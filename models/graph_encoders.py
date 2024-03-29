import torch
from torch import nn
import torch.nn.functional as F
import utils
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree
from transformers import AutoModel


###################################
# Graph Convolutional Network (baseline)
# From paper "Semi-Supervised Classification with Graph Convolutional Networks" by Thomas N. Kipf and Max Welling
# (https://arxiv.org/abs/1609.02907v4)
###################################

class GCNEncoder(nn.Module):
    def __init__(self, num_node_features, n_layers_conv, n_layers_out, nout, nhid, graph_hidden_channels):
        super(GCNEncoder, self).__init__()
        if n_layers_conv < 1:
            raise ValueError("GCN encoder must use at least one convolution layer")
        if n_layers_out < 2:
            raise ValueError("Out MLP must have at least two layers")
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, graph_hidden_channels))
        for i in range(n_layers_conv-1):
            self.convs.append(GCNConv(graph_hidden_channels, graph_hidden_channels))
        self.mol_mlp = nn.Sequential()
        self.mol_mlp.append(nn.Linear(graph_hidden_channels, nhid))
        self.mol_mlp.append(nn.ReLU())
        for _ in range(1, n_layers_out-1):
            self.mol_mlp.append(nn.Linear(nhid, nhid))
            self.mol_mlp.append(nn.ReLU())
        self.mol_mlp.append(nn.Linear(nhid, nout))

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        for i in range(len(self.convs)-1):
            x = self.convs[i](x, edge_index)
            x = x.relu()
        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_mlp(x)
        return x




##################################
# Graphormer
# From paper "Do Transformers Really Perform Bad for Graph Representation?" by Chengxuan Ying, Tianle Cai et al.
# (https://arxiv.org/abs/2106.05234)
# The implementation below is split into multiple classes.
##################################

class Graphormer_AttentionHead(nn.Module):
    def __init__(self, hidden_dim, dim_k):
        super(Graphormer_AttentionHead, self).__init__()
        self.dim_k = dim_k
        self.proj_Q = nn.Linear(hidden_dim, dim_k)
        self.proj_K = nn.Linear(hidden_dim, dim_k)
        self.proj_V = nn.Linear(hidden_dim, dim_k)
    
    def forward(self, H, b_matrix, attention_mask):
        Q = self.proj_Q(H)
        K = self.proj_K(H)
        logits = torch.where(attention_mask, torch.mm(Q, torch.transpose(K, 0, 1))/self.dim_k**.5 + b_matrix, float("-inf"))
        weights = torch.softmax(logits, -1)
        V = self.proj_V(H)
        return torch.mm(weights, V)


class Graphormer_MHA(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(Graphormer_MHA, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"Hidden dimension {hidden_dim} is not a multiple of num_heads={num_heads}")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_k = hidden_dim // num_heads
        self.heads = []
        for _ in range(num_heads):
            self.heads.append(Graphormer_AttentionHead(self.hidden_dim, self.dim_k))
        self.heads = nn.ModuleList(self.heads)
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, H, b_matrix, attention_mask):
        outs = [head(H, b_matrix, attention_mask) for head in self.heads]
        return self.proj_out(torch.cat(outs, -1))


class Graphormer_EncoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dim_ff):
        super(Graphormer_EncoderBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.mha = Graphormer_MHA(hidden_dim, num_heads)
        self.ffn_lin1 = nn.Linear(hidden_dim, dim_ff)
        self.ffn_lin2 = nn.Linear(dim_ff, hidden_dim)

    def forward(self, H, b_matrix, attention_mask):
        x = self.mha(F.layer_norm(H, (self.hidden_dim,)), b_matrix, attention_mask) + H
        x1 = F.relu(self.ffn_lin1(F.layer_norm(x, (self.hidden_dim,))))
        x2 = self.ffn_lin2(x1)
        return x+x2


class GraphormerEncoder(nn.Module):
    def __init__(self, num_layers, num_node_features, hidden_dim, num_heads, dim_ff=2048, max_degree=100, max_sp_distance=100):
        super(GraphormerEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.max_degree = max_degree
        self.max_sp_distance = max_sp_distance
        self.init_proj = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(Graphormer_EncoderBlock(hidden_dim, num_heads, dim_ff))
        self.layers = nn.ModuleList(self.layers)
        self.centrality_encoding = nn.Embedding(max_degree+1, self.hidden_dim)
        self.distance_biases = nn.Parameter(torch.randn((max_sp_distance+2,))) # last element is for unconnected pairs of nodes
        self.virtnode_dist_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, graph_batch):
        x = graph_batch.x
        sp_matrix = utils.compute_shortest_path_matrix(graph_batch).to(x.device)
        sp_matrix = torch.min(sp_matrix, torch.tensor(self.max_sp_distance, device=x.device))
        transform = utils.VirtualNodeBatch()
        graph_batch = transform(graph_batch)

        # Weight mask for zeroing attention weights that span across two different graphs
        attention_mask = graph_batch.batch[:, None] == graph_batch.batch[None, :]

        # b matrix, shared by all layers
        ii_orig = torch.arange(0, graph_batch.orig_num_nodes, device=x.device)[:, None]
        jj_orig = torch.arange(0, graph_batch.orig_num_nodes, device=x.device)[None, :]
        B_matrix = torch.full((graph_batch.num_nodes, graph_batch.num_nodes), self.distance_biases[-1].item(), device=x.device)
        B_matrix[ii_orig, jj_orig] = self.distance_biases[sp_matrix[ii_orig, jj_orig]]

        ii = torch.arange(0, graph_batch.num_nodes, device=x.device)[:, None]
        jj = torch.arange(0, graph_batch.num_nodes, device=x.device)[None, :]
        virtual_edges = torch.argwhere(( (graph_batch.is_virtual_node[ii]) & (~(graph_batch.is_virtual_node[jj])) & 
                (graph_batch.batch[ii] == graph_batch.batch[jj]) ) |
                ( (~graph_batch.is_virtual_node[ii]) & (graph_batch.is_virtual_node[jj]) & 
                (graph_batch.batch[ii] == graph_batch.batch[jj]) ))
        B_matrix[virtual_edges.T[0], virtual_edges.T[1]] = self.virtnode_dist_bias

        B_matrix[torch.arange(graph_batch.num_nodes, device=x.device), torch.arange(graph_batch.num_nodes, device=x.device)] = self.distance_biases[0]

        #print(graph_batch, graph_batch.num_nodes, torch.max(graph_batch.edge_index), graph_batch.is_node_attr('x'))
        degrees = torch.minimum(torch.tensor(self.max_degree, dtype=torch.long, device=x.device), torch.tensor(degree(graph_batch.edge_index[0], graph_batch.num_nodes), dtype=torch.long, device=x.device))
        x = self.init_proj(x)
        # Initial embedding of virtual nodes
        virtual_node_init = torch.zeros((graph_batch.batch_size, self.hidden_dim), device=x.device)
        x = torch.cat([x, virtual_node_init], dim=0)
        x += self.centrality_encoding(degrees)
        for i in range(self.num_layers):
            x = self.layers[i](x, B_matrix, attention_mask)
        return x[graph_batch.virtual_node_index]        



###################################
# Graph Isomorphism Network (GIN)
# From paper "How Powerful are Graph Neural Networks?" by Keyulu Xu, Weihua Hu et al.
# (https://arxiv.org/abs/1810.00826)
###################################

class GINEncoder(nn.Module):
    def __init__(self, num_layers, num_node_features, interm_hidden_dim, hidden_dim, out_interm_dim, out_dim):
        super(GINEncoder, self).__init__()
        if num_layers < 1:
            raise ValueError("GIN must have at least one layer")
        self.num_layers = num_layers
        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.out_interm_dim = out_interm_dim
        self.out_dim = out_dim
        self.layers = nn.ModuleList()
        firstMLP = nn.Sequential(
            nn.Linear(num_node_features, interm_hidden_dim),
            nn.ReLU(),
            nn.Linear(interm_hidden_dim, hidden_dim)
        )
        self.layers.append(GINConv(firstMLP))
        for i in range(1, num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, interm_hidden_dim),
                nn.ReLU(),
                nn.Linear(interm_hidden_dim, hidden_dim)
            )
            self.layers.append(GINConv(mlp))
        self.finalMLP = nn.Sequential(
            nn.Linear(num_node_features + num_layers*hidden_dim, out_interm_dim),
            nn.ReLU(),
            nn.Linear(out_interm_dim, out_dim)
        )
    
    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        readouts = [global_add_pool(x, batch)]
        for layer in self.layers:
            x = layer(x, edge_index)
            readouts.append(global_add_pool(x, batch))
        x = torch.cat(readouts, dim=-1)
        return self.finalMLP(x)



###############################
# GraphSAGE
# From paper "Inductive Representation Learning on Large Graphs" by William L. Hamilton et al.
# (https://arxiv.org/abs/1706.02216)
###############################

class GraphSAGE(nn.Module):
    def __init__(self, num_node_features, nout, nhid, nhid_ff, num_layers = 2):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(num_node_features, nhid))
        for i in range(num_layers - 2):
            self.convs.append(SAGEConv(nhid, nhid))
        self.convs.append(SAGEConv(nhid, nhid_ff))
        self.final_post_process1 = nn.Linear(nhid_ff, nhid_ff)
        self.final_post_process2 = nn.Linear(nhid_ff, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
            else:
                x = F.normalize(x, p=2, dim=-1)
        x = global_mean_pool(x, batch)
        x = self.final_post_process1(x).relu()
        x = self.final_post_process2(x)
        return x




###############################
# Graph Attention Networks (GAT)
# From paper "Graph Attention Networks" by Petar Veličković, Guillem Cucurull et al.
# (https://arxiv.org/abs/1710.10903)
###############################
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, nhid_ff, nhid = 256, nout = 768, num_layers = 3, dropout=0.5, alpha=0.2):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv1 = GATv2Conv(num_node_features, nhid, heads=4, dropout=dropout, concat=True, negative_slope=alpha)
        self.conv2 = GATv2Conv(nhid * 4, nhid, heads=4, dropout=dropout, concat=True, negative_slope=alpha)  
        self.conv3 = GATv2Conv(nhid * 4, nhid_ff, heads=6, dropout=dropout, concat=False, negative_slope=alpha)
        self.final_post_process1 = nn.Linear(nhid_ff, nhid_ff)
        self.final_post_process2 = nn.Linear(nhid_ff, nout)

    def forward(self, graph_batch):
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        # 1st layer
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 2nd layer
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 3rd layer
        x = self.conv3(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        
        x = global_mean_pool(x, batch)
        x = self.final_post_process1(x).relu()
        x = self.final_post_process2(x)
        return x


