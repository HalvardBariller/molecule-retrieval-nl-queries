import torch
from torch import nn
import torch.nn.functional as F
import utils
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import degree
from transformers import AutoModel



class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
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
        self.init_proj = nn.Linear(num_node_features, hidden_dim)
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(Graphormer_EncoderBlock(hidden_dim, num_heads, dim_ff))
        self.centrality_encoding = nn.Embedding(max_degree+1, self.hidden_dim)
        self.distance_biases = nn.Parameter(torch.randn((max_sp_distance+2,))) # last element is for unconnected pairs of nodes
        self.virtnode_dist_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, graph_batch):
        x = graph_batch.x
        sp_matrix = utils.compute_shortest_path_matrix(graph_batch)
        sp_matrix = torch.min(sp_matrix, torch.tensor(self.max_sp_distance))
        transform = utils.VirtualNodeBatch()
        graph_batch = transform(graph_batch)

        # Weight mask for zeroing attention weights that span across two different graphs
        attention_mask = graph_batch.batch[:, None] == graph_batch.batch[None, :]

        # b matrix, shared by all layers
        ii_orig = torch.arange(0, graph_batch.orig_num_nodes)[:, None]
        jj_orig = torch.arange(0, graph_batch.orig_num_nodes)[None, :]
        B_matrix = torch.full((graph_batch.num_nodes, graph_batch.num_nodes), self.distance_biases[-1].item(), device=x.device)
        B_matrix[ii_orig, jj_orig] = self.distance_biases[sp_matrix[ii_orig, jj_orig]]

        ii = torch.arange(0, graph_batch.num_nodes)[:, None]
        jj = torch.arange(0, graph_batch.num_nodes)[None, :]
        virtual_edges = torch.argwhere(( (graph_batch.is_virtual_node[ii]) & (~(graph_batch.is_virtual_node[jj])) & 
                (graph_batch.batch[ii] == graph_batch.batch[jj]) ) |
                ( (~graph_batch.is_virtual_node[ii]) & (graph_batch.is_virtual_node[jj]) & 
                (graph_batch.batch[ii] == graph_batch.batch[jj]) ))
        B_matrix[virtual_edges.T[0], virtual_edges.T[1]] = self.virtnode_dist_bias

        B_matrix[torch.arange(graph_batch.num_nodes), torch.arange(graph_batch.num_nodes)] = self.distance_biases[0]

        #print(graph_batch, graph_batch.num_nodes, torch.max(graph_batch.edge_index), graph_batch.is_node_attr('x'))
        degrees = torch.minimum(torch.tensor(self.max_degree, dtype=torch.long), torch.tensor(degree(graph_batch.edge_index[0], graph_batch.num_nodes), dtype=torch.long))
        x = self.init_proj(x)
        # Initial embedding of virtual nodes
        virtual_node_init = torch.zeros((graph_batch.batch_size, self.hidden_dim), device=x.device)
        x = torch.cat([x, virtual_node_init], dim=0)
        x += self.centrality_encoding(degrees)
        for i in range(self.num_layers):
            x = self.layers[i](x, B_matrix, attention_mask)
        return x[graph_batch.virtual_node_index]        