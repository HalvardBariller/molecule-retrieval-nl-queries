import torch
from torch import nn
import torch.nn.functional as F
import utils
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
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
        self.proj_Q = nn.Linear(hidden_dim, dim_k)
        self.proj_K = nn.Linear(hidden_dim, dim_k)
        self.proj_V = nn.Linear(hidden_dim, dim_k)
    
    def forward(self, H, shortest_path_matrix):
        Q = self.proj_Q(H)
        K = self.proj_K(H)
        raise NotImplementedError



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
        
    def forward(self, H, shortest_path_matrix):
        raise NotImplementedError

class Graphormer_EncoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dim_ff):
        super(Graphormer_EncoderBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.mha = Graphormer_MHA(hidden_dim, num_heads)
        self.ffn_lin1 = nn.Linear(hidden_dim, dim_ff)
        self.ffn_lin2 = nn.Linear(dim_ff, hidden_dim)

    def forward(self, H, shortest_path_matrix):
        x = self.mha(F.layer_norm(H, self.hidden_dim), shortest_path_matrix) + H
        x1 = F.relu(self.ffn_lin1(F.layer_norm(x, self.hidden_dim)))
        x2 = self.ffn_lin2(x1)
        return x+x2




class GraphormerEncoder(nn.Module):
    def __init__(self, num_layers, num_node_features, hidden_dim, num_heads, dim_ff=2048, max_degree=100):
        super(GraphormerEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.max_degree = max_degree
        self.init_proj = nn.Linear(num_node_features, hidden_dim)
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(Graphormer_EncoderBlock(hidden_dim, num_heads, dim_ff))
        self.centrality_encoding = nn.Embedding(max_degree+1, self.hidden_dim)

    def forward(self, graph_batch):
        x = graph_batch.x
        sp_matrix = utils.compute_shortest_path_matrix(graph_batch)
        transform = utils.VirtualNodeBatch()
        graph_batch = transform(graph_batch)
        degrees = torch.minimum(torch.tensor(self.max_degree), torch.tensor(degree(graph_batch.edge_index[0], graph_batch.num_nodes)))
        x = self.init_proj(x)
        virtual_node_init = torch.zeros((graph_batch.batch_size, self.hidden_dim), device=x.device)
        x = torch.cat([x, virtual_node_init], dim=0)
        x += self.centrality_encoding[degrees]
        for i in range(self.num_layers):
            x = self.layers[i](x, sp_matrix)
        return x[graph_batch.virtual_node_index]        
    
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
