from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel

import sys

from .graph_encoders import GraphEncoder, GraphormerEncoder
from .text_encoders import TextEncoder



class Model(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels):
        super(Model, self).__init__()
        #self.graph_encoder = GraphEncoder(num_node_features, nout, nhid, graph_hidden_channels)
        self.graph_encoder = GraphormerEncoder(num_layers=6, num_node_features=num_node_features, hidden_dim=nout, num_heads=32)
        self.text_encoder = TextEncoder(model_name)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder

