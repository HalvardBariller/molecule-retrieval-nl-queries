from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel

import sys

from .graph_encoders import GraphEncoder
from .text_encoders import TextEncoder



class Model(nn.Module):
    def __init__(self, graph_encoder, text_encoder):
        super(Model, self).__init__()
        self.graph_encoder = graph_encoder
        self.text_encoder = text_encoder
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder

