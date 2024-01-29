from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, graph_encoder, text_encoder):
        super(Model, self).__init__()
        self.graph_encoder = graph_encoder
        self.text_encoder = text_encoder
        self.projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 512)
            )
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        #graph_encoded = self.projection(graph_encoded)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        #text_encoded = self.projection(text_encoded)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder

