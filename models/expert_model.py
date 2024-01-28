import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel

import sys


class ExpertModel(nn.Module):
    def __init__(self, num_models):
        super(ExpertModel, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_models))
    
    def forward(self, predictors):
        # predictors shape: (batch_size, num_predictors, H)
        weighted_predictors = predictors * self.weights.unsqueeze(-1)
        return weighted_predictors.sum(dim=1)
