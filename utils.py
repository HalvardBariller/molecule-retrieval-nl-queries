from dataloader.dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.metrics import label_ranking_average_precision_score
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
import time
import os
import pandas as pd
import wandb
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
import copy



def compute_embeddings_valid(model, val_dataset, device, batch_size):
    # Set the model to evaluation mode
    model.eval()

    # Initialize DataLoader for the validation dataset
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Lists to store embeddings
    text_embeddings = []
    graph_embeddings = []

    # Generate embeddings
    for batch in val_loader:
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        graph_batch = batch.to(device)

        # Forward pass to get embeddings
        x_graph, x_text = model(graph_batch, input_ids, attention_mask)

        # Store embeddings
        graph_embeddings.extend(x_graph.detach().cpu().numpy())
        text_embeddings.extend(x_text.detach().cpu().numpy())


    # Create a mapping from CID to index
    cid_to_index = {cid: idx for idx, cid in enumerate(val_dataset.cids)}

    num_texts = len(val_dataset)
    num_graphs = num_texts  # Assuming each text description corresponds to a unique graph
    y_true = np.zeros((num_texts, num_graphs))

    for idx in range(num_texts):
        cid = val_dataset.idx_to_cid[idx]
        graph_idx = cid_to_index[cid]  # Map the CID to the correct index
        y_true[idx, graph_idx] = 1


    return graph_embeddings, text_embeddings, y_true



def compute_similarities_LRAP(graph_embeddings, text_embeddings, y_true):

    # Cosine similarity
    similarity_cos = cosine_similarity(text_embeddings, graph_embeddings)
    # Adjusted cosine similarity
    text_embeddings_centered = text_embeddings - np.mean(text_embeddings, axis=0)
    graph_embeddings_centered = graph_embeddings - np.mean(graph_embeddings, axis=0)
    similarity_adjcos = cosine_similarity(text_embeddings_centered, graph_embeddings_centered)
    # Dot product
    similarity_dot = np.matmul(text_embeddings, np.transpose(graph_embeddings))
    # Euclidean similarity
    similarity_euc = - pairwise_distances(text_embeddings, graph_embeddings, metric='euclidean')
    # Minkowski similarity
    similarity_min = - pairwise_distances(text_embeddings, graph_embeddings, metric='minkowski')
    
    similarity = np.mean([similarity_cos, similarity_adjcos, similarity_dot
                           #similarity_euc, similarity_min
                           ], axis=0)

    similarity_normalized = np.mean([similarity_cos / np.max(similarity_cos, axis=1)[:,None],
                                    similarity_adjcos / np.max(similarity_adjcos, axis=1)[:,None],
                                    similarity_dot / np.max(similarity_dot, axis=1)[:,None]
                                    #similarity_euc / np.max(similarity_euc, axis=1)[:,None],
                                    #similarity_min / np.max(similarity_min, axis=1)[:,None]
                                    ], axis=0)
                                        

    # Compute LRAP
    val_lrap = label_ranking_average_precision_score(y_true, similarity)
    val_lrap_normalized = label_ranking_average_precision_score(y_true, similarity_normalized)
    print("LRAP cosine:", label_ranking_average_precision_score(y_true, similarity_cos))
    print("LRAP adjcos:", label_ranking_average_precision_score(y_true, similarity_adjcos))
    print("LRAP dot:", label_ranking_average_precision_score(y_true, similarity_dot))
    print("LRAP euc:", label_ranking_average_precision_score(y_true, similarity_euc))
    print("LRAP mink:", label_ranking_average_precision_score(y_true, similarity_min))

    print('LRAP:', val_lrap)
    print('LRAP normalized:', val_lrap_normalized)

    #print("Validation LRAP Score:", val_lrap)
    return val_lrap, val_lrap_normalized


def make_predictions(graph_embeddings, text_embeddings, save_file = True):
    """Make predictions on the test set and save them to a CSV file.
    --------------
    Parameters:
    graph_embeddings: list of numpy arrays
        List of graph embeddings.
    text_embeddings: list of numpy arrays
        List of text embeddings.
    """
    # Cosine similarity
    similarity_cos = cosine_similarity(text_embeddings, graph_embeddings)
    # Adjusted cosine similarity
    text_embeddings_centered = text_embeddings - np.mean(text_embeddings, axis=0)
    graph_embeddings_centered = graph_embeddings - np.mean(graph_embeddings, axis=0)
    similarity_adjcos = cosine_similarity(text_embeddings_centered, graph_embeddings_centered)
    # Dot product
    similarity_dot = np.matmul(text_embeddings, np.transpose(graph_embeddings))
    # Euclidean similarity
    similarity_euc = - pairwise_distances(text_embeddings, graph_embeddings, metric='euclidean')
    # Minkowski similarity
    similarity_min = - pairwise_distances(text_embeddings, graph_embeddings, metric='minkowski')
    
    similarity = np.mean([similarity_cos, similarity_adjcos, similarity_dot, similarity_euc, similarity_min], axis=0)

    similarity_normalized = np.mean([similarity_cos / np.max(similarity_cos, axis=1)[:,None],
                                    similarity_adjcos / np.max(similarity_adjcos, axis=1)[:,None],
                                    similarity_dot / np.max(similarity_dot, axis=1)[:,None]
                                    #similarity_euc / np.max(similarity_euc, axis=1)[:,None],
                                    #similarity_min / np.max(similarity_min, axis=1)[:,None]
                                    ], axis=0)
    if save_file:
        solution = pd.DataFrame(similarity_normalized)
        solution['ID'] = solution.index
        solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
        solution.to_csv('submission.csv', index=False)
    else:
        return similarity_normalized
    

def prepare_submission_file(similarities, submission_file_name='submission'):
    """Prepare the submission file.
    --------------
    Parameters:
    similarities: numpy array
        Array of similarities.
    """
    solution = pd.DataFrame(similarities)
    solution['ID'] = solution.index
    solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
    solution.to_csv(submission_file_name + '.csv', index=False)
    
    

def compute_shortest_path_matrix(graph, to_undirected=True):
    """Compute the matrix of shortest paths of the given PyTorch Geometric graph.
    
    Parameters:
    graph: torch_geometric.data.Data
        The PyTorch Geometric graph for which to compute the shortest path matrix.
    to_undirected: boolean, optional
        Whether to consider the graph to be undirected."""
    nx_graph = to_networkx(graph, to_undirected=to_undirected)
    sp = nx.shortest_path_length(nx_graph)
    ret = -1*torch.ones((nx_graph.number_of_nodes(), nx_graph.number_of_nodes()), dtype=torch.long)
    for source, lendict in sp:
        for target, dist in lendict.items():
            ret[source, target] = dist
    return ret

# An adaptation of torch_geometric.transforms.virtual_node.VirtualNode for batched inputs.
class VirtualNodeBatch(BaseTransform):
    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        row, col = data.edge_index
        batch = data.batch
        edge_type = data.get('edge_type', torch.zeros_like(row))
        num_nodes = data.num_nodes
        assert num_nodes is not None

        arange = torch.arange(num_nodes, device=row.device)
        full = batch + num_nodes
        row = torch.cat([row, arange, full], dim=0)
        col = torch.cat([col, full, arange], dim=0)
        edge_index = torch.stack([row, col], dim=0)

        new_type = edge_type.new_full((num_nodes, ), int(edge_type.max()) + 1)
        edge_type = torch.cat([edge_type, new_type, new_type + 1], dim=0)

        new_batch = torch.arange(data.batch_size, device=batch.device)
        batch = torch.cat([batch, new_batch], dim=0)
        
        virtual_node_index = torch.arange(num_nodes, num_nodes+data.batch_size, device=batch.device)
        is_virtual_node = torch.zeros((num_nodes+data.batch_size,), dtype=torch.bool, device=batch.device)
        is_virtual_node[num_nodes:] = True

        old_data = copy.copy(data)
        for key, value in old_data.items():
            if key == 'edge_index' or key == 'edge_type' or key == 'batch':
                continue

            if isinstance(value, torch.Tensor):
                dim = old_data.__cat_dim__(key, value)
                size = list(value.size())

                fill_value = None
                if key == 'edge_weight':
                    size[dim] = 2 * num_nodes
                    fill_value = 1.
                elif old_data.is_edge_attr(key):
                    size[dim] = 2 * num_nodes
                    fill_value = 0.
                elif old_data.is_node_attr(key):
                    size[dim] = data.batch_size
                    fill_value = 0.

                if fill_value is not None:
                    new_value = value.new_full(size, fill_value)
                    data[key] = torch.cat([value, new_value], dim=dim)

        data.edge_index = edge_index
        data.edge_type = edge_type
        data.batch = batch
        data.virtual_node_index = virtual_node_index
        data.is_virtual_node = is_virtual_node

        if 'num_nodes' in data:
            data.num_nodes = num_nodes + data.batch_size
        
        data.orig_num_nodes = num_nodes

        return data

