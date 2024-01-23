from dataloader.dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.metrics import label_ranking_average_precision_score
from models.Model import Model
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
    
    similarity = np.mean([similarity_cos, similarity_adjcos, similarity_dot, similarity_euc, similarity_min], axis=0)

    similarity_normalized = np.mean([similarity_cos / np.max(similarity_cos, axis=1)[:,None],
                                    similarity_adjcos / np.max(similarity_adjcos, axis=1)[:,None],
                                    similarity_dot / np.max(similarity_dot, axis=1)[:,None],
                                    similarity_euc / np.max(similarity_euc, axis=1)[:,None],
                                    similarity_min / np.max(similarity_min, axis=1)[:,None]], axis=0)
                                        

    # Compute LRAP
    val_lrap = label_ranking_average_precision_score(y_true, similarity)
    val_lrap_normalized = label_ranking_average_precision_score(y_true, similarity_normalized)
    print('LRAP:', val_lrap)
    print('LRAP normalized:', val_lrap_normalized)

    #print("Validation LRAP Score:", val_lrap)
    return val_lrap


def make_predictions(graph_embeddings, text_embeddings):
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
                                    similarity_dot / np.max(similarity_dot, axis=1)[:,None],
                                    similarity_euc / np.max(similarity_euc, axis=1)[:,None],
                                    similarity_min / np.max(similarity_min, axis=1)[:,None]], axis=0)
    
    solution = pd.DataFrame(similarity_normalized)
    solution['ID'] = solution.index
    solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
    solution.to_csv('submission2.csv', index=False)
    
    

