from dataloader.dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.loader import DataLoader 
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.metrics import label_ranking_average_precision_score
from models.Model import Model
from models.expert_model import ExpertModel
from models.graph_encoders import GINEncoder, GraphormerEncoder, GraphSAGE
from models.text_encoders import TextEncoder
import numpy as np
from transformers import AutoTokenizer
import torch
import os
import pandas as pd
from utils import compute_embeddings_valid, compute_similarities_LRAP, make_predictions, prepare_submission_file
import argparse
from tqdm import tqdm


import warnings
warnings.simplefilter("ignore", category=UserWarning)


batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


aggregated_predictions_val = []
labels_val = []


model_1 = {"model_path": "gin_6layers_roberta_8131.pt",
           "model_name": "roberta-base",
           "graph_encoder": GINEncoder(num_layers=6, num_node_features=300, interm_hidden_dim=600, 
                                       hidden_dim=300, out_interm_dim=600, out_dim=768),
           "text_encoder": TextEncoder("roberta-base")}

model_2 = {"model_path": "graphormer_model.pt",
              "model_name": "roberta-base",
              "graph_encoder": GraphormerEncoder(num_layers = 6, num_node_features = 300, hidden_dim = 768, num_heads = 32),
              "text_encoder": TextEncoder("roberta-base")}

models = [model_1, model_2]

for model in models:
    model_path, model_name = model["model_path"], model["model_name"]
    graph_encoder, text_encoder = model["graph_encoder"], model["text_encoder"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)

    model = Model(graph_encoder, text_encoder)
    model.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded:", model_path)
    model.eval()
    # Val predictions
    graph_embeddings_val, text_embeddings_val, y_true_val = compute_embeddings_valid(model, val_dataset, device, batch_size)
    print("Val embeddings computed")
    labels_val.append(y_true_val)
    predictions_val = make_predictions(graph_embeddings_val, text_embeddings_val, save_file=False)
    aggregated_predictions_val.append(predictions_val)
    
print("Predictions computed, training ensemble model...")

# Sanity check for labels_val
for i in range(len(labels_val)):
    assert np.array_equal(labels_val[i], labels_val[0]), "labels_val[{}] is not equal to labels_val[0]".format(i)

######### Naive Ensemble model (average of predictions) #########

ensemble_predictions_val = np.zeros_like(predictions_val)

for predictions_val in aggregated_predictions_val:
    ensemble_predictions_val += predictions_val
ensemble_predictions_val /= len(aggregated_predictions_val)

print("Ensemble model trained")
print("LRAP val:", label_ranking_average_precision_score(labels_val[0], ensemble_predictions_val))



print("Predictions ready!")


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=0, perplexity = 5)
# Plot graph embeddings and text embeddings for the first 10 graphs in the validation set using y_true_val as labels
graph_embeddings_val_tsne = tsne.fit_transform(np.array(graph_embeddings_val[:10]))
text_embeddings_val_tsne = tsne.fit_transform(np.array(text_embeddings_val[:10]))
y_true_val_tsne = np.argmax(y_true_val[:10])

fig, ax = plt.subplots(1, 1, figsize=(20, 10))




#ax.scatter(graph_embeddings_val_tsne[:, 0], graph_embeddings_val_tsne[:, 1], c=y_true_val_tsne, label = "Graph embeddings")
#ax.scatter(text_embeddings_val_tsne[:, 0], text_embeddings_val_tsne[:, 1], c=y_true_val_tsne, label = "Text embeddings")
ax.legend()
ax.set_title("Graph and text embeddings for the first 10 graphs in the validation set")
plt.show()




