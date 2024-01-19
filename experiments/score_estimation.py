from data.dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
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

for file in os.listdir('./'):
    if 'model' in file:
        save_path = os.path.join('./', file)    
        print("Best model loaded")

model_name = 'distilbert-base-uncased'
# model_name = 'allenai/scibert_scivocab_uncased'


tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(model_name=model_name, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300) # nout = bert model hidden dim
model.to(device)

checkpoint = torch.load(save_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

graph_model = model.get_graph_encoder()
text_model = model.get_text_encoder()

batch_size = 64

val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


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
    



# Compute cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(text_embeddings, graph_embeddings)

# Compute LRAP
val_lrap = label_ranking_average_precision_score(y_true, similarity)

print("Validation LRAP Score:", val_lrap)