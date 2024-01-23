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

for file in os.listdir('./'):
    if 'model' in file:
        save_path = os.path.join('./', file)    
        print("Best model loaded")

model_name = 'distilbert-base-uncased'
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

# val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_cids_dataset = GraphDataset(root='./data/', gt=gt, split='test_cids')
test_text_dataset = TextDataset(file_path='./data/test_text.txt', tokenizer=tokenizer)
idx_to_cid = test_cids_dataset.get_idx_to_cid()
test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

# # Set the model to evaluation mode
# model.eval()

# # Initialize DataLoader for the validation dataset
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# # Lists to store embeddings
# text_embeddings = []
# graph_embeddings = []

# # Generate embeddings
# for batch in val_loader:
#     input_ids = batch.input_ids.to(device)
#     attention_mask = batch.attention_mask.to(device)
#     graph_batch = batch.to(device)

#     # Forward pass to get embeddings
#     x_graph, x_text = model(graph_batch, input_ids, attention_mask)

#     # Store embeddings
#     graph_embeddings.extend(x_graph.detach().cpu().numpy())
#     text_embeddings.extend(x_text.detach().cpu().numpy())

# # Create a mapping from CID to index
# cid_to_index = {cid: idx for idx, cid in enumerate(val_dataset.cids)}

# num_texts = len(val_dataset)
# num_graphs = num_texts  # Assuming each text description corresponds to a unique graph
# y_true = np.zeros((num_texts, num_graphs))

# for idx in range(num_texts):
#     cid = val_dataset.idx_to_cid[idx]
#     graph_idx = cid_to_index[cid]  # Map the CID to the correct index
#     y_true[idx, graph_idx] = 1

graph_embeddings = []
for batch in test_loader:
    for output in graph_model(batch.to(device)):
        graph_embeddings.append(output.tolist())

test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
text_embeddings = []
for batch in test_text_loader:
    for output in text_model(batch['input_ids'].to(device), 
                             attention_mask=batch['attention_mask'].to(device)):
        text_embeddings.append(output.tolist())

# Compute cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances

similarity = cosine_similarity(text_embeddings, graph_embeddings)
# Dot product
similarity_dot = np.matmul(text_embeddings, np.transpose(graph_embeddings))
# Cosine similarity
similarity_cosine = cosine_similarity(text_embeddings, graph_embeddings)

# Adjusted cosine similarity
# First center the vectors
text_embeddings_centered = text_embeddings - np.mean(text_embeddings, axis=0)
graph_embeddings_centered = graph_embeddings - np.mean(graph_embeddings, axis=0)
# Then compute cosine similarity
similarity_adjusted_cosine = cosine_similarity(text_embeddings_centered, graph_embeddings_centered)

# Minkowski distance 
similarity_minkowski = 1 - pairwise_distances(text_embeddings, graph_embeddings, metric='minkowski', p = 2)
# Euclidean distance
similarity_euclidean = 1 - pairwise_distances(text_embeddings, graph_embeddings, metric='euclidean')

# Compute LRAP for each metric
# val_lrap_dot = label_ranking_average_precision_score(y_true, similarity_dot)
# val_lrap_cosine = label_ranking_average_precision_score(y_true, similarity_cosine)
# val_lrap_adjusted_cosine = label_ranking_average_precision_score(y_true, similarity_adjusted_cosine)
# val_lrap_minkowski = label_ranking_average_precision_score(y_true, similarity_minkowski)
# val_lrap_euclidean = label_ranking_average_precision_score(y_true, similarity_euclidean)
# # Mixture of similarity metrics
# val_lrap_mixture = label_ranking_average_precision_score(y_true, np.mean([similarity_dot, 
#                                                                           similarity_cosine, 
                                                                        #   similarity_adjusted_cosine,
                                                                        #   similarity_minkowski,
                                                                        #   similarity_euclidean], axis=0))

# print("Validation LRAP Score (Dot Product):", val_lrap_dot)
# print("Validation LRAP Score (Cosine):", val_lrap_cosine)
# print("Validation LRAP Score (Adjusted Cosine):", val_lrap_adjusted_cosine)
# print("Validation LRAP Score (Minkowski):", val_lrap_minkowski)
# print("Validation LRAP Score (Euclidean):", val_lrap_euclidean)
# print("Validation LRAP Score (Mixture):", val_lrap_mixture)



#### Model 2 ####
print("Model 2")

save_path_2 = './Models/model_2_scibert.pt'


model_name_2 = 'allenai/scibert_scivocab_uncased'
tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)
gt_2 = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset_2 = GraphTextDataset(root='./data/', gt=gt_2, split='val', tokenizer=tokenizer_2)
train_dataset_2 = GraphTextDataset(root='./data/', gt=gt_2, split='train', tokenizer=tokenizer_2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_2 = Model(model_name=model_name_2, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300) # nout = bert model hidden dim
model_2.to(device)

checkpoint_2 = torch.load(save_path_2, map_location=device)
model_2.load_state_dict(checkpoint_2['model_state_dict'])
model_2.eval()

graph_model_2 = model_2.get_graph_encoder()
text_model_2 = model_2.get_text_encoder()

# Lists to store embeddings
graph_embeddings_2 = []
for batch in test_loader:
    for output in graph_model_2(batch.to(device)):
        graph_embeddings_2.append(output.tolist())

test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
text_embeddings_2 = []
for batch in test_text_loader:
    for output in text_model_2(batch['input_ids'].to(device), 
                             attention_mask=batch['attention_mask'].to(device)):
        text_embeddings_2.append(output.tolist())




from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

# Dot product
similarity_dot_2 = np.matmul(text_embeddings_2, np.transpose(graph_embeddings_2))

# Cosine similarity
similarity_cosine_2 = cosine_similarity(text_embeddings_2, graph_embeddings_2)

# Adjusted cosine similarity
# First center the vectors
text_embeddings_centered_2 = text_embeddings_2 - np.mean(text_embeddings_2, axis=0)
graph_embeddings_centered_2 = graph_embeddings_2 - np.mean(graph_embeddings_2, axis=0)
# Then compute cosine similarity
similarity_adjusted_cosine_2 = cosine_similarity(text_embeddings_centered_2, graph_embeddings_centered_2)

# Minkowski distance 
similarity_minkowski_2 = 1 - pairwise_distances(text_embeddings_2, graph_embeddings_2, metric='minkowski', p = 2)
# Euclidean distance
similarity_euclidean_2 = 1 - pairwise_distances(text_embeddings_2, graph_embeddings_2, metric='euclidean')

# # Compute LRAP for each metric
# val_lrap_dot_2 = label_ranking_average_precision_score(y_true, similarity_dot_2)
# val_lrap_cosine_2 = label_ranking_average_precision_score(y_true, similarity_cosine_2)
# val_lrap_adjusted_cosine_2 = label_ranking_average_precision_score(y_true, similarity_adjusted_cosine_2)
# val_lrap_minkowski_2 = label_ranking_average_precision_score(y_true, similarity_minkowski_2)
# val_lrap_euclidean_2 = label_ranking_average_precision_score(y_true, similarity_euclidean_2)
# # Mixture of similarity metrics
# val_lrap_mixture_2 = label_ranking_average_precision_score(y_true, np.mean([similarity_dot_2, 
#                                                                           similarity_cosine_2, 
#                                                                           similarity_adjusted_cosine_2,
#                                                                           similarity_minkowski_2,
#                                                                           similarity_euclidean_2], axis=0))

# print("Validation LRAP Score (Dot Product):", val_lrap_dot_2)
# print("Validation LRAP Score (Cosine):", val_lrap_cosine_2)
# print("Validation LRAP Score (Adjusted Cosine):", val_lrap_adjusted_cosine_2)
# print("Validation LRAP Score (Minkowski):", val_lrap_minkowski_2)
# print("Validation LRAP Score (Euclidean):", val_lrap_euclidean_2)
# print("Validation LRAP Score (Mixture):", val_lrap_mixture_2)




#### Mixture ####

print("Mixture of models")

# Mixture of similarity metrics
# val_lrap_mixture_models = label_ranking_average_precision_score(y_true, np.mean([similarity_dot, 
#                                                                           similarity_cosine, 
#                                                                           similarity_adjusted_cosine,
#                                                                           similarity_minkowski,
#                                                                           similarity_euclidean,
#                                                                           similarity_dot_2, 
#                                                                           similarity_cosine_2, 
#                                                                           similarity_adjusted_cosine_2,
#                                                                           similarity_minkowski_2,
#                                                                           similarity_euclidean_2], axis=0))

# Solution 

solution = pd.DataFrame(np.mean([similarity_dot,
                                 similarity_cosine, 
                                 similarity_adjusted_cosine,
                                 similarity_minkowski,
                                 similarity_euclidean,
                                 similarity_dot_2, 
                                 similarity_cosine_2, 
                                 similarity_adjusted_cosine_2,
                                 similarity_minkowski_2,
                                 similarity_euclidean_2], axis=0)
                                 )
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv('submission.csv', index=False)