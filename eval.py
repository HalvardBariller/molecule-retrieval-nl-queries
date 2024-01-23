from dataloader.dataloader import GraphTextDataset, GraphDataset, TextDataset
#from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader 
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.metrics import label_ranking_average_precision_score
from models.Model import Model
from models.graph_encoders import GraphEncoder
from models.text_encoders import TextEncoder
import numpy as np
from transformers import AutoTokenizer
import torch
import os
import pandas as pd
from utils import compute_embeddings_valid, compute_similarities_LRAP, make_predictions
import argparse

from losses.contrastive_loss import contrastive_loss

import warnings
warnings.simplefilter("ignore", category=UserWarning)

argparser = argparse.ArgumentParser(description="Evaluates the given model on the validation set and generates submission using the model.")
argparser.add_argument("model_path")
args = argparser.parse_args()

## Model
model_name = 'distilbert-base-uncased'

batch_size = 64

tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
#train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

graph_encoder = GraphEncoder(num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300) # nout = bert model hidden dim
text_encoder = TextEncoder(model_name)
model = Model(graph_encoder, text_encoder)
model.to(device)


print("Model path:", args.model_path)
print('Loading model...')
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model_state_dict'])
print("Model loaded")
model.eval()

# Compute validation LRAP
val_loader = DataLoader(val_dataset, batch_size=batch_size)
val_loss = 0        
for batch in val_loader:
    input_ids = batch.input_ids
    batch.pop('input_ids')
    attention_mask = batch.attention_mask
    batch.pop('attention_mask')
    graph_batch = batch
    x_graph, x_text = model(graph_batch.to(device), 
                            input_ids.to(device), 
                            attention_mask.to(device))
    current_loss = contrastive_loss(x_graph, x_text)   
    val_loss += current_loss.item()

print('Validation loss: ', str(val_loss/len(val_loader)) )

# LRAP computation
graph_embeddings, text_embeddings, y_true = compute_embeddings_valid(model, val_dataset, device, batch_size)
lrap_current_valid = compute_similarities_LRAP(graph_embeddings, text_embeddings, y_true)
print("Validation LRAP Score:", lrap_current_valid)

graph_model = model.get_graph_encoder()
text_model = model.get_text_encoder()

test_cids_dataset = GraphDataset(root='./data/', gt=gt, split='test_cids')
test_text_dataset = TextDataset(file_path='./data/test_text.txt', tokenizer=tokenizer)

idx_to_cid = test_cids_dataset.get_idx_to_cid()

test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

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


from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(text_embeddings, graph_embeddings)

solution = pd.DataFrame(similarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
#solution.to_csv('submission.csv', index=False)
        
make_predictions(text_embeddings, graph_embeddings)

print("Predictions ready!")