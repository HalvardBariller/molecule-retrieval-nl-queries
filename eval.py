from dataloader.dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.loader import DataLoader 
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.metrics import label_ranking_average_precision_score
from models.Model import Model
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
import time
import os
import pandas as pd
import wandb
from utils import compute_embeddings_valid, compute_similarities_LRAP, make_predictions

from losses.contrastive_loss import contrastive_loss

import warnings
warnings.simplefilter("ignore", category=UserWarning)

## Model
model_name = 'distilbert-base-uncased'


batch_size = 32

for file in os.listdir('./'):
    if 'woven' in file:
        save_path = os.path.join('./', file)    
        print("Best model loaded:", file)

tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(model_name=model_name, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300) # nout = bert model hidden dim
model.to(device)


print("Model path", save_path)
print('loading best model...')
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

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


# from sklearn.metrics.pairwise import cosine_similarity

# similarity = cosine_similarity(text_embeddings, graph_embeddings)


# solution = pd.DataFrame(similarity)
# solution['ID'] = solution.index
# solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
# solution.to_csv('submission2201.csv', index=False)
        
make_predictions(text_embeddings, graph_embeddings)

print("Predictions ready!")