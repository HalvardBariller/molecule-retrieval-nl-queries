from dataloader.dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.loader import DataLoader 
from torch.utils.data import DataLoader as TorchDataLoader
from models.Model import Model
from models.graph_encoders import GINEncoder
from models.text_encoders import TextEncoder
import numpy as np
from transformers import AutoTokenizer
import torch
from utils import make_predictions
import argparse
from tqdm import tqdm


argparser = argparse.ArgumentParser(description="Evaluates the given model on the validation set and generates submission using the model.")
argparser.add_argument("model_path")
args = argparser.parse_args()


########################
#     TEXT ENCODER     #
########################
model_name = 'distilbert-base-uncased'


batch_size = 32

tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################
#    GRAPH ENCODER     #
########################
graph_encoder = GINEncoder(num_layers=5, num_node_features=300, interm_hidden_dim=600, hidden_dim=300, out_interm_dim=600, out_dim=768) # nout = bert model hidden dim


text_encoder = TextEncoder(model_name)
model = Model(graph_encoder, text_encoder)
model.to(device)


print("Model path:", args.model_path)
print('Loading model...')
checkpoint = torch.load(args.model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print("Model loaded")
model.eval()

graph_model = model.get_graph_encoder()
text_model = model.get_text_encoder()

test_cids_dataset = GraphDataset(root='./data/', gt=gt, split='test_cids')
test_text_dataset = TextDataset(file_path='./data/test_text.txt', tokenizer=tokenizer)

idx_to_cid = test_cids_dataset.get_idx_to_cid()

test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

graph_embeddings = []
print("Computing graph embeddings...")
for batch in tqdm(test_loader):
    for output in graph_model(batch.to(device)):
        graph_embeddings.append(output.tolist())

test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
text_embeddings = []
print("Computing text embeddings...")
for batch in tqdm(test_text_loader):
    for output in text_model(batch['input_ids'].to(device), 
                             attention_mask=batch['attention_mask'].to(device)):
        text_embeddings.append(output.tolist())

        
make_predictions(graph_embeddings, text_embeddings)

print("Predictions ready!")