from dataloader.dataloader import GraphTextDataset
from torch_geometric.loader import DataLoader
from models.Model import Model
from models.graph_encoders import GCNEncoder, GraphormerEncoder, GINEncoder, GraphSAGE, GAT
from models.text_encoders import TextEncoder
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
import time
import os
import wandb
from tqdm import tqdm
from utils import compute_embeddings_valid, compute_similarities_LRAP
import argparse
from datetime import datetime
from losses.contrastive_loss import contrastive_loss, contrastive_loss_with_cosine, negative_sampling_contrastive_loss


## Initialization

# normally not needed?
import warnings
warnings.simplefilter("ignore", category=UserWarning)

argparser = argparse.ArgumentParser(description="Trains a model.")
argparser.add_argument("-n", "--name", help="Run name (for Weights and Biases)")
argparser.add_argument("-m", "--pretrained-model", help="Load pretrained model from file")
args = argparser.parse_args()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

curr_date = datetime.now().strftime("%y-%m-%dT%H-%M-%S")
if args.name is not None:
    artifact_name = f"{args.name}-{curr_date}.pt"
else:
    artifact_name = f"model-{curr_date}.pt"



#############################
#      HYPERPARAMETERS      #
#############################
# Set the training hyperparameters and Weights and Biases configuration below.
run = wandb.init(
        project="2nd run - ALTEGRAD",
        name=args.name,
        config={
            "epochs": 50,
            "batch_size": 32,
            "lr_text": 2e-5,
            "lr_graph": 4e-5
            })

config = wandb.config
nb_epochs = wandb.config.epochs
batch_size = wandb.config.batch_size
learning_rate_text = wandb.config.lr_text
learning_rate_graph = wandb.config.lr_graph



########################
#     TEXT ENCODER     #
########################
# Set the Hugging Face model name of the text encoder to use in the model in the variable `model_name`.
#model_name = 'distilbert-base-uncased'
#model_name = 'allenai/scibert_scivocab_uncased'
#model_name = 'allenai/biomed_roberta_base'
model_name = 'roberta-base'
#model_name = 'DeepChem/ChemBERTa-77M-MLM'

text_encoder = TextEncoder(model_name)



## Dataset definition and dataloading
tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

num_workers = 12

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers, pin_memory = True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers, pin_memory = True)



###########################
#      GRAPH ENCODER      #
###########################
# Define your graph encoder here. Below are some of the encoders that we tried.
#graph_encoder = GINEncoder(num_layers=6, num_node_features=300, interm_hidden_dim=600, hidden_dim=300, out_interm_dim=600, out_dim=768) # nout = bert model hidden dim
#graph_encoder = GraphormerEncoder(num_layers = 6, num_node_features = 300, hidden_dim = 768, num_heads = 32)
#graph_encoder = GCNEncoder(num_node_features=300, n_layers_conv=5, n_layers_out=3, nout=768, nhid=300, graph_hidden_channels=300)
#graph_encoder = GraphSAGE(num_node_features = 300, nout = 768, nhid = 300, nhid_ff = 600, num_layers = 2)
graph_encoder = AttentiveFP(num_node_features = 300, nout = 768, nhid = 300, nhid_ff = 600, num_layers = 2)



## Building and possibly loading the model
model = Model(graph_encoder, text_encoder)
model.to(device)

if args.pretrained_model is not None:
    print("Model path:", args.pretrained_model)
    print('Loading model...')
    checkpoint = torch.load(args.pretrained_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded")


## Optimizer
no_decay = ['bias', 'LayerNorm.weight']
optimizer_text_grouped_parameters = [
    {'params': [p for n, p in model.get_text_encoder().named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.get_text_encoder().named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer_text = optim.AdamW(optimizer_text_grouped_parameters, lr=learning_rate_text,
                                betas=(0.9, 0.98))
optimizer_graph = optim.AdamW(model.get_graph_encoder().parameters(), lr=learning_rate_graph,
                                betas=(0.9, 0.98), weight_decay=0.01)

## Learning rate scheduler
start_factor = 1.0 if args.pretrained_model is None else 0.3
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=3) * start_factor
scheduler_text = torch.optim.lr_scheduler.LinearLR(optimizer_text, start_factor=start_factor, end_factor=0.3, total_iters=nb_epochs)
scheduler_graph = torch.optim.lr_scheduler.LinearLR(optimizer_graph, start_factor=start_factor, end_factor=0.1, total_iters=nb_epochs)

## PyTorch AMP
scaler = GradScaler()




#################
# TRAINING LOOP
#################
epoch = 0
loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 50
best_lrap = 0

for i in range(nb_epochs):
    print('-----EPOCH{}-----'.format(i+1))
    model.train()
    for j, batch in enumerate(tqdm(train_loader)):
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch

        with autocast(dtype=torch.float16): # PyTorch AMP
            x_graph, x_text = model(graph_batch.to(device), 
                                    input_ids.to(device), 
                                    attention_mask.to(device))
            
            ## Classical contrastive loss
            #current_loss = contrastive_loss(x_graph, x_text)
            current_loss = contrastive_loss_with_cosine(x_graph, x_text)

        optimizer_graph.zero_grad()
        optimizer_text.zero_grad()
        scaler.scale(current_loss).backward()
        scaler.step(optimizer_graph)
        scaler.step(optimizer_text)
        scaler.update()
        loss += current_loss.item()
        
        count_iter += 1
        if count_iter % printEvery == 0:
            time2 = time.time()
            print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                        time2 - time1, loss/printEvery))
            losses.append(loss)
            loss = 0 


    # End of epoch: compute validation loss
    model.eval()       
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
        #current_loss = contrastive_loss(x_graph, x_text)   
        current_loss = contrastive_loss_with_cosine(x_graph, x_text)
        val_loss += current_loss.item()
    print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)) )


    # LRAP computation
    graph_embeddings, text_embeddings, y_true = compute_embeddings_valid(model, val_dataset, device, batch_size)
    lrap_current_valid, lrap_current_valid_norm = compute_similarities_LRAP(graph_embeddings, text_embeddings, y_true)
    best_lrap = max(best_lrap, max(lrap_current_valid, lrap_current_valid_norm))
    print("Validation LRAP Score:", lrap_current_valid)


    # Save model checkpoint
    if best_lrap == max(lrap_current_valid, lrap_current_valid_norm):
        print("Score improved, saving checkpoint...")
        # Remove previous checkpoints
        for file in os.listdir('./'):
            if 'model.pt' in file:
                os.remove(file)
            if args.pretrained_model is not None and args.pretrained_model in file:
                os.remove(file)
        save_path = os.path.join('./', str(i)+'model.pt')
        torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_graph_state_dict': optimizer_graph.state_dict(),
        'optimizer_text_state_dict': optimizer_text.state_dict(),
        'validation_accuracy': val_loss,
        'loss': loss,
        }, save_path)
        print('checkpoint saved to: {}'.format(save_path))
        # Uncomment the lines below to upload the weights to Weights and Biases
        # artifact = wandb.Artifact(name=artifact_name, type="model")
        # artifact.add_file(local_path=save_path)
        # run.log_artifact(artifact)
        # print("Artifact uploaded")

    wandb.log({"loss": loss, 
               "val_loss": val_loss,
               "LRAP_valid": lrap_current_valid,
               "LRAP_valid_normalized": lrap_current_valid_norm})

    scheduler_text.step()
    scheduler_graph.step()

