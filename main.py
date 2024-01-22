from dataloader.dataloader import GraphTextDataset, GraphDataset, TextDataset
#from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader 
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.metrics import label_ranking_average_precision_score
from models.Model import Model
from models.graph_encoders import GraphEncoder, GraphormerEncoder
from models.text_encoders import TextEncoder
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
import time
import os
import sys
import pandas as pd
import wandb
from utils import compute_embeddings_valid, compute_similarities_LRAP, make_predictions

from losses.contrastive_loss import contrastive_loss

import warnings
warnings.simplefilter("ignore", category=UserWarning)

## Model
model_name = 'distilbert-base-uncased'
# model_name = 'allenai/scibert_scivocab_uncased'


tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

## Hyperparameters
wandb.init(
        project="2nd run - ALTEGRAD",
        name=sys.argv[1] if len(sys.argv) >= 2 else None,
        config={
            "epochs": 20,
            "batch_size": 64,
            "lr": 2e-5
            })
config = wandb.config

# nb_epochs = 5
# batch_size = 32
# learning_rate = 2e-5
nb_epochs = wandb.config.epochs
batch_size = wandb.config.batch_size
learning_rate = wandb.config.lr


## Early Stopping Parameters
patience = 7  # Number of epochs to wait for improvement
# delta = 0.02  # Minimum change to qualify as an improvement
patience_counter = 0


num_workers = 12

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers, pin_memory = True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers, pin_memory = True)

graph_encoder = GraphEncoder(num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300) # nout = bert model hidden dim
text_encoder = TextEncoder(model_name)
model = Model(graph_encoder, text_encoder)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                betas=(0.9, 0.98),
                                weight_decay=0.01)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=nb_epochs)
scaler = GradScaler()


epoch = 0
loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 50
best_validation_loss = 1000000

for i in range(nb_epochs):
    print('-----EPOCH{}-----'.format(i+1))
    model.train()
    for batch in train_loader:
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch

        with autocast(dtype=torch.float16):
        
            x_graph, x_text = model(graph_batch.to(device), 
                                    input_ids.to(device), 
                                    attention_mask.to(device))
        
            
            current_loss = contrastive_loss(x_graph, x_text)
            
        optimizer.zero_grad()
        scaler.scale(current_loss).backward()
        #current_loss.backward()
        scaler.step(optimizer)
        scaler.update()
        #optimizer.step()
        loss += current_loss.item()
        
        count_iter += 1
        if count_iter % printEvery == 0:
            time2 = time.time()
            print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                        time2 - time1, loss/printEvery))
            losses.append(loss)
            loss = 0 
            
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
        current_loss = contrastive_loss(x_graph, x_text)   
        val_loss += current_loss.item()

    # Early stopping check
    if  val_loss < best_validation_loss:
        patience_counter = 0
    else:
        patience_counter += 1
    best_validation_loss = min(best_validation_loss, val_loss)
    print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)) )

    # LRAP computation
    graph_embeddings, text_embeddings, y_true = compute_embeddings_valid(model, val_dataset, device, batch_size)
    lrap_current_valid = compute_similarities_LRAP(graph_embeddings, text_embeddings, y_true)
    print("Validation LRAP Score:", lrap_current_valid)

    # Save model checkpoint
    if best_validation_loss==val_loss:
        print('validation loss improved saving checkpoint...')
        # Remove previous checkpoints
        for file in os.listdir('./'):
            if 'model.pt' in file:
                os.remove(file)
        save_path = os.path.join('./', str(i)+'model.pt')
        torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_accuracy': val_loss,
        'loss': loss,
        }, save_path)
        print('checkpoint saved to: {}'.format(save_path))

    wandb.log({"loss": loss, 
               "val_loss": val_loss, 
               "best_val_loss": best_validation_loss,
               #"LRAP_train": lrap_current_train,
              "LRAP_valid": lrap_current_valid})
    
    if patience_counter >= patience:
        print(f'Early stopping triggered after epoch {i+1}. Ending training.')
        break

    scheduler.step()


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
# solution.to_csv('submission.csv', index=False)
        
make_predictions(text_embeddings, graph_embeddings)