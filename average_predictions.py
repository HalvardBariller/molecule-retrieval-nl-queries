from dataloader.dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.loader import DataLoader 
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.metrics import label_ranking_average_precision_score
from models.Model import Model
from models.expert_model import ExpertModel
from models.graph_encoders import GINEncoder, GraphormerEncoder
from models.text_encoders import TextEncoder
import numpy as np
from transformers import AutoTokenizer
import torch
import os
import pandas as pd
from utils import compute_embeddings_valid, compute_similarities_LRAP, make_predictions, prepare_submission_file
import argparse
from tqdm import tqdm

#from losses.contrastive_loss import contrastive_loss

import warnings
warnings.simplefilter("ignore", category=UserWarning)


batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#aggregated_predictions_train = []
aggregated_predictions_val = []
aggregated_predictions_test = []
#labels_train = []
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
    #train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)
    model = Model(graph_encoder, text_encoder)
    model.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded:", model_path)
    model.eval()
    #graph_embeddings_train, text_embeddings_train, y_true_train = compute_embeddings_valid(model, train_dataset, device, batch_size)
    #print("Train embeddings computed")
    ########################################
    ## TOO HEAVY ?
    #labels_train.append(y_true_train)
    #predictions_train = make_predictions(graph_embeddings_train, text_embeddings_train, save_file=False)
    ## TEST WITH RESTRICED TRAIN SET
    #labels_train.append(y_true_train[:5000])
    #predictions_train = make_predictions(graph_embeddings_train[:5000], text_embeddings_train[:5000], save_file=False)
    ########################################
    #aggregated_predictions_train.append(predictions_train)
    #print("Train predictions computed")
    graph_embeddings_val, text_embeddings_val, y_true_val = compute_embeddings_valid(model, val_dataset, device, batch_size)
    print("Val embeddings computed")
    labels_val.append(y_true_val)
    predictions_val = make_predictions(graph_embeddings_val, text_embeddings_val, save_file=False)
    aggregated_predictions_val.append(predictions_val)

    # Test predictions
    graph_model = model.get_graph_encoder()
    text_model = model.get_text_encoder()
    test_cids_dataset = GraphDataset(root='./data/', gt=gt, split='test_cids')
    test_text_dataset = TextDataset(file_path='./data/test_text.txt', tokenizer=tokenizer)
    idx_to_cid = test_cids_dataset.get_idx_to_cid()
    test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)
    graph_embeddings = []
    print("Computing graph embeddings test...")
    for batch in tqdm(test_loader):
        for output in graph_model(batch.to(device)):
            graph_embeddings.append(output.tolist())
    test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
    text_embeddings = []
    print("Computing text embeddings test...")
    for batch in tqdm(test_text_loader):
        for output in text_model(batch['input_ids'].to(device), 
                                attention_mask=batch['attention_mask'].to(device)):
            text_embeddings.append(output.tolist())
    predictions_test = make_predictions(graph_embeddings, text_embeddings, save_file=False)
    aggregated_predictions_test.append(predictions_test)

print("Predictions computed, training ensemble model...")

# Sanity checks for labels
# for i in range(len(labels_train)):
#     assert np.array_equal(labels_train[i], labels_train[0])
for i in range(len(labels_val)):
    assert np.array_equal(labels_val[i], labels_val[0]), "labels_val[{}] is not equal to labels_val[0]".format(i)

######### Naive Ensemble model (average of predictions) #########

#ensemble_predictions_train = np.zeros_like(predictions_train)
ensemble_predictions_val = np.zeros_like(predictions_val)

# for predictions_train in aggregated_predictions_train:
#     ensemble_predictions_train += predictions_train
# ensemble_predictions_train /= len(aggregated_predictions_train)

for predictions_val in aggregated_predictions_val:
    ensemble_predictions_val += predictions_val
ensemble_predictions_val /= len(aggregated_predictions_val)

print("Ensemble model trained")
print("LRAP val:", label_ranking_average_precision_score(labels_val[0], ensemble_predictions_val))

ensemble_predictions_test = np.zeros_like(predictions_test)
for predictions_test in aggregated_predictions_test:
    ensemble_predictions_test += predictions_test
ensemble_predictions_test /= len(aggregated_predictions_test)

prepare_submission_file(ensemble_predictions_test, "submission_ensemble")

print("Predictions ready!")

######### Expert model #########


# def prepare_batch_aggregation(predictions):
#     M = len(predictions)
#     N, H = predictions[0].shape
#     batches = []
#     for i in range(N):
#         instance_data = [arr[i, :] for arr in predictions]
#         instance_batch = np.stack(instance_data, axis=0)
#         batches.append(instance_batch)
#     batch_data = np.stack(batches, axis=0)
#     tensor_input = torch.tensor(batch_data, dtype=torch.float32)
#     return tensor_input

# predictions_train = prepare_batch_aggregation(aggregated_predictions_train)
# predictions_val = prepare_batch_aggregation(aggregated_predictions_val)

# CE = torch.nn.CrossEntropyLoss()

# # Train expert model
# expert_model = ExpertModel(num_models=len(aggregated_predictions_train))
# expert_model.to(device)
# optimizer = torch.optim.Adam(expert_model.parameters(), lr=0.001)
# num_epochs = 100
# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#     output = expert_model(predictions_train.to(device))
#     expected = torch.tensor(labels_train[0], dtype=torch.long).to(device)
#     # Expected will work ? Need to check if I want solely the index of the max value
#     loss = CE(output, expected)
#     loss.backward()
#     optimizer.step()
#     print("Epoch", epoch, "loss:", loss.item())



