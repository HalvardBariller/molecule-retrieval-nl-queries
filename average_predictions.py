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
from utils import compute_embeddings_valid, compute_similarities_LRAP, make_predictions, prepare_submission_file
from tqdm import tqdm
import gc

#import warnings
#warnings.simplefilter("ignore", category=UserWarning)


batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


aggregated_predictions_val = []
aggregated_predictions_test = []
labels_val = []

torch.cuda.empty_cache()
gc.collect()


########################
#        MODELS        #
########################
# Define the models below. Do not forget to update the `models` list if you change the number of models.
# "model_name" refers to the name of the text encoder model on Hugging Face.
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

# Sanity check for labels_val
for i in range(len(labels_val)):
    assert np.array_equal(labels_val[i], labels_val[0]), "labels_val[{}] is not equal to labels_val[0]".format(i)



######### Soft Voting Ensemble model (sum of similarities) #########

ensemble_predictions_val = np.zeros_like(predictions_val)

for predictions_val in aggregated_predictions_val:
    ensemble_predictions_val += predictions_val

print("Ensemble model trained")
print("LRAP val:", label_ranking_average_precision_score(labels_val[0], ensemble_predictions_val))

ensemble_predictions_test = np.zeros_like(predictions_test)
for predictions_test in aggregated_predictions_test:
    ensemble_predictions_test += predictions_test


####### Hard Voting Ensemble model (sum of ranks) #########

hard_ensemble_predictions_val = np.zeros_like(predictions_val)

for predictions_val in aggregated_predictions_val:
    sorted_predictions_val = np.argsort(predictions_val, axis=1)
    ranks = np.argsort(sorted_predictions_val, axis=1)
    hard_ensemble_predictions_val += ranks

print("Hard Voting Ensemble model trained")
print("LRAP val:", label_ranking_average_precision_score(labels_val[0], hard_ensemble_predictions_val))

hard_ensemble_predictions_test = np.zeros_like(predictions_test)
for predictions_test in aggregated_predictions_test:
    sorted_predictions_test = np.argsort(predictions_test, axis=1)
    ranks = np.argsort(sorted_predictions_test, axis=1)
    hard_ensemble_predictions_test += ranks



# Best model
    
if label_ranking_average_precision_score(labels_val[0], ensemble_predictions_val) > label_ranking_average_precision_score(labels_val[0], hard_ensemble_predictions_val):
    print("Best model: Soft Ensemble model")
    prepare_submission_file(ensemble_predictions_test, "submission_ensemble")
else:
    print("Best model: Hard Voting Ensemble model")
    prepare_submission_file(hard_ensemble_predictions_test, "submission_ensemble_hard_voting")

print("Predictions ready!")