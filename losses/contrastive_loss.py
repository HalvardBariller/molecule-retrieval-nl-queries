import torch
import numpy as np

CE = torch.nn.CrossEntropyLoss()
def contrastive_loss(v1, v2):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  labels = torch.arange(logits.shape[0], device=v1.device)
  return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)


  
def contrastive_loss_with_cosine(v1, v2):
  """
  Contrastive loss with cosine similarity
  ----------------
  This loss adds to the classical contrastive loss the cosine similarity between the two vectors.
  It also includes a temperature parameter to scale the logits, following the CLIP paper. 
  """
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1)) * np.exp(0.07)
  v1_norm = v1 / v1.norm(dim=1)[:, None]
  v2_norm = v2 / v2.norm(dim=1)[:, None]
  logits_cosine = torch.matmul(v1_norm,torch.transpose(v2_norm, 0, 1)) * np.exp(0.07)
  labels = torch.arange(logits.shape[0], device=v1.device)
  return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels) + CE(logits_cosine, labels) + CE(torch.transpose(logits_cosine, 0, 1), labels)
