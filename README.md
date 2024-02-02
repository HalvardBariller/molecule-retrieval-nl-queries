# molecule-retrieval-nl-queries
Molecule Retrieval with Natural Language Queries. Challenge for the MVA Class "Advanced Learning for Text and Graph Data", 2023.

# Description of runnable files

In all these runnable files, hyperparameters, model definitions and all the code that needs
to be changed according to your needs are indicated with a comment header, such as:
```
###########################
#      GRAPH ENCODER      #
###########################
# Define your graph encoder here. Below are some of the encoders that we tried.
```

## `main.py`

Trains a model. The model to train including hyperparameters is defined in this file (but the actual implementation is split into several files).
The results are logged on [Weights and Biases](https://wandb.ai). Can also resume training from a checkpoint file.

Usage: `python main.py [-n run_name_for_wandb] [-m path/to/pretrained_model.pt]` (make sure that the right model is defined in the code before using a pretrained model)

## `eval.py`

Computes predictions on the test set for a single pre-trained model.

Usage: after defining the model corresponding to your checkpoint in `eval.py`, run: `python eval.py path/to/model.pt`

## `average_predictions.py`

Computes predictions by aggregating several models in two ways:
- by summing the normalized average similarity scores of each model,
- by summing the ranks computed with the similarity scores of each model (hard voting).

Usage: define the models and set the checkpoint paths directly in the code, then run `python average_predictions.py`.