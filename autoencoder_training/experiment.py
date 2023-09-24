import wandb

from transformers import AutoModel
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from einops import rearrange
from circuitsvis.activations import text_neuron_activations
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
import openai
from experiment_configs import ExperimentConfig, experiment_config_A

import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, l1_coef):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True)
        )

        self.l1_coef = l1_coef

        self.decoder_weight = nn.Parameter(torch.randn(hidden_size, input_size))
        nn.init.orthogonal_(self.decoder_weight)

    def forward(self, x):
        features = self.encoder(x)

        normalized_decoder_weight = F.normalize(self.decoder_weight, p=2, dim=1)
        reconstruction = torch.matmul(features, normalized_decoder_weight)

        return features, reconstruction

    def decoder(self, features):
        normalized_decoder_weight = F.normalize(self.decoder_weight, p=2, dim=1)

        return torch.matmul(features, normalized_decoder_weight)

def run_experiment(experiment_config: ExperimentConfig):
    '''
    Part 1 of IMDb experiment:
    1. Compute parameter divergence and sorts layers by parameter divergence between m_base and m_rlhf.
    2. Extract activations for the train split of IMDb prefixes.
    3. Train autoencoders on the extracted activations.
    4. Measure loss of the autoencoder on the IMDb test dataset.
    '''

    wandb_project_name = 'Autoencoder training'

    wandb.login()
    run = wandb.init(project=wandb_project_name)

    hyperparameters = experiment_config.hyperparameters
    wandb.log(hyperparameters)

    base_model_name = experiment_config.base_model_name
    policy_model_name = experiment_config.policy_model_name
    
    wandb.log({'base_model_name': base_model_name, 'policy_model_name': policy_model_name})

    train_dataset = load_dataset("imdb", split="train")
    test_dataset = load_dataset("imdb", split="test")

    input_data = preprocess(train_dataset, tokenizer, 960).to(device)
    test_data = preprocess(test_dataset, tokenizer, 960).to(device)

    hyperparameters_clone = hyperparameters.copy()
    sorted_layers = find_layers(m_base, m_rlhf)[:3]
    all_autoencoders = {}

    for layer_index in sorted_layers:
        layer_name = f"layers.{layer_index}.mlp"
        print(f"Layer: {layer_name}")

        for hidden_size in hidden_sizes:
            print(f"Training autoencoder with hidden size: {hidden_size}")
            hyperparameters_clone['hidden_size'] = hidden_size

            autoencoders = feature_representation(m_base, layer_name, input_data, hyperparameters_clone, device)

            if layer_name not in all_autoencoders:
                all_autoencoders[layer_name] = []

            all_autoencoders[layer_name].extend(autoencoders)

            test_activations = get_layer_activations(m_base, layer_name, test_data, device)
            test_activations_tensor = test_activations.detach().clone().to(device)
            test_activations_tensor = test_activations_tensor.squeeze(1)
            test_dataset = TensorDataset(test_activations_tensor)
            test_data_loader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)

            for autoencoder in autoencoders:
                test_loss = measure_performance(autoencoder, test_data_loader, device)
                print(f'Test Loss for Autoencoder with hidden size {hidden_size}: {test_loss:.4f}')


run_experiment(experiment_config=experiment_config_A)
