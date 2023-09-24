import wandb

from transformers import AutoModel
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch

from experiment_configs import ExperimentConfig, experiment_config_A

import torch.nn as nn

from models.sparse_autoencoder import SparseAutoencoder
from network_helper_functions import find_layers, get_layer_activations
from training import feature_representation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def preprocess(dataset, tokenizer, limit):
    texts = [x['text'] for x in dataset.select(range(limit))]
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    return inputs

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

    m_base = AutoModel.from_pretrained(base_model_name).to(device)
    m_rlhf = AutoModel.from_pretrained(policy_model_name).to(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer_rlhf = AutoTokenizer.from_pretrained(policy_model_name)
    tokenizer_rlhf.pad_token = tokenizer_rlhf.eos_token
    train_dataset = load_dataset("imdb", split="train")
    test_dataset = load_dataset("imdb", split="test")

    input_data = preprocess(train_dataset, tokenizer, 960).to(device)
    test_data = preprocess(test_dataset, tokenizer, 960).to(device)

    train_dataset_base = preprocess(load_dataset("imdb", split="train"), tokenizer, 96)
    input_data_base = {'input_ids': train_dataset_base['input_ids'].to(device)}
    train_dataset_rlhf = preprocess(load_dataset("imdb", split="train"), tokenizer_rlhf, 96)
    input_data_rlhf = {'input_ids': train_dataset_rlhf['input_ids'].to(device)}

    sorted_layers = find_layers(m_base, m_rlhf)

    autoencoders_base_big = {}
    autoencoders_base_small = {}

    autoencoders_rlhf_big = {}
    autoencoders_rlhf_small = {}

    hidden_sizes = sorted(hyperparameters['hidden_sizes']).copy()
    small_hidden_size = hidden_sizes[0]

    for layer_index in sorted_layers:
        print(layer_index)
        for hidden_size in hidden_sizes:
            print(hidden_size)

            hyperparameters_copy = hyperparameters.copy()
            hyperparameters_copy['hidden_size'] = hidden_size

            autoencoder_base = feature_representation(
                m_base, f'layers.{sorted_layers[layer_index]}.mlp',
                input_data_base, hyperparameters_copy, device
            )

            target_autoencoders_base = autoencoders_base_big if hidden_size > small_hidden_size else autoencoders_base_small

            target_autoencoders_base[str(layer_index)] = autoencoder_base

            autoencoder_rlhf = feature_representation(m_rlhf, f'layers.{sorted_layers[layer_index]}.mlp',
                                                      input_data_rlhf, hyperparameters_copy, device)

            target_autoencoders_rlhf = autoencoders_rlhf_big if hidden_size > small_hidden_size else autoencoders_rlhf_small

            target_autoencoders_rlhf[str(layer_index)] = autoencoder_rlhf


run_experiment(experiment_config=experiment_config_A)
