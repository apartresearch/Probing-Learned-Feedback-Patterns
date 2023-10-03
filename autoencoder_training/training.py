
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import wandb

from network_helper_functions import get_layer_activations
from models.sparse_autoencoder import SparseAutoencoder
from utils.helper_functions import batch


def train_autoencoder(autoencoder, input_texts, hyperparameters, device, label, model, tokenizer, layer_name):
    criterion = nn.MSELoss()
    batch_size = hyperparameters['batch_size']
    optimizer = optim.Adam(autoencoder.parameters(), lr=hyperparameters['learning_rate'])
    num_batches = int(len(input_texts) / batch_size)

    for epoch in range(hyperparameters['num_epochs']):
        all_losses = []
        all_sparsity_losses = []
        all_reconstruction_losses = []
        all_true_sparsity_losses = []

        wandb.define_metric(f"loss_{label}", summary="min")
        wandb.define_metric(f"reconstruction_loss_{label}", summary="min")
        wandb.define_metric(f"sparsity_loss_{label}", summary="min")
        wandb.define_metric(f"true_sparsity_loss_{label}", summary="min")

        for input_batch in tqdm(batch(input_texts, batch_size), total=num_batches):
            activations_batch = get_layer_activations(
                model=model, layer_name=layer_name, input_texts=input_batch, tokenizer=tokenizer,
                device=device, hyperparameters=hyperparameters
            )
            data = activations_batch.to(device)

            optimizer.zero_grad()
            features, reconstruction = autoencoder(data)

            sparsity_loss = autoencoder.l1_coef * torch.norm(features, 1, dim=-1).mean()
            true_sparsity_loss = torch.norm(features, 0, dim=-1).mean()

            reconstruction_loss = criterion(reconstruction, data)
            loss = reconstruction_loss + sparsity_loss

            all_losses.append(loss.cpu().detach().numpy())
            all_reconstruction_losses.append(reconstruction_loss.cpu().detach().numpy())
            all_sparsity_losses.append(sparsity_loss.cpu().detach().numpy())
            all_true_sparsity_losses.append(true_sparsity_loss.cpu().detach().numpy())

            loss.backward()
            optimizer.step()

            wandb.log({
                f"loss_{label}": loss,
                f"normalized_reconstruction_loss_{label}": reconstruction_loss,
                f"sparsity_loss_{label}": sparsity_loss,
                f"true_sparsity_loss_{label}": true_sparsity_loss
            })


        print(f"Epoch [{epoch+1}/{hyperparameters['num_epochs']}], Loss: {loss.item():.4f}")
        print("Reconstruction Loss: ", reconstruction_loss.item())
        print("Sparsity Loss: ", sparsity_loss.item())
        print("True sparsity loss: ", true_sparsity_loss.item())


def train_decoder(autoencoder, data_loader, encoded_data_loader, hyperparameters, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam([autoencoder.decoder_weight], lr=hyperparameters['learning_rate'])

    for epoch in range(hyperparameters['num_epochs']):
        for (encoded_data_batch, original_data_batch) in zip(encoded_data_loader, data_loader):
            encoded_data = encoded_data_batch.to(device)
            original_data = original_data_batch.to(device)

            optimizer.zero_grad()
            features = autoencoder.decoder(encoded_data)
            reconstruction_loss = criterion(features, original_data)

            loss = reconstruction_loss
            loss.backward()
            optimizer.step()

        print(f"Encoder Epoch [{epoch+1}/{hyperparameters['num_epochs']}], Loss: {loss.item():.4f}")

def train_encoder(autoencoder, data_loader, hyperparameters, device):
    optimizer = optim.Adam(autoencoder.encoder.parameters(), lr=hyperparameters['learning_rate'])

    for epoch in range(hyperparameters['num_epochs']):
        for batch in data_loader:
            data = batch.to(device)

            optimizer.zero_grad()
            features = autoencoder.encoder(data)

            sparsity_loss = autoencoder.l1_coef * torch.norm(features, 1, dim=-1).mean()
            loss = sparsity_loss

            loss.backward()
            optimizer.step()

        print(f"Encoder Epoch [{epoch+1}/{hyperparameters['num_epochs']}], Loss: {loss.item():.4f}")

def feature_representation(
        model, tokenizer, layer_name, input_texts, hyperparameters,
        device, num_autoencoders=1, label='default'
):
    '''
    base_activations = get_layer_activations_batched(m_base, layer_name, input_data, device)
    base_activations_tensor = base_activations.detach().clone()
    base_activations_tensor = base_activations_tensor.squeeze(1)

    input_size = base_activations_tensor.size(-1)
    base_dataset = TensorDataset(base_activations_tensor)
    base_data_loader = DataLoader(base_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
    '''
    batch_size = hyperparameters['batch_size']
    print('\nBatch size is ' + str(hyperparameters['batch_size']) + '\n')

    # Get batch without popping
    first_batch = input_texts[:batch_size].copy()

    first_activations_tensor = get_layer_activations(
        model=model, tokenizer=tokenizer, layer_name=layer_name, input_texts=first_batch,
        device=device, hyperparameters=hyperparameters
    ).detach().clone().squeeze(1)

    input_size = first_activations_tensor.size(-1)
    print(f'Input size is {input_size}.')

    autoencoders = []

    for i in range(num_autoencoders):
        local_label = f'{layer_name}_{label}_{i}'
        hidden_size = input_size * hyperparameters['hidden_size_multiple']
        autoencoder = SparseAutoencoder(input_size, hidden_size=hidden_size, l1_coef=hyperparameters['l1_coef']).to(device)
        train_autoencoder(
            autoencoder=autoencoder, input_texts=input_texts, hyperparameters=hyperparameters,
            device=device, label=local_label,
            model=model, tokenizer=tokenizer, layer_name=layer_name)
        autoencoders.append(autoencoder)

    return autoencoders