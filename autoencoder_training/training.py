
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import torch.nn as nn

from network_helper_functions import get_layer_activations
from models.sparse_autoencoder import SparseAutoencoder

def train_autoencoder(autoencoder, data_loader, hyperparameters, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=hyperparameters['learning_rate'])

    for epoch in range(hyperparameters['num_epochs']):
        for batch in data_loader:
            data = batch[0].to(device)

            optimizer.zero_grad()
            features, reconstruction = autoencoder(data)

            sparsity_loss = autoencoder.l1_coef * torch.norm(features, 1, dim=-1).mean()

            reconstruction_loss = criterion(reconstruction, data)
            loss = reconstruction_loss + sparsity_loss

            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{hyperparameters['num_epochs']}], Loss: {loss.item():.4f}")
        print("Reconstruction Loss: ", reconstruction_loss.item())
        print("Sparsity Loss: ", sparsity_loss.item())


def train_decoder(autoencoder, data_loader, encoded_data_loader, hyperparameters, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam([autoencoder.decoder_weight], lr=hyperparameters['learning_rate'])

    for epoch in range(hyperparameters['num_epochs']):
        for (encoded_data_batch, original_data_batch) in zip(encoded_data_loader, data_loader):
            encoded_data = encoded_data_batch[0].to(device)
            original_data = original_data_batch[0].to(device)

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
            data = batch[0].to(device)

            optimizer.zero_grad()
            features = autoencoder.encoder(data)

            sparsity_loss = autoencoder.l1_coef * torch.norm(features, 1, dim=-1).mean()
            loss = sparsity_loss

            loss.backward()
            optimizer.step()

        print(f"Encoder Epoch [{epoch+1}/{hyperparameters['num_epochs']}], Loss: {loss.item():.4f}")

def feature_representation(m_base, layer_name, input_data, hyperparameters, device, num_autoencoders=1):
    base_activations = get_layer_activations(m_base, layer_name, input_data, device)
    base_activations_tensor = base_activations.detach().clone()
    base_activations_tensor = base_activations_tensor.squeeze(1)

    input_size = base_activations_tensor.size(1)

    base_dataset = TensorDataset(base_activations_tensor)
    base_data_loader = DataLoader(base_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)

    autoencoders = []
    for i in range(num_autoencoders):
        autoencoder = SparseAutoencoder(input_size, hyperparameters['hidden_size'], hyperparameters['l1_coef']).to(device)
        train_autoencoder(autoencoder, base_data_loader, hyperparameters, device)
        autoencoders.append(autoencoder)

    return autoencoders