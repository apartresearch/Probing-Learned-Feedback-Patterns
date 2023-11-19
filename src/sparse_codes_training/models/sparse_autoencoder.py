"""
A sparse autoencoder, trained on activations of an LLM on a dataset.
"""

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from torch import nn
from torch import optim
from tqdm import tqdm
from typing import List

from sparse_codes_training.network_helper_functions import get_layer_activations
from utils.transformer_utils import batch

class SparseAutoencoder(nn.Module):
    """
    This autoencoder is trained on activations of a LLM on a dataset.
    """
    def __init__(self, input_size: int, hidden_size: int, l1_coef: float, weights_tied=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.weights_tied = weights_tied

        self.kwargs = {'input_size': input_size, 'hidden_size': hidden_size, 'l1_coef': l1_coef}
        self.l1_coef = float(l1_coef)

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU()
        )

        self.decoder = nn.Linear(self.hidden_size, self.input_size, bias=True)

        # Initialize the linear layers
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes weights via xavier uniform.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Noramlizes encoder weight, applies encoder, and then the decoder.
        """
        self.encoder[0].weight.data = F.normalize(self.encoder[0].weight, p=2, dim=1)
        features = self.encoder(x)

        if self.weights_tied:
            encoder_weight = self.encoder[0].weight.t()
            reconstruction = torch.matmul(features, encoder_weight) + self.decoder.bias
        else:
            reconstruction = self.decoder(features)

        return features, reconstruction


    def train_model(
            self, input_texts: List[str], hyperparameters: dict, model_device: str,
            autoencoder_device: str, label: str, model, tokenizer, layer_name: str
    ):
        """
        Train on the activations on texts.
        """
        criterion = nn.MSELoss()
        batch_size = hyperparameters['batch_size']
        optimizer = optim.Adam(self.parameters(), lr=hyperparameters['learning_rate'])
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
                    device=model_device, hyperparameters=hyperparameters
                )
                data = activations_batch.to(autoencoder_device)

                optimizer.zero_grad()
                features, reconstruction = self.forward(data)

                sparsity_loss = self.l1_coef * torch.norm(features, 1, dim=-1).mean()
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


            avg_loss = np.average(all_losses)
            avg_reconstruction_loss = np.average(all_reconstruction_losses)
            avg_sparsity_loss = np.average(all_sparsity_losses)
            avg_true_sparsity_loss = np.average(all_true_sparsity_losses)

            print(f"Epoch [{epoch+1}/{hyperparameters['num_epochs']}], Loss: {avg_loss:.4f}")
            print(f"Avg. reconstruction Loss: {avg_reconstruction_loss}")
            print(f"Avg. sparsity Loss: {avg_sparsity_loss}")
            print(f"Avg. true sparsity loss: {avg_true_sparsity_loss}")
