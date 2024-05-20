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

from reward_analyzer.sparse_codes_training.experiment_helpers.layer_activations_handler import LayerActivationsHandler
from reward_analyzer.utils.transformer_utils import batch

class SparseAutoencoder(nn.Module):
    """
    This autoencoder is trained on activations of a LLM on a dataset.
    """
    def __init__(self, input_size: int, hidden_size: int, l1_coef: float, tied_weights=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.tied_weights = tied_weights

        self.kwargs = {'input_size': input_size, 'hidden_size': hidden_size, 'l1_coef': l1_coef}
        self.l1_coef = float(l1_coef)

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU()
        )

        if self.tied_weights:
            print('\nNo explicit decoder created, only bias vector.')
            self.bias = nn.Parameter(torch.zeros(self.input_size))
        else:
            print('\nCreating explicit decoder matrix.')
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
        Normalizes encoder weight, applies encoder, and then the decoder.
        """
        x = x.to(dtype=self.encoder[0].weight.dtype)
        self.encoder[0].weight.data = F.normalize(self.encoder[0].weight, p=2, dim=1)
        features = self.encoder(x)

        if self.tied_weights:
            encoder_weight = self.encoder[0].weight
            reconstruction = torch.matmul(features, encoder_weight) + self.bias
        else:
            reconstruction = self.decoder(features)

        return features, reconstruction


    def train_model(
            self, input_texts: List[str], hyperparameters: dict, model_device: str,
            autoencoder_device: str, label: str, activations_handler: LayerActivationsHandler, tokenizer, layer_name: str
    ):
        """
        Train on the activations on texts.
        """
        criterion = nn.MSELoss()
        batch_size = hyperparameters['batch_size']
        optimizer = optim.Adam(self.parameters(), lr=hyperparameters['learning_rate'])
        num_batches = int(len(input_texts) / batch_size)

        wandb.define_metric(f"loss_{label}", summary="min")
        wandb.define_metric(f"reconstruction_loss_{label}", summary="min")
        wandb.define_metric(f"sparsity_loss_{label}", summary="min")
        wandb.define_metric(f"true_sparsity_loss_{label}", summary="min")

        wandb.define_metric("base_mmcs_results", summary="min")
        wandb.define_metric("rlhf_mmcs_results", summary="min")

        for epoch in range(hyperparameters['num_epochs']):
            all_losses = []
            all_sparsity_losses = []
            all_reconstruction_losses = []
            all_true_sparsity_losses = []


            for input_batch in tqdm(batch(input_texts, batch_size), total=num_batches):

                activations_batch = activations_handler.get_layer_activations(
                    layer_name=layer_name, input_texts=input_batch, tokenizer=tokenizer,
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

            print(f"Epoch [{epoch+1}/{hyperparameters['num_epochs']}] on {label}, Loss: {avg_loss:.4f}")
            print(f"Final reconstruction Loss on {label}: {all_reconstruction_losses[-1]}")
            print(f"Final sparsity Loss on {label}: {all_sparsity_losses[-1]}")
            print(f"Final true sparsity loss on {label}: {all_true_sparsity_losses[-1]}")
