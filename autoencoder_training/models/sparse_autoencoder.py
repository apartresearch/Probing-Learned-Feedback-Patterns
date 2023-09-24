import torch
import torch.nn as nn

import torch.nn.functional as F

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