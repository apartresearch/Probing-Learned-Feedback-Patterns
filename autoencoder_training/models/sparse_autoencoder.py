import torch
import torch.nn as nn

import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, l1_coef):
        super(SparseAutoencoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.kwargs = {'input_size': input_size, 'hidden_size': hidden_size, 'l1_coef': l1_coef}
        self.l1_coef = float(l1_coef)


        self.encoder_weight = nn.Parameter(torch.randn(self.hidden_size, input_size))
        nn.init.orthogonal_(self.encoder_weight)

        self.encoder_bias = nn.Parameter(torch.zeros(self.hidden_size))
        self.decoder_bias = nn.Parameter(torch.zeros(input_size))

    def forward(self, x):
        normalized_encoder_weight = F.normalize(self.encoder_weight, p=2, dim=1)

        features = F.linear(x, normalized_encoder_weight, self.encoder_bias)
        features = F.relu(features)

        reconstruction = F.linear(features, normalized_encoder_weight.t(), self.decoder_bias)

        return features, reconstruction