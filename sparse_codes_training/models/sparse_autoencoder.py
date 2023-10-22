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

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU()
        )

        # Decoder layers
        self.decoder = nn.Linear(self.hidden_size, self.input_size)

        # Initialize the linear layers
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        encoded = self.encoder(x)
        reconstruction = self.decoder(encoded)
        return encoded, reconstruction

    def forward(self, x):
        self.encoder[0].weight.data = F.normalize(self.encoder[0].weight, p=2, dim=1)

        features = self.encoder(x)

        reconstruction = F.linear(features, normalized_encoder_weight.t(), self.decoder_bias)

        return features, reconstruction