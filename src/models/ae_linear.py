import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearAutoencoder(nn.Module):
    """
    A linear autoencoder model as described in the Russo et al. paper.
    Encoder: 140 -> 32 -> 8 (latent)
    Decoder: 8 -> 32 -> 140
    """

    def __init__(self, input_dim=140, hidden_dim1=32, latent_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_layer1 = nn.Linear(input_dim, hidden_dim1)
        self.encoder_layer2 = nn.Linear(hidden_dim1, latent_dim)

        # Decoder
        self.decoder_layer1 = nn.Linear(latent_dim, hidden_dim1)
        self.decoder_layer2 = nn.Linear(hidden_dim1, input_dim)

    def forward(self, x):
        # Ensure input is flattened if it has an extra channel dimension
        # Expected input shape: (batch_size, seq_len, 1)
        if x.dim() == 3 and x.shape[2] == 1:
            x = x.squeeze(2)  # Flatten to (batch_size, seq_len)
        elif x.dim() != 2:
            raise ValueError(
                f"Input tensor must be 2D (batch_size, seq_len) or 3D (batch_size, seq_len, 1), but got {x.dim()} dimensions.")

        # Encoder forward pass, with RELU as the activation func.
        encoded = F.relu(self.encoder_layer1(x))
        latent = self.encoder_layer2(encoded)

        # Decoder forward pass
        decoded = F.relu(self.decoder_layer1(latent))
        reconstructed = self.decoder_layer2(decoded)

        return reconstructed, latent
