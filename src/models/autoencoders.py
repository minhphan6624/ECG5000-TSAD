import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim=140):
        super().__init()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLu(),
            nn.Linear(64, 32)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
