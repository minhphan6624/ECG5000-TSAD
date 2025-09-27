import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAE(nn.Module):
    """
    LSTM Autoencoder model as described in the Russo et al. paper.
    Architecture:
    - Encoder: 3 unidirectional LSTM layers
    - Latent space: 8-dimensional
    - Decoder: 3 unidirectional LSTM layers
    Input shape: (batch_size, sequence_length, input_features)
    """
    def __init__(self, sequence_length=140, input_features=1, hidden_dim=64, latent_dim=8, num_layers=3):
        super().__init__()
        self.sequence_length = sequence_length
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_features, # Corrected: number of features per time step
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        # Linear layer to map the final encoder hidden state to the latent space
        self.encoder_to_latent = nn.Linear(hidden_dim, latent_dim)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim, # Input to decoder LSTM is the latent representation
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        # Linear layer to map decoder LSTM output back to the original input features
        self.decoder_to_output = nn.Linear(hidden_dim, input_features) # Corrected: output features

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_features)

        # Encoder forward pass
        encoder_outputs, encoder_hidden = self.encoder_lstm(x)

        # Get the last hidden state of the encoder
        last_hidden_state = encoder_hidden[0][-1, :, :] # shape: (batch, hidden_dim)

        # Map to latent space
        latent = self.encoder_to_latent(last_hidden_state) # shape: (batch, latent_dim)

        # Prepare latent vector for decoder input
        # Replicate the latent vector across the sequence length
        seq_len = x.size(1)
        decoder_input = latent.unsqueeze(1).repeat(1, seq_len, 1) # shape: (batch, seq_len, latent_dim)

        # Decoder forward pass
        decoder_outputs, _ = self.decoder_lstm(decoder_input)

        # Map decoder outputs back to the original input features
        reconstructed = self.decoder_to_output(decoder_outputs) # shape: (batch, seq_len, input_features)

        return reconstructed, latent

