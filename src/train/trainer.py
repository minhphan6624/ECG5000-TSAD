import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os


# Function to add Gaussian noise to a Pytorch Tensor
def add_gaussian_noise(data, stddev=0.05):
    if not isinstance(data, torch.Tensor):
        raise TypeError("Input data must be a torch.Tensor")

    noise = torch.randn_like(data) * stddev
    return data + noise


class Trainer:
    """
    A generic trainer class for PyTorch models.
    Handles training loop, validation, early stopping, and model saving.
    """

    def __init__(self, model, device, train_loader, val_loader, criterion, optimizer, noise_stddev=0.05, checkpoint_dir="../checkpoints", patience=3):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion  # Loss function
        self.optimizer = optimizer
        self.noise_stddev = noise_stddev

        self.best_val_loss = float('inf')
        self.patience = patience
        self.bad_epochs = 0

        # Ensure checkpoint directory exists
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(
            self.checkpoint_dir, "best_model.pt")  # Default checkpoint path

    def run_epoch(self, train=True):
        """Runs a single epoch for training or validation."""
        if train:
            self.model.train()  # Set model to training mode
            loader = self.train_loader
            prefix = "Train"
        else:
            self.model.eval()   # Set model to evaluation mode
            loader = self.val_loader
            prefix = "Val"

        total_loss = 0.0
        num_samples = 0

        for batch in loader:
            # Handle cases where loader might return only features or features+labels
            if isinstance(batch, (list, tuple)):
                # Store the label, might need it later
                inputs = batch[0].to(self.device)
                original_inputs = inputs
            else:
                inputs = batch.to(self.device)
                original_inputs = inputs  # Store the original input for evaluating denoising AE

            # Add noise for denoising autoencoders (only during training)
            if train:
                noisy_inputs = add_gaussian_noise(
                    inputs, stddev=self.noise_stddev)
            else:
                noisy_inputs = inputs

            # Forward pass
            reconstructed, latent = self.model(noisy_inputs)

            # Calculate loss against the original, clean data
            loss = self.criterion(reconstructed, original_inputs)

            # Backward pass and optimization if training
            if train:
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping as per paper's example (optional)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)

        avg_loss = total_loss / num_samples
        print(f"{prefix} Epoch {self.current_epoch+1}: Avg. Loss = {avg_loss:.6f}")
        return avg_loss

    def train(self, num_epochs=200):
        """Starts the training process."""

        print(f"Starting training for {num_epochs} epochs...")
        print(f"Using device: {self.device}")
        print(
            f"Optimizer: {type(self.optimizer).__name__}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        print(f"Loss function: {type(self.criterion).__name__}")
        print(f"Noise stddev: {self.noise_stddev}")
        print(f"Early stopping patience: {self.patience}")

        # Training loop starts
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            train_loss = self.run_epoch(train=True)
            val_loss = self.run_epoch(train=False)

            # Early stopping logic
            # Use a small tolerance to avoid stopping due to minor fluctuations
            if val_loss < self.best_val_loss - 1e-5:
                self.best_val_loss = val_loss
                self.bad_epochs = 0
                # Save the model checkpoint
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(
                    f"Validation loss improved. Saving model to {self.checkpoint_path}")
            else:
                self.bad_epochs += 1

            if self.bad_epochs >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print("Training finished.")
        print(f"Best validation loss achieved: {self.best_val_loss:.6f}")
        # Load the best model weights
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        print(f"Loaded best model from {self.checkpoint_path}")

        return self.best_val_loss, epoch + 1  # Return best loss and total epochs run
