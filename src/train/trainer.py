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

    def __init__(self, model, device, train_loader, val_loader, criterion, optimizer, noise_stddev=0.05, checkpoint_dir="checkpoints"):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion  # Loss function
        self.optimizer = optimizer
        self.noise_stddev = noise_stddev

        self.best_val_loss = float('inf')
        self.patience = 3
        self.bad_epochs = 0

        # Ensure checkpoint directory exists
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(
            self.checkpoint_dir, "best_model.pt")  # Default checkpoint path

    def _run_epoch(self, train=True):
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
                # Store the label, just in case hihi
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

            if train:
                # Backward pass and optimization
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

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            train_loss = self._run_epoch(train=True)
            val_loss = self._run_epoch(train=False)

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

    def evaluate(self, test_loader, test_labels, threshold_fn, device):
        """
        Evaluates the trained model on the test set.
        This method would typically be in src/eval/metrics.py or similar,
        but is included here for completeness of the training process.
        """
        print("\nEvaluating model on test set...")
        self.model.eval()
        reconstruction_errors = []
        true_labels = []

        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                inputs = inputs.to(device)

                # For evaluation, we don't add noise to the input
                reconstructed, _ = self.model(inputs)

                # Calculate reconstruction error (e.g., Mean Absolute Error or MSE)
                # Paper uses MSE for loss, but MAE is often used for error calculation
                # Let's stick to MAE for error calculation as in the notebook example
                # If paper implies MSE for error, this needs adjustment.
                # The notebook used .abs().mean(dim=(1,2)) which is MAE per sample.
                errors = torch.abs(
                    reconstructed - inputs).mean(dim=(1, 2)).cpu().numpy()

                reconstruction_errors.extend(errors)
                true_labels.extend(labels.cpu().numpy())

        reconstruction_errors = np.array(reconstruction_errors)
        true_labels = np.array(true_labels)

        # Use the provided thresholding function to get predictions
        threshold, predictions = threshold_fn(
            reconstruction_errors, true_labels)

        # Calculate metrics (e.g., accuracy, precision, recall, F1, ROC AUC)
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0)
        roc_auc = roc_auc_score(
            true_labels, reconstruction_errors)  # Use errors for AUC

        cm = confusion_matrix(true_labels, predictions)

        print(f"Test Reconstruction MAE Threshold: {threshold:.6f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("Confusion Matrix:\n", cm)

        return {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm
        }


# if __name__ == '__main__':
#     # This is a basic example to show how the Trainer class might be used.
#     # In a real scenario, you would import models and loaders from other modules.
#     print("Running trainer example...")

#     # Mock components for demonstration
#     class MockModel(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.linear = nn.Linear(140, 140)  # Simple mock layer

#         def forward(self, x):
#             # Simulate reconstruction and latent space
#             # Ensure input is flattened if it has an extra channel dimension
#             if x.dim() == 3 and x.shape[2] == 1:
#                 x_flat = x.squeeze(2)
#             else:
#                 x_flat = x
#             reconstructed = self.linear(x_flat).unsqueeze(
#                 2)  # Add channel back if needed
#             latent = torch.randn(x.size(0), 8).to(x.device)  # Mock latent
#             return reconstructed, latent

#     # Mock DataLoaders
#     class MockDataLoader:
#         def __init__(self, data, labels=None, batch_size=32, shuffle=False):
#             self.data = data
#             self.labels = labels
#             self.batch_size = batch_size
#             self.shuffle = shuffle
#             self.dataset_size = len(data)
#             self.num_batches = (self.dataset_size +
#                                 batch_size - 1) // batch_size
#             self.current_batch_idx = 0

#         def __iter__(self):
#             self.current_batch_idx = 0
#             if self.shuffle:
#                 # Simple shuffle for mock data
#                 indices = np.arange(self.dataset_size)
#                 np.random.shuffle(indices)
#                 self.data = self.data[indices]
#                 if self.labels is not None:
#                     self.labels = self.labels[indices]
#             return self

#         def __next__(self):
#             if self.current_batch_idx >= self.num_batches:
#                 raise StopIteration

#             start = self.current_batch_idx * self.batch_size
#             end = min(start + self.batch_size, self.dataset_size)

#             batch_data = self.data[start:end]
#             if self.labels is not None:
#                 batch_labels = self.labels[start:end]
#                 batch = (batch_data, batch_labels)
#             else:
#                 batch = batch_data

#             self.current_batch_idx += 1
#             return batch

#         def __len__(self):
#             return self.num_batches

#     # Mock data
#     mock_X_train = torch.randn(500, 140, 1)
#     mock_X_val = torch.randn(200, 140, 1)
#     mock_X_test = torch.randn(300, 140, 1)
#     mock_y_test = np.random.randint(0, 2, 300)

#     mock_train_loader = MockDataLoader(
#         mock_X_train, batch_size=64, shuffle=True)
#     mock_val_loader = MockDataLoader(mock_X_val, batch_size=64, shuffle=False)
#     mock_test_loader = MockDataLoader(
#         mock_X_test, labels=mock_y_test, batch_size=64, shuffle=False)

#     # Mock threshold function (simple quantile for example)
#     def mock_threshold_fn(errors, labels):
#         # This should ideally be the actual thresholding logic from paper/eval script
#         # For this mock, we'll use a simple quantile
#         threshold = np.quantile(errors, 0.95)  # Example threshold
#         predictions = (errors > threshold).astype(int)
#         return threshold, predictions

#     # Initialize components
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = MockModel().to(device)
#     criterion = nn.MSELoss()  # Using MSE Loss as per paper's mention
#     optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

#     # Initialize Trainer
#     trainer = Trainer(
#         model=model,
#         device=device,
#         train_loader=mock_train_loader,
#         val_loader=mock_val_loader,
#         criterion=criterion,
#         optimizer=optimizer,
#         noise_stddev=0.05,  # Match paper's value
#         checkpoint_dir="checkpoints"  # Specify checkpoint directory
#     )

#     # Train the model (for a few epochs for demonstration)
#     best_loss, epochs_run = trainer.train(
#         num_epochs=5)  # Reduced epochs for example

#     # Evaluate the model
#     # Note: The evaluate method needs the actual thresholding function and test data
#     # For this mock, we'll just print a message.
#     print("\nMock evaluation step would happen here.")
#     # metrics = trainer.evaluate(mock_test_loader, mock_y_test, mock_threshold_fn, device)
#     # print(f"Mock evaluation metrics: {metrics}")
