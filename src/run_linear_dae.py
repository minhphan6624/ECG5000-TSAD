import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# Import custom modules
from utils.data_loader import load_dataset, get_data_loaders
from models.ae_linear import LinearAutoencoder
from train.trainer import Trainer
from eval.thresholding import calculate_threshold, predict_with_threshold
from eval.evaluate_model import calculate_metrics, evaluate_model
from eval.visualizations import plot_error_distribution, plot_roc_pr_curves, plot_reconstructions

# --- Configuration ---
# Data parameters
DATA_DIR = "data/ECG5000/"

# Model parameters
INPUT_DIM = 140
HIDDEN_DIM1 = 32
LATENT_DIM = 8

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 200
NOISE_STDDEV = 0.05
PATIENCE = 3
CHECKPOINT_DIR = "checkpoints"


def main():
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and prepare datasets
    try:
        X_train, X_val, X_test, y_test = load_dataset(data_dir=DATA_DIR)
        print("Datasets prepared successfully.")
        print(
            f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        print(f"Number of anomalies in test set: {np.sum(y_test)}")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure the 'data' directory exists and contains ECG5000_TRAIN.txt and ECG5000_TEST.txt.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return

    # Create DataLoaders
    train_loader, val_loader, test_loader = get_data_loaders(
        X_train, X_val, X_test, y_test, batch_size=BATCH_SIZE, shuffle_train=True
    )
    print("DataLoaders created.")

    # 2. Initialize Model, Loss, Optimizer
    model = LinearAutoencoder(
        input_dim=INPUT_DIM,
        hidden_dim1=HIDDEN_DIM1,
        latent_dim=LATENT_DIM
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    print("Model, loss, and optimizer initialized.")
    print(f"Model architecture:\n{model}")

    # 3. Initialize Trainer
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        noise_stddev=NOISE_STDDEV,
        checkpoint_dir=CHECKPOINT_DIR,
        patience=PATIENCE
    )
    print("Trainer initialized.")

    # 4. Train the model
    print("\n--- Starting Training ---")
    best_val_loss, epochs_run = trainer.train(num_epochs=NUM_EPOCHS)
    print("--- Training Finished ---")

    # 5. Evaluate the model
    print("\n--- Starting Evaluation ---")

    # Get reconstruction errors and true labels from the model
    reconstruction_errors_test, true_labels_test = evaluate_model(
        model, test_loader, device)

    # Separate errors for threshold calculation
    normal_errors_test = reconstruction_errors_test[true_labels_test == 0]
    anomaly_errors_test = reconstruction_errors_test[true_labels_test == 1]

    if len(normal_errors_test) == 0 or len(anomaly_errors_test) == 0:
        print("Error: Not enough normal or anomaly samples in the test set for threshold calculation.")
        return

    # Calculate threshold using the paper's method
    threshold = calculate_threshold(normal_errors_test, anomaly_errors_test)
    print(f"Calculated threshold using paper's method: {threshold:.6f}")

    # Make predictions using the calculated threshold
    predictions_test = predict_with_threshold(
        reconstruction_errors_test, threshold)

    # Calculate and print final metrics
    final_metrics = calculate_metrics(
        true_labels_test, predictions_test, reconstruction_errors_test)

    print("\n--- Performance Metrics ---")
    for key, value in final_metrics.items():
        if key == 'confusion_matrix':
            print(f"{key}:\n{value}")
        else:
            print(f"{key}: {value:.4f}")
    print("-------------------------\n")

    # 6. Generate Visualizations
    print("\n--- Generating Visualizations ---")

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Get original inputs from the test loader for reconstruction plots
    all_original_inputs = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, _ = batch
            all_original_inputs.extend(inputs.cpu().numpy())
    all_original_inputs = np.array(all_original_inputs)

    # Get reconstructed outputs for reconstruction plots
    model.eval()
    all_reconstructed_outputs = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, _ = batch
            inputs = inputs.to(device)
            reconstructed, _ = model(inputs)
            all_reconstructed_outputs.extend(reconstructed.cpu().numpy())
    all_reconstructed_outputs = np.array(all_reconstructed_outputs)

    # Plot Error Distribution
    plot_error_distribution(normal_errors_test, anomaly_errors_test, save_path="results/error_distribution_linear_dae.png")

    # Plot ROC and PR Curves
    plot_roc_pr_curves(true_labels_test, reconstruction_errors_test,
                       save_path_roc="results/roc_curve_linear_dae.png",
                       save_path_pr="results/pr_curve_linear_dae.png")

    # Plot Reconstructions
    plot_reconstructions(all_original_inputs, all_reconstructed_outputs, true_labels_test,
                         num_samples=5, save_path_prefix="results/reconstruction_linear_dae_")

    print("--- Visualizations Generated ---")
    print("--- Evaluation Finished ---")


if __name__ == "__main__":
    main()
