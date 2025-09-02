# Project Progress: ECG5000 Unsupervised Anomaly Detection

This document tracks the progress made in replicating and extending the methodology from "Russo et al. (2025) — Unsupervised Anomaly Detection in ECG Signals Using Denoising Autoencoders: A Comparative Study".

## Phase 1: Denoising Linear Autoencoder Implementation (Completed)

The core pipeline for the Denoising Linear Autoencoder has been successfully implemented and refactored for modularity. This phase focused on building a robust and reproducible system for data handling, model definition, training, and evaluation on a local machine.

### Completed Components:

*   **Data Loading and Preprocessing (`src/utils/data_loaders.py`):**
    *   Functions to load the ECG5000 dataset.
    *   Implementation of the precise 60% train, 20% validation, 20% test split for normal samples.
    *   Correct normalization (zero mean, unit variance) fitted only on training data.
    *   Data reshaping for PyTorch compatibility.
    *   Creation of PyTorch `DataLoader` objects.

*   **Denoising Linear Autoencoder Model (`src/models/ae_linear.py`):**
    *   Implementation of the linear autoencoder architecture: 140 -> 32 -> 8 (latent) -> 32 -> 140.
    *   Uses ReLU activation for hidden layers, with a linear latent space.

*   **Noise Injection (in `src/train/tra.py`):**
    *   
*   **Training Loop and Early Stopping (`src/train/trainer.py`):**
    *   A `Trainer` class that orchestrates the training process.
    *   Handles epoch loops, validation, and applies the denoising strategy.
    *   Implements early stopping based on validation loss improvement and model checkpointing.
    *   Contains a utility function (`add_gaussian_noise`) to apply Gaussian noise to input data during training, as required for denoising autoencoders.

*   **Evaluation Metrics (`src/eval/metrics.py`):**
    *   An `evaluate_model` function to run the trained model on the test set and return raw reconstruction errors and true labels.
    *   A `calculate_metrics` function to compute various performance metrics (accuracy, precision, recall, F1-score, ROC AUC, PR AUC, confusion matrix).

*   **Thresholding Logic (`src/eval/thresholding.py`):**
    *   `calculate_threshold`: Implements the paper's specific method for determining the anomaly detection threshold.
    *   `predict_with_threshold`: Uses the calculated threshold to classify samples as normal or anomalous.

*   **Main Execution Script (`run_linear_dae.py`):**
    *   Integrates all the above components.
    *   Loads data, initializes the model, trainer, and then trains the model.
    *   Orchestrates the evaluation process by calling `evaluate_model`, `calculate_threshold`, `predict_with_threshold`, and `calculate_metrics` to produce the final performance report.

## Next Steps:

*   Compare the results obtained from running `run_linear_dae.py` with the paper's reported accuracy (97.73%).
*   Implement the Convolutional and LSTM Autoencoders.
*   Implement Contractive Loss variants.
*   Prepare for running advanced models on Colab/cluster due to computational constraints.
