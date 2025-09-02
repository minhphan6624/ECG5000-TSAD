import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score
)


def evaluate_model(model, test_loader, device):
    """
    Evaluates the trained model on the test set and returns reconstruction errors and true labels.
    The threshold calculation and final metric computation will be handled by the caller.

    Args:
        model (nn.Module): The trained PyTorch model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): The device to run evaluation on ('cuda' or 'cpu').

    Returns:
        tuple: A tuple containing:
            - reconstruction_errors (np.ndarray): Reconstruction errors for each sample.
            - true_labels (np.ndarray): True labels for each sample.
    """
    print("\nEvaluating model on test set...")
    model.eval()
    reconstruction_errors = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = inputs.to(device)

            # For evaluation, we don't add noise to the input
            reconstructed, _ = model(inputs)

            # Calculate reconstruction error (MAE per sample)
            errors = torch.abs(
                reconstructed - inputs).mean(dim=(1, 2)).cpu().numpy()

            reconstruction_errors.extend(errors)
            true_labels.extend(labels.cpu().numpy())

    return np.array(reconstruction_errors), np.array(true_labels)


def calculate_metrics(y_true, y_pred, errors, threshold):
    """
    Calculates various performance metrics for anomaly detection and stores them in a dictionary.

    Args:
        y_true (np.ndarray): Ground truth labels (0 for normal, 1 for anomaly).
        y_pred (np.ndarray): Predicted labels (0 for normal, 1 for anomaly).
        errors (np.ndarray): Reconstruction errors for each sample.
        threshold (float): The threshold used for prediction.

    Returns:
        dict: A dictionary containing calculated metrics.
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray) or not isinstance(errors, np.ndarray):
        raise TypeError("y_true, y_pred, and errors must be numpy arrays.")
    if len(y_true) != len(y_pred) or len(y_true) != len(errors):
        raise ValueError("Input arrays must have the same length.")

    metrics = {}

    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # ROC AUC and PR AUC (using errors as scores)
    # Ensure there are both positive and negative classes for AUC calculation
    if len(np.unique(y_true)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_true, errors)
        metrics['pr_auc'] = average_precision_score(y_true, errors)
    else:
        # Cannot calculate if only one class is present
        metrics['roc_auc'] = np.nan
        metrics['pr_auc'] = np.nan

    # Confusion Matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    return metrics
