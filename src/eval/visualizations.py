import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def plot_error_distribution(errors_normal, errors_anomaly, save_path="results/error_distribution.png"):
    """
    Plots histograms of reconstruction errors for normal and anomalous data.

    Args:
        errors_normal (np.ndarray): Reconstruction errors for normal samples.
        errors_anomaly (np.ndarray): Reconstruction errors for anomaly samples.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(errors_normal, bins=50, alpha=0.7, label='Normal Data Errors', color='blue')
    plt.hist(errors_anomaly, bins=50, alpha=0.7, label='Anomaly Data Errors', color='red')
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error (MAE)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Error distribution plot saved to {save_path}")

def plot_roc_pr_curves(y_true, errors, save_path_roc="results/roc_curve.png", save_path_pr="results/pr_curve.png"):
    """
    Plots ROC and Precision-Recall curves.

    Args:
        y_true (np.ndarray): True labels (0 for normal, 1 for anomaly).
        errors (np.ndarray): Reconstruction errors (used as anomaly scores).
        save_path_roc (str): Path to save the ROC curve plot.
        save_path_pr (str): Path to save the PR curve plot.
    """
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, errors)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path_roc)
    plt.close()
    print(f"ROC curve plot saved to {save_path_roc}")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, errors)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path_pr)
    plt.close()
    print(f"Precision-Recall curve plot saved to {save_path_pr}")

def plot_reconstructions(original_samples, reconstructed_samples, labels, num_samples=5, save_path_prefix="results/reconstruction_"):
    """
    Plots original vs. reconstructed signals for a few normal and anomaly samples.

    Args:
        original_samples (np.ndarray): Original input samples.
        reconstructed_samples (np.ndarray): Reconstructed output samples.
        labels (np.ndarray): True labels for the samples.
        num_samples (int): Number of normal and anomaly samples to plot.
        save_path_prefix (str): Prefix for saving the reconstruction plots.
    """
    normal_indices = np.where(labels == 0)[0]
    anomaly_indices = np.where(labels == 1)[0]

    # Plot normal reconstructions
    for i in range(min(num_samples, len(normal_indices))):
        idx = normal_indices[i]
        plt.figure(figsize=(12, 4))
        plt.plot(original_samples[idx], label='Original')
        plt.plot(reconstructed_samples[idx], label='Reconstructed')
        plt.title(f'Normal Sample Reconstruction (Sample {idx})')
        plt.xlabel('Time Steps')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_path_prefix}normal_{idx}.png")
        plt.close()
        print(f"Normal reconstruction plot saved to {save_path_prefix}normal_{idx}.png")

    # Plot anomaly reconstructions
    for i in range(min(num_samples, len(anomaly_indices))):
        idx = anomaly_indices[i]
        plt.figure(figsize=(12, 4))
        plt.plot(original_samples[idx], label='Original')
        plt.plot(reconstructed_samples[idx], label='Reconstructed')
        plt.title(f'Anomaly Sample Reconstruction (Sample {idx})')
        plt.xlabel('Time Steps')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_path_prefix}anomaly_{idx}.png")
        plt.close()
        print(f"Anomaly reconstruction plot saved to {save_path_prefix}anomaly_{idx}.png")
