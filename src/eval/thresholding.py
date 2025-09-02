import numpy as np


def calculate_threshold(normal_errors, anomaly_errors):
    """
    Calculates the anomaly detection threshold based on the method described in the paper.
    Threshold = midpoint between (mean_normal + std_normal) and (mean_anomaly - std_anomaly).

    Args:
        normal_errors (np.ndarray): Reconstruction errors for normal samples.
        anomaly_errors (np.ndarray): Reconstruction errors for anomalous samples.
    """

    mean_normal = np.mean(normal_errors)
    std_normal = np.std(normal_errors)
    mean_anomaly = np.mean(anomaly_errors)
    std_anomaly = np.std(anomaly_errors)

    # Calculate the two points for the midpoint calculation
    point1 = mean_normal + std_normal
    point2 = mean_anomaly - std_anomaly

    # Calculate the threshold as the midpoint between point1 and point2
    threshold = (point1 + point2) / 2.0

    return threshold


def predict_with_threshold(errors, threshold):
    """
    Predicts labels based on reconstruction errors (np.ndarray) and a given threshold.
    0 for normal samples and 1 for anomalous samples.
    """

    predictions = (errors > threshold).astype(int)
    return predictions
