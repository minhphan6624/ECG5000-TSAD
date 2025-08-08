import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_ecg(path_to_data: str, normalize=True):
    train_path = f"{path_to_data}/ECG5000/ECG5000_TRAIN.txt"
    test_path = f"{path_to_data}/ECG5000/ECG5000_TEST.txt"

    # Load the datasets
    train_data = np.loadtxt(train_path)
    test_data = np.loadtxt(test_path)

    # Concatnate data
    data = np.concatenate(train_data, test_data)

    # Separate labels and features
    labels = data[:, 0]
    sequeneces = data[:, 1:]

    if normalize:
        scaler = StandardScaler
        sequences = scaler.fit_transform(sequences)

    return sequences, labels
