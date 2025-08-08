import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_ecg(path_to_data: str, normalize=True):
    train_path = f"{path_to_data}/raw/ECG5000/ECG5000_TRAIN.txt"
    test_path = f"{path_to_data}/raw/ECG5000/ECG5000_TEST.txt"

    # Load the datasets
    train_data = np.loadtxt(train_path)
    test_data = np.loadtxt(test_path)

    # Concatnate data
    data = np.concatenate([train_data, test_data], axis=0)

    # Separate labels and features
    labels = data[:, 0]
    sequences = data[:, 1:]

    if normalize:
        scaler = StandardScaler()
        sequences = scaler.fit_transform(sequences)

    return sequences, labels


def split_one_class(sequences, labels, normal_class=1):
    # Train dataset
    X_train = sequences[labels == normal_class]
    y_train = np.zeros(len(X_train))

    # Test dataset
    X_test = sequences[labels != normal_class]
    y_test = np.ones(len(X_test))

    return X_train, y_train, X_test, y_test
