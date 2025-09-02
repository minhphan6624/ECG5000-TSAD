import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Constants
CLASS_NORMAL = 1
RANDOM_STATE = 42  # For reproducibility


def load_dataset(data_dir="data"):
    """
    Loads, preprocesses, and splits the ECG5000 dataset.

    Args:
        data_dir (str): Directory where the dataset files are located.
        test_size_val (float): Proportion of normal samples to use for validation.
        test_size_test_normals (float): Proportion of the remaining normal samples to use for the test set.

    Returns:
        tuple: A tuple containing:
            - X_train (np.ndarray): Training features.
            - X_val (np.ndarray): Validation features.
            - X_test (np.ndarray): Test features.
            - y_test (np.ndarray): Test labels (0 for normal, 1 for anomaly).
    """

    # Load the datasets
    train_path = f"{data_dir}/ECG5000_TRAIN.txt"
    test_path = f"{data_dir}/ECG5000_TEST.txt"

    try:
        train_data = np.loadtxt(train_path)
        test_data = np.loadtxt(test_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset files not found in {data_dir}. Please ensure ECG5000_TRAIN.txt and ECG5000_TEST.txt are present.")

    # Stack the two datasets vertically
    all_data = np.vstack([train_data, test_data])

    # Split into features (X) and labels (y)
    # The first column is the label, the rest are features
    X_all, y_all = all_data[:, 1:], all_data[:, 0].astype(int)

    # Separate normal vs anomalous sequences
    X_normal = X_all[y_all == CLASS_NORMAL]
    X_anomaly = X_all[y_all != CLASS_NORMAL]

    # Split normal data into train, validation, and test sets
    # Paper uses 60% train, 20% val, 20% test for normals
    # train_test_split(X_normal, test_size=0.4) -> 60% train, 40% for val/test
    X_train, X_val_and_test_normals = train_test_split(
        X_normal,
        test_size=0.4,  # 40% for both val and test
        random_state=RANDOM_STATE
    )

    # Split the remaining 40% into validation (20%) and test_normals (20%)
    X_val, X_test_normals = train_test_split(
        X_val_and_test_normals,
        test_size=0.5,  # 0.5 of the 40% means 20% of the total
        random_state=RANDOM_STATE
    )

    # Construct the final test set: all anomalies + test normals
    X_test = np.vstack([X_test_normals, X_anomaly])
    y_test = np.hstack([                # Create corresponding labels: 0 for normal, 1 for anomaly
        np.zeros(len(X_test_normals), dtype=int),
        np.ones(len(X_anomaly), dtype=int)
    ])

    # Normalize (training data only)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Reshape data for PyTorch models (add channel dimension)
    # Expected shape: (N, n_len, features_num) -> (N, 140, 1)
    X_train = X_train[..., None]
    X_val = X_val[..., None]
    X_test = X_test[..., None]

    return X_train, X_val, X_test, y_test


def get_data_loaders(X_train, X_val, X_test, y_test, batch_size=32, shuffle_train=True):
    """
    Wraps the datasets into PyTorch DataLoaders.

    Args:
        X_train (np.ndarray): Training features.
        X_val (np.ndarray): Validation features.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        batch_size (int): Batch size for the DataLoaders.
        shuffle_train (bool): Whether to shuffle the training data.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - test_loader (DataLoader): DataLoader for the test set.
    """
    # Convert numpy arrays to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Create TensorDatasets
    # No labels needed for training input
    train_dataset = TensorDataset(X_train_t)
    # No labels needed for validation input
    val_dataset = TensorDataset(X_val_t)
    # Labels needed for testing
    test_dataset = TensorDataset(X_test_t, y_test_t)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
