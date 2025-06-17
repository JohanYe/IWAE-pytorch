import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch

def filter_dataset(dataset, digits):
    """
    Filter the dataset to only include specified digits.
    Args:
        dataset: The dataset to filter (e.g., MNIST).
        digits: List of digits to keep (e.g., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).
    Returns:
        Filtered dataset.
    """
    targets = dataset.targets.numpy()
    data = dataset.data.numpy()

    mask = np.isin(targets, digits)
    dataset.targets = torch.tensor(targets[mask])
    dataset.data = torch.tensor(data[mask])
    return dataset


# Data loading with skewed distribution
def skew_dataset(dataset, skew_digits, remove_ratio):
    """
    Skew the dataset by removing a percentage of specified digits.
    Args:
        dataset: The dataset to skew (e.g., MNIST).
        skew_digits: List of digits to skew (e.g., [1, 2, 3, 4, 5]).
        remove_ratio: Fraction of samples to remove for each digit (e.g., 0.8 for 80%).
    Returns:
        Skewed dataset.
    """
    targets = dataset.targets.numpy()
    data = dataset.data.numpy()

    mask = np.ones(len(targets), dtype=bool)
    for digit in skew_digits:
        digit_indices = np.where(targets == digit)[0]
        remove_count = int(len(digit_indices) * remove_ratio)
        remove_indices = np.random.choice(digit_indices, remove_count, replace=False)
        mask[remove_indices] = False

    dataset.targets = torch.tensor(targets[mask])
    dataset.data = torch.tensor(data[mask])
    return dataset


def Plot_loss_curve(train_list, test_dict):
    x_tst = list(test_dict.keys())
    y_tst = list(test_dict.values())
    train_x_vals = np.arange(len(train_list))
    plt.figure(2)
    plt.xlabel("Num Steps")
    plt.ylabel("ELBO")
    plt.title("ELBO Loss Curve")
    plt.plot(train_x_vals, train_list, label="train")
    plt.plot(x_tst, y_tst, label="tst")
    plt.legend(loc="best")
    plt.locator_params(axis="x", nbins=10)

    plt.show()
    return


def create_canvas(x):
    rows = 10
    columns = 10

    plt.figure(1)
    canvas = np.zeros((28 * rows, columns * 28))
    for i in range(rows):
        for j in range(columns):
            idx = i % columns + rows * j
            canvas[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = x[idx].reshape(
                (28, 28)
            )
    return canvas
