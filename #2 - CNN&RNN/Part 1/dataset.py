import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch import nn


def generate_dataset(n_samples):
    inputs, labels = make_moons(n_samples=n_samples)

    # visualization
    # visualize(inputs, labels)

    labels = OneHotEncoder(sparse=False).fit_transform(labels.reshape(len(labels), 1))
    return train_test_split(inputs, labels, test_size=0.3, random_state=42)


def visualize(inputs, labels):
    x1 = inputs[np.where(labels == 0)]
    x2 = inputs[np.where(labels == 1)]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x1[:, 0], x1[:, 1], "o", markerfacecolor="red", markeredgecolor="red")
    ax.plot(x2[:, 0], x2[:, 1], "o", markerfacecolor="blue", markeredgecolor="blue")
    plt.show()


if __name__ == "__main__":
    generate_dataset(100)
