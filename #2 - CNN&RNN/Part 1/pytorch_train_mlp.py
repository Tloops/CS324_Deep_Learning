from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np
import os

import torch

from pytorch_mlp import MLP
from pytorch_dataset import PointDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import json

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 20
EVAL_FREQ_DEFAULT = 1

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    predictions = torch.argmax(predictions, dim=1)
    targets = torch.argmax(targets, dim=1)
    return (predictions == targets).sum() / len(predictions)


def train():
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    data = PointDataset(n_samples=2000)
    train_size = int(len(data) * 0.7)
    test_size = len(data) - train_size
    train_dataset, test_dataset = random_split(data, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=test_size, num_workers=2)
    eval_train_dataloader = DataLoader(train_dataset, batch_size=train_size, num_workers=2)

    n_hidden = list(map(int, FLAGS.dnn_hidden_units.split(',')))
    eval_freq = FLAGS.eval_freq
    lr = FLAGS.learning_rate
    max_epoch = FLAGS.max_steps
    model = MLP(n_inputs=2, n_hidden=n_hidden, n_classes=2)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    print("\033[34m ---------- train start ---------- \033[0m")
    save_content = {
        "epoch": [],
        "acc_train": [],
        "acc_test": [],
    }

    for i in range(1, 1+max_epoch):
        model.train()
        for batch in train_dataloader:
            x, label = batch
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_func(y_pred, label)
            loss.backward()
            optimizer.step()

        if i % eval_freq == 0:
            model.eval()
            save_content["epoch"].append(i)

            for batch in eval_train_dataloader:
                x, label = batch
                y_pred = model(x)
            acc_train = accuracy(predictions=y_pred, targets=label)
            save_content["acc_train"].append(acc_train.item() * 100)

            for batch in test_dataloader:
                x, label = batch
                y_pred = model(x)
            acc_test = accuracy(predictions=y_pred, targets=label)
            save_content["acc_test"].append(acc_test.item() * 100)

            print('\033[31mepoch %4d/%4d, train acc %6.2f%%, test acc %6.2f%%\033[0m'
                  % (i, max_epoch, acc_train * 100, acc_test * 100))

    json_file_path = 'torch_accuracy.json'
    json_file = open(json_file_path, mode='w')
    json.dump(save_content, json_file, indent=4)


def main():
    """
    Main function
    """
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    main()
