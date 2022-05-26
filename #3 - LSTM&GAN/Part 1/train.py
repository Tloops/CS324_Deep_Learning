from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from lstm import LSTM

torch.autograd.set_detect_anomaly(True)


def train(config):
    # Initialize the model that we are going to use
    model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size)

    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=8)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    
    print(next(model.parameters()).device)

    print("\033[34m ---------- train start ---------- \033[0m")
    save_content = {
        "seq_len": config.input_length,
        "step": [],
        "accuracy": [],
        "loss": [],
    }
    print(config.input_length)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)

        if torch.cuda.is_available():
            batch_inputs = batch_inputs.cuda()
            batch_targets = batch_targets.cuda()
        out = model(batch_inputs)
        loss = criterion(out, batch_targets)
        optimizer.zero_grad()
        if step == 0:
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        optimizer.step()

        if step % 25 == 0:
            save_content["step"].append(step)
            total_correct = (out.argmax(1) == batch_targets).sum()
            accuracy = total_correct / config.batch_size
            save_content["accuracy"].append(accuracy.item())
            save_content["loss"].append(loss.item())
            print("Step: %d, Loss: %.2f, accuracy: %.2f" % (step, loss, accuracy))

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')

    json_file_path = 'accuracy_' + str(config.input_length) + '.json'
    json_file = open(json_file_path, mode='w')
    json.dump(save_content, json_file, indent=4)


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=1500, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)

    config = parser.parse_args()
    # Train the model
    train(config)
