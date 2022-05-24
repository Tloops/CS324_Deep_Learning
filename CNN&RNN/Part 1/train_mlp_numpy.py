from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset import generate_dataset
import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropy
import json

# Configurations
N_SAMPLE = 2000

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'  # '20'
LEARNING_RATE_DEFAULT = 1e-2  # 1e-2
MAX_EPOCHS_DEFAULT = 1500  # 1500
EVAL_FREQ_DEFAULT = 1  # 10
MODE_DEFAULT = "SGD"

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    n_sample = len(predictions)
    correct = 0
    for i in range(n_sample):
        pred, gt = predictions[i], targets[i]
        if (pred[0] > pred[1] and gt[0] > gt[1]) or (pred[0] < pred[1] and gt[0] < gt[1]):
            correct = correct + 1
    return correct / n_sample


def train(x_train, x_test, y_train, y_test, args, mode="BGD"):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    n_hidden = list(map(int, args.dnn_hidden_units.split(',')))
    eval_freq = args.eval_freq
    lr = args.learning_rate
    max_epoch = args.max_steps

    model = MLP(n_inputs=2, n_hidden=n_hidden, n_classes=2)
    loss_func = CrossEntropy()

    print("\033[34m ---------- " + mode + " train start ---------- \033[0m")
    save_content = {
        "mode": mode,
        "epoch": [],
        "acc_train": [],
        "acc_test": [],
    }

    for i in range(1, 1 + max_epoch):
        train_length = len(x_train)
        for j in range(train_length):
            x, y_gt = x_train[j], y_train[j]
            y_pred = model.forward(x)
            loss = loss_func.forward(y_pred, y_gt)

            dout = loss_func.backward(y_pred, y_gt)
            model.backward(dout)
            if mode == "SGD":
                model.step(lr, 1)
        if mode == "BGD":
            model.step(lr, train_length)

        if i % eval_freq == 0:
            save_content["epoch"].append(i)

            preds = np.zeros((len(x_train), 2))
            for j in range(len(x_train)):
                result = model.forward(x_train[j])
                preds[j][0], preds[j][1] = result[0], result[1]
            acc_train = accuracy(predictions=preds, targets=y_train)
            save_content["acc_train"].append(acc_train * 100)

            preds = np.zeros((len(x_test), 2))
            for j in range(len(x_test)):
                result = model.forward(x_test[j])
                preds[j][0], preds[j][1] = result[0], result[1]
            acc_test = accuracy(predictions=preds, targets=y_test)
            save_content["acc_test"].append(acc_test * 100)

            print('\033[31mepoch %4d/%4d, train acc %6.2f%%, test acc %6.2f%%\033[0m'
                  % (i, max_epoch, acc_train * 100, acc_test * 100))
            # if i == 100 and acc < 0.6:
            #     return train(x_train, x_test, y_train, y_test, args)

    json_file_path = 'numpy_accuracy.json'
    json_file = open(json_file_path, mode='w')
    json.dump(save_content, json_file, indent=4)


def main(args):
    """
    Main function
    """
    x_train, x_test, y_train, y_test = generate_dataset(N_SAMPLE)
    train(x_train, x_test, y_train, y_test, args, mode=args.mode)


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
    parser.add_argument('--mode', type=str, default=MODE_DEFAULT,
                        help='gradient descent method of the model')

    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    main(FLAGS)
