from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
from cnn_model import CNN
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
import torch

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 40000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = "./data/"

FLAGS = None


def train():
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    data_dir = FLAGS.data_dir
    eval_freq = FLAGS.eval_freq
    lr = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    max_epoch = FLAGS.max_steps

    train_data = torchvision.datasets.CIFAR10(root=data_dir,
                                              train=True,
                                              transform=torchvision.transforms.ToTensor(),
                                              download=True)
    test_data = torchvision.datasets.CIFAR10(root=data_dir,
                                             train=False,
                                             transform=torchvision.transforms.ToTensor(),
                                             download=True)
    train_data_size = len(train_data)
    test_data_size = len(test_data)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8)

    model = CNN(n_channels=3, n_classes=10)
    if torch.cuda.is_available():
        model = model.cuda()

    loss_func = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_func = loss_func.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\033[34m ---------- train start ---------- \033[0m")
    save_content = {
        "step": [],
        "acc_train": [],
        "acc_test": [],
    }

    step = 0
    while step < max_epoch:
        model.train()
        for batch in train_dataloader:
            x, label = batch
            if torch.cuda.is_available():
                x, label = x.cuda(), label.cuda()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_func(y_pred, label)
            loss.backward()
            optimizer.step()
            step += 1

            if step % eval_freq == 0:
                model.eval()
                save_content["step"].append(step)

                total_train_accuracy = 0
                for batch in train_dataloader:
                    x, label = batch
                    if torch.cuda.is_available():
                        x, label = x.cuda(), label.cuda()
                    outputs = model(x)
                    total_train_accuracy += (outputs.argmax(1) == label).sum()
                acc_train = total_train_accuracy / train_data_size * 100
                save_content["acc_train"].append(acc_train.item())

                total_test_accuracy = 0
                for batch in test_dataloader:
                    x, label = batch
                    if torch.cuda.is_available():
                        x, label = x.cuda(), label.cuda()
                    outputs = model(x)
                    total_test_accuracy += (outputs.argmax(1) == label).sum()
                acc_test = total_test_accuracy / test_data_size * 100
                save_content["acc_test"].append(acc_test.item())

                print('\033[31mstep %4d/%4d, train acc %.2f%%, test acc %.2f%%\033[0m'
                      % (step, max_epoch, acc_train, acc_test))

            if step >= max_epoch:
                break

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
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
