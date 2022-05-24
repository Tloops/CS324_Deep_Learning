from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch import nn


class CNN(nn.Module):

    def __init__(self, n_channels, n_classes):
        """
        Initializes CNN object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
        """
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            ConvAll(in_channels=n_channels, out_channels=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvAll(in_channels=64, out_channels=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvAll(in_channels=128, out_channels=256),
            ConvAll(in_channels=256, out_channels=256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvAll(in_channels=256, out_channels=512),
            ConvAll(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvAll(in_channels=512, out_channels=512),
            ConvAll(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=n_classes)
        )

    def forward(self, x):
        """
        Performs forward pass of the input.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        return self.model(x)


class ConvAll(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvAll, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)
