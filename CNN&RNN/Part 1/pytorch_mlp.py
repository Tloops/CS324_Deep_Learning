import torch.nn as nn
import torch

class MLP(nn.Module):
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super().__init__()
        list = []
        last_channels = n_inputs
        for i in range(len(n_hidden)):
            list.append(nn.Linear(in_features=last_channels, out_features=n_hidden[i]))
            list.append(nn.ReLU())
            last_channels = n_hidden[i]
        list.append(nn.Linear(in_features=last_channels, out_features=n_classes))
        self.network = nn.Sequential(*list)

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        out = self.network(x)
        return out
