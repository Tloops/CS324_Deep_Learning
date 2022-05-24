from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        self.Linear = []
        self.Relu = []
        prev_n = n_inputs
        for n_hid in n_hidden:
            self.Linear.append(Linear(prev_n, n_hid))
            prev_n = n_hid
            self.Relu.append(ReLU())
        self.Linear_out = Linear(prev_n, n_classes)
        self.SoftMax = SoftMax()

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        prev_val = np.array(x)
        for i in range(len(self.Linear)):
            linear, relu = self.Linear[i], self.Relu[i]
            new_val = linear.forward(prev_val)
            # print(new_val)
            new_val = relu.forward(new_val)
            # print(new_val)
            prev_val = np.array(new_val)
        x2 = self.Linear_out.forward(prev_val)
        # print(x2)
        out = self.SoftMax.forward(x2)
        # print(out)
        return out

    def backward(self, dout):
        """
        Performs backward propagation pass given the loss gradients. 
        Args:
            dout: gradients of the loss
        """
        dz = self.SoftMax.backward(dout)
        dx = self.Linear_out.backward(dz)

        prev_grad = np.array(dx)
        for i in reversed(range(len(self.Linear))):
            linear, relu = self.Linear[i], self.Relu[i]
            new_grad = linear.backward(relu.backward(prev_grad))
            prev_grad = np.array(new_grad)

    def step(self, lr, size):
        for linear in self.Linear:
            linear.step(lr, size)
        self.Linear_out.step(lr, size)
