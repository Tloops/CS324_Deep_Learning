import numpy as np
from math import log


class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Module initialisation.
        Args:
            in_features: input dimension
            out_features: output dimension
        TODO:
        1) Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
        std = 0.0001.
        2) Initialize biases self.params['bias'] with 0. 
        3) Initialize gradients with zeros.
        """
        self.input = np.zeros(in_features)
        self.params = {
            'weight': np.random.normal(loc=0, scale=0.7, size=(out_features, in_features)),
            'bias': np.zeros(out_features)
        }
        self.grads = {
            'weight': np.zeros((out_features, in_features)),
            'bias': np.zeros(out_features)
        }

    def forward(self, x):
        """
        Forward pass (i.e., compute output from input).
        Args:
            x: input to the module
        Returns:
            out: output of the module
        Hint: Similarly to pytorch, you can store the computed values inside the object
        and use them in the backward pass computation. This is true for *all* forward methods of *all* modules in this class
        """
        self.input = np.array(x)
        out = self.params['weight'].dot(x) + self.params['bias']
        return out

    def backward(self, dout):
        """
        Backward pass (i.e., compute gradient).
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to 
        layer parameters in self.grads['weight'] and self.grads['bias']. 
        """
        self.grads['weight'] += dout.reshape((-1, 1)).dot(self.input.reshape((-1, 1)).transpose())
        self.grads['bias'] += np.array(dout)
        dx = self.params['weight'].transpose().dot(dout)
        return dx

    def step(self, lr, size):
        self.params['weight'] -= lr * self.grads['weight'] / size
        self.params['bias'] -= lr * self.grads['bias'] / size
        self.grads['weight'].fill(0)
        self.grads['bias'].fill(0)


class ReLU(object):
    x = None

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
        """
        self.x = x
        out = np.array(x)
        out[out < 0] = 0
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        dx = np.array(dout)
        dx[self.x < 0] = 0
        return dx


class SoftMax(object):
    out = None

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
    
        TODO:
        Implement forward pass of the module. 
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        
        """
        b = x.max()
        y = np.exp(x - b)
        out = y / y.sum()
        self.out = np.array(out)
        return out

    def backward(self, dout):
        """
        Backward pass. 
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        n = len(dout)
        dx = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    dx[i][j] = self.out[i] * (1 - self.out[i])
                else:
                    dx[i][j] = - self.out[i] * self.out[j]
        dx = dx.dot(dout)
        return dx


class CrossEntropy(object):
    def forward(self, x, y):
        """
        Forward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            out: cross entropy loss
        """
        out = -(y * np.log(x+1e-5)).sum()
        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            dx: gradient of the loss with respect to the input x.
        """
        dx = - y / (x+1e-5)
        return dx
