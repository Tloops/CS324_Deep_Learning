import numpy as np

class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=1e2, learning_rate=1e-2):
        """
        Initializes perceptron object.
        Args:
            n_inputs: number of inputs.
            max_epochs: maximum number of training cycles.
            learning_rate: magnitude of weight changes at each training cycle
        """
        self.n_inputs = n_inputs
        self.max_epochs = max_epochs
        self.lr = learning_rate
        self.weight = np.zeros(n_inputs)
        
    def forward(self, input):
        """
        Predict label from input 
        Args:
            input: array of dimension equal to n_inputs.
        """
        out = self.weight.transpose().dot(input)
        label = 1 if out > 0 else -1
        return label
        
    def train(self, training_inputs, labels):
        """
        Train the perceptron
        Args:
            training_inputs: list of numpy arrays of training points.
            labels: arrays of expected output value for the corresponding point in training_inputs.
        """
        for j in range(1, 1+int(self.max_epochs)):
            permutation = np.random.permutation(training_inputs.shape[0])
            x_train = training_inputs[permutation, :]
            y_train = labels[permutation]
            for i in range(len(x_train)):
                input, label = x_train[i], y_train[i]
                pred_label = self.forward(input)
                if pred_label != label:
                    self.weight += self.lr * (label * input)
            train_acc = self.test(x_train, y_train)
            if(j % 10 == 0):
                print('epoch %03d: acc -> %.1f' % (j, train_acc))
        print('Train finished!')
    
    def test(self, test_inputs, labels):
        length = len(test_inputs)
        cnt_correct = 0
        for i in range(length):
            input, label = test_inputs[i], labels[i]
            pred_label = self.forward(input)
            if pred_label == label:
                cnt_correct = cnt_correct + 1
        return cnt_correct / length * 100
