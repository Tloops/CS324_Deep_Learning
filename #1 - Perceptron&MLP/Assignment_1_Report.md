# Assignment 1 Report

## Part I: the perceptron

> This part aims to implement and test a simple artificial neuron.

For convenience to generate data with different distributions, train the perceptron and test it, I use the `jupyter notebook`. The file is at `./Part 1/train.ipynb`. Also, there is also a python file `./Part 1/perceptron` describing the perceptron object.

### Task 1

To generate two Gaussian distributions and sample 200 points, the python library `numpy` provides a really useful function: `numpy.random.normal()`. With the parameter `size` of this function set to `(200, 2)`, the task is easily finished. By changing the parameter `loc` and `scale`, we can change the distribution's **mean** and **variance**, which will be modified in *Task 4*.

So I successfully generated the input, split them to train and test datasets, label them with 1 and -1, and finally combine them together with function `numpy.concatenate()`.

### Task 2

I implemented all the functions in `perceptron.py` and I also added a function called `test()` to the class. It can initialize the model, forward, train and test:

- **initialize**: 

  The perceptron object has four fields to be initialized

  - `n_inputs`: the dimension of the input vectors
  - `max_epochs`: the max number of training iterations, default `1e2`
  - `lr`: the learning rate, default `1e-2`
  - `weight`: the parameter of the perceptron units, has the same size as `n_inputs`

- **forward**:

  Predict the label according to the formula below:
  $$
  y' = sgn(w_t^T x_i)
  $$
  Here, $y'$ stands for the predicted label, $w_t$ stands for the weight of the perceptron at the t-th iteration and $x_i$ stands for the i-th input.

- **train**:

  Training procedure pseudocode:

  ```pseudocode
  for i from 1 to max_epochs do
  	shuffle training_input and labels by the same way
  	for input, y_gt in training_input, labels do
          y_pred := forward(input)
          if y_pred != y_gt then
              w := w + lr * y_gt * input
          end if
      end for
  end for
  ```

- **test**:

  Use the test data to go through the perceptron and return the accuracy.

### Task 3

For training, the settings of the hyperparameters are as follow:

1. n_input: `2`
2. learning rate: `1e-2`
3. max_epoch: `1e2`

The core code are in the file `train.ipynb` , part 2.

### Task 4

In this part, we try to train the perceptron with three kinds of data distributions:

- **normal**

  - loc1, loc2: 5, -5
  - sigma1, sigma2: 2, 2

- **2 Gaussians are too close (mean is close)**

  - loc1, loc2: 1, -1
  - sigma1, sigma2: 2, 2

- **variance is too high**
- loc1, loc2: 5, -5
  
- sigma1, sigma2: 10, 10

Also, the 3 graphs below is provided to show the distribution of the corresponding distribution.

<img src="./Part 1/img/normal.png" alt="normal" style="zoom: 67%;" /><img src="./Part 1/img/mean.png" alt="mean" style="zoom:67%;" /><img src="./Part 1/img/variance.png" alt="variance" style="zoom:67%;" />

#### Test accuracy: 

- **normal**: 100.0%
- **2 Gaussians are too close (mean is close)**: 76.2%
- **variance is too high**: 78.8%

From the three graphs above, it is obvious why the result will be like this. For normal case, the points are divided into two parts clearly, so by intuition, it is easy to draw a line to separate them into two parts. For other 2 cases, the two types of points are mixed to each other, and it is hard to classify them.

In conclusion, this perceptron is efficient with some particular input. However, there are still some cases that it doesn't perform so well.



## Part II: MLP progress

> This part aims to implement an MLP and its training procedure from scratch.

Except the three template code files `mlp_numpy.py`, `module.py` and `train_mlp_numpy.py` provided by the teacher, I add a file named `dataset.py` to generate the dataset and visualize the points in it. 

`dataset.py` uses the method `make_moons` to produce data points, the module `OneHotEncoder` to transform the dataset labels to one-hot encoding and the method `train_test_split` to split the dataset to train set and test set by 7:3.

### Task 1

`modules.py`: implements the basic components of the MLP, including **Linear**, **ReLU**, **SoftMax** and **CrossEntropy**. All of the four modules have 3 methods: `__init__`, `forward` and `backward`. `__init__` is used to declear and initialize the variables, `forward` is used to compute the outputs, `backward` is used to compute the gradients.

- **Linear**: a simple linear layer. The weights are initialized by normal distribution with mean = 0 and std = 0.0001. The bias are initialized with 0.
  $$
  out = Wx\\
  grad_{weight} = Dx\\
  grad_{bias} = D\\
  grad_{x} = W^TD
  $$
  where $W$ is the weights, $x$ is the input and $D$ is the gradients of the output.

- **ReLU**: Linear rectification function layer, a kind of activation function which is frequently used in deep learning area. 
  $$
  out =
  \begin{cases}
  x,\ x \geq 0\\
  0,\ x < 0
  \end{cases}\\
  grad_x =
  \begin{cases}
  D,\ x \geq 0\\
  0,\ x < 0
  \end{cases}
  $$
  where $x$ is the input and $D$ is the gradients of the output.

- **SoftMax**: make the output values between the interval 0 and 1.
  $$
  out = \frac {e^x} {\sum_{i=1}^{d_N} e^{x_i}}\\
  g_{x_{ij}} = 
  \begin{cases}
  out_i\ \cdot (1-out_i),\ i = j\\
  -out_i \cdot out_j,\ i \neq j
  \end{cases}\\
  grad_x = g\ \cdot\ D
  $$
  where $x$ is the input and $D$ is the gradients of the output.

- **CrossEntropy**: loss function.
  $$
  out = -\sum y_{gt}log(y_{pred})\\
  grad_{y_{pred}} = - y_{gt} / y_{pred}
  $$
  where $y_{gt}$ is the ground truth label, and the $y_{pred}$ is the predicted value.

`mlp_numpy.py`: a MLP module implemented by `numpy`. `__init__` initializes the layers of the MLP. `forward` is used to compute the outputs. `backward` is used to compute the gradients. The network architecture is shown below (if using the default arguments):

<img src="./Part 2/img/structure.png" alt="structure"  />

### Task 2

In this task, a batch gradient descent trainer should be implemented. Using the data generated and separated in `dataset.py`, we pass them to the train  method.

In the train method, we do the training as the following pseudocode:

```pseudocode
for i from 1 to max_epoch
	for all data in training set
		Calculate the output of x by the model
		Compute the loss between the output and ground truth
		Call backward method to do back propagation
	end for
	Update parameters of the model
	
	if i % eval_freq == 0
		Evaluate the accuracy of current model on test set
	end if
end for
```

### Task 3

Training settings: 

- `Hidden units`: [20]
- `Learning rate`: 1e-1
- `max_epoch`: 1500
- `eval_freq`: 5

To draw the accuracy curves, I create a notebook `visualize.ipynb` and the package `matplotlib` is used. One of the experiments results in the chart below:

<img src="./Part 2/img/accuracy_curve_BGD.jpg" alt="accuracy_curve_BGD" style="zoom:67%;" />



## Part III: stochastic gradient descent

### Task 1

To implement SGD, we can just change the way we `step` the gradient.

```pseudocode
for i from 1 to max_epoch
	for all data in training set
		Calculate the output of x by the model
		Compute the loss between the output and ground truth
		Call backward method to do back propagation
		if mode == "SGD"
			Update parameters of the model
		end if
	end for
	if mode == "BGD"
		Update parameters of the model
	end if
	
	if i % eval_freq == 0
		Evaluate the accuracy of current model on test set
	end if
end for
```

### Task 2

Training settings: 

- `Hidden units`: [20]
- `Learning rate`: 1e-2
- `max_epoch`: 1500
- `eval_freq`: 1

<img src="./Part 2/img/accuracy_curve_SGD.jpg" alt="accuracy_curve_SGD" style="zoom:67%;" />

### Analysis of Part 2 and 3

From the two curves chart shown above, we can easily compare the two gradient descent method. 

For BGD, it provides steady but slow steps, and for SGD, it provides fluctuant but fast steps.

