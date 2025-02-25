# Custom-Neural-Network-for-Diabetes-Classification

This project implements a neural network from scratch using Python and NumPy to classify diabetes data. It demonstrates how to build a simple feedforward neural network with manual implementations of forward and backward passes, various activation functions, dropout regularization, loss functions, and different optimization algorithms.
___
#### Table of Contents
* <ul>Introduction</ul>
* <ul>Features</ul>
* <ul>Dataset</ul>
* <ul>Installation</ul>
* <ul>Usage</ul>
* <ul>Project Structure</ul>
* <ul>Experimental Results</ul>
* <ul>Conclusion</ul>
* <ul>License</ul>

___
## Introduction

This project demonstrates how to build and train a neural network from scratch without relying on high-level deep learning libraries. The network is designed for binary classification on a diabetes dataset. It includes custom implementations of:

* <ul>Dense (fully connected) layers with L1/L2 regularization.</ul>
* <ul>Activation functions such as ReLU and Softmax.</ul>
* <ul>A dropout layer for regularization.</ul>
* <ul>Loss functions, including Categorical Cross-Entropy.</ul>
* <ul>Multiple optimization algorithms: SGD with momentum, Adagrad, RMSprop, and Adam.</ul>
The code also illustrates how to perform forward and backward propagation manually, update parameters using various optimizers, and evaluate model performance.

___
## Features
<ul>Custom Layers:

* <ul>Dense Layers: Implements weight and bias initialization, forward pass, and backward propagation with regularization.</ul>
* <ul>Activation Functions: ReLU for non-linearity and Softmax for output probability distribution.</ul>
* <ul>Dropout: Prevents overfitting by randomly deactivating a fraction of neurons during training.</ul></ul>

<ul>Loss Calculation:

* <ul>Categorical Cross-Entropy Loss: Computes the loss and its gradient with respect to predictions.</ul></ul>

<ul>Optimizers:

* <ul>SGD with Momentum: Standard gradient descent enhanced with momentum.</ul>
* <ul>Adagrad: Adaptive learning rate algorithm that adjusts the learning rate per parameter.</ul>
* <ul>RMSprop: Uses an exponentially decaying average of squared gradients for adaptive learning rates.</ul>
* <ul>Adam: Combines momentum and RMSprop techniques with bias correction for parameter updates.</ul></ul>

<ul>Training & Evaluation:

* <ul>Splits the diabetes dataset into training and testing sets.</ul>
* <ul>Trains the model over multiple epochs, printing loss, accuracy, and learning rate at regular intervals.</ul>
* <ul>Evaluates the model performance on a held-out test set.</ul></ul>

___
## Dataset

The dataset used is a diabetes dataset (diabetes.csv.xls), which contains several health indicators and an Outcome column indicating whether the patient has diabetes (binary classification). The dataset is preprocessed using pandas and split into training and testing sets using scikit-learn's train_test_split.

___
## Installation

#### Prerequisites
Ensure you have Python installed (Python 3.x is recommended). Install the required packages using pip:
```python
pip install numpy pandas scikit-learn
```

#### Clone the Repository

Clone this repository to your local machine:

```
git clone https://github.com/your-username/diabetes-neural-network.git
cd diabetes-neural-network
```

___
## Usage

Run the notebook or Python script to train and evaluate the neural network:

```
python neural_network.py
```

The program will:

* <ul>Load and preprocess the diabetes dataset.</ul>
* <ul>Initialize the network layers, loss function, and an optimizer (you can switch between Adagrad, SGD, RMSprop, or Adam).</ul>
* <ul>Train the network while printing the epoch number, accuracy, loss, and current learning rate.</ul>
* <ul>Finally, evaluate the model on the test set and print the validation accuracy and loss.</ul>

___
## Project Structure

<ul>Layers and Activations:

* <ul>Dense: Implements a fully connected layer with forward and backward propagation.</ul>
* <ul>ReLU: Activation function applying the rectified linear unit.</ul>
* <ul>Dropout: Regularization technique to prevent overfitting.</ul>
* <ul>Softmax: Activation function for output layer to compute class probabilities.</ul></ul>

<ul>Loss Functions:

* <ul>Loss_CategoricalCrossEntropy: Computes the cross-entropy loss and its gradient.</ul>
* <ul>Softmax_CrossEntropy: Combines softmax activation with categorical cross-entropy loss.</ul></ul>

<ul>Optimizers:

* <ul>SGD: Implements stochastic gradient descent with momentum.</ul>
* <ul>Adagrad: Adjusts the learning rate based on the sum of squared gradients.</ul>
* <ul>RMSprop: Uses a decaying average of squared gradients for adaptive learning rates.</ul>
* <ul>Adam: Combines momentum and RMSprop with bias correction.</ul></ul>

<ul>Training & Testing:

* <ul>The training loop feeds the data through the network, computes the loss (data + regularization), backpropagates the gradients, and updates parameters using the chosen optimizer.</ul>
* <ul>After training, the model is evaluated on the test set.</ul></ul>

___
## Experimental Results

Multiple optimizers were tested in this project. The experimental results on the diabetes dataset were as follows:

<ul>Adagrad:
  
* <ul>Achieved a validation accuracy of approximately 62.3% with a loss of 0.786.</ul>
* <ul>Adagrad’s adaptive learning rate per parameter helped the model converge more efficiently.</ul></ul>

<ul>SGD with Momentum, RMSprop, and Adam:
  
* <ul>These optimizers reached around 59.1% accuracy on the validation set, with slightly different loss values.</ul></ul>

___
## Conclusion

Based on the experiments conducted, Adagrad demonstrated superior performance over other optimizers for this diabetes classification task. Its ability to adaptively adjust the learning rate for each parameter based on historical gradient information appears to contribute to more efficient convergence and improved overall accuracy.
