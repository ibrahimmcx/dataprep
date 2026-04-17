import numpy as np
from .activations import Activation

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Xavier/Glorot Initialization for weights
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
        self.bias = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        # Gradients
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # Update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

class ActivationLayer(Layer):
    def __init__(self, activation: Activation):
        super().__init__()
        self.activation = activation

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation.forward(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        # element-wise multiplication by derivative
        return output_error * self.activation.derivative(self.input)
