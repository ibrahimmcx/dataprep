import numpy as np
from .layers import Layer, Dense, ActivationLayer
from .activations import Activation, ReLU, Sigmoid, Tanh, Softmax

class Sequential:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer: Layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        # Sample by sample prediction
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i:i+1] # Keep 2D shape (1, features)
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return np.array(result).squeeze()

    def fit(self, x_train, y_train, epochs, learning_rate, verbose=True):
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j:j+1]
                for layer in self.layers:
                    output = layer.forward(output)

                # compute loss
                err += self.loss(y_train[j:j+1], output)

                # backward propagation
                error = self.loss_prime(y_train[j:j+1], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            # calculate average error
            err /= samples
            if verbose and (i+1) % 10 == 0:
                print(f'Epoch {i+1}/{epochs}  error={err:.6f}')
