import numpy as np
import random

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.biases = np.zeros(output_size)

    def forward(self, input):
        self.last_input_shape = input.shape
        self.last_input = input
        input = input.flatten()
        self.last_input_flat = input
        return np.dot(input, self.weights) + self.biases