import numpy as np
from numba import cuda, jit
import time

class Dense:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weight = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(output_size) - 0.5

    def forward_seq(self, input):
        output = np.zeros((input.shape[0], self.output_size))
        for i in range(input.shape[0]):
            for j in range(self.output_size):
                output[i, j] = np.dot(input[i], self.weight[:, j]) + self.bias[j]
                output[i, j] = max(0, output[i, j])  # ReLU activation
        return output

@jit(nopython=True)
def forward_jit(input, weight, bias):
    output = np.zeros((input.shape[0], weight.shape[1]))
    for i in range(input.shape[0]):
        for j in range(weight.shape[1]):
            output[i, j] = np.dot(input[i], weight[:, j]) + bias[j]
            output[i, j] = max(0, output[i, j])  # ReLU activation
    return output

@cuda.jit
def forward_cuda(input, weight, bias, output):
    i, j = cuda.grid(2)
    if i < input.shape[0] and j < weight.shape[1]:
        val = 0
        for k in range(weight.shape[0]):
            val += input[i, k] * weight[k, j]
        output[i, j] = max(0, val + bias[j])  # ReLU activation

# Generating synthetic data
input_size = 512  # Size based on previous layer output shape (4, 32, 512)
output_size = 63  # Final output size
batch_size = 256
input_data = np.random.rand(batch_size, input_size) - 0.5

# Creating Dense layer
dense_layer = Dense(input_size, output_size)

# Sequential Execution
seq_start = time.time()
output_seq = dense_layer.forward_seq(input_data)
seq_end = time.time()

# JIT Execution
jit_start = time.time()
output_jit = forward_jit(input_data, dense_layer.weight, dense_layer.bias)
jit_end = time.time()

# CUDA Execution
output_cuda = np.zeros((batch_size, output_size))
threadsperblock = (16, 16)
blockspergrid_x = int(np.ceil(batch_size / threadsperblock[0]))
blockspergrid_y = int(np.ceil(output_size / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

cuda_start = time.time()
forward_cuda[blockspergrid, threadsperblock](input_data, dense_layer.weight, dense_layer.bias, output_cuda)
cuda_end = time.time()

# Timing and Error Analysis
print("Dense Layer Execution Times")
print(f"Time Sequential: {seq_end - seq_start}")
print(f"Time JIT: {jit_end - jit_start}")
print(f"Time CUDA: {cuda_end - cuda_start}")

print(f"Error between Sequential and JIT: {np.sum(np.abs(output_seq - output_jit))}")
print(f"Error between Sequential and CUDA: {np.sum(np.abs(output_seq - output_cuda))}")
