import numpy as np
from numba import jit, cuda
import time
import math

class BatchNormalization:
    def __init__(self, input_size, epsilon=1e-5, momentum=0.9):
        self.gamma = np.ones(input_size)
        self.beta = np.zeros(input_size)
        self.epsilon = epsilon
        self.momentum = momentum
        self.running_mean = np.zeros(input_size)
        self.running_var = np.ones(input_size)

    def forward(self, x, training=True):
        if training:
            # Calculate mean and variance for the current batch
            batch_mean = [sum(x[:, j]) / len(x) for j in range(len(x[0]))]
            batch_var = [(sum([(x[i, j] - batch_mean[j]) ** 2 for i in range(len(x))]) / len(x)) for j in range(len(x[0]))]
            
            # Normalize the input
            normalized_x = [[(x[i, j] - batch_mean[j]) / (batch_var[j] + self.epsilon) ** 0.5 for j in range(len(x[0]))] for i in range(len(x))]
            
            # Scale and shift
            out = [[self.gamma[j] * normalized_x[i][j] + self.beta[j] for j in range(len(x[0]))] for i in range(len(x))]
            
            # Update running mean and variance
            self.running_mean = [(self.momentum * self.running_mean[j] + (1 - self.momentum) * batch_mean[j]) for j in range(len(batch_mean))]
            self.running_var = [(self.momentum * self.running_var[j] + (1 - self.momentum) * batch_var[j]) for j in range(len(batch_var))]
            
            return np.array(out)

@jit(nopython=True)
def forward_batchnorm_jit(input, gamma, beta, mean, var, epsilon):
    normalized_input = (input - mean) / np.sqrt(var + epsilon)
    output = gamma * normalized_input + beta
    return output

@cuda.jit
def forward_batchnorm_cuda(input, gamma, beta, mean, var, epsilon, output):
    i, j = cuda.grid(2)
    if i < input.shape[0] and j < input.shape[1]:
        # Use double precision for all operations
        inp_val = input[i, j] - mean[j]
        var_eps = var[j] + epsilon
        normalized_input = inp_val / math.sqrt(var_eps)
        output[i, j] = gamma[j] * normalized_input + beta[j]

@cuda.jit
def forward_batchnorm_cuda_3D(input, gamma, beta, mean, var, epsilon, output):
    i, j, k = cuda.grid(3)
    if i < input.shape[0] and j < input.shape[1] and k == 0:  # Use k as a dummy dimension
        # Use double precision for all operations
        inp_val = input[i, j] - mean[j]
        var_eps = var[j] + epsilon
        normalized_input = inp_val / math.sqrt(var_eps)
        output[i, j] = gamma[j] * normalized_input + beta[j]


# Generating synthetic data
input_size = 512  # Size based on previous layer output shape
batch_size = 256
input_data = np.random.rand(batch_size, input_size) - 0.5

# Creating BatchNormalization layer
batchnorm_layer = BatchNormalization(input_size)

# Sequential Execution (training)
seq_start = time.time()
output_bn_seq_train = batchnorm_layer.forward(input_data, training=True)
seq_end = time.time()


# JIT Execution
batch_mean = np.mean(input_data, axis=0)
batch_var = np.var(input_data, axis=0)
jit_start = time.time()
output_bn_jit = forward_batchnorm_jit(input_data, batchnorm_layer.gamma, batchnorm_layer.beta, batch_mean, batch_var, batchnorm_layer.epsilon)
jit_end = time.time()

# CUDA Execution
output_bn_cuda = np.zeros_like(input_data)

# Define 2D grid and block dimensions
threadsperblock = (16, 16)
blockspergrid_x = int(np.ceil(batch_size / threadsperblock[0]))
blockspergrid_y = int(np.ceil(input_size / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)
cuda_start = time.time()
forward_batchnorm_cuda[blockspergrid, threadsperblock](input_data, batchnorm_layer.gamma, batchnorm_layer.beta, batch_mean, batch_var, batchnorm_layer.epsilon, output_bn_cuda)
cuda_end = time.time()

# Assuming input_data has shape (batch_size, height, width)

output_bn_cuda_3D = np.zeros_like(input_data)

# Define 3D grid and block dimensions
threadsperblock = (8, 8, 8)
blockspergrid_x = int(np.ceil(batch_size / threadsperblock[0]))
blockspergrid_y = int(np.ceil(input_size / threadsperblock[1]))
blockspergrid_z = 1  # k dimension is just a dummy to facilitate 3D block structure
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

cuda_start_3D = time.time()
forward_batchnorm_cuda[blockspergrid, threadsperblock](input_data, batchnorm_layer.gamma, batchnorm_layer.beta, batch_mean, batch_var, batchnorm_layer.epsilon, output_bn_cuda_3D)
cuda_end_3D = time.time()



# Timing and Error Analysis
print("BatchNormalization Execution Times")
print(f"Time Sequential (Training): {seq_end - seq_start}")
print(f"Time JIT: {jit_end - jit_start}")
print(f"Time CUDA: {cuda_end - cuda_start}")
print(f"Time CUDA 3D: {cuda_end_3D - cuda_start_3D}")

# Calculating Errors
print(f"Error between Sequential (Training) and JIT: {np.mean(np.abs(output_bn_seq_train - output_bn_jit))}")
print(f"Error between Sequential (Training) and CUDA: {np.mean(np.abs(output_bn_seq_train - output_bn_cuda))}")
print(f"Error between Sequential (Training) and CUDA: {np.mean(np.abs(output_bn_seq_train - output_bn_cuda_3D))}")
