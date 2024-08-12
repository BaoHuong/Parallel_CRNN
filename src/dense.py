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
        for i in range(len(input)):
            for j in range(self.output_size):
              val = 0
              for k in range(self.input_size):
                val += input[i, k] * self.weight[k, j]
              output[i][j] = max(0, val + self.bias[j])  # ReLU activation
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


@cuda.jit
def forward_cuda_optimized(input, weight, bias, output):
    # Define shared memory arrays for faster access
    shared_input = cuda.shared.array(shape=(16, 16), dtype=np.float32)
    shared_weight = cuda.shared.array(shape=(16, 16), dtype=np.float32)

    # Get thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x  # Block width
    bh = cuda.blockDim.y  # Block height

    row, col = cuda.grid(2)
    # Initialize output value
    val = 0.0

    # Loop over input and weight blocks
    for i in range((input.shape[1] + 16 - 1) // 16):
        # Load input and weights into shared memory
        if row < input.shape[0] and i * 16 + ty < input.shape[1]:
            shared_input[tx, ty] = input[row, i * 16 + ty]
        else:
            shared_input[tx, ty] = 0.0

        if col < weight.shape[1] and i * 16 + tx < weight.shape[0]:
            shared_weight[tx, ty] = weight[i * 16 + tx, col]
        else:
            shared_weight[tx, ty] = 0.0

        # Synchronize threads to ensure shared memory is fully loaded
        cuda.syncthreads()

        # Perform computation
        for j in range(16):
            val += shared_input[tx, j] * shared_weight[j, ty]

        # Synchronize again before loading the next block
        cuda.syncthreads()

    # Write the result back to global memory with ReLU activation
    if row < output.shape[0] and col < output.shape[1]:
        output[row, col] = max(0, val + bias[col])


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

# Updated CUDA Execution Timing
output_cuda_optimized = np.zeros((batch_size, output_size))

threadsperblock = (16, 16)
blockspergrid_x = int(np.ceil(batch_size / threadsperblock[0]))
blockspergrid_y = int(np.ceil(output_size / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

cuda_start_optimized = time.time()
forward_cuda_optimized[blockspergrid, threadsperblock](input_data, dense_layer.weight, dense_layer.bias, output_cuda_optimized)
cuda_end_optimized = time.time()

# Timing and Error Analysis
print("Optimized CUDA Dense Layer Execution Time")
print(f"Time CUDA Optimized: {cuda_end_optimized - cuda_start_optimized}")
print(f"Error between Sequential and CUDA shared memory: {np.mean(np.abs(output_seq - output_cuda_optimized))}")

# Timing and Error Analysis
print("Dense Layer Execution Times")
print(f"Time Sequential: {seq_end - seq_start}")
print(f"Time JIT: {jit_end - jit_start}")
print(f"Time CUDA: {cuda_end - cuda_start}")
#print(f"Time CUDA 3D Block: {cuda_end_3d - cuda_start_3d}")

print(f"Error between Sequential and JIT: {np.mean(np.abs(output_seq - output_jit))}")
print(f"Error between Sequential and CUDA: {np.mean(np.abs(output_seq - output_cuda))}")
#print(f"Error between Sequential and CUDA 3D: {np.sum(np.abs(output_seq - output_cuda_3d.transpose(0, 2, 1)))}")  # Transpose output_cuda_3d for comparison
