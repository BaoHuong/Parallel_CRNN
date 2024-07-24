import numpy as np
import math
import time
from numba import jit, cuda

# Normalization Layer Implementation
class Normalization():
    def __init__(self, numKernel):
        self.numKernel = numKernel
        self.mean = np.zeros(numKernel)
        self.variance = np.zeros(numKernel)

    def compute_mean_variance(self, data, shape):
        for i in range(self.numKernel):
            self.mean[i] = np.mean(data[i])
            self.variance[i] = np.var(data[i])

    def forward(self, input, shape, epsilon=1e-5):
        self.compute_mean_variance(input, shape)
        output = np.zeros_like(input, dtype=float)
        for i in range(self.numKernel):
            for j in range(shape[0] * shape[1]):
                output[i][j] = (input[i][j] - self.mean[i]) / np.sqrt(self.variance[i] + epsilon)
        return output

# JIT-compiled normalization forward pass
@jit(nopython=True)
def forward_jit(input, output, mean, variance, shape, numKernel, epsilon=1e-5):
    for i in range(numKernel):
        for j in range(shape[0] * shape[1]):
            output[i][j] = (input[i][j] - mean[i]) / np.sqrt(variance[i] + epsilon)
    return output

# CUDA-compiled normalization forward pass
@cuda.jit()
def forward_cuda(input, output, mean, variance, shape, numKernel, epsilon=1e-5): # Added epsilon as an argument
    ir, ic = cuda.grid(2)
    if ir < numKernel and ic < shape[0] * shape[1]:
        # Use CUDA's sqrt function from the math library
        output[ir][ic] = (input[ir][ic] - mean[ir]) / math.sqrt(variance[ir] + epsilon)

# Example usage:
w = 128
h = 32
a = np.random.rand(1, w * h)
aSize = (h, w)

layer = Normalization(1)

# Sequential execution
s_start = time.time()
output_seq = layer.forward(a, aSize)
s_end = time.time()

print("Layer Normalization - Sequential")
print(f'Time seq: \t{s_end - s_start}')

# JIT execution
mean = np.mean(a, axis=1)
variance = np.var(a, axis=1)
output_jit = np.zeros_like(a, dtype=float)

jit_start = time.time()
forward_jit(a, output_jit, mean, variance, aSize, 1)
jit_end = time.time()

print("Layer Normalization - JIT")
print(f'Time jit: \t{jit_end - jit_start}')
print(f'Error between seq and jit: \t{np.sum(np.abs(output_seq - output_jit))}')

# CUDA execution
output_cuda = np.zeros_like(a, dtype=float)
block_size = (16, 16)
grid_size = (math.ceil(1 / block_size[0]), math.ceil((h * w) / block_size[1]))

cuda_start = time.time()
forward_cuda[grid_size, block_size](a, output_cuda, mean, variance, aSize, 1, 1e-5) # Pass epsilon to forward_cuda
cuda_end = time.time()

print("Layer Normalization - CUDA")
print(f'Time cuda: \t{cuda_end - cuda_start}')
print(f'Error between seq and cuda: \t{np.sum(np.abs(output_seq - output_cuda))}')


import os
from PIL import Image

# Function to load and preprocess images from the dataset
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert('L')
        if img is not None:
            img = img.resize((128, 32))
            img = np.array(img).astype(np.float32).flatten()
            images.append(img)
    return np.array(images)

# Load images
folder_path = 'path_to_synthetic_word_dataset'
images = load_images_from_folder(folder_path)

# Normalize images
num_images = len(images)
w, h = 128, 32
input_data = images.reshape((num_images, 1, w * h))
aSize = (h, w)

layer = Normalization(1)

# Sequential execution on dataset
s_start = time.time()
output_seq = np.zeros_like(input_data)
for i in range(num_images):
    output_seq[i] = layer.forward(input_data[i], aSize)
s_end = time.time()

print("Sequential Normalization on Dataset")
print(f'Time seq: \t{s_end - s_start}')

# JIT execution on dataset
output_jit = np.zeros_like(input_data, dtype=float)
mean = np.mean(input_data, axis=1)
variance = np.var(input_data, axis=1)

jit_start = time.time()
for i in range(num_images):
    forward_jit(input_data[i], output_jit[i], mean[i], variance[i], aSize, 1)
jit_end = time.time()

print("JIT Normalization on Dataset")
print(f'Time jit: \t{jit_end - jit_start}')
print(f'Error between seq and jit: \t{np.sum(np.abs(output_seq - output_jit))}')

# CUDA execution on dataset
output_cuda = np.zeros_like(input_data, dtype=float)
block_size = (16, 16)
grid_size = (math.ceil(1 / block_size[0]), math.ceil((h * w) / block_size[1]))

cuda_start = time.time()
for i in range(num_images):
    forward_cuda[grid_size, block_size](input_data[i], output_cuda[i], mean[i], variance[i], aSize, 1)
cuda_end = time.time()

print("CUDA Normalization on Dataset")
print(f'Time cuda: \t{cuda_end - cuda_start}')
print(f'Error between seq and cuda: \t{np.sum(np.abs(output_seq - output_cuda))}')
