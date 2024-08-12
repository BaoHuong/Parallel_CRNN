import numpy as np
import time
from numba import cuda, jit
import os
from PIL import Image
import math

# Define a synthetic dataset path
synthetic_dataset = '/content/Synthetic_Word_Dataset'

# Load images from the dataset
def load_images(dataset_path):
    images = []
    for img_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_name)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_array = np.array(img).astype(np.float32)  # Convert to float32
        images.append(img_array)
    return images

# Define a simple batch normalization layer
def batch_normalization(inputs):
    # Calculate mean
    sum_values = 0.0
    n = inputs.shape[0] * inputs.shape[1]

    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            sum_values += inputs[i, j]

    mean = sum_values / n

    # Calculate variance
    sum_squared_diff = 0.0
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            sum_squared_diff += (inputs[i, j] - mean) ** 2

    var = sum_squared_diff / n

    # Normalize
    normalized = [[0.0 for _ in range(inputs.shape[1])] for _ in range(inputs.shape[0])]
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            normalized[i][j] = (inputs[i, j] - mean) / (var + 1e-5) ** 0.5

    return normalized

# Sequential execution of batch normalization
def sequential_batch_norm(images):
    normalized_images = []
    for img in images:
        normalized_img = batch_normalization(img)
        normalized_images.append(normalized_img)
    return normalized_images

# CUDA kernel for batch normalization
@cuda.jit
def cuda_batch_normalization(inputs, outputs, mean, var, epsilon):
    x, y = cuda.grid(2)
    if x < inputs.shape[0] and y < inputs.shape[1]:
        # Normalize
        outputs[x, y] = (inputs[x, y] - mean[0, 0]) / ((var[0, 0] + epsilon) ** 0.5)

# Batch normalization with CUDA
def batch_norm_cuda(images):
    cuda_images = []
    for img in images:
        mean = np.mean(img, axis=(0, 1), keepdims=True).astype(np.float32)
        var = np.var(img, axis=(0, 1), keepdims=True).astype(np.float32)
        epsilon = 1e-5  # Small constant to prevent division by zero

        output = np.zeros_like(img)
        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(img.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(img.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # Launch the CUDA kernel
        cuda_batch_normalization[blockspergrid, threadsperblock](img, output, mean, var, epsilon)
        cuda.synchronize()  # Ensure kernel completion

        cuda_images.append(output)

    return cuda_images

# JIT-accelerated batch normalization
@jit(nopython=True, parallel=True)
def jit_batch_normalization(inputs):
    # Calculate mean and variance without using the 'axis' argument
    mean = np.sum(inputs) / (inputs.shape[0] * inputs.shape[1])
    var = np.sum((inputs - mean)**2) / (inputs.shape[0] * inputs.shape[1])
    normalized = (inputs - mean) / np.sqrt(var + 1e-5)
    return normalized

# Batch normalization with JIT
def batch_norm_jit(images):
    jit_images = []
    for img in images:
        normalized_img = jit_batch_normalization(img)
        jit_images.append(normalized_img)
    return jit_images

# Optimized CUDA kernel for batch normalization using 3D blocks
@cuda.jit
def cuda_batch_normalization_3d(inputs, outputs, mean, var, epsilon):
    x, y, z = cuda.grid(3)
    if x < inputs.shape[0] and y < inputs.shape[1] and z < inputs.shape[2]:
        outputs[x, y, z] = (inputs[x, y, z] - mean[z]) / ((var[z] + epsilon) ** 0.5)

def batch_norm_cuda_3d(images):
    cuda_3D_images = []
    for img in images:
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)  # Add a dummy channel dimension for uniformity

        mean = np.mean(img, axis=(0, 1), dtype=np.float32)
        var = np.var(img, axis=(0, 1), dtype=np.float32)
        epsilon = 1e-5  # Small constant to prevent division by zero

        # Prepare output array
        output = np.zeros_like(img)

        # Set up CUDA grid and block dimensions
        threadsperblock = (16, 16, 4)  # Optimized block size
        blockspergrid_x = int(np.ceil(img.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(img.shape[1] / threadsperblock[1]))
        blockspergrid_z = int(np.ceil(img.shape[2] / threadsperblock[2]))
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

        # Launch the CUDA kernel
        cuda_batch_normalization_3d[blockspergrid, threadsperblock](img, output, mean, var, epsilon)
        cuda.synchronize()  # Ensure kernel completion

        cuda_3D_images.append(output)

    return cuda_3D_images

@cuda.jit
def cuda_shared_memory_batch_normalization(inputs, outputs, mean, var, epsilon):
    # Define shared memory
    shared_inputs = cuda.shared.array(shape=(16, 16), dtype=np.float32)

    # Get the thread indices
    x, y = cuda.grid(2)

    # Load data into shared memory
    if x < inputs.shape[0] and y < inputs.shape[1]:
        shared_inputs[cuda.threadIdx.x, cuda.threadIdx.y] = inputs[x, y]
    cuda.syncthreads()

    # Perform batch normalization
    if x < inputs.shape[0] and y < inputs.shape[1]:
        norm_val = (shared_inputs[cuda.threadIdx.x, cuda.threadIdx.y] - mean[0, 0]) / ((var[0, 0] + epsilon) ** 0.5)
        outputs[x, y] = norm_val

def batch_norm_cuda_shared_memory(images):
    cuda_shared_images = []
    for img in images:
        mean = np.mean(img, axis=(0, 1), keepdims=True).astype(np.float32)
        var = np.var(img, axis=(0, 1), keepdims=True).astype(np.float32)
        epsilon = 1e-5  # Small constant to prevent division by zero

        output = np.zeros_like(img)
        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(img.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(img.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # Launch the CUDA kernel
        cuda_shared_memory_batch_normalization[blockspergrid, threadsperblock](img, output, mean, var, epsilon)
        cuda.synchronize()  # Ensure kernel completion

        cuda_shared_images.append(output)

    return cuda_shared_images


# Load dataset
images = load_images(synthetic_dataset)

# Measure time for sequential execution
start_time_seq = time.time()
sequential_output = sequential_batch_norm(images)
end_time_seq = time.time()
time_seq = end_time_seq - start_time_seq

# Measure time for CUDA execution
start_time_cuda = time.time()
cuda_output = batch_norm_cuda(images)
end_time_cuda = time.time()
time_cuda = end_time_cuda - start_time_cuda

# Measure time for JIT execution
start_time_jit = time.time()
jit_output = batch_norm_jit(images)
end_time_jit = time.time()
time_jit = end_time_jit - start_time_jit

# Measure time for 3D execution
start_time_3D = time.time()
Cuda3D_output = batch_norm_cuda_3d(images)
end_time_3D = time.time()
time_3D = end_time_3D - start_time_3D

# Measure time for Shared Memory execution
start_time_shared = time.time()
Cuda_shared_output = batch_norm_cuda_shared_memory(images)
end_time_shared = time.time()
time_shared = end_time_shared - start_time_shared


# Error analysis between sequential and CUDA outputs
error_seq_cuda = np.mean([np.abs(seq_img - cuda_img).mean() # Calculate mean for each image pair
                         for seq_img, cuda_img in zip(sequential_output, cuda_output)])

# Error analysis between sequential and JIT outputs
error_seq_jit = np.mean([np.abs(seq_img - jit_img).mean() # Calculate mean for each image pair
                        for seq_img, jit_img in zip(sequential_output, jit_output)])

# Error analysis between sequential and JIT outputs
# Ensure the shapes match by expanding the sequential output
error_seq_3D = np.mean([np.abs(np.expand_dims(seq_img, axis=-1) - cuda_3D_img).mean()
                        for seq_img, cuda_3D_img in zip(sequential_output, Cuda3D_output)])

# Compare optimized CUDA results with sequential results
error_seq_shared = np.mean([np.abs(seq_img - cuda_shared_images).mean() # Calculate mean for each image pair
                        for seq_img, cuda_shared_images in zip(sequential_output, Cuda_shared_output)])


# Results
print(f"Sequential Execution Time: {time_seq:.3f} seconds")
print(f"JIT Execution Time: {time_jit:.3f} seconds")

print(f"CUDA Execution Time: {time_cuda:.3f} seconds")
print(f"CUDA 3D Execution Time: {time_3D:.3f} seconds")
#print(f"Optimized CUDA with Shared Memory Execution Time: {time_cuda_shared:.6f} seconds")
print(f"CUDA Shared Memory Execution Time: {time_shared:.3f} seconds")

print(f"Error between Sequential and CUDA Outputs: {error_seq_cuda:.3f}")
print(f"Error between Sequential and JIT Outputs: {error_seq_jit:.3f}")
print(f"Error between Sequential and CUDA 3D Outputs: {error_seq_3D:.6f}")

print(f"Error between Sequential and shared Outputs: {error_seq_shared:.6f}")
