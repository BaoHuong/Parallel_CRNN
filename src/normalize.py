import numpy as np
import random
import string

# Function to generate random words
def generate_random_word(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

# Function to create a synthetic dataset
def generate_synthetic_dataset(num_samples, word_length):
    dataset = []
    labels = []
    for _ in range(num_samples):
        word = generate_random_word(word_length)
        word_vector = [ord(char) for char in word]  # Convert each character to its ASCII value
        dataset.append(word_vector)
        labels.append(random.randint(0, 1))  # Random binary label
    return np.array(dataset), np.array(labels)

num_samples = 1000
word_length = 32

# Function to standardize the dataset
def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data