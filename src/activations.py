import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return z > 0

def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)  # max per column
    return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)  # sum per column
