import numpy as np

def relu(z):
    """Apply ReLU activation — returns max(0, z) elementwise."""
    return np.maximum(0, z)

def relu_derivative(z):
    """Return binary mask of ReLU derivative — 1 where z > 0, else 0."""
    return z > 0

def softmax(z):
    """Apply numerically stable softmax — converts logits to probabilities."""
    z = z - np.max(z, axis=0, keepdims=True)  # max per column
    return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)  # sum per column
