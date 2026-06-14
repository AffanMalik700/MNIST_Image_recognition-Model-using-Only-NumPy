import numpy as np
import pandas as pd
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
import struct

def load_images(filepath):
    """Load MNIST image binary file and return normalized float64 array of shape (n, 784)."""
    with open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>4I', f.read(16))
        pixels = np.frombuffer(f.read(), dtype=np.uint8)
        #conversion to matrix
        pixels = pixels.reshape(num_images, rows * cols)
        # normalization
        pixels = pixels.astype(np.float64) / 255.0
    return pixels

def load_labels(filepath):
    """Load MNIST label binary file and return uint8 array of shape (n,)."""
    with open(filepath, 'rb') as f :
        magic , num_labels = struct.unpack('>2I' ,f.read(8))
        labels  = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def one_hot_encoding(labels, num_classes=10):
    """Convert integer labels to one-hot encoded matrix of shape (n, num_classes)."""
    y = labels
    encoded = np.zeros((labels.shape[0], num_classes))
    for i in range(labels.shape[0]):
        encoded[i, y[i]] = 1
    return encoded

def load_data():
    """Load and split MNIST data into train, validation and test sets."""
    #Loading Train in the dataframe
    train_data = load_images(os.path.join(BASE_DIR, 'data', 'train-images.idx3-ubyte'))
    train_labels = load_labels(os.path.join(BASE_DIR, 'data', 'train-labels.idx1-ubyte'))
    m , n = train_data.shape
    #Loading Training in the dataframe
    X_train = train_data[int(0.2*m):m , :]
    y_train = train_labels[int(0.2*m):m]
    #Loading Cross Validation in the dataframe
    X_val = train_data[0:int(0.2*m) , :]
    y_val = train_labels[0:int(0.2*m)]
    #Load Testing in the dataframe
    X_test = load_images(os.path.join(BASE_DIR, 'data', 't10k-images.idx3-ubyte'))
    y_test = load_labels(os.path.join(BASE_DIR, 'data', 't10k-labels.idx1-ubyte'))

    return X_train, y_train, X_val , y_val, X_test, y_test

