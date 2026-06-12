import numpy as np
import pandas as pd

import struct

def load_images(filepath):
    with open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>4I', f.read(16))
        pixels = np.frombuffer(f.read(), dtype=np.uint8)
        pixels = pixels.reshape(num_images, rows * cols)
        pixels = pixels.astype(np.float64) / 255.0
    return pixels

def load_labels(filepath):
    with open(filepath, 'rb') as f :
        magic , num_labels = struct.unpack('>2I' ,f.read(8))
        labels  = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def one_hot_encoding(labels, num_classes=10):
    y = labels
    encoded = np.zeros((labels.shape[0], num_classes))
    for i in range(labels.shape[0]):
        encoded[i, y[i]] = 1
    return encoded

def load_data():
    train_data = load_images('../data/train-images.idx3-ubyte')
    train_labels = load_labels('../data/train-labels.idx1-ubyte')
    m , n = train_data.shape

    X_train = train_data[int(0.2*m):m , :]
    y_train = train_labels[int(0.2*m):m]

    X_val = train_data[0:int(0.2*m) , :]
    y_val = train_labels[0:int(0.2*m)]

    X_test  = load_images('../data/t10k-images.idx3-ubyte').T
    y_test  = load_labels('../data/t10k-labels.idx1-ubyte')
    return X_train, y_train, X_val , y_val, X_test, y_test

