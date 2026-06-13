import numpy as np
import sys
import os
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
try:
    from .network import *
    from .loss import *
    from .data_loader import *
    from .activations import *
except ImportError:
    from network import *
    from loss import *
    from data_loader import *
    from activations import *

x_train, y_train , x_val, y_val,  x_test, y_test =  load_data()
def predict(W1, B1, W2, B2, W3, B3, X):
    _, _, _, _, _, A3 = forward_pass(W1, B1, W2, B2, W3, B3, X)
    return np.argmax(A3, axis=0)

def accuracy(A3, y_true):
    predictions = np.argmax(A3, axis=0)
    return np.mean(predictions == y_true) * 100


def save_model(W1, B1, W2, B2, W3, B3):
    os.makedirs(MODELS_DIR, exist_ok=True)
    np.save(os.path.join(MODELS_DIR, 'W1.npy'), W1)
    np.save(os.path.join(MODELS_DIR, 'B1.npy'), B1)
    np.save(os.path.join(MODELS_DIR, 'W2.npy'), W2)
    np.save(os.path.join(MODELS_DIR, 'B2.npy'), B2)
    np.save(os.path.join(MODELS_DIR, 'W3.npy'), W3)
    np.save(os.path.join(MODELS_DIR, 'B3.npy'), B3)

def load_model():
    W1 = np.load(os.path.join(MODELS_DIR, 'W1.npy'))
    B1 = np.load(os.path.join(MODELS_DIR, 'B1.npy'))
    W2 = np.load(os.path.join(MODELS_DIR, 'W2.npy'))
    B2 = np.load(os.path.join(MODELS_DIR, 'B2.npy'))
    W3 = np.load(os.path.join(MODELS_DIR, 'W3.npy'))
    B3 = np.load(os.path.join(MODELS_DIR, 'B3.npy'))
    return W1, B1, W2, B2, W3, B3

def train(X_train, Y_train, epochs, learning_rate):
    W1, B1, W2, B2, W3, B3 = init_weights()
    Y = one_hot_encoding(Y_train).T

    batch_size = 48
    num_batches = X_train.shape[0] // batch_size

    for epoch in range(epochs):
        indices = np.random.permutation(X_train.shape[0])
        X_train = X_train[indices]
        Y_train = Y_train[indices]
        Y = Y[:, indices]

        for i in range(num_batches):
            X_batch = X_train[i * 48:i * 48 + 48, :]
            Y_batch = Y[:, i*48:i*48+48]  # slice columns, not rows
        # 1. forward pass
            z1, z2, z3, A1, A2, A3 = forward_pass(W1, B1, W2, B2, W3, B3, X_batch)
        # 2. compute loss
            loss = cross_entropy_loss(Y_batch, A3)
        # 3. backward pass
            dW1 , dB1 , dW2 , dB2 , dW3, dB3 = backward_pass(W1, B1, W2, B2, W3, B3, z1, z2, A1, A2 ,z3, A3 , X_batch , Y_batch )
        # 4. update parameters
            W1, B1, W2, B2, W3, B3 = update_parameter(W1, B1, W2, B2, W3, B3, dW1 , dB1 , dW2 , dB2 , dW3, dB3 , learning_rate)
        # 5. print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            _, _, _, _, _, A3_full = forward_pass(W1, B1, W2, B2, W3, B3, X_train)
            acc = accuracy(A3_full, Y_train)
            _, _, _, _, _, A3_val = forward_pass(W1, B1, W2, B2, W3, B3, x_val)
            val_acc = accuracy(A3_val, y_val)
            _, _, _, _, _, A3_test = forward_pass(W1, B1, W2, B2, W3, B3, x_test)
            test_acc = accuracy(A3_test, y_test)
            print(f"Epoch {epoch+1} | Loss: {loss:.8f} | Train: {acc:.2f}% | Val: {val_acc:.2f}% | test: {test_acc:.2f}%")
    return W1, B1, W2, B2, W3, B3


if __name__ == '__main__':
    W1, B1, W2, B2, W3, B3 = train(x_train, y_train, epochs=100, learning_rate=0.1)
    save_model(W1, B1, W2, B2, W3, B3)





