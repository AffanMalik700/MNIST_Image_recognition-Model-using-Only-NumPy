import numpy as np
from data_loader import *
from activations import *
x_train, y_train , x_val, y_val,  x_test, y_test =  load_data()

def init_weights():
    W1 = np.random.randn(64,784) * 0.05
    B1 = np.zeros((64,1))
    W2 = np.random.randn(16,64) * 0.05
    B2 = np.zeros((16,1))
    W3 = np.random.randn(10,16) * 0.05
    B3 = np.zeros((10,1))
    return W1,B1,W2,B2,W3,B3

def forward_pass(W1,B1,W2,B2,W3,B3,X):
    X = X.T
    z1 = np.matmul(W1,X) + B1
    A1 = relu(z1)
    z2 = np.matmul(W2,A1) + B2
    A2 = relu(z2)
    z3 = np.matmul(W3,A2) + B3
    A3 = softmax(z3)
    return z1, z2 , z3 , A1 , A2 , A3

# W1, B1, W2, B2, W3, B3 = init_weights()
# z1, z2, z3, A1, A2, A3 = forward_pass(W1, B1, W2, B2, W3, B3, x_train)
# print(z1.shape)  # expect (64, 60000)
# print(A1.shape)  # expect (64, 60000)
# print(A3.shape)  # expect (10, 60000)
# W1,B1,W2,B2,W3,B3 = init_weights()
# forward_pass(W1,B1,W2,B2,W3,B3,x_train[:,0])
# Y = one_hot_encoding(y_train, 10)

def backward_pass(W1, B1, W2, B2, W3, B3, z1, z2, A1, A2 ,z3, A3 ,X , Y ):
    n = X.shape[0]

    # Output layer
    dz3 = A3 - Y
    dW3 = (1/n) * dz3 @ A2.T
    dB3 = (1/n) * np.sum(dz3 , axis=1, keepdims=True)

    # Hidden layer 2
    dz2 = (W3.T @ dz3) * relu_derivative(z2)
    dW2 = (1/n) * dz2 @ A1.T
    dB2 = (1/n) * np.sum(dz2 , axis=1, keepdims=True)

    # Hidden layer 1
    dz1 = (W2.T @ dz2) * relu_derivative(z1)
    dW1 = (1/n) * dz1 @ X
    dB1 = (1/n) * np.sum(dz1 , axis=1, keepdims=True)

    return dW1 , dB1 , dW2 , dB2 , dW3, dB3
def update_parameter(W1, B1, W2, B2, W3, B3, dW1 , dB1 , dW2 , dB2 , dW3, dB3 , learning_rate):
    W1 = W1 - learning_rate * dW1
    B1 = B1 - learning_rate * dB1
    W2 = W2 - learning_rate * dW2
    B2 = B2 - learning_rate * dB2
    W3 = W3 - learning_rate * dW3
    B3 = B3 - learning_rate * dB3
    return W1, B1, W2, B2, W3, B3
# x_train = x_train.T
# x_test = x_test.T

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
