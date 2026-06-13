import numpy as np

def cross_entropy_loss(y_true, y_pred , W1, W2, W3,lambda_reg):
    n = y_pred.shape[1]
    return (-1 / n) * np.sum(y_true * np.log(y_pred+ 1e-8)) + (lambda_reg / (2 * n)) * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
