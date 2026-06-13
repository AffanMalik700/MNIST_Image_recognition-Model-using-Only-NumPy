# MNIST Image Recognition — Neural Network from Scratch

A fully connected neural network built using **only NumPy** — no TensorFlow, no PyTorch, no scikit-learn. Every component is implemented from scratch including forward propagation, backpropagation, and mini-batch gradient descent.

**Final accuracy: 98.1% on validation set, 97.9% on test set.**

---

## Project Structure

```
MNIST_ImageRecognition/
│
├── data/
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
│
├── src/
│   ├── data_loader.py      # Binary file parser, normalizer, one-hot encoder
│   ├── activations.py      # ReLU, ReLU derivative, Softmax
│   ├── loss.py             # Cross-entropy loss
│   ├── network.py          # Weight init, forward pass, backward pass, parameter update
│   └── train.py            # Mini-batch training loop, accuracy, validation
│
└── main.py                 # Entry point
```

---

## Network Architecture

```
Input Layer       784 neurons   (28×28 pixels, flattened)
Hidden Layer 1    128 neurons   (ReLU activation)
Hidden Layer 2     64 neurons   (ReLU activation)
Output Layer       10 neurons   (Softmax activation — one per digit 0–9)
```

---

## Implementation Details

### Data Pipeline (`data_loader.py`)
- Parses raw MNIST `.idx-ubyte` binary files using Python's `struct` module
- Normalizes pixel values from `uint8 [0, 255]` to `float64 [0, 1]`
- Splits 60,000 training images into 48,000 train / 12,000 validation
- One-hot encodes integer labels into `(n, 10)` matrices

### Activations (`activations.py`)
- **ReLU**: `max(0, z)` — used in hidden layers to avoid vanishing gradients
- **Softmax**: numerically stable implementation with max subtraction trick to prevent overflow
- **ReLU derivative**: returns binary mask `(z > 0)` used in backpropagation

### Loss (`loss.py`)
- Cross-entropy loss with epsilon clipping (`1e-8`) to prevent `log(0)` overflow

### Forward Pass (`network.py`)
```
Z1 = W1 @ X + B1  →  A1 = ReLU(Z1)
Z2 = W2 @ A1 + B2 →  A2 = ReLU(Z2)
Z3 = W3 @ A2 + B3 →  A3 = Softmax(Z3)
```

### Backpropagation (`network.py`)
Gradients computed via chain rule, no loops:
```
dZ3 = A3 - Y                          (softmax + cross-entropy simplification)
dW3 = (1/n) * dZ3 @ A2.T
dZ2 = (W3.T @ dZ3) * ReLU'(Z2)
dW2 = (1/n) * dZ2 @ A1.T
dZ1 = (W2.T @ dZ2) * ReLU'(Z1)
dW1 = (1/n) * dZ1 @ X
```

### Training (`train.py`)
- Mini-batch gradient descent with batch size 32
- Data shuffled every epoch to improve gradient estimates
- Validation accuracy tracked every 10 epochs

---

## Hyperparameters

| Parameter     | Value  |
|---------------|--------|
| Learning rate | 0.1    |
| Batch size    | 32     |
| Epochs        | 100    |
| Hidden layer 1| 128    |
| Hidden layer 2| 64     |
| Weight init   | `randn * 0.1` |

---

## Results

| Epoch | Train Acc | Val Acc | Test Acc |
|-------|-----------|---------|----------|
| 10    | 99.31%    | 97.54%  | 97.52%   |
| 30    | 100.00%   | 98.11%  | 97.85%   |
| 100   | 100.00%   | 98.05%  | 97.84%   |

---

## How to Run

**1. Download MNIST data** into the `data/` folder from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

**2. Install dependency**
```bash
pip install numpy
```

**3. Run training**
```bash
cd src
python train.py
```

---

## Key Concepts Implemented

- **Vanishing gradient problem** — why ReLU outperforms Sigmoid in deep networks
- **Numerical stability** — softmax max-subtraction trick, log-epsilon clipping
- **Vectorized operations** — all computations use matrix operations, no Python loops in forward/backward pass
- **Mini-batch gradient descent** — faster convergence than full-batch, better generalization than single-sample
- **Overfitting detection** — train vs validation accuracy gap monitoring
