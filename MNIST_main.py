import numpy as np

from src.data_loader import load_data
from src.train import predict , load_model

_, _, _, _, x_test, y_test = load_data()

W1, B1, W2, B2, W3, B3 = load_model()

Pred = predict(W1, B1, W2, B2, W3, B3 , x_test)

print(Pred[:10])
print(y_test[:10])

