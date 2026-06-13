import numpy as np

from src.data_loader import load_data
from src.train import predict , load_model ,accuracy

_, _, _, _, x_test, y_test = load_data()

W1, B1, W2, B2, W3, B3 = load_model()

Pred = predict(W1, B1, W2, B2, W3, B3 , x_test)

print("Prediction of My Model on test set (First 10 images): ",Pred[:10])
print("True Labels of images on test set : ",y_test[:10])
print("Accuracy on test set : ", np.mean(Pred == y_test) * 100)

