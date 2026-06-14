import matplotlib.pyplot as plt
from src.data_loader import load_data
from src.train import predict , load_model

_, _, _, _, x_test, y_test = load_data()

W1, B1, W2, B2, W3, B3 = load_model()

Pred = predict(W1, B1, W2, B2, W3, B3 , x_test)

def show_image(image, predicted, true_label):
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted} | True: {true_label}")
    plt.axis('off')
    plt.show()

for i in range(5):
    i = 2*i + 32
    show_image(x_test[i], Pred[i], y_test[i])

# print("Prediction of My Model on test set (First 10 images):",Pred[:10])
# print("True Labels of images on test set : ",y_test[:10])
# print("Accuracy on test set : ", np.mean(Pred == y_test) * 100)

