import numpy as np
from tensorflow.keras.datasets import mnist
from app.models.nn_from_scratch import NeuralNetwork

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten and normalize
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

def one_hot(label, num_classes=10):
    vec = np.zeros(num_classes)
    vec[label] = 1.0
    return vec

nn = NeuralNetwork(
    input_size = 784,
    hidden_size = 100,
    output_size = 10,
    learning_rate = 0.1
)

epochs = 5

for epoch in range(epochs):
    for x, y in zip(x_train, y_train):
        nn.backward(x, one_hot(y))
    print(f"Epock {epoch+1}/{epochs} complete")


correct = 0

for x, y in zip(x_test, y_test):
    pred = nn.predict(x)
    if pred == y:
        correct += 1

accuracy = correct / len(x_test)
print("Test accuracy:", accuracy)