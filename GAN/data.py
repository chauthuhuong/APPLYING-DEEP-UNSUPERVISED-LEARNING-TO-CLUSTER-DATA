import numpy as np
from tensorflow.keras.datasets import mnist

def load_mnist_data():
    """
    Load and preprocess MNIST dataset.
    """
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)  # Reshape to include channel
    return x_train, y_train
