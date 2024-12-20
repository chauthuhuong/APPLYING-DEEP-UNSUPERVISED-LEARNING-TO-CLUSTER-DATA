import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from tensorflow.keras.datasets import mnist

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    return x_train, y_train

def normalize_data(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def construct_graph(features, k=10):
    adjacency = kneighbors_graph(features, n_neighbors=k, mode='connectivity', include_self=True)
    return adjacency
