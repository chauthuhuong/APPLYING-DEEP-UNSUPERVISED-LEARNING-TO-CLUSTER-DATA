import numpy as np
from tensorflow.keras.datasets import mnist

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))  #Gộp dữ liệu train và test
    y = np.concatenate((y_train, y_test))
    x = x.astype('float32') / 255.0  # Chuẩn hóa về [0, 1]
    x = x.reshape([-1, 28, 28, 1])  #ConvNet
    #x = x.reshape([-1, 784])  #FC
    print(f'MNIST loaded: {x.shape[0]} samples')
    return x, y

def load_mnist_test():
    _, (x_test, y_test) = mnist.load_data()
    x = x_test.astype('float32') / 255.0  
    x = x.reshape([-1, 28, 28, 1])  
    print(f'MNIST test set loaded: {x.shape[0]} samples')
    return x, y_test


