import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, Cropping2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ClusteringLayer(tf.keras.layers.Layer):
    """
    Tính phân phối xác suất cụm dựa trên khoảng cách giữa
    điểm nhúng và các trọng số cụm (cluster centers).
    """
    def __init__(self, n_clusters, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters

    def build(self, input_shape):
        # Tạo trọng số cụm (cluster centers)
        self.input_dim = input_shape[-1]
        self.clusters = self.add_weight(
            shape=(self.n_clusters, self.input_dim),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs, **kwargs):
        # Tính phân phối xác suất cụm q_ij
        q = 1.0 / (1.0 + tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.clusters), axis=2))
        q = q / tf.reduce_sum(q, axis=1, keepdims=True)  # Chuẩn hóa thành xác suất
        return q


def build_fc_autoencoder(input_dim=784, latent_dim=10):
    #Lớp en
    inputs = Input(shape=(input_dim,), name="input")
    x = Dense(500, activation="relu")(inputs)
    x = Dense(500, activation="relu")(x)
    x = Dense(2000, activation="relu")(x)
    embedding = Dense(latent_dim, name="embedding")(x)

    #Lớp de
    x = Dense(2000, activation="relu")(embedding)
    x = Dense(500, activation="relu")(x)
    x = Dense(500, activation="relu")(x)
    outputs = Dense(input_dim, activation="sigmoid")(x)

    autoencoder = Model(inputs, outputs, name="Autoencoder")
    encoder = Model(inputs, embedding, name="Encoder")
    return autoencoder, encoder

def build_conv_autoencoder(input_shape=(28, 28, 1), latent_dim=10):
    inputs = Input(shape=input_shape, name="input")
    x = Conv2D(32, kernel_size=5, strides=2, activation="relu", padding="same")(inputs)  
    x = Conv2D(64, kernel_size=5, strides=2, activation="relu", padding="same")(x)   
    x = Conv2D(128, kernel_size=3, strides=2, activation="relu", padding="same")(x)     
    x = Flatten()(x)
    embedding = Dense(latent_dim, name="embedding")(x)  

    x = Dense(4 * 4 * 128, activation="relu")(embedding)  
    x = Reshape((4, 4, 128))(x)
    x = Conv2DTranspose(64, kernel_size=3, strides=2, activation="relu", padding="same")(x)  
    x = Conv2DTranspose(32, kernel_size=3, strides=2, activation="relu", padding="same")(x)  
    x = Conv2DTranspose(1, kernel_size=3, strides=2, activation="sigmoid", padding="same")(x)  
    outputs = Cropping2D(cropping=((2, 2), (2, 2)))(x)  

    autoencoder = Model(inputs, outputs, name="Autoencoder")
    encoder = Model(inputs, embedding, name="Encoder")
    return autoencoder, encoder


def build_dec_model(autoencoder, encoder, n_clusters):
    #kết hợp CL 
    clustering_layer = ClusteringLayer(n_clusters, name="clustering")(encoder.output)
    model = Model(inputs=encoder.input, outputs=clustering_layer)
    return model


def get_data_augmenter():
    return ImageDataGenerator(
        rotation_range=10,           # Xoay ảnh tối đa 10 độ
        width_shift_range=0.1,       # Dịch ngang tối đa 10%
        height_shift_range=0.1,      # Dịch dọc tối đa 10%
        zoom_range=0.1,              # Phóng to/thu nhỏ tối đa 10%
    )
