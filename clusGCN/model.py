import tensorflow as tf
from tensorflow.keras import layers, Model

class GCN(Model):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5):
        super(GCN, self).__init__()
        self.layers_list = []
        self.batch_norms = []

        for dim in hidden_dims:
            self.layers_list.append(layers.Dense(dim, activation=None))
            self.batch_norms.append(layers.BatchNormalization())
        self.fc_out = layers.Dense(output_dim)

        self.dropout_rate = dropout_rate
        self.activation = tf.keras.activations.elu  
    def call(self, x, training=False):
        residual = x
        for i, layer in enumerate(self.layers_list):
            x = layer(x)
            x = self.batch_norms[i](x, training=training)
            x = self.activation(x)
            x = layers.Dropout(self.dropout_rate)(x, training=training)
            if i % 2 == 1: 
                residual = tf.keras.layers.Dense(x.shape[-1])(residual)
                x = x + residual  
        return self.fc_out(x)
