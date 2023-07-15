from spektral.layers import GATConv
import tensorflow as tf


class GraphAttentionNetwork(tf.keras.models.Model):
    def __init__(self, n_label=1, activation='relu', output_activation='linear', use_bias=True, **kwargs):
        super(GraphAttentionNetwork, self).__init__()
        self.__layers = []
        for i in range(4):
            self.__layers.append(GATConv(2 ** (i + 3), activation=activation, kernel_initializer=tf.keras.initializers.Zeros()))
        self.__layers.append(tf.keras.layers.Flatten())
        self.__layers.append(tf.keras.layers.Dense(n_label, activation=output_activation, use_bias=use_bias, kernel_initializer=tf.keras.initializers.Zeros()))

    def call(self, inputs, training=None, mask=None):
        x, a = inputs
        for layer in self.__layers:
            x = layer([x, a])
        return x
