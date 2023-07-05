import tensorflow as tf


class FeedForwardNetwork(tf.keras.models.Model):
    def __init__(self, n_label=1, activation='relu', output_activation='linear', use_bias=True):
        super(FeedForwardNetwork, self).__init__()
        self.__layers = []
        for i in range(4):
            self.__layers.append(tf.keras.layers.Dense(2 ** (i + 3), activation=activation, use_bias=use_bias))
        self.__layers.append(tf.keras.layers.Dense(n_label, activation=output_activation, use_bias=use_bias))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.__layers:
            x = layer(x)
        return x
