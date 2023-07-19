import tensorflow as tf


class RecurrentNetwork(tf.keras.models.Model):
    def __init__(self, n_embeddings: int, n_label=1, activation='relu', output_activation='linear', use_bias=True, **kwargs):
        super(RecurrentNetwork, self).__init__()
        model = tf.keras.Sequential()
        forward_layer = tf.keras.layers.LSTM(units=n_embeddings, return_sequences=False, activation=activation, kernel_initializer=tf.keras.initializers.Zeros())
        backward_layer = tf.keras.layers.LSTM(units=n_embeddings, return_sequences=False, activation=activation, kernel_initializer=tf.keras.initializers.Zeros(), go_backwards=True)
        model.add(tf.keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer))
        model.add(tf.keras.layers.Dense(n_label, use_bias=use_bias))
        model.add(tf.keras.layers.Activation(output_activation))
        self.model = model

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)
