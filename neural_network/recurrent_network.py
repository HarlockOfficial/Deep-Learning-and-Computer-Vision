import tensorflow as tf


class RecurrentNetwork(tf.keras.models.Model):
    def __init__(self, n_label=1, activation='relu', output_activation='linear', use_bias=True, n_input_features=2):
        super(RecurrentNetwork, self).__init__()
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(input_dim=n_input_features, output_dim=n_label))
        model.add(tf.keras.layers.LSTM(units=128, return_sequences=True, activation=activation))
        model.add(tf.keras.layers.Dense(units=n_label, activation=output_activation, use_bias=use_bias))
        self.model = model

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)
