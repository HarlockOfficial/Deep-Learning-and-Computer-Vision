import tensorflow as tf


class RecurrentNetwork(tf.keras.models.Model):
    def __init__(self, n_embeddings:int, n_label=1, activation='relu', output_activation='linear', use_bias=True, **kwargs):
        super(RecurrentNetwork, self).__init__()
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(input_dim=n_embeddings, output_dim=n_label))
        model.add(tf.keras.layers.LSTM(units=1, return_sequences=True, activation=activation))
        self.model = model

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)
