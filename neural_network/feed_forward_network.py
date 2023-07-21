import tensorflow as tf
import training.utility as utility

class FeedForwardNetwork(tf.keras.models.Model):
    def __init__(self, n_label=1, activation='relu', output_activation='linear', use_bias=True, **kwargs):
        super(FeedForwardNetwork, self).__init__()

        input_layer_rnn = tf.keras.Input(shape=(1,))
        input_layer_gnn = tf.keras.Input(shape=(1,))
        input_layer_chemical = tf.keras.Input(shape=(15,))
        first_layer_rnn = tf.keras.layers.Dense(256, activation=activation, use_bias=use_bias, kernel_initializer=tf.keras.initializers.Zeros())
        first_layer_gnn = tf.keras.layers.Dense(256, activation=activation, use_bias=use_bias, kernel_initializer=tf.keras.initializers.Zeros())
        first_layer_chemical = tf.keras.layers.Dense(256, activation=activation, use_bias=use_bias, kernel_initializer=tf.keras.initializers.Zeros())
        rnn_edge = first_layer_rnn(input_layer_rnn)
        gnn_edge = first_layer_gnn(input_layer_gnn)
        chemical_edge = first_layer_chemical(input_layer_chemical)
        x = tf.keras.layers.concatenate([rnn_edge, gnn_edge, chemical_edge])
        x = tf.keras.layers.Dense(256, activation=activation, use_bias=use_bias,kernel_initializer=tf.keras.initializers.Zeros())(x)
        x = tf.keras.layers.Dense(256, activation=activation, use_bias=use_bias, kernel_initializer=tf.keras.initializers.Zeros())(x)
        x = tf.keras.layers.Dense(256, activation=activation, use_bias=use_bias,kernel_initializer=tf.keras.initializers.Zeros())(x)
        output = tf.keras.layers.Dense(n_label, activation=output_activation, use_bias=use_bias, kernel_initializer=tf.keras.initializers.Zeros())(x)
        self.__model = tf.keras.Model(inputs=[input_layer_rnn, input_layer_gnn, input_layer_chemical], outputs=output)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        utility.logger.debug("x: {}".format(x))
        return self.__model(x)
