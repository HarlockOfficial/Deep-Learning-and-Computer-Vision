import tensorflow as tf

from dgl.nn.tensorflow import GraphConv


class GraphConvolutionalNetwork(tf.keras.models.Model):
    def __init__(self, n_label=1, activation='relu', output_activation='linear', use_bias=True, **kwargs):
        super(GraphConvolutionalNetwork, self).__init__()
        reg = tf.keras.regularizers.l2(2.5e-4)
        self.__dropout_0 = tf.keras.layers.Dropout(rate=0.5)
        self.__convolution_0 = GraphConv(activation=activation, bias=use_bias)
        self.__dropout_1 = tf.keras.layers.Dropout(rate=0.5)
        self.__convolution_1 = GraphConv(out_feats=n_label, activation=output_activation, bias=use_bias)

    def call(self, inputs, training=None, mask=None):
        x, a = inputs
        x = self.__dropout_0(x)
        x = self.__convolution_0([x, a])
        x = self.__dropout_1(x)
        return self.__convolution_1([x, a])
