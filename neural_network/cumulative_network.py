import os

import tensorflow as tf

from dotenv import load_dotenv
load_dotenv()

from spektral.layers import GCNConv


class OurNetwork(tf.keras.models.Model):
    def __init__(self, n_label=1, n_embeddings: int = int(os.getenv('MAX_INPUT')), n_channels=16, activation='relu', output_activation='linear', use_bias=True, l2_reg=2.5e-4, dropout_rate=0.5, **kwargs):
        super(OurNetwork, self).__init__()
        # regularizer
        self.__reg = tf.keras.regularizers.l2(l2_reg)
        self.__n_embeddings = n_embeddings
        self.__n_label = n_label
        self.__n_channels = n_channels
        self.__activation = activation
        self.__output_activation = output_activation
        self.__use_bias = use_bias
        self.__dropout_rate = dropout_rate
        self.__model = None

    def build(self, input_shape):
        # [tensor(row, col), tensor(row, col), tensor(row,col)]
        # Input layers
        ## input physio-chemical features
        input_physio_chemical_features = tf.keras.Input(shape=(15, ))
        ## input aminoacid chain with positions
        input_aminoacid_chain = tf.keras.Input(shape=(20, ))
        ## input contact matrix
        ## (not put shape, since the matrix is a square matrix and
        ## depends on the number of residues)
        input_contact_matrix = tf.keras.Input(shape=(input_shape[1][-1], ))

        # Input dense layers
        first_layer_chemical_features = tf.keras.layers.Dense(15, activation=self.__activation, use_bias=self.__use_bias, kernel_initializer=tf.keras.initializers.Zeros())
        first_layer_amino_acid = tf.keras.layers.Dense(20, activation=self.__activation, use_bias=self.__use_bias, kernel_initializer=tf.keras.initializers.Zeros())
        first_layer_contact_matrix = tf.keras.layers.Dense(input_shape[1][-1], activation=self.__activation, use_bias=self.__use_bias, kernel_initializer=tf.keras.initializers.Zeros())

        # Input layers edges
        chemical_features_edge = first_layer_chemical_features(input_physio_chemical_features)
        amino_acid_edge = first_layer_amino_acid(input_aminoacid_chain)
        contact_matrix_edge = first_layer_contact_matrix(input_contact_matrix)

        # x is now input for the GCN
        #x_gcn = tf.keras.layers.concatenate([amino_acid_edge, contact_matrix_edge])

        # GCN
        # x_gcn = tf.keras.layers.Dropout(rate=self.__dropout_rate)(x_gcn)
        # here blows up, because gcn conv expects a n input dimension, GCNConv class, line 79
        x_gcn = GCNConv(self.__n_channels, activation=self.__activation, kernel_regularizer=self.__reg,
                        use_bias=self.__use_bias)([amino_acid_edge, contact_matrix_edge])
        x_gcn = tf.keras.layers.Dropout(rate=self.__dropout_rate)(x_gcn)
        x_gcn = GCNConv(self.__n_channels, activation=self.__activation, kernel_regularizer=self.__reg,
                        use_bias=self.__use_bias)([x_gcn, contact_matrix_edge])
        x_gcn = tf.keras.layers.Dropout(rate=self.__dropout_rate)(x_gcn)
        gcn_out = tf.keras.layers.Dense(self.__n_label, activation=self.__output_activation, use_bias=self.__use_bias, kernel_initializer=tf.keras.initializers.Zeros(), kernel_regularizer=self.__reg)(x_gcn)

        # BiRNN
        forward_layer = tf.keras.layers.LSTM(units=self.__n_embeddings, return_sequences=False, activation=self.__activation,
                                             kernel_initializer=tf.keras.initializers.Zeros(), input_shape=(20, 1))
        tmp_layer = tf.keras.layers.Bidirectional(forward_layer, input_shape=(20,1))
        reshape = tf.keras.layers.Reshape((20, 1))(amino_acid_edge)
        x_BiRNN = tmp_layer(reshape)
        x_BiRNN = tf.keras.layers.Dense(self.__n_label, use_bias=self.__use_bias)(x_BiRNN)
        BiRNN_out = tf.keras.layers.Activation(activation=self.__output_activation)(x_BiRNN)

        # Concatenate
        x = tf.keras.layers.concatenate([gcn_out, BiRNN_out, input_physio_chemical_features])

        # fully connected layers
        x = tf.keras.layers.Dense(256, activation=self.__activation, use_bias=self.__use_bias,kernel_initializer=tf.keras.initializers.Zeros())(x)
        x = tf.keras.layers.Dense(256, activation=self.__activation, use_bias=self.__use_bias, kernel_initializer=tf.keras.initializers.Zeros())(x)
        x = tf.keras.layers.Dense(256, activation=self.__activation, use_bias=self.__use_bias,kernel_initializer=tf.keras.initializers.Zeros())(x)

        output = tf.keras.layers.Dense(self.__n_label, activation=self.__output_activation, use_bias=self.__use_bias, kernel_initializer=tf.keras.initializers.Zeros())(x)

        self.__model = tf.keras.Model(inputs=[input_aminoacid_chain, input_contact_matrix, input_physio_chemical_features], outputs=output)

    def call(self, inputs, training=None, mask=None):
        return self.__model(inputs)
