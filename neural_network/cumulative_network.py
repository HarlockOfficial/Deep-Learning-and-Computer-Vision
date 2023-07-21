import os

import tensorflow as tf

from dotenv import load_dotenv
load_dotenv()

from spektral.layers import GCNConv, GlobalSumPool


class OurNetwork(tf.keras.models.Model):
    def __init__(self, contact_matrix_shape, n_label=1, n_embeddings=os.getenv('MAX_INPUT'), n_channels=16, activation='relu', output_activation='linear', use_bias=True, l2_reg=2.5e-4, dropout_rate=0.5, **kwargs):
        super(OurNetwork, self).__init__()

        # regularizer
        reg = tf.keras.regularizers.l2(l2_reg)

        # Input layers
        ## input physio-chemical features
        input_physio_chemical_features = tf.keras.Input(shape=(15,))
        ## input aminoacid chain with positions
        input_aminoacid_chain = tf.keras.Input(shape=(20,))
        ## input contact matrix
        ## (not put shape, since the matrix is a square matrix and
        ## depends on the number of residues)
        input_contact_matrix = tf.keras.Input(shape=(contact_matrix_shape, ))

        # Input dense layers
        first_layer_chemical_features = tf.keras.layers.Dense(256, activation=activation, use_bias=use_bias, kernel_initializer=tf.keras.initializers.Zeros())
        first_layer_amino_acid = tf.keras.layers.Dense(256, activation=activation, use_bias=use_bias, kernel_initializer=tf.keras.initializers.Zeros())
        first_layer_contact_matrix = tf.keras.layers.Dense(256, activation=activation, use_bias=use_bias, kernel_initializer=tf.keras.initializers.Zeros())

        # Input layers edges
        chemical_features_edge = first_layer_chemical_features(input_physio_chemical_features)
        amino_acid_edge = first_layer_amino_acid(input_aminoacid_chain)
        contact_matrix_edge = first_layer_contact_matrix(input_contact_matrix)

        # x is now input for the GCN
        x_gcn = tf.keras.layers.concatenate([amino_acid_edge, contact_matrix_edge])

        # GCN
        x_gcn = tf.keras.layers.Dropout(rate=dropout_rate)(x_gcn)
        # here blows up, because gcn conv expects a n input dimension, GCNConv class, line 79
        x_gcn = GCNConv(n_channels, activation=activation, kernel_regularizer=reg, use_bias=use_bias)(x_gcn)
        x_gcn = GlobalSumPool()(x_gcn)
        x_gcn = tf.keras.layers.Dropout(rate=dropout_rate)(x_gcn)
        gcn_out = tf.keras.layers.Dense(n_label, activation=output_activation, use_bias=use_bias, kernel_initializer=tf.keras.initializers.Zeros(), kernel_regularizer=reg)(x_gcn)

        # BiRNN
        forward_layer = tf.keras.layers.LSTM(units=n_embeddings, return_sequences=False, activation=activation,
                                             kernel_initializer=tf.keras.initializers.Zeros(), kernel_regularizer=reg)
        backward_layer = tf.keras.layers.LSTM(units=n_embeddings, return_sequences=False, activation=activation,
                                              kernel_initializer=tf.keras.initializers.Zeros(), kernel_regularizer=reg, go_backwards=True)
        x_BiRNN = tf.keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer)(amino_acid_edge)
        BiRNN_out = tf.keras.layers.Dense(n_label, activation=output_activation, use_bias=use_bias, kernel_initializer=tf.keras.initializers.Zeros(), kernel_regularizer=reg)(x_BiRNN)

        # Concatenate
        x = tf.keras.layers.concatenate([gcn_out, BiRNN_out, chemical_features_edge])

        # fully connected layers
        x = tf.keras.layers.Dense(256, activation=activation, use_bias=use_bias,kernel_initializer=tf.keras.initializers.Zeros())(x)
        x = tf.keras.layers.Dense(256, activation=activation, use_bias=use_bias, kernel_initializer=tf.keras.initializers.Zeros())(x)
        x = tf.keras.layers.Dense(256, activation=activation, use_bias=use_bias,kernel_initializer=tf.keras.initializers.Zeros())(x)

        output = tf.keras.layers.Dense(n_label, activation=output_activation, use_bias=use_bias, kernel_initializer=tf.keras.initializers.Zeros())(x)

        self.__model = tf.keras.Model(inputs=[input_aminoacid_chain, input_contact_matrix, input_physio_chemical_features], outputs=output)

    def call(self, inputs, training=None, mask=None):
        return self.__model(inputs)
