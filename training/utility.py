import os
import numpy as np
import tensorflow as tf


def train_network(model: type[tf.keras.models.Model], n_output_labels, x_train, y_train):
    model = model(n_label=n_output_labels, activation='relu', output_activation='linear', use_bias=True)
    if os.path.exists('data/models/graph_convolutional_network/weights.h5'):
        model.load_weights('data/models/graph_convolutional_network/weights.h5')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(x_train, y_train, epochs=100, batch_size=64, use_multiprocessing=True, verbose=1)
    if not os.path.exists('data/models/graph_convolutional_network'):
        os.makedirs('data/models/graph_convolutional_network')
    model.save_weights('data/models/graph_convolutional_network/weights.h5')
    model.save('data/models/graph_convolutional_network/model.h5')
    return model


def test_network(model: type[tf.keras.models.Model], n_output_labels, x_test, y_test):
    model = model(n_label=n_output_labels, activation='relu', output_activation='linear', use_bias=True)
    model.load_weights('data/models/graph_convolutional_network/weights.h5')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model.evaluate(x_test, y_test, batch_size=64, use_multiprocessing=True, verbose=1)
