import os

import tensorflow as tf


def train_network(model: tf.keras.models.Model, model_name: str, x_train, y_train):
    if os.path.exists(f'data/models/{model_name}/weights.tf'):
        model.load_weights(f'data/models/{model_name}/weights.tf', save_format='tf')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(x_train, y_train, epochs=100, batch_size=232, use_multiprocessing=True, verbose=1)
    if not os.path.exists(f'data/models/{model_name}'):
        os.makedirs(f'data/models/{model_name}')
    model.save_weights(f'data/models/{model_name}/weights.tf', save_format='tf')
    model.save(f'data/models/{model_name}/model.tf', save_format='tf')
    return model


def test_network(model: tf.keras.models.Model, model_name: str, x_test, y_test):
    model.load_weights(f'data/models/{model_name}/weights.tf', save_format='tf')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model.evaluate(x_test, y_test, batch_size=64, use_multiprocessing=True, verbose=1)
