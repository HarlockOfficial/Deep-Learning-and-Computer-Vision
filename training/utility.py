import os

import tensorflow as tf


def train_network(model: type[tf.keras.models.Model], model_name: str, n_output_labels, x_train, y_train):
    m = model(n_label=n_output_labels, activation='relu', output_activation='linear', use_bias=True)
    if os.path.exists(f'data/models/{model_name}/weights.h5'):
        m.load_weights(f'data/models/{model_name}/weights.h5')
    m.compile(optimizer='adam', loss='mse', metrics=['mae'])
    m.fit(x_train, y_train, epochs=100, batch_size=64, use_multiprocessing=True, verbose=1)
    if not os.path.exists(f'data/models/{model_name}'):
        os.makedirs(f'data/models/{model_name}')
    m.save_weights(f'data/models/{model_name}/weights.h5')
    m.save(f'data/models/{model_name}/model.h5')
    return m


def test_network(model: type[tf.keras.models.Model], model_name: str, n_output_labels, x_test, y_test):
    model = model(n_label=n_output_labels, activation='relu', output_activation='linear', use_bias=True)
    model.load_weights(f'data/models/{model_name}/weights.h5')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model.evaluate(x_test, y_test, batch_size=64, use_multiprocessing=True, verbose=1)
