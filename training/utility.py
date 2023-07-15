import os

import tensorflow as tf


def train_network(model: tf.keras.models.Model, model_name: str, x_train, y_train=None):
    if os.path.exists(f'data/models/{model_name}/weights.tf'):
        model.load_weights(f'data/models/{model_name}/weights.tf', save_format='tf')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    if y_train is None:
        from spektral.data.loaders import SingleLoader
        loader = SingleLoader(x_train)
        model.fit(loader.load(), steps_per_epoch=100, epochs=100, batch_size=x_train.size(), use_multiprocessing=True, verbose=1) #change epochs to around 7000
    else:
        model.fit(x=x_train, y=y_train, epochs=100, batch_size=len(x_train), use_multiprocessing=True, verbose=1)

    if not os.path.exists(f'data/models/{model_name}'):
        os.makedirs(f'data/models/{model_name}')
    model.save_weights(f'data/models/{model_name}/weights.tf', save_format='tf')
    return model


def test_network(model: tf.keras.models.Model, model_name: str, x_test, y_test):
    model.load_weights(f'data/models/{model_name}/weights.tf', save_format='tf')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    if y_test is None:
        from spektral.data.loaders import SingleLoader
        loader = SingleLoader(x_test)
        result = model.evaluate(loader.load(), steps_per_epoch=100, epochs=100, batch_size=x_test.size(),
                                use_multiprocessing=True, verbose=1)  # change epochs to around 7000
    else:
        result = model.evaluate(x=x_test, y=y_test, epochs=100, batch_size=len(x_test),
                                use_multiprocessing=True, verbose=1)

    return result
