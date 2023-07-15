import os

import tensorflow as tf
import numpy as np
from neural_network import RecurrentNetwork
from spektral.models.gcn import GCN
from neural_network import FeedForwardNetwork



def train_network(model: tf.keras.models.Model, model_name: str, x_train, y_train=None):
    if os.path.exists(f'data/models/{model_name}/'):
        with open(f'data/models/{model_name}/model.json', 'r') as f:
            model = tf.keras.models.model_from_json(f.read(), custom_objects={'RecurrentNetwork':RecurrentNetwork, 'GCN':GCN, 'FeedForwardNetwork':FeedForwardNetwork})

        weights = np.load(f'data/models/{model_name}/weight.json.npy', allow_pickle=True)

        if model_name == 'feed_forward_network':
            model.set_weights(weights)
            model.build(input_shape=(None, 269))
        elif model_name == 'recurrent_network':
            model.set_weights(weights)
            model.build(input_shape=(None, 22))
        else:
            from spektral.data.loaders import SingleLoader
            loader = SingleLoader(x_train, epochs=1)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            model.fit(loader.load())
            model.set_weights(weights)

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    if y_train is None:
        from spektral.data.loaders import SingleLoader
        loader = SingleLoader(x_train)
        model.fit(loader.load(), steps_per_epoch=10, epochs=10, batch_size=x_train.size(), use_multiprocessing=True, verbose=1) #change epochs to around 7000
    else:
        model.fit(x=x_train, y=y_train, epochs=10, batch_size=len(x_train), use_multiprocessing=True, verbose=1)

    if not os.path.exists(f'data/models/{model_name}'):
        os.makedirs(f'data/models/{model_name}')

    with open(f'data/models/{model_name}/model.json', 'w') as f:
        f.write(model.to_json())

    np.save(f'data/models/{model_name}/weight.json', model.get_weights(), allow_pickle=True)

    return model


def test_network(model: tf.keras.models.Model, model_name: str, x_test, y_test):
    model.load_weights(f'data/models/{model_name}/weights.tf')
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
