import os

import tensorflow as tf
import numpy as np
import preprocessing.utility as utility

logger = utility.default_logger(__file__)


def f1_m(y_true, y_pred):
    logger.debug('true: ' + y_true)
    logger.debug('pred: ' + y_pred)

    max_value = tf.math.reduce_max(y_pred)
    max_value_true = tf.math.reduce_max(y_true)

    logger.debug('max-value-predicted: ' + max_value)
    logger.debug('max-value-true: ' + str(max_value_true))

    y_pred = tf.math.greater(y_pred, tf.constant([0.5], dtype=tf.float32))
    logger.debug('thresholded-pred: ' + y_pred)

    precision_metric = tf.keras.metrics.Precision()
    precision_metric.update_state(y_true, y_pred)
    precision = precision_metric.result().numpy()
    logger.debug("Precision: " + str(precision))
    recall_metric = tf.keras.metrics.Recall()
    recall_metric.update_state(y_true, y_pred)
    recall = recall_metric.result().numpy()
    logger.debug("Recall: " + str(recall))
    return 2*((precision*recall)/(precision+recall+tf.keras.backend.epsilon()))


def train_network(model: tf.keras.models.Model, model_name: str, x_train, y_train=None):
    if os.path.exists(f'data/models/{model_name}/weights.tf'):
        model.load_weights(f'data/models/{model_name}/weights.tf', save_format='tf')
    model.compile(optimizer='adam', loss='mse', metrics=[f1_m], run_eagerly=True )

    if y_train is None:
        from spektral.data.loaders import SingleLoader
        loader = SingleLoader(x_train)
        #model.fit(loader.load(), steps_per_epoch=100, epochs=100, batch_size=x_train.size(), use_multiprocessing=True, verbose=1) #change epochs to around 7000
        model.fit(loader.load(), steps_per_epoch=10, epochs=10, batch_size=x_train.size(), use_multiprocessing=True,
                  verbose=1)
    else:
        model.fit(x=x_train, y=y_train, epochs=100, batch_size=len(x_train), use_multiprocessing=True, verbose=1)

    if not os.path.exists(f'data/models/{model_name}'):
        os.makedirs(f'data/models/{model_name}')
    model.save_weights(f'data/models/{model_name}/weights.tf', save_format='tf')
    return model

    """
    if os.path.exists(f'data/models/{model_name}/'):
        #with open(f'data/models/{model_name}/model.json', 'r') as f:
        #    model = tf.keras.models.model_from_json(f.read(), custom_objects={'RecurrentNetwork':RecurrentNetwork, 'GCN':GCN, 'FeedForwardNetwork':FeedForwardNetwork})

        weights = np.load(f'data/models/{model_name}/weight.json.npy', allow_pickle=True)

        if model_name == 'feed_forward_network':
            model.build(input_shape=(None, 1037))
            model.set_weights(weights)
            model.build(input_shape=(None, 1037))
        elif model_name == 'recurrent_network':
            model.build(input_shape=(None, 22))
            model.set_weights(weights)
            model.build(input_shape=(None, 22))
        else:
            from spektral.data.loaders import SingleLoader
            loader = SingleLoader(x_train, epochs=1)
            model.compile(optimizer='adam', loss='mse', metrics=[f1_m], run_eagerly=True)
            model.fit(loader.load())
            model.set_weights(weights)

    model.compile(optimizer='adam', loss='mse', metrics=[f1_m], run_eagerly=True)

    if y_train is None:
        from spektral.data.loaders import SingleLoader
        loader = SingleLoader(x_train)
        model.fit(loader.load(), steps_per_epoch=10, epochs=100, batch_size=x_train.size(), use_multiprocessing=True, verbose=1) #change epochs to around 7000
    else:
        model.fit(x=x_train, y=y_train, epochs=100, batch_size=len(x_train), use_multiprocessing=True, verbose=1)

    if not os.path.exists(f'data/models/{model_name}'):
        os.makedirs(f'data/models/{model_name}')

    with open(f'data/models/{model_name}/model.json', 'w') as f:
        f.write(model.to_json())
    print(tf.shape(model.get_weights()))
    input("---------------------------------")
    np.save(f'data/models/{model_name}/weight.json', model.get_weights(), allow_pickle=True)"""
    return model




def test_network(model: tf.keras.models.Model, model_name: str, x_test, y_test=None):

    if os.path.exists(f'data/models/{model_name}/'):

        weights = np.load(f'data/models/{model_name}/weight.json.npy', allow_pickle=True)

        if model_name == 'feed_forward_network':
            model.build(input_shape=(None, 1037))
            model.set_weights(weights)
            model.build(input_shape=(None, 1037))
            model.compile(optimizer='adam', loss='mse', metrics=[f1_m], run_eagerly=True)
        elif model_name == 'recurrent_network':
            model.build(input_shape=(None, 22))
            model.set_weights(weights)
            model.build(input_shape=(None, 22))
            model.compile(optimizer='adam', loss='mse', metrics=[f1_m], run_eagerly=True)
        else:
            from spektral.data.loaders import SingleLoader
            loader = SingleLoader(x_test, epochs=1)
            model.compile(optimizer='adam', loss='mse', metrics=[f1_m], run_eagerly=True)
            model.evaluate(loader.load())
            model.set_weights(weights)


    if y_test is None:
        from spektral.data.loaders import SingleLoader
        loader = SingleLoader(x_test, epochs=1)
        result = model.evaluate(loader.load(), steps=1, batch_size=x_test.size(),
                                use_multiprocessing=True, verbose=1)  # change epochs to around 7000
    else:
        result = model.evaluate(x=x_test, y=y_test, steps=1, batch_size=len(x_test),
                                use_multiprocessing=True, verbose=1)

    return model, result
