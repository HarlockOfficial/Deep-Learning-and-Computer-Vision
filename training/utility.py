import os

import keras.models
import spektral.layers
from dotenv import load_dotenv

import neural_network.cumulative_network

load_dotenv()

import tensorflow as tf
import preprocessing.utility as utility

logger = utility.default_logger(__file__)


def f1_m(y_true, y_pred):
    logger.debug('true: ' + str(y_true))
    logger.debug('pred: ' + str(y_pred))

    max_value = tf.math.reduce_max(y_pred)
    max_value_true = tf.math.reduce_max(y_true)

    logger.debug('max-value-predicted: ' + str(max_value))
    logger.debug('max-value-true: ' + str(max_value_true))

    y_pred = tf.math.greater(y_pred, tf.constant([0.5], dtype=tf.float32))
    logger.debug('thresholded-pred: ' + str(y_pred))

    precision_metric = tf.keras.metrics.Precision()
    precision_metric.update_state(y_true, y_pred)
    precision = precision_metric.result().numpy()
    logger.debug("Precision: " + str(precision))
    recall_metric = tf.keras.metrics.Recall()
    recall_metric.update_state(y_true, y_pred)
    recall = recall_metric.result().numpy()
    logger.debug("Recall: " + str(recall))
    return 2*((precision*recall)/(precision+recall+tf.keras.backend.epsilon()))


def train_network(model: tf.keras.models.Model, model_name: str, x_train, y_train=None, validation_data=None, batch_size=None):
    if batch_size is None:
        batch_size = len(x_train)
    logger.info("Does save path exist?")
    if os.path.exists('C:\\Users\\Filippo\\Projects\\Deep-Learning-and-Computer-Vision\\data\\models'):
        if os.path.isfile('C:\\Users\\Filippo\\Projects\\Deep-Learning-and-Computer-Vision\\data\\models\\our_network.index'):
            logger.info('Loading model from file')
            model.build(input_shape=[tf.shape(x_train[0]), tf.shape(x_train[1]), tf.shape(x_train[2])])
            model = model.load_weights('C:\\Users\\Filippo\\Projects\\Deep-Learning-and-Computer-Vision\\data\\models\\our_network')

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_m], run_eagerly=True )

    # https://keras.io/guides/serialization_and_saving/

    logger.debug("Model name: " + str(model_name))
    if y_train is None:
        from spektral.data.loaders import SingleLoader
        loader = SingleLoader(x_train)
        loader_validation = SingleLoader(validation_data, epochs=10)
        batch_size = x_train.size()
        result = model.fit(loader.load(), steps_per_epoch=10, epochs=10, batch_size=batch_size, use_multiprocessing=True,
                  verbose=1, validation_data=loader_validation.load(), validation_steps=10)
    else:
        logger.info("Batch size: " + str(batch_size))
        #checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'data/models/{model_name}/weights.ckpt', verbose=1, save_weights_only=True)
        result = model.fit(x=x_train, y=y_train, epochs=1, batch_size=batch_size, use_multiprocessing=True, verbose=1, steps_per_epoch=1) #TODO back to 10 epochs

    logger.debug("Model summary: ")
    summary_string = []
    model.summary(print_fn=lambda x: summary_string.append(x))
    logger.debug('\n'.join(summary_string))

    if result:
        logger.debug("History model name: " + str(model_name) + " History: " + str(result.history))

    model.save_weights('C:\\Users\\Filippo\\Projects\\Deep-Learning-and-Computer-Vision\\data\\models\\our_network')

    return model, result

# https://github.com/tensorflow/tensorflow/issues/29545#issuecomment-520062408

def test_network(model: tf.keras.models.Model, model_name: str, x_test, y_test=None):
    model.load_weights(f'data/models/{model_name}/weights.tf')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_m], run_eagerly=True)

    if y_test is None:
        from spektral.data.loaders import SingleLoader
        loader = SingleLoader(x_test, epochs=10)
        result = model.evaluate(loader.load(), steps=10, batch_size=x_test.size(),
                                use_multiprocessing=True, verbose=1)  # change epochs to around 7000
    else:
        result = model.evaluate(x=x_test, y=y_test, steps=10, batch_size=len(x_test),
                                use_multiprocessing=True, verbose=1)

    if result:
        logger.debug("History model name: " + str(model_name) + " History: " + str(result))

    return model, result
