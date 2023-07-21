import os

from dotenv import load_dotenv
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


def train_network(model: tf.keras.models.Model, model_name: str, x_train, y_train=None, validation_data=None):
    if os.path.exists(f'data/models/{model_name}/'):
        logger.info('Loading model from file')
        model.load_weights(f'data/models/{model_name}/weights.tf')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_m], run_eagerly=True )

    logger.debug("Model name: " + str(model_name))

    if y_train is None:
        from spektral.data.loaders import SingleLoader
        loader = SingleLoader(x_train)
        loader_validation = SingleLoader(validation_data, epochs=int(os.getenv('TRAIN_STEPS')))
        result = model.fit(loader.load(), steps_per_epoch=int(os.getenv('TRAIN_STEPS')), epochs=int(os.getenv('TRAIN_STEPS')), batch_size=x_train.size(), use_multiprocessing=True,
                  verbose=1, validation_data=loader_validation.load(), validation_steps=int(os.getenv('TRAIN_STEPS')))
    else:
       result = model.fit(x=x_train, y=y_train, epochs=int(os.getenv('TRAIN_STEPS')), batch_size=len(x_train), use_multiprocessing=True, verbose=1, validation_data=validation_data, validation_steps=int(os.getenv('TRAIN_STEPS')), steps_per_epoch=int(os.getenv('TRAIN_STEPS')))

    logger.debug("Model summary: ")
    summary_string = []
    model.summary(print_fn=lambda x: summary_string.append(x))
    logger.debug('\n'.join(summary_string))

    if result:
        logger.debug("History model name: " + str(model_name) + " History: " + str(result.history))


    if not os.path.exists(f'data/models/{model_name}'):
        os.makedirs(f'data/models/{model_name}')
    model.save_weights(f'data/models/{model_name}/weights.tf', save_format='tf')
    return model, result


def test_network(model: tf.keras.models.Model, model_name: str, x_test, y_test=None):
    model.load_weights(f'data/models/{model_name}/weights.tf')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_m], run_eagerly=True)

    if y_test is None:
        from spektral.data.loaders import SingleLoader
        loader = SingleLoader(x_test, epochs=int(os.getenv('TEST_STEPS')))
        result = model.evaluate(loader.load(), steps=int(os.getenv('TEST_STEPS')), batch_size=x_test.size(),
                                use_multiprocessing=True, verbose=1)  # change epochs to around 7000
    else:
        result = model.evaluate(x=x_test, y=y_test, steps=int(os.getenv('TEST_STEPS')), batch_size=len(x_test),
                                use_multiprocessing=True, verbose=1)

    if result:
        logger.debug("History model name: " + str(model_name) + " History: " + str(result))

    return model, result
