from neural_network import FeedForwardNetwork
from training import utility


def train_feed_forward_network(n_output_labels, x_train, y_train, validation_data):
    ffn = FeedForwardNetwork(n_label=n_output_labels, activation='sigmoid', output_activation='linear', use_bias=True)
    return utility.train_network(ffn, "feed_forward_network", x_train, y_train, validation_data)


def test_feed_forward_network(n_output_labels, x_test, y_test):
    ffn = FeedForwardNetwork(n_label=n_output_labels, activation='sigmoid', output_activation='linear', use_bias=True)
    return utility.test_network(ffn, "feed_forward_network", x_test, y_test)
