from neural_network import FeedForwardNetwork
from training import utility


def train_graph_attention_network(n_output_labels, x_train, y_train):
    return utility.train_network(FeedForwardNetwork, n_output_labels, x_train, y_train)


def test_graph_attention_network(n_output_labels, x_test, y_test):
    return utility.test_network(FeedForwardNetwork, n_output_labels, x_test, y_test)