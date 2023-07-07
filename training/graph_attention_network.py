from neural_network import GraphAttentionNetwork
from training import utility


def train_graph_attention_network(n_output_labels, x_train, y_train):
    return utility.train_network(GraphAttentionNetwork, n_output_labels, x_train, y_train)


def test_graph_attention_network(n_output_labels, x_test, y_test):
    return utility.test_network(GraphAttentionNetwork, n_output_labels, x_test, y_test)
