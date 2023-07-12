from neural_network import GraphAttentionNetwork
from training import utility


def train_graph_attention_network(n_output_labels, x_train, y_train):
    gan = GraphAttentionNetwork(n_label=n_output_labels, activation='relu', output_activation='linear', use_bias=True)
    return utility.train_network(gan, "graph_attention_network", x_train, y_train)


def test_graph_attention_network(n_output_labels, x_test, y_test):
    gan = GraphAttentionNetwork(n_label=n_output_labels, activation='relu', output_activation='linear', use_bias=True)
    return utility.test_network(gan, "graph_attention_network", x_test, y_test)
