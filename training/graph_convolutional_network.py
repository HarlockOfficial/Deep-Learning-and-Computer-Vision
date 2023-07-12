from neural_network import GraphConvolutionalNetwork

from training import utility


def train_graph_convolutional_network(n_output_labels, x_train, y_train):
    gcn = GraphConvolutionalNetwork(n_label=n_output_labels, activation='relu', output_activation='softmax', use_bias=True)
    return utility.train_network(gcn, "graph_convolutional_network", x_train, y_train)


def test_graph_convolutional_network(n_output_labels, x_test, y_test):
    gcn = GraphConvolutionalNetwork(n_label=n_output_labels, activation='relu', output_activation='softmax', use_bias=True)
    return utility.test_network(gcn, "graph_convolutional_network", x_test, y_test)
