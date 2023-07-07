from spektral.models.gcn import GCN

from training import utility


def train_graph_convolutional_network(n_output_labels, x_train, y_train):
    return utility.train_network(GCN, "graph_convolutional_network", n_output_labels, x_train, y_train)


def test_graph_convolutional_network(n_output_labels, x_test, y_test):
    return utility.test_network(GCN, "graph_convolutional_network", n_output_labels, x_test, y_test)
