from spektral.models.gcn import GCN

from training import utility


def train_graph_convolutional_network(n_output_labels, x_train, y_train):
    gcn = GCN(n_labels=n_output_labels, activation='relu', output_activation='linear', use_bias=True)
    return utility.train_network(gcn, "graph_convolutional_network", x_train, y_train)


def test_graph_convolutional_network(n_output_labels, x_test, y_test):
    gcn = GCN(n_labels=n_output_labels, activation='relu', output_activation='linear', use_bias=True)
    return utility.test_network(gcn, "graph_convolutional_network", x_test, y_test)
