from neural_network import GraphConvolutionalNetwork

from training import utility

from spektral.models.gcn import GCN

def train_graph_convolutional_network(n_output_labels, x_train):
    #gcn = GraphConvolutionalNetwork(n_label=n_output_labels, activation='relu', output_activation='softmax', use_bias=True)
    gcn = GCN(n_labels=n_output_labels, activation='sigmoid', output_activation='linear', use_bias=True)
    return utility.train_network(gcn, "graph_convolutional_network", x_train)


def test_graph_convolutional_network(n_output_labels, x_test):
    gcn = GCN(n_labels=n_output_labels, activation='sigmoid', output_activation='linear', use_bias=True)
    return utility.test_network(gcn, "graph_convolutional_network", x_test)
