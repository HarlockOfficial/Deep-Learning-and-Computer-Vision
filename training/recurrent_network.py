import training.utility as utility
from neural_network import RecurrentNetwork


def train_recurrent_network(n_output_labels, x_train, y_train):
    return utility.train_network(RecurrentNetwork, "recurrent_network", n_output_labels, x_train, y_train)


def test_recurrent_network(n_output_labels, x_test, y_test):
    return utility.test_network(RecurrentNetwork, "recurrent_network", n_output_labels, x_test, y_test)
