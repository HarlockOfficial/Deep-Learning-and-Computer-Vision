from neural_network import OurNetwork
from training import utility


def train_our_network(contact_matrix_shape, x_train, y_train, validation_data):
    net = OurNetwork(contact_matrix_shape=contact_matrix_shape)
    return utility.train_network(net, x_train, y_train, validation_data)


def test_our_network(contact_matrix_shape, x_test, y_test):
    net = OurNetwork(contact_matrix_shape=contact_matrix_shape)
    return utility.test_network(net, x_test, y_test)
