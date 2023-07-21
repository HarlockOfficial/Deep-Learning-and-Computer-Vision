from neural_network import OurNetwork
from training import utility


def train_our_network(x_train, y_train, validation_data):
    net = OurNetwork()
    return utility.train_network(net, 'our_network', x_train, y_train, validation_data, batch_size=len(x_train[0]))


def test_our_network(x_test, y_test):
    net = OurNetwork()
    return utility.test_network(net, 'our_network', x_test, y_test)
