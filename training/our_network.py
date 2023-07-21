from neural_network import OurNetwork
from training import utility


def train_our_network(x_train, y_train, validation_data):
    net = OurNetwork()
    return utility.train_network(net, x_train, y_train, validation_data)


def test_our_network(x_test, y_test):
    net = OurNetwork()
    return utility.test_network(net, x_test, y_test)
