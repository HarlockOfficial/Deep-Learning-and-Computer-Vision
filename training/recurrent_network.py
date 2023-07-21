import training.utility as utility
from neural_network import RecurrentNetwork


def train_recurrent_network(n_output_labels, x_train, y_train, validation_data):
    rnn = RecurrentNetwork(n_embeddings=n_output_labels, n_label=1, activation='relu', output_activation='sigmoid', use_bias=True)
    return utility.train_network(rnn, "recurrent_network", x_train, y_train, validation_data)


def test_recurrent_network(n_output_labels, x_test, y_test):
    rnn = RecurrentNetwork(n_embeddings=n_output_labels, n_label=1, activation='relu',
                           output_activation='sigmoid', use_bias=True)
    return utility.test_network(rnn, "recurrent_network", x_test, y_test)
