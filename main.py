import preprocessing.utility as utility
import training.recurrent_network

logger = utility.default_logger(__file__)


def preprocessing_rnn_gnn(dataset_path: str, output_path: str = None):
    """
        This function will be used to preprocess the dataset.
        It will return the preprocessed data.
    """
    logger.info("Preprocessing the dataset")
    import preprocessing
    preprocessed_rrn_data = preprocessing.rnn_preprocessing.extract_rnn_data(dataset_path)
    if output_path is not None:
        preprocessing.rnn_preprocessing.dump_to_file(preprocessed_rrn_data, output_path)
    # TODO solve, dont know what should be the output here
    expected_rnn_results = None
    # expected_rnn_results = preprocessing.rnn_preprocessing.extract_expected_results(dataset_path)

    preprocessed_gnn_data = preprocessing.gnn_preprocessing.extract_gnn_data(dataset_path)
    if output_path is not None:
        preprocessing.gnn_preprocessing.dump_to_file(preprocessed_gnn_data, output_path)
    distance_matrix = preprocessing.gnn_preprocessing.create_distance_matrix(preprocessed_gnn_data)
    expected_gnn_results = preprocessing.gnn_preprocessing.create_contact_matrix(distance_matrix)
    del distance_matrix

    return preprocessed_rrn_data, expected_rnn_results, preprocessed_gnn_data, expected_gnn_results


def preprocess_chemical_features(chemical_features_path: str, output_path: str = None):
    import preprocessing
    preprocessed_chemical_features = preprocessing.ffnn_preprocessing.extract_all_chemical_features(
        chemical_features_path)
    if output_path is not None:
        preprocessing.ffnn_preprocessing.dump_to_file(preprocessed_chemical_features, output_path)
    return preprocessed_chemical_features


def main(dataset_path: str, chemical_features_path: str, output_path=None):
    preprocessed_rrn_data, expected_rnn_results, preprocessed_gnn_data, expected_gnn_results = preprocessing_rnn_gnn(
        dataset_path, output_path)
    # preprocessed_chemical_features = preprocess_chemical_features(chemical_features_path, output_path)

    rnn_model = training.recurrent_network.train_recurrent_network(preprocessed_rrn_data, expected_rnn_results)
    gnn_model = training.graph_convolutional_network.train_graph_convolutional_network(preprocessed_gnn_data,
                                                                                       expected_gnn_results)
    # ffnn_model = training.feed_forward_neural_network.train_feed_forward_neural_network(preprocessed_chemical_features, expected_gnn_results)
    logger.info("Training finished")


if __name__ == "__main__":
    """
        This file is the main file of the project.
        It will be used to run the project.
        The following arguments are required:
            - path to the dataset
            - path to the chemical features
            - eventually, the path to the output folder
    """
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("dataset_path", help="Path to the dataset")
    args.add_argument("chemical_features_path", help="Path to the chemical features")
    args.add_argument("-o", "--output", default=None, help="Path to the output folder")
    utility.add_default_parameters(args)

    args = args.parse_args()
    utility.default_logging(args, logger)

    main(args.dataset_path, args.chemical_features_path, args.output)
