import preprocessing.utility as utility
import training.recurrent_network

logger = utility.default_logger(__file__)


def preprocessing_rnn_gnn(pdb_path: str, interaction_distance: float, output_path: str = None):
    """
        This function will be used to preprocess the dataset.
        It will return the preprocessed data.
    """
    logger.info("Preprocessing the dataset")
    import preprocessing
    preprocessed_rrn_data = preprocessing.rnn_preprocessing.extract_rnn_data(pdb_path)
    if output_path is not None:
        preprocessing.rnn_preprocessing.dump_to_file(preprocessed_rrn_data, output_path)
    expected_results = preprocessing.determine_interface\
        .compute_interface(interaction_distance=interaction_distance, pdb_path=pdb_path)
    if expected_results is None:
        logger.critical("Could not compute the interacting interface for the dataset.")
        exit(1)

    preprocessed_gnn_data = preprocessing.gnn_preprocessing.extract_gnn_data(pdb_path)
    if output_path is not None:
        preprocessing.gnn_preprocessing.dump_to_file(preprocessed_gnn_data, output_path)
    distance_matrix = preprocessing.gnn_preprocessing.create_distance_matrix(preprocessed_gnn_data)
    contact_matrix = preprocessing.gnn_preprocessing.create_contact_matrix(distance_matrix)
    aminoacid_list = [x[0: 3] for x in preprocessed_gnn_data]
    del distance_matrix

    return preprocessed_rrn_data, expected_results, (aminoacid_list, contact_matrix)


def preprocess_chemical_features(chemical_features_path: str, output_path: str = None):
    import preprocessing
    preprocessed_chemical_features = preprocessing.ffnn_preprocessing.extract_all_chemical_features(
        chemical_features_path)
    if output_path is not None:
        preprocessing.ffnn_preprocessing.dump_to_file(preprocessed_chemical_features, output_path)
    return preprocessed_chemical_features


def main(pdb_path: str, chemical_features_path: str, interaction_distance: float, output_path=None):
    logger.info("Obtaining preprocessed data")
    preprocessed_rrn_data, expected_results, preprocessed_gnn_data = preprocessing_rnn_gnn(
        pdb_path, interaction_distance, output_path)
    logger.info("Obtaining preprocessed chemical features")
    preprocessed_chemical_features = preprocess_chemical_features(chemical_features_path, output_path)

    logger.info("Assuming all data have same length")
    assert len(preprocessed_rrn_data) == len(preprocessed_gnn_data[0]) == len(preprocessed_gnn_data[1]) == len(
        expected_results)

    logger.info("Training the RNN")
    rnn_model = training.recurrent_network.\
        train_recurrent_network(len(preprocessed_rrn_data), preprocessed_rrn_data, expected_results)
    logger.info("Training the GCN")
    gnn_model = training.graph_convolutional_network.\
        train_graph_convolutional_network(len(preprocessed_rrn_data), preprocessed_gnn_data, expected_results)

    logger.info("Predicting RNN results")
    rnn_result = rnn_model.predict(preprocessed_rrn_data)
    logger.info("Predicting GCN results")
    gnn_result = gnn_model.predict(preprocessed_gnn_data)

    logger.error(f"{rnn_result}\n\n{gnn_result}\n\n{preprocessed_chemical_features}")
    return None

    input_vector = (rnn_result, gnn_result, preprocessed_chemical_features)

    ffnn_model = training.feed_forward_network.\
        train_feed_forward_network(len(preprocessed_rrn_data), input_vector, expected_results)
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
    args.add_argument("pdb_path", help="Path to the pdb")
    args.add_argument("chemical_features_path", help="Path to the chemical features")
    args.add_argument("--interaction_distance", default=6.0, type=float, help="Interaction distance")
    args.add_argument("-o", "--output", default=None, help="Path to the output folder")
    utility.add_default_parameters(args)

    args = args.parse_args()
    utility.default_logging(args, logger)

    logger.info("Starting the program")
    main(args.pdb_path, args.chemical_features_path, args.output)
    logger.info("Program finished")
