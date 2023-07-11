import preprocessing.utility as utility
import training.recurrent_network
import tensorflow as tf
logger = utility.default_logger(__file__)


def preprocessing_rnn_gnn(pdb_path: str, interaction_distance: float = 6.0, output_path: str = None):
    """
        This function will be used to preprocess the dataset.
        It will return the preprocessed data.
    """
    logger.info("Preprocessing the dataset")
    import preprocessing
    preprocessed_rrn_data = preprocessing.rnn_preprocessing.extract_rnn_data(pdb_path)
    if output_path is not None:
        preprocessing.rnn_preprocessing.dump_to_file_csv(preprocessed_rrn_data, output_path + "/preprocessed_rnn.csv")
    expected_results = preprocessing.determine_interface \
        .compute_interface(interaction_distance=interaction_distance, pdb_path=pdb_path)
    if expected_results is None:
        logger.critical("Could not compute the interacting interface for the dataset.")
        exit(1)

    logger.debug(str(expected_results))

    preprocessed_gnn_data = preprocessing.gnn_preprocessing.extract_gnn_data(pdb_path)
    if output_path is not None:
        preprocessing.gnn_preprocessing.dump_to_file_csv(preprocessed_gnn_data, output_path + "/preprocessed_gnn.csv")
    distance_matrix = preprocessing.gnn_preprocessing.create_distance_matrix(preprocessed_gnn_data)
    contact_matrix = preprocessing.gnn_preprocessing.create_contact_matrix(distance_matrix)
    aminoacid_list = [x[0: 3] for x in preprocessed_gnn_data]
    del distance_matrix
    rnn_input_one_hot_encoding, rnn_different_protein_names_index, rnn_different_residue_names_index = utility.to_one_hot_encoding_input_for_rnn(
        preprocessed_rrn_data)
    output_vector_one_hot_encoding, output_different_protein_names_index, output_different_residue_names_index = utility.to_one_hot_encoding_output(
        expected_results)
    gcn_input_vector_one_hot_encoding, gcn_different_protein_names_index, gcn_different_residue_names_index = utility.to_one_hot_encoding_input_for_gcn(
        aminoacid_list, contact_matrix)

    assert len(rnn_different_protein_names_index) == len(output_different_protein_names_index) == len(gcn_different_protein_names_index)
    assert len(rnn_different_residue_names_index) == len(output_different_residue_names_index) == len(gcn_different_residue_names_index)

    for key in rnn_different_protein_names_index.keys():
        for key2 in output_different_protein_names_index.keys():
            for key3 in gcn_different_protein_names_index.keys():
                assert rnn_different_protein_names_index[key] == output_different_protein_names_index[key] == gcn_different_protein_names_index[key]
                assert rnn_different_protein_names_index[key2] == output_different_protein_names_index[key2] == gcn_different_protein_names_index[key2]
                assert rnn_different_protein_names_index[key3] == output_different_protein_names_index[key3] == gcn_different_protein_names_index[key3]

    return rnn_input_one_hot_encoding, output_vector_one_hot_encoding, gcn_input_vector_one_hot_encoding, output_different_protein_names_index, output_different_residue_names_index


def preprocess_chemical_features(chemical_features_path: str, output_path: str = None):
    import preprocessing
    preprocessed_chemical_features = preprocessing.ffnn_preprocessing.extract_all_chemical_features(
        chemical_features_path)
    if output_path is not None:
        preprocessing.ffnn_preprocessing.dump_to_file_csv(preprocessed_chemical_features,
                                                          output_path + "/preprocessed_ffnn.csv")
    return preprocessed_chemical_features


def main(pdb_path: str, chemical_features_path: str, interaction_distance: float = 6.0, output_path=None):
    logger.info("Obtaining preprocessed data")
    preprocessed_rnn_data, expected_results, preprocessed_gnn_data, \
        different_protein_names_index, different_residue_names_index = preprocessing_rnn_gnn(
        pdb_path, interaction_distance, output_path)
    logger.info("Obtaining preprocessed chemical features")
    preprocessed_chemical_features = preprocess_chemical_features(chemical_features_path, output_path)

    logger.info("Assuming all data have same length")
    logger.debug(len(expected_results))
    logger.debug(str(expected_results))
    assert len(preprocessed_rnn_data) == len(preprocessed_gnn_data) == len(expected_results)

    logger.info("Training the RNN")

    tensor_pre_array = tf.convert_to_tensor(preprocessed_rnn_data)
    tensor_exp_array = tf.convert_to_tensor(expected_results)

    preprocessed_rnn_data_reshaped = tf.reshape(tensor_pre_array, (len(tensor_pre_array[0]), len(tensor_pre_array)))
    expected_rnn_data_reshaped = tf.reshape(tensor_exp_array, (len(tensor_exp_array[0]), len(tensor_exp_array)))

    rnn_model = training.recurrent_network. \
        train_recurrent_network(len(expected_results), preprocessed_rnn_data_reshaped, expected_rnn_data_reshaped)
    logger.info("Training the GCN")
    gnn_model = training.graph_convolutional_network. \
        train_graph_convolutional_network(preprocessed_gnn_data, expected_results)

    logger.info("Predicting RNN results")
    rnn_result = rnn_model.predict(preprocessed_rnn_data)
    logger.info("Predicting GCN results")
    gnn_result = gnn_model.predict(preprocessed_gnn_data)

    logger.error(f"{rnn_result}\n\n{gnn_result}\n\n{preprocessed_chemical_features}")

    input_vector = utility.to_one_hot_encoding_input_for_ffnn(rnn_result, gnn_result, preprocessed_chemical_features,
                                                              different_residue_names_index)

    ffnn_model = training.feed_forward_network. \
        train_feed_forward_network(len(expected_results), input_vector, expected_results)
    logger.info("Training finished")

    return rnn_model, gnn_model, ffnn_model, different_protein_names_index, different_residue_names_index


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
    main(args.pdb_path, args.chemical_features_path, args.interaction_distance, args.output)
    logger.info("Program finished")
