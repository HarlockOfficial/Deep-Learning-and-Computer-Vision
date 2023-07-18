import tensorflow as tf

import preprocessing.utility as utility
import training.recurrent_network

logger = utility.default_logger(__file__)


def preprocessing_rnn_gnn(pdb_path: str, interaction_distance: float = 6.0, output_path: str = None, different_protein_names_index = None, different_residue_names_index = None):
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
    rnn_input_one_hot_encoding, different_protein_names_index, different_residue_names_index = utility.to_one_hot_encoding_input_for_rnn(
        preprocessed_rrn_data, different_protein_names_index, different_residue_names_index)
    gcn_input_vector_one_hot_encoding, different_protein_names_index, different_residue_names_index = utility.to_one_hot_encoding_input_for_gcn(
        aminoacid_list, different_protein_names_index, different_residue_names_index)

    import neural_network.utility.gcn_dataset as gcn_dataset
    import numpy as np

    dataset = gcn_dataset.MyDataset(gcn_input_vector_one_hot_encoding.numpy(), contact_matrix.numpy(), np.array(expected_results))
    return rnn_input_one_hot_encoding, expected_results, gcn_input_vector_one_hot_encoding, contact_matrix, different_protein_names_index, different_residue_names_index, dataset, aminoacid_list


def preprocess_chemical_features(chemical_features_path: str, output_path: str = None):
    import preprocessing
    preprocessed_chemical_features = preprocessing.ffnn_preprocessing.extract_all_chemical_features(
        chemical_features_path)
    if output_path is not None:
        preprocessing.ffnn_preprocessing.dump_to_file_csv(preprocessed_chemical_features,
                                                          output_path + "/preprocessed_ffnn.csv")
    return preprocessed_chemical_features


def train_whole_network_on_a_file(pdb_path: str, chemical_features_path: str, interaction_distance: float = 6.0, chemical_features=None, output_path=None, different_protein_names_index=None, different_residue_names_index=None):
    logger.info("Obtaining preprocessed data")
    preprocessed_rnn_data, expected_results, preprocessed_gnn_data, contact_matrix, \
        different_protein_names_index, different_residue_names_index, dataset, aminoacid_list = preprocessing_rnn_gnn(
        pdb_path, interaction_distance, output_path, different_protein_names_index, different_residue_names_index)
    logger.info("Preprocessed RNN data length " + str(len(preprocessed_rnn_data[0])))
    logger.info("Preprocessed GNN data length " + str(len(preprocessed_gnn_data[0])))
    logger.info("Obtaining preprocessed chemical features")
    if chemical_features is None:
        preprocessed_chemical_features = preprocess_chemical_features(chemical_features_path, output_path)
    else:
        preprocessed_chemical_features = chemical_features

    logger.info("Assuming all data have same length")
    logger.debug(len(expected_results))
    logger.debug(str(expected_results))
    assert len(preprocessed_rnn_data) == len(preprocessed_gnn_data) == len(expected_results)

    logger.info("Training the RNN")

    #print(str(preprocessed_rnn_data), str(expected_results), str(len(preprocessed_rnn_data)),
    #      str(len(expected_results)))
    #input('------------------------------------------------------------------')
    preprocessed_rnn_data, expected_results = utility.balance_classes(preprocessed_rnn_data, expected_results)
    #input('------------------------------------------------------------------')
    #print(str(preprocessed_rnn_data), str(expected_results), str(len(preprocessed_rnn_data)), str(len(expected_results)))
    #input('------------------------------------------------------------------')
    tensor_pre_array = tf.convert_to_tensor(preprocessed_rnn_data)
    tensor_exp_array = tf.convert_to_tensor(expected_results)

    from dotenv import load_dotenv
    load_dotenv()
    import os

    rnn_model = training.recurrent_network. \
        train_recurrent_network(2*int(os.getenv('MAX_INPUT')), tensor_pre_array, tensor_exp_array)
        #train_recurrent_network(len(expected_results), tensor_pre_array, tensor_exp_array)

    logger.info("Training the GCN")
    gnn_model = training.graph_convolutional_network. \
        train_graph_convolutional_network(1, dataset)
        #train_graph_convolutional_network(int(os.getenv('MAX_INPUT')), dataset)
        #train_graph_convolutional_network(len(expected_results), dataset)

    logger.info("Predicting RNN results")
    rnn_result = rnn_model.predict(preprocessed_rnn_data, batch_size=len(preprocessed_rnn_data))
    logger.info("Predicting GCN results")
    logger.debug(rnn_result)
    logger.debug(preprocessed_gnn_data)
    logger.debug(contact_matrix)
    gnn_result = gnn_model.predict(x=[preprocessed_gnn_data.numpy(), contact_matrix.numpy()], batch_size=len(preprocessed_gnn_data))

    logger.debug(f"{rnn_result}\n\n{gnn_result}\n\n{preprocessed_chemical_features}")

    input_vector = utility.to_one_hot_encoding_input_for_ffnn(rnn_result, gnn_result, preprocessed_chemical_features, aminoacid_list)

    logger.info("Training the FFN")
    logger.info("Preprocessed FNN data length " + str(len(input_vector[0])))
    ffnn_model = training.feed_forward_network. \
        train_feed_forward_network(2*int(os.getenv('MAX_INPUT')), input_vector, expected_results)
        #train_feed_forward_network(len(expected_results), input_vector, expected_results)
    logger.info("Training finished")

    return rnn_model, gnn_model, ffnn_model, different_protein_names_index, different_residue_names_index, aminoacid_list, preprocessed_chemical_features


def test_whole_network_on_a_file(pdb_path, chemical_features_path, interaction_distance: float = 6.0, chemical_features=None, output_path=None, different_protein_names_index = None, different_residue_names_index = None):
    logger.info("Obtaining preprocessed data")
    preprocessed_rnn_data, expected_results, preprocessed_gnn_data, contact_matrix, \
        different_protein_names_index, different_residue_names_index, dataset, aminoacid_list = preprocessing_rnn_gnn(
        pdb_path, interaction_distance, output_path, different_protein_names_index, different_residue_names_index)
    logger.info("Obtaining preprocessed chemical features")
    if chemical_features is None:
        preprocessed_chemical_features = preprocess_chemical_features(chemical_features_path, output_path)
    else:
        preprocessed_chemical_features = chemical_features

    logger.info("Assuming all data have same length")
    logger.debug(len(expected_results))
    logger.debug(str(expected_results))
    assert len(preprocessed_rnn_data) == len(preprocessed_gnn_data) == len(expected_results)

    logger.info("Testing the RNN")

    tensor_pre_array = tf.convert_to_tensor(preprocessed_rnn_data)
    tensor_exp_array = tf.convert_to_tensor(expected_results)

    rnn_model, rnn_result = training.recurrent_network. \
        test_recurrent_network(len(expected_results), tensor_pre_array, tensor_exp_array)
    logger.info("RNN result: " + str(rnn_result))
    logger.info("Training the GCN")
    gnn_model, gnn_result = training.graph_convolutional_network. \
        test_graph_convolutional_network(len(expected_results), dataset)
    logger.info("GCN result: " + str(gnn_result))

    logger.info("Predicting RNN results")
    rnn_result = rnn_model.predict(preprocessed_rnn_data, batch_size=len(preprocessed_rnn_data))
    logger.info("Predicting GCN results")
    logger.debug(rnn_result)
    logger.debug(preprocessed_gnn_data)
    logger.debug(contact_matrix)
    gnn_result = gnn_model.predict(x=[preprocessed_gnn_data.numpy(), contact_matrix.numpy()],
                                   batch_size=len(preprocessed_gnn_data))

    logger.debug(f"{rnn_result}\n\n{gnn_result}\n\n{preprocessed_chemical_features}")

    input_vector = utility.to_one_hot_encoding_input_for_ffnn(rnn_result, gnn_result, preprocessed_chemical_features,
                                                              aminoacid_list)

    logger.info("Testing the FFN")
    ffnn_model = training.feed_forward_network. \
        test_feed_forward_network(len(expected_results), input_vector, expected_results)
    logger.info("Training finished")

    return rnn_model, gnn_model, ffnn_model, different_protein_names_index, different_residue_names_index, aminoacid_list, preprocessed_chemical_features


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
    rnn_model, gnn_model, ffnn_model, different_protein_names_index, different_residue_names_index, aminoacid_list, \
    preprocess_chemical_features = train_whole_network_on_a_file(args.pdb_path, args.chemical_features_path,
                                                                 args.interaction_distance, args.output)
    logger.info("Trained RNN, with configs:\n" + str(rnn_model.get_config()))
    logger.info("Trained GCNN, with configs:\n" + str(gnn_model.get_config()))
    logger.info("Trained FFNN, with configs:\n" + str(ffnn_model.get_config()))

    logger.info("Starting testing")
    rnn_model, gnn_model, ffnn_model, _, _, _, \
    preprocess_chemical_features = test_whole_network_on_a_file(args.pdb_path, args.chemical_features_path,
                                                                args.interaction_distance, preprocess_chemical_features,
                                                                args.output)
    logger.info("Tested RNN, with configs:\n" + str(rnn_model.get_config()))
    logger.info("Tested GCNN, with configs:\n" + str(gnn_model.get_config()))
    logger.info("Tested FFNN, with configs:\n" + str(ffnn_model.get_config()))

    logger.info("Program finished")
