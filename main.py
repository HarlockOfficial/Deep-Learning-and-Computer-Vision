import random

import numpy as np
import tensorflow as tf

import preprocessing
import preprocessing.utility as utility
import training.recurrent_network

logger = utility.default_logger(__file__)


def preprocessing_rnn_gnn(pdb_path: str, interaction_distance: float = 6.0, output_path: str = None,
                          different_protein_names_index = None, different_residue_names_index = None):
    """
        This function will be used to preprocess the dataset.
        It will return the preprocessed data.
    """
    logger.info("Preprocessing the dataset")

    extract_data = preprocessing.utility.extract_data(pdb_path)
    #if output_path is not None:
    #    preprocessing.rnn_preprocessing.dump_to_file_csv(extract_data, output_path + "/preprocessed.csv")
    #    TODO: fix later
    expected_results = preprocessing.determine_interface \
        .compute_interface(interaction_distance=interaction_distance, pdb_path=pdb_path)
    if expected_results is None:
        logger.critical("Could not compute the interacting interface for the dataset.")
        exit(1)

    new_extract_data = []
    for protein_name, residue_id, residue_name, center_of_mass_x, center_of_mass_y, center_of_mass_z in extract_data:
        new_extract_data.append([protein_name, residue_id, residue_name, center_of_mass_x, center_of_mass_y,
                                 center_of_mass_z])

    logger.debug("Expected results main: " + str(expected_results))
    import pandas as pd
    df = pd.DataFrame(new_extract_data, columns=['protein_name', 'residue_id', 'residue_name', 'center_of_mass_x',
                                                 'center_of_mass_y', 'center_of_mass_z'])
    del new_extract_data
    if max(expected_results) != min(expected_results):
        extract_data, expected_results = preprocessing.utility.balance_classes(df, expected_results)
    else:
        #There is only one class so no resample
        extract_data = df
        logger.info("There is only one class so no resample")

    new_extract_data, new_expected_results = preprocessing.utility.split_data(extract_data, expected_results)
    out = []
    for extract_data, expected_results in zip(new_extract_data, new_expected_results):
        #if output_path is not None:
            #preprocessing.gnn_preprocessing.dump_to_file_csv(expected_results, output_path + "/preprocessed.csv")
        # TODO: fix later
        distance_matrix = preprocessing.gnn_preprocessing.create_distance_matrix(extract_data)
        contact_matrix = preprocessing.gnn_preprocessing.create_contact_matrix(distance_matrix)
        del distance_matrix
        #The aminoacid list contains protein name, residue number and residue name which is the actual input for the GCN

        aminoacid_list = [x[0: 3] for x in extract_data]
        input_one_hot_encoding, _ = utility.to_one_hot_encoding_input(
            aminoacid_list, different_residue_names_index)

        import neural_network.utility.gcn_dataset as gcn_dataset
        import numpy as np
        input_tensor = tf.convert_to_tensor(value=input_one_hot_encoding, dtype=tf.float32)
        dataset = gcn_dataset.MyDataset(input_tensor.numpy(), contact_matrix.numpy(), np.array(expected_results))
        logger.debug("Expected results: " + str(expected_results))

        out.append((aminoacid_list, expected_results, input_one_hot_encoding, contact_matrix, different_protein_names_index, \
            different_residue_names_index, dataset, aminoacid_list))
    return out


def preprocess_chemical_features(chemical_features_path: str, output_path: str = None):
    import preprocessing
    preprocessed_chemical_features = preprocessing.ffnn_preprocessing.extract_all_chemical_features(
        chemical_features_path)
    if output_path is not None:
        preprocessing.ffnn_preprocessing.dump_to_file_csv(preprocessed_chemical_features,
                                                          output_path + "/preprocessed_ffnn.csv")
    return preprocessed_chemical_features


def train_our_network_on_a_file(pdb_path: str, chemical_features_path: str, interaction_distance: float = 6.0, chemical_features=None, output_path=None, different_protein_names_index=None, different_residue_names_index=None, pdb_validation_path=None):
    logger.info("Obtaining preprocessed data")
    for preprocessed_rnn_data, expected_results, preprocessed_gnn_data, contact_matrix, \
            _, _, _, aminoacid_list in preprocessing_rnn_gnn(
        pdb_path, interaction_distance, output_path, different_protein_names_index, different_residue_names_index):

        validation_rnn_data, validation_expected_results, validation_gnn_data, validation_contact_matrix, \
            validation_different_protein_names_index, _, validation_dataset, validation_aminoacid_list = random.choice(
            preprocessing_rnn_gnn(pdb_validation_path, interaction_distance, output_path, different_protein_names_index,
                                  different_residue_names_index))

        logger.debug("Preprocessed data length (using only one of the data since they're identical): " + str(
            len(preprocessed_rnn_data[0])))
        logger.info("Obtaining preprocessed chemical features")

        if chemical_features is None:
            preprocessed_chemical_features = preprocess_chemical_features(chemical_features_path, output_path)
        else:
            preprocessed_chemical_features = chemical_features

        logger.info("Assuming all data have same length")
        logger.debug(len(expected_results))
        logger.debug(str(expected_results))
        assert len(preprocessed_rnn_data) == len(preprocessed_gnn_data) == len(expected_results)

        logger.info("Training the Network")

        if len(preprocessed_rnn_data[0]) == 3:
            preprocessed_rnn_data, _ = utility.to_one_hot_encoding_input(preprocessed_rnn_data,
                                                                         different_residue_names_index)

        if len(validation_rnn_data[0]) == 3:
            validation_rnn_data, _ = utility.to_one_hot_encoding_input(validation_rnn_data,
                                                                       different_residue_names_index)

        tensor_pre_array = tf.convert_to_tensor(preprocessed_rnn_data)
        tensor_exp_array = tf.convert_to_tensor(expected_results)

        chemical_features = utility.to_one_hot_encoding_input_for_ffnn(preprocessed_chemical_features, aminoacid_list)
        tensor_chem_data = tf.convert_to_tensor(chemical_features)

        from dotenv import load_dotenv
        load_dotenv()

        model, result = training.our_network.train_our_network(contact_matrix.get_shape().as_list()[1], [tensor_pre_array, contact_matrix, tensor_chem_data],
                                                               tensor_exp_array,
                                                               validation_data=(tf.convert_to_tensor(validation_rnn_data),
                                                                                tf.convert_to_tensor(validation_expected_results)))
        logger.info("Training finished")

    return None, None, model, different_protein_names_index, different_residue_names_index, aminoacid_list, preprocessed_chemical_features


def train_whole_network_on_a_file(pdb_path: str, chemical_features_path: str, interaction_distance: float = 6.0, chemical_features=None, output_path=None, different_protein_names_index=None, different_residue_names_index=None, pdb_validation_path=None):
    logger.info("Obtaining preprocessed data")
    for preprocessed_rnn_data, expected_results, preprocessed_gnn_data, contact_matrix, \
        _, _, dataset, aminoacid_list in preprocessing_rnn_gnn(
        pdb_path, interaction_distance, output_path, different_protein_names_index, different_residue_names_index):

        validation_rnn_data, validation_expected_results, validation_gnn_data, validation_contact_matrix, \
        validation_different_protein_names_index, _, validation_dataset, validation_aminoacid_list = random.choice(preprocessing_rnn_gnn(pdb_validation_path, interaction_distance, output_path, different_protein_names_index, different_residue_names_index))

        logger.debug("Preprocessed data length (using only one of the data since they're identical)" + str(len(preprocessed_rnn_data[0])))
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

        #input('------------------------------------------------------------------')
        #print(str(preprocessed_rnn_data), str(expected_results), str(len(preprocessed_rnn_data)), str(len(expected_results)))
        #input('------------------------------------------------------------------')
        if len(preprocessed_rnn_data[0]) == 3:
            preprocessed_rnn_data, _ = utility.to_one_hot_encoding_input(preprocessed_rnn_data, different_residue_names_index)

        if len(validation_rnn_data[0]) == 3:
            validation_rnn_data, _ = utility.to_one_hot_encoding_input(validation_rnn_data, different_residue_names_index)

        tensor_pre_array = tf.convert_to_tensor(preprocessed_rnn_data)
        tensor_exp_array = tf.convert_to_tensor(expected_results)

        from dotenv import load_dotenv
        load_dotenv()
        import os

        rnn_model, rnn_result = training.recurrent_network. \
            train_recurrent_network(int(os.getenv('MAX_INPUT')), tensor_pre_array, tensor_exp_array,
                                    validation_data=(tf.convert_to_tensor(validation_rnn_data),
                                                     tf.convert_to_tensor(validation_expected_results)))
            #train_recurrent_network(len(expected_results), tensor_pre_array, tensor_exp_array)

        logger.info("Training the GCN")
        gnn_model, gnn_result = training.graph_convolutional_network. \
            train_graph_convolutional_network(1, dataset, validation_data=validation_dataset)
            #train_graph_convolutional_network(int(os.getenv('MAX_INPUT')), dataset)
            #train_graph_convolutional_network(len(expected_results), dataset)

        logger.info("Predicting RNN results")
        #rnn_result = rnn_model.predict(preprocessed_rnn_data, batch_size=len(preprocessed_rnn_data))
        #logger.debug(rnn_result)
        logger.info("Predicting GCN results")
        #gnn_tensor_input = tf.convert_to_tensor(preprocessed_gnn_data, dtype=tf.float32)
        #logger.debug(gnn_tensor_input)
        #logger.debug(contact_matrix)
        #gnn_result = gnn_model.predict(x=[gnn_tensor_input.numpy(), tf.cast(contact_matrix, dtype=tf.float32).numpy()], batch_size=len(preprocessed_gnn_data))

        gnn_result = rnn_result = expected_results
        logger.debug(f"{rnn_result}\n\n{gnn_result}\n\n{preprocessed_chemical_features}")

        input_vector = utility.to_one_hot_encoding_input_for_ffnn(preprocessed_chemical_features, aminoacid_list)

        logger.info("Predicting RNN validation results")
        #validation_rnn_result = rnn_model.predict(validation_rnn_data, batch_size=len(validation_rnn_data))
        validation_rnn_result = validation_expected_results
        logger.debug(validation_rnn_result)
        logger.info("Predicting GCN validation results")
        #validation_gnn_tensor_input = tf.convert_to_tensor(validation_gnn_data, dtype=tf.float32)
        #logger.debug(validation_gnn_tensor_input)
        #logger.debug(validation_contact_matrix)
        #validation_gnn_result = gnn_model.predict(x=[validation_gnn_tensor_input.numpy(), tf.cast(validation_contact_matrix, dtype=tf.float32).numpy()], batch_size=len(validation_gnn_data))
        validation_gnn_result = validation_expected_results
        logger.debug(f"{validation_rnn_result}\n\n{validation_gnn_result}\n\n{preprocessed_chemical_features}")
        validation_vector = utility.to_one_hot_encoding_input_for_ffnn(preprocessed_chemical_features, validation_aminoacid_list)

        logger.info("Training the FFN")
        logger.info("Preprocessed FNN data length " + str(len(input_vector[0])))

        '''
         Since 1 neuron is createrd for each input, using the len of the expected results as output creates a matrix
         we actually need one predicition for each neuron. 
        '''

        ffnn_model, ffn_resutl = training.feed_forward_network. \
            train_feed_forward_network(1, [np.array(rnn_result), np.array(gnn_result), np.array(input_vector)], np.array(expected_results), validation_data=((np.array(validation_rnn_result), np.array(validation_gnn_result), np.array(validation_vector)), np.array(validation_expected_results)))
            #train_feed_forward_network(len(expected_results), input_vector, expected_results)
        logger.info("Training finished")

    return rnn_model, gnn_model, ffnn_model, different_protein_names_index, different_residue_names_index, aminoacid_list, preprocessed_chemical_features


def test_whole_network_on_a_file(pdb_path, chemical_features_path, interaction_distance: float = 6.0, chemical_features=None, output_path=None, different_protein_names_index = None, different_residue_names_index = None):
    logger.info("Obtaining preprocessed data")
    for preprocessed_rnn_data, expected_results, preprocessed_gnn_data, contact_matrix, \
            _, _, dataset, aminoacid_list in preprocessing_rnn_gnn(
        pdb_path, interaction_distance, output_path, different_protein_names_index, different_residue_names_index):

        logger.debug("Preprocessed data length (using only one of the data since they're identical)" + str(
            len(preprocessed_rnn_data[0])))
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

        from dotenv import load_dotenv
        load_dotenv()
        import os

        if len(preprocessed_rnn_data[0]) == 3:
            preprocessed_rnn_data, _ = utility.to_one_hot_encoding_input(preprocessed_rnn_data, different_residue_names_index)

        tensor_pre_array = tf.convert_to_tensor(preprocessed_rnn_data)
        tensor_exp_array = tf.convert_to_tensor(expected_results)

        rnn_model, rnn_result = training.recurrent_network. \
            test_recurrent_network(int(os.getenv('MAX_INPUT')), tensor_pre_array, tensor_exp_array)
        logger.info("RNN result: " + str(rnn_result))
        logger.info("Testing the GCN")
        gnn_model, gnn_result = training.graph_convolutional_network. \
            test_graph_convolutional_network(1, dataset)
        logger.info("GCN result: " + str(gnn_result))

        logger.info("Predicting RNN results")
        rnn_result = rnn_model.predict(preprocessed_rnn_data, batch_size=len(preprocessed_rnn_data))
        logger.info("Predicting GCN results")
        logger.debug(rnn_result)
        logger.debug(preprocessed_gnn_data)
        logger.debug(contact_matrix)
        gnn_tensor_input = tf.convert_to_tensor(preprocessed_gnn_data, dtype=tf.float32)
        contact_matrix_tensor = tf.cast(contact_matrix, dtype=tf.float32)
        gnn_result = gnn_model.predict(x=[gnn_tensor_input.numpy(), contact_matrix_tensor.numpy()],
                                       batch_size=len(preprocessed_gnn_data))

        logger.debug(f"{rnn_result}\n\n{gnn_result}\n\n{preprocessed_chemical_features}")

        input_vector = utility.to_one_hot_encoding_input_for_ffnn(preprocessed_chemical_features,
                                                                  aminoacid_list)

        logger.info("Testing the FFN")
        ffnn_model, ffn_results = training.feed_forward_network. \
            test_feed_forward_network(1, [np.array(rnn_result), np.array(gnn_result), np.array(input_vector)], np.array(expected_results))
        logger.info("Testing finished")

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

    ordered_names_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                          'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                          'THR', 'TRP', 'TYR', 'VAL']

    preprocessed_chemical_features = None
    different_residue_names_index = dict()
    for index, x in enumerate(ordered_names_list):
        different_residue_names_index[x] = index

    rnn_model, gnn_model, ffnn_model, _, different_residue_names_index, _, \
    preprocess_chemical_features = train_whole_network_on_a_file(args.pdb_path, args.chemical_features_path,
                                                                 args.interaction_distance, args.output,
                                                                 different_residue_names_index=different_residue_names_index)

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
