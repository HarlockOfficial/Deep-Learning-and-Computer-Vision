import argparse
from typing import Tuple, List, Union, Dict, Any
import tensorflow as tf
from tensorflow import Tensor


def add_default_parameters(parser: argparse.ArgumentParser):
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    parser.add_argument('--debug_log_file_path', type=str, default='data/logs/debug.log',
                        help='the path to the debug log file')
    parser.add_argument('--verbose', action='store_true', help='enable verbose mode')
    parser.add_argument('--verbose_log_file_path', type=str, default='data/logs/verbose.log',
                        help='the path to the verbose log file')
    parser.add_argument('--warning', action='store_true', help='enable warning mode')
    parser.add_argument('--warning_log_file_path', type=str, default='data/logs/warning.log',
                        help='the path to the warning log file')
    parser.add_argument('--error', action='store_true', help='enable error mode')
    parser.add_argument('--error_log_file_path', type=str, default='data/logs/error.log',
                        help='the path to the error log file')
    parser.add_argument('--critical', action='store_true', help='enable critical mode')
    parser.add_argument('--critical_log_file_path', type=str, default='data/logs/critical.log',
                        help='the path to the critical log file')


def default_logging(args, logger):
    if args.critical:
        logger.enable('CRITICAL')
        logger.add_file_path('CRITICAL', args.critical_log_file_path)
    if args.error:
        logger.enable('ERROR')
        logger.add_file_path('ERROR', args.error_log_file_path)
    if args.warning:
        logger.enable('WARNING')
        logger.add_file_path('WARNING', args.warning_log_file_path)
    if args.verbose:
        logger.enable('INFO')
        logger.add_file_path('INFO', args.verbose_log_file_path)
    if args.debug:
        logger.enable('DEBUG')
        logger.add_file_path('DEBUG', args.debug_log_file_path)


def default_logger(file):
    import os
    import sys

    if os.path.dirname(os.path.dirname(os.path.abspath(file))) not in sys.path:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(file))))

    from my_logging import Logger

    return Logger()


def get_residue_name_and_protein_name(residue, chain, dataset_file_name, logger):
    residue_name = residue.get_resname()
    logger.debug("residue name: " + str(residue_name))
    #protein_name = datset_file_name.split('/')[-1].split('.')[0][:-1] + chain.get_id()
    protein_name = chain.get_id()
    logger.debug("protein name: " + str(protein_name))
    residue_id = residue.get_full_id()[3][1]
    logger.debug("residue id: " + str(residue_id))
    return residue_name, residue_id, protein_name


def euclidean_distance(position1: tuple[float, float, float], position2: tuple[float, float, float]):
    import math
    return math.sqrt(sum([(position1[i] - position2[i]) ** 2 for i in range(len(position1))]))


def to_one_hot_encoding_input_for_rnn(rnn_input: list[tuple[str, int, str]]) -> tuple[list[list[int]], dict[str, int], dict[str, int]]:
    # process input
    different_protein_names_index = dict()
    different_residue_names_index = dict()
    for index, (protein_name, _, residue_name) in enumerate(rnn_input):
        if protein_name not in different_protein_names_index:
            different_protein_names_index[protein_name] = len(different_protein_names_index)
        if residue_name not in different_residue_names_index:
            different_residue_names_index[residue_name] = len(different_residue_names_index)

    amount_different_protein_names = len(different_protein_names_index.keys())
    amount_different_residue_names = len(different_residue_names_index.keys())

    protein_name_zero_vector = [0] * amount_different_protein_names
    residue_name_zero_vector = [0] * amount_different_residue_names

    rnn_input_one_hot_encoding = []
    for index, (protein_name, residue_id, residue_name) in enumerate(rnn_input):
        protein_name_one_hot_encoding = protein_name_zero_vector.copy()
        residue_name_one_hot_encoding = residue_name_zero_vector.copy()

        protein_name_one_hot_encoding[different_protein_names_index[protein_name]] = 1
        residue_name_one_hot_encoding[different_residue_names_index[residue_name]] = 1

        rnn_input_one_hot_encoding.append(protein_name_one_hot_encoding + [residue_id] + residue_name_one_hot_encoding)

    return rnn_input_one_hot_encoding, different_protein_names_index, different_residue_names_index


def to_one_hot_encoding_input_for_gcn(aminoacid_list: list[tuple[str, int, str]]) -> \
        tuple[Any, dict[str, int], dict[str, int]]:
    gcn_input_vector_one_hot_encoding, different_protein_names_index, different_residue_names_index = \
        to_one_hot_encoding_input_for_rnn(aminoacid_list)
    return tf.convert_to_tensor(value=gcn_input_vector_one_hot_encoding, dtype=tf.float32), different_protein_names_index, \
           different_residue_names_index


def to_one_hot_encoding_input_for_ffnn(rnn_result: list[list[int]], gnn_result: list[list[int]],
                                       preprocessed_chemical_features: dict[str, dict[str, float]],
                                       different_residue_names_index: dict[str, int]) ->\
        list[list[float]]:
    ffnn_input_vector_one_hot_encoding = []
    for element_list in rnn_result:
        ffnn_input_vector_one_hot_encoding.append([float(x) for x in element_list])

    for index, element in enumerate(rnn_result):
        ffnn_input_vector_one_hot_encoding[index].extend([float(x) for x in gnn_result[index]])

    one_hot_encoded_aminoacid_list = []
    for element in rnn_result:
        one_hot_encoded_aminoacid_list.append(element[:len(different_residue_names_index.keys())])
    aminoacid_name_list = []
    for aminoacid_list in one_hot_encoded_aminoacid_list:
        index = aminoacid_list.index(1)
        for key in different_residue_names_index.keys():
            if different_residue_names_index[key] == index:
                aminoacid_name_list.append(key)
                break

    for index, element in enumerate(ffnn_input_vector_one_hot_encoding):
        for chemical_feature in preprocessed_chemical_features.keys():
            ffnn_input_vector_one_hot_encoding[index].append(
                preprocessed_chemical_features[chemical_feature][aminoacid_name_list[index]])

    return ffnn_input_vector_one_hot_encoding
