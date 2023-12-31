import argparse
from typing import Any, Union, Tuple, List

import pandas as pd
import tensorflow as tf
from Bio.PDB import PDBParser
from imblearn.over_sampling import SMOTENC


def is_hetero(res):
    return res.get_full_id()[3][0] != ' '


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

    from my_logging import logger_instance

    return logger_instance


def get_residue_name_and_protein_name(residue, chain, dataset_file_name, logger):
    residue_name = residue.get_resname()
    protein_name = chain.get_id()
    residue_id = residue.get_full_id()[3][1]
    return residue_name, residue_id, protein_name


def euclidean_distance(position1: tuple[float, float, float], position2: tuple[float, float, float]):
    import math
    return math.sqrt(sum([(position1[i] - position2[i]) ** 2 for i in range(len(position1))]))


def extract_data(dataset_file_name: str) -> list[tuple[str, int, str, float, float, float]]:
    """
        Using BioPython, reads the provided pdb input file.
        For each amino acid obtains the center of mass, the residue name and the related protein name.

        :param dataset_file_name: the name of the pdb input file
        :returns a list of tuples, with one entry for each amino acid.
            The entry is a tuple containing:
                - the center of mass (x, y, z)
                - the residue name
                - the protein name
    """
    p = PDBParser(PERMISSIVE=True)
    structure = p.get_structure('protein', dataset_file_name)
    out = []

    for chain in structure.get_chains():
        logger.debug("processing chain: " + str(chain))
        for residue in chain:
            if is_hetero(residue):
                logger.debug("skipping residue couse it's heteroatm: " + str(residue))
                continue
            if residue.get_resname() == 'HOH':
                logger.debug("skipping residue: " + str(residue))
                continue
            center_of_mass = residue.center_of_mass()
            residue_name, residue_id, protein_name = get_residue_name_and_protein_name(residue, chain,
                                                                                       dataset_file_name, logger)
            out.append(
                (protein_name, residue_id, residue_name, center_of_mass[0], center_of_mass[1], center_of_mass[2]))

    for index, (protein_name, _, residue_name, center_of_mass_x, center_of_mass_y, center_of_mass_z) in enumerate(out):
        out[index] = (protein_name, index, residue_name, center_of_mass_x, center_of_mass_y, center_of_mass_z)

    """from dotenv import load_dotenv
    load_dotenv()
    import os

    for i in range(len(out), int(os.getenv('MAX_INPUT'))):
        out.append((0, 0, 0, 0, 0, 0))
    """
    return out


def to_one_hot_encoding_input(input_vector: list[tuple[str, int, str]], different_residue_names_index: dict[str, int]) -> \
        tuple[list[list[int]], dict[str, int]]:
    amount_different_residue_names = len(different_residue_names_index.keys())
    residue_name_zero_vector = [0] * amount_different_residue_names

    input_one_hot_encoding = []
    for index, (protein_name, residue_id, residue_name) in enumerate(input_vector):
        residue_name_one_hot_encoding = residue_name_zero_vector.copy()

        if residue_name != 0:
            residue_name_one_hot_encoding[different_residue_names_index[residue_name]] = 1

        input_one_hot_encoding.append(residue_name_one_hot_encoding)

    logger.debug("input_one_hot_encoding: " + str(input_one_hot_encoding))

    return input_one_hot_encoding, different_residue_names_index


def to_one_hot_encoding_input_for_ffnn(preprocessed_chemical_features: dict[str, dict[str, float]],
                                       aminoacid_list: list[tuple[str, int, str]]) -> \
        list[list[float]]:
    ffnn_input_vector_one_hot_encoding = []

    for index, element in enumerate(aminoacid_list):
        aminoacid_name = element[2]
        aminoacid_features = preprocessed_chemical_features.values()
        tmp = []
        ffnn_input_vector_one_hot_encoding.append(tmp)
        for feature_dict in aminoacid_features:
            feature_value = feature_dict[aminoacid_name]
            ffnn_input_vector_one_hot_encoding[index].append(feature_value)

    """
    if aminoacid_name == 0:
        feature_value = 0.0
    else:
        feature_value = feature_dict[aminoacid_name]
    ffnn_input_vector_one_hot_encoding[index].append(feature_value)

    
    from dotenv import load_dotenv
    load_dotenv()
    import os

    for i in range(len(ffnn_input_vector_one_hot_encoding), int(os.getenv('MAX_INPUT'))):
        tmp = []
        for j in range(len(ffnn_input_vector_one_hot_encoding[i])):
            tmp.append(0)
        ffnn_input_vector_one_hot_encoding.append(tmp)
    """
    return ffnn_input_vector_one_hot_encoding


def balance_classes(x_train, y_train):
    logger.debug("x_train: " + str(x_train))
    logger.debug("y_train: " + str(y_train))
    num_of_ones = len(list(filter(lambda x: x == 1, y_train)))
    num_of_zeros = len(list(filter(lambda x: x == 0, y_train)))
    logger.debug("num of ones: " + str(num_of_ones))
    logger.debug("num of zeros: " + str(num_of_zeros))

    if num_of_ones <= 1 or num_of_zeros <= 1:
        return x_train, y_train

    if num_of_ones > num_of_zeros and num_of_zeros < 5:
        smt = SMOTENC(random_state=42, categorical_features=['protein_name', 'residue_name'],
                      k_neighbors=num_of_zeros - 1)
    elif num_of_zeros > num_of_ones and num_of_ones < 5:
        smt = SMOTENC(random_state=42, categorical_features=['protein_name', 'residue_name'],
                      k_neighbors=num_of_ones - 1)
    else:
        smt = SMOTENC(random_state=42, categorical_features=['protein_name', 'residue_name'])
    x_res, y_res = smt.fit_resample(x_train, y_train)
    return x_res, y_res


logger = default_logger(__file__)


def split_data(extract_data: pd.DataFrame, expected_results: pd.DataFrame):
    """
    columns of extract_data ['protein_name', 'residue_id', 'residue_name', 'center_of_mass_x',
                                                 'center_of_mass_y', 'center_of_mass_z'])
    i have to split both extract_data and expected_results in vectors of vectors
    there will be one vector for each protein name, and each vector will contain the data for each residue
    extract_data and expected_results match on row index value
    """
    new_extract_data = []
    new_expected_results = []
    protein_name_index = dict()
    for _, protein_name in enumerate(extract_data['protein_name']):
        if protein_name not in protein_name_index:
            protein_name_index[protein_name] = len(protein_name_index)
            new_extract_data.append([])
            new_expected_results.append([])

    logger.debug("protein_name_index: " + str(protein_name_index))

    for i, element_row in extract_data.iterrows():
        new_extract_data[protein_name_index[element_row['protein_name']]].append(element_row.to_list())
        new_expected_results[protein_name_index[element_row['protein_name']]].append(expected_results[i])

    logger.debug("new_extract_data: " + str(new_extract_data))
    logger.debug("new_expected_results: " + str(new_expected_results))
    return new_extract_data, new_expected_results
