import json
import os
from typing import Dict, Union

from dotenv import load_dotenv
import requests

import utility

logger = utility.default_logger(__file__)


def extract_all_chemical_features(feature_list_file_path: str) -> Dict[str, Dict[str, float]]:
    """
        Read the feature codes from the file feature_list_file_path.
        Extract the chemical features using the function extract_chemical_feature.
        Return a dictionary of dictionaries, where the key is the feature code and the value
        is the dictionary containing the chemical features.
    """
    with open(feature_list_file_path, 'r') as f:
        feature_list = f.readlines()

    feature_list = [feature.strip() for feature in feature_list]
    logger.debug('Feature list: {}'.format(feature_list))
    out = dict()
    for feature in feature_list:
        out[feature] = extract_chemical_feature(feature)
    logger.debug('Out: {}'.format(out))
    return out


def extract_chemical_feature(feature_code: str) -> Dict[str, float]:
    """
        Extract chemical features from the data provided by the website
        reachable using the environment variable 'AAINDEX_WEBSITE_URL'.
        The request should be made to the url AAINDEX_WEBSITE_URL + feature_code.
        The response is a html file.
    """
    load_dotenv()
    request = requests.get(os.environ['AAINDEX_WEBSITE_URL'] + feature_code)
    content = request.text.split('\n')
    logger.debug('Content: {}'.format(content))
    # assuming table header is always the same and does not change
    try:
        start_line_index = [index for index, line in enumerate(content)
                            if line.strip().startswith('I    A/L')][0]
    except IndexError:
        logger.error('The feature code {} is not valid.'.format(feature_code))
        return dict()

    meaningful_lines = content[start_line_index + 1:start_line_index + 3]
    logger.debug('Meaningful lines: {}'.format(meaningful_lines))

    meaningful_lines = ' '.join(meaningful_lines).split(' ')
    meaningful_lines = [line.strip() for line in meaningful_lines if line.strip() != '']
    logger.debug('Meaningful lines: {}'.format(meaningful_lines))

    ordered_names_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                          'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                          'THR', 'TRP', 'TYR', 'VAL']
    out = dict()
    for index, name in enumerate(ordered_names_list):
        out[name] = float(meaningful_lines[index])
    logger.debug('Out: {}'.format(out))
    return out


def dump_to_file(feature_dict: Union[Dict[str, float], Dict[str, Dict[str, float]]], output_file_path: str):
    """
        Dump the data to a file in json format.
    """
    with open(output_file_path, 'w') as f:
        json.dump(feature_dict, f)


def dump_to_file_csv(feature_dict: Union[Dict[str, float], Dict[str, Dict[str, float]]], output_file_path: str):
    """
        Dump the data to a file in csv format.
    """
    with open(output_file_path, 'w') as f:
        if all([isinstance(x, float) for x in feature_dict.values()]):
            f.write('feature_code,feature_value\n')
            for feature_code, feature_value in feature_dict.items():
                f.write('{},{}\n'.format(feature_code, feature_value))
        elif all([isinstance(x, dict) for x in feature_dict.values()]):
            f.write('feature_code,feature_name,feature_value\n')
            for feature_code, feature_dict in feature_dict.items():
                for feature_name, feature_value in feature_dict.items():
                    f.write('{},{},{}\n'.format(feature_code, feature_name, feature_value))


if __name__ == '__main__':
    """
    Requires as parameter the filepath of the pdb input file.
    Prints the list of tuples obtained from the read_data function.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_list_file_path', type=str,
                        help='the path to the file containing the list of feature codes')
    parser.add_argument('--chemical_feature_code', type=str, help='the chemical feature code to extract')
    parser.add_argument('--json_path', type=str, default=None, help='the path to the json file')
    parser.add_argument('--csv_path', type=str, default=None, help='the path to the csv file')
    utility.add_default_parameters(parser)
    parser.add_argument('--output_file_path', type=str, default=None, help='the path to the output file')

    args = parser.parse_args()

    utility.default_logging(args, logger)
    if args.feature_list_file_path is not None:
        data = extract_all_chemical_features(args.feature_list_file_path)
    elif args.chemical_feature_code is not None:
        data = extract_chemical_feature(args.chemical_feature_code)
    else:
        logger.critical('Either feature_list_file_path or chemical_feature_code must be specified.')
        exit(1)
    if args.json_path is not None:
        dump_to_file(data, args.json_path)
    if args.csv_path is not None:
        dump_to_file_csv(data, args.csv_path)
    if args.output_file_path is None:
        print(data)
    else:
        dump_to_file(data, args.output_file_path)
