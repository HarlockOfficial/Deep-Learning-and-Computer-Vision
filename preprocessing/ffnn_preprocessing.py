import json
import os
from typing import Dict

from dotenv import load_dotenv
import requests

import preprocessing.utility as utility

logger = utility.default_logger(__file__)


def extract_chemical_features(feature_code: str) -> Dict[str, float]:
    """
        Extract chemical features from the data provided by the website
        reachable using the envirnonment variable 'AAINDEX_WEBSITE_URL'.
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


def dump_to_file(feature_dict: Dict[str, float], output_file_path: str):
    """
        Dump the data to a file named feature_code.txt.
    """
    with open(output_file_path, 'w') as f:
        f.write(json.dumps(feature_dict))


if __name__ == '__main__':
    """
    Requires as parameter the filepath of the pdb input file.
    Prints the list of tuples obtained from the read_data function.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('chemical_feature_code', type=str, help='the chemical feature code to extract')
    utility.add_default_parameters(parser)
    parser.add_argument('--output_file_path', type=str, default=None, help='the path to the output file')

    args = parser.parse_args()

    utility.default_logging(args, logger)

    data = extract_chemical_features(args.chemical_feature_code)
    if args.output_file_path is None:
        print(data)
    else:
        dump_to_file(data, args.output_file_path)
