import json
import os

from Bio.PDB import PDBParser

import utility

logger = utility.default_logger(__file__)


def dump_to_file(pdb_data: list[tuple[str, int, str]], output_file_path: str):
    """
        Dumps the provided data to the provided output file path.

        :param pdb_data: the data to be dumped
        :param output_file_path: the output file path
    """
    with open(output_file_path, 'w') as f:
        json.dump(pdb_data, f)


def dump_to_file_csv(pdb_data: list[tuple[str, int, str]], output_file_path: str):
    """
        Dumps the provided data to the provided output file path.

        :param pdb_data: the data to be dumped
        :param output_file_path: the output file path
    """
    with open(output_file_path, 'w') as f:
        for protein_name, residue_id, residue_name in pdb_data:
            f.write(protein_name + ',' + str(residue_id) + ',' + residue_name + '\n')


if __name__ == '__main__':
    """
        Requires as parameter the filepath of the pdb input file.
        Prints the list of tuples obtained from the read_data function.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb_file_path', type=str, help='the path to the pdb input file')
    parser.add_argument('--csv', action='store_true', help='if set, the output will be in csv format')
    parser.add_argument('--json', action='store_true', help='if set, the output will be in json format')
    utility.add_default_parameters(parser)
    parser.add_argument('--output_file_path', type=str, default=None, help='the path to the output file')

    args = parser.parse_args()

    utility.default_logging(args, logger)

    data = extract_rnn_data(args.pdb_file_path)
    if args.output_file_path is None:
        print(data)
    else:
        if args.csv:
            dump_to_file_csv(data, args.output_file_path)
        elif args.json:
            dump_to_file(data, args.output_file_path)
        else:
            logger.critical("No output format specified")
            exit(1)
