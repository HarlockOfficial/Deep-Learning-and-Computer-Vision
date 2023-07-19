import json

import pandas
from Bio.PDB import PDBParser

import utility
import tensorflow as tf

logger = utility.default_logger(__file__)


def create_distance_matrix(aminoacid_mass_center: pandas.DataFrame) -> list[list[tuple[str, int, str, str, int, str, float]]]:
    """
        Creates a distance matrix from the provided list of amino acids.

        :param aminoacid_mass_center: the list of amino acids
        :returns a distance matrix
    """
    out = []
    for i, row_i in aminoacid_mass_center.iterrows():
        out.append([])
        for j, row_j in aminoacid_mass_center.iterrows():
            out[i].append((row_i['protein_name'], row_i['residue_id'],
                           row_i['residue_name'],
                           utility.euclidean_distance((row_i['center_of_mass_x'], row_i['center_of_mass_y'], row_i['center_of_mass_z']),
                                                      (row_j['center_of_mass_x'], row_j['center_of_mass_y'], row_j['center_of_mass_z']))))
    return out


def create_contact_matrix(distance_matrix: list[list[tuple[str, int, str, str, int, str, float]]], threshold: float = 12) -> tf.Tensor:
    """
        Creates a contact matrix from the provided distance matrix.
    """

    outer = []

    for i in range(len(distance_matrix)):
        outer.append([])
        for j in range(len(distance_matrix)):
            if distance_matrix[i][j][-1] <= threshold:
                outer[i].append(1)
            else:
                outer[i].append(0)

    out = tf.constant(value=outer, shape=(len(distance_matrix), len(distance_matrix)), dtype=tf.int32)
    #sparse = tf.sparse.from_dense(out)
    return out


def dump_to_file_matrix(matrix: list[list[float]], output_file_path: str):
    """
        Dumps the provided data to the provided output file path.

        :param matrix: the data to be dumped
        :param output_file_path: the output file path
    """
    with open(output_file_path, 'w') as f:
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                f.write(str(matrix[i][j]) + ",")
            f.write("\n")


def dump_to_file(pdb_data: list[tuple[str, int, str, float, float, float]], output_file_path: str):
    """
        Dumps the provided data to the provided output file path.

        :param pdb_data: the data to be dumped
        :param output_file_path: the output file path
    """
    with open(output_file_path, 'w') as f:
        json.dump(pdb_data, f)


def dump_to_file_csv(pdb_data: list[tuple[str, int, str, float, float, float]], output_file_path: str):
    """
        Dumps the provided data to the provided output file path.

        :param pdb_data: the data to be dumped
        :param output_file_path: the output file path
    """
    with open(output_file_path, 'w') as f:
        for entry in pdb_data:
            f.write(str(entry[0]) + "," + str(entry[1]) + "," + str(entry[2]) + "," + str(entry[3]) + "," + str(entry[4]) + "," + str(entry[5]) + "\n")


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
    parser.add_argument('--csv_distance_matrix_path', type=str, default=None, help='the path to the output file')
    parser.add_argument('--csv_contact_matrix_path', type=str, default=None, help='the path to the output file')
    utility.add_default_parameters(parser)
    parser.add_argument('--output_file_path', type=str, default=None, help='the path to the output file')

    args = parser.parse_args()

    utility.default_logging(args, logger)

    data = extract_gnn_data(args.pdb_file_path)
    if args.csv_distance_matrix_path is not None:
        dump_to_file_matrix(create_distance_matrix(data), args.csv_distance_matrix_path)
    if args.csv_contact_matrix_path is not None:
        dump_to_file_matrix(create_contact_matrix(create_distance_matrix(data)), args.csv_contact_matrix_path)

    if args.output_file_path is None:
        print(data)
    else:
        if args.csv:
            dump_to_file_csv(data, args.output_file_path)
        elif args.json:
            dump_to_file(data, args.output_file_path)
        else:
            logger.critical("no output format specified")
            exit(1)
