import json

from Bio.PDB import PDBParser

import utility

logger = utility.default_logger(__file__)


def create_distance_matrix(aminoacid_mass_center: list[tuple[str, int, str, float, float, float]]) -> list[list[tuple[str, int, str, str, int, str, float]]]:
    """
        Creates a distance matrix from the provided list of amino acids.

        :param aminoacid_mass_center: the list of amino acids
        :returns a distance matrix
    """
    out = []
    for i in range(len(aminoacid_mass_center)):
        out.append([])
        for j in range(len(aminoacid_mass_center)):
            out[i].append((aminoacid_mass_center[i][0], aminoacid_mass_center[i][1], aminoacid_mass_center[i][2], aminoacid_mass_center[j][0], aminoacid_mass_center[j][1], aminoacid_mass_center[j][2], utility.euclidean_distance(aminoacid_mass_center[i][3:], aminoacid_mass_center[j][3:])))
    return out


def create_contact_matrix(distance_matrix: list[list[tuple[str, int, str, str, int, str, float]]], threshold: float = 12) -> list[list[int]]:
    """
        Creates a contact matrix from the provided distance matrix.
    """
    out = []
    for i in range(len(distance_matrix)):
        out.append([])
        for j in range(len(distance_matrix)):
            if distance_matrix[i][j][-1] <= threshold:
                out[i].append(1)
            else:
                out[i].append(0)
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


def extract_gnn_data(dataset_file_name: str) -> list[tuple[str, int, str, float, float, float]]:
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
            logger.debug("processing residue: " + str(residue))
            if residue.get_resname() == 'HOH':
                logger.debug("skipping residue: " + str(residue))
                continue
            center_of_mass = residue['CA'].get_coord()
            logger.debug("center of mass: " + str(center_of_mass))
            residue_name, residue_id, protein_name = utility.get_residue_name_and_protein_name(residue, chain,
                                                                                   dataset_file_name, logger)
            out.append((protein_name, residue_id, residue_name, center_of_mass[0], center_of_mass[1], center_of_mass[2]))
    return out


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
