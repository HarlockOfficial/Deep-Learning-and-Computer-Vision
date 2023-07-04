import json
from typing import List, Tuple

from Bio.PDB import PDBParser

import utility

logger = utility.default_logger(__file__)


def create_distance_matrix(aminoacid_mass_center: List[Tuple[float, float, float, str, str]]) -> List[List[float]]:
    """
        Creates a distance matrix from the provided list of amino acids.

        :param aminoacid_mass_center: the list of amino acids
        :returns a distance matrix
    """
    out = []
    for i in range(len(aminoacid_mass_center)):
        out.append([])
        for j in range(len(aminoacid_mass_center)):
            out[i].append(utility.euclidean_distance(aminoacid_mass_center[i][:3], aminoacid_mass_center[j][:3]))
    return out


def create_contact_matrix(distance_matrix: List[List[float]], threshold: float = 12) -> List[List[int]]:
    """
        Creates a contact matrix from the provided distance matrix.
    """
    out = []
    for i in range(len(distance_matrix)):
        out.append([])
        for j in range(len(distance_matrix)):
            if distance_matrix[i][j] <= threshold:
                out[i].append(1)
            else:
                out[i].append(0)
    return out


def extract_gnn_data(dataset_file_name: str) -> List[Tuple[float, float, float, str, str]]:
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
            residue_name, protein_name = utility.get_residue_name_and_protein_name(residue, chain,
                                                                                   dataset_file_name, logger)
            out.append((center_of_mass[0], center_of_mass[1], center_of_mass[2],
                        residue_name, protein_name))
    return out


def dump_to_file(pdb_data: List[Tuple[float, float, float, str, str]], output_file_path: str):
    """
        Dumps the provided data to the provided output file path.

        :param pdb_data: the data to be dumped
        :param output_file_path: the output file path
    """
    with open(output_file_path, 'w') as f:
        json.dump(pdb_data, f)


if __name__ == '__main__':
    """
    Requires as parameter the filepath of the pdb input file.
    Prints the list of tuples obtained from the read_data function.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb_file_path', type=str, help='the path to the pdb input file')
    utility.add_default_parameters(parser)
    parser.add_argument('--output_file_path', type=str, default=None, help='the path to the output file')

    args = parser.parse_args()

    utility.default_logging(args, logger)

    data = extract_gnn_data(args.pdb_file_path)
    if args.output_file_path is None:
        print(data)
    else:
        dump_to_file(data, args.output_file_path)
