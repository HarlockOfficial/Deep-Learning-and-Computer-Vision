import json
from typing import List, Tuple

from Bio.PDB import PDBParser

import utility

logger = utility.default_logger(__file__)


def extract_rnn_data(dataset_file_name: str) -> List[Tuple[str, str]]:
    """
        Using BioPython, reads the provided pdb input file.
        For each amino acid obtains the residue name and the related protein name.

        :param dataset_file_name: the name of the pdb input file
        :returns a list of tuples, with one entry for each amino acid.
            The entry is a tuple containing:
                - the residue name
                - the protein name
    """
    p = PDBParser(PERMISSIVE=True)
    structure = p.get_structure('protein', dataset_file_name)
    out = []
    for chain in structure.get_chains():
        for residue in chain:
            logger.debug("processing residue: " + str(residue))
            if residue.get_resname() == 'HOH':
                logger.debug("skipping residue: " + str(residue))
                continue
            residue_name, protein_name = utility.get_residue_name_and_protein_name(residue, chain,
                                                                                   dataset_file_name, logger)
            out.append((residue_name, protein_name))
    return out


def dump_to_file(pdb_data: List[Tuple[str, str]], output_file_path: str):
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

    data = extract_rnn_data(args.pdb_file_path)
    if args.output_file_path is None:
        print(data)
    else:
        dump_to_file(data, args.output_file_path)