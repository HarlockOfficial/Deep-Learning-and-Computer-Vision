import json
import os.path
from typing import List, Tuple

from Bio.PDB import PDBParser

import sys
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_logging import Logger

logger = Logger()


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
    for model in structure:
        logger.debug("processing model: " + str(model))
        for chain in model:
            logger.debug("processing chain: " + str(chain))
            for residue in chain:
                logger.debug("processing residue: " + str(residue))
                if residue.get_resname() == 'HOH':
                    logger.debug("skipping residue: " + str(residue))
                    continue
                center_of_mass = residue['CA'].get_coord()
                logger.debug("center of mass: " + str(center_of_mass))
                residue_name = residue.get_resname()
                logger.debug("residue name: " + str(residue_name))
                protein_name = dataset_file_name.split('/')[-1].split('.')[0]
                logger.debug("protein name: " + str(protein_name))
                out.append((center_of_mass[0], center_of_mass[1], center_of_mass[2], residue_name, protein_name))
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
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    parser.add_argument('--debug_log_file_path', type=str, default='pdb_data/logs/debug.log', help='the path to the debug log file')
    parser.add_argument('--verbose', action='store_true', help='enable verbose mode')
    parser.add_argument('--verbose_log_file_path', type=str, default='pdb_data/logs/verbose.log', help='the path to the verbose log file')
    parser.add_argument('--warning', action='store_true', help='enable warning mode')
    parser.add_argument('--warning_log_file_path', type=str, default='pdb_data/logs/warning.log', help='the path to the warning log file')
    parser.add_argument('--error', action='store_true', help='enable error mode')
    parser.add_argument('--error_log_file_path', type=str, default='pdb_data/logs/error.log', help='the path to the error log file')
    parser.add_argument('--critical', action='store_true', help='enable critical mode')
    parser.add_argument('--critical_log_file_path', type=str, default='pdb_data/logs/critical.log', help='the path to the critical log file')
    parser.add_argument('--output_file_path', type=str, default=None, help='the path to the output file')

    args = parser.parse_args()
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

    data = extract_gnn_data(args.pdb_file_path)
    if args.output_file_path is None:
        print(data)
    else:
        dump_to_file(data, args.output_file_path)
