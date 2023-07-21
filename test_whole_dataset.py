import os

import main
import preprocessing.utility as utility

logger = utility.default_logger(__file__)


def test(pdb_folder_path: str, chemical_features_path: str, interaction_distance: float = 6.0, output_path=None):
    """
        For each pdb file in the pdb folder, it will train the model.
    """
    logger.info("Starting the testing")

    if output_path is not None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)


    ordered_names_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                          'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                          'THR', 'TRP', 'TYR', 'VAL']

    preprocessed_chemical_features = None
    different_residue_names_index = dict()
    for index, x in enumerate(ordered_names_list):
        different_residue_names_index[x] = index

    for pdb_file in os.listdir(pdb_folder_path):
        if pdb_file.endswith(".pdb"):
            pdb_path = pdb_folder_path + "/" + pdb_file
            logger.info("Training for file: " + pdb_path)
            rnn_model, gnn_model, ffnn_model, different_protein_names_index, _, \
            aminoacid_list, preprocessed_chemical_features = main.\
                test_whole_network_on_a_file(pdb_path, chemical_features_path, interaction_distance,
                                             preprocessed_chemical_features, output_path,
                                             different_residue_names_index = different_residue_names_index)
            logger.info("Testing for file: " + pdb_path + " finished")
    logger.info('Testing finished')


if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("pdb_folder_path", help="Path to the pdb folder")
    args.add_argument("chemical_features_path", help="Path to the chemical features file")
    args.add_argument("--interaction_distance", default=6.0, type=float, help="Interaction distance")
    args.add_argument("-o", "--output", default=None, help="Path to the output folder")
    utility.add_default_parameters(args)

    args = args.parse_args()

    utility.default_logging(args, logger)

    test(args.pdb_folder_path, args.chemical_features_path, args.interaction_distance, args.output)
    logger.info("Training finished")
