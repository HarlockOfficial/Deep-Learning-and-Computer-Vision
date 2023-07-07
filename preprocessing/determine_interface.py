# computes the truth value for every residue in the pdb files
import gzip
import os
import re
from glob import glob

from Bio.PDB import NeighborSearch
from Bio.PDB.PDBParser import PDBParser

import utility

logger = utility.default_logger(__file__)

# regex used to distinguish hydrogen atoms
_hydrogen = re.compile("[123 ]*H.*")


def is_hydrogen(atm):
    return _hydrogen.match(atm.get_id())


def is_hetero(res):
    return res.get_full_id()[3][0] != ' '


def get_binding_partner_id(pdb_id):
    partner_id = pdb_id.split("_")
    partner_id[-1] = "ag" if partner_id[-1] == "ab" else "ab"
    return "_".join(partner_id)


def main(interaction_distance: float = 6.0):
    pdb_parser = PDBParser(QUIET=True, PERMISSIVE=True)
    logger.info("Computing interacting interface for all PDB files.")
    for dataset in ["testing", "training", "validation"]:  # for each dataset partition
        logger.info(f"Computing interacting interface for {dataset} dataset.")
        folder_path = f"data/{dataset}/"
        if not os.path.exists(folder_path):
            logger.info(f"Folder {folder_path} does not exist!")
            continue
        for pdb_file_1 in sorted(glob(f"{folder_path}/*.pdb")):
            logger.info(f"Computing interacting interface for {pdb_file_1}.")
            pdb_id_1 = re.sub("\.pdb$", "", os.path.basename(pdb_file_1))
            pdb_id_2 = get_binding_partner_id(pdb_id_1)
            logger.info(f"Computing interacting interface for {pdb_id_1} - {pdb_id_2}.")
            pdb_file_2 = f"data/{dataset}/{pdb_id_2}.pdb"
            logger.info(f"Computing interacting interface for {pdb_file_2}.")

            p1_structure = pdb_parser.get_structure(pdb_id_1, pdb_file_1)
            p2_structure = pdb_parser.get_structure(pdb_id_2, pdb_file_2)

            p1_heavy_atoms = [atom for atom in p1_structure.get_atoms() if not is_hydrogen(atom)]
            p2_heavy_atoms = [atom for atom in p2_structure.get_atoms() if not is_hydrogen(atom)]

            p1_tree = NeighborSearch(p1_heavy_atoms)
            p1_interface = set()

            for atom in p2_heavy_atoms:
                nb_list = p1_tree.search(atom.coord, interaction_distance, "R")
                p1_interface = p1_interface.union(nb_list)

            if not p1_interface:
                logger.info(f"No interface residues for PDB id {pdb_id_1}, {dataset} dataset!")

            destination_path = f"data/{dataset}/interaction_interface"
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)
            with gzip.open(f"{destination_path}/{pdb_id_1}_interface.txt.gz", "wt") as out_p1:
                for residue in p1_structure.get_residues():
                    if is_hetero(residue):
                        continue
                    chain_id = residue.get_full_id()[2]
                    residue_id = residue.get_full_id()[3][1]
                    residue_insertion_code = residue.get_full_id()[3][2]
                    residue_name = residue.get_resname()
                    is_interface = 1 if residue in p1_interface else 0
                    out_p1.write("%s,%s,%s,%s,%d\n" % (chain_id, residue_id, residue_insertion_code,
                                                       residue_name, is_interface))


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser(description="Computes the interacting interface for all PDB files.")
    arg_parser.add_argument("--interaction_distance", type=float, default=6.0)
    utility.add_default_parameters(arg_parser)
    args = arg_parser.parse_args()
    utility.default_logging(args, logger)

    main(args.interaction_distance)
