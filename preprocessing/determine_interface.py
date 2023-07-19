# computes the truth value for every residue in the pdb files
import json
import os
import re
from glob import glob
from typing import Union

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


def get_binding_partner_id(partner_pdb_id):
    partner_id = partner_pdb_id.split("_")
    partner_id[-1] = "ag" if partner_id[-1] == "ab" else "ab"
    return "_".join(partner_id)


def compute_interacting_interface(folder_path: str, pdb_parser: PDBParser, interaction_distance: float) \
        -> Union[None, dict[str, list[tuple[str, int, str, int]]]]:
    out = dict()
    if not os.path.exists(folder_path):
        logger.info(f"Folder {folder_path} does not exist!")
        return None
    for pdb_file_1 in sorted(glob(f"{folder_path}/*.pdb")):
        pdb_id_1 = re.sub("\.pdb$", "", os.path.basename(pdb_file_1))
        if pdb_id_1 not in out.keys():
            out[pdb_id_1] = list()
        out[pdb_id_1].append(compute_interacting_interface_for_single_file(pdb_file_1, pdb_parser, interaction_distance))
    return out


def dump_to_file(determined_interface: Union[
    list[tuple[str, int, str, int]], dict[str, list[tuple[str, int, str, int]]]],
                 destination_path: str):
    """
        Dumps the determined interface to a json file.
    """
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    if isinstance(determined_interface, dict):
        for interface_dataset, dataset_items in determined_interface.items():
            with open(f"{destination_path}/{interface_dataset}_interface.json", "w") as f:
                json.dump(dataset_items, f)
    else:
        with open(f"{destination_path}/interface.json", "w") as f:
            json.dump(determined_interface, f)


def dump_to_file_csv(determined_interface: Union[
    list[tuple[str, int, str, int]], dict[str, list[tuple[str, int, str, int]]]],
                     destination_path: str):
    """
        Dumps the determined interface to a csv file.
    """
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    if isinstance(determined_interface, dict):
        for interface_dataset, dataset_items in determined_interface.items():
            with open(f"{destination_path}/{interface_dataset}_interface.csv", "w") as f:
                f.write("chain_id,residue_id,residue_name,is_interface\n")
                for row in dataset_items:
                    f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")
    else:
        with open(f"{destination_path}/interface.csv", "w") as f:
            f.write("chain_id,residue_id,residue_name,is_interface\n")
            for row in determined_interface:
                f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")


def compute_interacting_interface_for_single_file(pdb_path: str, pdb_parser: PDBParser, interaction_distance: float = 6.0):
    logger.info(f"Computing interacting interface for {pdb_path}.")

    pdb_id_1 = re.sub("\.pdb$", "", os.path.basename(pdb_path))
    folder_path = os.path.dirname(pdb_path)
    pdb_id_2 = get_binding_partner_id(pdb_id_1)

    logger.info(f"Computing interacting interface for {pdb_id_1} - {pdb_id_2}.")

    pdb_file_2 = f"{folder_path}/{pdb_id_2}.pdb"

    logger.info(f"Computing interacting interface for {pdb_file_2}.")

    p1_structure = pdb_parser.get_structure(pdb_id_1, pdb_path)
    p2_structure = pdb_parser.get_structure(pdb_id_2, pdb_file_2)

    p1_heavy_atoms = [atom for atom in p1_structure.get_atoms() if not is_hydrogen(atom)]
    p2_heavy_atoms = [atom for atom in p2_structure.get_atoms() if not is_hydrogen(atom)]

    p1_tree = NeighborSearch(p1_heavy_atoms)
    p1_interface = set()

    for atom in p2_heavy_atoms:
        logger.debug(str(type(interaction_distance)) + " " + str(interaction_distance))
        nb_list = p1_tree.search(atom.coord, interaction_distance, "R")
        p1_interface = p1_interface.union(nb_list)

    if not p1_interface:
        logger.info(f"No interface residues for PDB id {pdb_id_1}, in folder {folder_path}")
    out = []
    for residue in p1_structure.get_residues():
        if is_hetero(residue):
            continue
        is_interface = 1 if residue in p1_interface else 0
        out.append(is_interface)

    """from dotenv import load_dotenv
    load_dotenv()

    for i in range(len(out), int(os.getenv('MAX_INPUT'))):
        out.append(0)

    logger.debug("Len out: " + str(len(out)))
    """
    return out


def compute_interface(interaction_distance: float = 6.0, pdb_path: str = None) -> \
        Union[None, list[int]]:
    pdb_parser = PDBParser(QUIET=True, PERMISSIVE=True)
    if pdb_path is None:
        logger.critical(
            "No dataset list provided. Please provide a list of datasets to compute the interacting interface for.")
        exit(1)
    logger.info(f"Computing interacting interface for {pdb_path} dataset.")
    result = compute_interacting_interface_for_single_file(pdb_path, pdb_parser, interaction_distance)
    return result


def compute_all_interfaces(interaction_distance: float = 6.0, dataset_list: list[str] = None) -> \
        Union[None, dict[str, dict[str, list[tuple[str, int, str, int]]]]]:
    pdb_parser = PDBParser(QUIET=True, PERMISSIVE=True)
    if dataset_list is None:
        logger.critical(
            "No dataset list provided. Please provide a list of datasets to compute the interacting interface for.")
        return None
    logger.info("Computing interacting interface for all PDB files.")
    out = dict()
    for curr_dataset in dataset_list:  # for each dataset partition
        logger.info(f"Computing interacting interface for {curr_dataset} dataset.")
        result = compute_interacting_interface(f"data/{curr_dataset}", pdb_parser, interaction_distance)
        if result is None:
            result = dict()
        out[curr_dataset] = result
    return out


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Computes the interacting interface for all PDB files.")
    arg_parser.add_argument("--interaction_distance", type=float, default=6.0)
    arg_parser.add_argument("--dataset_list", type=str, nargs="+", default=["training", "testing", "validation"])
    arg_parser.add_argument("--json_paths", type=str, nargs="+",
                            default=["data/training/interacting_interface/", "data/testing/interacting_interface/",
                                     "data/validation/interacting_interface/"])
    arg_parser.add_argument("--csv_paths", type=str, nargs="+",
                            default=["data/training/interacting_interface/", "data/testing/interacting_interface/",
                                     "data/validation/interacting_interface/"])
    utility.add_default_parameters(arg_parser)
    args = arg_parser.parse_args()
    utility.default_logging(args, logger)

    interfaces_result = compute_all_interfaces(args.interaction_distance, args.dataset_list)
    for dataset, items in interfaces_result.items():
        dataset_index = args.dataset_list.index(dataset)
        if len(args.json_paths) > dataset_index:
            for pdb_id, interface in items.items():
                dump_to_file(interface, args.json_paths[dataset_index] + pdb_id)
        if len(args.csv_paths) > dataset_index:
            for pdb_id, interface in items.items():
                dump_to_file_csv(interface, args.csv_paths[dataset_index] + pdb_id)
