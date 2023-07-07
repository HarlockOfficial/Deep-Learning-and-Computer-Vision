import argparse
from typing import Tuple


def add_default_parameters(parser: argparse.ArgumentParser):
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    parser.add_argument('--debug_log_file_path', type=str, default='data/logs/debug.log',
                        help='the path to the debug log file')
    parser.add_argument('--verbose', action='store_true', help='enable verbose mode')
    parser.add_argument('--verbose_log_file_path', type=str, default='data/logs/verbose.log',
                        help='the path to the verbose log file')
    parser.add_argument('--warning', action='store_true', help='enable warning mode')
    parser.add_argument('--warning_log_file_path', type=str, default='data/logs/warning.log',
                        help='the path to the warning log file')
    parser.add_argument('--error', action='store_true', help='enable error mode')
    parser.add_argument('--error_log_file_path', type=str, default='data/logs/error.log',
                        help='the path to the error log file')
    parser.add_argument('--critical', action='store_true', help='enable critical mode')
    parser.add_argument('--critical_log_file_path', type=str, default='data/logs/critical.log',
                        help='the path to the critical log file')


def default_logging(args, logger):
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


def default_logger(file):
    import os
    import sys

    if os.path.dirname(os.path.dirname(os.path.abspath(file))) not in sys.path:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(file))))

    from my_logging import Logger

    return Logger()


def get_residue_name_and_protein_name(residue, chain, dataset_file_name, logger):
    residue_name = residue.get_resname()
    logger.debug("residue name: " + str(residue_name))
    protein_name = dataset_file_name.split('/')[-1].split('.')[0][:-1] + chain.get_id()
    logger.debug("protein name: " + str(protein_name))
    residue_id = residue.get_full_id()[3][1]
    logger.debug("residue id: " + str(residue_id))
    return residue_name, residue_id, protein_name


def euclidean_distance(position1: Tuple[float, float, float], position2: Tuple[float, float, float]):
    import math
    return math.sqrt(sum([(position1[i] - position2[i]) ** 2 for i in range(len(position1))]))
