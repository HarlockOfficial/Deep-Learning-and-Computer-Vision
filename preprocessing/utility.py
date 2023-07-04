import argparse


def add_default_parameters(parser: argparse.ArgumentParser):
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    parser.add_argument('--debug_log_file_path', type=str, default='pdb_data/logs/debug.log',
                        help='the path to the debug log file')
    parser.add_argument('--verbose', action='store_true', help='enable verbose mode')
    parser.add_argument('--verbose_log_file_path', type=str, default='pdb_data/logs/verbose.log',
                        help='the path to the verbose log file')
    parser.add_argument('--warning', action='store_true', help='enable warning mode')
    parser.add_argument('--warning_log_file_path', type=str, default='pdb_data/logs/warning.log',
                        help='the path to the warning log file')
    parser.add_argument('--error', action='store_true', help='enable error mode')
    parser.add_argument('--error_log_file_path', type=str, default='pdb_data/logs/error.log',
                        help='the path to the error log file')
    parser.add_argument('--critical', action='store_true', help='enable critical mode')
    parser.add_argument('--critical_log_file_path', type=str, default='pdb_data/logs/critical.log',
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
