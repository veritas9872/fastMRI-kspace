from pathlib import Path
import argparse
import logging
import time
import json


def initialize(ckpt_dir):
    ckpt_path = Path(ckpt_dir)  # If a string is entered, convert to Path object.
    if not ckpt_path.exists():
        print('Making checkpoint directory: ', ckpt_path)
        try:
            ckpt_path.mkdir()
        except OSError as e:
            print(e)
            print('Could not make Checkpoint root directory')

    run_number = sum([ckpt.is_dir() for ckpt in ckpt_path.iterdir()]) + 1
    time_string = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))
    run_name = f'Trial {run_number:02d}  {time_string}'

    try:
        run_ckpt_path = ckpt_path / run_name
        run_ckpt_path.mkdir()
        print(f'Created Checkpoint Directory {run_ckpt_path}')
    except OSError as e:
        print(e)
        print('Could not make Checkpoint run directory')

    print('Starting', run_name)

    return run_number, run_name


def save_dict_as_json(dict_data, log_dir, save_name):
    file_dir = Path(log_dir, f'{save_name}.json')
    with open(file_dir, mode='w') as jf:
        json.dump(dict_data, jf, indent=2, sort_keys=True)


def get_logger(name, save_file=None):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    for handler in logger.handlers:  # Initialize existing logger.
        logger.removeHandler(handler)

    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    if save_file:
        f_handler = logging.FileHandler(str(save_file) + '.log', mode='w')
        f_handler.setLevel(logging.INFO)
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    return logger


# def create_logger(name, save_folder=None, save_name='event_log.log'):
#     """
#     Modified logger created to save all outputs onto a single file.
#     The file being saved now has a fixed name, allowing all
#     :param name:
#     :param save_folder:
#     :param save_name:
#     :return:
#     """
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.INFO)
#
#     # Handler for display to stdout.
#     c_handler = logging.StreamHandler()
#     c_handler.setLevel(logging.INFO)
#     c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
#     c_handler.setFormatter(c_format)
#     logger.addHandler(c_handler)
#
#     if save_folder and save_name:  # Both variables must be present to write a log file to disk.
#         save_path = Path(save_folder)
#         assert save_path.exists(), 'Specified path does not exist.'
#         save_path = save_path / save_name
#
#         f_handler = logging.FileHandler(save_path, mode='x')
#         f_handler.setLevel(logging.INFO)
#         f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         f_handler.setFormatter(f_format)
#         logger.addHandler(f_handler)
#
#     return logger






def create_arg_parser(**overrides):
    parser = argparse.ArgumentParser(description='Simple argument parser for placing default arguments as desired.')
    parser.set_defaults(**overrides)
    return parser
