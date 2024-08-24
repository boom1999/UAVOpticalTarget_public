# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/5/3 19:38
# @Software: PyCharm

import glob
import os
import re
from typing import Tuple

import torch


def load_data(data_type: str) -> Tuple[torch.Tensor, str]:
    """
    Load positioning data.
    :param data_type:
    :return: p_position in tensor, latest_file_path in str
    """
    # Loading positioning data.
    # Set the directory path and pattern for the file names.
    dir_path = os.path.join(os.getcwd().split('OpticalTarget')[0], 'OpticalTarget', 'data', 'pt')
    if not os.path.exists(dir_path):
        raise Exception('The .pt data directory does not exist.')
    pattern = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}'
    if data_type == 'full_err':
        # Get a list of all .pt files in the directory and its subdirectories that match the pattern.
        files = glob.glob(os.path.join(dir_path, '**', 'full_err_*.pt'), recursive=True)
        if not files:
            # if the list is empty, raise a custom exception.
            raise Exception('No matching full-err files found')
        files.sort(key=lambda full_err: re.search(pattern, full_err).group(), reverse=True)
        # Select the latest location file.
        latest_file_path = files[0]
        # full-err location data
        p_position = torch.load(latest_file_path)
        return p_position, latest_file_path

    elif data_type == 'no_err':
        files_ne = glob.glob(os.path.join(dir_path, '**', 'no_err_*.pt'), recursive=True)
        if not files_ne:
            # if the list is empty, raise a custom exception.
            raise Exception('No matching no-err files found')
        files_ne.sort(key=lambda full_err: re.search(pattern, full_err).group(), reverse=True)
        # Select the latest location file.
        latest_file_path_ne = files_ne[0]

        # err-free location data
        p_position_ne = torch.load(latest_file_path_ne)
        return p_position_ne, latest_file_path_ne

    elif data_type == 'round_trip':
        files_rt = glob.glob(os.path.join(dir_path, '**', 'round_trip_*.pt'), recursive=True)
        if not files_rt:
            # if the list is empty, raise a custom exception.
            raise Exception('No matching round-trip files found')
        files_rt.sort(key=lambda full_err: re.search(pattern, full_err).group(), reverse=True)
        # Select the latest location file.
        latest_file_path_rt = files_rt[0]

        # round-trip location data
        p_position_rt = torch.load(latest_file_path_rt)
        return p_position_rt, latest_file_path_rt
    elif data_type == 'target_data':
        files_td = glob.glob(os.path.join(dir_path, '**', 'target_data_*.pt'), recursive=True)
        if not files_td:
            # if the list is empty, raise a custom exception.
            raise Exception('No matching target-data files found')
        files_td.sort(key=lambda full_err: re.search(pattern, full_err).group(), reverse=True)
        # Select the latest location file.
        latest_file_path_td = files_td[0]

        # target-data location data
        p_position_td = torch.load(latest_file_path_td)
        return p_position_td, latest_file_path_td
    else:
        raise Exception('No matching files found')
