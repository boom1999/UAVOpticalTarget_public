# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/7/5 15:05
# @Software: PyCharm

import os

import torch


def load_data_fold(fold_num):
    """
    Load positioning data.
    :param fold_num: the number of fold: 0, 1, 2, 3, 4,
    :return: train_data_list, valid_data_list, test_data_list
    """
    dir_path = os.path.join(os.getcwd().split('OpticalTarget')[0], 'OpticalTarget', 'data', 'fold_data')
    if not os.path.exists(dir_path):
        raise Exception('The dataset directory does not exist, please check if you split it or not.')
    train_data = torch.load(os.path.join(dir_path, 'train_data_fold' + str(fold_num) + '.pt'))
    valid_data = torch.load(os.path.join(dir_path, 'valid_data_fold' + str(fold_num) + '.pt'))
    test_data = torch.load(os.path.join(dir_path, 'test_data_fold' + str(fold_num) + '.pt'))
    return train_data.squeeze(3), valid_data.squeeze(3), test_data.squeeze(3)
