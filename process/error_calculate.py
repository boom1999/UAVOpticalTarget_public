# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/5/4 19:30
# @Software: PyCharm

import torch


def error_calculate(real_data, target_data) -> tuple:
    """
    Calculate the error of the error data and the real data.
    :param real_data: The real data in k system (B_k_ne, L_k_ne, H_k_ne) and in g system (x_g_ne, y_g_ne, z_g_ne).
    :param target_data: The error data in k system (B_k_ne, L_k_ne, H_k_ne) and in g system (x_g_ne, y_g_ne, z_g_ne).
    :return: The error of the predicted data and the real data.
    """
    real_data_k = real_data[0]
    real_data_g = real_data[1]
    target_data_k = target_data[0]
    target_data_g = target_data[1]

    # Error = Target - Real. torch.Size([length, 3])
    error_data = torch.abs(target_data_k[:, :, 0] - real_data_k[0].squeeze(1))

    # compute error in latitude, longitude and altitude
    lat_max = torch.max(error_data[:, 0])
    lat_mean = torch.mean(error_data[:, 0])
    lon_max = torch.max(error_data[:, 1])
    lon_mean = torch.mean(error_data[:, 1])
    alt_max = torch.max(error_data[:, 2])
    alt_mean = torch.mean(error_data[:, 2])
    # compute distance error
    distance = torch.norm(real_data_g - target_data_g, dim=1, p="fro").squeeze(1)
    distance_max = torch.max(distance)
    distance_mean = torch.mean(distance)

    return lat_max, lat_mean, lon_max, lon_mean, alt_max, alt_mean, distance_max, distance_mean, distance
