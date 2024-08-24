# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/7/4 16:06
# @Software: PyCharm

import os
import subprocess
import sys
import time
from datetime import datetime

import torch
from tqdm import tqdm

import model.round_trip as round_trip_model

begin_time = time.time()


# def u_distribution(length):
#     """
#     A U-shaped random distribution with a length of length is generated between -1 and 1,
#     and the probability density is the smallest at 0.
#     :param length: The length of the data.
#     :return: transformed_data: torch.size([length]).
#     """
#     gaussian_data = torch.randn(length)
#     normalized_data = 2 * (gaussian_data - gaussian_data.min()) / (gaussian_data.max() - gaussian_data.min()) - 1
#     positive_data = normalized_data[torch.gt(normalized_data, 0)]
#     negative_data = normalized_data[torch.lt(normalized_data, 0)]
#     transformed_positive = 1 - positive_data
#     transformed_negative = -1 - negative_data
#     transformed_data = torch.cat((transformed_positive, transformed_negative))
#     transformed_data = transformed_data[torch.randperm(transformed_data.shape[0])]
#     return transformed_data


def gaussian_normalized(length):
    """
    A Gaussian distribution with a length of length is generated between -1 and 1,
    :param length: The length of the data.
    :return: transformed_data: torch.size([length]).
    """
    gaussian_data = torch.randn(length)
    normalized_data = 2 * (gaussian_data - gaussian_data.min()) / (gaussian_data.max() - gaussian_data.min()) - 1
    return normalized_data


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Repeat times.
generate_times = round_trip_model.generate_times
# Length of the sequence.
target_times = round_trip_model.target_times

# Phi, Gamma, Theta all zoom into (-1~1).
phi_zoom = torch.FloatTensor(generate_times).uniform_(-1)
gamma_zoom = gaussian_normalized(generate_times)
theta_zoom = gaussian_normalized(generate_times)

# The largest zoom for one angle, use a mixed Gaussian distribution that peaks at both -1 and 1.
# mean1, mean2 = -1, 1
# std1, std2 = 0.3, 0.3
# random_zoom = torch.cat([
#     torch.randn(generate_times // 2) * std1 + mean1,
#     torch.randn(generate_times // 2) * std2 + mean2
# ])
# # Scrambles the scaling distribution.
# random_zoom = random_zoom[torch.randperm(random_zoom.shape[0])]

# Use gaussian distribution.
random_zoom = torch.randn(generate_times)

# Small fixed zooms.
phi_small_fixed_zoom = gaussian_normalized(generate_times)
gamma_small_fixed_zoom = gaussian_normalized(generate_times)
theta_small_fixed_zoom = gaussian_normalized(generate_times)
alpha_small_fixed_zoom = gaussian_normalized(generate_times)
beta_small_fixed_zoom = gaussian_normalized(generate_times)
iterator = tqdm(range(generate_times))
progress_bar = tqdm(iterator, total=generate_times)
n_1, n_2, n_3, n_4, n_5 = None, None, None, None, None
for data_index in progress_bar:
    progress_bar.set_description(f'Generating data {data_index + 1}/{generate_times}')
    attitude_angle_zooms = [phi_zoom[data_index], gamma_zoom[data_index], theta_zoom[data_index]]
    small_fixed_zooms = [phi_small_fixed_zoom[data_index], gamma_small_fixed_zoom[data_index],
                         theta_small_fixed_zoom[data_index], alpha_small_fixed_zoom[data_index],
                         beta_small_fixed_zoom[data_index], ]
    _, _, t_n_1, t_n_2, t_n_3, t_n_4, t_n_5, t_n_1_inverse, t_n_2_inverse, t_n_3_inverse, t_n_4_inverse, t_n_5_inverse,\
        _, _, _, _, _, _, _, _, _, t_k_1, t_k_2, t_k_3, t_k_4, t_k_5, \
        regression_data_1, regression_data_2, regression_data_3, regression_data_4, regression_data_5, _, _, \
        real_k, uav_k = round_trip_model.generate_target_data(attitude_angle_zooms=attitude_angle_zooms,
                                                              small_fixed_zooms=small_fixed_zooms,
                                                              fixed_error_zoom=random_zoom[data_index])
    regression_data_1 = regression_data_1.unsqueeze(2)
    regression_data_2 = regression_data_2.unsqueeze(2)
    regression_data_3 = regression_data_3.unsqueeze(2)
    regression_data_4 = regression_data_4.unsqueeze(2)
    regression_data_5 = regression_data_5.unsqueeze(2)
    real_k = torch.broadcast_to(real_k, (target_times, real_k.shape[0])).unsqueeze(-1)
    t_n_1, t_n_2, t_n_3, t_n_4, t_n_5 = \
        torch.cat((t_n_1, t_n_1_inverse, regression_data_1, t_k_1, real_k, uav_k), dim=1), \
        torch.cat((t_n_2, t_n_2_inverse, regression_data_2, t_k_2, real_k, uav_k), dim=1), \
        torch.cat((t_n_3, t_n_3_inverse, regression_data_3, t_k_3, real_k, uav_k), dim=1), \
        torch.cat((t_n_4, t_n_4_inverse, regression_data_4, t_k_4, real_k, uav_k), dim=1), \
        torch.cat((t_n_5, t_n_5_inverse, regression_data_5, t_k_5, real_k, uav_k), dim=1)
    if data_index == 0:
        n_1, n_2, n_3, n_4, n_5 = t_n_1.unsqueeze(0), t_n_2.unsqueeze(0), t_n_3.unsqueeze(0), \
            t_n_4.unsqueeze(0), t_n_5.unsqueeze(0)
    else:
        n_1, n_2, n_3, n_4, n_5 = \
            torch.cat((n_1, t_n_1.unsqueeze(0)), dim=0), torch.cat((n_2, t_n_2.unsqueeze(0)), dim=0), \
            torch.cat((n_3, t_n_3.unsqueeze(0)), dim=0), torch.cat((n_4, t_n_4.unsqueeze(0)), dim=0), \
            torch.cat((n_5, t_n_5.unsqueeze(0)), dim=0)
label_1 = torch.zeros((generate_times, target_times, 1, 1))
label_2, label_3, label_4, label_5 = label_1 + 1, label_1 + 2, label_1 + 3, label_1 + 4
n_1, n_2, n_3, n_4, n_5 = torch.cat((n_1, label_1), dim=2), torch.cat((n_2, label_2), dim=2), \
    torch.cat((n_3, label_3), dim=2), torch.cat((n_4, label_4), dim=2), torch.cat((n_5, label_5), dim=2)
# every n data information:
# # Shape: (generate_times, target_times, 32, 1)
# # dim 0: generate_times.
# # dim 1: target_times.
# # dim 2: 27-dimensional data :
# # # # # [0:3]   lat_n_relative(target_x), lon_n_relative(target_y), alt_n_relative(target_z)
# # # # # [3:6]   lat_n_relative(target_x), lon_n_relative(target_y), alt_n_relative(target_z) for inverse mirror
# # # # # [6:11]  phi, gamma, theta, alpha, beta
# # # # # [11:13] alpha_inverse, beta_inverse
# # # # # [13:18] phi_delta, gamma_delta, theta_delta, alpha_delta, beta_delta (in deg)
# # # # # [18:22] laser_distance, uav_coordinate_x, uav_coordinate_y, uav_coordinate_z,
# # # # # [22:25] The coordinates of the original positioning target point in k system lat, lon, height
# # # # # [25:28] real coordinates in k system lat, lon, height
# # # # # [28:31] uav coordinates in k system lat, lon, height
# # # # # [31]    label
# # dim 3: 1.
concatenated_data = torch.cat((n_1, n_2, n_3, n_4, n_5), dim=0)
# Randomly shuffle the data.
shuffled_data = concatenated_data[torch.randperm(concatenated_data.size()[0])]
# Save the data.
dir_path = os.path.abspath(os.path.join(os.getcwd().split('OpticalTarget')[0], 'OpticalTarget', 'data', 'pt'))
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
file_name = f'target_data_{timestamp}.pt'
torch.save(shuffled_data, os.path.join(dir_path, file_name))

# Split dataset.
python_exe_path = sys.executable
script_path = os.path.abspath(os.path.join(os.getcwd().split('OpticalTarget')[0],
                                           'OpticalTarget', 'process', 'split_dataset.py'))
script_result = subprocess.run([python_exe_path, script_path])
if script_result.returncode != 0:
    print('Split dataset failed.')
    exit(1)
else:
    print('Split dataset succeeded.')

end_time = time.time()
print("Total time elapsed: {:.4f}s".format(end_time - begin_time))
torch.cuda.empty_cache()
