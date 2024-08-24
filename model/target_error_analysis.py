# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/4/26 16:10
# @Software: PyCharm

import os
import time
from datetime import datetime

import torch

import config.parameters_init as parameters_init
import process.rotation as rotation
from process import error_calculate

# -------------------------- 0. Default setting. --------------------------

times = parameters_init.times
length = parameters_init.length

begin_time = time.time()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- 1. When there is no error. --------------------------

p_k_ne, \
    phi_b_ne, gamma_b_ne, theta_b_ne, \
    phi_delta_i_b_ne, gamma_delta_i_b_ne, theta_delta_i_b_ne, \
    phi_delta_v_b_ne, gamma_delta_v_b_ne, theta_delta_v_b_ne, \
    alpha_c_ne, beta_c_ne, R = parameters_init.parameter_initialization_no_err(length)

# Error-free laser ranging distance.
t_c_ne = torch.zeros(length, 3, 1, dtype=torch.float64)
t_c_ne[:, 2, :] = R.unsqueeze(1)

# Error-free UAV location position parameters.
B_ne = p_k_ne[0]  # lat
L_ne = p_k_ne[1]  # lon
H_ne = p_k_ne[2]  # height

# NOTE: t --> target, p --> UAV, t_*_ne --> * system, _ne --> no error, _e --> error.
p_g_ne = rotation.k_2_g(B_ne, L_ne, H_ne)

# When there is no error, t_b_ne = t_t_ne.
t_t_ne = rotation.c_2_t(t_c_ne, alpha_c_ne, beta_c_ne)
t_b_ne = rotation.t_2_b(t_t_ne, phi_delta_i_b_ne, gamma_delta_i_b_ne, theta_delta_i_b_ne,
                        phi_delta_v_b_ne, gamma_delta_v_b_ne, theta_delta_v_b_ne)

t_n_ne = rotation.b_2_n(t_b_ne, phi_b_ne, gamma_b_ne, theta_b_ne)
t_g_ne = rotation.n_2_g(t_n_ne, B_ne, L_ne, p_g_ne)
t_k_ne = rotation.g_2_k(t_g_ne)

# Compute the error between systems rotation and laser ranging distance.
err_real_laser = (R - torch.norm(t_g_ne - p_g_ne, dim=1, p="fro").squeeze(1))[0]
print("The error between systems rotation and laser ranging distance is: ", err_real_laser.item(), 'm.', '\n')

# Save the error-free parameters.
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
dir_path = os.path.abspath(os.path.join(os.getcwd(), "..", "data", "pt"))
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
file_name = f'no_err_{timestamp}.pt'
# FORMAT: t_ne[0] = (B_ne, L_ne, H_ne); t_ne[1] = (x_ne, y_ne, z_ne)
t_ne = torch.stack([t_k_ne, t_g_ne], dim=0)
torch.save(t_ne, os.path.join(dir_path, file_name))

# -------------------------- 2. When there is full-error. --------------------------

p_k, \
    phi_b_e, gamma_b_e, theta_b_e, \
    phi_delta_i_b_e, gamma_delta_i_b_e, theta_delta_i_b_e, \
    phi_delta_v_b_e, gamma_delta_v_b_e, theta_delta_v_b_e, \
    alpha_c_e, beta_c_e, distance_c = parameters_init.parameter_initialization_full_err(length)

# Error laser ranging distance.
t_c_e = torch.zeros(length, 3, 1, dtype=torch.float64)
t_c_e[:, 2, :] = distance_c.unsqueeze(1)

# Error UAV location position parameters.
B_e = p_k[0]  # lat
L_e = p_k[1]  # lon
H_e = p_k[2]  # height

# NOTE: t --> target, p --> UAV, t_*_ne --> * system, _ne --> no error, _e --> error.
p_g_e = rotation.k_2_g(B_e, L_e, H_e)
# When there is error, t_b_e != t_t_e.
t_t_e = rotation.c_2_t(t_c_e, alpha_c_e, beta_c_e)
t_b_e = rotation.t_2_b(t_t_e, phi_delta_i_b_e, gamma_delta_i_b_e, theta_delta_i_b_e,
                       phi_delta_v_b_e, gamma_delta_v_b_e, theta_delta_v_b_e)

t_n_e = rotation.b_2_n(t_b_e, phi_b_e, gamma_b_e, theta_b_e)
t_g_e = rotation.n_2_g(t_n_e, B_e, L_e, p_g_e)
t_k_e = rotation.g_2_k(t_g_e)

# Save the error parameters.
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
file_name = f'full_err_{timestamp}.pt'
t_e = torch.stack([t_k_e, t_g_e], dim=0)
torch.save(t_e, os.path.join(dir_path, file_name))
lat_max, lat_mean, lon_max, lon_mean, alt_max, alt_mean, distance_max, distance_mean = \
    error_calculate(real_data=t_ne, target_data=t_e)[0:8]
print("------full_err------")
print(f"lat_max: {lat_max:.4f}°, lat_mean: {lat_mean:.4f}°, lon_max: {lon_max:.4f}°, lon_mean: {lon_mean:.4f}°, "
      f"alt_max: {alt_max:.4f}m, alt_mean: {alt_mean:.4f}m, distance_max: {distance_max:.4f}m, "
      f"distance_mean: {distance_mean:.4f}m.")
print("--------------------\n")

# -------------------------- 3. Calculation time and freeing GPU memory. --------------------------
end_time = time.time()
print("Total time elapsed: {:.4f}s".format(end_time - begin_time))
torch.cuda.empty_cache()
