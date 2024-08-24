# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/5/16 17:15
# @Software: PyCharm

import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch

import config.parameters_init as parameters_init
import process.rotation as rotation
from process.load_data import load_data
from tools import save_fig

length = 50

begin_time = time.time()

# -------------------------- 1. When there is no error. --------------------------

p_k_ne, \
    phi_b_ne, gamma_b_ne, theta_b_ne, \
    phi_delta_i_b_ne, gamma_delta_i_b_ne, theta_delta_i_b_ne, \
    phi_delta_v_b_ne, gamma_delta_v_b_ne, theta_delta_v_b_ne, \
    alpha_c_ne, beta_c_ne, R = parameters_init.parameter_initialization_no_err(length)

# no-err location data
real_data, data_name = load_data('no_err')
# real_k_data = real_data[0]
real_g_data = real_data[1][0:length]

# -------------------------- 2. When there is error. --------------------------

p_k, \
    phi_b_e, gamma_b_e, theta_b_e, \
    phi_delta_i_b_e, gamma_delta_i_b_e, theta_delta_i_b_e, \
    phi_delta_v_b_e, gamma_delta_v_b_e, theta_delta_v_b_e, \
    alpha_c_e, beta_c_e, distance_c = parameters_init.parameter_analysis(length)

# No practical effect, eight decimal places are displayed when debugging.
torch.set_printoptions(precision=16)

# -------------------------- 3. Independent control variable analysis module. --------------------------

# Set the save path.
save_dir_path = os.path.abspath(os.path.join(
    os.getcwd().split('OpticalTarget')[0], 'OpticalTarget', 'data', 'fig', 'variable_analysis'))

# 3.1. Analyze the error of the laser ranging distance.

t_c_1 = torch.zeros(length, 3, 1, dtype=torch.float64)
t_c_1[:, 2, :] = distance_c.unsqueeze(1)

B_ne = p_k_ne[0]  # lat
L_ne = p_k_ne[1]  # lon
H_ne = p_k_ne[2]  # height

p_g_ne = rotation.k_2_g(B_ne, L_ne, H_ne)

t_t_1 = rotation.c_2_t(t_c_1, alpha_c_ne, beta_c_ne)
t_b_1 = rotation.t_2_b(t_t_1, phi_delta_i_b_ne, gamma_delta_i_b_ne, theta_delta_i_b_ne,
                       phi_delta_v_b_ne, gamma_delta_v_b_ne, theta_delta_v_b_ne)

t_n_1 = rotation.b_2_n(t_b_1, phi_b_ne, gamma_b_ne, theta_b_ne)
t_g_1 = rotation.n_2_g(t_n_1, B_ne, L_ne, p_g_ne)
# t_k_1 = rotation.g_2_k(t_g_1)

distance_err_1 = torch.norm(real_g_data - t_g_1, dim=1, p="fro").squeeze(1)
plot_x_1 = distance_c - R
plot_y_1 = distance_err_1

# Sampling 1/50th of the data points for plotting
# changing step ************
# step = 50
step = 1
indices = torch.arange(0, len(plot_x_1), step=step, dtype=torch.int64)

plot_x_1 = plot_x_1[indices]
plot_y_1 = plot_y_1[indices]

# Create the plot.
fig_1 = plt.figure(1, dpi=600)
plt.scatter(plot_x_1, plot_y_1, s=20, c='r', marker='*', label='distance')

# Customize the plot
plt.xlabel('Laser ranging distance err (m)', fontsize=10, fontweight='bold')
plt.ylabel('Target distance err (m)', fontsize=10, fontweight='bold')
plt.title('Influence of the laser ranging distance error', fontsize=12, fontweight='bold')
plt.grid(True, linewidth=0.5, alpha=0.5)
plt.legend(loc='upper right')

# Display the plot
plt.show()

# Save the plot.
file_time = (os.path.basename(data_name).split('.')[0]).split('_')
time_stamp = file_time[2] + '_' + file_time[3]
save_name_1 = os.path.join('distance_err_' + time_stamp + '.png')
save_fig(fig_1, save_name_1, save_dir_path)

# 3.2. Analyze the error of the UAV GPS location parameters.

t_c_ne = torch.zeros(length, 3, 1, dtype=torch.float64)
t_c_ne[:, 2, :] = R.unsqueeze(1)

B_2 = p_k[0]  # lat_err
L_2 = p_k[1]  # lon_err

# 3.2.1 Analyze the error of the UAV GPS location parameters (lat_err).
p_g_2_1 = rotation.k_2_g(B_2, L_ne, H_ne)
t_t_2 = rotation.c_2_t(t_c_ne, alpha_c_ne, beta_c_ne)
t_b_2 = rotation.t_2_b(t_t_2, phi_delta_i_b_ne, gamma_delta_i_b_ne, theta_delta_i_b_ne,
                       phi_delta_v_b_ne, gamma_delta_v_b_ne, theta_delta_v_b_ne)
t_n_2 = rotation.b_2_n(t_b_2, phi_b_ne, gamma_b_ne, theta_b_ne)
t_g_2_1 = rotation.n_2_g(t_n_2, B_2, L_ne, p_g_2_1)
# t_k_2_1 = rotation.g_2_k(t_g_2_1)

# 3.2.2 Analyze the error of the UAV GPS location parameters (lon_err).

p_g_2_2 = rotation.k_2_g(B_ne, L_2, H_ne)
# ... Same as above.
t_g_2_2 = rotation.n_2_g(t_n_2, B_ne, L_2, p_g_2_2)
# t_k_2_2 = rotation.g_2_k(t_g_2_2)

# 3.2.3 Plot the error of the UAV GPS location parameters.

distance_err_2_1 = torch.norm(real_g_data - t_g_2_1, dim=1, p="fro").squeeze(1)
plot_x_2_1 = (p_k[0] - p_k_ne[0])[indices]
plot_y_2_1 = distance_err_2_1[indices]

distance_err_2_2 = torch.norm(real_g_data - t_g_2_2, dim=1, p="fro").squeeze(1)
plot_x_2_2 = (p_k[1] - p_k_ne[1])[indices]
plot_y_2_2 = distance_err_2_2[indices]

# Create the plot.
fig_2 = plt.figure(2, dpi=600)
plt.scatter(plot_x_2_1, plot_y_2_1, s=20, c='r', marker='*', label='latitude')
plt.scatter(plot_x_2_2, plot_y_2_2, s=20, c='g', marker='1', label='longitude')

# Customize the plot
plt.xlabel('Lat and Lon err (degree)', fontsize=10, fontweight='bold')
plt.ylabel('Target distance err (m)', fontsize=10, fontweight='bold')
plt.title('Influence of the GPS error (lat and lon)', fontsize=12, fontweight='bold')
plt.grid(True, linewidth=0.5, alpha=0.5)
plt.legend(loc='upper right')

# Display the plot
plt.show()

# Save the plot.
save_name_2 = os.path.join('lat_lon_err_' + time_stamp + '.png')
save_fig(fig_2, save_name_2, save_dir_path)

# 3.3. Analyze the error of the UAV attitude parameters.

H_3 = p_k[2]  # height_err

p_g_3 = rotation.k_2_g(B_ne, L_ne, H_3)
t_t_3 = rotation.c_2_t(t_c_ne, alpha_c_ne, beta_c_ne)
t_b_3 = rotation.t_2_b(t_t_3, phi_delta_i_b_ne, gamma_delta_i_b_ne, theta_delta_i_b_ne,
                       phi_delta_v_b_ne, gamma_delta_v_b_ne, theta_delta_v_b_ne)
t_n_3 = rotation.b_2_n(t_b_3, phi_b_ne, gamma_b_ne, theta_b_ne)
t_g_3 = rotation.n_2_g(t_n_3, B_ne, L_ne, p_g_3)
# t_k_3 = rotation.g_2_k(t_g_3)

distance_err_3 = torch.norm(real_g_data - t_g_3, dim=1, p="fro").squeeze(1)
plot_x_3 = (H_3 - p_k_ne[2])[indices]
plot_y_3 = distance_err_3[indices]

# Create the plot.
fig_3 = plt.figure(3, dpi=600)
plt.scatter(plot_x_3, plot_y_3, s=20, c='r', marker='*', label='height')

# Customize the plot
plt.xlabel('GPS height err (m)', fontsize=10, fontweight='bold')
plt.ylabel('Target distance err (m)', fontsize=10, fontweight='bold')
plt.title('Influence of the GPS error (height)', fontsize=12, fontweight='bold')
plt.grid(True, linewidth=0.5, alpha=0.5)
plt.legend(loc='upper right')

# Display the plot
plt.show()

# Save the plot.
save_name_3 = os.path.join('height_err_' + time_stamp + '.png')
save_fig(fig_3, save_name_3, save_dir_path)

# 3.4. Analyze the error of the attitude angle parameters.

# 3.4.1 Analyze the error of the UAV attitude angle parameters (Yaw angle).

t_t_ne = rotation.c_2_t(t_c_ne, alpha_c_ne, beta_c_ne)
t_b_ne = rotation.t_2_b(t_t_ne, phi_delta_i_b_ne, gamma_delta_i_b_ne, theta_delta_i_b_ne,
                        phi_delta_v_b_ne, gamma_delta_v_b_ne, theta_delta_v_b_ne)
t_n_4_1 = rotation.b_2_n(t_b_ne, phi_b_e, gamma_b_ne, theta_b_ne)
t_g_4_1 = rotation.n_2_g(t_n_4_1, B_ne, L_ne, p_g_ne)
# t_k_4_1 = rotation.g_2_k(t_g_4_1)

distance_err_4_1 = torch.norm(real_g_data - t_g_4_1, dim=1, p="fro").squeeze(1)
plot_x_4_1 = (phi_b_e - phi_b_ne)[indices]
plot_y_4_1 = distance_err_4_1[indices]

# 3.4.2 Analyze the error of the UAV attitude angle parameters (Pitch angle).

t_n_4_2 = rotation.b_2_n(t_b_ne, phi_b_ne, gamma_b_e, theta_b_ne)
t_g_4_2 = rotation.n_2_g(t_n_4_2, B_ne, L_ne, p_g_ne)
# t_k_4_2 = rotation.g_2_k(t_g_4_2)

distance_err_4_2 = torch.norm(real_g_data - t_g_4_2, dim=1, p="fro").squeeze(1)
plot_x_4_2 = (gamma_b_e - gamma_b_ne)[indices]
plot_y_4_2 = distance_err_4_2[indices]

# 3.4.3 Analyze the error of the UAV attitude angle parameters (Roll angle).

t_n_4_3 = rotation.b_2_n(t_b_ne, phi_b_ne, gamma_b_ne, theta_b_e)
t_g_4_3 = rotation.n_2_g(t_n_4_3, B_ne, L_ne, p_g_ne)
# t_k_4_3 = rotation.g_2_k(t_g_4_3)

distance_err_4_3 = torch.norm(real_g_data - t_g_4_3, dim=1, p="fro").squeeze(1)
plot_x_4_3 = (theta_b_e - theta_b_ne)[indices]
plot_y_4_3 = distance_err_4_3[indices]

# Create the plot.
fig_4 = plt.figure(4, dpi=600)
plt.scatter(plot_x_4_1, plot_y_4_1, s=20, c='r', marker='*', label='X-axis (Yaw angle)')
plt.scatter(plot_x_4_2, plot_y_4_2, s=20, c='g', marker='1', label='Y-axis (Pitch angle)')
plt.scatter(plot_x_4_3, plot_y_4_3, s=20, c='b', marker='x', label='Z-axis (Roll angle)')

# Customize the plot
plt.xlabel('Attitude angle err (degree)', fontsize=10, fontweight='bold')
plt.ylabel('Target distance err (m)', fontsize=10, fontweight='bold')
plt.title('Influence of the attitude angle error', fontsize=12, fontweight='bold')
plt.grid(True, linewidth=0.5, alpha=0.5)
plt.legend(loc='upper right')

# Display the plot
plt.show()

# Save the plot.
save_name_4 = os.path.join('attitude_angle_err_' + time_stamp + '.png')
save_fig(fig_4, save_name_4, save_dir_path)

# 3.5. Analyze the error of the optical axis stabilized platform parameters.
# The influence of installation error and vibration error is the same,
# and only the influence of installation error is listed here.

# 3.5.1 Yaw angle.

t_b_5_1 = rotation.t_2_b(t_t_ne, phi_delta_i_b_e, gamma_delta_i_b_ne, theta_delta_i_b_ne,
                         phi_delta_v_b_ne, gamma_delta_v_b_ne, theta_delta_v_b_ne)
t_n_5_1 = rotation.b_2_n(t_b_5_1, phi_b_ne, gamma_b_ne, theta_b_ne)
t_g_5_1 = rotation.n_2_g(t_n_5_1, B_ne, L_ne, p_g_ne)
# t_k_5_1 = rotation.g_2_k(t_g_5_1)

distance_err_5_1 = torch.norm(real_g_data - t_g_5_1, dim=1, p="fro").squeeze(1)
plot_x_5_1 = (phi_delta_i_b_e - phi_delta_i_b_ne)[indices]
plot_y_5_1 = distance_err_5_1[indices]

# 3.5.2 Pitch angle.

t_b_5_2 = rotation.t_2_b(t_t_ne, phi_delta_i_b_ne, gamma_delta_i_b_e, theta_delta_i_b_ne,
                         phi_delta_v_b_ne, gamma_delta_v_b_ne, theta_delta_v_b_ne)
t_n_5_2 = rotation.b_2_n(t_b_5_2, phi_b_ne, gamma_b_ne, theta_b_ne)
t_g_5_2 = rotation.n_2_g(t_n_5_2, B_ne, L_ne, p_g_ne)
# t_k_5_2 = rotation.g_2_k(t_g_5_2)

distance_err_5_2 = torch.norm(real_g_data - t_g_5_2, dim=1, p="fro").squeeze(1)
plot_x_5_2 = (gamma_delta_i_b_e - gamma_delta_i_b_ne)[indices]
plot_y_5_2 = distance_err_5_2[indices]

# 3.5.3 Roll angle.

t_b_5_3 = rotation.t_2_b(t_t_ne, phi_delta_i_b_ne, gamma_delta_i_b_ne, theta_delta_i_b_e,
                         phi_delta_v_b_ne, gamma_delta_v_b_ne, theta_delta_v_b_ne)
t_n_5_3 = rotation.b_2_n(t_b_5_3, phi_b_ne, gamma_b_ne, theta_b_ne)
t_g_5_3 = rotation.n_2_g(t_n_5_3, B_ne, L_ne, p_g_ne)
# t_k_5_3 = rotation.g_2_k(t_g_5_3)

distance_err_5_3 = torch.norm(real_g_data - t_g_5_3, dim=1, p="fro").squeeze(1)
plot_x_5_3 = (theta_delta_i_b_e - theta_delta_i_b_ne)[indices]
plot_y_5_3 = distance_err_5_3[indices]

# Create the plot.
fig_5 = plt.figure(5, dpi=600)
plt.scatter(plot_x_5_1, plot_y_5_1, s=20, c='r', marker='*', label='X-axis (Yaw angle)')
plt.scatter(plot_x_5_2, plot_y_5_2, s=20, c='g', marker='1', label='Y-axis (Pitch angle)')
plt.scatter(plot_x_5_3, plot_y_5_3, s=20, c='b', marker='x', label='Z-axis (Roll angle)')

# Customize the plot
plt.xlabel('The optical axis stabilized platform parameters (degree)', fontsize=10, fontweight='bold')
plt.ylabel('Target distance err (m)', fontsize=10, fontweight='bold')
plt.title('Influence of the installation error', fontsize=12, fontweight='bold')
plt.grid(True, linewidth=0.5, alpha=0.5)
plt.legend(loc='upper right')

# Display the plot
plt.show()

# Save the plot.
save_name_5 = os.path.join('installation_err_' + time_stamp + '.png')
save_fig(fig_5, save_name_5, save_dir_path)

# 3.6. Analyze the error of the camera optical axis pointing parameters.

# 3.6.1 Azimuth angle of camera.

t_t_6_1 = rotation.c_2_t(t_c_ne, alpha_c_e, beta_c_ne)
t_b_6_1 = rotation.t_2_b(t_t_6_1, phi_delta_i_b_ne, gamma_delta_i_b_ne, theta_delta_i_b_ne,
                         phi_delta_v_b_ne, gamma_delta_v_b_ne, theta_delta_v_b_ne)
t_n_6_1 = rotation.b_2_n(t_b_6_1, phi_b_ne, gamma_b_ne, theta_b_ne)
t_g_6_1 = rotation.n_2_g(t_n_6_1, B_ne, L_ne, p_g_ne)
# t_k_6_1 = rotation.g_2_k(t_g_6_1)

distance_err_6_1 = torch.norm(real_g_data - t_g_6_1, dim=1, p="fro").squeeze(1)
plot_x_6_1 = (alpha_c_e - alpha_c_ne)[indices]
plot_y_6_1 = distance_err_6_1[indices]

# 3.6.2 Pitch angle of camera.

t_t_6_2 = rotation.c_2_t(t_c_ne, alpha_c_ne, beta_c_e)
t_b_6_2 = rotation.t_2_b(t_t_6_2, phi_delta_i_b_ne, gamma_delta_i_b_ne, theta_delta_i_b_ne,
                         phi_delta_v_b_ne, gamma_delta_v_b_ne, theta_delta_v_b_ne)
t_n_6_2 = rotation.b_2_n(t_b_6_2, phi_b_ne, gamma_b_ne, theta_b_ne)
t_g_6_2 = rotation.n_2_g(t_n_6_2, B_ne, L_ne, p_g_ne)
# t_k_6_2 = rotation.g_2_k(t_g_6_2)

distance_err_6_2 = torch.norm(real_g_data - t_g_6_2, dim=1, p="fro").squeeze(1)
plot_x_6_2 = (beta_c_e - beta_c_ne)[indices]
plot_y_6_2 = distance_err_6_2[indices]

# Create the plot.
fig_6 = plt.figure(6, dpi=600)
plt.scatter(plot_x_6_1, plot_y_6_1, s=20, c='r', marker='*', label='Azimuth angle of camera.')
plt.scatter(plot_x_6_2, plot_y_6_2, s=20, c='g', marker='1', label='Pitch angle of camera.')

# Customize the plot
plt.xlabel('The camera optical axis pointing parameters (degree)', fontsize=10, fontweight='bold')
plt.ylabel('Target distance err (m)', fontsize=10, fontweight='bold')
plt.title('Influence of the camera optical axis pointing error', fontsize=12, fontweight='bold')
plt.grid(True, linewidth=0.5, alpha=0.5)
plt.legend(loc='upper right')

# Display the plot
plt.show()

# Save the plot.
save_name_6 = os.path.join('camera_angle_err_' + time_stamp + '.png')
save_fig(fig_6, save_name_6, save_dir_path)

# -------------------------- 4. Quantitative analysis of error influencing factors --------------------------

# 4.1 Laser ranging error.

mean_laser = torch.mean(torch.diff(distance_err_1[int(length / 2):]) / torch.diff((distance_c - R)[int(length / 2):]))

# 4.2 GPS positioning error (lat and lon).
# Since the latitude and longitude are differentiated according to the length,
# nan will appear if the spacing is too small, so the numerical value after sampling is used for approximation.
mean_lat = torch.mean(torch.diff(plot_y_2_1) / torch.diff(plot_x_2_1))
mean_lon = torch.mean(torch.diff(plot_y_2_2) / torch.diff(plot_x_2_2))

# 4.3 GPS positioning error (alt).

mean_alt = torch.mean(torch.diff(distance_err_3) / torch.diff((H_3 - p_k_ne[2])))

# 4.4 Attitude angle error.

mean_yaw = torch.mean(torch.diff(distance_err_4_1) / torch.diff((phi_b_e - phi_b_ne)))
mean_pitch = torch.mean(torch.diff(distance_err_4_2) / torch.diff((gamma_b_e - gamma_b_ne)))
mean_roll = torch.mean(torch.diff(distance_err_4_3) / torch.diff((theta_b_e - theta_b_ne)))

# 4.5 Installation error (the same as vibration error).

mean_yaw_install = torch.mean(torch.diff(distance_err_5_1) / torch.diff((phi_delta_i_b_e - phi_delta_v_b_ne)))
mean_pitch_install = torch.mean(torch.diff(distance_err_5_2) / torch.diff((gamma_delta_i_b_e - gamma_delta_v_b_ne)))
mean_roll_install = torch.mean(torch.diff(distance_err_5_3) / torch.diff((theta_delta_i_b_e - theta_delta_v_b_ne)))

# 4.6 Camera optical axis pointing error.

mean_camera_azimuth = torch.mean(torch.diff(distance_err_6_1) / torch.diff((alpha_c_e - alpha_c_ne)))
mean_camera_pitch = torch.mean(torch.diff(distance_err_6_2) / torch.diff((beta_c_e - beta_c_ne)))

print('mean_laser: ', mean_laser.item(), ' m/m\n',
      'mean_lat: ', mean_lat.item() * 1e-5, ' m/1e-5 °\n',
      'mean_lon: ', mean_lon.item() * 1e-5, ' m/1e-5 °\n',
      'mean_alt: ', mean_alt.item(), 'm/m\n',
      'mean_yaw: ', mean_yaw.item() * 1e-2, ' m/1e-2 °\n',
      'mean_pitch: ', mean_pitch.item() * 1e-2, ' m/1e-2 °\n',
      'mean_roll: ', mean_roll.item() * 1e-2, ' m/1e-2 °\n',
      'mean_yaw_install: ', mean_yaw_install.item() * 1e-2, ' m/1e-2 °\n',
      'mean_pitch_install: ', mean_pitch_install.item() * 1e-2, ' m/1e-2 °\n',
      'mean_roll_install: ', mean_roll_install.item() * 1e-2, ' m/1e-2 °\n',
      'mean_camera_azimuth: ', mean_camera_azimuth.item() * 1e-2, ' m/1e-2 °\n',
      'mean_camera_pitch: ', mean_camera_pitch.item() * 1e-2, ' m/1e-2 °\n')

# Save data for Origin.
# 1. Laser ranging distance.
data_1 = torch.stack((distance_c - R, distance_err_1), dim=1)
# 2. GPS positioning error (lat and lon).
data_2_1 = torch.stack((p_k[0] - p_k_ne[0], distance_err_2_1), dim=1)
data_2_2 = torch.stack((p_k[1] - p_k_ne[1], distance_err_2_2), dim=1)
# 3. GPS positioning error (alt).
data_3 = torch.stack((H_3 - p_k_ne[2], distance_err_3), dim=1)
# 4. Attitude angle error.
data_4_1 = torch.stack((phi_b_e - phi_b_ne, distance_err_4_1), dim=1)
data_4_2 = torch.stack((gamma_b_e - gamma_b_ne, distance_err_4_2), dim=1)
data_4_3 = torch.stack((theta_b_e - theta_b_ne, distance_err_4_3), dim=1)
# 5. Installation error (the same as vibration error).
data_5_1 = torch.stack((phi_delta_i_b_e - phi_delta_v_b_ne, distance_err_5_1), dim=1)
data_5_2 = torch.stack((gamma_delta_i_b_e - gamma_delta_v_b_ne, distance_err_5_2), dim=1)
data_5_3 = torch.stack((theta_delta_i_b_e - theta_delta_v_b_ne, distance_err_5_3), dim=1)
# 6. Camera optical axis pointing error.
data_6_1 = torch.stack((alpha_c_e - alpha_c_ne, distance_err_6_1), dim=1)
data_6_2 = torch.stack((beta_c_e - beta_c_ne, distance_err_6_2), dim=1)
# 7. Save data.
# 7.1 Combine the data.
# shape: [12, 100, 2]
data = torch.cat((data_1, data_2_1, data_2_2, data_3, data_4_1, data_4_2, data_4_3,
                  data_5_1, data_5_2, data_5_3, data_6_1, data_6_2), dim=1)
# 7.2 Save the data.
dir_path_csv = os.path.abspath(os.path.join(os.getcwd().split('OpticalTarget')[0], 'OpticalTarget', 'data', 'csv'))
if not os.path.exists(dir_path_csv):
    os.makedirs(dir_path_csv)
file_name_csv = f'variable_analysis_{time_stamp}.csv'
columns = ['laser_error', 'target_err_1', 'lat_err', 'target_err_2', 'lon_err', 'target_err_3',
           'height_err', 'target_err_4', 'yaw_err', 'target_err_5', 'pitch_err', 'target_err_6',
           'roll_err', 'target_err_7', 'yaw_install_err', 'target_err_8', 'pitch_install_err', 'target_err_9',
           'roll_install_err', 'target_err_10', 'camera_azimuth_err', 'target_err_11',
           'camera_pitch_err', 'target_err_12']
df = pd.DataFrame(data.numpy(), columns=columns)
df.to_csv(os.path.join(dir_path_csv, file_name_csv), index=False)

end_time = time.time()
print("Total time elapsed: {:.4f}s".format(end_time - begin_time))
