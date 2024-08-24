# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/5/29 10:10
# @Software: PyCharm

import math
import os
import random

import torch

import config.parameters_init as parameters_init
import process.rotation as rotation
from process import load_data
from tools import camera_calculate


def radius_calculate(lat_rad):
    """
    Calculate the radius of the earth at a given latitude and longitude.
    :param lat_rad: latitude in radians
    :return: radius of the earth
    """
    a = rotation.EARTH_A
    b = rotation.EARTH_B
    numerator = (a ** 2 * torch.cos(lat_rad)) ** 2 + (b ** 2 * torch.sin(lat_rad)) ** 2
    denominator = (a * torch.cos(lat_rad)) ** 2 + (b * torch.sin(lat_rad)) ** 2
    earth_radius = torch.sqrt(numerator / denominator)
    return earth_radius


def split_circle(lat_o, lon_o, radius_o, n):
    """
    Input the longitude and latitude of the cooperation target point, the radius of the test point flying around,
    the number of divisions, output the longitude and latitude of the starting point of the calibration flight.
    :param lat_o: Latitude of the cooperation target point.
    :param lon_o: Longitude of the cooperation target point.
    :param radius_o: The radius of the point flying around.
    :param n: The number of divisions.
    :return: Lat and lon in shape torch.size(n, 2)
    """
    # Get a more precise radius.
    earth_radius = radius_calculate(lat_o)
    earth_alpha = torch.arcsin(radius_o / earth_radius)
    radius_o = radius_o * torch.cos(earth_alpha)

    points = torch.zeros(n, 2, dtype=torch.float64)
    delta_theta_rad = torch.tensor(2 * math.pi / n, dtype=torch.float64)
    for i in range(n):
        # 0~90, 270~360: positive; 90~270: negative
        delta_lat = radius_o * torch.cos(i * delta_theta_rad)
        # 0~180: negative; 180~360: positive
        delta_lon = - radius_o * torch.sin(i * delta_theta_rad)

        delta_lat_rad = delta_lat / earth_radius
        delta_lon_rad = delta_lon / (earth_radius * torch.cos(torch.deg2rad(lat_o)))

        points[i] = torch.tensor((lat_o + torch.rad2deg(delta_lat_rad), lon_o + torch.rad2deg(delta_lon_rad)))

    return points.unsqueeze(-1)


def convert_angle(angle):
    """
    Convert angles from 0~360 to 0~180 and -180~0
    :param angle: Yaw angle in deg (0~360).
    :return: Yaw angle in deg counterclockwise 0~180 degrees, clockwise 0~-180 degrees.
    """
    if angle > 180:
        angle = angle - 360
    return angle


target_times = parameters_init.target_times
generate_times = parameters_init.times
attitude_delta_phi = parameters_init.attitude_delta_phi
attitude_delta_gamma = parameters_init.attitude_delta_gamma
attitude_delta_theta = parameters_init.attitude_delta_theta
phi_delta_i = parameters_init.phi_delta_i
gamma_delta_i = parameters_init.gamma_delta_i
theta_delta_i = parameters_init.theta_delta_i
axis_delta_i = parameters_init.axis_delta_i
phi_delta_v = parameters_init.phi_delta_v
gamma_delta_v = parameters_init.gamma_delta_v
theta_delta_v = parameters_init.theta_delta_v
axis_delta_v_phi = parameters_init.axis_delta_v_phi
axis_delta_v_gamma = parameters_init.axis_delta_v_gamma
axis_delta_v_theta = parameters_init.axis_delta_v_theta

initial_latitude = torch.tensor(parameters_init.Bp, dtype=torch.float64)
initial_longitude = torch.tensor(parameters_init.Lp, dtype=torch.float64)
initial_altitude = torch.tensor(parameters_init.Hp, dtype=torch.float64)

# Initial position of cooperation target point uav_coordinates
real_coordinates = load_data('no_err')[0][0][0].squeeze(1)
time_stamp_parts = (os.path.basename(load_data('no_err')[1]).split('.')[0]).split('_')
date_part = time_stamp_parts[2]
time_part = time_stamp_parts[3]
time_stamp = date_part + '_' + time_part

# Initialize the attitude angle for forward flight path.
# phi_original = torch.tensor(parameters_init.phi, dtype=torch.float64)
gamma_original = torch.tensor(parameters_init.gamma, dtype=torch.float64)
theta_original = torch.tensor(parameters_init.theta, dtype=torch.float64)

# Distance between target points.
target_delta_distance = torch.tensor(60.0, dtype=torch.float64)

# Computes initial test point trajectory radius, regardless of height.
round_r = torch.norm(rotation.k_2_g(real_coordinates[0], real_coordinates[1], torch.tensor(0, dtype=torch.float64)) -
                     rotation.k_2_g(initial_latitude, initial_longitude, torch.tensor(0, dtype=torch.float64)),
                     dim=1, p="fro").squeeze()
# num of initial points
N = 360
# lat and lon of initial points (torch.size([n, 2]))
circle_points = split_circle(real_coordinates[0], real_coordinates[1], round_r, N)

# Initial position of UAV (N directions.).
uav_circle_coordinates = torch.zeros(N, target_times, 3, 1, dtype=torch.float64)
uav_circle_coordinates[:, 0, 0, :] = circle_points[:, 0]
uav_circle_coordinates[:, 0, 1, :] = circle_points[:, 1]


# Generate data.
def generate_target_data(delta_phi=torch.tensor(20, dtype=torch.float64), gamma=gamma_original, theta=theta_original,
                         attitude_angle_zooms=None, small_fixed_zooms=None, fixed_error_zoom=1.0):

    # Add random for cooperation target point (height).
    height_gaussian = torch.randn(generate_times)
    # height_gaussian_nor = 2 * (height_gaussian - height_gaussian.min()) / (height_gaussian.max() -
    #                                                                        height_gaussian.min()) - 1
    uav_height_delta_zoom = height_gaussian[random.randrange(0, target_times)]
    uav_circle_coordinates[:, 0, 2, :] = initial_altitude + 200 * uav_height_delta_zoom

    direction_index = random.randrange(0, N)
    uav_coordinates = uav_circle_coordinates[direction_index]

    # attitude angle zooms
    # The uav needs to fly as level as possible.
    if attitude_angle_zooms is None:
        attitude_angle_zooms = [1.0, 0.0, 0.0]
    phi_zoom = attitude_angle_zooms[0]
    gamma_zoom = attitude_angle_zooms[1]
    theta_zoom = attitude_angle_zooms[2]

    # small fixed error zooms
    if small_fixed_zooms is None:
        small_fixed_zooms = [0.3, 0.3, 0.3, 0.3, 0.3]
    phi_small_zoom = small_fixed_zooms[0] * fixed_error_zoom
    gamma_small_zoom = small_fixed_zooms[1] * fixed_error_zoom
    theta_small_zoom = small_fixed_zooms[2] * fixed_error_zoom
    alpha_small_zoom = small_fixed_zooms[3] * fixed_error_zoom
    beta_small_zoom = small_fixed_zooms[4] * fixed_error_zoom

    true_phi = (360 / N * direction_index + 180) % 360
    delta_phi = delta_phi * phi_zoom
    phi = convert_angle(true_phi + delta_phi)
    phi_rad = torch.deg2rad(phi)
    gamma = gamma * gamma_zoom
    gamma_rad = torch.deg2rad(gamma)
    theta = theta * theta_zoom
    theta_rad = torch.deg2rad(theta)

    # (Dead Reckoning, DR) Calculate the coordinates of the UAV in the next moment.
    for i in range(1, target_times):
        # Calculate the distance between the initial position and the target point.
        radius_now = radius_calculate(torch.deg2rad(uav_coordinates[i - 1, 0, :]))
        # delta_x --> height
        delta_x = target_delta_distance * torch.sin(gamma_rad)
        # delta_y --> lon(left is positive)
        delta_y = target_delta_distance * torch.cos(gamma_rad) * torch.sin(phi_rad)
        # delta_z --> lat
        delta_z = target_delta_distance * torch.cos(gamma_rad) * torch.cos(phi_rad)

        delta_latitude_rad = delta_z / radius_now
        delta_longitude_rad = delta_y / (torch.cos(torch.deg2rad(uav_coordinates[i - 1, 0, :])) * radius_now)

        # Update the coordinate position.
        uav_coordinates[i, 0, :] = uav_coordinates[i - 1, 0, :] + torch.rad2deg(delta_latitude_rad)
        uav_coordinates[i, 1, :] = uav_coordinates[i - 1, 1, :] - torch.rad2deg(delta_longitude_rad)
        uav_coordinates[i, 2, :] = uav_coordinates[i - 1, 2, :] - delta_x * (gamma > 0)

    uav_coordinates_g = rotation.k_2_g(uav_coordinates[:, 0], uav_coordinates[:, 1], uav_coordinates[:, 2])
    real_coordinates_g = rotation.k_2_g(real_coordinates[0], real_coordinates[1], real_coordinates[2])

    # Calculate the distance between the target point and the UAV.
    laser_distance = torch.norm(uav_coordinates_g - real_coordinates_g, dim=1, p="fro")

    # Calculate the coordinates in N system.
    real_coordinates_n, uav_coordinates_n = rotation.g_2_n(real_coordinates_g, real_coordinates[0],
                                                           real_coordinates[1], uav_coordinates_g)

    alpha_rad_target, beta_rad_target = \
        camera_calculate(phi_rad.repeat(target_times).unsqueeze(0), gamma_rad.repeat(target_times).unsqueeze(0),
                         theta_rad.repeat(target_times).unsqueeze(0), laser_distance.unsqueeze(0),
                         uav_coordinates_n.unsqueeze(0), real_coordinates_n.repeat(target_times, 1, 1).unsqueeze(0))
    alpha = torch.rad2deg(alpha_rad_target)
    beta = torch.rad2deg(beta_rad_target)

    # Add another method to calculate the axis angle of the camera.
    alpha_reverse = (alpha + 180) % 360
    beta_reverse = - beta - 180

    # Generate the attitude angle parameters of the UAV in B system.
    att_angle_std = torch.tensor([attitude_delta_phi, attitude_delta_gamma, attitude_delta_theta])
    att_angle_means = torch.tensor([phi, gamma, theta])
    att_angle = torch.randn((3, target_times))
    att_angle = att_angle * att_angle_std.view(-1, 1) + att_angle_means.view(-1, 1)

    # Generate the installation error parameters of the UAV in T system.
    install_err_std = torch.tensor([axis_delta_i, axis_delta_i, axis_delta_i])
    install_err_means = torch.tensor([phi_delta_i, gamma_delta_i, theta_delta_i])
    install_err = torch.randn((3, target_times))
    install_err = install_err * install_err_std.view(-1, 1) + install_err_means.view(-1, 1)

    # Generate the vibration error parameters of the UAV in T system.
    vibra_err_delta_max = torch.tensor([axis_delta_v_phi, axis_delta_v_gamma, axis_delta_v_theta])
    vibra_err_means = torch.tensor([phi_delta_v, gamma_delta_v, theta_delta_v])
    vibra_err = torch.FloatTensor(3, target_times).uniform_(-1)
    vibra_err = vibra_err * vibra_err_delta_max.view(-1, 1) + vibra_err_means.view(-1, 1)

    delta_phi, delta_gamma, delta_theta = att_angle[0], att_angle[1], att_angle[2]
    phi_delta_i_b_e, gamma_delta_i_b_e, theta_delta_i_b_e = install_err[0], install_err[1], install_err[2]
    phi_delta_v_b_e, gamma_delta_v_b_e, theta_delta_v_b_e = vibra_err[0], vibra_err[1], vibra_err[2]

    camera_delta = (torch.randn(target_times) * parameters_init.camera_delta).clone().detach().to(torch.float64)

    # Add no fixed error.
    phi_deg_no_err = delta_phi
    gamma_deg_no_err = delta_gamma
    theta_deg_no_err = delta_theta
    alpha_deg_no_err = alpha + camera_delta
    beta_deg_no_err = beta + camera_delta
    # The other method to calculate the axis angle of the camera.
    alpha_deg_no_err_reverse = alpha_reverse + camera_delta
    beta_deg_no_err_reverse = beta_reverse + camera_delta

    # Add small fixed error.
    phi_deg_small = phi_deg_no_err + torch.tensor(parameters_init.phi_fixed_delta, dtype=torch.float64) * phi_small_zoom
    gamma_deg_small = gamma_deg_no_err + torch.tensor(parameters_init.gamma_fixed_delta,
                                                      dtype=torch.float64) * gamma_small_zoom
    theta_deg_small = theta_deg_no_err + torch.tensor(parameters_init.theta_fixed_delta,
                                                      dtype=torch.float64) * theta_small_zoom
    alpha_deg_small = alpha_deg_no_err + torch.tensor(parameters_init.alpha_fixed_delta,
                                                      dtype=torch.float64) * alpha_small_zoom
    beta_deg_small = beta_deg_no_err + torch.tensor(parameters_init.beta_fixed_delta,
                                                    dtype=torch.float64) * beta_small_zoom
    # The other method
    alpha_deg_small_reverse = alpha_deg_no_err_reverse + torch.tensor(parameters_init.alpha_fixed_delta,
                                                                      dtype=torch.float64) * alpha_small_zoom
    beta_deg_small_reverse = beta_deg_no_err_reverse + torch.tensor(parameters_init.beta_fixed_delta,
                                                                    dtype=torch.float64) * beta_small_zoom

    # Add fixed error.
    phi_deg = phi_deg_no_err + torch.tensor(parameters_init.phi_fixed_delta, dtype=torch.float64) * fixed_error_zoom
    gamma_deg = gamma_deg_no_err + torch.tensor(parameters_init.gamma_fixed_delta,
                                                dtype=torch.float64) * fixed_error_zoom
    theta_deg = theta_deg_no_err + torch.tensor(parameters_init.theta_fixed_delta,
                                                dtype=torch.float64) * fixed_error_zoom
    alpha_deg = alpha_deg_no_err + torch.tensor(parameters_init.alpha_fixed_delta,
                                                dtype=torch.float64) * fixed_error_zoom
    beta_deg = beta_deg_no_err + torch.tensor(parameters_init.beta_fixed_delta, dtype=torch.float64) * fixed_error_zoom
    # # The other method
    alpha_deg_reverse = alpha_deg_no_err_reverse + torch.tensor(parameters_init.alpha_fixed_delta,
                                                                dtype=torch.float64) * fixed_error_zoom
    beta_deg_reverse = beta_deg_no_err_reverse + torch.tensor(parameters_init.beta_fixed_delta,
                                                              dtype=torch.float64) * fixed_error_zoom

    # Calculate all systematic and non-systematic errors to prepare for subsequent visualization results.

    # Small error
    phi_delta_sum_small = phi_deg_small - phi
    gamma_delta_sum_small = gamma_deg_small - gamma
    theta_delta_sum_small = theta_deg_small - theta
    alpha_delta_sum_small = alpha_deg_small - alpha
    beta_delta_sum_small = beta_deg_small - beta
    #
    # Whether the calculation difference of the total amount of error between inverted mirrors is less than e-15,
    # the same method is used.
    # alpha_delta_sum_small_reverse = alpha_deg_small_reverse - alpha_reverse
    # beta_delta_sum_small_reverse = beta_deg_small_reverse - beta_reverse
    #

    # Big error
    phi_delta_sum = phi_deg - phi
    gamma_delta_sum = gamma_deg - gamma
    theta_delta_sum = theta_deg - theta
    alpha_delta_sum = alpha_deg - alpha
    beta_delta_sum = beta_deg - beta
    #
    # The calculation difference between the total amount of error and whether it is a reverse mirror is smaller than
    # that of e-15, so it can be ignored directly and the same method is used.
    # alpha_delta_sum_reverse = alpha_deg_reverse - alpha_reverse
    # beta_delta_sum_reverse = beta_deg_reverse - beta_reverse
    #

    # No error round trip.
    t_c_n = torch.zeros(target_times, 3, 1, dtype=torch.float64)
    t_c_n[:, 2, :] = laser_distance
    t_t_n = rotation.c_2_t(t_c_n, alpha_deg_no_err, beta_deg_no_err)
    t_b_n = rotation.t_2_b(t_t_n, phi_delta_i_b_e, gamma_delta_i_b_e, theta_delta_i_b_e,
                           phi_delta_v_b_e, gamma_delta_v_b_e, theta_delta_v_b_e)
    t_n_n = rotation.b_2_n(t_b_n, phi_deg_no_err, gamma_deg_no_err, theta_deg_no_err)
    t_g_n = rotation.n_2_g(t_n_n, real_coordinates[0].expand(target_times), real_coordinates[1].expand(target_times),
                           uav_coordinates_g)
    t_k_n = rotation.g_2_k(t_g_n)

    # All small fixed error round trip.
    t_c_s = torch.zeros(target_times, 3, 1, dtype=torch.float64)
    t_c_s[:, 2, :] = laser_distance
    t_t_s = rotation.c_2_t(t_c_s, alpha_deg_small, beta_deg_small)
    t_b_s = rotation.t_2_b(t_t_s, phi_delta_i_b_e, gamma_delta_i_b_e, theta_delta_i_b_e,
                           phi_delta_v_b_e, gamma_delta_v_b_e, theta_delta_v_b_e)
    t_n_s = rotation.b_2_n(t_b_s, phi_deg_small, gamma_deg_small, theta_deg_small)
    t_g_s = rotation.n_2_g(t_n_s, real_coordinates[0].expand(target_times), real_coordinates[1].expand(target_times),
                           uav_coordinates_g)
    t_k_s = rotation.g_2_k(t_g_s)

    # Fixed error in phi round trip.
    # Positive mirror measurement.
    t_c_1 = torch.zeros(target_times, 3, 1, dtype=torch.float64)
    t_c_1[:, 2, :] = laser_distance
    t_t_1 = rotation.c_2_t(t_c_1, alpha_deg_small, beta_deg_small)
    t_b_1 = rotation.t_2_b(t_t_1, phi_delta_i_b_e, gamma_delta_i_b_e, theta_delta_i_b_e,
                           phi_delta_v_b_e, gamma_delta_v_b_e, theta_delta_v_b_e)
    t_n_1 = rotation.b_2_n(t_b_1, phi_deg, gamma_deg_small, theta_deg_small)
    t_g_1 = rotation.n_2_g(t_n_1, real_coordinates[0].expand(target_times), real_coordinates[1].expand(target_times),
                           uav_coordinates_g)
    t_k_1 = rotation.g_2_k(t_g_1)
    # Inverted mirror measurement.
    t_t_1_inverse = rotation.c_2_t(t_c_1, alpha_deg_small_reverse, beta_deg_small_reverse)
    t_b_1_inverse = rotation.t_2_b(t_t_1_inverse, phi_delta_i_b_e, gamma_delta_i_b_e, theta_delta_i_b_e,
                                   phi_delta_v_b_e, gamma_delta_v_b_e, theta_delta_v_b_e)
    t_n_1_inverse = rotation.b_2_n(t_b_1_inverse, phi_deg, gamma_deg_small, theta_deg_small)
    #
    # t_g_1_inverse = rotation.n_2_g(t_n_1_inverse, real_coordinates[0].expand(target_times),
    #                                real_coordinates[1].expand(target_times), uav_coordinates_g)
    # t_k_1_inverse = rotation.g_2_k(t_g_1_inverse)
    #
    # Pack the data for regression in angle phi.
    regression_data_1 = torch.stack((
        torch.deg2rad(phi_deg), torch.deg2rad(gamma_deg_small), torch.deg2rad(theta_deg_small),
        torch.deg2rad(alpha_deg_small), torch.deg2rad(beta_deg_small),
        torch.deg2rad(alpha_deg_small_reverse), torch.deg2rad(beta_deg_small_reverse),
        phi_delta_sum, gamma_delta_sum_small, theta_delta_sum_small, alpha_delta_sum_small, beta_delta_sum_small,
        laser_distance.squeeze(), uav_coordinates_n.squeeze()[:, 0],
        uav_coordinates_n.squeeze()[:, 1], uav_coordinates_n.squeeze()[:, 2]), dim=1)

    # Fixed error in gamma round trip.
    # Positive mirror measurement.
    t_c_2 = torch.zeros(target_times, 3, 1, dtype=torch.float64)
    t_c_2[:, 2, :] = laser_distance
    t_t_2 = rotation.c_2_t(t_c_2, alpha_deg_small, beta_deg_small)
    t_b_2 = rotation.t_2_b(t_t_2, phi_delta_i_b_e, gamma_delta_i_b_e, theta_delta_i_b_e,
                           phi_delta_v_b_e, gamma_delta_v_b_e, theta_delta_v_b_e)
    t_n_2 = rotation.b_2_n(t_b_2, phi_deg_small, gamma_deg, theta_deg_small)
    t_g_2 = rotation.n_2_g(t_n_2, real_coordinates[0].expand(target_times), real_coordinates[1].expand(target_times),
                           uav_coordinates_g)
    t_k_2 = rotation.g_2_k(t_g_2)
    # Inverted mirror measurement.
    t_t_2_inverse = rotation.c_2_t(t_c_2, alpha_deg_small_reverse, beta_deg_small_reverse)
    t_b_2_inverse = rotation.t_2_b(t_t_2_inverse, phi_delta_i_b_e, gamma_delta_i_b_e, theta_delta_i_b_e,
                                   phi_delta_v_b_e, gamma_delta_v_b_e, theta_delta_v_b_e)
    t_n_2_inverse = rotation.b_2_n(t_b_2_inverse, phi_deg_small, gamma_deg, theta_deg_small)
    #
    # t_g_2_inverse = rotation.n_2_g(t_n_2_inverse, real_coordinates[0].expand(target_times),
    #                                real_coordinates[1].expand(target_times), uav_coordinates_g)
    # t_k_2_inverse = rotation.g_2_k(t_g_2_inverse)
    #
    # Pack the data for regression in angle gamma.
    regression_data_2 = torch.stack((
        torch.deg2rad(phi_deg_small), torch.deg2rad(gamma_deg), torch.deg2rad(theta_deg_small),
        torch.deg2rad(alpha_deg_small), torch.deg2rad(beta_deg_small),
        torch.deg2rad(alpha_deg_small_reverse), torch.deg2rad(beta_deg_small_reverse),
        phi_delta_sum_small, gamma_delta_sum, theta_delta_sum_small, alpha_delta_sum_small, beta_delta_sum_small,
        laser_distance.squeeze(), uav_coordinates_n.squeeze()[:, 0],
        uav_coordinates_n.squeeze()[:, 1], uav_coordinates_n.squeeze()[:, 2]), dim=1)

    # Fixed error in theta round trip.
    # Positive mirror measurement.
    t_c_3 = torch.zeros(target_times, 3, 1, dtype=torch.float64)
    t_c_3[:, 2, :] = laser_distance
    t_t_3 = rotation.c_2_t(t_c_3, alpha_deg_small, beta_deg_small)
    t_b_3 = rotation.t_2_b(t_t_3, phi_delta_i_b_e, gamma_delta_i_b_e, theta_delta_i_b_e,
                           phi_delta_v_b_e, gamma_delta_v_b_e, theta_delta_v_b_e)
    t_n_3 = rotation.b_2_n(t_b_3, phi_deg_small, gamma_deg_small, theta_deg)
    t_g_3 = rotation.n_2_g(t_n_3, real_coordinates[0].expand(target_times), real_coordinates[1].expand(target_times),
                           uav_coordinates_g)
    t_k_3 = rotation.g_2_k(t_g_3)
    # Inverted mirror measurement.
    t_t_3_inverse = rotation.c_2_t(t_c_3, alpha_deg_small_reverse, beta_deg_small_reverse)
    t_b_3_inverse = rotation.t_2_b(t_t_3_inverse, phi_delta_i_b_e, gamma_delta_i_b_e, theta_delta_i_b_e,
                                   phi_delta_v_b_e, gamma_delta_v_b_e, theta_delta_v_b_e)
    t_n_3_inverse = rotation.b_2_n(t_b_3_inverse, phi_deg_small, gamma_deg_small, theta_deg)
    #
    # t_g_3_inverse = rotation.n_2_g(t_n_3_inverse, real_coordinates[0].expand(target_times),
    #                                real_coordinates[1].expand(target_times), uav_coordinates_g)
    # t_k_3_inverse = rotation.g_2_k(t_g_3_inverse)
    #
    # Pack the data for regression in angle theta.
    regression_data_3 = torch.stack((
        torch.deg2rad(phi_deg_small), torch.deg2rad(gamma_deg_small), torch.deg2rad(theta_deg),
        torch.deg2rad(alpha_deg_small), torch.deg2rad(beta_deg_small),
        torch.deg2rad(alpha_deg_small_reverse), torch.deg2rad(beta_deg_small_reverse),
        phi_delta_sum_small, gamma_delta_sum_small, theta_delta_sum, alpha_delta_sum_small, beta_delta_sum_small,
        laser_distance.squeeze(), uav_coordinates_n.squeeze()[:, 0],
        uav_coordinates_n.squeeze()[:, 1], uav_coordinates_n.squeeze()[:, 2]), dim=1)

    # Fixed error in alpha round trip.
    # Positive mirror measurement.
    t_c_4 = torch.zeros(target_times, 3, 1, dtype=torch.float64)
    t_c_4[:, 2, :] = laser_distance
    t_t_4 = rotation.c_2_t(t_c_4, alpha_deg, beta_deg_small)
    t_b_4 = rotation.t_2_b(t_t_4, phi_delta_i_b_e, gamma_delta_i_b_e, theta_delta_i_b_e,
                           phi_delta_v_b_e, gamma_delta_v_b_e, theta_delta_v_b_e)
    t_n_4 = rotation.b_2_n(t_b_4, phi_deg_small, gamma_deg_small, theta_deg_small)
    t_g_4 = rotation.n_2_g(t_n_4, real_coordinates[0].expand(target_times), real_coordinates[1].expand(target_times),
                           uav_coordinates_g)
    t_k_4 = rotation.g_2_k(t_g_4)
    # Inverted mirror measurement.
    t_t_4_inverse = rotation.c_2_t(t_c_4, alpha_deg_reverse, beta_deg_small_reverse)
    t_b_4_inverse = rotation.t_2_b(t_t_4_inverse, phi_delta_i_b_e, gamma_delta_i_b_e, theta_delta_i_b_e,
                                   phi_delta_v_b_e, gamma_delta_v_b_e, theta_delta_v_b_e)
    t_n_4_inverse = rotation.b_2_n(t_b_4_inverse, phi_deg_small, gamma_deg_small, theta_deg_small)
    #
    # t_g_4_inverse = rotation.n_2_g(t_n_4_inverse, real_coordinates[0].expand(target_times),
    #                                real_coordinates[1].expand(target_times), uav_coordinates_g)
    # t_k_4_inverse = rotation.g_2_k(t_g_4_inverse)
    #
    # Pack the data for regression in angle alpha.
    regression_data_4 = torch.stack((
        torch.deg2rad(phi_deg_small), torch.deg2rad(gamma_deg_small), torch.deg2rad(theta_deg_small),
        torch.deg2rad(alpha_deg), torch.deg2rad(beta_deg_small),
        torch.deg2rad(alpha_deg_reverse), torch.deg2rad(beta_deg_small_reverse),
        phi_delta_sum_small, gamma_delta_sum_small, theta_delta_sum_small, alpha_delta_sum, beta_delta_sum_small,
        laser_distance.squeeze(), uav_coordinates_n.squeeze()[:, 0],
        uav_coordinates_n.squeeze()[:, 1], uav_coordinates_n.squeeze()[:, 2]), dim=1)

    # Fixed error in beta round trip.
    # Positive mirror measurement.
    t_c_5 = torch.zeros(target_times, 3, 1, dtype=torch.float64)
    t_c_5[:, 2, :] = laser_distance
    t_t_5 = rotation.c_2_t(t_c_5, alpha_deg_small, beta_deg)
    t_b_5 = rotation.t_2_b(t_t_5, phi_delta_i_b_e, gamma_delta_i_b_e, theta_delta_i_b_e,
                           phi_delta_v_b_e, gamma_delta_v_b_e, theta_delta_v_b_e)
    t_n_5 = rotation.b_2_n(t_b_5, phi_deg_small, gamma_deg_small, theta_deg_small)
    t_g_5 = rotation.n_2_g(t_n_5, real_coordinates[0].expand(target_times), real_coordinates[1].expand(target_times),
                           uav_coordinates_g)
    t_k_5 = rotation.g_2_k(t_g_5)
    # Inverted mirror measurement.
    t_t_5_inverse = rotation.c_2_t(t_c_5, alpha_deg_small_reverse, beta_deg_reverse)
    t_b_5_inverse = rotation.t_2_b(t_t_5_inverse, phi_delta_i_b_e, gamma_delta_i_b_e, theta_delta_i_b_e,
                                   phi_delta_v_b_e, gamma_delta_v_b_e, theta_delta_v_b_e)
    t_n_5_inverse = rotation.b_2_n(t_b_5_inverse, phi_deg_small, gamma_deg_small, theta_deg_small)
    #
    # t_g_5_inverse = rotation.n_2_g(t_n_5_inverse, real_coordinates[0].expand(target_times),
    #                                real_coordinates[1].expand(target_times), uav_coordinates_g)
    # t_k_5_inverse = rotation.g_2_k(t_g_5_inverse)
    #
    # Pack the data for regression in angle beta.
    regression_data_5 = torch.stack((
        torch.deg2rad(phi_deg_small), torch.deg2rad(gamma_deg_small), torch.deg2rad(theta_deg_small),
        torch.deg2rad(alpha_deg_small), torch.deg2rad(beta_deg),
        torch.deg2rad(alpha_deg_small_reverse), torch.deg2rad(beta_deg_reverse),
        phi_delta_sum_small, gamma_delta_sum_small, theta_delta_sum_small, alpha_delta_sum_small, beta_delta_sum,
        laser_distance.squeeze(), uav_coordinates_n.squeeze()[:, 0],
        uav_coordinates_n.squeeze()[:, 1], uav_coordinates_n.squeeze()[:, 2]), dim=1)

    # Use the N coordinate system data for classification tasks (relative coordinates)
    n_coordinates = [t_n_n, t_n_s, t_n_1, t_n_2, t_n_3, t_n_4, t_n_5]
    for tensor in n_coordinates:
        tensor += uav_coordinates_n
    n_coordinates_inverse = [t_n_1_inverse, t_n_2_inverse, t_n_3_inverse, t_n_4_inverse, t_n_5_inverse]
    for tensor in n_coordinates_inverse:
        tensor += uav_coordinates_n

    return t_n_n, t_n_s, t_n_1, t_n_2, t_n_3, t_n_4, t_n_5, \
        t_n_1_inverse, t_n_2_inverse, t_n_3_inverse, t_n_4_inverse, t_n_5_inverse, \
        t_g_n, t_g_s, t_g_1, t_g_2, t_g_3, t_g_4, t_g_5, \
        t_k_n, t_k_s, t_k_1, t_k_2, t_k_3, t_k_4, t_k_5, \
        regression_data_1, regression_data_2, regression_data_3, regression_data_4, regression_data_5, \
        real_coordinates_n, real_coordinates_g, real_coordinates, uav_coordinates


# The N coordinate system data is used for classification tasks (relative coordinates),
# the G coordinate system data is not used yet, and the K coordinate system data is used for drawing.
n_n, n_s, n_1, n_2, n_3, n_4, n_5, \
    _, _, _, _, _, \
    g_n, g_s, g_1, g_2, g_3, g_4, g_5, \
    k_n, k_s, k_1, k_2, k_3, k_4, k_5, \
    _, _, _, _, _, real_n, real_g, _, _ = generate_target_data()

# Calculate the error.
# 1. No fixed error.
error_n = torch.norm(g_n - real_g, dim=1, p="fro")
error_n_max, error_n_mean = torch.max(error_n), torch.mean(error_n)
# 2. Small fixed error.
error_s = torch.norm(g_s - real_g, dim=1, p="fro")
error_s_max, error_s_mean = torch.max(error_s), torch.mean(error_s)
# 3. Fixed error in phi.
error_1 = torch.norm(g_1 - real_g, dim=1, p="fro")
error_1_max, error_1_mean = torch.max(error_1), torch.mean(error_1)
# 4. Fixed error in gamma.
error_2 = torch.norm(g_2 - real_g, dim=1, p="fro")
error_2_max, error_2_mean = torch.max(error_2), torch.mean(error_2)
# 5. Fixed error in theta.
error_3 = torch.norm(g_3 - real_g, dim=1, p="fro")
error_3_max, error_3_mean = torch.max(error_3), torch.mean(error_3)
# 6. Fixed error in alpha.
error_4 = torch.norm(g_4 - real_g, dim=1, p="fro")
error_4_max, error_4_mean = torch.max(error_4), torch.mean(error_4)
# 7. Fixed error in beta.
error_5 = torch.norm(g_5 - real_g, dim=1, p="fro")
error_5_max, error_5_mean = torch.max(error_5), torch.mean(error_5)

error_data = torch.stack((torch.zeros(target_times, 1), error_n, error_s,
                          error_1, error_2, error_3, error_4, error_5), dim=0)

# Save the data.
# Save the .pt file.
dir_path_pt = os.path.abspath(os.path.join(os.getcwd(), "..", "data", "pt"))
if not os.path.exists(dir_path_pt):
    os.makedirs(dir_path_pt)
file_name_pt = f'round_trip_{time_stamp}.pt'
round_coordinates_ans = torch.stack(((real_coordinates.unsqueeze(1).unsqueeze(0)).expand(target_times, 3, 1),
                                     k_n, k_s, k_1, k_2, k_3, k_4, k_5), dim=0)
round_coordinates_ans_n = torch.stack((real_n.expand(target_times, 3, 1),
                                       n_n, n_s, n_1, n_2, n_3, n_4, n_5), dim=0)
round_coordinates_ans = torch.cat((round_coordinates_ans, error_data.unsqueeze(2), round_coordinates_ans_n), dim=2)
# Shape: (8, target_times, 7, 1)
# dim 0: 8 different round trip.
# dim 1: target_times.
# dim 2: 4 different coordinates: lat, lon, alt, error, lat_n, lon_n, alt_n.
# dim 3: 1.
torch.save(round_coordinates_ans, os.path.join(dir_path_pt, file_name_pt))
