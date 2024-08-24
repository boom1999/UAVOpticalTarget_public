# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/8/7 15:05
# @Software: PyCharm

import math

import torch

import process.rotation as rotation


def camera_calculate(phi_rad, gamma_rad, theta_rad, laser_distance, uav_coordinates_n, real_coordinates_n):
    """
    Re-calculate the azimuth and pitch angle of the camera.
    :param phi_rad: Yaw angle of UAV.
    :param gamma_rad: Pitch angle of UAV.
    :param theta_rad: Roll angle of UAV.
    :param laser_distance: Laser ranging distance.
    :param uav_coordinates_n: The UAV positions in the N coordinate system.
    :param real_coordinates_n: The real positions in the N coordinate system.
    :return: The azimuth and pitch angle of the camera.
    """
    num_points = uav_coordinates_n.shape[0]
    target_points = uav_coordinates_n.shape[1]

    # Flatten the tensors.
    phi_rad_flat = phi_rad.view(num_points * target_points, 1).squeeze(-1)
    gamma_rad_flat = gamma_rad.view(num_points * target_points, 1).squeeze(-1)
    theta_rad_flat = theta_rad.view(num_points * target_points, 1).squeeze(-1)
    laser_distance_flat = laser_distance.view(num_points * target_points, 1)
    uav_coordinates_n_flat = uav_coordinates_n.view(num_points * target_points, 3, 1)
    real_coordinates_n_flat = real_coordinates_n.view(num_points * target_points, 3, 1)

    # Calculate the middle angle of the camera.
    middle_angle = rotation.rotation_z(theta_rad_flat) @ rotation.rotation_y(gamma_rad_flat) @ rotation.rotation_x(
        phi_rad_flat) @ (real_coordinates_n_flat - uav_coordinates_n_flat)

    # Calculate the axis angle of the camera.
    # Change the angle from the range of [-pi, pi] to the range of [0, 2pi].
    alpha_mask_index = torch.nonzero((middle_angle[:, 2] <= 0) | (middle_angle[:, 1] >= 0))[:, 0]
    alpha_rad = torch.atan2(middle_angle[:, 2], middle_angle[:, 1]) - math.pi / 2
    alpha_rad[alpha_mask_index] = 2 * math.pi + torch.atan2(middle_angle[alpha_mask_index, 2],
                                                            middle_angle[alpha_mask_index, 1]) - math.pi / 2
    beta_rad = torch.arcsin(middle_angle[:, 0] / laser_distance_flat)

    alpha_rad_target = alpha_rad.squeeze(1)
    beta_rad_target = beta_rad.squeeze(1)
    return alpha_rad_target, beta_rad_target
