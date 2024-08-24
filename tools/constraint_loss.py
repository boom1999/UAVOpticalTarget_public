# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/8/1 16:07
# @Software: PyCharm

import torch
import torch.nn as nn
import process.rotation as rotation


def calculate_constraint_loss(points, phi, gamma, theta, alpha, beta, distance, real_k=None, uav_k=None, need_k=False):
    """
    Use the regression parameters to calculate the constraint loss,
    these five angles are the output of the network(_hat).
    :param points: The UAV positions in the N coordinate system.
    :param phi: Yaw angle of UAV.
    :param gamma: Pitch angle of UAV.
    :param theta: Roll angle of UAV.
    :param alpha: Azimuth angle of camera.
    :param beta: Pitch angle of camera.
    :param distance: Laser ranging distance.
    :param real_k: Real coordinates in k system lat, lon, height.
    :param uav_k: Uav coordinates in k system lat, lon, height.
    :param need_k: Whether to compute the coordinates in the k system.
    :return: The constraint error, MSE_loss.
    """
    target_loss = nn.MSELoss()

    num_points = points.shape[0]
    target_points = points.shape[1]
    in_device = points.device

    # Flatten the tensors.
    points_flat = points.view(num_points * target_points, 3, 1)
    phi_flat = torch.rad2deg(phi.view(num_points * target_points, 1))
    gamma_flat = torch.rad2deg(gamma.view(num_points * target_points, 1))
    theta_flat = torch.rad2deg(theta.view(num_points * target_points, 1))
    alpha_flat = torch.rad2deg(alpha.view(num_points * target_points, 1))
    beta_flat = torch.rad2deg(beta.view(num_points * target_points, 1))
    distance_flat = distance.view(num_points * target_points, 1)

    t_c_n = torch.zeros(num_points * target_points, 3, 1, dtype=torch.float64).to(in_device)
    t_c_n[:, 2, :] = distance_flat

    t_t_n = rotation.c_2_t(t_c_n, alpha_flat, beta_flat)
    # The equations from system t to system b are omitted first, and will be restored later depending on the situation.
    t_n_n = rotation.b_2_n(t_t_n, phi_flat, gamma_flat, theta_flat)
    diff = t_n_n + points_flat

    error_mean = torch.mean(torch.norm(diff, dim=1, p="fro"))
    loss = target_loss(diff, torch.zeros_like(diff))

    if need_k:
        real_k_flat = real_k.view(num_points * target_points, 3, 1)
        uav_k_flat = uav_k.view(num_points * target_points, 3, 1)
        uav_g = rotation.k_2_g(uav_k_flat[:, 0], uav_k_flat[:, 1], uav_k_flat[:, 2])
        t_g_n = rotation.n_2_g(t_n_n, real_k_flat[:, 0].squeeze(), real_k_flat[:, 1].squeeze(), uav_g)
        t_k_n = rotation.g_2_k(t_g_n)
        t_k_n = t_k_n.view(num_points, target_points, 3)
    else:
        t_k_n = None

    return error_mean, loss, t_k_n
