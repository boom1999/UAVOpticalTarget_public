# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/4/26 15:09
# @Software: PyCharm

import math

import torch

# -------------------------- WGS84 Earth's parameters --------------------------
# semi-major axes
EARTH_A = 6378137.0
# semi-minor axes
EARTH_B = 6356752.314245
# Earth's first eccentricity
earth_e1 = math.sqrt((math.pow(EARTH_A, 2) - math.pow(EARTH_B, 2)) / math.pow(EARTH_A, 2))
# Earth's second eccentricity
earth_e2 = math.sqrt((math.pow(EARTH_A, 2) - math.pow(EARTH_B, 2)) / math.pow(EARTH_B, 2))

# -------------------------- Check if GPU is available --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------- 1. 3D transformation matrix. --------------------------
def rotation_x(phi: torch.Tensor) -> torch.Tensor:
    """
    The coordinate system rotates phi degrees around the X circle. And the yaw angle of UAV.
    :param phi: The angle of rotation.
    :return: The rotation matrix.
    """
    if torch.numel(phi) == 1:
        # Handles rotation matrices of scalar type.
        r = torch.tensor([[1, 0, 0],
                         [0, torch.cos(phi), torch.sin(phi)],
                         [0, -torch.sin(phi), torch.cos(phi)]])
    else:
        # Handles rotation matrices of tensor type.
        in_device = phi.device
        length = phi.shape[0]
        r = torch.zeros((length, 3, 3), dtype=torch.float64).to(in_device)

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        r[:, 0, 0] = 1.0
        r[:, 1, 1] = cos_phi
        r[:, 1, 2] = sin_phi
        r[:, 2, 1] = -sin_phi
        r[:, 2, 2] = cos_phi
    return r


def rotation_y(gamma: torch.Tensor) -> torch.Tensor:
    """
    The coordinate system rotates gamma degrees around the Y circle. And the pitch angle of UAV.
    :param gamma: The angle of rotation.
    :return: The rotation matrix.
    """
    if torch.numel(gamma) == 1:
        # Handles rotation matrices of scalar type.
        r = torch.tensor([[torch.cos(gamma), 0, -torch.sin(gamma)],
                          [0, 1, 0],
                          [torch.sin(gamma), 0, torch.cos(gamma)]])
    else:
        # Handles rotation matrices of tensor type.
        in_device = gamma.device
        length = gamma.shape[0]
        r = torch.zeros((length, 3, 3), dtype=torch.float64).to(in_device)

        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)

        r[:, 0, 0] = cos_gamma
        r[:, 0, 2] = -sin_gamma
        r[:, 1, 1] = 1.0
        r[:, 2, 0] = sin_gamma
        r[:, 2, 2] = cos_gamma
    return r


def rotation_z(theta: torch.Tensor) -> torch.Tensor:
    """
    The coordinate system rotates theta degrees around the Z circle. And the roll angle of UAV.
    :param theta: The angle of rotation.
    :return: The rotation matrix.
    """
    if torch.numel(theta) == 1:
        # Handles rotation matrices of scalar type.
        r = torch.tensor([[torch.cos(theta), torch.sin(theta), 0],
                          [-torch.sin(theta), torch.cos(theta), 0],
                          [0, 0, 1]])
    else:
        # Handles rotation matrices of tensor type.
        in_device = theta.device
        length = theta.shape[0]
        r = torch.zeros((length, 3, 3), dtype=torch.float64).to(in_device)

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r[:, 0, 0] = cos_theta
        r[:, 0, 1] = sin_theta
        r[:, 1, 0] = -sin_theta
        r[:, 1, 1] = cos_theta
        r[:, 2, 2] = 1.0
    return r


# -------------------------- 2. Coordinate system transformation process. --------------------------
# Rot^A_B: A sequence of rotation matrices for transforming from coordinate system A to B.
# t_c: The coordinates of the target point in the camera coordinate system.
# t_g: The coordinates of the target point in the Cartesian coordinate system of the earth.
# p_g: The coordinates of the UAV in the Cartesian coordinate system of the earth
# B, L, H: The latitude, longitude and altitude of the UAV.

# 1. t_c = Rot^C_T * Rot^T_B * Rot^B_N * Rot^N_G * (t_g - p_g) ==>
#    t_g = (Rot^C_T * Rot^T_B * Rot^B_N * Rot^N_G)^(-1) * t_c + p_g ==>
#    (A * B * C)^(-1) = C^(-1) * B^(-1) * A^(-1) ==>
#    t_g = (Rot^N_G)^(-1) * (Rot^B_N)^(-1) * (Rot^T_B)^(-1) * (Rot^C_T)^(-1) * t_c + p_g
# 2. t_g ==> B, L, H


def c_2_t(t_c, alpha_degrees, beta_degrees):
    """
    Apply rotation function to each value in t_c. (Rot^C_T)^(-1) * t_c
    :param t_c: The coordinates of the target point in the camera coordinate system (c).
    :param alpha_degrees: The angle of rotation around the X circle.
    :param beta_degrees: The angle of rotation around the Y circle.
    :return: The coordinates of the target point in the base coordinate system (t).
    """
    # Convert angle from degrees to radians.
    alpha_radians = torch.deg2rad(alpha_degrees).squeeze()
    beta_radians = torch.deg2rad(beta_degrees).squeeze()
    # Rotated matrix.
    rotated_12 = rotation_x(alpha_radians)
    rotated_13 = rotation_y(beta_radians)
    # Coordinate transformation formula.
    t_t = torch.linalg.inv(rotated_12) @ torch.linalg.inv(rotated_13) @ t_c
    return t_t


def t_2_b(t_t, phi_delta_i_b_degrees, gamma_delta_i_b_degrees, theta_delta_i_b_degrees,
          phi_delta_v_b_degrees, gamma_delta_v_b_degrees, theta_delta_v_b_degrees):
    """
    Apply rotation function to each value from t system to b system. (Rot^T_B)^(-1) * (Rot^C_T)^(-1) * t_c
    :param t_t: The coordinates of the target point in the base coordinate system (t).
    :param phi_delta_i_b_degrees: The installation error of the camera around the X circle.
    :param gamma_delta_i_b_degrees: The installation error of the camera around the Y circle.
    :param theta_delta_i_b_degrees: The installation error of the camera around the Z circle.
    :param phi_delta_v_b_degrees: The vibration error of the IMU around the X circle.
    :param gamma_delta_v_b_degrees: The vibration error of the IMU around the Y circle.
    :param theta_delta_v_b_degrees: The vibration error of the IMU around the Z circle.
    :return: The coordinates of the target point in the body coordinate system (b).
    """
    # Convert angle from degrees to radians.
    phi_delta_i_b_radians = torch.deg2rad(phi_delta_i_b_degrees).squeeze()
    gamma_delta_i_b_radians = torch.deg2rad(gamma_delta_i_b_degrees).squeeze()
    theta_delta_i_b_radians = torch.deg2rad(theta_delta_i_b_degrees).squeeze()
    phi_delta_v_b_radians = torch.deg2rad(phi_delta_v_b_degrees).squeeze()
    gamma_delta_v_b_radians = torch.deg2rad(gamma_delta_v_b_degrees).squeeze()
    theta_delta_v_b_radians = torch.deg2rad(theta_delta_v_b_degrees).squeeze()
    # Rotated matrix.
    rotated_6 = rotation_x(phi_delta_i_b_radians)
    rotated_7 = rotation_y(gamma_delta_i_b_radians)
    rotated_8 = rotation_z(theta_delta_i_b_radians)
    rotated_9 = rotation_x(phi_delta_v_b_radians)
    rotated_10 = rotation_y(gamma_delta_v_b_radians)
    rotated_11 = rotation_z(theta_delta_v_b_radians)
    # Coordinate transformation formula.
    t_b = torch.linalg.inv(rotated_6) @ torch.linalg.inv(rotated_7) @ torch.linalg.inv(rotated_8) @ \
        torch.linalg.inv(rotated_9) @ torch.linalg.inv(rotated_10) @ torch.linalg.inv(rotated_11) @ t_t
    return t_b


def b_2_n(t_b, phi_b_degrees, gamma_b_degrees, theta_b_degrees):
    """
    Apply rotation function to each value from the body coordinate system
    to the local horizontal geographic coordinate system (n). (Rot^B_N)^(-1) * (Rot^T_B)^(-1) * (Rot^C_T)^(-1) * t_c
    :param t_b: The coordinates of the target point in the body coordinate system (b).
    :param phi_b_degrees: The latitude of the target point.
    :param gamma_b_degrees: The longitude of the target point.
    :param theta_b_degrees: The height of the target point.
    :return: The coordinates of the target point in the local horizontal geographic coordinate system (n).
    """
    # Convert angle from degrees to radians.
    phi_b_radians = torch.deg2rad(phi_b_degrees).squeeze()
    gamma_b_radians = torch.deg2rad(gamma_b_degrees).squeeze()
    theta_b_radians = torch.deg2rad(theta_b_degrees).squeeze()
    # Rotated matrix.
    rotated_3 = rotation_x(phi_b_radians)
    rotated_4 = rotation_y(gamma_b_radians)
    rotated_5 = rotation_z(theta_b_radians)
    # Coordinate transformation formula.
    t_n = torch.linalg.inv(rotated_3) @ torch.linalg.inv(rotated_4) @ torch.linalg.inv(rotated_5) @ t_b
    return t_n


def n_2_g(t_n, B_k_degrees, L_k_degrees, p_g):
    """
    Apply rotation function to each value from the local horizontal geographic coordinate system (n)
    to space Cartesian coordinate system (g).
    (Rot^N_G)^(-1) * (Rot^B_N)^(-1) * (Rot^T_B)^(-1) * (Rot^C_T)^(-1) * t_c + p_g
    :param t_n: The coordinates of the target point in the local horizontal geographic coordinate system (n).
    :param B_k_degrees: The latitude of the IMU.
    :param L_k_degrees: The longitude of the IMU.
    :param p_g: The coordinates of the IMU in the space Cartesian coordinate system (g).
    :return: The coordinates of the target point in the space Cartesian coordinate system (g).
    """
    # Convert angle from degrees to radians.
    L_k_radians = torch.deg2rad(L_k_degrees).squeeze()
    B_k_radians = torch.deg2rad(B_k_degrees).squeeze()
    # Rotated matrix.
    rotated_1 = rotation_z(L_k_radians)
    rotated_2 = rotation_y(-B_k_radians)
    # Coordinate transformation formula.
    t_g = torch.linalg.inv(rotated_1) @ torch.linalg.inv(rotated_2) @ t_n + p_g
    return t_g


def g_2_n(t_g, B_k_degrees, L_k_degrees, p_g):
    """
    Apply rotation function to each value from the space Cartesian coordinate system (g)
    to the local horizontal geographic coordinate system (n).
    (Rot^B_N) * (Rot^T_B) * (Rot^C_T) * (Rot^N_G) * (t_g - p_g)
    :param t_g: The relative coordinate origin in the space Cartesian coordinate system (g).
    :param B_k_degrees: The latitude of the relative coordinate.
    :param L_k_degrees: The longitude of the relative coordinate.
    :param p_g: The coordinates of the UAV in the space Cartesian coordinate system (g).
    :return: The coordinates of the target point in the local horizontal geographic coordinate system (n).
    """
    in_device = t_g.device
    rotated_1 = torch.zeros(3, 3, dtype=torch.float64).to(in_device)
    rotated_2 = torch.zeros(3, 3, dtype=torch.float64).to(in_device)
    # Convert angle from degrees to radians.
    L_k_radians = torch.deg2rad(L_k_degrees)
    B_k_radians = torch.deg2rad(B_k_degrees)
    rotated_1[:, :] = rotation_z(L_k_radians)
    rotated_2[:, :] = rotation_y(-B_k_radians)
    t_n = rotated_2 @ rotated_1 @ (t_g - t_g)
    p_n = rotated_2 @ rotated_1 @ (p_g - t_g)
    return t_n, p_n


def k_2_g(B_k_degrees, L_k_degrees, H_k):
    """
    B, L, H: (Lat, Lon, Alt) ==> t_g(x, y, z)
    :param B_k_degrees: Geodetic latitude, angle system,
    specifying that south latitude is negative, and the range is [-90, 90].
    :param L_k_degrees: Geodetic longitude, angle system,
    west longitude is specified as negative, the range is [-180, 180].
    :param H_k: Altitude, earth height, unit m.
    :return: t_g(x, y, z), unit m
    """
    in_device = B_k_degrees.device
    # Check the input type.
    if isinstance(B_k_degrees, float):
        B_k_degrees = torch.tensor(B_k_degrees)
    if isinstance(L_k_degrees, float):
        L_k_degrees = torch.tensor(L_k_degrees)
    if isinstance(H_k, float):
        H_k = torch.tensor(H_k)
    length = B_k_degrees.shape[0] if len(B_k_degrees.shape) > 0 else 1
    if B_k_degrees.dim() == 1:
        B_k_degrees = B_k_degrees.unsqueeze(1)
        L_k_degrees = L_k_degrees.unsqueeze(1)
        H_k = H_k.unsqueeze(1)
    p_g = torch.zeros(length, 3, 1, dtype=torch.float64).to(in_device)
    # Convert angle from degrees to radians.
    B_k_radians = torch.deg2rad(B_k_degrees)
    L_k_radians = torch.deg2rad(L_k_degrees)
    # The radius of curvature of the prime vertical.
    r_pv = EARTH_A / torch.sqrt(1 - (earth_e1 * torch.sin(B_k_radians)).pow(2))
    # Handles scalar tensors and vector tensors.
    if len(B_k_degrees.shape) == 0:
        B_k_radians = B_k_radians.unsqueeze(0)
        L_k_radians = L_k_radians.unsqueeze(0)
        H_k = H_k.unsqueeze(0)
        r_pv = r_pv.unsqueeze(0)
    p_g[:, 0, :] = (r_pv + H_k) * torch.cos(B_k_radians) * torch.cos(L_k_radians)
    p_g[:, 1, :] = (r_pv + H_k) * torch.cos(B_k_radians) * torch.sin(L_k_radians)
    p_g[:, 2, :] = (r_pv * (1 - math.pow(earth_e1, 2)) + H_k) * torch.sin(B_k_radians)
    return p_g


def g_2_k(t_g):
    """
    t_g(x, y, z) ==> B, L, H: (Lat, Lon, Alt)
    :param t_g: t_g(x, y, z), unit m.
    :return: t_k(B, L, H), geodetic latitude, longitude and altitude, unit m.
    """
    x_g = t_g[:, 0, :]
    y_g = t_g[:, 1, :]
    z_g = t_g[:, 2, :]
    # Auxiliary quantity theta_earth.
    theta_earth = torch.arctan(EARTH_A * z_g / (EARTH_B * torch.sqrt(x_g.pow(2) + y_g.pow(2))))
    B_k_radians = torch.arctan((z_g + EARTH_B * math.pow(earth_e2, 2) * (torch.sin(theta_earth)).pow(3))
                               / (torch.sqrt(x_g.pow(2) + y_g.pow(2)) - EARTH_A * math.pow(earth_e1, 2)
                                  * (torch.cos(theta_earth)).pow(3)))

    # East longitude, west longitude conversion.
    # length = x_g.shape[0]
    # L_k_radians = torch.zeros(length, 1)
    # L_k_degrees = torch.zeros(length, 1)
    # for i in range(length):
    #     if x_g[i] == 0 and y_g[i] > 0:
    #         L_k_radians[i] = 90.0
    #     elif x_g[i] == 0 and y_g[i] < 0:
    #         L_k_radians[i] = -90.0
    #     elif x_g[i] < 0 <= y_g[i]:
    #         L_k_radians[i] = torch.arctan(y_g[i] / x_g[i])
    #         L_k_degrees[i] = torch.rad2deg(L_k_radians[i])
    #         L_k_degrees[i] = L_k_degrees[i] + 180
    #     elif x_g[i] < 0 and y_g[i] < 0:
    #         L_k_radians[i] = torch.arctan(y_g[i] / x_g[i])
    #         L_k_degrees[i] = torch.rad2deg(L_k_radians[i])
    #         L_k_degrees[i] = L_k_degrees[i] - 180
    #     else:  # x_g > 0
    #         L_k_radians[i] = torch.arctan(y_g[i] / x_g[i])
    #         L_k_degrees[i] = torch.rad2deg(L_k_radians[i])

    # Use vectorization to reduce if-else loop judgments.
    L_k_radians = torch.where(
        x_g == 0,
        torch.sign(y_g) * math.pi / 2,
        torch.atan(y_g / x_g)
    )
    L_k_degrees = torch.rad2deg(L_k_radians)
    L_k_degrees += 180 * (x_g < 0) * (y_g >= 0)
    L_k_degrees -= 180 * (x_g < 0) * (y_g < 0)
    # The radius of curvature of the prime vertical.
    r_pv = EARTH_A / torch.sqrt(1 - (earth_e1 * torch.sin(B_k_radians)).pow(2))
    H_K = torch.sqrt(x_g.pow(2) + y_g.pow(2)) / torch.cos(B_k_radians) - r_pv
    # Convert angle from radians to degrees.
    B_k_degrees = torch.rad2deg(B_k_radians)
    t_k = torch.stack((B_k_degrees, L_k_degrees, H_K), dim=1)
    return t_k
