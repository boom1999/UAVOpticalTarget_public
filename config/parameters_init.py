# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/4/26 16:10
# @Software: PyCharm

import argparse

import torch


def show_parameters(args):
    args_dict = args.__dict__
    print('The parameters of the UAV target simulation are as following......')
    for key in args_dict:
        print(key, ':', args_dict[key])
    print('\n')


def parameters_setting():
    # UAV target simulation parameters.

    sim_parser = argparse.ArgumentParser(description='UAV target simulation parameter settings.')
    # Number of simulations.
    sim_parser.add_argument('--times', type=int, default=10000, help='Targeting repeat times.')
    # Length of the random sequence.
    sim_parser.add_argument('--length', type=int, default=5000, help='Length of the random sequence.')
    # The location parameters of the UAV.
    sim_parser.add_argument('--Hp', type=float, default=4000.0, help='The height at which the UAV is flown.')
    sim_parser.add_argument('--Hp_delta', type=float, default=10.0, help='The standard deviation of '
                                                                         'the altitude at which the UAV is flown.')
    sim_parser.add_argument('--Lp', type=float, default=118.825632, help='The longitude at which the UAV is flown.')
    sim_parser.add_argument('--Bp', type=float, default=32.032975, help='The latitude at which the UAV is flown.')
    sim_parser.add_argument('--LB_delta', type=float, default=0.0002, help='The standard deviation of '
                                                                           'the latitude and longitude error.')
    # Attitude Angle Parameters of the UAV. All angles entered are in degrees.
    # For phi, positive angle means that the drone turns left relative to the flight path.
    sim_parser.add_argument('--phi', type=float, default=10.0, help='Yaw angle of UAV.')
    # For gamma, the positive angle means that the nose of the drone is down relative to the horizontal plane.
    sim_parser.add_argument('--gamma', type=float, default=3.0, help='Pitch angle of UAV.')
    # For theta, the positive angle means that the right wing of the drone is down relative to the horizontal plane.
    sim_parser.add_argument('--theta', type=float, default=4.0, help='Roll angle of UAV.')
    sim_parser.add_argument('--attitude_delta_phi', type=float, default=0.1,
                            help='The standard deviation of the UAV attitude angle (phi) error.')
    sim_parser.add_argument('--attitude_delta_gamma', type=float, default=0.02,
                            help='The standard deviation of the UAV attitude angle (gamma) error.')
    sim_parser.add_argument('--attitude_delta_theta', type=float, default=0.02,
                            help='The standard deviation of the UAV attitude angle (theta) error.')

    sim_parser.add_argument('--phi_delta_i', type=float, default=0.0, help='The installation error of '
                                                                           'the yaw angle of UAV.')
    sim_parser.add_argument('--gamma_delta_i', type=float, default=0.0, help='The installation error of '
                                                                             'the pitch angle of UAV.')
    sim_parser.add_argument('--theta_delta_i', type=float, default=0.0, help='The installation error of '
                                                                             'the roll angle of UAV.')
    sim_parser.add_argument('--phi_delta_v', type=float, default=0.0, help='The vibration error of '
                                                                           'the yaw angle of UAV.')
    sim_parser.add_argument('--gamma_delta_v', type=float, default=0.0, help='The vibration error of '
                                                                             'the pitch angle of UAV.')
    sim_parser.add_argument('--theta_delta_v', type=float, default=0.0, help='The vibration error of '
                                                                             'the roll angle of UAV.')
    sim_parser.add_argument('--axis_delta_i', type=float, default=0.1,
                            help='The standard deviation of the optical axis stabilized platform error.')
    sim_parser.add_argument('--axis_delta_v_phi', type=float, default=0.05,
                            help='Uniformly distributed error maximum in the yaw direction for the angular.')
    sim_parser.add_argument('--axis_delta_v_gamma', type=float, default=0.1,
                            help='Uniformly distributed error maximum in the pitch direction for the angular.')
    sim_parser.add_argument('--axis_delta_v_theta', type=float, default=0.1,
                            help='Uniformly distributed error maximum in the roll direction for the angular.')

    # The rotation parameters of the optoelectronic platform.
    # For alpha, the positive angle is vertically left, and the negative angle is vertically right.
    sim_parser.add_argument('--alpha', type=float, default=30.0, help='Azimuth angle of camera.')
    # For beta, the positive angle is horizontal upward, and the negative angle is horizontal downward.
    sim_parser.add_argument('--beta', type=float, default=-50.0, help='Pitch angle of camera.')
    sim_parser.add_argument('--camera_delta', type=float, default=0.05, help='The standard deviation of '
                                                                             'the camera optical axis pointing error.')
    # LiDAR ranging parameters.
    sim_parser.add_argument('--R', type=float, default=5000.0, help='Laser ranging distance.')
    sim_parser.add_argument('--R_delta', type=float, default=3.0, help='The standard deviation of '
                                                                       'the distance error of the laser ranging.')
    sim_parser.add_argument('--target_times', type=int, default=128, help='The times for cooperative targeting.')

    # Fixed error parameters.
    sim_parser.add_argument('--phi_fixed_delta', type=float, default=2.0, help='Fixed error (deg) of yaw angle.')
    sim_parser.add_argument('--gamma_fixed_delta', type=float, default=2.0, help='Fixed error (deg) of pitch angle.')
    sim_parser.add_argument('--theta_fixed_delta', type=float, default=2.0, help='Fixed error (deg) of roll angle.')
    sim_parser.add_argument('--alpha_fixed_delta', type=float, default=2.0, help='Fixed error (deg) of azimuth angle.')
    sim_parser.add_argument('--beta_fixed_delta', type=float, default=2.0, help='Fixed error (deg) of pitch angle.')

    args = sim_parser.parse_args([])

    show_parameters(args)
    return args


args_sim_parser = parameters_setting()
# for attr in vars(args_sim_parser):
#     exec(f"{attr} = args_sim_parser.{attr}")

times = args_sim_parser.times
length = args_sim_parser.length
Hp = args_sim_parser.Hp
Hp_delta = args_sim_parser.Hp_delta
Lp = args_sim_parser.Lp
Bp = args_sim_parser.Bp
LB_delta = args_sim_parser.LB_delta
phi = args_sim_parser.phi
gamma = args_sim_parser.gamma
theta = args_sim_parser.theta
attitude_delta_phi = args_sim_parser.attitude_delta_phi
attitude_delta_gamma = args_sim_parser.attitude_delta_gamma
attitude_delta_theta = args_sim_parser.attitude_delta_theta
phi_delta_i = args_sim_parser.phi_delta_i
gamma_delta_i = args_sim_parser.gamma_delta_i
theta_delta_i = args_sim_parser.theta_delta_i
axis_delta_i = args_sim_parser.axis_delta_i
phi_delta_v = args_sim_parser.phi_delta_v
gamma_delta_v = args_sim_parser.gamma_delta_v
theta_delta_v = args_sim_parser.theta_delta_v
axis_delta_v_phi = args_sim_parser.axis_delta_v_phi
axis_delta_v_gamma = args_sim_parser.axis_delta_v_gamma
axis_delta_v_theta = args_sim_parser.axis_delta_v_theta
alpha = args_sim_parser.alpha
beta = args_sim_parser.beta
camera_delta = args_sim_parser.camera_delta
R = args_sim_parser.R
R_delta = args_sim_parser.R_delta
target_times = args_sim_parser.target_times
phi_fixed_delta = args_sim_parser.phi_fixed_delta
gamma_fixed_delta = args_sim_parser.gamma_fixed_delta
theta_fixed_delta = args_sim_parser.theta_fixed_delta
alpha_fixed_delta = args_sim_parser.alpha_fixed_delta
beta_fixed_delta = args_sim_parser.beta_fixed_delta


def parameter_initialization_no_err(length_random=length):
    """
    Initialize the parameters of the UAV and the camera without error.
    :param length_random: The number of random samples.
    :return:
    """
    # Generate the location parameters of the UAV in k system.
    B_ne = torch.ones(length_random) * Bp
    L_ne = torch.ones(length_random) * Lp
    H_ne = torch.ones(length_random) * Hp
    p_k_position_ne = torch.stack((B_ne, L_ne, H_ne), dim=0)

    # Error-free camera optical axis pointing parameters.
    alpha_c_ne = torch.ones(length_random) * alpha
    beta_c_ne = torch.ones(length_random) * beta

    # Error-free UAV attitude angle parameters of installation and vibration err.
    phi_delta_i_b_ne = torch.zeros(length_random)
    gamma_delta_i_b_ne = torch.zeros(length_random)
    theta_delta_i_b_ne = torch.zeros(length_random)
    phi_delta_v_b_ne = torch.zeros(length_random)
    gamma_delta_v_b_ne = torch.zeros(length_random)
    theta_delta_v_b_ne = torch.zeros(length_random)

    # Error-free UAV attitude angle parameters.
    phi_b_ne = torch.ones(length_random) * phi
    gamma_b_ne = torch.ones(length_random) * gamma
    theta_b_ne = torch.ones(length_random) * theta

    # Generate the LiDAR ranging
    distance_ne = torch.ones(length_random) * R

    return p_k_position_ne, \
        phi_b_ne, gamma_b_ne, theta_b_ne, phi_delta_i_b_ne, gamma_delta_i_b_ne, theta_delta_i_b_ne, \
        phi_delta_v_b_ne, gamma_delta_v_b_ne, theta_delta_v_b_ne, alpha_c_ne, beta_c_ne, distance_ne


def parameter_initialization_full_err(length_random=length):
    """
    Initialize the parameters of the UAV and the camera with error.
    :param length_random: The number of random samples.
    :return:
    """
    # Define the mean and standard deviation for UAV position in K system.
    p_k_std = torch.tensor([LB_delta, LB_delta, Hp_delta])
    p_k_means = torch.tensor([Bp, Lp, Hp])
    # Generate a random 3x1000 tensor with values sampled from a normal distribution
    # with a mean of 0 and standard deviation of 1
    p_k_position = torch.randn((3, length_random))
    p_k_position = p_k_position * p_k_std.view(-1, 1) + p_k_means.view(-1, 1)

    # Generate the attitude angle parameters of the UAV in B system.
    att_angle_std = torch.tensor([attitude_delta_phi, attitude_delta_gamma, attitude_delta_theta])
    att_angle_means = torch.tensor([phi, gamma, theta])
    att_angle = torch.randn((3, length_random))
    att_angle = att_angle * att_angle_std.view(-1, 1) + att_angle_means.view(-1, 1)

    # Generate the installation error parameters of the UAV in T system.
    install_err_std = torch.tensor([axis_delta_i, axis_delta_i, axis_delta_i])
    install_err_means = torch.tensor([phi_delta_i, gamma_delta_i, theta_delta_i])
    install_err = torch.randn((3, length_random))
    install_err = install_err * install_err_std.view(-1, 1) + install_err_means.view(-1, 1)

    # Generate the vibration error parameters of the UAV in T system.
    vibra_err_delta_max = torch.tensor([axis_delta_v_phi, axis_delta_v_gamma, axis_delta_v_theta])
    vibra_err_means = torch.tensor([phi_delta_v, gamma_delta_v, theta_delta_v])
    vibra_err = torch.FloatTensor(3, length_random).uniform_(-1)
    vibra_err = vibra_err * vibra_err_delta_max.view(-1, 1) + vibra_err_means.view(-1, 1)

    # Generate the camera optical axis pointing parameters of the UAV in C system.
    camera_axis_std = torch.tensor([camera_delta, camera_delta])
    camera_axis_means = torch.tensor([alpha, beta])
    camera_axis = torch.randn((2, length_random))
    camera_axis = camera_axis * camera_axis_std.view(-1, 1) + camera_axis_means.view(-1, 1)

    # Generate the LiDAR parameters of the UAV in C system.
    distance_std = torch.tensor([R_delta])
    distance_means = torch.tensor([R])
    distance = torch.randn(length_random)
    distance = distance * distance_std + distance_means

    return \
        p_k_position, \
        att_angle[0], att_angle[1], att_angle[2], \
        install_err[0], install_err[1], install_err[2], \
        vibra_err[0], vibra_err[1], vibra_err[2], \
        camera_axis[0], camera_axis[1], distance


def parameter_analysis(length_random=length):
    """
    The latitude and longitude are evenly distributed from small to large, -0.0005~0.0005.
    and the height is also uniformly distributed from low to high -50~50.
    :param length_random: The number of random samples.
    :return:
    """
    LB_start_value = 0
    LB_end_value = 0.0005
    Hp_start_value = 0
    Hp_end_value = 50
    LB_series = torch.linspace(LB_start_value, LB_end_value, length_random)
    Hp_series = torch.linspace(Hp_start_value, Hp_end_value, length_random)
    p_k_series = torch.stack([LB_series, LB_series, Hp_series], dim=0)
    p_k_means = torch.tensor([Bp, Lp, Hp])
    p_k_position = p_k_means.view(-1, 1) + p_k_series

    # The attitude angle parameters error is uniformly distributed from low to high 0~0.5.
    attitude_delta_start_value = 0
    attitude_delta_end_value = 0.5
    attitude_delta_series = torch.linspace(attitude_delta_start_value, attitude_delta_end_value, length_random)
    att_angle_means = torch.tensor([phi, gamma, theta])
    att_angle = att_angle_means.view(-1, 1) + attitude_delta_series

    # Generate the installation error is uniformly distributed from low to high 0~0.5.
    axis_delta_start_value = 0
    axis_delta_end_value = 0.5
    axis_delta_series = torch.linspace(axis_delta_start_value, axis_delta_end_value, length_random)
    install_err_means = torch.tensor([phi_delta_i, gamma_delta_i, theta_delta_i])
    install_err = install_err_means.view(-1, 1) + axis_delta_series
    vibra_err_means = torch.tensor([phi_delta_v, gamma_delta_v, theta_delta_v])
    vibra_err = vibra_err_means.view(-1, 1) + axis_delta_series

    # The camera optical axis error is uniformly distributed from low to high 0~0.5.
    camera_delta_start_value = 0
    camera_delta_end_value = 0.5
    camera_delta_series = torch.linspace(camera_delta_start_value, camera_delta_end_value, length_random)
    camera_axis_means = torch.tensor([alpha, beta])
    camera_axis = camera_axis_means.view(-1, 1) + camera_delta_series

    # The laser distance error is also uniformly distributed from low to high 0~50.
    distance_start_value = 0
    distance_end_value = 50
    distance_series = torch.linspace(distance_start_value, distance_end_value, length_random)
    distance_means = torch.tensor([R])
    distance = distance_means + distance_series

    return \
        p_k_position, \
        att_angle[0], att_angle[1], att_angle[2], \
        install_err[0], install_err[1], install_err[2], \
        vibra_err[0], vibra_err[1], vibra_err[2], \
        camera_axis[0], camera_axis[1], distance
