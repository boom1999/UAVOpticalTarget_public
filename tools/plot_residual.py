# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/9/9 16:29
# @Software: PyCharm

import os

import matplotlib.pyplot as plt
from tools import save_fig
import pandas as pd
import numpy as np


def plot_angles(angle_name, real_data, predict_data, fold_num, iterator, target_times=1):
    """
    Plot and save the true angle and predicted angle curves.
    :param angle_name: Five angles: Phi, Gamma, Theta, Alpha, Beta.
    :param real_data: True angle data.
    :param predict_data: Output angle data from the model.
    :param fold_num: The fold number.
    :param iterator: The iterator number.
    :param target_times: The number of target points in one flight series.
    :return: None.
    """
    curves_dir = os.path.abspath(os.path.join(
        os.getcwd().split('OpticalTarget')[0], 'OpticalTarget', 'data', 'fig', 'angles'))
    curves_dir = os.path.join(curves_dir, angle_name)
    save_name = os.path.join(angle_name + '_fold_' + str(fold_num) + "_iter_" + str(iterator) + '.png')

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
    sample_size = real_data.shape[1]
    real_data = np.mean(real_data.reshape(-1, target_times), axis=1)
    predict_data = np.mean(predict_data.reshape(-1, target_times), axis=1)
    indices = np.random.choice(len(real_data), size=sample_size, replace=False)

    sample_real_data = real_data[indices]
    sample_predict_data = predict_data[indices]

    axs[0].plot(sample_real_data, label=angle_name + ' real delta')
    axs[0].set_xlabel('Data Index')
    axs[0].set_ylabel('Angle delta (deg)')
    axs[0].set_title('Real delta for angle ' + angle_name)
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.5)

    axs[1].plot(sample_real_data-sample_predict_data, label=angle_name+' residual')
    axs[1].set_xlabel('Data Index')
    axs[1].set_ylabel('Angle delta (deg)')
    axs[1].set_title('Prediction residual for angle ' + angle_name)
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # Set the ordinate range of the right subplot to match that of the left subplot.
    axs[1].set_ylim(axs[0].get_ylim())

    plt.tight_layout()
    plt.show()
    save_fig(fig, save_name, curves_dir)


def save_output(original_data, predict_data, fold_num, iterator):
    """
    Save the true angle and predicted angle curves to csv.
    Input shape numpy.ndarray shape: (batch_size, target_times, 9)
    :param original_data: Five true angle_delta data, true label and true target coordinates(k system).
    :param predict_data: Five predicted angle_delta data, predicted label and predicted target coordinates(k system).
    :param fold_num: The fold number.
    :param iterator: The iterator number.
    :return: None.
    """
    csv_dir = os.path.abspath(os.path.join(
        os.getcwd().split('OpticalTarget')[0], 'OpticalTarget', 'data', 'csv'))
    # Save the .csv file.
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    csv_name = os.path.join(csv_dir, 'predict_fold_' + str(fold_num) + "_iter_" + str(iterator) + '.csv')

    # Reshape
    output_data = np.concatenate((original_data, predict_data), axis=-1)
    df = pd.DataFrame(output_data)
    columns = pd.MultiIndex.from_arrays([["real"] * 9 + ["predict"] * 9,
                                         ["phi", "gamma", "theta", "alpha", "beta", "lat", "lon", "alt", "label"] * 2])
    df.columns = columns
    # The data set is too large, do not use EXCEL to read it (the upper limit of the number of rows is exceeded).
    df.to_csv(csv_name, header=True, index=False)
