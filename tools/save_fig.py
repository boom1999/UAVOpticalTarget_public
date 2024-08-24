# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/5/6 10:12
# @Software: PyCharm

def save_fig(fig, save_name, save_dir_path=None):
    """
    Save the plot.
    :param fig: The current figure.
    :param save_name: The name of the file to save.
    :param save_dir_path: The path to save the plot (optional).
    :return:
    """
    import os.path
    # Check if file already exists
    if save_dir_path is None:
        save_dir_path = \
            os.path.abspath(os.path.join(os.getcwd().split('OpticalTarget')[0], 'OpticalTarget', 'data', 'fig'))
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    save_path = os.path.join(save_dir_path, save_name)
    # Check if file already exists
    if os.path.exists(save_path):
        os.remove(save_path)
    fig.savefig(save_path, dpi=600)
    return
