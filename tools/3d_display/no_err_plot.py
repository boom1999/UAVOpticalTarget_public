# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/5/2 14:45
# @Software: PyCharm


import os.path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from process import load_data
from tools import save_fig

# Load error-free positioning data.
p_k_position_ne = load_data('no_err')[0][0][0]

# Create a 3D plot.
fig = plt.figure(dpi=600)
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

# Plot the coordinate as point in 3D space.
ax.scatter(p_k_position_ne[1], p_k_position_ne[0], p_k_position_ne[2], c='b', marker='*', s=20, label='error-free')

# Set the axis limits.
ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
ax.set_zlim(ax.get_zlim()[0], ax.get_zlim()[1])

np_64_x = np.float64(p_k_position_ne[1].item())
np_64_y = np.float64(p_k_position_ne[0].item())
np_64_z = np.float64(p_k_position_ne[2].item())
ax.plot([np_64_x, np_64_x], [np_64_y, np_64_y], [ax.get_zlim()[0], np_64_z],
        linestyle='dashed', color='black', alpha=1/5)
ax.plot([np_64_x, np_64_x], [np_64_y, ax.get_ylim()[1]], [np_64_z, np_64_z],
        linestyle='dashed', color='black', alpha=1/5)
ax.plot([ax.get_xlim()[0], np_64_x], [np_64_y, np_64_y], [np_64_z, np_64_z],
        linestyle='dashed', color='black', alpha=1/5)

ax.scatter(p_k_position_ne[1], p_k_position_ne[0], ax.get_zlim()[0], c='black', marker='x', s=5)
ax.scatter(ax.get_xlim()[0], p_k_position_ne[0], p_k_position_ne[2], c='black', marker='x', s=5)
ax.scatter(p_k_position_ne[1], ax.get_ylim()[1], p_k_position_ne[2], c='black', marker='x', s=5)

# Set the axis limits.
ax.ticklabel_format(useOffset=False, style='plain')
# Set the axis labels.
ax.set_xlabel('Longitude (degree)', fontsize=8, fontweight='bold')
ax.set_ylabel('Latitude (degree)', fontsize=8, fontweight='bold')
ax.set_zlabel('Altitude (m)', fontsize=8, fontweight='bold')

# Get the current figure.
pig = plt.gcf()

# Show the plot.
plt.show()
# Save the plot.
save_name = os.path.basename(load_data('no_err')[1]).split('.')[0] + '.png'
save_fig(fig, save_name)
