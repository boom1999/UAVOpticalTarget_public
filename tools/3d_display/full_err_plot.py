# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/5/2 15:54
# @Software: PyCharm

import os.path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from process import error_calculate, load_data
from tools import save_fig

# full-err location data
p_position = load_data('full_err')[0]

# no-err location data
p_position_ne = load_data('no_err')[0]

# Calculate the distances between predicted and real points.
distance = error_calculate(p_position, p_position_ne)[8]

# Create a 3D plot.
fig = plt.figure(dpi=600)
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

# Extract x, y, z coordinates from tensor
p_k_position = p_position[0].squeeze(2)
lat = p_k_position[:, 0]
lon = p_k_position[:, 1]
height = p_k_position[:, 2]

# Plot the coordinate as point in 3D space.
scatter = ax.scatter(lon, lat, height, c=distance, marker='^', s=10, alpha=2/5, cmap='viridis')
dummy_scatter = ax.scatter([], [], [], marker='*', s=10, label='Real target point')
cbar = fig.colorbar(scatter, location='right', pad=0.1, shrink=0.7, aspect=15)
cbar.set_label('Error distance (m)')
ax.legend(handles=[dummy_scatter], loc='best', markerscale=2, fontsize=10)

# Set the axis limits.
ax.ticklabel_format(useOffset=False, style='plain')
# Set the axis labels.
ax.set_xlabel('Longitude (degree)', fontsize=8, fontweight='bold')
ax.set_ylabel('Latitude (degree)', fontsize=8, fontweight='bold')
ax.yaxis.labelpad = 10
ax.set_zlabel('Altitude (m)', fontsize=8, fontweight='bold')

# Show the plot.
plt.show()
# Save the plot.
save_name = os.path.basename(load_data('full_err')[1]).split('.')[0] + '.png'
save_fig(fig, save_name)
