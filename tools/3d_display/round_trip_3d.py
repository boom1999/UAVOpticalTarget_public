# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/6/27 19:39
# @Software: PyCharm


import os.path

import torch
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from process import load_data
from tools import save_fig

# Load round trip positioning data.
round_trip_data, data_path = load_data('round_trip')
p_k_real, p_k_n, p_k_s, p_k_1, p_k_2, p_k_3, p_k_4, p_k_5 = round_trip_data
p_k_real = p_k_real[0]

# Create a 3D plot.
fig = plt.figure(dpi=600)
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

# Plot the coordinate as point in 3D space.
ax.scatter(p_k_real[1], p_k_real[0], p_k_real[2], c='b', marker='*', s=20, label='real')

# Get the color map.
cmap = plt.get_cmap('tab10')
colors = cmap.colors


# Plot the convex hull.
# def the 3d smooth convex hull function
def plot_3d_smooth_convex_hull(x, y, z, input_ax: Axes3D, color='r'):
    """
    Plot the 3d convex hull.
    :param x: The x coordinate.
    :param y: The y coordinate.
    :param z: The z coordinate.
    :param input_ax: The Axes3D object.
    :param color: The color of the convex hull.
    :return: None
    """
    x = x.squeeze()
    y = y.squeeze()
    z = z.squeeze()
    points = torch.stack((x, y, z), dim=1)
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        simplex_points = points[simplex]
        polygon = Poly3DCollection([simplex_points], alpha=0.1)
        polygon.set_facecolor(color)
        input_ax.add_collection3d(polygon)
    return None


# Plot the coordinate as point in 3D space.
# 1. Plot no-fixed error coordinates.
lat_n = p_k_n[:, 0]
lon_n = p_k_n[:, 1]
height_n = p_k_n[:, 2]
# scatter_n = ax.scatter(lon_n, lat_n, height_n, marker='*', s=5, alpha=3/5, label='no-fixed')
# 2. Plot small-fixed error coordinates.
lat_s = p_k_s[:, 0]
lon_s = p_k_s[:, 1]
height_s = p_k_s[:, 2]
# scatter_s = ax.scatter(lon_s, lat_s, height_s, marker='*', s=5, alpha=3/5, label='small-fixed')
# 3. Plot fixed error coordinates in phi.
lat_1 = p_k_1[:, 0]
lon_1 = p_k_1[:, 1]
height_1 = p_k_1[:, 2]
scatter_1 = ax.scatter(lon_1, lat_1, height_1, marker='.', s=5, alpha=3 / 5, label='phi')
plot_3d_smooth_convex_hull(lon_1, lat_1, height_1, ax, color=colors[0])
# 4. Plot fixed error coordinates in gamma.
lat_2 = p_k_2[:, 0]
lon_2 = p_k_2[:, 1]
height_2 = p_k_2[:, 2]
scatter_2 = ax.scatter(lon_2, lat_2, height_2, marker='.', s=5, alpha=3 / 5, label='gamma')
plot_3d_smooth_convex_hull(lon_2, lat_2, height_2, ax, color=colors[1])
# 5. Plot fixed error coordinates in theta.
lat_3 = p_k_3[:, 0]
lon_3 = p_k_3[:, 1]
height_3 = p_k_3[:, 2]
scatter_3 = ax.scatter(lon_3, lat_3, height_3, marker='.', s=5, alpha=3 / 5, label='theta')
plot_3d_smooth_convex_hull(lon_3, lat_3, height_3, ax, color=colors[2])
# 6. Plot fixed error coordinates in alpha.
lat_4 = p_k_4[:, 0]
lon_4 = p_k_4[:, 1]
height_4 = p_k_4[:, 2]
scatter_4 = ax.scatter(lon_4, lat_4, height_4, marker='^', s=5, alpha=3 / 5, label='alpha')
plot_3d_smooth_convex_hull(lon_4, lat_4, height_4, ax, color=colors[3])
# 7. Plot fixed error coordinates in beta.
lat_5 = p_k_5[:, 0]
lon_5 = p_k_5[:, 1]
height_5 = p_k_5[:, 2]
scatter_5 = ax.scatter(lon_5, lat_5, height_5, marker='^', s=5, alpha=3 / 5, label='beta')
plot_3d_smooth_convex_hull(lon_5, lat_5, height_5, ax, color=colors[4])

# Calibrate the boundary of real coordinates.
np_64_x = np.float64(p_k_real[1].item())
np_64_y = np.float64(p_k_real[0].item())
np_64_z = np.float64(p_k_real[2].item())
ax.text(np_64_x, np_64_y, np_64_z, 'Real point', ha='right', va='bottom', fontsize=8)
ax.plot([np_64_x, np_64_x], [np_64_y, np_64_y], [ax.get_zlim()[0], np_64_z],
        linestyle='dashed', color='black', alpha=1 / 5)
ax.plot([np_64_x, np_64_x], [np_64_y, ax.get_ylim()[1]], [np_64_z, np_64_z],
        linestyle='dashed', color='black', alpha=1 / 5)
ax.plot([ax.get_xlim()[0], np_64_x], [np_64_y, np_64_y], [np_64_z, np_64_z],
        linestyle='dashed', color='black', alpha=1 / 5)

ax.scatter(p_k_real[1], p_k_real[0], ax.get_zlim()[0], c='black', marker='*', s=1)
ax.scatter(ax.get_xlim()[0], p_k_real[0], p_k_real[2], c='black', marker='*', s=1)
ax.scatter(p_k_real[1], ax.get_ylim()[1], p_k_real[2], c='black', marker='*', s=1)

# # Set the axis limits.
# x_min, x_max = np.min(np.concatenate([lon_n.numpy(), lon_s.numpy(),
#                       lon_1.numpy(), lon_2.numpy(), lon_3.numpy(), lon_4.numpy(), lon_5.numpy()])), \
#     np.max(np.concatenate([lon_n.numpy(), lon_s.numpy(),
#            lon_1.numpy(), lon_2.numpy(), lon_3.numpy(), lon_4.numpy(), lon_5.numpy()]))
# y_min, y_max = np.min(np.concatenate([lat_n.numpy(), lat_s.numpy(),
#                       lat_1.numpy(), lat_2.numpy(), lat_3.numpy(), lat_4.numpy(), lat_5.numpy()])), \
#     np.max(np.concatenate([lat_n.numpy(), lat_s.numpy(),
#            lat_1.numpy(), lat_2.numpy(), lat_3.numpy(), lat_4.numpy(), lat_5.numpy()]))
# z_min, z_max = np.min(np.concatenate([height_n.numpy(), height_s.numpy(),
#                       height_1.numpy(), height_2.numpy(), height_3.numpy(), height_4.numpy(), height_5.numpy()])), \
#     np.max(np.concatenate([height_n.numpy(), height_s.numpy(),
#            height_1.numpy(), height_2.numpy(), height_3.numpy(), height_4.numpy(), height_5.numpy()]))

# Set the axis limits.
ax.ticklabel_format(useOffset=False, style='plain')
# Set the axis labels.
ax.set_xlabel('Longitude (degree)', fontsize=8, fontweight='bold')
ax.set_ylabel('Latitude (degree)', fontsize=8, fontweight='bold')
ax.yaxis.labelpad = 10
ax.set_zlabel('Altitude (m)', fontsize=8, fontweight='bold')
plt.legend(loc='best', fontsize=8)  # Show the plot.
plt.show()
# Save the plot.
save_name = os.path.basename(data_path).split('.')[0] + '.png'
save_fig(fig, save_name)
