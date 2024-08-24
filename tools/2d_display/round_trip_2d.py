# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/7/16 18:46
# @Software: PyCharm

import os.path

import matplotlib.pyplot as plt
import torch
from scipy.spatial import ConvexHull

from process import load_data
from tools import save_fig

# 1. Load round trip positioning data.
round_trip_data, data_path = load_data('round_trip')
save_name_pre = os.path.basename(data_path).split('.')[0]
p_k_real, p_k_n, p_k_s, p_k_1, p_k_2, p_k_3, p_k_4, p_k_5 = round_trip_data
p_k_real = p_k_real[0]

# 2. Get the coordinate of the points.
# Shape: (8, target_times, 7, 1)
# dim 0: 8 different round trip.
# dim 1: target_times.
# dim 2: 4 different coordinates: lat, lon, alt, error, lat_n, lon_n, alt_n.
# dim 3: 1.

# 2.1. Get the coordinate of the points in no-fixed error.

lat_n, lon_n, height_n, error_n, lat_n_n, lon_n_n, height_n_n = p_k_n.unbind(1)

# 2.2. Get the coordinate of the points in small-fixed error.
lat_s, lon_s, height_s, error_s, lat_s_n, lon_s_n, height_s_n = p_k_s.unbind(1)

# 2.3. Get the coordinate of the points in phi.
lat_1, lon_1, height_1, error_1, lat_1_n, lon_1_n, height_1_n = p_k_1.unbind(1)

# 2.4. Get the coordinate of the points in gamma.
lat_2, lon_2, height_2, error_2, lat_2_n, lon_2_n, height_2_n = p_k_2.unbind(1)

# 2.5. Get the coordinate of the points in theta.
lat_3, lon_3, height_3, error_3, lat_3_n, lon_3_n, height_3_n = p_k_3.unbind(1)

# 2.6. Get the coordinate of the points in alpha.
lat_4, lon_4, height_4, error_4, lat_4_n, lon_4_n, height_4_n = p_k_4.unbind(1)

# 2.7. Get the coordinate of the points in beta.
lat_5, lon_5, height_5, error_5, lat_5_n, lon_5_n, height_5_n = p_k_5.unbind(1)


# 3. Plot the 2D figure.

# def the smooth convex hull function
def plot_smooth_convex_hull(x, y, input_fig: plt, color='r'):
    """
    Plot the smooth convex hull.
    :param x: The x coordinate of the points.
    :param y: The y coordinate of the points.
    :param input_fig: The input figure.
    :param color: The color of the convex hull.
    :return: None
    """
    x = x.squeeze()
    y = y.squeeze()
    points = torch.stack((x, y), dim=1)
    # Get the convex hull coordinates of the points.
    hull = ConvexHull(points.numpy())
    vertices = points[hull.vertices]
    # Connect the first and last points (the same).
    vertices = torch.cat((vertices, vertices[0:1]), dim=0)
    # Plot the convex hull.
    input_fig.fill(vertices[:, 0], vertices[:, 1], alpha=0.2, color=color)
    return None


# import numpy as np
#
# from alphashape import alphashape
#
#
# I originally wanted to use the rolling ball algorithm alphashape to draw a smoother boundary, but later found that
# the magnitude difference between latitude, longitude and height was too large, and the effect was not good.
# Then I considered using alphashape to obtain the indices of the boundary in the coordinate system of N or G,
# which is similar in magnitude. The idea and implementation are correct when applied to the K coordinate system.
# After the coordinate points are converted to the K coordinate system, the two-dimensional rendering effect is
# not very good, so I first use the convex hull algorithm above.
# def plot_smooth_alpha_shape(x, y, x_n, y_n, input_fig: plt, color='r', alpha_value=0.15):
#     """
#     Plot the smooth convex hull using Alpha Shape.
#     :param x: The x coordinate of the points.
#     :param y: The y coordinate of the points.
#     :param x_n: The x coordinate of the points in n system.
#     :param y_n: The y coordinate of the points in n system.
#     :param input_fig: The input figure.
#     :param color: The color of the convex hull.
#     :param alpha_value: The alpha parameter for Alpha Shape.
#     :return: None
#     """
#     x_n = x_n.squeeze().numpy()
#     y_n = y_n.squeeze().numpy()
#     points = np.stack((x_n, y_n), axis=1)
#
#     def plot_polygon(polygon, fig, original_x, original_y):
#         # Use iteration to handle multiple boundaries.
#         if polygon.geom_type == 'Polygon':
#             coors_x, coors_y = polygon.exterior.coords.xy
#             # Get the indices of the boundary point in the original sequence in the N coordinate system.
#             indices = []
#             for i in range(len(coors_x)):
#                 mask = np.logical_and(original_x == coors_x[i], original_y == coors_y[i])
#                 match_index = np.where(mask)[0]
#                 if len(match_index) > 0:
#                     indices.append(match_index[0])
#             indices = np.array(indices)
#             # Plot the convex hull in the K coordinate system.
#             fig.fill(x[indices], y[indices], alpha=0.2, color=color)
#         elif polygon.geom_type == 'MultiPolygon':
#             for p in polygon.geoms:
#                 plot_polygon(p, fig, original_x, original_y)
#     # Compute the alpha shape
#     hull = alphashape(points, alpha_value)
#     # Plot the convex hull
#     plot_polygon(hull, input_fig, x_n, y_n)
#     return None


# Get the color map.
cmap = plt.get_cmap('tab10')
colors = cmap.colors

# 3.1. Create a 2D plot in XoY plane.
fig_xoy = plt.figure(figsize=(10, 10), dpi=600)
plt.scatter(p_k_real[1], p_k_real[0], c='b', marker='*', s=100, label='real')
plt.text(p_k_real[1], p_k_real[0], 'Real point', ha='right', va='bottom')
# plt.scatter(lon_n, lat_n, label='no-fixed')
# plt.scatter(lon_s, lat_s, label='small-fixed')
plt.scatter(lon_1, lat_1, marker='.', s=15, label='phi')
plot_smooth_convex_hull(lon_1, lat_1, plt, colors[0])
plt.scatter(lon_2, lat_2, marker='.', s=15, label='gamma')
plot_smooth_convex_hull(lon_2, lat_2, plt, colors[1])
plt.scatter(lon_3, lat_3, marker='.', s=15, label='theta')
plot_smooth_convex_hull(lon_3, lat_3, plt, colors[2])
plt.scatter(lon_4, lat_4, marker='^', s=15, label='alpha')
plot_smooth_convex_hull(lon_4, lat_4, plt, colors[3])
plt.scatter(lon_5, lat_5, marker='^', s=15, label='beta')
plot_smooth_convex_hull(lon_5, lat_5, plt, colors[4])
plt.legend()
plt.ticklabel_format(useOffset=False, style='plain')
plt.xlabel('Longitude (degree)', fontsize=10, fontweight='bold')
plt.ylabel('Latitude (degree)', fontsize=10, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title('Plot the target points in XoY plane', fontsize=14, fontweight='bold')
plt.grid()
plt.show()
save_name = save_name_pre + '_XoY' + '.png'
save_fig(fig_xoy, save_name)

# 3.2. Create a 2D plot in XoZ plane.
fig_xoz = plt.figure(figsize=(10, 10), dpi=600)
plt.scatter(p_k_real[1], p_k_real[2], c='b', marker='*', s=100, label='real')
plt.text(p_k_real[1], p_k_real[2], 'Real point', ha='right', va='bottom')
# plt.scatter(lon_n, height_n, label='no-fixed')
# plt.scatter(lon_s, height_s, label='small-fixed')
plt.scatter(lon_1, height_1, marker='.', s=15, label='phi')
plot_smooth_convex_hull(lon_1, height_1, plt, colors[0])
plt.scatter(lon_2, height_2, marker='.', s=15, label='gamma')
plot_smooth_convex_hull(lon_2, height_2, plt, colors[1])
plt.scatter(lon_3, height_3, marker='.', s=15, label='theta')
plot_smooth_convex_hull(lon_3, height_3, plt, colors[2])
plt.scatter(lon_4, height_4, marker='^', s=15, label='alpha')
plot_smooth_convex_hull(lon_4, height_4, plt, colors[3])
plt.scatter(lon_5, height_5, marker='^', s=15, label='beta')
plot_smooth_convex_hull(lon_5, height_5, plt, colors[4])
plt.legend()
plt.ticklabel_format(useOffset=False, style='plain')
plt.xlabel('Longitude (degree)', fontsize=10, fontweight='bold')
plt.ylabel('Altitude (m)', fontsize=10, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title('Plot the target points in XoZ plane', fontsize=14, fontweight='bold')
plt.grid()
plt.show()
save_name = save_name_pre + '_XoZ' + '.png'
save_fig(fig_xoz, save_name)

# 3.3. Create a 2D plot in YoZ plane.
fig_yoz = plt.figure(figsize=(10, 10), dpi=600)
plt.scatter(p_k_real[0], p_k_real[2], c='b', marker='*', s=100, label='real')
plt.text(p_k_real[0], p_k_real[2], 'Real point', ha='right', va='bottom')
# plt.scatter(lat_n, height_n, label='no-fixed')
# plt.scatter(lat_s, height_s, label='small-fixed')
plt.scatter(lat_1, height_1, marker='.', s=15, label='phi')
plot_smooth_convex_hull(lat_1, height_1, plt, colors[0])
plt.scatter(lat_2, height_2, marker='.', s=15, label='gamma')
plot_smooth_convex_hull(lat_2, height_2, plt, colors[1])
plt.scatter(lat_3, height_3, marker='.', s=15, label='theta')
plot_smooth_convex_hull(lat_3, height_3, plt, colors[2])
plt.scatter(lat_4, height_4, marker='^', s=15, label='alpha')
plot_smooth_convex_hull(lat_4, height_4, plt, colors[3])
plt.scatter(lat_5, height_5, marker='^', s=15, label='beta')
plot_smooth_convex_hull(lat_5, height_5, plt, colors[4])
plt.legend()
plt.ticklabel_format(useOffset=False, style='plain')
plt.xlabel('Latitude (degree)', fontsize=10, fontweight='bold')
plt.ylabel('Altitude (m)', fontsize=10, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title('Plot the target points in YoZ plane', fontsize=14, fontweight='bold')
plt.grid()
plt.show()
save_name = save_name_pre + '_YoZ' + '.png'
save_fig(fig_yoz, save_name)
