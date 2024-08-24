# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/5/3 20:15
# @Software: PyCharm

import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from process import error_calculate, load_data
from tools import save_fig

# no-err location data
real_data = load_data('no_err')[0]
real_k_data = real_data[0]
real_g_data = real_data[1]
# full-err location data
target_data, target_name = load_data('full_err')
target_k_data = target_data[0]
target_g_data = target_data[1]

# compute error in latitude
error_data = (target_k_data[:, :, ] - real_k_data[0]).squeeze(2)

# plot histogram of error
fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

# set the number of bins
bins = int(real_k_data.shape[0] / 100)

# plot error in latitude
# The bin with the most occurrences is the darkest.
max_index0_0 = int(torch.argmax(torch.histc(error_data[:, 0], bins=bins)))
ax0_0 = sns.histplot(error_data[:, 0], kde=True, ax=axs[0, 0], bins=bins)
palette0_0 = sns.color_palette("crest", n_colors=max_index0_0-1) + \
             sns.color_palette("crest_r", n_colors=bins-max_index0_0+1)
for i, rect in enumerate(ax0_0.patches):
    color = palette0_0[i]
    rect.set_color(color)
axs[0, 0].set_title(f"Latitude", fontsize=14, fontweight='bold')
axs[0, 0].set_xlabel("Latitude error (degree)", fontsize=12, fontweight='bold')
axs[0, 0].set_ylabel("Count (times)", fontsize=12, fontweight='bold')

# plot error in longitude
max_index0_1 = int(torch.argmax(torch.histc(error_data[:, 1], bins=bins)))
ax0_1 = sns.histplot(error_data[:, 1], kde=True, ax=axs[0, 1], bins=bins)
palette0_1 = sns.color_palette("crest", n_colors=max_index0_1-1) + \
             sns.color_palette("crest_r", n_colors=bins-max_index0_1+1)
for i, rect in enumerate(ax0_1.patches):
    color = palette0_1[i]
    rect.set_color(color)
axs[0, 1].set_title(f"Longitude", fontsize=14, fontweight='bold')
axs[0, 1].set_xlabel("Longitude error (degree)", fontsize=12, fontweight='bold')
axs[0, 1].set_ylabel("Count (times)", fontsize=12, fontweight='bold')

# plot error in altitude
max_index1_0 = int(torch.argmax(torch.histc(error_data[:, 2], bins=bins)))
ax1_0 = sns.histplot(error_data[:, 2], kde=True, ax=axs[1, 0], bins=bins)
palette1_0 = sns.color_palette("flare", n_colors=max_index1_0-1) + \
             sns.color_palette("flare_r", n_colors=bins-max_index1_0+1)
for i, rect in enumerate(ax1_0.patches):
    color = palette1_0[i]
    rect.set_color(color)
axs[1, 0].set_title(f"Altitude ", fontsize=14, fontweight='bold')
axs[1, 0].set_xlabel("Altitude error (m)", fontsize=12, fontweight='bold')
axs[1, 0].set_ylabel("Count (times)", fontsize=12, fontweight='bold')

# compute distance error distance.shape = torch.Size([series_length])
distance = error_calculate(target_data, real_data)[8]

# plot error in distance
max_index1_1 = int(torch.argmax(torch.histc(distance, bins=bins)))
ax1_1 = sns.histplot(distance, kde=True, ax=axs[1, 1], bins=bins)
palette1_1 = sns.color_palette("magma_r", n_colors=max_index1_1-1) + \
             sns.color_palette("magma", n_colors=bins-max_index1_1+1)
for i, rect in enumerate(ax1_1.patches):
    color = palette1_1[i]
    rect.set_color(color)
axs[1, 1].set_title(f"Distance", fontsize=14, fontweight='bold')
axs[1, 1].set_xlabel("Distance error (m)", fontsize=12, fontweight='bold')
axs[1, 1].set_ylabel("Count (times)", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# Save the plot.
save_name = os.path.basename(target_name).split('.')[0] + 'error_statistics.png'
save_fig(fig, save_name)
