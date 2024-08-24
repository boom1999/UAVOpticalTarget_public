# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/7/3 20:22
# @Software: PyCharm

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from process import load_data
from tools import save_fig

round_trip_data, round_trip_data_path = load_data('round_trip')
time_stamp_parts = (os.path.basename(round_trip_data_path).split('.')[0]).split('_')
date_part = time_stamp_parts[2]
time_part = time_stamp_parts[3]
time_stamp = date_part + '_' + time_part

# Save the .csv file.
dir_path_csv = os.path.abspath(os.path.join(os.getcwd().split('OpticalTarget')[0], 'OpticalTarget', 'data', 'csv'))
if not os.path.exists(dir_path_csv):
    os.makedirs(dir_path_csv)
file_name_csv = f'round_trip_{time_stamp}.csv'
# Shape of round_trip_data: (8, target_times, 7, 1)
# dim 0: 8 different round trip.
# dim 1: target_times.
# dim 2: 4 different coordinates: lat, lon, alt, error, lat_n, lon_n, alt_n.
# dim 3: 1.
round_coordinates_ans = round_trip_data[:, :, 0:4].squeeze(3).numpy()
reshaped_round_coordinates_ans = np.empty((128, 32))
for i in range(32):
    layer_index = i // 4
    column_index = i % 4
    reshaped_round_coordinates_ans[:, i] = round_coordinates_ans[layer_index, :, column_index]
header_1 = ["real"] * 4 + ["no_fixed"] * 4 + ["small_fixed"] * 4 + \
           ["phi"] * 4 + ["gamma"] * 4 + ["theta"] * 4 + ["alpha"] * 4 + ["beta"] * 4
header_2 = ["latitude", "longitude", "altitude", "error"] * 8
columns = pd.MultiIndex.from_arrays([header_1, header_2])
pd.DataFrame(reshaped_round_coordinates_ans, columns=columns).to_csv(os.path.join(dir_path_csv, file_name_csv),
                                                                     header=True, index=True)
# Plot the error influence curve.
error_data = round_coordinates_ans[:, :, 3]
df = pd.DataFrame(error_data)
fig = plt.figure(figsize=(10, 6), dpi=600)
for i in range(8):
    smoothed = df.iloc[i].rolling(window=5, center=True).mean()
    plt.plot(smoothed, label=header_1[i*4])
plt.xlabel('Data Index')
plt.ylabel('Error distance (m)')
plt.title('Error influence curve')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
save_name = os.path.join('error_curve_' + time_stamp + '.png')
save_fig(fig, save_name)
