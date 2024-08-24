# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/7/5 10:50
# @Software: PyCharm

import os

import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from process import load_data

# Load data
original_data, original_data_path = load_data('target_data')

# Split data into 5 folds
fold_data_indices = []
label = original_data[:, :, 31, :][:, 0, :].squeeze(1)
for train_val_indices, test_indices in \
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(original_data, label):
    # train data + val data : test data = 4 : 1
    train_val_data_indices = train_val_indices.tolist()
    test_data_indices = test_indices.tolist()

    fold_data_indices.append((train_val_data_indices, test_data_indices))

# Save data
dir_path = os.path.abspath(os.path.join(os.getcwd().split('OpticalTarget')[0], 'OpticalTarget', 'data', 'fold_data'))
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
train_data_path = os.path.join(dir_path, "train_data")
val_data_path = os.path.join(dir_path, "valid_data")
test_data_path = os.path.join(dir_path, "test_data")

for fold, (train_val_indices, test_indices) in enumerate(fold_data_indices, start=0):
    train_val_data = original_data[train_val_indices]
    test_data = original_data[test_indices]

    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.25, random_state=42)
    # train data : val data = 3 : 1
    train_data = original_data[train_indices]
    val_data = original_data[val_indices]

    # train data : val data : test data = 3 : 1 : 1
    torch.save(train_data, f"{train_data_path}_fold{fold}.pt")

    torch.save(val_data, f"{val_data_path}_fold{fold}.pt")

    torch.save(test_data, f"{test_data_path}_fold{fold}.pt")
