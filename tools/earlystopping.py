# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/7/5 21:21
# @Software: PyCharm

import os

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, iterator, fold_num, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accs = 0
        self.F1 = 0
        self.F2 = 0
        self.F3 = 0
        self.F4 = 0
        self.F5 = 0
        self.val_loss_min = np.Inf
        self.iter = str(iterator)
        self.fold_num = str(fold_num)

    def __call__(self, val_loss, accs, F1, F2, F3, F4, F5, model, model_name):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
            self.F5 = F5
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                print("BEST Accuracy: {:.4f} | "
                      "Phi F1: {:.4f} | Gamma F1: {:.4f} | Theta F1: {:.4f} | Alpha F1: {:.4f} | Beta F1: {:.4f}"
                      .format(self.accs, self.F1, self.F2, self.F3, self.F4, self.F5))
        else:
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
            self.F5 = F5
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...\n'.format(self.val_loss_min,
                                                                                            val_loss))
        model_save_path = os.path.join(os.getcwd().split('OpticalTarget')[0], 'OpticalTarget', 'output',
                                       "uav-target_fold_" + self.fold_num + "_iter_" + self.iter + '.pt')
        torch.save(model.state_dict(), model_save_path)
        self.val_loss_min = val_loss
