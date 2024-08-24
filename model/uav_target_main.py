# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/7/5 10:09
# @Software: PyCharm

import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchsummaryX import summary
from tqdm import tqdm

from model.uav_target_model import Net
from process import load_data_fold
from tools import EarlyStopping, calculate_constraint_loss, evaluation5class

input_size = 13
hidden_size = 128
num_layers = 2
num_classes = 5
num_reg = 5
heads = 8

summary_printed = False


def train_LSTM(x_train, x_val, learning_rate, model_weight_decay, model_patience,
               model_n_epochs, model_batch_size, iterator, fold_num):
    global summary_printed
    model = Net(in_feats=input_size, hid_feats=hidden_size, num_layers=num_layers,
                class_feats=num_classes, reg_feats=num_reg, num_heads=heads, batch_first=True).to(device)
    target_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=model_weight_decay)
    early_stopping = EarlyStopping(iterator=iterator, fold_num=fold_num, patience=model_patience, verbose=True)

    # Loss used for gradient descent iterations.
    train_losses = []
    # Classification accuracy.
    train_accs = []
    # Classification loss.
    train_class_losses = []
    # Regression loss.
    train_reg_losses = []
    # Actual positioning error (unit/meter).
    train_target_costs = []
    # Reduced margin of error (units/percentage).
    train_target_decreases = []

    val_losses = []
    val_accs = []
    val_class_losses = []
    val_reg_losses = []
    val_target_costs = []
    val_target_decreases = []

    # torch.Size([generate_times*5*0.6, target_times(sequence_length), 32])
    x_train_data = x_train[:, :, 0:31]
    # torch.Size([generate_times*5*0.6])
    x_train_label = x_train[:, :, 31][:, 0].type(torch.int64)
    # torch.Size([generate_times*5*0.2, target_times(sequence_length), 32])
    x_val_data = x_val[:, :, 0:31]
    # torch.Size([generate_times*5*0.2])
    x_val_label = x_val[:, :, 31][:, 0].type(torch.int64)

    # Summary initial delta for 5 angles.
    init_phi_delta = [list() for _ in range(5)]
    init_gamma_delta = [list() for _ in range(5)]
    init_theta_delta = [list() for _ in range(5)]
    init_alpha_delta = [list() for _ in range(5)]
    init_beta_delta = [list() for _ in range(5)]
    for i in range(0, 5):
        init_phi_delta_i = torch.mean(torch.abs(torch.mean((x_val_data[x_val_label == i][:, :, 13]), dim=1))).item()
        init_phi_delta[i].append(init_phi_delta_i)
        init_gamma_delta_i = torch.mean(torch.abs(torch.mean((x_val_data[x_val_label == i][:, :, 14]), dim=1))).item()
        init_gamma_delta[i].append(init_gamma_delta_i)
        init_theta_delta_i = torch.mean(torch.abs(torch.mean((x_val_data[x_val_label == i][:, :, 15]), dim=1))).item()
        init_theta_delta[i].append(init_theta_delta_i)
        init_alpha_delta_i = torch.mean(torch.abs(torch.mean((x_val_data[x_val_label == i][:, :, 16]), dim=1))).item()
        init_alpha_delta[i].append(init_alpha_delta_i)
        init_beta_delta_i = torch.mean(torch.abs(torch.mean((x_val_data[x_val_label == i][:, :, 17]), dim=1))).item()
        init_beta_delta[i].append(init_beta_delta_i)

    print("Initial delta series needed to be predicted: \n")
    print("Delta series in phi: Phi {:.4f} | Gamma {:.4f} | Theta {:.4f} | Alpha {:.4f} | Beta {:.4f}\n"
          "Delta series in gamma: Phi {:.4f} | Gamma {:.4f} | Theta {:.4f} | Alpha {:.4f} | Beta {:.4f}\n"
          "Delta series in theta: Phi {:.4f} | Gamma {:.4f} | Theta {:.4f} | Alpha {:.4f} | Beta {:.4f}\n"
          "Delta series in alpha: Phi {:.4f} | Gamma {:.4f} | Theta {:.4f} | Alpha {:.4f} | Beta {:.4f}\n"
          "Delta series in beta: Phi {:.4f} | Gamma {:.4f} | Theta {:.4f} | Alpha {:.4f} | Beta {:.4f}"
          .format(np.mean(init_phi_delta[0]), np.mean(init_gamma_delta[0]), np.mean(init_theta_delta[0]),
                  np.mean(init_alpha_delta[0]), np.mean(init_beta_delta[0]),
                  np.mean(init_phi_delta[1]), np.mean(init_gamma_delta[1]), np.mean(init_theta_delta[1]),
                  np.mean(init_alpha_delta[1]), np.mean(init_beta_delta[1]),
                  np.mean(init_phi_delta[2]), np.mean(init_gamma_delta[2]), np.mean(init_theta_delta[2]),
                  np.mean(init_alpha_delta[2]), np.mean(init_beta_delta[2]),
                  np.mean(init_phi_delta[3]), np.mean(init_gamma_delta[3]), np.mean(init_theta_delta[3]),
                  np.mean(init_alpha_delta[3]), np.mean(init_beta_delta[3]),
                  np.mean(init_phi_delta[4]), np.mean(init_gamma_delta[4]), np.mean(init_theta_delta[4]),
                  np.mean(init_alpha_delta[4]), np.mean(init_beta_delta[4])))

    if not summary_printed:
        summary(model, torch.randn(model_batch_size, x_train_data.shape[1], input_size).to(device))
        summary_printed = True

    train_loader = DataLoader(
        dataset=TensorDataset(x_train_data, x_train_label),
        batch_size=model_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True)
    val_loader = DataLoader(
        dataset=TensorDataset(x_val_data, x_val_label),
        batch_size=model_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True)
    log_dir = os.path.abspath(os.path.join(os.getcwd().split('OpticalTarget')[0], 'OpticalTarget',
                                           'logs', 'tensorboard',
                                           begin_time_stamp, f"iter_{iterator}", f"fold_{fold_num}"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Terminal: tensorboard --logdir=logs
    # Open: http://localhost:6006/ to see the result
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_graph(model, input_to_model=torch.randn(model_batch_size, x_train_data.shape[1], input_size).to(device))
    # Initialize the indicators.
    accs, F1, F2, F3, F4, F5 = 0, 0, 0, 0, 0, 0
    for epoch in range(model_n_epochs):
        # The overall loss of the two tasks in this epoch.
        avg_loss = []
        # The accuracy of the classification task in this epoch.
        avg_acc = []
        # The loss of the classification task in this epoch.
        avg_class_loss = []
        # The loss of the regression task in this epoch.
        avg_reg_loss = []
        # The actual positioning error (unit/meter) in this epoch.
        avg_target_cost = []
        # The reduced margin of error (units/percentage) in this epoch.
        avg_target_decrease = []

        tqdm_train_loader = tqdm(train_loader)
        for Batch_data, Batch_data_labels in tqdm_train_loader:
            model.train()
            Batch_data = Batch_data.to(device)
            Batch_data_labels = Batch_data_labels.to(device)
            # t_c, p_n(x, y, z) don't need to be trained.
            # # dim 2: 27-dimensional data :
            # # # # # [0:3]   lat_n_relative(target_x), lon_n_relative(target_y), alt_n_relative(target_z)
            # # # # # [3:6]   x, y, z for inverse mirror
            # # # # # [6:11]  phi, gamma, theta, alpha, beta
            # # # # # [11:13] alpha_inverse, beta_inverse
            out_labels, out_regs = model(Batch_data[:, :, 0:13])

            # real and uav coordinates in k system
            # target_k_0 = Batch_data[:, :, 22:25]
            # real_k = Batch_data[:, :, 25:28]
            # uav_k = Batch_data[:, :, 28:31]

            # Regression for constraint loss.
            phi_delta, gamma_delta, theta_delta, alpha_delta, beta_delta = \
                torch.split(out_regs, split_size_or_sections=1, dim=1)
            phi_hat, gamma_hat, theta_hat, alpha_hat, beta_hat = torch.unbind(Batch_data[:, :, 6:11], dim=2)
            phi, gamma, theta, alpha, beta = (phi_hat - phi_delta).unsqueeze(-1), \
                (gamma_hat - gamma_delta).unsqueeze(-1), (theta_hat - theta_delta).unsqueeze(-1), \
                (alpha_hat - alpha_delta).unsqueeze(-1), (beta_hat - beta_delta).unsqueeze(-1)

            distance = Batch_data[:, :, 18]
            p_n = Batch_data[:, :, 19:22]

            # Reformat the shape of the tensors.
            p_n = p_n.unsqueeze(-1)
            distance = distance.unsqueeze(-1)

            # original_cons_cost, _ = calculate_constraint_loss(p_n, phi_hat.unsqueeze(-1), gamma_hat.unsqueeze(-1),
            #                                                   theta_hat.unsqueeze(-1), alpha_hat.unsqueeze(-1),
            #                                                   beta_hat.unsqueeze(-1), distance)
            original_cons_cost = torch.mean(torch.norm(Batch_data[:, :, 0:3], dim=2, p="fro"))
            original_loss = target_loss(Batch_data[:, :, 0:3], torch.zeros_like(Batch_data[:, :, 0:3]))
            cons_cost, cons_loss, _ = calculate_constraint_loss(p_n, phi, gamma, theta, alpha, beta, distance)
            cons_lost_decrease = (original_cons_cost - cons_cost) / original_cons_cost * 100
            # add target loss to avg_target_loss
            avg_target_cost.append(cons_cost.item())
            avg_target_decrease.append(cons_lost_decrease.item())

            # Compute loss
            final_loss_class = F.nll_loss(out_labels, Batch_data_labels)
            # MSE loss
            final_loss_reg = cons_loss / original_loss
            loss = 0.7 * final_loss_reg + 0.3 * final_loss_class
            # zero the parameter gradients
            optimizer.zero_grad()
            # bp loss
            loss.backward()
            # add loss to avg_loss, use item() to get float data, otherwise return a low precision tensor
            avg_class_loss.append(final_loss_class.item())
            avg_reg_loss.append(final_loss_reg.item())
            avg_loss.append(loss.item())
            # update parameters of net
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            # Calculate the number of correct predictions, 'torch.eq()' if the match is returned 1, otherwise 0
            correct = pred.eq(Batch_data_labels).sum().item()
            train_acc = correct / len(Batch_data_labels)
            avg_acc.append(train_acc)

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))
        train_class_losses.append(np.mean(avg_class_loss))
        train_reg_losses.append(np.mean(avg_reg_loss))
        train_target_costs.append(np.mean(avg_target_cost))
        train_target_decreases.append(np.mean(avg_target_decrease))
        writer.add_scalar('Train/Loss', train_losses[epoch], epoch)
        writer.add_scalar('Train/Accuracy', train_accs[epoch], epoch)
        writer.add_scalars('Train/Class_Reg_Loss', {'class': train_class_losses[epoch],
                                                    'reg': train_reg_losses[epoch]}, epoch)
        writer.add_scalar('Train/Target_Error', train_target_costs[epoch], epoch)
        writer.add_scalar('Train/Target_Error_Decrease', train_target_decreases[epoch], epoch)

        # Evaluate validation set performance separately, deactivates dropout during validation run.
        temp_val_loss = []
        temp_val_acc = []
        temp_val_class_loss = []
        temp_val_reg_loss = []
        temp_target_cost = []
        temp_target_decrease = []
        # initialize the list of predicted targets and ground-truth targets
        temp_phi_delta = [list() for _ in range(5)]
        temp_gamma_delta = [list() for _ in range(5)]
        temp_theta_delta = [list() for _ in range(5)]
        temp_alpha_delta = [list() for _ in range(5)]
        temp_beta_delta = [list() for _ in range(5)]

        temp_val_Acc_all = []
        temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1 = [], [], [], []
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2 = [], [], [], []
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3 = [], [], [], []
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], []
        temp_val_Acc5, temp_val_Prec5, temp_val_Recll5, temp_val_F5 = [], [], [], []
        model.eval()

        tqdm_val_loader = tqdm(val_loader)
        for Batch_data, Batch_data_labels in tqdm_val_loader:
            Batch_data = Batch_data.to(device)
            Batch_data_labels = Batch_data_labels.to(device)
            val_out_labels, val_out_regs = model(Batch_data[:, :, 0:13])

            # real and uav coordinates in k system
            # target_k_0 = Batch_data[:, :, 22:25]
            # real_k = Batch_data[:, :, 25:28]
            # uav_k = Batch_data[:, :, 28:31]

            # Regression for constraint loss.
            phi_delta, gamma_delta, theta_delta, alpha_delta, beta_delta = \
                torch.split(val_out_regs, split_size_or_sections=1, dim=1)
            phi_hat, gamma_hat, theta_hat, alpha_hat, beta_hat = torch.unbind(Batch_data[:, :, 6:11], dim=2)
            phi, gamma, theta, alpha, beta = (phi_hat - phi_delta).unsqueeze(-1), \
                (gamma_hat - gamma_delta).unsqueeze(-1), (theta_hat - theta_delta).unsqueeze(-1), \
                (alpha_hat - alpha_delta).unsqueeze(-1), (beta_hat - beta_delta).unsqueeze(-1)

            distance = Batch_data[:, :, 18]
            p_n = Batch_data[:, :, 19:22]

            # Reformat the shape of the tensors.
            p_n = p_n.unsqueeze(-1)
            distance = distance.unsqueeze(-1)

            # original_cons_cost, _ = calculate_constraint_loss(p_n, phi_hat.unsqueeze(-1), gamma_hat.unsqueeze(-1),
            #                                                   theta_hat.unsqueeze(-1), alpha_hat.unsqueeze(-1),
            #                                                   beta_hat.unsqueeze(-1), distance)
            original_cons_cost = torch.mean(torch.norm(Batch_data[:, :, 0:3], dim=2, p="fro"))
            original_loss = target_loss(Batch_data[:, :, 0:3], torch.zeros_like(Batch_data[:, :, 0:3]))
            cons_cost, cons_loss, _ = calculate_constraint_loss(p_n, phi, gamma, theta, alpha, beta, distance)
            cons_lost_decrease = (original_cons_cost - cons_cost) / original_cons_cost * 100
            # add target loss to avg_target_loss
            temp_target_cost.append(cons_cost.item())
            temp_target_decrease.append(cons_lost_decrease.item())

            # Compute loss
            val_loss_class = F.nll_loss(val_out_labels, Batch_data_labels)
            val_loss_reg = cons_loss / original_loss
            loss = 0.7 * val_loss_reg + 0.3 * val_loss_class
            temp_val_class_loss.append(val_loss_class.item())
            temp_val_reg_loss.append(val_loss_reg.item())
            temp_val_loss.append(loss.item())
            _, val_pred = val_out_labels.max(dim=1)
            correct = val_pred.eq(Batch_data_labels).sum().item()

            # Temporarily save the delta of five angles.
            for i in range(0, 5):
                temp_phi_delta_i = torch.mean(torch.abs(
                    torch.rad2deg(phi_delta[Batch_data_labels == i]).squeeze(-1) -
                    torch.mean(Batch_data[Batch_data_labels == i][:, :, 13], dim=1))).item()
                temp_phi_delta[i].append(temp_phi_delta_i)
                temp_gamma_delta_i = torch.mean(torch.abs(
                    torch.rad2deg(gamma_delta[Batch_data_labels == i]).squeeze(-1) -
                    torch.mean(Batch_data[Batch_data_labels == i][:, :, 14], dim=1))).item()
                temp_gamma_delta[i].append(temp_gamma_delta_i)
                temp_theta_delta_i = torch.mean(torch.abs(
                    torch.rad2deg(theta_delta[Batch_data_labels == i]).squeeze(-1) -
                    torch.mean(Batch_data[Batch_data_labels == i][:, :, 15], dim=1))).item()
                temp_theta_delta[i].append(temp_theta_delta_i)
                temp_alpha_delta_i = torch.mean(torch.abs(
                    torch.rad2deg(phi_delta[Batch_data_labels == i]).squeeze(-1) -
                    torch.mean(Batch_data[Batch_data_labels == i][:, :, 16], dim=1))).item()
                temp_alpha_delta[i].append(temp_alpha_delta_i)
                temp_beta_delta_i = torch.mean(torch.abs(
                    torch.rad2deg(beta_delta[Batch_data_labels == i]).squeeze(-1) -
                    torch.mean(Batch_data[Batch_data_labels == i][:, :, 17], dim=1))).item()
                temp_beta_delta[i].append(temp_beta_delta_i)

            val_acc = correct / len(Batch_data_labels)
            temp_val_acc.append(val_acc)

            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, \
                Acc4, Prec4, Recll4, F4, Acc5, Prec5, Recll5, F5 = evaluation5class(val_pred, Batch_data_labels)
            temp_val_Acc_all.append(Acc_all)
            temp_val_Acc1.append(Acc1), temp_val_Prec1.append(Prec1), temp_val_Recll1.append(
                Recll1), temp_val_F1.append(F1)
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2)
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3)
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_Acc5.append(Acc5), temp_val_Prec5.append(Prec5), temp_val_Recll5.append(
                Recll5), temp_val_F5.append(F5)

        val_losses.append(np.mean(temp_val_loss))
        val_accs.append(np.mean(temp_val_acc))
        val_class_losses.append(np.mean(temp_val_class_loss))
        val_reg_losses.append(np.mean(temp_val_reg_loss))
        val_target_costs.append(np.mean(temp_target_cost))
        val_target_decreases.append(np.mean(temp_target_decrease))
        writer.add_scalar('Val/Loss', val_losses[epoch], epoch)
        writer.add_scalar('Val/Accuracy', val_accs[epoch], epoch)
        writer.add_scalars('Val/Class_Reg_Loss', {'class': val_class_losses[epoch],
                                                  'reg': val_reg_losses[epoch]}, epoch)
        writer.add_scalar('Val/Target_Error', val_target_costs[epoch], epoch)
        writer.add_scalar('Val/Target_Error_Decrease', val_target_decreases[epoch], epoch)

        tqdm.write(
            "Iter {:03d} | Fold {:01d} | Epoch {:05d}\n"
            "Train_Loss {:.4f} | Train_Accuracy {:.4f} | Train_target_Error {:.4f}m | Train_target_Decrease {:.2f}%\n"
            "Val_Loss {:.4f} | Val_Accuracy {:.4f} | Val_target_Error {:.4f}m | Val_target_Decrease {:.2f}%\n"
            "Residual error of angle phi  : Phi {:.4f} | Gamma {:.4f} | Theta {:.4f} | Alpha {:.4f} | Beta {:.4f}\n"
            "Residual error of angle gamma: Phi {:.4f} | Gamma {:.4f} | Theta {:.4f} | Alpha {:.4f} | Beta {:.4f}\n"
            "Residual error of angle theta: Phi {:.4f} | Gamma {:.4f} | Theta {:.4f} | Alpha {:.4f} | Beta {:.4f}\n"
            "Residual error of angle alpha: Phi {:.4f} | Gamma {:.4f} | Theta {:.4f} | Alpha {:.4f} | Beta {:.4f}\n"
            "Residual error of angle beta : Phi {:.4f} | Gamma {:.4f} | Theta {:.4f} | Alpha {:.4f} | Beta {:.4f}"
            .format(
                int(iterator), int(fold_num), epoch,
                np.mean(train_losses), np.mean(train_accs), np.mean(avg_target_cost), np.mean(avg_target_decrease),
                np.mean(temp_val_loss), np.mean(temp_val_acc), np.mean(temp_target_cost),
                np.mean(temp_target_decrease),
                np.mean(temp_phi_delta[0]), np.mean(temp_gamma_delta[0]), np.mean(temp_theta_delta[0]),
                np.mean(temp_alpha_delta[0]), np.mean(temp_beta_delta[0]),
                np.mean(temp_phi_delta[1]), np.mean(temp_gamma_delta[1]), np.mean(temp_theta_delta[1]),
                np.mean(temp_alpha_delta[1]), np.mean(temp_beta_delta[1]),
                np.mean(temp_phi_delta[2]), np.mean(temp_gamma_delta[2]), np.mean(temp_theta_delta[2]),
                np.mean(temp_alpha_delta[2]), np.mean(temp_beta_delta[2]),
                np.mean(temp_phi_delta[3]), np.mean(temp_gamma_delta[3]), np.mean(temp_theta_delta[3]),
                np.mean(temp_alpha_delta[3]), np.mean(temp_beta_delta[3]),
                np.mean(temp_phi_delta[4]), np.mean(temp_gamma_delta[4]), np.mean(temp_theta_delta[4]),
                np.mean(temp_alpha_delta[4]), np.mean(temp_beta_delta[4])))

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4)),
               'C5:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc5), np.mean(temp_val_Prec5),
                                                       np.mean(temp_val_Recll5), np.mean(temp_val_F5))]
        print('results:', res)
        accs = np.mean(temp_val_acc)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        F5 = np.mean(temp_val_F5)
        early_stopping(np.mean(temp_val_loss), accs, F1, F2, F3, F4, F5, model, "uav-target")
        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.accs
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            F5 = early_stopping.F5
            break

    writer.close()
    return accs, F1, F2, F3, F4, F5


def show_Hyperparameters(args):
    argsDict = args.__dict__
    print('The training settings are as following')
    for key in argsDict:
        print(key, ':', argsDict[key])
    print('\n')


# Training settings
def parameters_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping.')
    parser.add_argument('--batch_size', type=int, default=256, help='Size of batch.')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
    args = parser.parse_args()
    show_Hyperparameters(args)
    return args


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

args_sim_parser = parameters_setting()
n_epochs = args_sim_parser.n_epochs
patience = args_sim_parser.patience
batch_size = args_sim_parser.batch_size
iterations = args_sim_parser.iterations
lr = args_sim_parser.lr
weight_decay = args_sim_parser.weight_decay
dropout = args_sim_parser.dropout

begin_time = time.time()
begin_time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
print('Start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(begin_time)), '\n')

valid_acc = []
Phi_F1 = []
Gamma_F1 = []
Theta_F1 = []
Alpha_F1 = []
Beta_F1 = []

fold0_x_train, fold0_x_val, _ = load_data_fold(0)
fold1_x_train, fold1_x_val, _ = load_data_fold(1)
fold2_x_train, fold2_x_val, _ = load_data_fold(2)
fold3_x_train, fold3_x_val, _ = load_data_fold(3)
fold4_x_train, fold4_x_val, _ = load_data_fold(4)

for iteration in range(iterations):
    accs0, F1_0, F2_0, F3_0, F4_0, F5_0 = \
        train_LSTM(fold0_x_train, fold0_x_val, lr, weight_decay, patience, n_epochs, batch_size, iteration, 0)
    accs1, F1_1, F2_1, F3_1, F4_1, F5_1 = \
        train_LSTM(fold1_x_train, fold1_x_val, lr, weight_decay, patience, n_epochs, batch_size, iteration, 1)
    accs2, F1_2, F2_2, F3_2, F4_2, F5_2 = \
        train_LSTM(fold2_x_train, fold2_x_val, lr, weight_decay, patience, n_epochs, batch_size, iteration, 2)
    accs3, F1_3, F2_3, F3_3, F4_3, F5_3 = \
        train_LSTM(fold3_x_train, fold3_x_val, lr, weight_decay, patience, n_epochs, batch_size, iteration, 3)
    accs4, F1_4, F2_4, F3_4, F4_4, F5_4 = \
        train_LSTM(fold4_x_train, fold4_x_val, lr, weight_decay, patience, n_epochs, batch_size, iteration, 4)

    # valid acc is mean of 5-fold.
    valid_acc.append((accs0 + accs1 + accs2 + accs3 + accs4) / 5)
    # F1 is also mean of 5-fold.
    Phi_F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
    Gamma_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
    Theta_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
    Alpha_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
    Beta_F1.append((F5_0 + F5_1 + F5_2 + F5_3 + F5_4) / 5)

# 全部除以迭代次数iterations取平均
print("Total_Test_Accuracy: {:.4f} | Phi F1: {:.4f} | Gamma F1: {:.4f} | Theta F1: {:.4f} | "
      "Alpha F1: {:.4f} | Beta F1: {:.4f}".format(sum(valid_acc) / iterations, sum(Phi_F1) / iterations,
                                                  sum(Gamma_F1) / iterations, sum(Theta_F1) / iterations,
                                                  sum(Alpha_F1) / iterations, sum(Beta_F1) / iterations))

end_time = time.time()
print('End time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
print("Total time elapsed: {:.4f}s".format(end_time - begin_time))
torch.cuda.empty_cache()
