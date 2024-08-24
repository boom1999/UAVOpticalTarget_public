# -*- coding: utf-8 -*-
# @Author  : Heisenberg
# @Time    : 2023/7/8 17:09
# @Software: PyCharm

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torchsummaryX import summary
from tqdm import tqdm

from model.uav_target_model import Net
from process import load_data_fold
from tools import calculate_constraint_loss, evaluation5class, save_fig, save_output, plot_angles

input_size = 13
hidden_size = 128
num_layers = 2
num_classes = 5
num_reg = 5
heads = 8

summary_printed = False
target_times = 1


def predict_LSTM(x_test, model_batch_size, iterator, fold_num):
    global summary_printed
    global target_times

    model_path = os.path.join(os.getcwd().split('OpticalTarget')[0], 'OpticalTarget', 'output',
                              "uav-target_fold_" + str(fold_num) + "_iter_" + str(iterator) + '.pt')
    model = Net(in_feats=input_size, hid_feats=hidden_size, num_layers=num_layers,
                class_feats=num_classes, reg_feats=num_reg, num_heads=heads, batch_first=True).to(device)
    target_loss = nn.MSELoss()
    model.load_state_dict(torch.load(model_path))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters for this model:  {total_params}")

    # torch.Size([generate_times*5*0.2, target_times(sequence_length), 32])
    x_test_data = x_test[:, :, 0:31]
    # torch.Size([generate_times*5*0.2])
    x_test_label = x_test[:, :, 31][:, 0].type(torch.int64)

    if not summary_printed:
        summary(model, torch.randn(model_batch_size, x_test_data.shape[1], input_size).to(device))
        summary_printed = True

    test_loader = DataLoader(
        dataset=TensorDataset(x_test_data, x_test_label),
        batch_size=model_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True)

    temp_test_losses = []
    temp_test_accs = []
    temp_test_target_costs = []
    temp_test_target_decrease = []
    temp_test_Acc_all = []
    temp_test_Acc1, temp_test_Prec1, temp_test_Recll1, temp_test_F1 = [], [], [], []
    temp_test_Acc2, temp_test_Prec2, temp_test_Recll2, temp_test_F2 = [], [], [], []
    temp_test_Acc3, temp_test_Prec3, temp_test_Recll3, temp_test_F3 = [], [], [], []
    temp_test_Acc4, temp_test_Prec4, temp_test_Recll4, temp_test_F4 = [], [], [], []
    temp_test_Acc5, temp_test_Prec5, temp_test_Recll5, temp_test_F5 = [], [], [], []

    tqdm_test_loader = tqdm(test_loader)
    real_labels = []
    predict_labels = []
    real_phi, real_gamma, real_theta, real_alpha, real_beta = [], [], [], [], []
    predict_phi, predict_gamma, predict_theta, predict_alpha, predict_beta = [], [], [], [], []
    t_k_0, t_k_1 = [], []
    for Batch_data, Batch_data_labels in tqdm_test_loader:
        model.eval()
        Batch_data = Batch_data.to(device)
        Batch_data_labels = Batch_data_labels.to(device)
        out_labels, out_regs = model(Batch_data[:, :, 0:13])

        # real and uav coordinates in k system
        target_k_0 = Batch_data[:, :, 22:25]
        real_k = Batch_data[:, :, 25:28]
        uav_k = Batch_data[:, :, 28:31]

        # For regression
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
        cons_cost, cons_loss, target_k_1 = calculate_constraint_loss(
            p_n, phi, gamma, theta, alpha, beta, distance, real_k, uav_k, need_k=True)
        cons_loss_decrease = (original_cons_cost - cons_cost) / original_cons_cost * 100

        # add constraint loss.
        temp_test_target_costs.append(cons_cost.item())
        temp_test_target_decrease.append(cons_loss_decrease.item())

        # Compute loss
        test_loss_reg = cons_loss / original_loss
        test_loss_class = F.nll_loss(out_labels, Batch_data_labels)
        loss = test_loss_reg * 0.7 + test_loss_class * 0.3
        temp_test_losses.append(loss.item())
        _, test_pred = out_labels.max(dim=1)
        correct = test_pred.eq(Batch_data_labels).sum().item()
        val_acc = correct / len(Batch_data_labels)
        temp_test_accs.append(val_acc)

        # Save data for confusion matrix and angles.
        real_labels.append(Batch_data_labels)
        predict_labels.append(test_pred)

        target_times = Batch_data[:, :, 0].shape[1]

        real_phi.append(Batch_data[:, :, 13].cpu().detach().numpy())
        predict_phi.append(torch.rad2deg(phi_delta).repeat(1, target_times).cpu().detach().numpy())

        real_gamma.append(Batch_data[:, :, 14].cpu().detach().numpy())
        predict_gamma.append(torch.rad2deg(gamma_delta).repeat(1, target_times).cpu().detach().numpy())

        real_theta.append(Batch_data[:, :, 15].cpu().detach().numpy())
        predict_theta.append(torch.rad2deg(theta_delta).repeat(1, target_times).cpu().detach().numpy())

        real_alpha.append(Batch_data[:, :, 16].cpu().detach().numpy())
        predict_alpha.append(torch.rad2deg(alpha_delta).repeat(1, target_times).cpu().detach().numpy())

        real_beta.append(Batch_data[:, :, 17].cpu().detach().numpy())
        predict_beta.append(torch.rad2deg(beta_delta).repeat(1, target_times).cpu().detach().numpy())

        t_k_0.append(target_k_0.cpu().detach().numpy())
        t_k_1.append(target_k_1.cpu().detach().numpy())

        # 计算每个类别的指标
        Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, \
            Acc4, Prec4, Recll4, F4, Acc5, Prec5, Recll5, F5 = evaluation5class(test_pred, Batch_data_labels)

        temp_test_Acc_all.append(Acc_all)

        temp_test_Acc1.append(Acc1), temp_test_Prec1.append(Prec1), temp_test_Recll1.append(
            Recll1), temp_test_F1.append(F1)
        temp_test_Acc2.append(Acc2), temp_test_Prec2.append(Prec2), temp_test_Recll2.append(
            Recll2), temp_test_F2.append(F2)
        temp_test_Acc3.append(Acc3), temp_test_Prec3.append(Prec3), temp_test_Recll3.append(
            Recll3), temp_test_F3.append(F3)
        temp_test_Acc4.append(Acc4), temp_test_Prec4.append(Prec4), temp_test_Recll4.append(
            Recll4), temp_test_F4.append(F4)
        temp_test_Acc5.append(Acc5), temp_test_Prec5.append(Prec5), temp_test_Recll5.append(
            Recll5), temp_test_F5.append(F5)

    # Plot the heatmap of confusion matrix.
    class_labels = ['Phi', 'Gamma', 'Theta', 'Alpha', 'Beta']
    test_confusion_matrix = confusion_matrix(
        torch.cat(real_labels).cpu().numpy(), torch.cat(predict_labels).cpu().numpy())
    class_totals = test_confusion_matrix.sum(axis=1)
    normalized_matrix = test_confusion_matrix.astype('float') / class_totals[:, np.newaxis]
    fig = plt.figure(figsize=(10, 8), dpi=600)
    sns.set(style="whitegrid")
    heatmap = sns.heatmap(normalized_matrix, annot=True, fmt='.2%', cmap='Blues',
                          annot_kws={"size": 14, "color": "black"},
                          xticklabels=class_labels, yticklabels=class_labels)
    color_bar = heatmap.collections[0].colorbar
    color_bar.set_ticks(np.linspace(0, 0.8, 5))
    color_bar.set_ticklabels(['0%', '20%', '40%', '60%', '80%'])
    color_bar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    plt.xlabel('Predicted Labels', fontsize=14, fontweight='bold')
    plt.ylabel('True Labels', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    heatmap_dir = os.path.abspath(os.path.join(
        os.getcwd().split('OpticalTarget')[0], 'OpticalTarget', 'data', 'fig', 'heatmap'))
    heatmap_name = os.path.join('heatmap_fold_' + str(fold_num) + "_iter_" + str(iterator) + '.png')
    save_fig(fig, heatmap_name, heatmap_dir)

    # Transform (batch_num, batch_size, target_times) to (batch_size, target_times, 1)
    real_phi = np.array(real_phi)
    predict_phi = np.array(predict_phi)
    real_gamma = np.array(real_gamma)
    predict_gamma = np.array(predict_gamma)
    real_theta = np.array(real_theta)
    predict_theta = np.array(predict_theta)
    real_alpha = np.array(real_alpha)
    predict_alpha = np.array(predict_alpha)
    real_beta = np.array(real_beta)
    predict_beta = np.array(predict_beta)

    # Plot the true angle and predicted angle curves.
    plot_angles('X raw (Phi and Alpha)', real_phi + real_alpha, predict_phi + predict_alpha,
                fold_num, iterator, target_times=target_times)
    plot_angles('Y raw (Gamma)', real_gamma, predict_gamma, fold_num, iterator, target_times=target_times)
    plot_angles('Z raw (Theta)', real_theta, predict_theta, fold_num, iterator, target_times=target_times)
    plot_angles('Y raw (Beta)', real_beta, predict_beta, fold_num, iterator, target_times=target_times)

    # Save the true angle and predicted angle curves to csv (five angles).
    real_angles = np.concatenate((np.expand_dims(real_phi.reshape(-1), axis=1),
                                  np.expand_dims(real_gamma.reshape(-1), axis=1),
                                  np.expand_dims(real_theta.reshape(-1), axis=1),
                                  np.expand_dims(real_alpha.reshape(-1), axis=1),
                                  np.expand_dims(real_beta.reshape(-1), axis=1)), axis=-1)
    real_labels = [label.unsqueeze(1).expand(batch_size, target_times) for label in real_labels]
    real_labels = np.expand_dims(np.array(torch.cat(real_labels).cpu().numpy()).reshape(-1), axis=1)
    t_k_0 = np.array(t_k_0).reshape(-1, 3)
    original_data = np.concatenate((real_angles, t_k_0, real_labels), axis=-1)

    predict_angles = np.concatenate((np.expand_dims(predict_phi.reshape(-1), axis=1),
                                     np.expand_dims(predict_gamma.reshape(-1), axis=1),
                                     np.expand_dims(predict_theta.reshape(-1), axis=1),
                                     np.expand_dims(predict_alpha.reshape(-1), axis=1),
                                     np.expand_dims(predict_beta.reshape(-1), axis=1)), axis=-1)
    predict_labels = [label.unsqueeze(1).expand(batch_size, target_times) for label in predict_labels]
    predict_labels = np.expand_dims(np.array(torch.cat(predict_labels).cpu().numpy()).reshape(-1), axis=1)
    t_k_1 = np.array(t_k_1).reshape(-1, 3)
    predict_data = np.concatenate((predict_angles, t_k_1, predict_labels), axis=-1)

    save_output(original_data, predict_data, fold_num, iterator)

    accs = np.mean(temp_test_accs)
    target_loss = np.mean(temp_test_target_costs)
    target_decrease = np.mean(temp_test_target_decrease)
    F1 = np.mean(temp_test_F1)
    F2 = np.mean(temp_test_F2)
    F3 = np.mean(temp_test_F3)
    F4 = np.mean(temp_test_F4)
    F5 = np.mean(temp_test_F5)
    tqdm.write(
        "Iter: {:d}|Fold: {:d} | Test_Accuracy: {:.4f} | Target_Loss: {:.4f}m | Target_Decrease: {:.2f}%\n"
        "Phi F1: {:.4f} | Gamma F1: {:.4f} | Theta F1: {:.4f} | Alpha F1: {:.4f} | Beta F1: {:.4f}"
        .format(iterator, fold_num, accs, target_loss, target_decrease, F1, F2, F3, F4, F5))
    return accs, target_loss, target_decrease, F1, F2, F3, F4, F5


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

iterations = 10
batch_size = 512
test_acc = []
mean_target_loss = []
mean_target_decrease = []
Phi_F1 = []
Gamma_F1 = []
Theta_F1 = []
Alpha_F1 = []
Beta_F1 = []

_, _, fold_0_x_test = load_data_fold(0)
_, _, fold_1_x_test = load_data_fold(1)
_, _, fold_2_x_test = load_data_fold(2)
_, _, fold_3_x_test = load_data_fold(3)
_, _, fold_4_x_test = load_data_fold(4)

for iteration in range(iterations):
    accs0, target_loss_0, target_decrease_0, F1_0, F2_0, F3_0, F4_0, F5_0 = \
        predict_LSTM(fold_0_x_test, batch_size, iteration, fold_num=0)
    accs1, target_loss_1, target_decrease_1, F1_1, F2_1, F3_1, F4_1, F5_1 = \
        predict_LSTM(fold_1_x_test, batch_size, iteration, fold_num=1)
    accs2, target_loss_2, target_decrease_2, F1_2, F2_2, F3_2, F4_2, F5_2 = \
        predict_LSTM(fold_2_x_test, batch_size, iteration, fold_num=2)
    accs3, target_loss_3, target_decrease_3, F1_3, F2_3, F3_3, F4_3, F5_3 = \
        predict_LSTM(fold_3_x_test, batch_size, iteration, fold_num=3)
    accs4, target_loss_4, target_decrease_4, F1_4, F2_4, F3_4, F4_4, F5_4 = \
        predict_LSTM(fold_4_x_test, batch_size, iteration, fold_num=4)

    # valid acc is mean of 5-fold.
    test_acc.append((accs0 + accs1 + accs2 + accs3 + accs4) / 5)
    # target loss is mean of 5-fold.
    mean_target_loss.append((target_loss_0 + target_loss_1 + target_loss_2 + target_loss_3 + target_loss_4) / 5)
    # target decrease is mean of 5-fold.
    mean_target_decrease.append((target_decrease_0 + target_decrease_1 + target_decrease_2 + target_decrease_3 +
                                 target_decrease_4) / 5)
    # F1 is also mean of 5-fold.
    Phi_F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
    Gamma_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
    Theta_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
    Alpha_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
    Beta_F1.append((F5_0 + F5_1 + F5_2 + F5_3 + F5_4) / 5)

print("Total_Test_Accuracy: {:.4f} | Total_Target_Loss: {:.4f} | Total_Target_Decrease: {:.4f}\n"
      "Phi F1: {:.4f} | Gamma F1: {:.4f} | Theta F1: {:.4f} | Alpha F1: {:.4f} | Beta F1: {:.4f}".
      format(sum(test_acc) / iterations, sum(Phi_F1) / iterations,
             sum(mean_target_loss) / iterations, sum(mean_target_decrease) / iterations,
             sum(Gamma_F1) / iterations, sum(Theta_F1) / iterations,
             sum(Alpha_F1) / iterations, sum(Beta_F1) / iterations))

torch.cuda.empty_cache()
