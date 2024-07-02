from Model import Cox_PASNet
from SubNetwork_SparseCoding import dropout_mask, s_mask
from datetime import datetime
import torch
import torch.optim as optim
import copy
from scipy.interpolate import interp1d
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

dtype = torch.FloatTensor


def trainCoxPASNet(train_x, train_ylabel,
                   eval_x, eval_ylabel, pathway_mask,
                   input, Pathway_layer, Hidden_layer, output,
                   Learning_Rate, L2, Num_Epochs, Dropout_Rate, colour,
                   task='Diagnosis1', num_classes: int = 1,
                   device=torch.device('cpu'),
                   eval_every: int = 50):

    print(f"Current Task is {task}")
    net = Cox_PASNet(input, Pathway_layer,
                     Hidden_layer, output,
                     pathway_mask, num_classes=num_classes)

    net = net.to(device)

    opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay=L2)

    for epoch in tqdm(range(1, Num_Epochs + 1), desc="Training Epochs", colour= colour, total=Num_Epochs):
        print('Epochs:',epoch,'______________',datetime.now())
        net.train()
        opt.zero_grad()
        net.fc1 = dropout_mask(Pathway_layer, Dropout_Rate[0])
        net.fc2 = dropout_mask(Hidden_layer, Dropout_Rate[1])

        x_pre = net(train_x)
        if num_classes == 2:
            x_pre = torch.sigmoid(x_pre).squeeze()

        loss = net.criterion.forward(x_pre, train_ylabel)
        loss.backward()
        opt.step()

        net.Layer1.weight.data = net.Layer1.weight.data.mul(
            net.pathway_mask)


        fc1_grad = copy.deepcopy(net.Layer2.weight._grad.data)
        fc2_grad = copy.deepcopy(net.Layer3.weight._grad.data)
        fc1_grad_mask = torch.where(fc1_grad == 0, fc1_grad, torch.ones_like(fc1_grad))
        fc2_grad_mask = torch.where(fc2_grad == 0, fc2_grad, torch.ones_like(fc2_grad))

        net_Layer2_weight = copy.deepcopy(net.Layer2.weight.data)
        net_Layer3_weight = copy.deepcopy(net.Layer3.weight.data)


        net_state_dict = net.state_dict()


        copy_net = copy.deepcopy(net)
        copy_state_dict = copy_net.state_dict()
        for name, param in copy_state_dict.items():

            if not "weight" in name:
                continue

            if "Layer1" in name:
                continue

            if "Layer4" in name:
                break

            if "Layer2" in name:
                active_param = net_Layer2_weight.mul(fc1_grad_mask)
            if "Layer3" in name:
                active_param = net_Layer3_weight.mul(fc2_grad_mask)
            nonzero_param_1d = active_param[active_param != 0]
            if nonzero_param_1d.size(
                    0) == 0:
                break
            copy_param_1d = copy.deepcopy(nonzero_param_1d)

            S_set = torch.arange(100, -1, -1)[1:]
            copy_param = copy.deepcopy(active_param)
            S_loss = []
            for S in S_set:
                param_mask = s_mask(sparse_level=S.item(), param_matrix=copy_param, nonzero_param_1D=copy_param_1d,
                                    dtype=dtype)
                transformed_param = copy_param.mul(param_mask)
                copy_state_dict[name].copy_(transformed_param)
                copy_net.train()
                y_tmp = copy_net(train_x)
                if num_classes == 2:
                    y_tmp = torch.sigmoid(y_tmp.squeeze())
                loss_tmp = net.criterion.forward(y_tmp, train_ylabel)
                S_loss.append(loss_tmp.detach().cpu())

            interp_S_loss = interp1d(S_set, S_loss, kind='cubic')
            interp_S_set = torch.linspace(min(S_set), max(S_set), steps=100)
            interp_loss = interp_S_loss(interp_S_set)
            optimal_S = interp_S_set[np.argmin(interp_loss)]
            optimal_param_mask = s_mask(sparse_level=optimal_S.item(), param_matrix=copy_param,
                                        nonzero_param_1D=copy_param_1d, dtype=dtype)
            if "Layer2" in name:
                final_optimal_param_mask = torch.where(fc1_grad_mask == 0, torch.ones_like(fc1_grad_mask),
                                                       optimal_param_mask)
                optimal_transformed_param = net_Layer2_weight.mul(final_optimal_param_mask)
            if "Layer3" in name:
                final_optimal_param_mask = torch.where(fc2_grad_mask == 0, torch.ones_like(fc2_grad_mask),
                                                       optimal_param_mask)
                optimal_transformed_param = net_Layer3_weight.mul(final_optimal_param_mask)

            copy_state_dict[name].copy_(optimal_transformed_param)
            net_state_dict[name].copy_(optimal_transformed_param)

        if epoch % eval_every == 0:
            net.train()
            train_pred = net(train_x)
            if num_classes == 2:
                train_pred = torch.sigmoid(train_pred.squeeze())
            train_loss = net.criterion.forward(train_pred, train_ylabel)


            net.eval()
            eval_pred = net(eval_x)
            if num_classes == 2:
                eval_pred = torch.sigmoid(eval_pred)
            eval_loss = net.criterion.forward(eval_pred, eval_ylabel)
            if task == 'Diagnosis1':
                train_scores = torch.sigmoid(train_pred)
                eval_scores = torch.sigmoid(eval_pred)
            else:
                train_scores = torch.softmax(train_pred, dim=-1).squeeze()
                eval_scores = torch.softmax(eval_pred, dim=-1).squeeze()
            if task == 'Diagnosis2':
                train_auc = roc_auc_score(train_ylabel.cpu(),
                                          train_scores.squeeze().detach().cpu(),
                                          multi_class='ovo')
                eval_auc = roc_auc_score(eval_ylabel.cpu(),
                                         eval_scores.squeeze().detach().cpu(),
                                         multi_class='ovo')
            else:
                train_auc = roc_auc_score(train_ylabel.cpu(),
                                          train_scores.squeeze().detach().cpu())
                eval_auc = roc_auc_score(eval_ylabel.cpu(),
                                         eval_scores.squeeze().detach().cpu())

            print(f"EPOCH [{epoch}]: Loss in Train: ", train_loss.item())
    return (train_loss, eval_loss, train_auc, eval_auc)
