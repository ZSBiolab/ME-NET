from SubNetwork_SparseCoding import dropout_mask, s_mask
from datetime import datetime
import torch
import torch.optim as optim
import copy
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import os.path
from sklearn.metrics import roc_auc_score, roc_curve,auc
import matplotlib.pyplot as plt
dtype = torch.FloatTensor


def InterpretCoxPASNet(net, x, ylabel, Pathway_layer, Hidden_layer,
                       Learning_Rate, L2, Num_Epochs, Dropout_Rate, outpath,task,path_0,
                       num_classes: int = 2):

    opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay=L2)

    for epoch in range(Num_Epochs + 1):
        print('Epochs:',epoch,'___',datetime.now())
        net.train()
        opt.zero_grad()
        net.fc1 = dropout_mask(Pathway_layer, Dropout_Rate[0])
        net.fc2 = dropout_mask(Hidden_layer, Dropout_Rate[1])

        pred = net.forward(x)

        if num_classes == 2:
            pred = torch.sigmoid(pred)

        loss = net.criterion.forward(pred, ylabel)
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
                y_tmp = copy_net(x)
                if num_classes == 2:
                    y_tmp = torch.sigmoid(y_tmp)
                loss_tmp = net.criterion.forward(y_tmp, ylabel)

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


    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists(os.path.join('results',path_0)):
        os.makedirs(os.path.join('results',path_0))
    torch.save(net.state_dict(), os.path.join('results',path_0,outpath))

    net.eval()
    eval_pred = net(x)
    if task == 'Diagnose1':
        eval_scores = torch.sigmoid(eval_pred)
    else:
        eval_scores = torch.softmax(eval_pred, dim=-1).squeeze()
    if task == 'Diagnosis2':
        auc = roc_auc_score(ylabel.detach().cpu(),
                                 eval_scores.squeeze().detach().cpu(),
                                 multi_class='ovo')
    else:
        auc = roc_auc_score(ylabel.detach().cpu(),
                                 eval_scores.squeeze().detach().cpu())
    fpr, tpr, thresholds = roc_curve(ylabel.detach().cpu(), eval_scores.detach().cpu())


    y_test_np = ylabel.detach().cpu()
    y_pred_np = eval_pred.detach().cpu()
    y_score_np = eval_scores.squeeze().detach().cpu()
    df = pd.DataFrame({'ylabel': y_test_np, 'eval_pred': y_pred_np,'eval_scores':y_score_np})
    df.to_csv(os.path.join('results',path_0,'_y_test_y_pred.csv'), index=False)
    df2 = pd.DataFrame({'fpr':fpr,'tpr':tpr,'thresholds':thresholds})
    df2.to_csv(os.path.join('results',path_0,'fpr_tpr.csv'), index=False)
    

    plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join('results',path_0,'_roc_curve.png')) # save the plot as a PNG file
