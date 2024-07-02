from SubNetwork_SparseCoding import dropout_mask, s_mask
from datetime import datetime
import torch
import torch.optim as optim
import copy
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import os.path
from scipy.stats import norm
from sklearn.metrics import  accuracy_score,roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, mean_squared_error, zero_one_loss,hamming_loss
import matplotlib.pyplot as plt
dtype = torch.FloatTensor

def AUC_CI(auc,tpr, label, alpha = 0.05):
    lowerb = []
    upperb = []
    label = np.array(label)#防止label不是array类型
    n1, n2 = np.sum(label == 1), np.sum(label == 0)
    q1 = auc / (2-auc)
    q2 = (2 * auc ** 2) / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 -1) * (q2 - auc ** 2)) / (n1 * n2))
    confidence_level = 1 - alpha
    z_lower, z_upper = norm.interval(confidence_level)
    for x in tpr:
        lb = x + z_lower * se
        ub = x + z_upper * se
        if lb < 0:
            lb = 0
        if ub > 1:
            ub = 1
        lowerb.append(lb)
        upperb.append(ub)
    return (lowerb, upperb,z_lower, z_upper)

def LoadNet_4test(net, x, ylabel,x_test, test_ylabel, Pathway_layer, Hidden_layer,
                  Learning_Rate, L2, Num_Epochs, Dropout_Rate, outpath,task,path_0,dc,
                  num_classes: int = 2):
    train_losses = []
    test_losses = []

    opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay=L2)

    for epoch in range(Num_Epochs + 1):
        print('Epochs:',epoch,'___',datetime.now())
        net.train()
        opt.zero_grad()

        net.fc1 = dropout_mask(Pathway_layer, Dropout_Rate[0])
        net.fc2 = dropout_mask(Hidden_layer, Dropout_Rate[1])

        pred = net.forward(x)
        test_pred = net.forward(x_test)  

        if num_classes == 2:
            pred = torch.sigmoid(pred)
            test_pred = torch.sigmoid(test_pred)
        test_loss = net.criterion.forward(test_pred, test_ylabel)
        loss = net.criterion.forward(pred, ylabel)
        loss.backward()
        opt.step()
        train_losses.append(loss.squeeze().detach().cpu())
        test_losses.append(test_loss.squeeze().detach().cpu())
        
        
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
        

    if not os.path.exists('results_load_test'):
        os.makedirs('results_load_test')
    if not os.path.exists(os.path.join('results_load_test',path_0)):
        os.makedirs(os.path.join('results_load_test',path_0))
    if os.path.exists(outpath):
        net.load_state_dict(torch.load(os.path.join('results4testing',path_0,outpath), map_location=dc))

        
    else:
        raise RuntimeError(f"Checkpoint {os.path.join('results4testing',path_0,outpath)} not exists !")
    print('''save weights and node values into files individually''')


    net.eval()
    eval_pred = net(x_test)
    if task == 'Diagnose1':
        eval_scores = torch.sigmoid(eval_pred)
    else:
        eval_scores = torch.softmax(eval_pred, dim=-1)
    if task == 'Diagnosis2':
        auc = roc_auc_score(test_ylabel.detach().cpu(),
                                 eval_scores.squeeze().detach().cpu(),
                                 multi_class='ovo')
    else:
        auc = roc_auc_score(test_ylabel.detach().cpu(),
                                 eval_scores.squeeze().detach().cpu())
    fpr, tpr, thresholds = roc_curve(test_ylabel.detach().cpu(), eval_scores.detach().cpu())
    print(auc)

    test_ylabel_np = test_ylabel.detach().cpu()
    y_pred_np = eval_pred.squeeze().detach().cpu()
    y_pred_rd = (y_pred_np>=0).int()
    y_score_np = eval_scores.squeeze().detach().cpu()
    
    drecall = recall_score(test_ylabel_np, y_pred_rd,average= "micro")
    dprecision=precision_score(test_ylabel_np, y_pred_rd,average= "micro")
    df1= f1_score(test_ylabel_np, y_pred_rd,average= "micro")
    MSE = mean_squared_error(test_ylabel_np, y_pred_rd, squared=False)
    daccuracy=accuracy_score(test_ylabel_np,y_pred_rd)
    zol = zero_one_loss(test_ylabel_np,y_pred_rd)
    dhamming_loss=hamming_loss(test_ylabel_np, y_pred_rd)
    
    df = pd.DataFrame({'test_ylabel': test_ylabel_np, 'eval_pred': y_pred_np,'eval_scores':y_score_np,'y_pred_rd':y_pred_rd})
    df.to_csv(os.path.join('results_load_test',path_0,'_y_test_y_pred.csv'), index=False)
    df2 = pd.DataFrame({'fpr':fpr,'tpr':tpr,'thresholds':thresholds})
    df2.to_csv(os.path.join('results_load_test',path_0,'fpr_tpr.csv'), index=False)
    df3 = pd.DataFrame({'recall':drecall,'precision':dprecision,'f1':df1,'MSE':MSE,'accuracy':daccuracy,'zol':zol,'dhamming_loss':dhamming_loss},index= [1])
    df3.to_csv(os.path.join('results_load_test',path_0,'other_parameter.csv'), index=False)

    df4 = pd.DataFrame({'Num_Epochs':list(range(Num_Epochs+1)),'train_losses':list(train_losses),'test_losses':list(test_losses)})
    df4.to_csv(os.path.join('results_load_test',path_0,'loss-epoch.csv'), index=False)

    CI_test = AUC_CI(auc,tpr, test_ylabel_np, 0.05)

    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color= 'coral',label='AUC = {:.2f}'.format(auc))
    ax.fill_between(fpr, CI_test[0], CI_test[1], color='coral', alpha=0.3, label='95% CI')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_facecolor('none')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)#坐标宽度
    ax.spines['left'].set_linewidth(1.5)
    ax.tick_params(axis='both', length=6, width=1, direction='out',color = 'black')
    plt.xticks(np.arange(0, 1.1, 0.2),color = 'black')
    plt.yticks(np.arange(0, 1.1, 0.2),color = 'black')
    ax.legend()
    plt.savefig(os.path.join('results_load_test',path_0,'_roc_curve.pdf')) 
    df0 = pd.DataFrame({'ylabel_real': test_ylabel_np,  'eval_pred': y_pred_np,'pre_scores':y_score_np,"y_pred_rd":y_pred_rd})
    df0.to_csv(os.path.join('results_load_test',path_0,'_y_test_y_pred.csv'), index=False)
    df2 = pd.DataFrame({'fpr':fpr,'tpr':tpr,'thresholds':thresholds})
    df2.to_csv(os.path.join('results_load_test',path_0,'fpr_tpr.csv'), index=False)
    df3 = pd.DataFrame({'recall':drecall,'precision':dprecision,'f1':df1,'MSE':MSE,'accuracy':daccuracy,'zol':zol,'dhamming_loss':dhamming_loss},index= [1])
    df3.to_csv(os.path.join('results_load_test',path_0,'other_parameter.csv'), index=False)
    plt.close()