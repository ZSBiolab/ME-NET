import os.path

from DataLoader import load_data, load_pathway
from Train import trainCoxPASNet

import torch
import numpy as np
# pip install typed-argument-parser
from params import Params


dtype = torch.FloatTensor


class Trainer:
    def __init__(self, args: Params):
        print("Trainer Initialization...")
        if torch.cuda.is_available() and not args.use_cpu:
            print("Use GPU...")
            self.device = torch.device(f"cuda:{args.gpu}")
        else:
            print("Use CPU...")
            self.device = torch.device('cpu')
        self.pathway_mask = load_pathway(os.path.join(args.data_path,
                                                      args.pmask),
                                         dtype, self.device)
        print(f"{self.pathway_mask.device=}")
        self.x_train, yd1_train, yd2_train = load_data(os.path.join(args.data_path,
                                                                    args.train),
                                                       dtype, self.device)
        self.x_valid, yd1_valid, yd2_valid = load_data(os.path.join(args.data_path,
                                                                    args.valid),
                                                       dtype,
                                                       self.device)
        self.x_test, yd1_test, yd2_test = load_data(os.path.join(args.data_path,
                                                                 args.test),
                                                    dtype, self.device)
        if args.task == 'Diagnosis1':
            self.train_ylabel = yd1_train.to(dtype=torch.float).squeeze()
            self.valid_ylabel = yd1_valid.to(dtype=torch.float).squeeze()
            self.test_ylabel = yd1_test.to(dtype=torch.float).squeeze()
            self.num_classes = 2
        else:
            self.train_ylabel = yd2_train.to(dtype=torch.long).squeeze()
            self.valid_ylabel = yd2_valid.to(dtype=torch.long).squeeze()
            self.test_ylabel = yd2_test.to(dtype=torch.long).squeeze()
            self.num_classes = 3

        self.opt_l2_loss = 0
        self.opt_lr_loss = 0
        self.num_epochs = args.grid_epoch  ### for grid search
        self.opt_loss = torch.tensor([float("Inf")]).to(self.device)
        ###if gpu is being used

        ###
        self.opt_valid_auc = 0
        self.opt_train_auc = 0
        self.Initial_Learning_Rate = [0.03, 0.01, 0.001, 0.00075]
        self.L2_Lambda = [0.1, 0.01, 0.005, 0.001]
        self.opt_L2_lambda = 0.0
        self.opt_lr = 0.0
        self.params = self.args = args
        self.Dropout_Rate = [0.7, 0.5]

    def grid_search(self):
        for l2 in self.L2_Lambda:
            for lr in self.Initial_Learning_Rate:
                loss_train, loss_valid, train_auc, valid_auc = trainCoxPASNet(self.x_train,
                                                                              self.train_ylabel,
                                                                              self.x_valid,
                                                                              self.valid_ylabel,
                                                                              self.pathway_mask,
                                                                              self.params.In_Nodes,
                                                                              self.args.Pathway_Nodes,
                                                                              self.params.Hidden_Nodes,
                                                                              self.params.Out_Nodes,
                                                                              lr, l2,
                                                                              self.num_epochs,
                                                                              self.Dropout_Rate,
                                                                              num_classes=self.num_classes,
                                                                              device=self.device)
                if loss_valid < self.opt_loss:
                    self.opt_l2_loss = l2
                    self.opt_lr_loss = lr
                    self.opt_loss = loss_valid
                    self.opt_train_auc = train_auc
                    self.opt_valid_auc = valid_auc
                print("L2: ", l2, "LR: ", lr, "Loss in Validation: ", loss_valid)
        print(f"Optimal Lr: {self.opt_lr_loss}")
        print(f"Optimal L2: {self.opt_l2_loss}")

    def train(self):
        self.grid_search()
        loss_train, loss_test, train_auc, test_auc = trainCoxPASNet(self.x_train,
                                                                    self.train_ylabel,
                                                                    self.x_test,
                                                                    self.test_ylabel,
                                                                    self.pathway_mask,
                                                                    self.params.In_Nodes,
                                                                    self.params.Pathway_Nodes,
                                                                    self.params.Hidden_Nodes,
                                                                    self.params.Out_Nodes,
                                                                    self.opt_lr_loss,
                                                                    self.opt_l2_loss,
                                                                    self.params.epoch,
                                                                    self.Dropout_Rate,
                                                                    task=self.params.task,
                                                                    num_classes=self.num_classes,
                                                                    device=self.device)
        print(f"Train Loss: {loss_train}")
        print(f"Test Loss: {loss_test}")
        print(f"Train AUC: {train_auc}")
        print(f"Test AUC: {test_auc}")

    def debug(self):
        loss_train, loss_test, train_auc, test_auc = trainCoxPASNet(self.x_train,
                                                                    self.train_ylabel,
                                                                    self.x_test,
                                                                    self.test_ylabel,
                                                                    self.pathway_mask,
                                                                    self.params.In_Nodes,
                                                                    self.params.Pathway_Nodes,
                                                                    self.params.Hidden_Nodes,
                                                                    self.params.Out_Nodes,
                                                                    self.opt_lr_loss,
                                                                    self.opt_l2_loss,
                                                                    2,
                                                                    self.Dropout_Rate,
                                                                    task=self.params.task,
                                                                    num_classes=self.num_classes,
                                                                    device=self.device,
                                                                    eval_every=1)
        print(f"Train Loss: {loss_train}")
        print(f"Test Loss: {loss_test}")
        print(f"Train AUC: {train_auc}")
        print(f"Test AUC: {test_auc}")


if __name__ == '__main__':
    args = Params().parse_args()
    trainer = Trainer(args=args)
    trainer.debug()
