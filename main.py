import os.path
from DataLoader import load_data, load_pathway
from Train import trainCoxPASNet
import torch
from params import Params
from Model import Cox_PASNet
from Train_for_Interpret import InterpretCoxPASNet
from Train_for_testing import InterpretNet_4test
from Load_for_testing import LoadNet_4test
dtype = torch.FloatTensor
import numpy as np


class Trainer:
    def __init__(self, args: Params):
        print("Trainer Initialization...")
        
        if torch.cuda.is_available() and not args.use_cpu:
            print("Use GPU...")
            self.device = torch.device(f"cuda:{args.gpu}")
        else:
            print("Use CPU...")
            self.device = torch.device('cpu')
        self.path_0 = args.path_0
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
        self.num_epochs = args.grid_epoch
        self.opt_loss = torch.tensor([float("Inf")]).to(self.device)

        self.opt_valid_auc = 0
        self.opt_train_auc = 0
        self.Initial_Learning_Rate = [0.03, 0.01, 0.001, 0.00075]
        self.L2_Lambda = [0.1, 0.01, 0.005, 0.001]
        self.opt_L2_lambda = 0.0
        self.opt_lr = 0.0
        self.params = self.args = args
        self.Dropout_Rate = [0.7, 0.5]

    def grid_search(self):
        colours = ['blue', 'green', 'red', 'yellow', 'purple', 'orange']
        for l2 in self.L2_Lambda:
            for idx, lr in enumerate(self.Initial_Learning_Rate):

                colour = colours[idx % len(colours)]
                loss_train, loss_valid, train_auc, valid_auc = trainCoxPASNet(self.x_train,
                                                                              self.train_ylabel,
                                                                              self.x_valid,
                                                                              self.valid_ylabel,
                                                                              self.pathway_mask,
                                                                              self.params.input,
                                                                              self.args.Pathway_Nodes,
                                                                              self.params.Hidden_layer,
                                                                              self.params.output,
                                                                              lr, l2,
                                                                              self.num_epochs,
                                                                              self.Dropout_Rate,
                                                                              colour=colour,
                                                                              task=self.params.task,
                                                                              num_classes=self.num_classes,
                                                                              device=self.device)
                if valid_auc > self.opt_valid_auc:
                    self.opt_l2_loss = l2
                    self.opt_lr_loss = lr
                    self.opt_loss = loss_valid
                    self.opt_train_auc = train_auc
                    self.opt_valid_auc = valid_auc
                print("L2: ", l2, "LR: ", lr, "Loss in Validation: ", loss_valid,"valid_AUC",valid_auc)
        print(f"Optimal Lr: {self.opt_lr_loss}")
        print(f"Optimal L2: {self.opt_l2_loss}")
        print(f"Optimal valid_AUC: {self.opt_valid_auc}")

    def train(self):

        self.grid_search()
        '''
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
        print(f"Training Results:")
        print(f"Train Loss: {loss_train}")
        print(f"Test Loss: {loss_test}")
        print(f"Train AUC: {train_auc}")
        print(f"Test AUC: {test_auc}")
        '''
        print(f"Optimal Lr: {self.opt_lr_loss}")
        print(f"Optimal L2: {self.opt_l2_loss}")
        np.savetxt(os.path.join('data/',self.path_0, 'pms.csv'),[self.opt_lr_loss,self.opt_l2_loss], delimiter=",")

    def train_interpret(self):
        print(f"Interpret Training...")
        model = Cox_PASNet(self.params.input,
                           self.params.Pathway_Nodes,
                           self.params.Hidden_layer,
                           self.params.output,
                           self.pathway_mask,
                           num_classes=self.num_classes).to(self.device)


        print(f"Load full data from {self.params.full_data}...")
        x, y1ladbels, y2labels = load_data(os.path.join(self.params.data_path, self.params.full_data),
                                           dtype, self.device)
        print('load to interpret')
        if self.params.task == 'Diagnosis1':
            ylabels = y1ladbels.to(dtype=torch.float).squeeze()
            num_classes = 2
        else:
            ylabels = y2labels.to(dtype=torch.float).squeeze()
            num_classes = 3
        print('training InterpretCoxPASNet')
        InterpretCoxPASNet(model, x, ylabels, self.params.Pathway_Nodes,
                           self.params.Hidden_layer,
                           Learning_Rate=self.params.interp_lr,
                           L2=self.params.interp_l2,
                           Dropout_Rate=self.Dropout_Rate,
                           Num_Epochs=self.params.interp_epoch,
                           outpath=self.params.ckpt,
                           num_classes=num_classes,
                           task=self.params.task,
                           path_0=self.path_0)
        print('Finish training InterpretCoxPASNet')

        if os.path.exists(self.params.ckpt):
            model.load_state_dict(torch.load(os.path.join('results',self.path_0,self.params.ckpt), map_location=self.device))

            
        else:
            raise RuntimeError(f"Checkpoint {os.path.join('results',self.path_0,self.params.ckpt)} not exists !")
        print('''save weights and node values into files individually''')
        w_Layer1 = model.Layer1.weight.data.cpu().detach().numpy()
        w_Layer2 = model.Layer2.weight.data.cpu().detach().numpy()
        w_Layer3 = model.Layer3.weight.data.cpu().detach().numpy()
        w_Layer4 = model.Layer4.weight.data.cpu().detach().numpy()

        if not os.path.exists('results'):
            os.makedirs('results')
        if not os.path.exists(os.path.join('results', self.path_0)):
            os.makedirs(os.path.join('results',self.path_0))
        np.savetxt(os.path.join('results', self.path_0,'w_Layer1.csv'),
                   w_Layer1, delimiter=",")
        np.savetxt(os.path.join('results', self.path_0,'w_Layer2.csv'),
                   w_Layer2, delimiter=",")
        np.savetxt(os.path.join('results',self.path_0, 'w_Layer3.csv'),
                   w_Layer3, delimiter=",")
        np.savetxt(os.path.join('results', self.path_0,'w_Layer4.csv'),
                   w_Layer4, delimiter=",")

        pathway_node = model.tanh(model.Layer1(x))
        hidden_node = model.tanh(model.Layer2(pathway_node))
        hidden_2_node = model.tanh(model.Layer3(hidden_node))
        lin_pred = model.Layer4(hidden_2_node)

        np.savetxt(os.path.join('results', self.path_0, "pathway_node.csv"),
                   pathway_node.cpu().detach().numpy(),
                   delimiter=",")
        np.savetxt(os.path.join('results', self.path_0, "hidden_node.csv"),
                   hidden_node.cpu().detach().numpy(),
                   delimiter=",")
        np.savetxt(os.path.join('results', self.path_0,"hidden_2_node.csv"),
                   hidden_2_node.cpu().detach().numpy(),
                   delimiter=",")
        np.savetxt(os.path.join('results', self.path_0, "lin_pred.csv"),
                   lin_pred.cpu().detach().numpy(),
                   delimiter=",")

    def load_tesing(self):
        print(f"Training for tesing...")
        model = Cox_PASNet(self.params.input,
                           self.params.Pathway_Nodes,
                           self.params.Hidden_layer,
                           self.params.output,
                           self.pathway_mask,
                           num_classes=self.num_classes).to(self.device)

        print(f"Load full data from {self.params.full_data}...")

        LoadNet_4test(model,
                       self.x_train,
                       self.train_ylabel,
                       self.x_test,
                       self.test_ylabel,
                       self.params.Pathway_Nodes,
                       self.params.Hidden_layer,
                       Learning_Rate=self.params.interp_lr,
                       L2=self.params.interp_l2,
                       Dropout_Rate=self.Dropout_Rate,
                       Num_Epochs=1,
                       outpath=self.params.ckpt,
                       num_classes=self.num_classes,
                       task=self.params.task,
                       path_0=self.path_0,
                       dc = self.device)

    def train_tesing(self):
        print(f"Training for tesing...")
        model = Cox_PASNet(self.params.input,
                           self.params.Pathway_Nodes,
                           self.params.Hidden_layer,
                           self.params.output,
                           self.pathway_mask,
                           num_classes=self.num_classes).to(self.device)

        print(f"Load full data from {self.params.full_data}...")

        InterpretNet_4test(model,
                           self.x_train,
                           self.train_ylabel,
                           self.x_test,
                           self.test_ylabel,
                           self.params.Pathway_Nodes,
                           self.params.Hidden_layer,
                           Learning_Rate=self.params.interp_lr,
                           L2=self.params.interp_l2,
                           Dropout_Rate=self.Dropout_Rate,
                           Num_Epochs=self.params.interp_epoch,
                           outpath=self.params.ckpt,
                           num_classes=self.num_classes,
                           task=self.params.task,
                           path_0=self.path_0)
        print('Finish training InterpretCoxPASNet')

        if os.path.exists(self.params.ckpt):
            model.load_state_dict(torch.load(os.path.join('results4testing',self.path_0,self.params.ckpt), map_location=self.device))

            
        else:
            raise RuntimeError(f"Checkpoint {os.path.join('results4testing',self.path_0,self.params.ckpt)} not exists !")
        print('''save weights and node values into files individually''')
        w_Layer1 = model.Layer1.weight.data.cpu().detach().numpy()
        w_Layer2 = model.Layer2.weight.data.cpu().detach().numpy()
        w_Layer3 = model.Layer3.weight.data.cpu().detach().numpy()
        w_Layer4 = model.Layer4.weight.data.cpu().detach().numpy()

        if not os.path.exists('results4testing'):
            os.makedirs('results4testing')
        if not os.path.exists(os.path.join('results4testing', self.path_0)):
            os.makedirs(os.path.join('results4testing',self.path_0))
        np.savetxt(os.path.join('results4testing', self.path_0,'w_Layer1.csv'),
                   w_Layer1, delimiter=",")
        np.savetxt(os.path.join('results4testing', self.path_0,'w_Layer2.csv'),
                   w_Layer2, delimiter=",")
        np.savetxt(os.path.join('results4testing',self.path_0, 'w_Layer3.csv'),
                   w_Layer3, delimiter=",")
        np.savetxt(os.path.join('results4testing', self.path_0,'w_Layer4.csv'),
                   w_Layer4, delimiter=",")

        pathway_node = model.tanh(model.Layer1(self.x_train))
        hidden_node = model.tanh(model.Layer2(pathway_node))
        hidden_2_node = model.tanh(model.Layer3(hidden_node))
        lin_pred = model.Layer4(hidden_2_node)

        np.savetxt(os.path.join('results4testing', self.path_0, "pathway_node.csv"),
                   pathway_node.cpu().detach().numpy(),
                   delimiter=",")
        np.savetxt(os.path.join('results4testing', self.path_0, "hidden_node.csv"),
                   hidden_node.cpu().detach().numpy(),
                   delimiter=",")
        np.savetxt(os.path.join('results4testing', self.path_0,"hidden_2_node.csv"),
                   hidden_2_node.cpu().detach().numpy(),
                   delimiter=",")
        np.savetxt(os.path.join('results4testing', self.path_0, "lin_pred.csv"),
                   lin_pred.cpu().detach().numpy(),
                   delimiter=",")
                   
    def debug(self):
        colour = 'green'
        loss_train, loss_test, train_auc, test_auc = trainCoxPASNet(self.x_train,
                                                                    self.train_ylabel,
                                                                    self.x_test,
                                                                    self.test_ylabel,
                                                                    self.pathway_mask,
                                                                    self.params.input,
                                                                    self.params.Pathway_Nodes,
                                                                    self.params.Hidden_layer,
                                                                    self.params.output,
                                                                    self.opt_lr_loss,
                                                                    self.opt_l2_loss,
                                                                    self.params.epoch,
                                                                    self.Dropout_Rate,
                                                                    colour=colour,
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
    if args.do_interp:

        trainer.train_interpret()
    elif args.do_test:

        trainer.train_tesing()
    elif args.do_load:

        trainer.load_tesing()
    else:
        trainer.train()

