import torch
import torch.nn as nn


class Cox_PASNet(nn.Module):
    def __init__(self, input, Pathway_layer, Hidden_layer, output, Pathway_Mask,
                 num_classes: int = 2):
        super(Cox_PASNet, self).__init__()
        self.act = nn.Tanh()
        self.pathway_mask = Pathway_Mask

        self.Layer1 = nn.Linear(input, Pathway_layer)

        self.Layer2 = nn.Linear(Pathway_layer, Hidden_layer)

        self.Layer3 = nn.Linear(Hidden_layer, output, bias=False)

        if num_classes == 2:

            self.Layer4 = nn.Linear(output, 1, bias=False)
        else:
            self.Layer4 = nn.Linear(output, num_classes, bias=False)
        self.Layer4.weight.data.uniform_(-0.001, 0.001)

        self.fc1 = torch.ones(Pathway_layer)
        self.fc2 = torch.ones(Hidden_layer)

        if torch.cuda.is_available():
            self.fc1 = self.fc1.cuda()
            self.fc2 = self.fc2.cuda()
        if num_classes == 2:

            self.criterion = nn.BCELoss()
        else:

            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):

        self.Layer1.weight.data = self.Layer1.weight.data.mul(self.pathway_mask)
        x = self.act(self.Layer1(x))
        if self.training == True:
            x = x.mul(self.fc1)
        x = self.act(self.Layer2(x))
        if self.training == True:
            x = x.mul(self.fc2)
        x = self.act(self.Layer3(x))

        x_pre = self.Layer4(x)

        return x_pre.squeeze()
