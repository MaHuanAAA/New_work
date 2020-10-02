import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 32)
        self.linear2 = torch.nn.Linear(32, 10)
        self.sf = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        y_pred = self.linear2(x)
        return y_pred

class Model2(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear3 = torch.nn.Linear(216, 54)
        self.linear4 = torch.nn.Linear(54, 10)
        self.sf = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.linear3(x))
        y_pred = self.linear4(x)
        return y_pred


