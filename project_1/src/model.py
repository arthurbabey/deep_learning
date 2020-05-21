import torch
import torch.optim as optim
from torch.nn import functional as F
from torch import nn


class bigConvNet_3(nn.Module):
    def __init__(self, nb_hidden=500, dp1=0.25, dp2=0.5):
        super(bigConvNet_3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 100, kernel_size=2)
        self.conv1_bn = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 200, kernel_size=2, stride = 1)
        self.conv2_bn = nn.BatchNorm2d(200)
        self.fc1 = nn.Linear(4*200, nb_hidden)
        self.fc1_bn = nn.BatchNorm1d(nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.dropout1 = nn.Dropout2d(dp1)
        self.dropout2 = nn.Dropout2d(dp2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), (2, 2)))
        x = self.dropout1(x)
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        x = x.view(-1, 4*200)
        x = self.dropout1(x)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class Siamese(nn.Module):

    def __init__(self, comparison):
        super(Siamese, self).__init__()
        self.model = bigConvNet_3(500, 0.5, 0.5)
        self.comparison = comparison

    def forward1(self, x):
        mid = self.model(x)
        return mid

    def forward2(self, mid1, mid2):
        mid = mid1 - mid2
        out = self.comparison(mid)
        return out
