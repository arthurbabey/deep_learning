import torch
import torch.optim as optim
from torch.nn import functional as F
from torch import nn


class predictive_model(nn.Module):
    """
    Best predictive model
    """
    def __init__(self, nb_hidden=500, dp1=0.25, dp2=0.5):
        super(predictive_model, self).__init__()
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
    """
    Siamese model, initializee a predictive model a need to be inialized with a comparison model
    """

    def __init__(self, comparison):
        super(Siamese, self).__init__()
        self.model = predictive_model(500, 0.5, 0.5)
        self.comparison = comparison

    def forward1(self, x):
        mid = self.model(x)
        return mid

    def forward2(self, mid1, mid2):
        mid = mid1 - mid2
        out = self.comparison(mid)
        return out

class ConvNet_1(nn.Module):
    def __init__(self, nb_hidden):
        super(ConvNet_1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size=2)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=2, stride = 1)
        self.fc1 = nn.Linear(64, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.dropout1 = nn.Dropout2d(0)
        self.dropout2 = nn.Dropout2d(0)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (2, 2)))
        x = self.dropout1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class ConvNet_2(nn.Module):
    def __init__(self, nb_hidden):
        super(ConvNet_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size=2)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=2, stride = 1)
        self.fc1 = nn.Linear(128, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.dropout1 = nn.Dropout2d(0)
        self.dropout2 = nn.Dropout2d(0)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (2, 2)))
        x = self.dropout1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 128)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class ConvNet_3(nn.Module):
    def __init__(self, nb_hidden):
        super(ConvNet_3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size=2)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=2, stride = 1)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.dropout1 = nn.Dropout2d(0)
        self.dropout2 = nn.Dropout2d(0)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (2, 2)))
        x = self.dropout1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 256)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class ConvNet_4(nn.Module):
    def __init__(self, nb_hidden):
        super(ConvNet_4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=2, stride = 1)
        self.fc1 = nn.Linear(512, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.dropout1 = nn.Dropout2d(0)
        self.dropout2 = nn.Dropout2d(0)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (2, 2)))
        x = self.dropout1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 512)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class bigConvNet_1(nn.Module):
    def __init__(self, nb_hidden):
        super(bigConvNet_1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride = 1)
        self.fc1 = nn.Linear(4*128, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.dropout1 = nn.Dropout2d(0)
        self.dropout2 = nn.Dropout2d(0)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (2, 2)))
        x = self.dropout1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 4*128)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class bigConvNet_2(nn.Module):
    def __init__(self, nb_hidden):
        super(bigConvNet_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size=2)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=2, stride = 1)
        self.fc1 = nn.Linear(4*256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.dropout1 = nn.Dropout2d(0)
        self.dropout2 = nn.Dropout2d(0)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (2, 2)))
        x = self.dropout1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 4*256)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class bigConvNet_3(nn.Module):
    def __init__(self, nb_hidden):
        super(bigConvNet_3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 100, kernel_size=2)
        self.conv2 = nn.Conv2d(100, 200, kernel_size=2, stride = 1)
        self.fc1 = nn.Linear(4*200, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.dropout1 = nn.Dropout2d(0)
        self.dropout2 = nn.Dropout2d(0)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (2, 2)))
        x = self.dropout1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 4*200)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class bigConvNet_4(nn.Module):
    def __init__(self, nb_hidden):
        super(bigConvNet_4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 128, kernel_size=2)
        self.conv2 = nn.Conv2d(128, 1024, kernel_size=2, stride = 1)
        self.fc1 = nn.Linear(4*1024, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.dropout1 = nn.Dropout2d(0)
        self.dropout2 = nn.Dropout2d(0)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (2, 2)))
        x = self.dropout1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 4*1024)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
