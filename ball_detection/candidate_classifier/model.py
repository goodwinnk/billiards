import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 12, 3, padding=1)
        self.conv3 = nn.Conv2d(12, 4, 3, padding=1)
        self.act = nn.PReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4 * 4 * 4, 24)
        self.fc2 = nn.Linear(24, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(-1, 4 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
