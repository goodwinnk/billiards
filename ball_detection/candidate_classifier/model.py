import torch
import torch.nn as nn
import torch.nn.functional as F


CLASSIFICATION_SCORE_THRESHOLD = .5
NET_INPUT_SIZE = 32, 32


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        self.act = nn.PReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4 * 4 * 8, 24)
        self.fc2 = nn.Linear(24, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(-1, 4 * 4 * 8)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
