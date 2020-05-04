import torch.nn as nn
import torch.nn.functional as F

from ball_detection.utils import BallType


NET_INPUT_SIZE = 32, 32


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 3, padding=1)
        self.act1 = nn.PReLU()
        self.conv2 = nn.Conv2d(20, 20, 3, padding=1)
        self.act2 = nn.PReLU()
        self.conv3 = nn.Conv2d(20, 20, 3, padding=1)
        self.act3 = nn.PReLU()
        self.act = nn.PReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4 * 4 * 20, 28)
        self.act4 = nn.PReLU()
        self.fc2 = nn.Linear(28, len(BallType))

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.pool(x)
        x = self.act2(self.conv2(x))
        x = self.pool(x)
        x = self.act3(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(-1, self.fc1.in_features)
        x = self.act4(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
