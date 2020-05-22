from torch import nn


class BallLocationFCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ops = nn.Sequential(
            nn.Conv2d(3, 5, 7, padding=3),
            nn.PReLU(),
            nn.Conv2d(5, 6, 7, padding=3),
            nn.PReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(6, 5, 7, padding=3),
            nn.PReLU(),
            nn.Conv2d(5, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.ops(x)
