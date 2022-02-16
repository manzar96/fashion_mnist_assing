import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(
            4,4))
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=32,
                               kernel_size=(3,3))

        self.mlp = nn.Sequential(
            nn.Linear(in_features=32*5*5, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10, bias=True)
        )

    # define forward function
    def forward(self, t):
        # conv 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # conv 2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # fc1
        t = t.reshape(-1, 32*5*5)
        t = self.mlp(t)

        return t