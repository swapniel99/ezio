import torch.nn as nn
from ezio.utils.ghost_batch_norm import GhostBatchNorm


class NetGBN_6(nn.Module):
    def __init__(self):
        super(NetGBN_6, self).__init__()
        self.cblock1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),    # Input -  28x28x1, Output -  28x28x8, RF - 3x3
            nn.ReLU(),
            GhostBatchNorm(num_features=8, num_splits=4),
            nn.Conv2d(8, 10, 3, padding=1, bias=False),   # Input -  28x28x8, Output - 28x28x10, RF - 5x5
            nn.ReLU(),
        )

        self.tblock1 = nn.Sequential(
            nn.MaxPool2d(2, 2),                           # Input - 28x28x10, Output - 14x14x10, RF - 6x6
        )

        self.cblock2 = nn.Sequential(
            GhostBatchNorm(num_features=10, num_splits=4),
            nn.Conv2d(10, 16, 3, padding=1, bias=False),  # Input - 14x14x10, Output - 14x14x16, RF - 10x10
            nn.ReLU(),
            GhostBatchNorm(num_features=16, num_splits=4),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),  # Input - 14x14x16, Output - 14x14x16, RF - 14x14
            nn.ReLU(),
        )

        self.tblock2 = nn.Sequential(
            nn.MaxPool2d(2, 2),                           # Input - 14x14x16, Output -   7x7x16, RF - 16x16
        )

        self.cblock3 = nn.Sequential(
            GhostBatchNorm(num_features=16, num_splits=4),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),  # Input -   7x7x16, Output -   7x7x16, RF - 24x24
            nn.ReLU(),
        )

        self.outblock = nn.Sequential(
            nn.AvgPool2d(7, 7),                           # Input -   7x7x16, Output -   1x1x16, RF - 48x48
            GhostBatchNorm(num_features=16, num_splits=4),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.ReLU(),
            GhostBatchNorm(num_features=32, num_splits=4),
            nn.Conv2d(32, 10, 1),
            nn.Flatten(),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.cblock1(x)
        x = self.tblock1(x)
        x = self.cblock2(x)
        x = self.tblock2(x)
        x = self.cblock3(x)
        x = self.outblock(x)
        return x
