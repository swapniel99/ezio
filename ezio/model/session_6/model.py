import torch.nn as nn


class Net_6(nn.Module):
    def __init__(self, dropout=0.0):
        self.dropout = dropout
        super(Net_6, self).__init__()
        self.cblock1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),    # Input -  28x28x1, Output -  28x28x8, RF - 3x3
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1, bias=False),    # Input -  28x28x8, Output -  28x28x8, RF - 5x5
            nn.Dropout(self.dropout),
            nn.ReLU(),
        )

        self.tblock1 = nn.Sequential(
            nn.MaxPool2d(2, 2),                           # Input -  28x28x8, Output -  14x14x8, RF - 6x6
        )

        self.cblock2 = nn.Sequential(
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),   # Input -  14x14x8, Output - 14x14x16, RF - 10x10
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),  # Input - 14x14x16, Output - 14x14x16, RF - 14x14
            nn.Dropout(self.dropout),
            nn.ReLU(),
        )

        self.tblock2 = nn.Sequential(
            nn.MaxPool2d(2, 2),                           # Input - 14x14x16, Output -   7x7x16, RF - 16x16
        )

        self.cblock3 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),  # Input -   7x7x16, Output -   7x7x32, RF - 24x24
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),  # Input -   7x7x32, Output -   7x7x32, RF - 32x32
            nn.Dropout(self.dropout),
            nn.ReLU(),
        )

        self.outblock = nn.Sequential(
            nn.AvgPool2d(7, 7),                           # Input -   7x7x32, Output -   1x1x32
            # It's a Fully Connected Layer... It's a Convolutional layer... It's Ant-Man!
            # Not using nn.Linear() because I made a promise :P
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 1, bias=False),             # Input -   1x1x32, Output -   1x1x32
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 10, 1),                         # Input -   1x1x32, Output -   1x1x10
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