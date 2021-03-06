import torch.nn as nn


class Net_7(nn.Module):
    def __init__(self):
        super(Net_7, self).__init__()
        self.cblock1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.tblock1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
        )

        self.cblock2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, dilation=2, padding=2, bias=False, groups=128),
            nn.Conv2d(128, 256, 1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, dilation=2, padding=2, bias=False, groups=256),
            nn.Conv2d(256, 256, 1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.tblock2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
        )

        self.cblock3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False, groups=256),
            nn.Conv2d(256, 512, 1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1, bias=False, groups=512),
            nn.Conv2d(512, 512, 1, padding=0, bias=False),
            nn.BatchNorm2d(512),

            nn.ReLU()
        )

        self.outblock = nn.Sequential(
            nn.AvgPool2d(7, 7),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 10, 1),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.cblock1(x)
        x = self.tblock1(x)
        x = self.cblock2(x)
        x = self.tblock2(x)
        x = self.cblock3(x)
        x = self.outblock(x)
        return x
