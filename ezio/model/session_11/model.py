import torch.nn as nn


class Net11(nn.Module):
    def __init__(self):
        super(Net11, self).__init__()

        self.preplayer = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.x1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.r1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.x2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.r2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool2d(4)

        self.fc = nn.Sequential(
            nn.Conv2d(512, 10, 1),
            nn.Flatten()
        )

    def forward(self, x):
        preplayer = self.preplayer(x)

        x1 = self.x1(preplayer)
        r1 = self.r1(x1)
        layer1 = x1 + r1

        layer2 = self.layer2(layer1)
        x2 = self.x2(layer2)
        r2 = self.r2(x2)

        layer3 = x2 + r2

        maxpool = self.maxpool(layer3)
        fc = self.fc(maxpool)

        return fc