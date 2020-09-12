import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, groups=1, dilation=1, padding=1, bias=False),
                      nn.BatchNorm2d(16),
                      nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(16, 16, 3, groups=1, dilation=2, padding=2, bias=False),
                      nn.BatchNorm2d(16),
                      nn.ReLU(),
                      nn.Dropout(0.05))

        # Transition 1
        self.pool1 = nn.MaxPool2d(2, 2) 

        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, 3, groups=1, dilation=1, padding=1, bias=False),
                      nn.BatchNorm2d(32),
                      nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(32, 32, 3, groups=1, dilation=1, padding=1, bias=False),
                      nn.BatchNorm2d(32),
                      nn.ReLU(),
                      nn.Dropout(0.1))

        # Transition 2
        self.pool2 = nn.MaxPool2d(2, 2) 

        self.conv5 = nn.Sequential(nn.Conv2d(32, 64, 3, groups=1, dilation=1, padding=1, bias=False),
                      nn.BatchNorm2d(64),
                      nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(64, 64, 3, groups=1, dilation=1, padding=1, bias=False),
                      nn.BatchNorm2d(64),
                      nn.ReLU(),
                      nn.Dropout(0.15))

        # Transition 3
        self.pool3 = nn.MaxPool2d(2, 2) 
        self.conv7 = nn.Sequential(nn.Conv2d(64, 64, 3, groups=64, dilation=1, padding=1, bias=False),
                                    nn.Conv2d(64, 128, 1, groups=1, dilation=1, bias=False),                                 
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(128, 128, 3, groups=128, dilation=1, padding=1, bias=False),
                                    nn.Conv2d(128, 128, 1, groups=1, dilation=1, bias=False),                                 
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        
        # GAP
        self.gap = nn.AvgPool2d(kernel_size=(4,4))
        self.conv9 = nn.Sequential(nn.Conv2d(128, 10, 1, padding=0, bias=False)) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = self.pool3(x)
        x = self.conv7(x)
        x = self.conv8(x)

        x = self.gap(x)
        x = self.conv9(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

net = Net()