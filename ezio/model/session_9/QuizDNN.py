import torch.nn as nn
import torch.nn.functional as F

def relu_apply(x, conv, bn):
    return F.relu(conv(bn(x)))

class QuizDNN(nn.Module):
    def __init__(self):
        super(QuizDNN, self).__init__()

        self.conv0 = nn.Conv2d(3, 128, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.bn6 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.pool3 = nn.AvgPool2d(8, 8)

        self.fc = nn.Conv2d(128, 10, 1, bias=True)

    def forward(self, x):
        x1 = F.relu(self.conv0(x))
        x2 = relu_apply(x1, self.conv1, self.bn1)
        x3 = relu_apply(x1 + x2, self.conv2, self.bn2)
        x4 = self.pool1(x1 + x2 + x3)

        x5 = relu_apply(x4, self.conv3, self.bn3)
        x6 = relu_apply(x4 + x5, self.conv4, self.bn4)
        x7 = relu_apply(x4 + x5 + x6, self.conv5, self.bn5)
        x8 = self.pool2(x5 + x6 + x7)

        x9 = relu_apply(x8, self.conv6, self.bn6)
        x10 = relu_apply(x8 + x9, self.conv7, self.bn7)
        x11 = relu_apply(x8 + x9 + x10, self.conv8, self.bn8)
        x12 = self.pool3(x11)

        x13 = self.fc(x12)

        return x13.view(-1, 10)

