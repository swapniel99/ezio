# S7 Solution

### Team Members
1. Swapnil Gusani
2. Karan Patel
3. Nikhil Kothari
4. Abhinesh Sankar

### Problem Statement
Achieve 80% accuracy on CIFAR-10 dataset using parameters under 1M. Must include Depthwise Seperable Convolutions and Dilated Convolutions. Global Receptive Field should be more than 44.

### Results
1. Best Accuracy - 87.58 % (19th Epoch)
2. Parameters - 925,706


### Model Package
[(EZIO Link)](https://github.com/swapniel99/ezio/tree/master/ezio) 


### Network Architecture
[(Model Architecture - Pytorch)](https://github.com/swapniel99/ezio/blob/master/ezio/model/session_7/model.py) 
```c
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [-1, 128, 30, 30]           3,456
       BatchNorm2d-2          [-1, 128, 30, 30]             256
              ReLU-3          [-1, 128, 30, 30]               0
            Conv2d-4          [-1, 128, 28, 28]         147,456
       BatchNorm2d-5          [-1, 128, 28, 28]             256
              ReLU-6          [-1, 128, 28, 28]               0
         MaxPool2d-7          [-1, 128, 14, 14]               0
            Conv2d-8          [-1, 128, 14, 14]           1,152
            Conv2d-9          [-1, 256, 14, 14]          32,768
      BatchNorm2d-10          [-1, 256, 14, 14]             512
             ReLU-11          [-1, 256, 14, 14]               0
           Conv2d-12          [-1, 256, 14, 14]           2,304
           Conv2d-13          [-1, 256, 14, 14]          65,536
      BatchNorm2d-14          [-1, 256, 14, 14]             512
             ReLU-15          [-1, 256, 14, 14]               0
        MaxPool2d-16            [-1, 256, 7, 7]               0
           Conv2d-17            [-1, 256, 7, 7]           2,304
           Conv2d-18            [-1, 512, 7, 7]         131,072
      BatchNorm2d-19            [-1, 512, 7, 7]           1,024
             ReLU-20            [-1, 512, 7, 7]               0
           Conv2d-21            [-1, 512, 7, 7]           4,608
           Conv2d-22            [-1, 512, 7, 7]         262,144
      BatchNorm2d-23            [-1, 512, 7, 7]           1,024
             ReLU-24            [-1, 512, 7, 7]               0
        AvgPool2d-25            [-1, 512, 1, 1]               0
      BatchNorm2d-26            [-1, 512, 1, 1]           1,024
           Conv2d-27            [-1, 512, 1, 1]         262,144
             ReLU-28            [-1, 512, 1, 1]               0
      BatchNorm2d-29            [-1, 512, 1, 1]           1,024
           Conv2d-30             [-1, 10, 1, 1]           5,130
          Flatten-31                   [-1, 10]               0
================================================================
Total params: 925,706
Trainable params: 925,706
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 9.55
Params size (MB): 3.53
Estimated Total Size (MB): 13.09
----------------------------------------------------------------
```
### Receptive Field
Global Receptive Field : 64

|         | kernel | dilation | padding | stride | Jout | Nout | RF |
|---------|--------|----------|---------|--------|------|------|----|
| input   |        |          |         |        | 1    | 32   | 1  |
| conv    | 3      | 1        | 0       | 1      | 1    | 30   | 3  |
| conv    | 3      | 1        | 0       | 1      | 1    | 28   | 5  |
| maxpool | 2      | 1        | 0       | 2      | 2    | 14   | 6  |
| conv    | 3      | 2        | 2       | 1      | 2    | 14   | 14 |
| conv    | 1      | 1        | 0       | 1      | 2    | 14   | 14 |
| conv    | 3      | 2        | 2       | 1      | 2    | 14   | 22 |
| conv    | 1      | 1        | 0       | 1      | 2    | 14   | 22 |
| maxpool | 2      | 1        | 0       | 2      | 4    | 7    | 24 |
| conv    | 3      | 1        | 1       | 1      | 4    | 7    | 32 |
| conv    | 1      | 1        | 0       | 1      | 4    | 7    | 32 |
| conv    | 3      | 1        | 1       | 1      | 4    | 7    | 40 |
| conv    | 1      | 1        | 0       | 1      | 4    | 7    | 40 |
| avgpool | 7      | 1        | 0       | 7      | 28   | 1    | 64 |
| conv    | 1      | 1        | 0       | 1      | 28   | 1    | 64 |
| conv    | 1      | 1        | 0       | 1      | 28   | 1    | 64 |
