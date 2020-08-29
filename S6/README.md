# S6 Solution
​
### Team Members
1. Swapnil Gusani
2. Karan Patel
3. Nikhil Kothari
4. Abhinesh Sankar
​
### Problem Statement
Compare regularization techniques 
1. L1
2. L2
3. L1-L2 combined
4. GBN
5. L1-L2-GBN
​
### Results
1. L1 - 99.30 %
2. L2 - 99.51 %
3. L1-L2 combined - 99.30 %
4. GBN - 99.25 %
5. L1-L2-GBN - 99.24 %
<br/><br/>

### Training Loss and Validation Accuracy
![Loss/Accuracy](https://raw.githubusercontent.com/swapniel99/ezio/master/S6/images/loss_accuracy.png  "Loss/Accuracy")

<br/><br/>
### Misclassified Examples

<div style="text-align:center"><img src="https://raw.githubusercontent.com/swapniel99/ezio/master/S6/images/ex.png"/></div>

<br/><br/>
### Network Diagram
![The Network](https://i.imgur.com/owE4034.png  "Model")

[(Tool used to create above diagram)](https://alexlenail.me/NN-SVG/LeNet.html) 
### Network Architecture
```c
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
            Conv2d-4           [-1, 10, 28, 28]             720
              ReLU-5           [-1, 10, 28, 28]               0
         MaxPool2d-6           [-1, 10, 14, 14]               0
       BatchNorm2d-7           [-1, 10, 14, 14]              20
            Conv2d-8           [-1, 16, 14, 14]           1,440
              ReLU-9           [-1, 16, 14, 14]               0
      BatchNorm2d-10           [-1, 16, 14, 14]              32
           Conv2d-11           [-1, 16, 14, 14]           2,304
             ReLU-12           [-1, 16, 14, 14]               0
        MaxPool2d-13             [-1, 16, 7, 7]               0
      BatchNorm2d-14             [-1, 16, 7, 7]              32
           Conv2d-15             [-1, 16, 7, 7]           2,304
             ReLU-16             [-1, 16, 7, 7]               0
        AvgPool2d-17             [-1, 16, 1, 1]               0
      BatchNorm2d-18             [-1, 16, 1, 1]              32
           Conv2d-19             [-1, 32, 1, 1]             512
             ReLU-20             [-1, 32, 1, 1]               0
      BatchNorm2d-21             [-1, 32, 1, 1]              64
           Conv2d-22             [-1, 10, 1, 1]             330
          Flatten-23                   [-1, 10]               0
       LogSoftmax-24                   [-1, 10]               0
================================================================
Total params: 7,878
Trainable params: 7,878
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.44
Params size (MB): 0.03
Estimated Total Size (MB): 0.47
----------------------------------------------------------------
```
<br/><br/>
### Analysis
1. L1 regularization has a sparse solution(lot of weights -> 0), hence it is hard on the training.
2. L1 generates a model that is simple and interpretable but cannot learn complex patterns.
3. L2 regularization penalizes the sum of square weights. It performed much better than L1 as it doesn't have a sparse solution, it is able to learn complex patterns.
4. L1 and L2 combined gave similar results to L1 alone.
5. We used 4 splits with batch size 64 in GBN. It performed slightly better than BN.
6. L1, L2 and GBN had results comparable to GBN alone but slightly worse than L1 + L2.
​
​
