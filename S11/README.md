# S11 Solution

### Team Members
1. Swapnil Gusani
2. Karan Patel
3. Nikhil Kothari
4. Abhinesh Sankar

### Problem Statement
Achieve 90% accuracy using the custom architecture. Use OneCyclePolicy. 
Use the Data Augmentation - RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8). Batch size = 512. Implement LR Range Test module.

### Results
1. Best Test Accuracy - 91.45 % (24th Epoch)


## Cyclic Graph
![Cyclic Graph](https://raw.githubusercontent.com/swapniel99/ezio/master/S11/images/cyclic_graph.png "Graph")

## One Cycle Policy (Accuracy vs Learning_rate)
![OCP](https://raw.githubusercontent.com/swapniel99/ezio/master/S11/images/ocp.png "Graph")

### Model Summary
```c
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 32, 32]          73,728
         MaxPool2d-5          [-1, 128, 16, 16]               0
       BatchNorm2d-6          [-1, 128, 16, 16]             256
              ReLU-7          [-1, 128, 16, 16]               0
            Conv2d-8          [-1, 128, 16, 16]         147,456
       BatchNorm2d-9          [-1, 128, 16, 16]             256
             ReLU-10          [-1, 128, 16, 16]               0
           Conv2d-11          [-1, 128, 16, 16]         147,456
      BatchNorm2d-12          [-1, 128, 16, 16]             256
             ReLU-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 256, 16, 16]         294,912
        MaxPool2d-15            [-1, 256, 8, 8]               0
      BatchNorm2d-16            [-1, 256, 8, 8]             512
             ReLU-17            [-1, 256, 8, 8]               0
           Conv2d-18            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-19            [-1, 512, 4, 4]               0
      BatchNorm2d-20            [-1, 512, 4, 4]           1,024
             ReLU-21            [-1, 512, 4, 4]               0
           Conv2d-22            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-23            [-1, 512, 4, 4]           1,024
             ReLU-24            [-1, 512, 4, 4]               0
           Conv2d-25            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-26            [-1, 512, 4, 4]           1,024
             ReLU-27            [-1, 512, 4, 4]               0
        MaxPool2d-28            [-1, 512, 1, 1]               0
           Conv2d-29             [-1, 10, 1, 1]           5,130
          Flatten-30                   [-1, 10]               0
================================================================
Total params: 6,573,130
Trainable params: 6,573,130
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.44
Params size (MB): 25.07
Estimated Total Size (MB): 31.53
----------------------------------------------------------------
````

### Model
[(Model)](https://github.com/swapniel99/ezio/blob/master/ezio/model/session_11/model.py) 

### LR Range Finder Code
[(LR Range Finder)](https://github.com/swapniel99/ezio/blob/master/ezio/utils/lr_range_test.py) 
