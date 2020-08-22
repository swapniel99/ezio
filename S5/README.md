# S5 Solution

### Problem Statement
Achieve >= **99.4%** accuracy consistently on the **MNIST** dataset in <= **15  epochs**.

### Results
Accuracy : **99.49%**

Params : **7878**
### Network Diagram
![The Network](https://i.imgur.com/owE4034.png  "Model")

[(Tool used to create above diagram)](https://alexlenail.me/NN-SVG/LeNet.html) 
### Steps used in training the model:
Step 1: (EVA5S5_1.ipynb) 
1. A basic structure of the model was initialized as the starting point.
2. In order to bring down the number of parameters, we used Global Average Pooling(GAP) **(nn.AvgPool2d())** before applying FC layer. GAP is a better alternative to flatten() as it helps preserve spatial information.

Step 2: (EVA5S5_2.ipynb)
1. Batch Normalization is a technique that standardizes the inputs to a layer for each mini-batch. 
2. As the previous interation of the model was not able to converge well on training dataset, we used Batch Normalization **(nn.BatchNorm2d())** to improve the convergence and efficiency of the model.

Step 3: (EVA5S5_3.ipynb)
1. The previous iteration of the model was overfitting.
2. Data Regularization techniques such as DropOut and Data Augmentation are used to reduce the gap between training and test accuracy.
3. We used Data Augmentation techinuqes such as **RandomPerspective** and **RandomRotation**.

Step 4: (EVA5S5_4.ipynb)
1. The loss graph of the previous iteration of the model was depicting a zig-zag pattern. This indicates that the learning rate was too high towards the later epochs of the training process.
2. We used **optim.lr_scheduler.ReduceLROnPlateau()** function in pytorch to reduce the learning rate incase of increase in the loss function. 

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

### Code Structure
```python
	
	# Convolution Block 1
	self.cblock1 = nn.Sequential(
		      conv -> relu ->
		bn -> conv -> relu
        )
        
        # Transition Block 1
        self.tblock1 = nn.Sequential(
        	maxpool
        )
        
        # Convolution Block 2
        self.cblock2 = nn.Sequential(
	    	bn -> conv -> relu ->
	    	bn -> conv -> relu
        )
        
        # Transition Block 2
        self.tblock2 = nn.Sequential(
	        maxpool
        )
        
        # Convolution Block 3
        self.cblock3 = nn.Sequential(
	    	bn -> conv -> relu
        )
        
        # Output Block
        self.outblock = nn.Sequential(
	        avgpool ->
	        # Dense layers start
	        bn -> conv -> relu ->
	        bn -> conv ->
	        flatten -> logsoftmax
        )
```

### Receptive Field Calculation
|         | kernel | padding | stride | Jout | Nout | RF |
|---------|--------|---------|--------|------|------|----|
| input   |        |         |        | 1    | 28   | 1  |
| conv    | 3      | 1       | 1      | 1    | 28   | 3  |
| conv    | 3      | 1       | 1      | 1    | 28   | 5  |
| maxpool | 2      | 0       | 2      | 2    | 14   | 6  |
| conv    | 3      | 1       | 1      | 2    | 14   | 10 |
| conv    | 3      | 1       | 1      | 2    | 14   | 14 |
| maxpool | 2      | 0       | 2      | 4    | 7    | 16 |
| conv    | 3      | 1       | 1      | 4    | 7    | 24 |
| avgpool | 7      | 0       | 7      | 28   | 1    | 48 |
| conv    | 1      | 0       | 1      | 28   | 1    | 48 |
| conv    | 1      | 0       | 1      | 28   | 1    | 48 |

### Description

The network consists of 3 convolution blocks. Each convolution block has 2 convolution layers. Few notable features :

1. ReLU used as activation layer.
2. Maxpool used after every 2 convolutions.
3. Batch normalisation applied right before every convolution.
4. Number of channels steadily increased from 8 to 16.
5. Avg pool applied after reaching image size 7x7 and dropping it to 1x1.
6. 2 FC layers applied after AvgPool using nn.conv2d of kernel size 1 and stride 1.
