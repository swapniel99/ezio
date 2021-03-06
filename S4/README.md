# S4 Solution
### Results
Accuracy : **99.61%**

Params : **19570**
### Network Diagram
![The Network](https://i.imgur.com/y3bj1OC.png  "Model")

[(Tool used to create above diagram)](https://alexlenail.me/NN-SVG/LeNet.html) 

### Network Architecture
```c
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
           Dropout-2            [-1, 8, 28, 28]               0
              ReLU-3            [-1, 8, 28, 28]               0
       BatchNorm2d-4            [-1, 8, 28, 28]              16
            Conv2d-5            [-1, 8, 28, 28]             576
           Dropout-6            [-1, 8, 28, 28]               0
              ReLU-7            [-1, 8, 28, 28]               0
         MaxPool2d-8            [-1, 8, 14, 14]               0
       BatchNorm2d-9            [-1, 8, 14, 14]              16
           Conv2d-10           [-1, 16, 14, 14]           1,152
          Dropout-11           [-1, 16, 14, 14]               0
             ReLU-12           [-1, 16, 14, 14]               0
      BatchNorm2d-13           [-1, 16, 14, 14]              32
           Conv2d-14           [-1, 16, 14, 14]           2,304
          Dropout-15           [-1, 16, 14, 14]               0
             ReLU-16           [-1, 16, 14, 14]               0
        MaxPool2d-17             [-1, 16, 7, 7]               0
      BatchNorm2d-18             [-1, 16, 7, 7]              32
           Conv2d-19             [-1, 32, 7, 7]           4,608
          Dropout-20             [-1, 32, 7, 7]               0
             ReLU-21             [-1, 32, 7, 7]               0
      BatchNorm2d-22             [-1, 32, 7, 7]              64
           Conv2d-23             [-1, 32, 7, 7]           9,216
          Dropout-24             [-1, 32, 7, 7]               0
             ReLU-25             [-1, 32, 7, 7]               0
        AvgPool2d-26             [-1, 32, 1, 1]               0
      BatchNorm2d-27             [-1, 32, 1, 1]              64
           Conv2d-28             [-1, 32, 1, 1]           1,024
          Dropout-29             [-1, 32, 1, 1]               0
             ReLU-30             [-1, 32, 1, 1]               0
      BatchNorm2d-31             [-1, 32, 1, 1]              64
           Conv2d-32             [-1, 10, 1, 1]             330
          Flatten-33                   [-1, 10]               0
       LogSoftmax-34                   [-1, 10]               0
================================================================
Total params: 19,570
Trainable params: 19,570
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.62
Params size (MB): 0.07
Estimated Total Size (MB): 0.70
----------------------------------------------------------------
```

### Code Structure
```python
	
	# Convolution Block 1
	self.cblock1 = nn.Sequential(
		      conv -> drop -> relu ->
		bn -> conv -> drop -> relu
        )
        
        # Transition Block 1
        self.tblock1 = nn.Sequential(
        	maxpool
        )
        
        # Convolution Block 2
        self.cblock2 = nn.Sequential(
	    	bn -> conv -> drop -> relu ->
	    	bn -> conv -> drop -> relu
        )
        
        # Transition Block 2
        self.tblock2 = nn.Sequential(
	        maxpool
        )
        
        # Convolution Block 3
        self.cblock3 = nn.Sequential(
	    	bn -> conv -> drop -> relu ->
	    	bn -> conv -> drop -> relu
        )
        
        # Output Block
        self.outblock = nn.Sequential(
	        avgpool ->
	        # Dense layers start
	        bn -> conv -> drop -> relu ->
	        bn -> conv ->
	        flatten -> logsoftmax
        )
```

### Description

The network consists of 3 convolution blocks. Each convolution block has 2 convolution layers. Few notable features :

1. Dropout 5% applied after every conv2d layer.
2. ReLU used as activation layer after dropout.
3. Maxpool used after every 2 convolutions.
4. Batch normalisation applied right before every convolution.
5. Number of channels steadily increased from 8 to 32.
6. Avg pool applied after reaching image size 7x7 and dropping it to 1x1.
7. 2 FC layers applied after AvgPool using nn.conv2d of kernel size 1 and stride 1.
8. Dropout between 2 FC layers set a bit higher to 30%.
