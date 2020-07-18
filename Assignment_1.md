## Q1 Channels and Kernels (according to EVA)
**Channel**: A channel is a collection or stream of information which has some quality or “feature” in common.
This feature defines the context in which some information becomes part of the channel.
For example: 
1. Left and Right audio in a song file are separate channels and being recorded on ‘left’ or ‘right’ mic is a feature.
2. In an airport, passengers waiting at their respective gates based on gate number printed on boarding pass form a channel at each gate. Here feature being a gate number.

**Kernel**: A kernel(also known as a feature extractor) is a function which transforms/extracts information such that it has a specific feature(in common).
The output of a kernel makes a channel.
For example:
1. A function which outputs first names only from a list of full names is a kernel.
2. A function which outputs names of movies of Amir Khan from the universe of all movies is a kernel.



## Q2 Why should we (nearly) always use 3x3 kernels?

Let us divide the answer into 2 parts:

**1. Why is an odd filter preferred over an even filter?**
- A pixel is considered a fundamental entity. A source pixel is the pixel around which a kernel is centered and we consolidate information from all the neighbouring pixels around it to calculate the output pixel. In even filters the source pixel is imaginary with fractional coordinates whereas in odd filters the source pixel is real and has integer coordinates.
- If we think of convolution as interpolation from the input pixels to the central pixel, we cannot do so with an even-sized filter.
- Suppose a part of the image contains all zeros except for one entry at a particular index. An odd filter will be able to capture this peak location while an even filter would not. We would want to preserve the location of peaks whenever we apply a filter.
- An even sized filter will lead to aliasing errors and distortions across the layers.
![](https://i.stack.imgur.com/YDusp.png)

**2.Why is the 3x3 kernel preferred over larger kernels like 5x5 ,7x7?**
- The features extracted by 3x3 will be highly local and hence, it will be able to capture more complex features. Whereas, the larger filters will extract more general features.
- A smaller kernel will be computationally more efficient. For example: To get a reception field of 5x5, we need to apply a 3x3 filter twice ((3*3)+(3*3) weights). Conversely, we will apply a 5x5 filter once.(5*5 weights).
- We use a 1×1 filter to down sample the depth or number of feature maps.
- Thus smaller kernels make deeper networks and larger make wider. The above points also provide the argument behind why deeper networks perform better than wider networks.
![](https://i.kym-cdn.com/photos/images/newsfeed/000/531/557/a88.jpg)




## Q3 Number of times to perform 3x3 convolutions to reach 1x1 from 199x199

### --------Specify number and detail the approach used. like how the size decreases by 2 after each 3x3-----

199x199 > 197x197 > 195x195 > 193x193 > 191x191 > 189x189 > 187x187 > 185x185 > 183x183 > 181x181 > 179x179 > 177x177 > 175x175 > 173x173 > 171x171 > 169x169 > 167x167 > 165x165 > 163x163 > 161x161 > 159x159 > 157x157 > 155x155 > 153x153 > 151x151 > 149x149 > 147x147 > 145x145 > 143x143 > 141x141 > 139x139 > 137x137 > 135x135 > 133x133 > 131x131 > 129x129 > 127x127 > 125x125 > 123x123 > 121x121 > 119x119 > 117x117 > 115x115 > 113x113 > 111x111 > 109x109 > 107x107 > 105x105 > 103x103 > 101x101 > 99x99 > 97x97 > 95x95 > 93x93 > 91x91 > 89x89 > 87x87 > 85x85 > 83x83 > 81x81 > 79x79 > 77x77 > 75x75 > 73x73 > 71x71 > 69x69 > 67x67 > 65x65 > 63x63 > 61x61 > 59x59 > 57x57 > 55x55 > 53x53 > 51x51 > 49x49 > 47x47 > 45x45 > 43x43 > 41x41 > 39x39 > 37x37 > 35x35 > 33x33 > 31x31 > 29x29 > 27x27 > 25x25 > 23x23 > 21x21 > 19x19 > 17x17 > 15x15 > 13x13 > 11x11 > 9x9 > 7x7 > 5x5 > 3x3 > 1x1

## Q4 How are kernels initialized? 

Kernels are initialized with random numbers. The range and distribution of these random numbers can be debated but they have to be random nonetheless.
Randomness helps the neural network to break symmetry faster and converge. It gives the initial noise or mutation required for its evolution.
For example: If kernels are initialised with 0 then the entire neural network is equivalent to one single neuron with 0. There is no non-linearity at this point. For NN to discover it takes a painful amount of time.
![](https://miro.medium.com/max/450/1*_wS_ul0act9fCT-b7SuONQ.png)


#### Kernel initialization methods in PyTorch

> From PyTorch documentation

1. uniform weights

   ```py
   torch.nn.init.uniform_(tensor, a=0.0, b=1.0)
   ```

2. normally distributed weights

   ```
   torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
   ```

3. Constant weights

   ```
   torch.nn.init.constant_(tensor, val)
   ```

4. All ones

   ```
   torch.nn.init.ones_(tensor)
   ```

5. All zeroes

   ```
   torch.nn.init.zeros_(tensor)
   ```

6. Identity matrix

   ```
   torch.nn.init.eye_(tensor)
   ```

7. Dirac delta function

   > **From PyTorch docs**
   >
   > Preserves the identity of the inputs in Convolutional layers, where as many input channels are preserved as possible. In case of groups>1, each group of channels preserves identity.

   ```
   torch.nn.init.dirac_(tensor, groups=1)
   ```

8. Xavier uniform

   ```
   torch.nn.init.xavier_uniform_(tensor, gain=1.0)
   ```

9. Xavier normal

   ```
   torch.nn.init.xavier_uniform_(tensor, gain=1.0)
   ```

10. Kaiming uniform

    ```
    torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
    ```

11. Kaiming normal

    ```
    torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
    ```

12. orthogonal initialization

    ```
    torch.nn.init.orthogonal_(tensor, gain=1)
    ```

13. Sparse initialization

    ```
    torch.nn.init.sparse_(tensor, sparsity, std=0.01)
    ```

## Q5 What happens during the training of a DNN?

To get an intuition, let’s see this analogy.

It’s like the way we learn in presence of our teacher who helps us get closer to the correct answer. Our purpose is to formulate an approximate function f(x) which outputs the correct value of y. The algorithm iterates over the data, makes predictions and the learning algorithm acts like a teacher to make the model learn and reduce the error/loss at every iteration.

A student’s learning reflects in the tests he takes and so, these tests are absolutely necessary and have tremendous importance for the student in order to learn further. A loss function is a test for our model and its value, the loss/error is the test result. Based on these, the teacher, back propagation algorithm, corrects the model and makes it learn.

Back propagation is an algorithm for supervised learning, that uses a technique called Gradient Descent to update the learnable parameters of the network, i.e. weights and biases, in order to minimise the error made by the network. Gradient descent is an iterative optimization algorithm that takes steps in the opposite direction to the gradient of a differentiable function in order to reach its local minima.

For minimising the loss function in our case, gradient descent would update the learnable parameters i.e. the kernel weights, the weights of the fully connected layers and the biases, with small steps in the opposite direction to the gradient of the loss function. After an iteration over the entire training data, called an epoch, gradient descent would update the learnable parameters in order to reduce the value of the loss function. As the iterations increase, the value of the loss function would decrease and would eventually reach its minimum.

The algorithm is called back-propagation because the error in prediction can only be determined in the last layer. And the error values of previous layers have to be derived from this in a backward direction layer by layer.