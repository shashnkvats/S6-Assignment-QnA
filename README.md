# Backpropagation

<p>This excel provides an overview of a simple feed-forward neural network with three layers: an input layer, a hidden layer, and an output layer. The network uses sigmoid activation functions and a mean squared error loss function. The backpropagation algorithm is used for training.</p>

## Network Architecture

The network consists of the following layers:
* **Input Layer**: This layer has two nodes, i<sub>1</sub> and i<sub>2</sub>
* **Hidden Layer**: This layer has two nodes, represented as h<sub>1</sub> and h<sub>2</sub> before the application of the sigmoid activation function, and a<sub>h1</sub> and a<sub>h2</sub> after the application of the sigmoid activation function.
* **Output Layer**: This layer has two nodes, represented as o<sub>1</sub> and o<sub>2</sub> before the application of the sigmoid activation function, and a<sub>o1</sub> and a<sub>o2</sub> after the application of the sigmoid activation function.

## Weights
The weights connecting the layers are represented as w<sub>1</sub>, w<sub>2</sub>, w<sub>3</sub>, w<sub>4</sub>, w<sub>5</sub>, w<sub>6</sub>, w<sub>7</sub>, and w<sub>8</sub>. The weights are used as follows:

h<sub>1</sub> = i<sub>1</sub>w<sub>1</sub> + i<sub>2</sub>w<sub>2</sub> <br>
h<sub>2</sub> = i<sub>1</sub>w<sub>3</sub> + i<sub>2</sub>w<sub>4</sub> <br>
o<sub>1</sub> = w<sub>5</sub>a<sub>1</sub> + w<sub>6</sub>a<sub>h2</sub> <br>
o<sub>2</sub> = w<sub>7</sub>a<sub>h1</sub> + w<sub>8</sub>a<sub>h2</sub> <br>

## Activation Function
The sigmoid function is used as the activation function in the hidden and output layers. After the application of the sigmoid function, the hidden layer nodes are represented as a<sub>h1</sub> and a<sub>h2</sub>, and the output layer nodes are represented as a<sub>o1</sub> and a<sub>o2</sub>.


## Loss Function
The network uses a mean squared error loss function. The targets for the output layer nodes are represented as t<sub>1</sub> and t<sub>2</sub>. The individual errors for the output nodes are calculated as follows:

E<sub>1</sub> = 1/2*(t<sub>1</sub> - a<sub>o1</sub>)<sup>2</sup>
E<sub>2</sub> = 1/2*(t<sub>2</sub> - a<sub>o2</sub>)<sup>2</sup>
The total loss E<sub>tot</sub> is the sum of E<sub>1</sub> and E<sub>2</sub>.

## Learning Rate
<p>The learning rate is a hyperparameter that determines the step size at each iteration while moving toward a minimum of a loss function. It plays a crucial role in the training of neural networks.</p> 

1. **Very Small Learning Rate**: If the learning rate is very small, the steps towards the minimum of the loss function will also be very small. This means the network will learn very slowly. While this might lead to a very precise convergence (because it takes small steps and is less likely to overshoot the minimum), it also means that the network will need a lot of time (i.e., many epochs) to converge, which can be computationally expensive. There's also a risk that the learning process gets stuck in a local minimum rather than finding the global minimum.
2. **Extremely Large Learning Rate**: If the learning rate is too large, the steps will also be large. This can cause the learning process to overshoot the minimum of the loss function and potentially result in divergence, meaning the network fails to learn. In the worst case, the loss function could become NaN (Not a Number), a situation known as "exploding gradients". This is usually a sign that the learning rate is too high.

<p>Hence, care must be taken while chosing the learning rate.</p>
<bold> The images showing the effect of change in Loss wrt LR is presnt in repo.</bold>


## Backpropagation
<p>After the forward pass through the network and the calculation of the loss, the backpropagation algorithm is used to update the weights in the network. This involves calculating the derivative of the loss with respect to each weight, and then adjusting the weights in the direction that reduces the loss.</p>

## Backpropagation Calculation
<p>Backpropagation is the method used to update the weights in the network. It involves calculating the partial derivatives of the total error with respect to the weights, and then adjusting the weights in the direction that reduces the error. The following equations are used to calculate these partial derivatives:</p>

![Screenshot](backpropagation.png)

* Calculate the partial derivative of the total error with respect to the weights between the hidden and output layers:<br>
    ∂E<sub>total</sub>/∂w<sub>5</sub> = (a<sub>01</sub> - t<sub>1</sub>) * a<sub>o1</sub> * (1 -a<sub>o1</sub>) *  a<sub>h1</sub> <br>
    ∂E<sub>total</sub>/∂w<sub>6</sub> = (a<sub>01</sub> - t<sub>1</sub>) * a<sub>o1</sub> * (1 -a<sub>o1</sub>) *  a<sub>h2</sub> <br>
    ∂E<sub>total</sub>/w<sub>7</sub> = (a<sub>02</sub> - t<sub>2</sub>) * a<sub>o2</sub> * (1 - a<sub>o2</sub>) *  a<sub>h1</sub> <br>
    ∂E<sub>total</sub>/∂w<sub>8</sub> = (a<sub>02</sub> - t<sub>2</sub>) * a<sub>o2</sub> * (1 - a<sub>o2</sub>) *  a<sub>h2</sub> <br>

* Calculate the partial derivative of the total error with respect to the outputs of the hidden layer:

    ∂E<sub>total</sub>/∂a<sub>h1</sub> = (a<sub>01</sub> - t<sub>1</sub>) * a<sub>o1</sub> * (1 - a<sub>o1</sub>) * w<sub>5</sub> +  (a<sub>02</sub> - t<sub>2</sub>) * a<sub>o2</sub> * (1 - a<sub>o2</sub>) * w<sub>7</sub> <br>
    ∂E<sub>total</sub>/∂a<sub>h2</sub> = (a<sub>01</sub> - t<sub>1</sub>) * a<sub>o1</sub> * (1 - a<sub>o1</sub>) * w<sub>6</sub> +  (a<sub>02</sub> - t<sub>2</sub>) * a<sub>o2</sub> * (1 - a<sub>o2</sub>) * w<sub>8</sub>

* Calculate the partial derivative of the total error with respect to the weights between the input and hidden layers:

    ∂E<sub>total</sub>/∂w<sub>1</sub> = ((a<sub>o1</sub> - t<sub>1</sub>) * a<sub>o1</sub> * (1 - a<sub>01</sub>) * w<sub>5</sub> +  (a<sub>o2</sub> - t<sub>2</sub>) * a<sub>o2</sub> * (1 - a<sub>o2</sub>) * w<sub>7</sub>) * a<sub>h1</sub> * (1 - a<sub>h1</sub>) * i<sub>1</sub> <br>
    ∂E<sub>total</sub>/∂w<sub>2</sub> = ((a<sub>o1</sub> - t<sub>1</sub>) * a<sub>o1</sub> * (1 - a<sub>01</sub>) * w<sub>5</sub> +  (a<sub>o2</sub> - t<sub>2</sub>) * a<sub>o2</sub> * (1 - a<sub>o2</sub>) * w<sub>7</sub>) * a<sub>h1</sub> * (1 - a<sub>h1</sub>) * i<sub>2</sub> <br>
    ∂E<sub>total</sub>/∂w<sub>3</sub> = ((a<sub>o1</sub> - t<sub>1</sub>) * a<sub>o1</sub> * (1 - a<sub>01</sub>) * w<sub>6</sub> +  (a<sub>o2</sub> - t<sub>2</sub>) * a<sub>o2</sub> * (1 - a<sub>o2</sub>) * w<sub>8</sub>) * a<sub>h2</sub> * (1 - a<sub>h2</sub>) * i<sub>1</sub> <br>
    ∂E<sub>total</sub>/∂w<sub>4</sub> = ((a<sub>o1</sub> - t<sub>1</sub>) * a<sub>o1</sub> * (1 - a<sub>01</sub>) * w<sub>6</sub> +  (a<sub>o2</sub> - t<sub>2</sub>) * a<sub>o2</sub> * (1 - a<sub>o2</sub>) * w<sub>8</sub>) * a<sub>h2</sub> * (1 - a<sub>h2</sub>) * i<sub>2</sub> <br>

<p>These partial derivatives are then used to update the weights in the network. The weights are adjusted in the direction that reduces the total error. This process is repeated for many iterations until the network is adequately trained and the total error is minimized.</p>


