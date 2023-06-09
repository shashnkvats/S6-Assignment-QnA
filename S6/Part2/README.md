# Convolutional Neural Network for MNIST Classification

<p> This notebook trains a Convolutional Neural Network (CNN) on the MNIST dataset using PyTorch. The MNIST dataset consists of handwritten digits from 0 to 9, and the goal of the network is to correctly classify these digits.</p>

## Network Architecture
<p>The network architecture consists of two convolutional layers, each followed by a batch normalization layer and a max pooling operation. After the convolutional layers, the data is flattened and passed through a fully connected layer, followed by a dropout layer for regularization, and finally through another fully connected layer to produce the output. The output of the network is a distribution over the 10 digit classes (0-9).</p>

## Training
<p>The network is trained using Stochastic Gradient Descent (SGD) with a learning rate of 0.01 and momentum of 0.9. The training process runs for 20 epochs. During each epoch, the network's parameters are updated to minimize the negative log likelihood loss between the network's predictions and the true labels.</p>

## Testing
<p>After each epoch, the network's performance is evaluated on a separate test set. The average loss and accuracy of the network on the test set are printed out.</p>

## Requirements
This notebook requires the following Python libraries:

1. PyTorch
2. torchvision
3. tqdm

## Results
<p>The model is able to achieve the accuracy of 99.12% on test dataset.</p>