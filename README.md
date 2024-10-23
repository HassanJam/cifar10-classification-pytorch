CIFAR-10 Image Classification using PyTorch

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. 
The dataset consists of 60,000 32x32 color images across 10 classes, and this model achieves 87% accuracy on the training data.

Table of Contents

Dataset

Model Architecture

Training Configuration

Prerequisites

Dataset
The CIFAR-10 dataset consists of 10 different classes of objects:

Airplane

Automobile

Bird

Cat

Deer

Dog

Frog

Horse

Ship

Truck

The dataset is split into:

50,000 training images
10,000 test images

Model Architecture

This CNN model is composed of the following layers:

Convolutional Layer: 32 filters, 3x3 kernel size

Convolutional Layer: 32 filters, 3x3 kernel size

Max Pooling: 2x2 window

Convolutional Layer: 64 filters, 3x3 kernel size

Convolutional Layer: 64 filters, 3x3 kernel size

Max Pooling: 2x2 window

Fully Connected Layer: 128 units

Output Layer: 10 classes (Softmax)

Training Configuration

Loss Function: Cross Entropy Loss

Optimizer: SGD with momentum

Learning Rate: 0.001

Weight Decay: 0.005

Momentum: 0.9

Batch Size: 64

Epochs: 20

Prerequisites

Make sure you have Python 3.x and the following libraries installed:

torch

torchvision

numpy
