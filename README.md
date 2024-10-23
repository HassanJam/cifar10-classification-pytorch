                                                                CIFAR-10 Image Classification using PyTorch
This project implements a simple Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset. The dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
S
Dataset
The CIFAR-10 dataset is used for training and testing the model. It contains 10 different classes of images:

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
The dataset is divided into:

50,000 training images
10,000 test images
Model Architecture
The model is a simple CNN consisting of the following layers:

Convolutional Layer: 3x3 kernels, 32 filters
Convolutional Layer: 3x3 kernels, 32 filters
Max Pooling: 2x2 window
Convolutional Layer: 3x3 kernels, 64 filters
Convolutional Layer: 3x3 kernels, 64 filters
Max Pooling: 2x2 window
Fully Connected Layer: 128 units
Output Layer: 10 classes (Softmax activation for classification)
Training
The model is trained using the following configuration:

Loss Function: Cross Entropy Loss
Optimizer: Stochastic Gradient Descent (SGD) with momentum
Learning Rate: 0.001
Weight Decay: 0.005
Momentum: 0.9
Batch Size: 64
Number of Epochs: 20
Prerequisites
Python 3.x
PyTorch
Torchvision
Numpy
