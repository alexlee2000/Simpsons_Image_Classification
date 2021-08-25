#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.


In terms of image transformations, after we convert images to grayscale as directed, we utilized horizontal, vertical and rotational transforms in 
order to alleviate overfitting and subsequently improve our model accuracy. I noticed that data augmentation has a large impact on how well the model
performs as it improved accuracy by a significant amount. The choice of loss function was the Cross Entropy Loss as it is used often in literature 
for classification problems such as ours. Cross Entropy Loss minimizes the distance between two probability distributions (predicted and target). 
The optimizer of choice was RMSprop which is similar to the gradient descent method with momentum. RMSprop restricts the osciallations in the vertical 
direction which enables us to increase our learning rate in order to take larger steps in the horizontal direction and thus converge faster. 
The architecture chosen was a convolutional network with 5 convolutional layers with relu activation followed by a fully connected layer with log 
softmax. I noticed that adding a convolutional layer improved the performance of the model significantly however I couldn't add another layer due to 
capacity issues and errors. Each convolutional layer has padding of 2 and kernel size 3 by 3 with every second layer followed by a batch normalization,
max pooling and dropout in order to improve accuracy and further generalize the model. After the fully connected layer we implement another dropout of 
50%. In terms of hyperparameters, the batch size was set to 32 as it showed to produce the best results, epoch was set to 100, learning rate was set 
to the default of 0.001 with a weight decay of 1e-6. The Validation set was set at 20% of the training set in order to improve generalization. 
Once I was convinced that my model was not overfitting anymore, I changed the test-validation split to 1.0. 
"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        # image data augmentation 
        trainTransform = transforms.Compose([transforms.Grayscale(1),                     
                                             transforms.RandomHorizontalFlip(p = 0.5),    
                                             transforms.RandomVerticalFlip(p = 0.5), 
                                             transforms.RandomRotation(degrees = (-30,30)),
                                             transforms.ToTensor(), 
                                             transforms.Normalize((0), (1))
                                             ])        
        return trainTransform
    elif mode == 'test':
        # ensuring test data is in grayscale
        trainTransform = transforms.Compose([transforms.ToTensor(), 
                                             transforms.Grayscale(1)
                                             ])     
        return trainTransform

############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 2)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 2)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels =  64, kernel_size = 3, padding = 2)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels =  64, kernel_size = 3, padding = 2)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels =  86, kernel_size = 3, padding = 2)

        self.fc_layer = nn.Linear(37926, 250)
        self.fc_output_layer = nn.Linear(250, 14) 
        
    def forward(self, t):
        # 1st convolutional layer 
        out = F.relu(self.conv1(t))

        # 2nd convolutional layer
        out = F.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = F.dropout(out, 0.25)

        # 3rd convolutional layer
        out = F.relu(self.conv3(out))
        
        # 4th convolutional layer 
        out = F.relu(self.conv4_bn(self.conv4(out)))
        out = F.max_pool2d(out, 2)
        out = F.dropout(out, 0.25)

        # 5th convolutional layer 
        out = F.relu(self.conv5(out))

        # fully connected layer 
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc_layer(out))
        out = F.dropout(out, 0.50)

        # output layer
        out = F.log_softmax(self.fc_output_layer(out), dim = 1)
        return out

net = Network()
lossFunc = nn.CrossEntropyLoss()

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 1.0
batch_size = 32 
epochs = 100
optimiser = optim.RMSprop(net.parameters(), lr=0.001, weight_decay=1e-6)