# Simpsons_Character_Classification
For this assignment you will be writing a Pytorch program that learns to classify 14 different Simpsons Characters using the grey scale images we provide.

![image](https://user-images.githubusercontent.com/43845085/130785055-c6fd9a37-9164-4ef9-9be9-ccd55094053f.png)

The provided file hw2main.py handles the following:
- loading the images from the data directory
- splitting the data into training and validation sets (in the ratio specified by train_val_split) 
- data transformation: images are loaded and converted to tensors; this allows the network to work with the data; you can optionally modify and add your own transformation steps, and you can specify different transformations for the training and testing phase if you wish
- loading the data using DataLoader() provided by pytorch with your specified batch_size in student.py
