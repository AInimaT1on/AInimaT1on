import cv2
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import time
import torch, requests, time, math
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from helper_functions import calc_accuracy
from sign_dataloaders import SignDataLoader
from sign_model import SignNN, SignNN2

def train_MyNN(x_train_loader,x_test_loader, model, lr, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()
    acc_test = []
    epoch_losses = []
    running_loss = 0
    max_accuracy = 0
    for epoch in range(epochs):
        running_losses = []
        print(f"EPOCH: {epoch+1}/{epochs}")

        for i, (images, labels) in enumerate(iter(x_train_loader)):
            #images.resize_(images.size()[0],784)
            print(images.shape)
            optimizer.zero_grad()
            #print(f"IMAGES SIZE {images.shape}")
            output = model.forward(images)


            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i%100 ==0:
                print(f"On Iteration: {i}, loss was: {round(running_loss/100, 4)}")
                running_losses.append(running_loss)
                running_loss = 0
        epoch_losses.append(loss)
        #
        # print("\n HERE \n")
        # print(epoch_losses)

        #### Validate
        model.eval()
        with torch.no_grad():
            acc = calc_accuracy(model, x_test_loader)
            acc_test.append(acc)
            if acc > max_accuracy:
                torch.save(model, 'set_path_name.pth')
                max_accuracy = acc

        model.train()

    print(f" THE MAX ACCURACY ACHIEVED {max_accuracy}")
    return model


################################################################################
train_transforms = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomRotation(30),
                            transforms.ColorJitter(),
                            #transforms.RandomPerspective(),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                             ])

test_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                             ])

train_data = SignDataLoader("data/train data/sign_mnist_train.csv", transform=train_transforms)
test_data = SignDataLoader("data/test data/sign_mnist_test.csv", transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=False)
#model = SignNN(784, 64, 32, 16)
model = SignNN2()
trained_model = train_MyNN(train_loader, test_loader, model , 0.0003, 10)
