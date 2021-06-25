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
from torchvision import datasets, transforms, models
from helper_functions import calc_accuracy
from sign_dataloaders import SignDataLoader
from sign_model import SignNN, SignNN2
import torchvision.models as models
from mobilenetv3 import mobilenetv3_large, mobilenetv3_small

def train_MyNN(x_train_loader,x_test_loader, model, lr, epochs):
    if torch.cuda.is_available():
        print("Yay cuda is working")
    else:
        print("cuda was not available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()
    acc_test = []
    epoch_losses = []
    running_loss = 0
    max_accuracy = 0
    model.train()
    model.to(device)

    for epoch in range(epochs):
        running_losses = []
        print(f"EPOCH: {epoch+1}/{epochs}")

        for i, (images, labels) in enumerate(iter(x_train_loader)):

            images, labels = images.to(device), labels.to(device)
            #images.resize_(images.size()[0],784)
            #print(images.shape)
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
            total_acc = []
            for images, labels in iter(x_test_loader):
                #images.resize_(images.size()[0],784)
                images, labels = images.to(device), labels.to(device)
                max_vals, max_indices = model(images).max(1)
                # assumes the first dimension is batch size
                n = max_indices.size(0)  # index 0 for extracting the # of elements
                # calulate acc (note .item() to do float division)
                acc = (max_indices == labels).sum().item() / n
                total_acc.append(acc)

            final_acc = sum(total_acc) / len(total_acc)
            print(f"The average accuracy across all tests: {final_acc}, test_size: {len(total_acc)}")
            acc_test.append(finl_acc)
            if final_acc > max_accuracy:
                torch.save(model, 'mobilenetv3_large100_img.pth')
                max_accuracy = final_acc

        model.train()

    print(f" THE MAX ACCURACY ACHIEVED {max_accuracy}")
    return model


################################################################################
train_transforms = transforms.Compose([
                            #transforms.ToPILImage(),
                            transforms.RandomRotation(30),
                            transforms.ColorJitter(),
                            #transforms.Grayscale(),
                            #transforms.RandomPerspective(),
                            transforms.RandomHorizontalFlip(),
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                             ])

test_transforms = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                             ])

#train_data = SignDataLoader("data/train data/sign_mnist_train.csv", transform=train_transforms)
#test_data = SignDataLoader("data/test data/sign_mnist_test.csv", transform=test_transforms)
train_data = datasets.ImageFolder("../data/signdata/Train", transform=train_transforms)# "../sign_datav2/Train" "../data/signdata/Train" "../sign_datav2/random_train_100"
test_data = datasets.ImageFolder("../data/signdata/Test", transform=test_transforms)#"../sign_datav2/Test" #"../data/signdata/Test" "../sign_datav2/random_test_100"
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
classifier = nn.Sequential(OrderedDict([
                            ("fc1", nn.Linear(in_features = 960, out_features=1280, bias=True)),
                            ("ReLU1", nn.ReLU()),
                            ("fc2", nn.Linear(in_features=1280, out_features=500, bias=True)),
                            ("ReLU2", nn.ReLU()),
                            ("fc3", nn.Linear(in_features=500, out_features=5, bias=True)),
                            #("OUT", nn.LogSoftmax(dim=1)),
                            ]))


model = mobilenetv3_large()
model.load_state_dict(torch.load("pretrained/mobilenetv3-large-1cd25616.pth"))#'pretrained/mobilenetv3-small-55df8e1f.pth'
for param in model.parameters():
    param.requires_grad = False
#model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True
#model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
#model = SignNN(784, 64, 32, 16)
#model = SignNN2()
#model = models.densenet121(pretrained=True)
#model = models.resnet18(pretrained=True)
#model = models.resnet34(pretrained=True)
#model = models.mobilenet_v3_small(pretrained=True)
#print(model)
model.classifier = classifier
trained_model = train_MyNN(train_loader, test_loader, model , 0.001, 10)
