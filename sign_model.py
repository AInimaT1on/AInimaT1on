import cv2
import numpy as np
import pandas as pd
from collections import OrderedDict
import time
import torch, requests, time, math
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

#model = SignNN(784, 64, 32, 16)
class SignNN(nn.Module):
    def __init__(self, input_dimensions, neurons_l1, neurons_l2, neurons_l3):
        super(SignNN, self).__init__()
        self.l1 = nn.Linear(input_dimensions, neurons_l1)
        self.l2 = nn.Linear(neurons_l1, neurons_l2)
        self.l3 = nn.Linear(neurons_l2, neurons_l3)
        self.final = nn.Linear(neurons_l3, 24)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        x = self.final(x)
        x = F.softmax(x, dim=1)
        return x

class SignNN2(nn.Module):
    def __init__(self):
        super(SignNN2, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=24)


    def forward(self, x):
        out = self.layer1(x)
        #print("done this")
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
