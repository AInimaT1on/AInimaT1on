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


class SignNN(nn.Module):
    def __init__(self, input_dimensions, neurons_l1, neurons_l2, neurons_l3):
        super(SignNN, self).__init__()
        self.l1 = nn.Linear(input_dimensions, neurons_l1)
        self.l2 = nn.Linear(neurons_l1, neurons_l2)
        self.l3 = nn.Linear(neurons_l2, neurons_l3)
        self.final = nn.Linear(neurons_l3, 26)

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
