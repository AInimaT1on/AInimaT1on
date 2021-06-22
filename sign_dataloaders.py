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

class SignDataLoader(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        super(SignDataLoader).__init__()
        self.train_df = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = self.train_df['label']
        self.images = self.train_df.drop(['label'], axis=1)
        print(len(self.labels.unique()))

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        img_np = np.asarray(self.images.iloc[idx]).reshape(28,28).astype('uint8')
        img_label = self.labels.iloc[idx]
        if self.transform is not None:
            img_processed = self.transform(img_np)
            return (img_processed, img_label)
        return (img_np, img_label)
