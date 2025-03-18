""" data accessibility processing
Author@Mingyang
"""
import os
import sys
import pickle

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from scipy.io import loadmat
from torchvision import transforms

import random
import re
from torch.utils.data import DataLoader
import operator

''' pytorch dataset loading
'''
# transform the images
class Trans:
    # if transform is given, we transform data using
    def __init__(self):
        pass

    def transforms_Train(self):
        transform_1 = transforms.Compose([

            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor()  
        ])

        return transform_1

    def transforms_Test(self):
        transform_2 = transforms.Compose([
            transforms.ToTensor(),
        ])

        return transform_2


class BatteryDataset(Dataset):
    def __init__(self, root, file_name, prefix, transform=None):
        super(BatteryDataset, self).__init__() 
        self.path = os.path.join(root, file_name)
        self.txt_root = root + '\\' + (prefix + file_name + '_dataset.txt')

        f = open(self.txt_root, 'r')
        data = f.readlines()

        imgs = []
        labels_value = []
        for line in data:
            word = line.split()
            imgs.append(os.path.join(self.path, word[2], word[0]))
            labels_value.append(int(word[1]))
        self.img = imgs
        self.label = labels_value
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]

        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)

        return img, label



