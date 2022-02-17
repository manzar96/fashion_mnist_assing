import os.path
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import random

from torch.utils.data import Dataset


class CXR(Dataset):
    """
    This class wraps the CXR Dataset.
    """
    def __init__(self, path, transforms=None):
        self.path = path
        self.transforms = transforms
        self.data, self.targets = self.read_data(path)

    def read_data(self,path):
        neg_dir = os.path.join(path,'G1/')
        pos_dir = os.path.join(path,'G2/')
        neg_files = os.listdir(neg_dir)
        pos_files = os.listdir(pos_dir)
        data = []
        targets = []
        for file in neg_files:
            img = Image.open(os.path.join(neg_dir, file)).convert("RGB")
            if self.transforms:
                img = self.transforms(img)
            data.append(img)
            targets.append(torch.tensor(0,dtype=torch.long))
        for file in pos_files:
            img = Image.open(os.path.join(pos_dir,file)).convert("RGB")
            if self.transforms:
                img=self.transforms(img)
            data.append(img)
            targets.append(torch.tensor(1,dtype=torch.long))
        data=torch.stack(data)
        targets = torch.stack(targets)
        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def normalize(self):
        self.data = self.data / 255.0



