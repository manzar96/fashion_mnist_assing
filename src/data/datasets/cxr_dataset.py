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
            img = Image.open(os.path.join(neg_dir,file)).convert('RGB')
            img = self.transforms(img)
            data.append(img)
            targets.append(torch.tensor(0,dtype=torch.long))
        import ipdb;ipdb.set_trace()
        for file in pos_files:
            img = Image.open(os.path.join(pos_dir,file)).convert('RGB')
            data.append(self.transforms(img))
            targets.append(torch.tensor(1,dtype=torch.long))
        data=torch.stack(data)
        targets = torch.stack(targets)
        import ipdb;
        ipdb.set_trace()
        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        if self.transforms:
        #     to be consistent we transform tensor to numpy and then to PIL
        #     image
            data = Image.fromarray(data.numpy(), mode="L")
            data = self.transforms(data)
        return data, self.targets[index]

    def normalize(self):
        self.data = self.data / 255.0



