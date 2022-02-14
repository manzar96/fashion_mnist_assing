import os.path
import torch
import torchvision
import matplotlib.pyplot as plt

from torch.utils.data import Dataset


class FashionMnist(Dataset):

    def __init__(self, path, train=False, transform=True):
        self.path = path
        if transform:
            self.transforms = torchvision.transforms.Compose(
                torchvision.transforms.ToTensor())
        else:
            self.transforms = None

        if os.path.exists(path):
            download = False
        else:
            download = True

        if train:
            set = torchvision.datasets.FashionMNIST(
                root=self.path,
                download=download,
                train=True,
            transform=self.transforms)
        else:
            set = torchvision.datasets.FashionMNIST(
                root=self.path,
                download=download,
                train=False,
                transform=self.transforms)

        self.data = set.data
        self.classes = set.classes
        self.classes2idx = set.class_to_idx
        self.idx2classes = {self.classes2idx[myclass]:myclass for myclass in \
                self.classes2idx}
        self.targets = set.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def normalize(self):
        self.data = self.data / 255.0

    def visualize(self,indexes_list):
        plt.figure(figsize=(10, 10))
        for index,im in enumerate(indexes_list):
            plt.subplot(5, 5, index + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.data[im], cmap=plt.cm.binary)
            plt.xlabel(self.idx2classes[self.targets[im].item()])
        plt.show()