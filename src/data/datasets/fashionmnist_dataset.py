import os.path
import torch
import torchvision
import matplotlib.pyplot as plt

from PIL import Image

from torch.utils.data import Dataset


class FashionMnist(Dataset):
    """
    FashionMnist Dataset Wraper. Used for loading fashion-mnist data.
    """
    def __init__(self, path, train=False, transforms=None):
        self.path = path
        # if transform:
        #     self.transforms = torchvision.transforms.Compose(
        #         torchvision.transforms.ToTensor())
        # else:
        #     self.transforms = None
        self.transforms = transforms
        if os.path.exists(path):
            download = False
        else:
            download = True

        if train:
            set = torchvision.datasets.FashionMNIST(
                root=self.path,
                download=download,
                train=True)
        else:
            set = torchvision.datasets.FashionMNIST(
                root=self.path,
                download=download,
                train=False)

        self.data = set.data
        self.classes = set.classes
        self.classes2idx = set.class_to_idx
        self.idx2classes = {self.classes2idx[myclass]:myclass for myclass in \
                self.classes2idx}
        self.targets = set.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]
        if self.transforms:
        #     to be consistent we transform tensor to numpy and then to PIL
        #     image
            data = Image.fromarray(data.numpy(), mode="L")
            data = self.transforms(data)
        return data, target

    def normalize(self):
        """
        Func to normzalize the grayscale images of fasahion-mnist dataset
        :return: data normalized
        """
        self.data = self.data / 255.0

    def visualize(self,indexes_list):
        """
        Func to visualize specific images of the dataset
        :param indexes_list: a list containing the indexes of images that
        you want to visualize
        """
        plt.figure(figsize=(10, 10))
        for index,im in enumerate(indexes_list):
            plt.subplot(5, 5, index + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.data[im], cmap=plt.cm.binary)
            plt.xlabel(self.idx2classes[self.targets[im].item()])
        plt.show()