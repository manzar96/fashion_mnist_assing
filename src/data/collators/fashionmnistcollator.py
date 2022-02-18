import torch


class FashionMnistCollator(object):
    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, batch):
        inputs, labels = map(list, zip(*batch))
        inputs = torch.stack(inputs).to(self.device)
        inputs = torch.flatten(inputs, start_dim=1,end_dim=2)
        labels = torch.stack(labels).to(self.device)
        return inputs,labels


class FashionMnistCNNCollator(object):
    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, batch):
        inputs, labels = map(list, zip(*batch))
        inputs = torch.stack(inputs).to(self.device)
        inputs = torch.unsqueeze(inputs,dim=1)
        labels = torch.stack(labels).to(self.device)
        return inputs,labels


class FashionMnistResNetCollator(object):
    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, batch):
        inputs, labels = map(list, zip(*batch))
        inputs = torch.stack(inputs).to(self.device)
        labels = torch.stack(labels).to(self.device)
        return inputs, labels