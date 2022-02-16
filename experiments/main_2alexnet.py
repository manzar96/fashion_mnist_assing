import torch
import torchvision
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD

from src.utils.parser import parse_args
from src.utils.config import Config
from src.data.datasets.fashionmnist_dataset import FashionMnist
from src.data.collators.fashionmnistcollator import FashionMnistCollator,\
    FashionMnistResNetCollator

from src.models.alexnet import AlexNet
from src.trainers.trainers import ClassificationTrainer
from src.utils.metrics import MetricsWraper


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

args = parse_args()

# make config
config = Config(args=args,device=DEVICE)

# load dataset
transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(
    256),torchvision.transforms.CenterCrop(
    224),torchvision.transforms.ToTensor()])

dataset_train = FashionMnist(path='./data/FashionMNIST', train=True,
                             transforms=transforms)

dataset_test = FashionMnist(path='./data/FashionMNIST', train=False,
                             transforms=transforms)

dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
collator_fn = FashionMnistResNetCollator(device=DEVICE)


# make model
model = AlexNet(num_classes=10, pretrained=True)

optimizer = Adam(model.parameters(), lr=config.lr , eps=1e-07)
# optimizer = SGD(model.parameters(), lr=config.lr)

metrics = MetricsWraper(['accuracy'])

# make trainer
trainer = ClassificationTrainer(model=model, optimizer=optimizer,
                                config=config, metrics=metrics)

trainer.kfoldvalidation(k_folds=5, epochs=config.epochs, dataset=dataset,
                        collator_fn=collator_fn,
                        batch_size=config.batch_size)





