import torch
import timm
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD

from src.utils.parser import parse_args
from src.utils.config import Config
from src.data.datasets.fashionmnist_dataset import FashionMnist
from src.data.collators.fashionmnistcollator import FashionMnistCollator,FashionMnistCNNCollator
from src.models.mlp import MLP
from src.models.cnn import CNN
from src.trainers.trainers import ClassificationTrainer
from src.utils.metrics import MetricsWraper


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

args = parse_args()

# make config
config = Config(args=args,device=DEVICE)

# load dataset
dataset_train = FashionMnist(path='./data/FashionMNIST', train=True)
dataset_train.normalize()
collator_fn = FashionMnistCollator(device=DEVICE)
train_loader = DataLoader(dataset_train, batch_size=config.batch_size,
                          drop_last=False, shuffle=False,
                          collate_fn=collator_fn)

dataset_test = FashionMnist(path='./data/FashionMNIST', train=False)
dataset_test.normalize()
collator_fn = FashionMnistCollator(device=DEVICE)
test_loader = DataLoader(dataset_test, batch_size=config.batch_size,
                          drop_last=False, shuffle=False,
                          collate_fn=collator_fn)

# used to visualize samples
# dataset_train.visualize([10,20,32,12,24,544,100,6,11,2343])

# make model
model = MLP(input_dim=28*28, out_dim=10)
# model = CNN()

optimizer = Adam(model.parameters(), lr=config.lr , eps=1e-07)
# optimizer = SGD(model.parameters(), lr=config.lr)

metrics = MetricsWraper(['accuracy'])

# make trainer
trainer = ClassificationTrainer(model=model, optimizer=optimizer,
                                config=config, metrics=metrics)

trainer.fit(config.epochs, train_loader, val_loader=None)

trainer.test(testloader=test_loader,loadckpt=None)





