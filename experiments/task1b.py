import torch
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD

from src.utils.parser import parse_args
from src.utils.config import Config
from src.data.datasets.fashionmnist_dataset import FashionMnist
from src.data.collators.fashionmnistcollator import FashionMnistCollator,FashionMnistCNNCollator
from src.models.mlp import MLP
from src.models.cnn import CNN,CNN2
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
dataset_test = FashionMnist(path='./data/FashionMNIST', train=False)
dataset_test.normalize()
dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
collator_fn = FashionMnistCNNCollator(device=DEVICE)


# make model
# model = MLP(input_dim=28*28, out_dim=10)
model = CNN()

optimizer = Adam(model.parameters(), lr=config.lr , eps=1e-07)
# optimizer = SGD(model.parameters(), lr=config.lr)

metrics = MetricsWraper(['accuracy'])

# make trainer
trainer = ClassificationTrainer(model=model, optimizer=optimizer,
                                config=config, metrics=metrics)

trainer.kfoldvalidation(k_folds=5, epochs=config.epochs, dataset=dataset,
                        collator_fn=collator_fn,
                        batch_size=config.batch_size)





