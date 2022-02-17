import torch
import timm
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD

from src.utils.parser import parse_args
from src.utils.config import Config
from src.data.datasets.cxr_dataset import CXR
from src.data.collators.cxrcollators import CXRXCeptionCollator
from src.models.mlp import MLP
from src.trainers.trainers import ClassificationTrainer
from src.utils.metrics import MetricsWraper


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

args = parse_args()

# make config
config = Config(args=args,device=DEVICE)

# load dataset
dataset = CXR(path='./data/CXR',transforms=None)
collator_fn = CXRXCeptionCollator(device=DEVICE)
# make model
# we load the pretrained XCeption Model and we add a classification head for
# binary classification
model = timm.create_model('xception', pretrained=True, num_classes=2)


optimizer = Adam(model.parameters(), lr=config.lr , eps=1e-07)
# optimizer = SGD(model.parameters(), lr=config.lr)

metrics = MetricsWraper(['accuracy'])

# make trainer
trainer = ClassificationTrainer(model=model, optimizer=optimizer,
                                config=config, metrics=metrics)

trainer.kfoldvalidation(k_folds=5, epochs=config.epochs, dataset=dataset_train,
                        collator_fn=collator_fn,
                        batch_size=config.batch_size)
