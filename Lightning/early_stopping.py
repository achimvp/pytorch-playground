import os
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from validate_and_test import LitAutoEncoder, Encoder, Decoder
from lightning import Trainer
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split

# Define the training, validation and testing dataset
train_set = MNIST(os.getcwd(), download=True, train=True, transform=transforms.ToTensor())
test_set = MNIST(os.getcwd(), download=True, train=False, transform=transforms.ToTensor())
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size],
                                    generator=seed)

train_loader = DataLoader(batch_size=256, dataset=train_set)
valid_loader = DataLoader(batch_size=256, dataset=valid_set)
test_loader = DataLoader(batch_size=256, dataset=test_set)

model = LitAutoEncoder(Encoder(), Decoder())
trainer = Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
