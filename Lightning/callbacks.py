import os
from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from validate_and_test import LitAutoEncoder, Encoder, Decoder
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")
    
    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
    

model = LitAutoEncoder(Encoder(), Decoder())

dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(batch_size=256, dataset=dataset, num_workers=7)

trainer = Trainer(callbacks=[MyPrintingCallback()], max_epochs=1)
trainer.fit(model=model, train_dataloaders=train_loader)
