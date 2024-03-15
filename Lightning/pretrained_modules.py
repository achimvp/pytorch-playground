import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from lightning import LightningModule, Trainer
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

class ImagenetTransferLearning(LightningModule):
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 10
        self.classifier = nn.Linear(num_filters, num_target_classes)
    
    def forward(self, x):
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        
        x = self.classifier(representations)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.cross_entropy(y_pred, y)
        return loss    
    
    def configure_optimizers(self):
        # define the optimizer for your model
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


train_set = CIFAR10(os.getcwd(), download=True, train=True, transform=transforms.ToTensor())
test_set = CIFAR10(os.getcwd(), download=True, train=False, transform=transforms.ToTensor())
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size],
                                    generator=seed)

train_loader = DataLoader(batch_size=256, dataset=train_set)
valid_loader = DataLoader(batch_size=256, dataset=valid_set)
test_loader = DataLoader(batch_size=256, dataset=test_set)

# finetune the model
model = ImagenetTransferLearning()
x,y = next(iter(test_loader))
y_pred = model(x)
print("Predicted class before training:", y_pred[0])
print("True class:", y[0])
plt.imshow(x[0].cpu().numpy().transpose((1,2,0)))
plt.show()
trainer = Trainer(max_epochs=5)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
y_pred = model(x)
print("Predicted class after training:", y_pred[0])
print("True class: ", y[0])
plt.imshow(x[0].cpu().numpy().transpose((1,2,0)))





