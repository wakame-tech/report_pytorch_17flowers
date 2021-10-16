"""
モデル定義
"""
from typing import Tuple
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import pytorch_lightning as ptl
import torchvision.transforms as transforms

from dataset import Flower17
from parameters import Parameters

# (batch, channel, h, w) -> (batch, 1)
Batch = Tuple[torch.Tensor, torch.Tensor]

class Model(ptl.LightningModule):
    def __init__(self, params: Parameters):
        super().__init__()

        self.params = params
        # TODO
        self.transform = transforms.Compose([
            # resize to (224, 224)
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.Grayscale(),
            # PIL.Image -> Tensor
            transforms.ToTensor(),
        ])

        # hyper parameters
        self.batch_size = 4

        # metrics
        self.train_acc = ptl.metrics.Accuracy()
        self.val_acc = ptl.metrics.Accuracy()
        self.test_acc = ptl.metrics.Accuracy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, h, w)
        print(x.shape)

        # TODO
        pass


    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_: torch.Tensor = self(x)
        loss: torch.Tensor = F.nll_loss(y_, y)
        print(f'loss: {loss.item()}')
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        x, t = batch
        y: torch.Tensor = self(x)
        loss: torch.Tensor = F.nll_loss(y, t)
        acc: torch.Tensor = self.val_acc(y, t)

        print(f'val_loss: {loss.item()}')
        print(f'val_acc: {acc.item()}')

        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def setup(self, stage=None):
        dataset = Flower17(params=self.params, transform=self.transform)
        print(f'image: {type(dataset[0][0])}, label: {type(dataset[0][1])}')
        train_length = int(len(dataset) * 0.9)
        val_length = len(dataset) - train_length
        self.train_dataset, self.val_dataset = random_split(dataset, [train_length, val_length])
        # TODO
        self.test_dataset = Flower17(params=self.params, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
