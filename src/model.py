"""
モデル定義
"""
from typing import Tuple
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
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
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # PIL.Image -> Tensor
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # hyper parameters
        self.batch_size = 128

        # metrics
        self.train_acc = ptl.metrics.Accuracy()
        self.val_acc = ptl.metrics.Accuracy()
        self.test_acc = ptl.metrics.Accuracy()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ResNet: (batch, channels, h, w) -> (batch, dim)
        y = self.model.forward(x)
        return y

    def training_step(self, batch: Batch, batch_idx: int):
        x, y = batch
        y_: torch.Tensor = self(x)
        loss = F.cross_entropy(y_, y)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        x, y = batch
        y_ = self(x)
        loss = F.cross_entropy(y_, y)
        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

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
