from typing import Any, Callable, Optional, Union
from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule
from torchvision import datasets, transforms


class DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir="data",
        num_workers=8,
        batch_size=128,
        pin_memory=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=True)
        self.dataset_class(self.data_dir, train=False, download=True)

    def setup(self, stage):
        # No train/val split used in original paper
        self.train = self.dataset_class(
            self.data_dir,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            ),  # Default transforms in the 'normal' mode.
        )
        self.test = self.dataset_class(
            self.data_dir,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            ),  # Default transforms in the 'normal' mode.
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    @property
    def num_channels(self) -> int:
        return self.dims[0]


class CIFAR100DataModule(DataModule):
    dims = (3, 32, 32)
    num_classes = 100

    dataset_class = datasets.CIFAR100
    mean = (0.5071598291397095, 0.4866936206817627, 0.44120192527770996)
    std = (0.2673342823982239, 0.2564384639263153, 0.2761504650115967)


class FashionMNISTDataModule(DataModule):
    dims = (1, 28, 28)
    num_classes = 10

    dataset_class = datasets.FashionMNIST
    mean = (0.1307,)
    std = (0.3081,)
