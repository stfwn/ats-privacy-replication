from typing import Any, Callable, Optional, Union, List
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from torchvision import datasets, transforms

from original.benchmark.comm import construct_policy


class DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        policy_list: List[List[int]] = None,
        num_workers: int = 8,
        batch_size: int = 128,
        pin_memory: bool = True,
        *args,
        **kwargs,
    ):
        if policy_list is None:
            policy_list = [[]]

        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.policy_list = policy_list
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=True)
        self.dataset_class(self.data_dir, train=False, download=True)

    def setup(self, stage):
        # Transforms in the 'aug' mode
        if any(p for p in self.policy_list):
            data_transform = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                construct_policy(self.policy_list),
            ]
        # Default transforms in the 'normal' mode.
        data_transform.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        data_transform = transforms.Compose(data_transform)

        # No train/val split used in original paper
        self.train = self.dataset_class(
            self.data_dir,
            train=True,
            transform=data_transform,
        )
        self.test = self.dataset_class(
            self.data_dir,
            train=False,
            transform=data_transform,
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

    def val_dataloader(self):
        """Original paper used test set as val set so we must too."""
        return self.test_dataloader()

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
