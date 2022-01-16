from typing import Any, Callable, Optional, Union
from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule
from torchvision import datasets, transforms


class FashionMNISTDataModule(LightningDataModule):
    # TODO

    mean = (0.1307,)
    std = (0.3081,)

    @property
    def num_channels(self):
        return self.dims[0]


class CIFAR100DataModule(LightningDataModule):
    name = "cifar_100"
    dims = (3, 32, 32)
    num_classes = 100

    mean = (0.5071598291397095, 0.4866936206817627, 0.44120192527770996)
    std = (0.2673342823982239, 0.2564384639263153, 0.2761504650115967)

    def __init__(
        self,
        data_dir: Optional[str] = "data",
        num_workers: int = 8,
        batch_size: int = 128,
        pin_memory: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            num_workers: How many workers to use for loading data
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory

    def prepare_data(self):
        datasets.CIFAR100(self.data_dir, train=True, download=True)
        datasets.CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage):
        # No train/val split used in original paper
        self.train = datasets.CIFAR100(
            self.data_dir,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            ),  # Default transforms in the 'normal' mode.
        )
        self.test = datasets.CIFAR100(
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
