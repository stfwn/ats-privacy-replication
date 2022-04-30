import os
import random
import torch

from typing import Callable, Optional, List, Type
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive

from policy import MetaPolicy


class DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        policy_list: List[List[int]] = None,
        num_workers: int = os.cpu_count(),
        batch_size: int = 128,
        pin_memory: bool = True,
        apply_policy_to_test: bool = False,
        tiny_subset_size: float = None,
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
        self.apply_policy_to_test = apply_policy_to_test
        self.tiny_subset_size = tiny_subset_size

        self.mean: torch.Tensor
        self.std: torch.Tensor

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=True)
        self.dataset_class(self.data_dir, train=False, download=True)

    def setup(self, stage):
        # 'crop' mode
        data_transform = [
            transforms.RandomCrop(self.dims[1], padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        # Additional transforms in 'aug' mode
        if any(p for p in self.policy_list):
            data_transform.extend(
                [
                    MetaPolicy.from_indices(self.policy_list),
                ]
            )
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
        if self.tiny_subset_size is not None:
            self.train = _get_random_subset(self.train, self.tiny_subset_size)

        if self.apply_policy_to_test and any(self.policy_list):
            test_transform = data_transform
        else:
            test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )
        self.test = self.dataset_class(
            self.data_dir,
            train=False,
            transform=test_transform,
        )
        if self.tiny_subset_size is not None:
            self.test = _get_random_subset(self.test, self.tiny_subset_size)

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
            shuffle=False,
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
    mean = torch.tensor(
        [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
    )
    std = torch.tensor(
        [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
    )


class FashionMNISTDataModule(DataModule):
    dims = (1, 28, 28)
    num_classes = 10

    dataset_class = datasets.FashionMNIST
    mean = torch.tensor((0.1307,))
    std = torch.tensor((0.3081,))


class TinyImageNet200(datasets.ImageFolder):
    base_folder = "tiny-imagenet-200"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    tgz_md5 = "90528d7ca1a48142e341f4ef8d21d0de"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self.train = train  # training set or test set
        self.root = root

        if download:
            self.download()

        super().__init__(
            self.datadir, transform=transform, target_transform=target_transform
        )

    @property
    def datadir(self):
        subdir = "train" if self.train else "val"
        return os.path.join(self.root, self.base_folder, subdir)

    def download(self) -> None:
        """Download the data if it doesn't exist already."""
        if os.path.exists(os.path.join(self.root, self.base_folder)):
            return

        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=self.tgz_md5
        )

        val_dir = os.path.join(self.root, self.base_folder, "val")
        val_img_dir = os.path.join(val_dir, "images")
        val_annot_f = os.path.join(val_dir, "val_annotations.txt")

        # Open and read val annotations text file
        val_img_dict = {}
        with open(val_annot_f) as f:
            for line in f:
                words = line.split("\t")
                val_img_dict[words[0]] = words[1]

        for img, folder in val_img_dict.items():
            newpath = os.path.join(val_dir, folder)
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            if os.path.exists(os.path.join(val_img_dir, img)):
                os.rename(
                    os.path.join(val_img_dir, img), os.path.join(newpath, img)
                )
        os.rmdir(val_img_dir)


class TinyImageNet200DataModule(DataModule):
    dims = (3, 64, 64)
    num_classes = 200

    dataset_class = TinyImageNet200
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])


def get_by_name(name: str) -> Type[DataModule]:
    options = {
        "cifar100": CIFAR100DataModule,
        "fmnist": FashionMNISTDataModule,
        "tiny-imagenet200": TinyImageNet200DataModule,
    }
    if dm := options.get(name, None):
        return dm
    raise ValueError(f"Unknown data module: {name}")


def _get_random_subset(dataset, size: float):
    dataset_indices = list(range(len(dataset)))
    random.shuffle(dataset_indices)
    dataset_len = int(len(dataset) * size)
    dataset_indices = dataset_indices[:dataset_len]
    return torch.utils.data.Subset(dataset, dataset_indices)
