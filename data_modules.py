from typing import Any, Callable, Optional, Union

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.datamodules import FashionMNISTDataModule
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR100


class FashionMNISTDataModule(FashionMNISTDataModule):
    @property
    def num_channels(self):
        return self.dims[0]


class CIFAR100DataModule(VisionDataModule):
    name = "cifar_100"
    dataset_cls = CIFAR100
    dims = (3, 32, 32)

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(  # type: ignore[misc]
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )

    @property
    def num_channels(self):
        return self.dims[0]

    @property
    def num_classes(self) -> int:
        return 100

    def default_transforms(self) -> Callable:
        return transforms.Compose([transforms.ToTensor()])
