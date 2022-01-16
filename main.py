from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import models
import data_modules

pl.seed_everything(23333, workers=True)


def main(args):
    data_module = {
        "fmnist": data_modules.FashionMNISTDataModule,
        "cifar100": data_modules.CIFAR100DataModule,
    }[args.dataset](
        data_dir=args.data_dir, num_workers=12, batch_size=args.batch_size
    )

    model = {"resnet20": models.ResNet20}[args.model](
        num_channels=data_module.num_channels,
        num_classes=data_module.num_classes,
        epochs=args.epochs,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=1,
        deterministic=True,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(every_n_epochs=10),
        ],
    )
    trainer.fit(model, data_module)

    # Evaluate on tasks
    #  trainer.test(model, data_module)
    ...


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-b", "--batch_size", default=128)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("-e", "--epochs", type=int, default=200)
    parser.add_argument("-m", "--model", required=True, choices=["resnet20"])
    parser.add_argument(
        "-d", "--dataset", choices=["fmnist", "cifar100"], required=True
    )
    args = parser.parse_args()
    main(args)
