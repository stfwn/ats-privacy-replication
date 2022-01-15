from argparse import ArgumentParser

import pytorch_lightning as pl

import models
import data_modules

pl.seed_everything(23333, workers=True)


def main(args):
    # Init model & data
    dm = {"fmnist": data_modules.FashionMNISTDataModule, "cifar100": data_modules.CIFAR100DataModule}[
        args.dataset
    ](
        data_dir=args.data_dir, num_workers=12
    )  # TODO: Add CIFAR100
    model = {"resnet20": models.ResNet20}[args.model](
        num_channels=dm.num_channels, num_classes=dm.num_classes, epochs=args.epochs
    )

    # Add transformations
    ...

    # Train model
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=1, deterministic=True)
    trainer.fit(model, dm)
    trainer.test(model, dm)

    # Evaluate on tasks
    ...


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("-e", "--epochs", type=int, default=200)
    parser.add_argument("-m", "--model", required=True, choices=["resnet20"])
    parser.add_argument("-d", "--dataset", choices=["fmnist", "cifar100"], required=True)
    args = parser.parse_args()
    main(args)
