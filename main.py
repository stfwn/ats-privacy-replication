# Stdlib
from argparse import ArgumentParser
import sys

# Third party
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl

from original.searchalg.search_best import find_best

sys.path.insert(0, "./original/")

# First party
import data_modules
import models
from policy_search import parallel_policy_search
import utils

pl.seed_everything(42, workers=True)


def main(args):
    if args.command == "train":
        train(args)
    if args.command == "search":
        search(args)
    else:
        print(args.command)
        raise ValueError


def train(args):
    data_module = {
        "fmnist": data_modules.FashionMNISTDataModule,
        "cifar100": data_modules.CIFAR100DataModule,
    }[args.dataset](
        data_dir=args.data_dir,
        policy_list=args.aug_list,
        num_workers=12,
        batch_size=args.batch_size,
    )

    model = {"resnet20": models.ResNet20, "convnet": models.ConvNet}[
        args.model
    ](
        num_channels=data_module.num_channels,
        num_classes=data_module.num_classes,
        **vars(args),  # pass all args just to log them
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            utils.PeriodicCheckpoint(every_n_epochs=10),
        ],
        devices="auto",
        accelerator="auto",
        default_root_dir=f"logs/{args.dataset}-{args.model}/training/{args.aug_list}/",
    )

    trainer.fit(model, data_module)


def search(args):
    parallel_policy_search(
        num_schemes=args.num_schemes,
        model=args.model,
        data=args.dataset,
        epochs=args.epochs,
        model_checkpoint=args.model_checkpoint,
        num_transform=args.num_transform,
        num_per_gpu=args.num_per_gpu,
        num_images=args.num_images,
        schemes=args.aug_list,
    )
    find_best(
        dataset_name=args.dataset,
        model_name=args.model,
        thresh_acc=args.thresh_acc,
        n=args.n,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    commands = parser.add_subparsers(
        title="command", help="Action to execute", dest="command"
    )

    # Policy search arguments
    search_parser = commands.add_parser(
        "search", help="Automatic transformation search"
    )
    search_parser.add_argument(
        "-m", "--model", choices=["ResNet20-4"], required=True
    )
    search_parser.add_argument(
        "-d", "--dataset", choices=["fmnist", "cifar100"], required=True
    )
    search_parser.add_argument("-n", "--num_schemes", type=int, default=1600)
    search_parser.add_argument("-g", "--num_per_gpu", type=int, default=20)
    search_parser.add_argument("-e", "--epochs", type=int, default=100)
    search_parser.add_argument("-t", "--num_transform", type=int, default=3)
    search_parser.add_argument("--model_checkpoint", type=str, required=True)
    search_parser.add_argument(
        "--num_images", default=1, type=int, help="Number of images."
    )
    search_parser.add_argument(
        "--thresh-acc",
        default=-85,
        required=False,
        type=int,
        help="Accuracy Score Threshold",
    )
    search_parser.add_argument(
        "--n",
        default=10,
        required=False,
        type=int,
        help="Maximum number of policies",
    )
    search_parser.add_argument(
        "--aug_list", default=None, type=utils.split_augmentations
    )

    # Model training arguments
    train_parser = commands.add_parser("train", help="Model training")
    train_parser.add_argument("-b", "--batch_size", default=128)
    train_parser.add_argument("--data-dir", default="./data")
    train_parser.add_argument("-e", "--epochs", type=int, default=200)
    train_parser.add_argument(
        "-m", "--model", required=True, choices=["convnet", "resnet20"]
    )
    train_parser.add_argument(
        "-d", "--dataset", choices=["cifar100", "fmnist"], required=True
    )
    train_parser.add_argument(
        "--aug_list", default="", type=utils.split_augmentations
    )
    train_parser.add_argument("--bugged-loss", action="store_true")

    args = parser.parse_args()
    main(args)
