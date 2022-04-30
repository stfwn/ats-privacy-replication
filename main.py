import torch
import pytorch_lightning as pl

import attacks
import data_modules
import models
import utils
import policy_search

from argparse import ArgumentParser
from pytorch_lightning.callbacks import LearningRateMonitor

pl.seed_everything(42, workers=True)


def main(args):
    if args.command == "attack":
        attack(args)
    elif args.command == "search":
        search(args)
    elif args.command == "test":
        test(args)
    elif args.command == "train":
        train(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


def attack(args):
    checkpoint_path = (
        args.checkpoint_path
        if args.checkpoint_path is not None
        else utils.find_checkpoint_path(args)
    )
    model = (
        models.get_by_name(args.model)
        .load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            num_classes=100,
            num_channels=3,
            epochs=5,
        )
        .to("cuda" if torch.cuda.is_available() else "cpu")
    )
    data_module = data_modules.get_by_name(args.dataset)(
        data_dir=args.data_dir,
        policy_list=args.aug_list,
        apply_policy_to_test=True,
    )
    (
        original_img,
        reconstruction,
        reconstruction_stats,
        psnr,
        img_mse,
        pred_mse,
    ) = attacks.attack(
        model=model,
        data_module=data_module,
        optimizer=args.optimizer,
        img_idx=args.image_index,
        max_iterations=args.max_iterations,
    )
    utils.log(
        args,
        kind="attacks",
        info={
            "args": vars(args),
            "original_img": original_img,
            "reconstruction": reconstruction,
            "psnr": psnr,
            "img_mse": img_mse,
            "pred_mse": pred_mse,
            "reconstruction_stats": reconstruction_stats,
        },
    )
    print("PSNR:", psnr)


def search(args):
    policy_search.parallel_policy_search(
        num_schemes=args.num_schemes,
        model=args.model,
        data=args.dataset,
        epochs=args.epochs,
        model_checkpoint=args.checkpoint_path,
        num_transform=args.num_transform,
        num_per_gpu=args.num_per_gpu,
        num_images=args.num_images,
        schemes=args.aug_list,
        data_dir=args.data_dir,
    )
    policy_search.find_best(
        dataset_name=args.dataset,
        model_name=args.model,
        thresh_acc=args.thresh_acc,
        n=args.n,
    )


def test(args):
    """Evaluate trained models on any dataset-policy combination."""
    model = models.get_by_name(args.model).load_from_checkpoint(
        checkpoint_path=(
            # Use a particular checkpoint for this model
            args.checkpoint_path
            if args.checkpoint_path
            # Or use the latest one for the given model-dataset-policy combination
            else utils.find_checkpoint_path(args)
        )
    )
    data_module = data_modules.get_by_name(args.dataset)(
        data_dir=args.data_dir,
        policy_list=args.aug_list,
        batch_size=args.batch_size,
    )
    trainer = pl.Trainer(
        deterministic=True,
        devices="auto",
        accelerator="auto",
    )
    trainer.test(model, data_module)


def train(args):
    if args.epochs is None:
        args.epochs = 200 if args.dataset == "cifar100" else 100

    data_module = data_modules.get_by_name(args.dataset)(
        data_dir=args.data_dir,
        policy_list=args.aug_list,
        batch_size=args.batch_size,
        tiny_subset_size=args.tiny_subset_size,
    )
    model = models.get_by_name(args.model)(
        num_channels=data_module.num_channels,
        num_classes=data_module.num_classes,
        **vars(args),  # pass all args to log them
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
        default_root_dir=utils.pl_log_root_dir(args),
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    dataset_choices = ["cifar100", "fmnist", "tiny-imagenet200"]
    model_choices = ["convnet", "resnet20"]

    parser = ArgumentParser()

    # Shared
    parser.add_argument("--data-dir", type=str, default="./data")

    commands = parser.add_subparsers(
        title="command",
        help="Action to execute",
        dest="command",
    )

    # Search
    search_parser = commands.add_parser(
        "search", help="Automatic transformation search"
    )
    search_parser.add_argument(
        "-m", "--model", choices=model_choices, required=True, type=str
    )
    search_parser.add_argument(
        "-d", "--dataset", choices=dataset_choices, required=True, type=str
    )
    search_parser.add_argument("-n", "--num-schemes", type=int, default=1600)
    search_parser.add_argument("-g", "--num-per-gpu", type=int, default=20)
    search_parser.add_argument("-e", "--epochs", type=int, default=100)
    search_parser.add_argument("-t", "--num-transform", type=int, default=3)
    search_parser.add_argument("--checkpoint-path", type=str, required=True)
    search_parser.add_argument(
        "--num-images", default=1, type=int, help="Number of images."
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
        "--aug-list", default=None, type=utils.split_augmentations
    )

    # Train
    train_parser = commands.add_parser("train", help="Model training")
    train_parser.add_argument("-b", "--batch-size", type=int, default=128)
    train_parser.add_argument("-e", "--epochs", type=int, default=None)
    train_parser.add_argument(
        "-m", "--model", required=True, choices=model_choices, type=str
    )
    train_parser.add_argument(
        "-d", "--dataset", choices=dataset_choices, required=True, type=str
    )
    train_parser.add_argument(
        "--aug-list", default="", type=utils.split_augmentations
    )
    train_parser.add_argument("--bugged-loss", action="store_true")
    train_parser.add_argument("--tiny-subset-size", type=float, default=None)
    train_parser.add_argument(
        "--defense", type=lambda x: x.split("-"), required=False, default=None
    )

    # Attack
    attack_parser = commands.add_parser(
        "attack", help="Perform reconstruction attack"
    )
    attack_parser.add_argument(
        "-m", "--model", required=True, choices=model_choices, type=str
    )
    attack_parser.add_argument(
        "-o",
        "--optimizer",
        default=None,
        required=True,
        choices=[  # See page 5 of paper for this list
            "zhu",  # (1)
            "inversed",  # (2) probably
            "inversed-LBFGS-sim",  # (3)
            "inversed-adam-L1",  # (4)
            "inversed-adam-L2",  # (5)
            "inversed-sgd-sim",  # (6)
            "inversefed-default",  # Additional that was not included in the paper
        ],
        help="Optimizer used by the recontruction machine",
        type=str,
    )
    attack_parser.add_argument(
        "-d", "--dataset", choices=dataset_choices, required=True, type=str
    )
    attack_parser.add_argument(
        "--aug-list", default="", type=utils.split_augmentations
    )
    attack_parser.add_argument("-i", "--image-index", default=0, type=int)
    attack_parser.add_argument(
        "-n",
        "--num-images",
        default=1,
        type=int,
        help="Number of images to be attacked, if bigger than 1 then next images just +1",
    )
    attack_parser.add_argument("--max-iterations", type=int, default=None)
    attack_parser.add_argument("--checkpoint-path", type=str, default=None)
    attack_parser.add_argument(
        "--defense", type=lambda x: x.split("-"), required=False, default=None
    )

    # Test
    test_parser = commands.add_parser("test", help="Test a model")
    test_parser.add_argument("-b", "--batch-size", default=128, type=int)
    test_parser.add_argument("-e", "--epochs", type=int, default=None)
    test_parser.add_argument(
        "-m", "--model", required=True, choices=model_choices, type=str
    )
    test_parser.add_argument(
        "-d", "--dataset", choices=dataset_choices, required=True, type=str
    )
    test_parser.add_argument(
        "--aug-list", default="", type=utils.split_augmentations
    )
    test_parser.add_argument("--checkpoint-path", type=str, default=None)
    test_parser.add_argument(
        "--defense", type=lambda x: x.split("-"), required=False, default=None
    )

    args = parser.parse_args()
    main(args)
