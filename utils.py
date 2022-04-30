from argparse import Namespace
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import List, Literal

import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from tensorflow.python.summary.summary_iterator import summary_iterator
import torch

EPOCH_RE = r"epoch=(\d+)"
VERSION_RE = r"version_(\d+)"


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            trainer.save_checkpoint(
                Path(self.dirpath) / f"epoch={trainer.current_epoch}.ckpt"
            )


def split_augmentations(aug_list: str) -> list:
    """
    In:  "1-2-3+4-5-6"
    Out: [[1,2,3], [4,5,6]]

    In: ''
    Out: [[]]
    """
    return [[int(a) for a in p.split("-") if a] for p in aug_list.split("+")]


def join_augmentations(aug_list: list) -> str:
    """
    In:  [[1,2,3], [4,5,6]]
    Out: "1-2-3+4-5-6"

    In: [[]]
    Out: 'none'
    """
    if joined := "+".join(["-".join([str(a) for a in l]) for l in aug_list]):
        return joined
    return "none"


def model_dir(args) -> str:
    return f"{args.dataset}-{args.model}" + (
        "-" + "-".join(args.defense) if args.defense else ""
    )


def pl_log_root_dir(args) -> str:
    """Returns appropriate root dir for PyTorch Lightning trainer given args"""
    logdir_base = Path("logs")
    return str(
        logdir_base
        / model_dir(args)
        / "training"
        / join_augmentations(args.aug_list)
    )


def find_checkpoint_path(args: Namespace) -> Path:
    """Finds latest checkpoint for dataset-model-auglist combo. Assumes a very
    specific set of arguments and a very specific logging structure."""
    logdir_base = Path("logs")
    versions_dir = (
        logdir_base
        / model_dir(args)
        / "training"
        / join_augmentations(args.aug_list)
        / "lightning_logs"
    )
    versions = versions_dir.iterdir()
    latest_version = max(
        versions, key=lambda p: int(re.search(VERSION_RE, p.name)[1])
    )
    checkpoints_dir = latest_version / "checkpoints"
    checkpoints = checkpoints_dir.iterdir()
    latest_checkpoint = max(
        checkpoints, key=lambda p: int(re.search(EPOCH_RE, p.name)[1])
    )
    return latest_checkpoint


def log(
    args: Namespace,
    kind: Literal["attacks", "training", "augmentations"],
    info: dict,
):
    logdir_base = Path("logs")
    if kind == "attacks":
        # E.g. logs/cifar100-resnet20/attacks/1-2-3/1-zhu.json
        logdir = (
            logdir_base
            / model_dir(args)
            / kind
            / join_augmentations(args.aug_list)
        )
        logdir.mkdir(parents=True, exist_ok=True)
        log_filename = str(args.image_index) + "-" + args.optimizer + ".pt"
        logpath = logdir / log_filename
        torch.save(info, logpath)
        print(f"Logged to {logpath}")
    else:
        raise NotImplementedError(f"Logging for {kind} is not implemented yet.")


"""Utility functions used in the notebook."""


def load_attack_log(
    dataset: str,
    model: str,
    policy: List[List[int]],
    img_idx: int,
    optimizer: str,
    defense: str = "",
):
    logdir_base = Path("logs")
    defense_str = "-" + defense if defense else ""
    try:
        return torch.load(
            logdir_base
            / f"{dataset}-{model}{defense_str}"
            / "attacks"
            / join_augmentations(policy)
            / f"{str(img_idx)}-{optimizer}.pt",
            map_location="cpu",
        )
    except FileNotFoundError:
        return torch.load(
            logdir_base
            / f"{dataset}-{model}{defense_str}"
            / "attacks"
            / join_augmentations(reversed(policy))
            / f"{str(img_idx)}-{optimizer}.pt",
            map_location="cpu",
        )


def load_stats_from_tfevents(path: str):
    stats = defaultdict(lambda: defaultdict(list))
    changed_epoch = False
    epoch = 0
    loss_train = []
    for summary in summary_iterator(str(path)):
        for v in summary.summary.value:
            metric = (
                "acc" if "acc" in v.tag else "loss" if "loss" in v.tag else None
            )
            etap = (
                "train"
                if "train" in v.tag
                else "val"
                if "val" in v.tag
                else None
            )
            if "epoch" in v.tag and epoch != v.simple_value:
                # For train loss calculation
                epoch = v.simple_value
                changed_epoch = True
            if etap is None or metric is None:
                # This is not important log
                continue
            if etap == "train" and metric == "loss":
                # Train loss is logged more than once in epoch -> use the mean
                loss_train.append(v.simple_value)
                if changed_epoch:
                    changed_epoch = False
                    stats[metric][etap].append(np.mean(loss_train))
                    loss_train = []
            else:
                stats[metric][etap].append(v.simple_value)
    return stats


def read_psnr(
    dataset: str,
    model: str,
    policy: List[List[int]],
    img_idx: int,
    optimizer: str,
    defense: str = "",
):
    log = load_attack_log(dataset, model, policy, img_idx, optimizer, defense)
    return log["psnr"]


def read_acc(
    dataset: str, model: str, aug_list: List[List[int]], defense: str = ""
):
    logdir = Path("logs")
    defense_str = "-" + defense if defense else ""
    try:
        events_filepath = list(
            logdir.glob(
                f"{dataset}-{model}{defense_str}/training/{join_augmentations(aug_list)}/lightning_logs/*/*tfevents*"
            )
        )[0]
    except:
        events_filepath = list(
            logdir.glob(
                f"{dataset}-{model}{defense_str}/training/{join_augmentations(reversed(aug_list))}/lightning_logs/*/*tfevents*"
            )
        )[0]
    stats = load_stats_from_tfevents(str(events_filepath))
    return stats["acc"]["val"][-1] * 100


def read_grad_sims(
    dataset: str, model: str, policy: List[int], image_idxs: List[int]
):
    logdir_base = Path("logs")
    augmentations_logdir = logdir_base / f"{dataset}-{model}" / "augmentations"
    if policy is None:
        fp = augmentations_logdir / "none.json"
    else:
        fp = augmentations_logdir / f"{'-'.join(map(str, policy))}.json"
    with open(fp) as f:
        results = json.load(f)
    try:
        return [results["gradsim"][idx] for idx in image_idxs]
    except KeyError:
        return [results["grad_sim"][idx] for idx in image_idxs]


def read_all_spris(dataset: str, model: str):
    logdir_base = Path("logs")
    augmentations_logdir = logdir_base / f"{dataset}-{model}" / "augmentations"
    policy_paths = list(augmentations_logdir.glob("*[0-9].json"))
    all_policies = [list(map(int, p.stem.split("-"))) for p in policy_paths]
    s_pris = []
    for policy, policy_path in zip(all_policies, policy_paths):
        with open(policy_path) as f:
            policy_results = json.load(f)
        s_pris.append(policy_results["S_pri"])
    num_images = len(s_pris[0])
    s_pris = np.mean(s_pris, axis=1)
    return s_pris, all_policies, num_images
