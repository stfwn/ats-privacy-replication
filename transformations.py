from __future__ import annotations
from functools import cached_property
import random

from pytorch_lightning import LightningModule, LightningDataModule
from torch.utils.data import DataLoader

import metrics


EMPTY_TRANSFORMATION = -1


class TransformationLibrary:
    def __init__(self, transformations: list[Transformation]):
        self.transformations = transformations

        # Assign an id to each transformation that we can track them with.
        for i, t in enumerate(self.transformations):
            t.id = i

    def random_policy(self, max_transformations: int = 3) -> Policy:
        transformations = []
        for _ in range(max_transformations):
            if t := self.random_transformation():
                transformations.append(t)
        return Policy(transformations)

    def random_transformation(self):
        i = random.randint(EMPTY_TRANSFORMATION, len(self.transformations))
        if i != EMPTY_TRANSFORMATION:
            return self.transformations[i]
        return None

    def compute_top_policies(
        self,
        n: int,
        accuracy_threshold: float,
        untrained_model: LightningModule,
        partially_trained_model: LightningModule,
        dataloader: DataLoader,
        max_transformations_per_policy: int = 3,
        logdir: str = "logs",
    ) -> list[Policy]:
        """Implements algorithm 1.

        Parameter correspondence to paper:
            self:                       P
            accuracy_threshold:         T_acc
            untrained_model:            M^r
            partially_trained_model:    M^s
            dataloader:                 D

        Returns:
            Optimal policy set containing the best `top_n` policies.
        """
        if len(self.top_policies) == n:
            return self.top_policies

        try_at_least = 1500  # Paper: C_max
        top_policies: list[Policy] = []
        i = 0
        while i < try_at_least or len(top_policies) < n:
            policy = self.random_policy(
                max_transformations=max_transformations_per_policy
            )
            s_acc = policy.compute_s_acc(untrained_model, dataloader)
            s_pri = policy.compute_s_pri(partially_trained_model, dataloader)

            if s_acc > accuracy_threshold:
                if len(top_policies) < n:
                    top_policies.append(policy)
                    i += 1
                    continue
                top_policies = sorted(top_policies, key=lambda p: p.s_pri)
                if s_pri < top_policies[-1].s_pri:
                    top_policies[-1] = policy
            i += 1
        self.top_policies: list[Policy] = top_policies
        return top_policies


class Policy:
    def __init__(self, transformations: list[Transformation]):
        self.transformations = transformations

    def compute_s_acc(self, model: LightningModule, dataloader: DataLoader):
        # TODO
        raise NotImplementedError
        # Apply self.transformations per batch in loop over dataloader
        self.s_acc: float = ...
        return self.s_acc

    def compute_s_pri(self, model: LightningModule, dataloader: DataLoader):
        # TODO
        raise NotImplementedError
        # Apply self.transformations per batch in loop over dataloader
        self.s_pri: float = ...
        return self.s_pri


class Transformation:
    id: int
    # TODO
