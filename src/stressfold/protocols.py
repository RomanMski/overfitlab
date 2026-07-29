"""Resampling protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from sklearn.model_selection import train_test_split

from .random import derive_seed


@dataclass(frozen=True, slots=True)
class HoldoutSplit:
    repeat: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    seed: int
    stratified: bool


def repeated_holdout(
    n_samples: int,
    y: np.ndarray,
    *,
    task: str,
    repeats: int,
    test_size: float,
    random_state: int,
) -> Iterator[HoldoutSplit]:
    """Yield deterministic independent shuffle-split replicates."""

    if n_samples < 4:
        raise ValueError("StressFold requires at least four rows")
    all_indices = np.arange(n_samples, dtype=int)
    for repeat in range(repeats):
        seed = derive_seed(random_state, "split", repeat)
        stratify = None
        stratified = False
        if task == "binary_classification":
            _, counts = np.unique(y, return_counts=True)
            # train_test_split gives a clearer error for impossible requested
            # sizes, but only request stratification when both classes can land
            # on both sides of the split.
            expected_test = int(np.ceil(n_samples * test_size))
            expected_train = n_samples - expected_test
            if (
                counts.min() < 2
                or expected_test < len(counts)
                or expected_train < len(counts)
            ):
                raise ValueError(
                    "The requested binary holdout cannot place both classes in train and test; "
                    "provide more minority-class rows or adjust test_size"
                )
            stratify = y
            stratified = True
        train_indices, test_indices = train_test_split(
            all_indices,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
        yield HoldoutSplit(
            repeat=repeat,
            train_indices=np.sort(train_indices),
            test_indices=np.sort(test_indices),
            seed=seed,
            stratified=stratified,
        )
