"""Splits déterministes train / validation / calibration / test.

Helper local provisoire : la branche ``feature/data-baseline`` fournira les
chargeurs officiels ; ce module permet à la branche probabilistic-models
d'être exécutable de bout en bout de façon indépendante et reproductible.
Les indices sont sauvegardés pour garantir le même protocole à toutes les
méthodes (CNN classique, MC Dropout, Deep Ensembles).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

STATS = {
    "fashion-mnist": {"mean": (0.2860,), "std": (0.3530,)},
    "cifar-10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
}


def make_splits(
    dataset: str,
    root: str = "data/raw",
    seed: int = 42,
    val_size: int = 5000,
    calib_size: int = 5000,
    indices_path: str | Path | None = None,
) -> dict[str, Dataset]:
    """Construit quatre splits disjoints à partir du split train officiel."""
    if dataset not in STATS:
        raise ValueError(f"dataset inconnu : {dataset!r}")
    stats = STATS[dataset]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ]
    )
    train_full = datasets.__dict__[
        "FashionMNIST" if dataset == "fashion-mnist" else "CIFAR10"
    ](root=root, train=True, download=False, transform=transform)
    test = datasets.__dict__[
        "FashionMNIST" if dataset == "fashion-mnist" else "CIFAR10"
    ](root=root, train=False, download=False, transform=transform)

    all_indices = np.arange(len(train_full))
    np.random.default_rng(seed).shuffle(all_indices)
    calibration = all_indices[:calib_size]
    validation = all_indices[calib_size : calib_size + val_size]
    train = all_indices[calib_size + val_size :]

    if indices_path is not None:
        path = Path(indices_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "sizes": {
                        "train": len(train),
                        "validation": len(validation),
                        "calibration": len(calibration),
                        "test": len(test),
                    },
                    "indices": {
                        "train": train.tolist(),
                        "validation": validation.tolist(),
                        "calibration": calibration.tolist(),
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "train": Subset(train_full, train.tolist()),
        "validation": Subset(train_full, validation.tolist()),
        "calibration": Subset(train_full, calibration.tolist()),
        "test": test,
    }
