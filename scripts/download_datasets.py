"""Télécharge les datasets du sujet 5 et écrit un manifeste reproductible.

Utilisation :
    python scripts/download_datasets.py

Les images sont stockées dans ``data/raw/`` (ignoré par Git). Le manifeste
``results/data_manifest.json`` contient les métadonnées vérifiables (taille,
sha256 des archives, effectifs, classes) sans dupliquer les données brutes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from torchvision import datasets


DATASETS = {
    "fashion-mnist": {
        "factory": datasets.FashionMNIST,
        "source": "https://github.com/zalandoresearch/fashion-mnist",
        "license": "MIT",
        "citation": "Xiao, Rasul & Vollgraf, arXiv:1708.07747 (2017)",
        "archive_dir": "FashionMNIST/raw",
        "archive_globs": ("*.gz",),
    },
    "cifar-10": {
        "factory": datasets.CIFAR10,
        "source": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "license": "MIT-like (voir page officielle)",
        "citation": "Krizhevsky, Learning Multiple Layers of Features from Tiny Images (2009)",
        "archive_dir": ".",
        "archive_globs": ("cifar-10-python.tar.gz",),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_info(root: Path, globs: tuple[str, ...]) -> list[dict]:
    found: dict[str, dict] = {}
    for pattern in globs:
        for path in sorted(root.rglob(pattern)):
            if path.is_file():
                found[str(path.relative_to(root).as_posix())] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
    return [{"path": name, **info} for name, info in sorted(found.items())]


def describe(name: str, root: Path) -> dict:
    spec = DATASETS[name]
    train = spec["factory"](root=str(root), train=True, download=True)
    test = spec["factory"](root=str(root), train=False, download=True)
    image, label = train[0]
    return {
        "name": name,
        "source": spec["source"],
        "license": spec["license"],
        "citation": spec["citation"],
        "train_size": len(train),
        "test_size": len(test),
        "classes": list(train.classes),
        "image_shape": list(np.asarray(image).shape),
        "first_train_label": train.classes[label],
        "archives": archive_info(root / spec["archive_dir"], spec["archive_globs"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("results/data_manifest.json"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
    )
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    manifest = {name: describe(name, args.root) for name in args.datasets}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for name, info in manifest.items():
        print(
            f"{name}: train={info['train_size']}, test={info['test_size']}, "
            f"shape={tuple(info['image_shape'])}, archives={len(info['archives'])}"
        )
    print(f"Manifeste écrit dans {args.manifest}")


if __name__ == "__main__":
    main()
