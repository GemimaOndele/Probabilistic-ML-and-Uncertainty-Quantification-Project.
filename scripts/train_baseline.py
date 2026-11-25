"""Entraîne le CNN baseline avec dropout (checkpoint réutilisé par MC Dropout).

Mission GitHub : #14 — [P0] Implémentation du Monte Carlo Dropout (étape 1/2).
Le dropout est actif pendant l'entraînement : sans cela, l'échantillonnage à
l'inférence ne correspondrait pas à la famille variationnelle apprise.

Exemple :
    python scripts/train_baseline.py --dataset fashion-mnist --epochs 3 --seed 42
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.splits import STATS, make_splits
from src.models.cnn import build_model
from src.utils import Timer, seed_everything, write_json


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1)}


def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = make_splits(
        args.dataset,
        root=args.root,
        seed=args.split_seed,
        val_size=args.val_size,
        calib_size=args.calib_size,
        indices_path=args.indices,
    )
    train_set = splits["train"]
    if args.limit_train is not None:
        train_set = torch.utils.data.Subset(
            train_set, list(range(min(args.limit_train, len(train_set))))
        )
    val_set = splits["validation"]
    if args.limit_val is not None:
        val_set = torch.utils.data.Subset(
            val_set, list(range(min(args.limit_val, len(val_set))))
        )

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.dataset, dropout_p=args.dropout_p).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    config = {
        "dataset": args.dataset,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout_p": args.dropout_p,
        "hidden": 128,
        "val_size": args.val_size,
        "calib_size": args.calib_size,
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
        "normalization": STATS[args.dataset],
        "device": str(device),
        "torch_version": str(torch.__version__),
    }
    history: dict = {"config": config, "epochs": []}
    best_acc, best_state = -1.0, None
    timer = Timer()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.numel()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
        train_metrics = {
            "loss": running_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
        }
        val_metrics = evaluate(model, val_loader, device)
        history["epochs"].append(
            {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        )
        print(
            f"epoch {epoch}/{args.epochs} | train {train_metrics['accuracy']:.4f} "
            f"| val {val_metrics['accuracy']:.4f}"
        )
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())

    history["best_validation_accuracy"] = best_acc
    history["train_seconds"] = timer.elapsed()
    history["example_counts"] = {
        "train": len(train_set),
        "validation": len(val_set),
        "calibration": len(splits["calibration"]),
        "test": len(splits["test"]),
    }
    write_json(args.history, history)

    checkpoint = {
        "model_state_dict": best_state if best_state is not None else model.state_dict(),
        "config": config,
        "best_validation_accuracy": best_acc,
    }
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    print(
        f"checkpoint: {args.checkpoint} | meilleure val accuracy: {best_acc:.4f} "
        f"| temps: {history['train_seconds']:.1f}s"
    )
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="fashion-mnist", choices=list(STATS))
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout-p", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--calib-size", type=int, default=5000)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--history", default=None)
    parser.add_argument("--indices", default=None)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint or (
        f"results/checkpoints/baseline_{args.dataset}_seed{args.seed}.pt"
    )
    args.history = args.history or (
        f"results/train_baseline_{args.dataset}_seed{args.seed}.json"
    )
    args.indices = args.indices or (
        f"results/splits/{args.dataset}_split{args.split_seed}.json"
    )
    return args


if __name__ == "__main__":
    train(parse_args())
