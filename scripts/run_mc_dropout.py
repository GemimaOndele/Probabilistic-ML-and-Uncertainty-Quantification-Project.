"""Inférence MC Dropout à partir du checkpoint baseline + incertitude épistémique.

Missions GitHub : #14 (dropout actif à l'inférence, BatchNorm en eval) et
#15 (probabilités MC, entropie prédictive, entropie attendue, information
mutuelle). Les probabilités ``[T, N, K]`` sont exportées pour la branche
d'évaluation (Omar) et un résumé JSON est produit (aucun résultat inventé).

Exemple :
    python scripts/run_mc_dropout.py --num-samples 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.splits import make_splits
from src.models.cnn import build_model
from src.uncertainty.mc_dropout import predict_mc, summarize_predictions
from src.utils import Timer, seed_everything, write_json


def deterministic_predictions(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Softmax déterministe (dropout désactivé) : référence CNN classique."""
    model.eval()
    probs_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs_list.append(torch.softmax(logits, dim=-1).cpu().numpy())
            labels_list.append(labels.numpy())
    probs = np.concatenate(probs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return probs, labels, probs.argmax(axis=1)


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.__version__ est un TorchVersion (sous-classe de str) dans la config
    # des checkpoints existants ; on autorise ce type pour weights_only=True.
    torch.serialization.add_safe_globals([type(torch.__version__)])
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    config = checkpoint["config"]
    model = build_model(config["dataset"], dropout_p=config["dropout_p"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    splits = make_splits(
        config["dataset"],
        root=args.root,
        seed=config.get("split_seed", config["seed"]),
        val_size=config["val_size"],
        calib_size=config["calib_size"],
    )
    dataset = splits[args.split]
    if args.limit is not None:
        dataset = torch.utils.data.Subset(
            dataset, list(range(min(args.limit, len(dataset))))
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    timer = Timer()
    probs_mc = predict_mc(
        model,
        torch.cat([images for images, _ in loader], dim=0),
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=device,
    )
    mc_seconds = timer.elapsed()

    deterministic_probs, labels, deterministic_pred = deterministic_predictions(
        model, loader, device
    )
    summary = summarize_predictions(probs_mc, labels)
    correct = summary["predicted_labels"] == labels
    plugin_nll = float(
        -np.log(
            np.clip(deterministic_probs[np.arange(len(labels)), labels], 1e-12, 1.0)
        ).mean()
    )

    result = {
        "method": "mc_dropout",
        "dataset": config["dataset"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_seed": config["seed"],
        "split": args.split,
        "num_examples": int(len(labels)),
        "num_samples_T": args.num_samples,
        "dropout_p": config["dropout_p"],
        "seed": args.seed,
        "device": str(device),
        "accuracy_deterministic": float((deterministic_pred == labels).mean()),
        "nll_deterministic": plugin_nll,
        "accuracy_mc": summary["accuracy"],
        "nll_mc": summary["nll"],
        "mean_predictive_entropy": float(summary["predictive_entropy"].mean()),
        "mean_expected_entropy": float(summary["expected_entropy"].mean()),
        "mean_mutual_information": float(summary["mutual_information"].mean()),
        "mean_mutual_information_correct": float(
            summary["mutual_information"][correct].mean()
        )
        if correct.any()
        else None,
        "mean_mutual_information_incorrect": float(
            summary["mutual_information"][~correct].mean()
        )
        if (~correct).any()
        else None,
        "inference_seconds_mc": mc_seconds,
    }
    write_json(args.output, result)
    print(
        f"MC Dropout T={args.num_samples} | accuracy={result['accuracy_mc']:.4f} "
        f"| NLL={result['nll_mc']:.4f} | MI moyen={result['mean_mutual_information']:.5f} "
        f"| temps={mc_seconds:.1f}s"
    )

    if args.probs:
        Path(args.probs).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.probs, probs_mc=probs_mc, labels=labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="results/checkpoints/baseline_fashion-mnist_seed42.pt"
    )
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--split", default="test", choices=["test", "calibration", "validation"])
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="results/mc_dropout_fashion-mnist_test.json")
    parser.add_argument("--probs", default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
