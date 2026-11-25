"""Évalue un Deep Ensembles avec dropout, à budget de calcul égal au MC Dropout.

Mission GitHub : #17 — [P2] Implémentation et comparaison des Deep Ensembles.
Chaque membre (seed indépendante, split identique) fournit ``T`` passes MC ;
la décomposition utiliser les ``M × T`` passes au total, exactement comme le
MC Dropout simple avec ``T=30``. L'information mutuelle des deux méthodes est
ainsi estimée par le même estimateur et peut être comparée directement.

Exemple (3 membres × 10 passes = 30 passes, comme MC Dropout T=30) :
    python scripts/run_ensembles.py --mc-samples 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.splits import make_splits
from src.models.cnn import build_model
from src.uncertainty.decomposition import decompose_uncertainty
from src.uncertainty.mc_dropout import predict_mc
from src.utils import Timer, seed_everything, write_json


def load_member(path: str, device: torch.device) -> tuple[torch.nn.Module, dict]:
    torch.serialization.add_safe_globals([type(torch.__version__)])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = build_model(config["dataset"], dropout_p=config["dropout_p"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members, configs, training_seconds = [], [], []
    for path in args.checkpoints:
        model, config = load_member(path, device)
        members.append(model)
        configs.append(config)
        history_path = Path(
            path.replace("checkpoints/baseline_", "train_baseline_").replace(".pt", ".json")
        )
        if history_path.exists():
            training_seconds.append(
                json.loads(history_path.read_text(encoding="utf-8"))["train_seconds"]
            )

    seeds = [config["seed"] for config in configs]
    split_seeds = {config.get("split_seed", config["seed"]) for config in configs}
    if len(split_seeds) != 1:
        raise ValueError(f"Les membres n'utilisent pas le meme split : {split_seeds}")
    base_config = configs[0]
    splits = make_splits(
        base_config["dataset"],
        root=args.root,
        seed=split_seeds.pop(),
        val_size=base_config["val_size"],
        calib_size=base_config["calib_size"],
    )
    dataset = splits[args.split]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    images = torch.cat([batch_images for batch_images, _ in loader], dim=0)
    labels = np.concatenate([batch_labels.numpy() for _, batch_labels in loader])

    timer = Timer()
    member_probs = [
        predict_mc(
            model,
            images,
            num_samples=args.mc_samples,
            batch_size=args.batch_size,
            device=device,
        )
        for model in members
    ]
    inference_seconds = timer.elapsed()

    probs_all = np.stack(member_probs)  # [M, T, N, K]
    decomposition = decompose_uncertainty(probs_all)
    p_mean = decomposition["probs_mean"]
    predicted = decomposition["predicted_labels"]
    correct = predicted == labels
    mutual_information = decomposition["mutual_information"]
    total_passes = decomposition["total_passes"]

    result = {
        "method": "deep_ensemble_mc_dropout",
        "dataset": base_config["dataset"],
        "split": args.split,
        "num_examples": int(len(labels)),
        "num_members": len(members),
        "member_seeds": seeds,
        "split_seed": base_config.get("split_seed", base_config["seed"]),
        "mc_samples_per_member": args.mc_samples,
        "total_stochastic_passes": total_passes,
        "device": str(device),
        "accuracy": float(correct.mean()),
        "nll": float(
            -np.log(np.clip(p_mean[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()
        ),
        "mean_predictive_entropy": float(decomposition["predictive_entropy"].mean()),
        "mean_expected_entropy": float(decomposition["expected_entropy"].mean()),
        "mean_mutual_information": float(mutual_information.mean()),
        "mean_mi_correct": float(mutual_information[correct].mean()),
        "mean_mi_incorrect": float(mutual_information[~correct].mean()),
        "mean_between_member_mi": float(decomposition["between_member_mi"].mean()),
        "training_seconds_per_member": training_seconds,
        "training_seconds_total": float(sum(training_seconds)),
        "inference_seconds_ensemble": inference_seconds,
    }

    mc_path = Path("results/mc_dropout_fashion-mnist_test.json")
    if mc_path.exists():
        mc = json.loads(mc_path.read_text(encoding="utf-8"))
        result["comparaison_mc_dropout"] = {
            "budget_egal_passes": total_passes == mc["num_samples_T"],
            "mc_total_passes": mc["num_samples_T"],
            "accuracy_mc": mc["accuracy_mc"],
            "nll_mc": mc["nll_mc"],
            "mean_mutual_information_mc": mc["mean_mutual_information"],
            "inference_seconds_mc": mc["inference_seconds_mc"],
        }

    write_json(args.output, result)
    if args.probs:
        Path(args.probs).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.probs, probs_mc_per_member=probs_all, labels=labels
        )
    print(
        f"Deep Ensemble M={len(members)} x T={args.mc_samples} "
        f"({total_passes} passes) | accuracy={result['accuracy']:.4f} "
        f"| NLL={result['nll']:.4f} | MI={result['mean_mutual_information']:.5f} "
        f"| train={result['training_seconds_total']:.1f}s "
        f"| inference={inference_seconds:.1f}s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "results/checkpoints/baseline_fashion-mnist_seed42.pt",
            "results/checkpoints/baseline_fashion-mnist_seed43.pt",
            "results/checkpoints/baseline_fashion-mnist_seed44.pt",
        ],
    )
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--split", default="test", choices=["test", "calibration", "validation"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/deep_ensembles_fashion-mnist.json")
    parser.add_argument(
        "--probs", default="results/probs_ensemble_fashion-mnist_test.npz"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
