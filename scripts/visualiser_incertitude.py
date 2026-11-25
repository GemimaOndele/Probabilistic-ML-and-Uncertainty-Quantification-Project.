"""Visualise les prédictions incertaines du CNN MC Dropout.

Mission GitHub : #16 — [P1] Visualisation des prédictions incertaines.
Produit une planche d'exemples corrects/erronés triés par information
mutuelle et un histogramme comparatif MI(correct) vs MI(erroné).

Exemple :
    python scripts/visualiser_incertitude.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

from src.uncertainty.mc_dropout import entropy
from src.utils import write_json


def select_indices(
    labels: np.ndarray,
    predicted: np.ndarray,
    mutual_information: np.ndarray,
    count: int,
) -> dict[str, list[int]]:
    """Sélectionne des exemples représentatifs pour la planche."""
    correct = np.where(predicted == labels)[0]
    incorrect = np.where(predicted != labels)[0]

    def top(indices: np.ndarray, highest: bool = True) -> list[int]:
        if len(indices) == 0:
            return []
        order = np.argsort(mutual_information[indices])
        if highest:
            order = order[::-1]
        return indices[order[:count]].tolist()

    return {
        "incorrect_high_mi": top(incorrect, highest=True),
        "correct_high_mi": top(correct, highest=True),
        "incorrect_low_mi": top(incorrect, highest=False),
    }


def plot_examples(
    raw_dataset,
    classes: list[str],
    selections: dict[str, list[int]],
    probs_mean: np.ndarray,
    predictive_entropy: np.ndarray,
    mutual_information: np.ndarray,
    labels: np.ndarray,
    predicted: np.ndarray,
    output: Path,
) -> None:
    titles = {
        "incorrect_high_mi": "Erreurs a forte incertitude epistémique",
        "correct_high_mi": "Bonnes prédictions a forte incertitude",
        "incorrect_low_mi": "Erreurs a faible incertitude (confusion)",
    }
    rows = [(name, indices) for name, indices in selections.items() if indices]
    n_cols = max(len(indices) for _, indices in rows)
    fig, axes = plt.subplots(
        len(rows), n_cols, figsize=(2.1 * n_cols, 2.5 * len(rows))
    )
    axes = np.atleast_2d(axes)
    for row, (name, indices) in enumerate(rows):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(indices):
                continue
            idx = indices[col]
            image = np.asarray(raw_dataset[idx][0])
            ax.imshow(image, cmap="gray")
            ax.set_title(
                f"{titles[name]}\n"
                f"vrai={classes[labels[idx]]} pred={classes[predicted[idx]]}\n"
                f"P={probs_mean[idx, predicted[idx]]:.2f} "
                f"H={predictive_entropy[idx]:.2f} MI={mutual_information[idx]:.2f}",
                fontsize=7,
            )
    fig.suptitle("Exemples Fashion-MNIST — MC Dropout (T=30)", fontsize=10)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_histogram(
    mutual_information: np.ndarray,
    correct: np.ndarray,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, float(np.percentile(mutual_information, 99.5)), 50)
    ax.hist(mutual_information[correct], bins=bins, alpha=0.7, label="correct")
    ax.hist(mutual_information[~correct], bins=bins, alpha=0.7, label="erroné")
    ax.set_xlabel("Information mutuelle (nats)")
    ax.set_ylabel("Nombre d'exemples")
    ax.set_title("Incertitude épistémique : prédictions correctes vs erronées")
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    data = np.load(args.probs)
    probs_mc = data["probs_mc"]
    labels = data["labels"]
    probs_mean = probs_mc.mean(axis=0)
    predicted = probs_mean.argmax(axis=1)
    predictive_entropy = entropy(probs_mean)
    expected_entropy = entropy(probs_mc).mean(axis=0)
    mutual_information = np.maximum(predictive_entropy - expected_entropy, 0.0)

    raw_dataset = datasets.FashionMNIST(
        root=args.root, train=False, download=False, transform=None
    )
    selections = select_indices(labels, predicted, mutual_information, args.count)
    plot_examples(
        raw_dataset,
        raw_dataset.classes,
        selections,
        probs_mean,
        predictive_entropy,
        mutual_information,
        labels,
        predicted,
        args.figure,
    )
    correct = predicted == labels
    plot_histogram(mutual_information, correct, args.histogram)

    report = {
        "dataset": "fashion-mnist",
        "split": "test",
        "num_examples": int(len(labels)),
        "num_samples_T": int(probs_mc.shape[0]),
        "source_probs": str(args.probs),
        "mean_mi_correct": float(mutual_information[correct].mean()),
        "mean_mi_incorrect": float(mutual_information[~correct].mean()),
        "selections": {
            name: [
                {
                    "index": int(idx),
                    "true_label": raw_dataset.classes[int(labels[idx])],
                    "predicted_label": raw_dataset.classes[int(predicted[idx])],
                    "confidence": float(probs_mean[idx, predicted[idx]]),
                    "predictive_entropy": float(predictive_entropy[idx]),
                    "mutual_information": float(mutual_information[idx]),
                }
                for idx in indices
            ]
            for name, indices in selections.items()
        },
        "figures": [str(args.figure), str(args.histogram)],
    }
    write_json(args.report, report)
    print(f"Figures: {args.figure}, {args.histogram}")
    print(f"Rapport: {args.report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probs", default="results/probs_mc_fashion-mnist_test.npz")
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--figure", type=Path, default=Path("results/figures/exemples_incertitude.png"))
    parser.add_argument(
        "--histogram", type=Path, default=Path("results/figures/histogramme_mi.png")
    )
    parser.add_argument(
        "--report", default="results/visualisation_incertitude_fashion-mnist.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
