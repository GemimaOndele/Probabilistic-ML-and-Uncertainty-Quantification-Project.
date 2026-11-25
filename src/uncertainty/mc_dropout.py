"""Inférence Monte Carlo Dropout et mesures d'incertitude épistémique.

Missions GitHub : #14 (MC Dropout) et #15 (quantification de l'incertitude).

Les probabilités produites ont la forme ``[T, N, K]`` conformément au contrat
technique ; elles peuvent être consommées telles quelles par le module
d'évaluation de la branche ``feature/evaluation-robustness``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from src.models.cnn import mc_dropout_mode

EPS = np.float32(1e-12)


def entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Entropie de Shannon ``-sum p log p`` le long de ``axis`` (en nats).

    Les probabilités sont tronquées à ``[EPS, 1]`` pour éviter ``0 log 0`` ;
    l'entropie d'une distribution uniforme sur K classes vaut ``log K``.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def predict_mc(
    model: nn.Module,
    inputs: Tensor,
    num_samples: int = 30,
    batch_size: int = 256,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Probabilités MC Dropout de forme ``[T, N, K]``.

    Le modèle est passé en ``model.eval()`` et seuls les dropout sont
    réactivés : BatchNorm utilise ses statistiques courantes (pas de bruit
    de batch parasite). Aucun gradient n'est calculé.
    """
    if num_samples < 1:
        raise ValueError("num_samples doit etre >= 1")
    device = device or next(model.parameters()).device
    model.to(device)
    x = inputs.to(device)
    n = x.shape[0]
    # Le nombre de classes est déduit de la première passe (sortie logits).
    with torch.no_grad(), mc_dropout_mode(model):
        num_classes = model(x[:1]).shape[-1]
        probs = np.empty((num_samples, n, num_classes), dtype=np.float32)
        for t in range(num_samples):
            for start in range(0, n, batch_size):
                batch = x[start : start + batch_size]
                logits = model(batch)
                p = torch.softmax(logits, dim=-1).cpu().numpy()
                probs[t, start : start + len(batch)] = p
    return probs


def summarize_predictions(probs_mc: np.ndarray, labels: np.ndarray | None = None) -> dict:
    """Statistiques de la distribution prédictive MC.

    Args:
        probs_mc: probabilités ``[T, N, K]`` issues de :func:`predict_mc`.
        labels: étiquettes optionnelles ``[N]`` pour accuracy et NLL.

    Returns:
        Dictionnaire de tableaux numpy et de métriques scalaires.
    """
    p = np.asarray(probs_mc, dtype=np.float32)
    if p.ndim != 3:
        raise ValueError(f"probs_mc doit etre [T, N, K], recu {p.shape}")
    t, n, k = p.shape
    if t < 1 or n < 1 or k < 2:
        raise ValueError(f"dimensions invalides : {p.shape}")

    p_mean = p.mean(axis=0)
    predictive_entropy = entropy(p_mean)
    expected_entropy = entropy(p).mean(axis=0)
    mutual_information = np.maximum(predictive_entropy - expected_entropy, 0.0)

    result: dict = {
        "num_samples": t,
        "num_examples": n,
        "num_classes": k,
        "probs_mc": p,
        "probs_mean": p_mean,
        "predicted_labels": p_mean.argmax(axis=1),
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
    }
    if labels is not None:
        y = np.asarray(labels, dtype=np.int64)
        if y.shape != (n,):
            raise ValueError(f"labels doit etre [{n}], recu {y.shape}")
        result["accuracy"] = float((result["predicted_labels"] == y).mean())
        result["nll"] = float(-np.log(np.clip(p_mean[np.arange(n), y], EPS, 1.0)).mean())
    return result
