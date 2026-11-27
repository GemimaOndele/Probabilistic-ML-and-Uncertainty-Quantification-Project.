"""Métriques de performance, de calibration et d'incertitude (Issue #19).

Conventions (cf. ``docs/contrat-technique.md``) :
- ``probs`` : probabilités prédictives ``[N, C]`` (pour MC Dropout, la moyenne des T passes) ;
- ``mc_probs`` : probabilités des T passes stochastiques ``[T, N, C]`` ;
- ``labels`` : étiquettes entières ``[N]``.
Tous les logarithmes sont naturels (nats).
"""
from __future__ import annotations

import numpy as np

from ._utils import EPS, check_mc_probs, check_probs


# --------------------------------------------------------------------------- performance
def accuracy(probs, labels) -> float:
    p, y = check_probs(probs, labels)
    return float(np.mean(p.argmax(axis=1) == y))


def nll(probs, labels) -> float:
    """Log-vraisemblance négative moyenne : -1/N Σ log p(y_i | x_i)."""
    p, y = check_probs(probs, labels)
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + EPS)))


def brier_score(probs, labels) -> float:
    """Score de Brier multiclasse : 1/N Σ_i Σ_c (p_ic - 1[y_i = c])²  (entre 0 et 2)."""
    p, y = check_probs(probs, labels)
    onehot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


# --------------------------------------------------------------------------- calibration
def reliability_bins(probs, labels, n_bins: int = 15) -> dict:
    """Statistiques par intervalle de confiance (bins de largeur égale sur [0, 1]).

    Retourne pour chaque bin : bornes, effectif, confiance moyenne et accuracy.
    Les bins vides ont une confiance et une accuracy ``nan``.
    """
    p, y = check_probs(probs, labels)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bin i = ]edges[i], edges[i+1]] ; une confiance de 0 tombe dans le premier bin
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(np.int64)
    sum_conf = np.bincount(idx, weights=conf, minlength=n_bins)
    sum_acc = np.bincount(idx, weights=correct, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.where(counts > 0, sum_conf / counts, np.nan)
        mean_acc = np.where(counts > 0, sum_acc / counts, np.nan)
    return {
        "edges": edges,
        "counts": counts,
        "confidence": mean_conf,
        "accuracy": mean_acc,
        "n": int(len(y)),
    }


def expected_calibration_error(probs, labels, n_bins: int = 15) -> float:
    """ECE (Guo et al., 2017) : Σ_b |B_b|/N · |acc(B_b) - conf(B_b)|."""
    b = reliability_bins(probs, labels, n_bins)
    mask = b["counts"] > 0
    gaps = np.abs(b["accuracy"][mask] - b["confidence"][mask])
    return float(np.sum(b["counts"][mask] / b["n"] * gaps))


def maximum_calibration_error(probs, labels, n_bins: int = 15) -> float:
    """MCE : écart maximal |acc - conf| sur les bins non vides."""
    b = reliability_bins(probs, labels, n_bins)
    mask = b["counts"] > 0
    return float(np.max(np.abs(b["accuracy"][mask] - b["confidence"][mask])))


# --------------------------------------------------------------------------- incertitude
def entropy(probs, axis: int = -1) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    return -np.sum(p * np.log(p + EPS), axis=axis)


def uncertainty_decomposition(mc_probs) -> dict:
    """Décomposition de l'incertitude à partir de T passes stochastiques ``[T, N, C]``.

    - entropie prédictive  H[ E_θ p(y|x,θ) ]          (incertitude totale)
    - entropie attendue    E_θ H[ p(y|x,θ) ]          (part aléatoire)
    - information mutuelle H[E p] - E H[p] = I(y; θ)  (part épistémique, ≥ 0)
    """
    p = check_mc_probs(mc_probs)
    mean = p.mean(axis=0)
    pred_ent = entropy(mean)
    exp_ent = entropy(p).mean(axis=0)
    mi = np.maximum(pred_ent - exp_ent, 0.0)
    return {
        "mean_probs": mean,
        "predictive_entropy": pred_ent,
        "expected_entropy": exp_ent,
        "mutual_information": mi,
        "max_prob": mean.max(axis=1),
        "variation_ratio": 1.0 - mean.max(axis=1),
    }


# --------------------------------------------------------------------------- synthèse
def classification_metrics(probs, labels, n_bins: int = 15) -> dict:
    """Toutes les métriques scalaires pour des probabilités ``[N, C]``."""
    p, y = check_probs(probs, labels)
    return {
        "accuracy": accuracy(p, y),
        "nll": nll(p, y),
        "brier": brier_score(p, y),
        "ece": expected_calibration_error(p, y, n_bins),
        "mce": maximum_calibration_error(p, y, n_bins),
        "n_samples": int(len(y)),
    }


def evaluate_mc(mc_probs, labels, n_bins: int = 15) -> dict:
    """Métriques de la distribution prédictive moyenne + incertitudes moyennes,
    séparées entre prédictions correctes et erronées."""
    dec = uncertainty_decomposition(mc_probs)
    res = classification_metrics(dec["mean_probs"], labels, n_bins)
    _, y = check_probs(dec["mean_probs"], labels)
    correct = dec["mean_probs"].argmax(axis=1) == y
    for key in ("predictive_entropy", "expected_entropy", "mutual_information"):
        v = dec[key]
        res[f"mean_{key}"] = float(v.mean())
        res[f"mean_{key}_correct"] = float(v[correct].mean()) if correct.any() else float("nan")
        res[f"mean_{key}_wrong"] = float(v[~correct].mean()) if (~correct).any() else float("nan")
    res["n_mc_samples"] = int(check_mc_probs(mc_probs).shape[0])
    return res
