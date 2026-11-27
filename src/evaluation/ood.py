"""Détection hors distribution (Issue #21).

Score d'anomalie : plus il est élevé, plus l'exemple est jugé OOD.
Les métriques (AUROC, AUPR, FPR@95%TPR) sont calculées en NumPy, en traitant
les exemples OOD comme la classe positive.
"""
from __future__ import annotations

import numpy as np

from ._utils import check_mc_probs, to_numpy
from .metrics import uncertainty_decomposition

SCORES = ("max_prob", "predictive_entropy", "mutual_information", "expected_entropy")


def uncertainty_scores(mc_probs) -> dict[str, np.ndarray]:
    """Scores d'anomalie à partir de probabilités ``[T, N, C]`` ou ``[N, C]``.

    ``max_prob`` est renvoyé sous forme ``1 - max p`` pour que « grand = OOD ».
    Avec un modèle déterministe (T = 1), l'information mutuelle est nulle.
    """
    dec = uncertainty_decomposition(check_mc_probs(mc_probs))
    return {
        "max_prob": 1.0 - dec["max_prob"],
        "predictive_entropy": dec["predictive_entropy"],
        "mutual_information": dec["mutual_information"],
        "expected_entropy": dec["expected_entropy"],
    }


def _binary_curves(scores_in, scores_out):
    s = np.concatenate([to_numpy(scores_in), to_numpy(scores_out)]).astype(np.float64)
    y = np.concatenate([np.zeros(len(scores_in)), np.ones(len(scores_out))])
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    # seuils distincts (gestion correcte des ex aequo)
    distinct = np.where(np.diff(s))[0]
    thr_idx = np.r_[distinct, len(s) - 1]
    tp = np.cumsum(y)[thr_idx]
    fp = (thr_idx + 1) - tp
    return tp, fp, y.sum(), len(y) - y.sum()


def auroc(scores_in, scores_out) -> float:
    tp, fp, P, N = _binary_curves(scores_in, scores_out)
    tpr = np.r_[0.0, tp / P]
    fpr = np.r_[0.0, fp / N]
    return float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))


def aupr(scores_in, scores_out) -> float:
    """Aire sous la courbe précision-rappel (average precision), OOD = positif."""
    tp, fp, P, _ = _binary_curves(scores_in, scores_out)
    precision = tp / (tp + fp)
    recall = tp / P
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def fpr_at_tpr(scores_in, scores_out, tpr_level: float = 0.95) -> float:
    """Taux de faux positifs (ID pris pour OOD) quand on détecte ``tpr_level`` des OOD."""
    tp, fp, P, N = _binary_curves(scores_in, scores_out)
    tpr = tp / P
    i = int(np.searchsorted(tpr, tpr_level, side="left"))
    return float(fp[min(i, len(fp) - 1)] / N)


def ood_detection_metrics(mc_probs_in, mc_probs_out, scores=SCORES) -> dict:
    """AUROC / AUPR / FPR@95 pour chaque score d'incertitude."""
    s_in, s_out = uncertainty_scores(mc_probs_in), uncertainty_scores(mc_probs_out)
    out = {}
    for name in scores:
        out[name] = {
            "auroc": auroc(s_in[name], s_out[name]),
            "aupr": aupr(s_in[name], s_out[name]),
            "fpr95": fpr_at_tpr(s_in[name], s_out[name], 0.95),
            "mean_in": float(np.mean(s_in[name])),
            "mean_out": float(np.mean(s_out[name])),
        }
    return out
