"""Prédiction conforme split (Issue #22).

Deux scores de non-conformité :
- ``lac`` (Sadinle et al., 2019) : s(x, y) = 1 - p_y(x) ; ensembles les plus petits ;
- ``aps`` (Romano et al., 2020) : masse cumulée jusqu'à la vraie classe,
  avec randomisation optionnelle ; couverture conditionnelle plus homogène.

Garantie (échangeabilité) : P(y ∈ C(x)) ≥ 1 - α, calibration sur le split
``calibration`` uniquement, évaluation sur ``test``.
"""
from __future__ import annotations

import math

import numpy as np

from ._utils import check_probs


def _aps_scores(p: np.ndarray, y: np.ndarray, u: np.ndarray | None) -> np.ndarray:
    order = np.argsort(-p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    rank = np.argmax(order == y[:, None], axis=1)
    s = cum[np.arange(len(y)), rank]
    if u is not None:
        s = s - u * sorted_p[np.arange(len(y)), rank]
    return s


class SplitConformalClassifier:
    def __init__(self, alpha: float = 0.1, method: str = "lac", randomized: bool = True,
                 seed: int = 0):
        if not 0 < alpha < 1:
            raise ValueError("alpha doit être dans ]0, 1[")
        if method not in ("lac", "aps"):
            raise ValueError("method doit valoir 'lac' ou 'aps'")
        self.alpha, self.method, self.randomized, self.seed = alpha, method, randomized, seed
        self.qhat: float | None = None

    def _u(self, n: int, rng) -> np.ndarray | None:
        return rng.random(n) if (self.method == "aps" and self.randomized) else None

    def fit(self, probs_cal, labels_cal) -> "SplitConformalClassifier":
        p, y = check_probs(probs_cal, labels_cal)
        rng = np.random.default_rng(self.seed)
        if self.method == "lac":
            s = 1.0 - p[np.arange(len(y)), y]
        else:
            s = _aps_scores(p, y, self._u(len(y), rng))
        n = len(s)
        level = min(1.0, math.ceil((n + 1) * (1 - self.alpha)) / n)
        self.qhat = float(np.quantile(s, level, method="higher"))
        self.n_cal = n
        return self

    def predict_sets(self, probs) -> np.ndarray:
        """Masque booléen ``[N, C]`` : True si la classe appartient à l'ensemble."""
        if self.qhat is None:
            raise RuntimeError("appeler fit() sur le split de calibration")
        p, _ = check_probs(probs)
        if self.method == "lac":
            return p >= 1.0 - self.qhat
        rng = np.random.default_rng(self.seed + 1)
        order = np.argsort(-p, axis=1)
        sorted_p = np.take_along_axis(p, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)
        u = self._u(len(p), rng)
        score = cum - (u[:, None] * sorted_p if u is not None else 0.0)
        keep_sorted = score <= self.qhat
        keep_sorted[:, 0] |= u is None  # version non randomisée : jamais d'ensemble vide
        mask = np.zeros_like(p, dtype=bool)
        np.put_along_axis(mask, order, keep_sorted, axis=1)
        return mask

    def evaluate(self, probs, labels) -> dict:
        p, y = check_probs(probs, labels)
        sets = self.predict_sets(p)
        sizes = sets.sum(axis=1)
        covered = sets[np.arange(len(y)), y]
        n_classes = p.shape[1]
        per_class = [float(covered[y == c].mean()) if np.any(y == c) else float("nan")
                     for c in range(n_classes)]
        return {
            "method": self.method,
            "alpha": self.alpha,
            "target_coverage": 1 - self.alpha,
            "qhat": self.qhat,
            "coverage": float(covered.mean()),
            "mean_set_size": float(sizes.mean()),
            "median_set_size": float(np.median(sizes)),
            "empty_set_rate": float(np.mean(sizes == 0)),
            "singleton_rate": float(np.mean(sizes == 1)),
            "min_class_coverage": float(np.nanmin(per_class)),
            "per_class_coverage": per_class,
        }
