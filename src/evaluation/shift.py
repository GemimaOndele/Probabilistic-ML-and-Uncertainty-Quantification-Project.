"""Robustesse sous distribution shift (Issue #21).

Perturbations synthétiques inspirées de CIFAR-10-C / Fashion-MNIST-C
(Hendrycks & Dietterich, 2019), avec 5 niveaux de sévérité.

Les images sont des tableaux ``[N, C, H, W]`` ou ``[N, H, W]`` à valeurs dans
[0, 1], **avant normalisation** : appliquer la normalisation du dataset dans
``predict_fn`` (ou après la perturbation).
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from ._utils import check_mc_probs, to_numpy
from .metrics import evaluate_mc

SEVERITIES = (1, 2, 3, 4, 5)

_PARAMS = {
    "gaussian_noise": (0.04, 0.06, 0.08, 0.10, 0.14),   # écart-type du bruit
    "salt_pepper": (0.01, 0.02, 0.03, 0.05, 0.07),      # proportion de pixels
    "gaussian_blur": (0.5, 0.75, 1.0, 1.5, 2.0),        # sigma (pixels)
    "contrast": (0.75, 0.5, 0.4, 0.3, 0.15),            # facteur de contraste
    "brightness": (0.1, 0.2, 0.3, 0.4, 0.5),            # décalage additif
    "occlusion": (0.1, 0.2, 0.3, 0.4, 0.5),             # côté du carré / côté image
    "rotation": (5, 10, 15, 25, 35),                    # angle en degrés
}
PERTURBATIONS = tuple(_PARAMS)


def _as_nchw(x: np.ndarray) -> tuple[np.ndarray, bool]:
    if x.ndim == 3:
        return x[:, None], True
    if x.ndim != 4:
        raise ValueError(f"images attendues [N, C, H, W] ou [N, H, W], reçu {x.shape}")
    return x, False


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, ((0, 0), (0, 0), (r, r), (r, r)), mode="reflect")
    h = sum(k[i] * pad[:, :, :, i:i + x.shape[3]] for i in range(len(k)))
    return sum(k[i] * h[:, :, i:i + x.shape[2], :] for i in range(len(k)))


def _rotate(x: np.ndarray, angle_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Rotation (plus proche voisin, fond noir), signe de l'angle tiré au hasard par image."""
    n, _, h, w = x.shape
    out = np.zeros_like(x)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    for i in range(n):
        a = np.deg2rad(angle_deg * rng.choice([-1, 1]))
        ys = np.cos(a) * (yy - cy) - np.sin(a) * (xx - cx) + cy
        xs = np.sin(a) * (yy - cy) + np.cos(a) * (xx - cx) + cx
        ys, xs = np.rint(ys).astype(int), np.rint(xs).astype(int)
        ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        out[i][:, ok] = x[i][:, ys[ok], xs[ok]]
    return out


def perturb(images, kind: str, severity: int, seed: int = 0) -> np.ndarray:
    """Applique la perturbation ``kind`` au niveau ``severity`` (1 à 5)."""
    if kind not in _PARAMS:
        raise ValueError(f"perturbation inconnue {kind!r} ; choix : {PERTURBATIONS}")
    if severity not in SEVERITIES:
        raise ValueError("severity doit être entre 1 et 5")
    rng = np.random.default_rng(seed)
    x, squeezed = _as_nchw(to_numpy(images).astype(np.float64))
    p = _PARAMS[kind][severity - 1]
    if kind == "gaussian_noise":
        x = x + rng.normal(0.0, p, size=x.shape)
    elif kind == "salt_pepper":
        mask = rng.random((x.shape[0], 1, *x.shape[2:])) < p
        salt = rng.random(mask.shape) < 0.5
        x = np.where(mask & salt, 1.0, np.where(mask & ~salt, 0.0, x))
    elif kind == "gaussian_blur":
        x = _blur(x, p)
    elif kind == "contrast":
        mean = x.mean(axis=(1, 2, 3), keepdims=True)
        x = (x - mean) * p + mean
    elif kind == "brightness":
        x = x + p
    elif kind == "occlusion":
        h, w = x.shape[2:]
        s = max(1, int(round(p * min(h, w))))
        x = x.copy()
        for i in range(x.shape[0]):
            r0, c0 = rng.integers(0, h - s + 1), rng.integers(0, w - s + 1)
            x[i, :, r0:r0 + s, c0:c0 + s] = 0.0
    elif kind == "rotation":
        x = _rotate(x, p, rng)
    x = np.clip(x, 0.0, 1.0)
    return x[:, 0] if squeezed else x


def evaluate_under_shift(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    images,
    labels,
    perturbations=PERTURBATIONS,
    severities=SEVERITIES,
    seed: int = 0,
    n_bins: int = 15,
) -> list[dict]:
    """Évalue un modèle sur données propres puis perturbées.

    ``predict_fn(images) -> probs`` doit renvoyer ``[N, C]`` (modèle déterministe)
    ou ``[T, N, C]`` (MC Dropout). Une ligne par (perturbation, sévérité) avec
    accuracy, NLL, Brier, ECE, incertitudes moyennes et temps d'inférence ;
    la ligne ``clean`` (sévérité 0) sert de référence.
    """
    y = to_numpy(labels).astype(np.int64)

    def run(x, kind, sev):
        t0 = time.perf_counter()
        probs = check_mc_probs(predict_fn(x))
        dt = time.perf_counter() - t0
        row = {"perturbation": kind, "severity": sev, **evaluate_mc(probs, y, n_bins)}
        row["inference_time_s"] = dt
        return row

    rows = [run(to_numpy(images), "clean", 0)]
    for kind in perturbations:
        for sev in severities:
            rows.append(run(perturb(images, kind, sev, seed=seed + sev), kind, sev))
    clean_acc = rows[0]["accuracy"]
    for r in rows:
        r["accuracy_drop"] = clean_acc - r["accuracy"]
    return rows


def summarize_shift(rows: list[dict]) -> dict:
    """Moyennes par sévérité (toutes perturbations confondues), hors ligne ``clean``."""
    out = {}
    for sev in sorted({r["severity"] for r in rows}):
        sel = [r for r in rows if r["severity"] == sev]
        out[sev] = {k: float(np.mean([r[k] for r in sel]))
                    for k in ("accuracy", "nll", "brier", "ece", "mean_mutual_information",
                              "mean_predictive_entropy")}
    return out
