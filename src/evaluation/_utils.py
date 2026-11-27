"""Utilitaires internes : conversion des entrées et validations de forme."""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def to_numpy(x) -> np.ndarray:
    """Convertit un tableau NumPy, une liste ou un tenseur PyTorch en ``np.ndarray``.

    PyTorch n'est pas importé : on se contente de détecter ``detach``/``cpu``.
    """
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def softmax(logits, axis: int = -1) -> np.ndarray:
    """Softmax numériquement stable."""
    z = to_numpy(logits).astype(np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def check_probs(probs, labels=None) -> tuple[np.ndarray, np.ndarray | None]:
    """Valide des probabilités ``[N, C]`` et des étiquettes entières ``[N]``."""
    p = to_numpy(probs).astype(np.float64)
    if p.ndim != 2:
        raise ValueError(f"probabilités attendues de forme [N, C], reçu {p.shape}")
    if np.any(p < -1e-6) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("les lignes de probabilités doivent être positives et sommer à 1")
    y = None
    if labels is not None:
        y = to_numpy(labels).astype(np.int64).reshape(-1)
        if y.shape[0] != p.shape[0]:
            raise ValueError(f"{p.shape[0]} prédictions mais {y.shape[0]} étiquettes")
        if y.min() < 0 or y.max() >= p.shape[1]:
            raise ValueError("étiquettes hors de l'intervalle [0, C)")
    return np.clip(p, 0.0, 1.0), y


def check_mc_probs(mc_probs) -> np.ndarray:
    """Valide des probabilités Monte Carlo ``[T, N, C]`` (une passe stochastique par T)."""
    p = to_numpy(mc_probs).astype(np.float64)
    if p.ndim == 2:
        p = p[None]
    if p.ndim != 3:
        raise ValueError(f"probabilités MC attendues de forme [T, N, C], reçu {p.shape}")
    if not np.allclose(p.sum(axis=2), 1.0, atol=1e-4):
        raise ValueError("chaque passe MC doit fournir des probabilités qui somment à 1")
    return np.clip(p, 0.0, 1.0)
