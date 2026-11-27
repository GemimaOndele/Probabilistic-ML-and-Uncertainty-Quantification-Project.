"""Temperature Scaling et diagrammes de fiabilité (Issue #20).

La température est ajustée **uniquement sur le split de calibration**, puis
appliquée telle quelle au test (et aux données décalées / OOD).

Fonctionne avec :
- des logits ``[N, C]`` (CNN classique) ;
- des probabilités MC ``[T, N, C]`` (MC Dropout) : chaque passe est remise à
  l'échelle via ``softmax(log p_t / T)``, ce qui équivaut à diviser ses logits
  par T, puis les passes sont moyennées.
"""
from __future__ import annotations

import math

import numpy as np

from ._utils import EPS, check_mc_probs, softmax, to_numpy
from .metrics import nll, reliability_bins


def _scale(scores: np.ndarray, temperature: float, from_probs: bool) -> np.ndarray:
    """Applique la température ; renvoie des probabilités ``[N, C]``."""
    if from_probs:  # scores = probabilités MC [T, N, C]
        return softmax(np.log(scores + EPS) / temperature, axis=-1).mean(axis=0)
    return softmax(scores / temperature, axis=-1)


class TemperatureScaler:
    """Calibration post-hoc à un paramètre (Guo et al., 2017).

    La NLL est convexe en β = 1/T ; on minimise donc par recherche dichotomique
    (section dorée) sur log T dans ``[log t_min, log t_max]``, sans dépendance
    à PyTorch ni SciPy.
    """

    def __init__(self, t_min: float = 0.05, t_max: float = 20.0, tol: float = 1e-5):
        self.t_min, self.t_max, self.tol = t_min, t_max, tol
        self.temperature: float | None = None
        self.nll_before: float | None = None
        self.nll_after: float | None = None

    @staticmethod
    def _prepare(scores, from_probs: bool) -> np.ndarray:
        if from_probs:
            return check_mc_probs(scores)
        z = to_numpy(scores).astype(np.float64)
        if z.ndim != 2:
            raise ValueError(f"logits attendus de forme [N, C], reçu {z.shape}")
        return z

    def fit(self, scores, labels, from_probs: bool = False) -> "TemperatureScaler":
        """Ajuste T sur le split de calibration.

        ``scores`` : logits ``[N, C]`` (``from_probs=False``) ou probabilités
        MC ``[T, N, C]`` / ``[N, C]`` (``from_probs=True``).
        """
        s = self._prepare(scores, from_probs)
        y = to_numpy(labels).astype(np.int64).reshape(-1)

        def loss(log_t: float) -> float:
            return nll(_scale(s, math.exp(log_t), from_probs), y)

        a, b = math.log(self.t_min), math.log(self.t_max)
        g = (math.sqrt(5) - 1) / 2
        c, d = b - g * (b - a), a + g * (b - a)
        fc, fd = loss(c), loss(d)
        while b - a > self.tol:
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - g * (b - a)
                fc = loss(c)
            else:
                a, c, fc = c, d, fd
                d = a + g * (b - a)
                fd = loss(d)
        self.temperature = float(math.exp((a + b) / 2))
        self.nll_before = loss(0.0)
        self.nll_after = loss(math.log(self.temperature))
        self._from_probs = from_probs
        return self

    def transform(self, scores, from_probs: bool | None = None) -> np.ndarray:
        """Renvoie les probabilités calibrées ``[N, C]``."""
        if self.temperature is None:
            raise RuntimeError("appeler fit() sur le split de calibration avant transform()")
        fp = self._from_probs if from_probs is None else from_probs
        return _scale(self._prepare(scores, fp), self.temperature, fp)

    def transform_mc(self, mc_probs) -> np.ndarray:
        """Remet à l'échelle chaque passe MC et renvoie ``[T, N, C]`` (utile pour
        recalculer entropie et information mutuelle après calibration)."""
        if self.temperature is None:
            raise RuntimeError("appeler fit() avant transform_mc()")
        p = check_mc_probs(mc_probs)
        return softmax(np.log(p + EPS) / self.temperature, axis=-1)

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "calibration_nll_before": self.nll_before,
            "calibration_nll_after": self.nll_after,
        }


def plot_reliability_diagram(probs, labels, n_bins: int = 15, title: str = "",
                             ax=None, path: str | None = None):
    """Diagramme de fiabilité (accuracy vs confiance) + histogramme des confiances.

    Matplotlib est importé à la demande. Si ``path`` est fourni, la figure est
    enregistrée. Renvoie la figure.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    from .metrics import expected_calibration_error

    b = reliability_bins(probs, labels, n_bins)
    ece = expected_calibration_error(probs, labels, n_bins)
    centers = (b["edges"][:-1] + b["edges"][1:]) / 2
    width = 1.0 / n_bins
    acc = np.nan_to_num(b["accuracy"])

    if ax is None:
        fig, (ax, ax_h) = plt.subplots(2, 1, figsize=(5, 6), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig, ax_h = ax.figure, None
    ax.bar(centers, acc, width=width, edgecolor="black", alpha=0.8, label="Accuracy")
    ax.bar(centers, np.nan_to_num(b["confidence"]) - acc, bottom=acc, width=width,
           color="tab:red", alpha=0.3, edgecolor="tab:red", label="Écart")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Calibration parfaite")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title} (ECE = {ece:.3f})".strip())
    ax.legend(loc="upper left")
    if ax_h is not None:
        ax_h.bar(centers, b["counts"] / max(b["n"], 1), width=width, edgecolor="black")
        ax_h.set_xlabel("Confiance")
        ax_h.set_ylabel("Proportion")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
