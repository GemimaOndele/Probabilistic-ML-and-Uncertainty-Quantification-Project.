"""Utilitaires communs : seeds, horodatage et sérialisation JSON."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Fixe les générateurs Python, NumPy et PyTorch pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def write_json(path: str | Path, payload: dict) -> None:
    """Écrit un JSON lisible, avec création automatique du dossier parent."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


class Timer:
    """Chronomètre simple pour les coûts d'entraînement / d'inférence."""

    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start
