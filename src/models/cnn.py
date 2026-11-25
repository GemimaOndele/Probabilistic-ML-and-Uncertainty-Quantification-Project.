"""CNN de classification retournant des logits, avec dropout utilisable en MC.

Contrat (docs/contrat-technique.md) : entrée image, sortie logits ``[N, C]`` ;
le dropout est actif à l'entraînement et peut être réactivé à l'inférence pour
l'inférence Monte Carlo, tandis que BatchNorm reste en mode ``eval``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from torch import Tensor, nn


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout2d."""

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DropoutCNN(nn.Module):
    """CNN compact pour Fashion-MNIST / CIFAR-10, sortie logits ``[N, C]``.

    Le même dropout est utilisé à l'entraînement et échantillonné à l'inférence
    via :func:`mc_dropout_mode`. Les dimensions sont calculées pour 28×28 et
    32×32 (après deux MaxPool : 7×7 et 8×8).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout_p: float = 0.25,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, dropout_p),
            ConvBlock(32, 64, dropout_p),
        )
        spatial = 7 if in_channels == 1 else 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * spatial * spatial, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))


def build_model(dataset: str, dropout_p: float = 0.25) -> DropoutCNN:
    """Fabrique le CNN selon le jeu de données du sujet 5."""
    if dataset not in {"fashion-mnist", "cifar-10"}:
        raise ValueError(f"dataset inconnu : {dataset!r}")
    return DropoutCNN(in_channels=1 if dataset == "fashion-mnist" else 3, dropout_p=dropout_p)


def dropout_modules(model: nn.Module) -> list[nn.Dropout]:
    """Retourne toutes les couches Dropout / Dropout2d du modèle."""
    return [m for m in model.modules() if isinstance(m, (nn.Dropout, nn.Dropout2d))]


@contextmanager
def mc_dropout_mode(model: nn.Module) -> Iterator[nn.Module]:
    """Contexte MC : BatchNorm en eval, uniquement les dropout en train.

    Les modes initiaux sont restaurés à la sortie, ce qui évite de laisser le
    modèle en train après l'inférence.
    """
    previous = {m: m.training for m in model.modules()}
    model.eval()
    for layer in dropout_modules(model):
        layer.train()
    try:
        yield model
    finally:
        for module, was_training in previous.items():
            module.train(was_training)
