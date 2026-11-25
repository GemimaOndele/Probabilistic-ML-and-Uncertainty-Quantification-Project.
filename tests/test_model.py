"""Contrat d'interface du CNN : logits [N, C] et modes dropout/BatchNorm."""

import torch
from torch import nn

from src.models.cnn import build_model, dropout_modules, mc_dropout_mode


def test_sortie_logits_et_forme():
    model = build_model("fashion-mnist", dropout_p=0.2)
    x = torch.randn(4, 1, 28, 28)
    assert model(x).shape == (4, 10)

    model_cifar = build_model("cifar-10", dropout_p=0.2)
    assert model_cifar(torch.randn(4, 3, 32, 32)).shape == (4, 10)


def test_mode_mc_dropout_active_uniquement_sur_les_dropouts():
    model = build_model("fashion-mnist", dropout_p=0.3)
    model.eval()
    assert not any(m.training for m in model.modules())

    with mc_dropout_mode(model):
        assert all(m.training for m in dropout_modules(model))
        assert all(
            not m.training for m in model.modules() if isinstance(m, nn.BatchNorm2d)
        )
    assert not any(m.training for m in model.modules())


def test_dataset_inconnu():
    import pytest

    with pytest.raises(ValueError):
        build_model("mnist")
