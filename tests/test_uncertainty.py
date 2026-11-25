"""Tests des probabilités MC et de la décomposition entropie / information mutuelle."""

import numpy as np
import torch

from src.models.cnn import build_model
from src.uncertainty.decomposition import decompose_uncertainty
from src.uncertainty.mc_dropout import entropy, predict_mc, summarize_predictions


def test_entropie_uniforme_et_one_hot():
    k = 4
    assert np.isclose(entropy(np.full(k, 1.0 / k)), np.log(k))
    one_hot = np.zeros(k)
    one_hot[2] = 1.0
    assert np.isclose(entropy(one_hot), 0.0, atol=1e-10)


def test_predict_mc_forme_et_variation():
    torch.manual_seed(0)
    model = build_model("fashion-mnist", dropout_p=0.3)
    model.eval()
    x = torch.randn(6, 1, 28, 28)
    probs = predict_mc(model, x, num_samples=5, batch_size=2)
    assert probs.shape == (5, 6, 10)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-6)
    # Avec dropout actif, les passes doivent différer.
    assert not np.allclose(probs[0], probs[1])


def test_predict_mc_sans_dropout_est_deterministe():
    torch.manual_seed(0)
    model = build_model("fashion-mnist", dropout_p=0.0)
    model.eval()
    probs = predict_mc(model, torch.randn(4, 1, 28, 28), num_samples=3)
    assert np.allclose(probs[0], probs[1], atol=1e-6)


def test_resume_entropies_et_information_mutuelle():
    # Deux passes désaccordées => information mutuelle strictement positive.
    p0 = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
    p1 = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    probs = np.stack([p0, p1])  # [T=2, N=2, K=2]
    summary = summarize_predictions(probs)
    assert summary["probs_mean"].shape == (2, 2)
    assert np.all(summary["mutual_information"] > 0.0)
    assert np.all(
        summary["predictive_entropy"]
        >= summary["expected_entropy"] - 1e-12
    )


def test_information_mutuelle_nulle_avec_une_passe():
    p = np.array([[[0.7, 0.3]]], dtype=np.float32)  # [T=1, N=1, K=2]
    summary = summarize_predictions(p)
    assert np.isclose(summary["mutual_information"][0], 0.0, atol=1e-12)


def test_decomposition_equivalente_si_membres_identiques():
    # Un ensemble de 3 membres identiques doit donner la meme decomposition
    # qu'un seul reseau avec les memes passes dropout.
    p = np.array(
        [[[0.8, 0.2], [0.3, 0.7]], [[0.7, 0.3], [0.4, 0.6]]], dtype=np.float32
    )  # [T=2, N=2, K=2]
    single = decompose_uncertainty(p)
    ensemble = decompose_uncertainty(np.stack([p, p, p]))  # [M=3, T=2, N=2, K=2]
    assert ensemble["total_passes"] == 6
    assert ensemble["num_passes_per_member"] == 2
    assert np.allclose(
        ensemble["mutual_information"], single["mutual_information"], atol=1e-6
    )
    assert np.allclose(ensemble["between_member_mi"], 0.0, atol=1e-6)


def test_decomposition_rejette_mauvaise_dimension():
    import pytest

    with pytest.raises(ValueError):
        decompose_uncertainty(np.zeros((2, 3), dtype=np.float32))


def test_metriques_avec_labels():
    p = np.array(
        [[[0.8, 0.2], [0.3, 0.7]], [[0.7, 0.3], [0.4, 0.6]]], dtype=np.float32
    )
    summary = summarize_predictions(p, labels=np.array([0, 1]))
    assert summary["accuracy"] == 1.0
    assert summary["nll"] > 0.0
