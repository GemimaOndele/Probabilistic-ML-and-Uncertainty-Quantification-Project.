import random
import pytest
from evaluation.contracts import validate_predictions
from experiment import seed_python


def test_valid_mc():
    validate_predictions(["a", "b"], [0, 1], [[.5,.5],[.2,.8]],
                         [[[.4,.6],[.1,.9]], [[.6,.4],[.3,.7]]])


@pytest.mark.parametrize("ids,labels,probs", [
    ([], [], []), (["a","a"], [0,1], [[.5,.5],[.5,.5]]),
    (["a"], [2], [[.5,.5]]), (["a"], [True], [[.5,.5]]),
    (["a"], [0], [[float("nan"),.5]]), (["a"], [0], [[-.1,1.1]]),
    (["a"], [0], [[.4,.4]]), (["a","b"], [0], [[.5,.5]]),
])
def test_invalid_predictions(ids, labels, probs):
    with pytest.raises(ValueError):
        validate_predictions(ids, labels, probs)


def test_mc_mismatch_and_empty():
    for mc in ([], [[[.8,.2]]], [[[.5,.5],[.5,.5]]]):
        with pytest.raises(ValueError):
            validate_predictions(["a"], [0], [[.5,.5]], mc)


def test_seed_replays_python_random():
    seed_python(42)
    first = [random.random() for _ in range(5)]
    seed_python(42)
    assert first == [random.random() for _ in range(5)]
