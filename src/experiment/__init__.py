"""Shared experiment setup."""
import random


def seed_python(seed: int) -> None:
    """Seed Python only; numerical framework seeds remain explicit."""
    random.seed(seed)
