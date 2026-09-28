"""Seeds must be derived deterministically, never via the built-in `hash()`,
and must reproduce bit-identical results."""
from __future__ import annotations

import numpy as np

from covfrenet.rng import trial_seed


def test_trial_seed_repeatable():
    a = trial_seed("S01", (1.5, 2), 3)
    b = trial_seed("S01", (1.5, 2), 3)
    assert a == b


def test_trial_seed_distinguishes_inputs():
    base = trial_seed("S01", (1.5, 2), 3)
    assert base != trial_seed("S02", (1.5, 2), 3)
    assert base != trial_seed("S01", (1.5, 3), 3)
    assert base != trial_seed("S01", (1.5, 2), 4)


def test_default_rng_bitwise_reproducible():
    seed = trial_seed("S01", (1.5, 2), 3)
    a = np.random.default_rng(seed).normal(size=100)
    b = np.random.default_rng(seed).normal(size=100)
    assert np.array_equal(a, b)
