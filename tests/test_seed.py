"""
test_seed.py
============
Unit tests for src/utils/seed.py.

Verifies:
- set_seed() sets consistent random states for random, numpy, and torch.
- Seeding is reproducible: two calls with the same seed produce the same
  random numbers.
- Invalid seed values raise ValueError.
- PYTHONHASHSEED environment variable is set.
"""

import os
import random

import numpy as np
import pytest
import torch
from src.utils.seed import DEFAULT_SEED, set_seed


class TestSetSeedReproducibility:
    @pytest.mark.smoke
    def test_random_reproducible(self) -> None:
        set_seed(42)
        val_a = random.random()
        set_seed(42)
        val_b = random.random()
        assert val_a == val_b

    @pytest.mark.smoke
    def test_numpy_reproducible(self) -> None:
        set_seed(42)
        arr_a = np.random.rand(10)
        set_seed(42)
        arr_b = np.random.rand(10)
        np.testing.assert_array_equal(arr_a, arr_b)

    @pytest.mark.smoke
    def test_torch_reproducible(self) -> None:
        set_seed(42)
        t_a = torch.rand(10)
        set_seed(42)
        t_b = torch.rand(10)
        assert torch.allclose(t_a, t_b)

    def test_different_seeds_produce_different_values(self) -> None:
        set_seed(1)
        val_1 = random.random()
        set_seed(2)
        val_2 = random.random()
        assert val_1 != val_2


class TestSetSeedEnvironment:
    def test_pythonhashseed_is_set(self) -> None:
        set_seed(99)
        assert os.environ["PYTHONHASHSEED"] == "99"

    def test_pythonhashseed_matches_seed(self) -> None:
        set_seed(123)
        assert os.environ["PYTHONHASHSEED"] == "123"


class TestSetSeedValidation:
    def test_invalid_negative_seed_raises(self) -> None:
        with pytest.raises(ValueError, match="Seed must be in"):
            set_seed(-1)

    def test_invalid_overflow_seed_raises(self) -> None:
        with pytest.raises(ValueError, match="Seed must be in"):
            set_seed(2**32)

    def test_boundary_seed_zero_ok(self) -> None:
        set_seed(0)  # must not raise

    def test_boundary_seed_max_ok(self) -> None:
        set_seed(2**32 - 1)  # must not raise


class TestDefaultSeed:
    def test_default_seed_is_integer(self) -> None:
        assert isinstance(DEFAULT_SEED, int)

    def test_default_seed_in_valid_range(self) -> None:
        assert 0 <= DEFAULT_SEED < 2**32


class TestFullDeterminism:
    def test_full_determinism_does_not_raise(self) -> None:
        """Enabling full_determinism must not crash on CPU."""
        set_seed(42, full_determinism=True)
