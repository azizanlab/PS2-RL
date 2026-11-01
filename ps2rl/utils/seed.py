"""Random seed helpers."""

from __future__ import annotations

import numpy as np
import jax


def make_prng_key(seed: int) -> jax.Array:
    """Create a JAX PRNG key."""
    return jax.random.PRNGKey(seed)


def make_numpy_rng(seed: int) -> np.random.Generator:
    """Create a NumPy RNG."""
    return np.random.default_rng(seed)

