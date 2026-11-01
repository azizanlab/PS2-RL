"""Safe-set interface S = {x : h_S(x) >= 0}.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import jax
import jax.numpy as jnp


class SafeSet(ABC):
    """Safe set defined by stacked differentiable rows h_S,j(x) >= 0."""

    @property
    @abstractmethod
    def num_constraints(self) -> int:
        """Number of scalar constraint rows."""

    @abstractmethod
    def values_and_grads(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Stacked row values h(x) and gradients dh/dx (shape (n_c,), (n_c, n))."""

    @abstractmethod
    def contains(self, x: jax.Array) -> jax.Array:
        """Boolean membership x in S."""

    def margin(self, x: jax.Array) -> jax.Array:
        """min_j h_S,j(x); +inf when the set has no rows."""
        vals, _ = self.values_and_grads(jnp.asarray(x))
        if vals.shape[0] == 0:
            return jnp.asarray(jnp.inf, dtype=jnp.asarray(x).dtype)
        return jnp.min(vals)


__all__ = ["SafeSet"]
