"""Base set B interface and the ellipsoidal instance both systems use.

``EllipsoidBaseSet``: the sublevel set {x : e(x)^T P e(x) <= base_set_c} 
of the paired base controller's DARE certificate. The base set is certified 
only *relative to its controller*, so the controller is the first constructor
argument and ``__post_init__`` enforces the input-feasibility bound.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Tuple

import jax
import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from ps2rl.base_controller.base_controller import DiscreteLQR


def _softplus(x: jax.Array | float) -> jax.Array:
    x_arr = jnp.asarray(x)
    return jnp.log1p(jnp.exp(-jnp.abs(x_arr))) + jnp.maximum(x_arr, 0.0)


class BaseSet(ABC):
    """Certified base set B."""

    @property
    @abstractmethod
    def num_constraints(self) -> int:
        """Number of scalar base-constraint rows."""

    @abstractmethod
    def values_and_grads(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """BCBF base rows h_B(x) and gradients dh_B/dx."""

    @abstractmethod
    def contains(self, x: jax.Array, atol: float = 0.0) -> jax.Array:
        """Boolean membership (backup hand-off / Phase-1 goal check)."""

    @abstractmethod
    def margin(self, x: jax.Array) -> jax.Array:
        """min_j h_B,j(x)."""

    @abstractmethod
    def smooth_distance(self, x: jax.Array) -> jax.Array:
        """Smoothed distance to B — reset-shell labeling only, never on a control path."""


@dataclass(frozen=True)
class EllipsoidBaseSet(BaseSet):
    """LQR-certified ellipsoid {x : e(x)^T P e(x) <= base_set_c}.

    P and the error map come from the paired controller's certificate.
    """

    controller: "DiscreteLQR"
    base_set_c: float
    smooth_gain: float = 20.0

    def __post_init__(self) -> None:
        base_set_c = float(self.base_set_c)
        if (not isfinite(base_set_c)) or base_set_c <= 0.0:
            raise ValueError(f"base_set_c must be positive and finite, got {self.base_set_c}")
        max_level = float(self.controller.max_certified_level)
        if base_set_c > max_level + 1e-12:
            raise ValueError(
                "base_set_c exceeds the controller's actuator-admissible bound: "
                f"got {base_set_c}, max {max_level}."
            )
        object.__setattr__(self, "base_set_c", base_set_c)
        object.__setattr__(self, "smooth_gain", float(self.smooth_gain))

    @property
    def num_constraints(self) -> int:
        return 1

    def _values_only(self, x: jax.Array) -> jax.Array:
        x_err = self.controller.error_state(jnp.asarray(x))
        p_matrix = jnp.asarray(self.controller.p_matrix, dtype=x_err.dtype)
        quad = x_err @ (p_matrix @ x_err)
        return jnp.asarray([self.base_set_c - quad], dtype=x_err.dtype)

    def values_and_grads(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        if self.controller.error_jacobian_is_identity:
            # Explicit -2 P e gradient (unicycle path, kept verbatim).
            x_arr = jnp.asarray(x)
            err = self.controller.error_state(x_arr)
            p_mat = jnp.asarray(self.controller.p_matrix, dtype=x_arr.dtype)
            quad = err @ (p_mat @ err)
            grad = -(2.0 * (p_mat @ err))
            h = jnp.asarray([self.base_set_c - quad], dtype=x_arr.dtype)
            dh = jnp.asarray([grad], dtype=x_arr.dtype)
            return h, dh
        # Autodiff through the (nonlinear) error map (quadrotor path, verbatim).
        x_arr = jnp.asarray(x)
        vals = self._values_only(x_arr)
        grads = jax.jacfwd(self._values_only)(x_arr)
        return vals, grads

    def contains(self, x: jax.Array, atol: float = 0.0) -> jax.Array:
        if self.controller.error_jacobian_is_identity:
            # Historical unicycle contains traced through values_and_grads.
            vals, _ = self.values_and_grads(x)
        else:
            vals = self._values_only(jnp.asarray(x))
        return jnp.all(vals >= -jnp.asarray(atol, dtype=vals.dtype))

    def margin(self, x: jax.Array) -> jax.Array:
        return jnp.min(self._values_only(jnp.asarray(x)))

    def smooth_distance(self, x: jax.Array) -> jax.Array:
        """Softplus-smoothed distance to B (0 well inside, ~excess outside)."""
        x_arr = jnp.asarray(x)
        gain = max(float(self.smooth_gain), 1e-6)
        x_err = self.controller.error_state(x_arr)
        p_matrix = jnp.asarray(self.controller.p_matrix, dtype=x_err.dtype)
        quad = jnp.einsum("...i,ij,...j->...", x_err, p_matrix, x_err)
        excess = quad - jnp.asarray(self.base_set_c, dtype=x_err.dtype)
        return _softplus(gain * excess) / gain


__all__ = ["BaseSet", "EllipsoidBaseSet", "_softplus"]
