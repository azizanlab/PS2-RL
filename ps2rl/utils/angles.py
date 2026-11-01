"""Angle helpers: wrap radians into ``(-pi, pi]``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def wrap_angle(theta: jax.Array) -> jax.Array:
    """Wrap ``theta`` (radians) into ``(-pi, pi]`` — JAX/f32 control+env path."""
    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


def wrap_angle_np(theta: np.ndarray | float) -> np.ndarray:
    """Wrap ``theta`` (radians) into ``(-pi, pi]`` — NumPy/float64 eval-metric path."""
    arr = np.asarray(theta, dtype=np.float64)
    return np.arctan2(np.sin(arr), np.cos(arr))


__all__ = ["wrap_angle", "wrap_angle_np"]
