"""Lane-keeping safe set.

The base set B and its LQR base controller live in the generic layer:

    controller = UnicycleDLQR.from_config(cfg)
    base_set   = EllipsoidBaseSet(controller, cfg.base_set_c)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Tuple

import jax.numpy as jnp

from ps2rl.sets.safe_sets import SafeSet


@dataclass(frozen=True)
class UnicycleSafeSet(SafeSet):
    """Box safe set C = {|y|<=y_max, |psi|<=psi_max}."""

    y_max: float = 1.8
    psi_max: float = pi / 3.0

    @property
    def num_constraints(self) -> int:
        return 4

    def values_and_grads(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        y, _, psi = x
        h = jnp.array(
            [
                self.y_max - y,
                self.y_max + y,
                self.psi_max - psi,
                self.psi_max + psi,
            ],
            dtype=x.dtype,
        )
        dh = jnp.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=x.dtype,
        )
        return h, dh

    def contains(self, x: jnp.ndarray) -> jnp.ndarray:
        return (jnp.abs(x[0]) <= self.y_max) & (jnp.abs(x[2]) <= self.psi_max)
