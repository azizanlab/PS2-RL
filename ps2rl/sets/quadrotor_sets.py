"""Quadrotor safe set (the hard-deck ceiling).

The base set B and its hover-LQR base controller live in the generic layer:

    controller = QuadrotorDLQR.from_config(cbf_cfg)
    base_set   = EllipsoidBaseSet(controller, cbf_cfg.base_set_c,
                                  smooth_gain=cbf_cfg.base_set_smooth_gain)

Quaternion / attitude helpers live in ``ps2rl.utils.quaternion``; the reduced
hover error map lives with the controller in ``ps2rl.base_controller.quadrotor_dlqr``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ps2rl.base_controller.quadrotor_dlqr import QuadrotorDLQR
from ps2rl.sets.safe_sets import SafeSet


@dataclass(frozen=True)
class QuadrotorSafeSet(SafeSet):
    z_max: float = 3.0

    @classmethod
    def from_cbf_config(cls, cfg: Any) -> "QuadrotorSafeSet":
        return cls(z_max=float(cfg.z_max))

    @property
    def num_constraints(self) -> int:
        return 1

    def values_and_grads(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        x_arr = jnp.asarray(x)
        val = jnp.asarray(self.z_max, dtype=x_arr.dtype) - x_arr[2]
        grad = jnp.zeros((10,), dtype=x_arr.dtype).at[2].set(-1.0)
        return jnp.stack([val]), jnp.stack([grad])

    def contains(self, x: jax.Array) -> jax.Array:
        x_arr = jnp.asarray(x)
        safe = jnp.asarray(True, dtype=jnp.bool_)
        safe = safe & (x_arr[..., 2] <= self.z_max)
        return safe


__all__ = [
    "QuadrotorDLQR",
    "QuadrotorSafeSet",
]
