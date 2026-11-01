"""Unicycle analytic backup policy + the unicycle action-clip helper.

The eq.-(8) handoff is the shared ``BackupPolicy.select_action`` combinator; the
unicycle callers wrap it with ``clip_backup_action`` (pre-clip of the raw action,
post-clip of the composed action) to preserve the historical clip placement.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ps2rl.backup_policy.backup_policy import ABP
from ps2rl.utils.policy import clip_to_box


def clip_backup_action(
    u: jax.Array,
    *,
    a_max: float,
    r_max: float,
) -> jax.Array:
    return clip_to_box(u, [-float(a_max), -float(r_max)], [float(a_max), float(r_max)])


@dataclass(frozen=True)
class UnicycleABP(ABP):
    """Analytic unicycle backup: the saturated LQR base controller, globally."""

    def sa_action(self, x: jax.Array) -> jax.Array:
        return self.base_controller.action(jnp.asarray(x))

    def action(self, x: jax.Array) -> jax.Array:
        u = self.base_controller.action(jnp.asarray(x))
        a_max = float(self.base_controller.u_high[0])
        r_max = float(self.base_controller.u_high[1])
        return clip_backup_action(u, a_max=a_max, r_max=r_max)


__all__ = [
    "UnicycleABP",
    "clip_backup_action",
]
