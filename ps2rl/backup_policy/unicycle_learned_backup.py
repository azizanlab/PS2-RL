"""Unicycle learned backup policy: checkpoint wrapper + LBP composition.

Moved from ``ps2rl.phase1_sa.unicycle_learned_policy`` (Batch 6b-2). The
checkpoint payload schema is unchanged; the metadata/payload validation
against the runtime CBF config stays with the CBF system module until the
Batch 6b-3 cil merge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Any, Dict

import jax
import jax.numpy as jnp

from ps2rl.backup_policy.backup_policy import LBP
from ps2rl.backup_policy.unicycle_analytic_backup import clip_backup_action
from ps2rl.utils.policy import ActorConfig, _coerce_actor_cfg, actor_mean_action
from ps2rl.sets.base_sets import EllipsoidBaseSet


@dataclass(frozen=True)
class LearnedUnicycleBackupPolicy:
    """Deterministic learned backup policy wrapper."""

    actor_params: Any
    actor_cfg: ActorConfig
    action_scale: jax.Array
    action_low: jax.Array
    action_high: jax.Array
    metadata: Dict[str, Any] = field(default_factory=dict)

    def _action_bounds(self) -> tuple[jax.Array, jax.Array]:
        return self.action_low, self.action_high

    def action_single(self, x: jax.Array) -> jax.Array:
        low, high = self._action_bounds()
        act = actor_mean_action(
            self.actor_params,
            x[None, :],
            self.action_scale,
            self.actor_cfg,
            action_low=self.action_low,
            action_high=self.action_high,
        )[0]
        return jnp.clip(act, low, high)

    def action_batch(self, x_batch: jax.Array) -> jax.Array:
        low, high = self._action_bounds()
        act = actor_mean_action(
            self.actor_params,
            x_batch,
            self.action_scale,
            self.actor_cfg,
            action_low=self.action_low,
            action_high=self.action_high,
        )
        return jnp.clip(act, low, high)


def load_learned_unicycle_backup_policy(path: str | Path) -> LearnedUnicycleBackupPolicy:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Learned backup policy checkpoint not found: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(payload)}")
    if "actor_params" not in payload:
        raise KeyError(f"Missing 'actor_params' in learned backup policy file: {ckpt_path}")
    if "actor_cfg" not in payload:
        raise KeyError(f"Missing 'actor_cfg' in learned backup policy file: {ckpt_path}")
    if "action_scale" not in payload:
        raise KeyError(f"Missing 'action_scale' in learned backup policy file: {ckpt_path}")

    if "action_low" not in payload or "action_high" not in payload:
        raise KeyError(f"Missing 'action_low'/'action_high' in learned backup policy file: {ckpt_path}")

    actor_cfg = _coerce_actor_cfg(payload["actor_cfg"])
    action_scale = jnp.asarray(payload["action_scale"], dtype=jnp.float32)
    action_low = jnp.asarray(payload["action_low"], dtype=jnp.float32)
    action_high = jnp.asarray(payload["action_high"], dtype=jnp.float32)
    actor_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), payload["actor_params"])
    return LearnedUnicycleBackupPolicy(
        actor_params=actor_params,
        actor_cfg=actor_cfg,
        action_scale=action_scale,
        action_low=action_low,
        action_high=action_high,
        metadata=dict(payload.get("metadata", {})),
    )


@dataclass(frozen=True)
class UnicycleLBP(LBP):
    """Learned unicycle backup: Phase-1 safe-arrival actor + LQR-on-B handoff."""

    learned: LearnedUnicycleBackupPolicy = None  # type: ignore[assignment]
    a_max: float = 0.0
    r_max: float = 0.0
    capture_set: EllipsoidBaseSet | None = None

    def sa_action(self, x: jax.Array) -> jax.Array:
        clip_low = jnp.asarray([-self.a_max, -self.r_max], dtype=jnp.float32)
        clip_high = jnp.asarray([self.a_max, self.r_max], dtype=jnp.float32)
        raw = self.learned.action_single(jnp.asarray(x, dtype=jnp.float32))
        raw = jnp.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw = jnp.clip(raw, clip_low, clip_high)
        return jnp.asarray(raw, dtype=x.dtype)

    def action(self, x: jax.Array) -> jax.Array:
        u = self.sa_action(x)
        if self.capture_set is not None:
            u = clip_backup_action(
                self.select_action(
                    x,
                    clip_backup_action(u, a_max=self.a_max, r_max=self.r_max),
                    self.capture_set,
                    controller=self.base_controller,
                ),
                a_max=self.a_max,
                r_max=self.r_max,
            )
        return clip_backup_action(u, a_max=self.a_max, r_max=self.r_max)


__all__ = [
    "LearnedUnicycleBackupPolicy",
    "UnicycleLBP",
    "load_learned_unicycle_backup_policy",
]
