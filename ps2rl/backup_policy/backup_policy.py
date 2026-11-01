"""Composed backup policy pi_b

The backup policy is the composition of a certified base controller pi_B
(inside the base set B) with a safe-arrival policy pi_SA (outside B):

    pi_b(x) = pi_B(x)  if x in B,   pi_SA(x)  otherwise.

``ABP`` uses a hand-designed analytic safe-arrival policy; 
``LBP`` wraps a Phase-1 learned safe-arrival actor. 
The BCBF engine consumes ``BackupPolicy.action`` as pi_b for backup rollouts; 
Phase-1's in-training handoff evaluation uses the same combinator 
(``select_action``) on the actor output arriving as data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
from typing import TYPE_CHECKING, Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.utils.policy import ActorConfig

if TYPE_CHECKING:  # pragma: no cover
    from ps2rl.base_controller.base_controller import BaseController
    from ps2rl.sets.base_sets import BaseSet


@dataclass(frozen=True)
class BackupPolicy(ABC):
    """pi_b = (pi_B on B) + (pi_SA off B)"""

    base_controller: "BaseController"
    base_set: "BaseSet"

    @abstractmethod
    def sa_action(self, x: jax.Array) -> jax.Array:
        """The outside-B safe-arrival policy pi_SA(x)."""

    @staticmethod
    def select_action(
        x: jax.Array,
        raw_action: jax.Array,
        base_set: "BaseSet",
        controller: "BaseController | None" = None,
    ) -> jax.Array:
        """Action-level combinator: base controller inside B, raw outside.

        The single home for the handoff select, callable without composing a
        BackupPolicy (Phase-1 training combines the current actor output, which
        arrives as data). ``controller`` defaults to the base set's certified
        controller.
        """
        x_arr = jnp.asarray(x)
        raw = jnp.asarray(raw_action, dtype=x_arr.dtype)
        ctrl = base_set.controller if controller is None else controller
        base = ctrl.action(x_arr)
        use_base = base_set.contains(x_arr)
        return jax.lax.select(use_base, base, raw)

    def action(self, x: jax.Array) -> jax.Array:
        """Handoff-base action: base controller inside B, safe-arrival outside."""
        x_arr = jnp.asarray(x)
        return self.select_action(x_arr, self.sa_action(x_arr), self.base_set, self.base_controller)


class ABP(BackupPolicy):
    """Analytic backup policy: hand-designed safe-arrival policy (ABP)."""


class LBP(BackupPolicy):
    """Learned backup policy: Phase-1 safe-arrival actor (LBP)."""


def save_learned_backup_policy(
    path: str | Path,
    *,
    actor_params: Any,
    actor_cfg: ActorConfig,
    action_scale: np.ndarray | jax.Array,
    action_low: np.ndarray | jax.Array,
    action_high: np.ndarray | jax.Array,
    metadata: Dict[str, Any] | None = None,
    metadata_before_bounds: bool = False,
) -> None:
    """Save a learned backup-policy checkpoint (shared by both systems)."""
    base = {
        "actor_params": actor_params,
        "actor_cfg": asdict(actor_cfg),
        "action_scale": np.asarray(action_scale, dtype=np.float32),
    }
    md = dict(metadata or {})
    bounds = {
        "action_low": np.asarray(action_low, dtype=np.float32),
        "action_high": np.asarray(action_high, dtype=np.float32),
    }
    payload = {**base, "metadata": md, **bounds} if metadata_before_bounds else {**base, **bounds, "metadata": md}
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


__all__ = ["ABP", "BackupPolicy", "LBP", "save_learned_backup_policy"]
