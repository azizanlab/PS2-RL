"""Control-invariant layer: BCBF-QP projection entry points.

Policy flow:
  raw policy action (from NN) -> backup-CBF QP projection -> safe action.
"""

from __future__ import annotations

from typing import Callable, Dict, NamedTuple, Tuple

import jax

from ps2rl.utils.policy import (  # noqa: F401
    ActorConfig,
    actor_mean_action,
    init_actor_params,
    sample_actor_action,
)


class BCBFProjectionOps(NamedTuple):

    project_with_info: Callable[[jax.Array, jax.Array], Tuple[jax.Array, jax.Array, Dict[str, jax.Array]]]
    project: Callable[[jax.Array, jax.Array], jax.Array]
    backup_policy: Callable[[jax.Array], jax.Array]
