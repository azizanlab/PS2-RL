"""Safe-arrival critic helpers.

The critic is parameterized as a continuation value in ``[0, 1]`` via a
sigmoid output. We then wrap that continuation value with exact terminal
semantics:

  - goal states map to 1
  - failure states map to 0
  - continuation states map to the learned continuation value
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp

from ps2rl.utils.networks import init_mlp_params, mlp_forward


GoalContainsFn = Callable[[jax.Array], jax.Array]
FailContainsFn = Callable[[jax.Array], jax.Array]


@dataclass(frozen=True)
class SafeArrivalCriticConfig:
    obs_dim: int = 3
    act_dim: int = 2
    hidden_sizes: tuple[int, ...] = (256, 256)


def _ensure_hidden_sizes(hidden_sizes: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(v) for v in hidden_sizes)


def init_sa_q_params(key: jax.Array, cfg: SafeArrivalCriticConfig):
    sizes = [int(cfg.obs_dim) + int(cfg.act_dim), *_ensure_hidden_sizes(cfg.hidden_sizes), 1]
    return init_mlp_params(key, sizes, final_scale=1e-2)


def init_twin_q_params(key: jax.Array, cfg: SafeArrivalCriticConfig):
    k1, k2 = jax.random.split(key, 2)
    return init_sa_q_params(k1, cfg), init_sa_q_params(k2, cfg)


def q_cont(q_params, obs: jax.Array, act: jax.Array) -> jax.Array:
    raw_q = mlp_forward(q_params, jnp.concatenate([obs, act], axis=-1))
    raw_q = jnp.squeeze(raw_q, axis=-1)
    return jax.nn.sigmoid(raw_q)


def q_full_from_flags(
    q_cont_value: jax.Array,
    goal: jax.Array,
    fail: jax.Array,
) -> jax.Array:
    goal_f = goal.astype(q_cont_value.dtype)
    fail_f = fail.astype(q_cont_value.dtype)
    cont_f = jnp.clip(1.0 - goal_f - fail_f, 0.0, 1.0)
    q_value = goal_f + cont_f * q_cont_value
    return jnp.clip(q_value, 0.0, 1.0)


def q_full(
    q_params,
    obs: jax.Array,
    act: jax.Array,
    goal_contains_fn: GoalContainsFn,
    fail_contains_fn: FailContainsFn,
) -> jax.Array:
    q_cont_value = q_cont(q_params, obs, act)
    goal = goal_contains_fn(obs)
    fail = fail_contains_fn(obs)
    return q_full_from_flags(q_cont_value, goal=goal, fail=fail)


__all__ = [
    "FailContainsFn",
    "GoalContainsFn",
    "SafeArrivalCriticConfig",
    "init_sa_q_params",
    "init_twin_q_params",
    "q_cont",
    "q_full",
    "q_full_from_flags",
]
