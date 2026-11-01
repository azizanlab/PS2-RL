"""Generic tanh-Gaussian actor (MLP init/apply, boxed sample/mean actions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ActorConfig:
    """Actor network and distribution hyper-parameters."""

    obs_dim: int = 3
    action_dim: int = 2
    hidden_sizes: Tuple[int, ...] = (256, 256)
    log_std_min: float = -5.0
    log_std_max: float = 2.0


def _coerce_actor_cfg(cfg_payload: Any) -> ActorConfig:
    """Rebuild an ``ActorConfig`` from a saved payload (dataclass or dict).
    """
    if isinstance(cfg_payload, ActorConfig):
        return cfg_payload
    if not isinstance(cfg_payload, dict):
        raise TypeError(f"Unsupported actor_cfg payload type: {type(cfg_payload)}")
    cfg_dict = dict(cfg_payload)
    if "hidden_sizes" in cfg_dict:
        cfg_dict["hidden_sizes"] = tuple(cfg_dict["hidden_sizes"])
    return ActorConfig(**cfg_dict)


def clip_to_box(u: jax.Array, low: Any, high: Any) -> jax.Array:
    """Clip ``u`` to the per-dimension box ``[low, high]``."""
    u = jnp.asarray(u)
    return jnp.clip(u, jnp.asarray(low, dtype=u.dtype), jnp.asarray(high, dtype=u.dtype))


def _init_linear(key: jax.Array, in_dim: int, out_dim: int, scale: float | None = None):
    if scale is None:
        scale = float(jnp.sqrt(2.0 / in_dim))
    w = scale * jax.random.normal(key, (in_dim, out_dim))
    b = jnp.zeros((out_dim,))
    return {"w": w, "b": b}


def _init_mlp(key: jax.Array, sizes: Sequence[int]):
    keys = jax.random.split(key, len(sizes) - 1)
    layers = []
    for i, (k, n_in, n_out) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        # small final layer init for actor stability
        scale = 1e-2 if i == len(sizes) - 2 else None
        layers.append(_init_linear(k, n_in, n_out, scale=scale))
    return {"layers": layers}


def _mlp_apply(params, x: jax.Array) -> jax.Array:
    h = x
    n_layers = len(params["layers"])
    for i, layer in enumerate(params["layers"]):
        h = h @ layer["w"] + layer["b"]
        if i != n_layers - 1:
            h = jax.nn.relu(h)
    return h


def init_actor_params(key: jax.Array, cfg: ActorConfig):
    sizes = [cfg.obs_dim, *cfg.hidden_sizes, 2 * cfg.action_dim]
    return _init_mlp(key, sizes)


def _actor_dist_params(params, obs: jax.Array, cfg: ActorConfig) -> Tuple[jax.Array, jax.Array]:
    out = _mlp_apply(params, obs)
    mean, log_std_raw = jnp.split(out, 2, axis=-1)
    # Smooth clamp of log std.
    t = jnp.tanh(log_std_raw)
    log_std = cfg.log_std_min + 0.5 * (cfg.log_std_max - cfg.log_std_min) * (t + 1.0)
    return mean, log_std


def _normal_log_prob(x: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    pre_sum = -0.5 * (((x - mean) / jnp.exp(log_std)) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    return jnp.sum(pre_sum, axis=-1)


def _resolve_action_box(
    action_scale: jax.Array,
    *,
    action_low: jax.Array | None = None,
    action_high: jax.Array | None = None,
) -> Tuple[jax.Array, jax.Array]:
    if (action_low is None) != (action_high is None):
        raise ValueError("action_low and action_high must be provided together")
    if action_low is None:
        action_low = -action_scale
        action_high = action_scale
    else:
        action_low = jnp.asarray(action_low, dtype=action_scale.dtype)
        action_high = jnp.asarray(action_high, dtype=action_scale.dtype)
    action_mid = 0.5 * (action_high + action_low)
    action_half_range = 0.5 * (action_high - action_low)
    return action_mid, action_half_range


def _stable_log_one_minus_tanh_sq(pre_tanh: jax.Array) -> jax.Array:
    log_two = jnp.log(jnp.asarray(2.0, dtype=pre_tanh.dtype))
    return 2.0 * (log_two - pre_tanh - jax.nn.softplus(-2.0 * pre_tanh))


def sample_actor_action(
    params,
    obs: jax.Array,
    key: jax.Array,
    action_scale: jax.Array,
    cfg: ActorConfig,
    *,
    action_low: jax.Array | None = None,
    action_high: jax.Array | None = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Sample Gaussian action squashed into a per-dimension box.

    Returns:
      action_raw_scaled, log_prob, mean_action_scaled
    """
    mean, log_std = _actor_dist_params(params, obs, cfg)
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, shape=mean.shape)
    pre_tanh = mean + std * eps
    action_mid, action_half_range = _resolve_action_box(
        action_scale,
        action_low=action_low,
        action_high=action_high,
    )
    tanh_action = jnp.tanh(pre_tanh)
    action = action_mid + action_half_range * tanh_action

    # SAC log-prob with stable affine-tanh correction.
    log_prob = _normal_log_prob(pre_tanh, mean, log_std)
    log_det = jnp.log(jnp.clip(action_half_range, min=1e-6)) + _stable_log_one_minus_tanh_sq(pre_tanh)
    log_prob -= jnp.sum(log_det, axis=-1)
    mean_action = action_mid + action_half_range * jnp.tanh(mean)
    return action, log_prob, mean_action


def actor_mean_action(
    params,
    obs: jax.Array,
    action_scale: jax.Array,
    cfg: ActorConfig,
    *,
    action_low: jax.Array | None = None,
    action_high: jax.Array | None = None,
) -> jax.Array:
    mean, _ = _actor_dist_params(params, obs, cfg)
    action_mid, action_half_range = _resolve_action_box(
        action_scale,
        action_low=action_low,
        action_high=action_high,
    )
    return action_mid + action_half_range * jnp.tanh(mean)


__all__ = [
    "ActorConfig",
    "actor_mean_action",
    "clip_to_box",
    "init_actor_params",
    "sample_actor_action",
]
