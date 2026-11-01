"""SAC network primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class CriticConfig:
    obs_dim: int = 3
    act_dim: int = 2
    hidden_sizes: tuple[int, ...] = (256, 256)


def _init_linear(key: jax.Array, in_dim: int, out_dim: int, scale: float | None = None):
    if scale is None:
        scale = float(jnp.sqrt(2.0 / in_dim))
    w = scale * jax.random.normal(key, (in_dim, out_dim))
    b = jnp.zeros((out_dim,))
    return {"w": w, "b": b}


def init_mlp_params(key: jax.Array, sizes: Sequence[int], final_scale: float | None = None):
    keys = jax.random.split(key, len(sizes) - 1)
    layers = []
    n_layers = len(sizes) - 1
    for i, (k, n_in, n_out) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        scale = final_scale if i == n_layers - 1 else None
        layers.append(_init_linear(k, n_in, n_out, scale=scale))
    return {"layers": layers}


def mlp_forward(params, x: jax.Array) -> jax.Array:
    h = x
    n_layers = len(params["layers"])
    for i, layer in enumerate(params["layers"]):
        h = h @ layer["w"] + layer["b"]
        if i != n_layers - 1:
            h = jax.nn.relu(h)
    return h


def init_q_params(key: jax.Array, cfg: CriticConfig):
    sizes = [cfg.obs_dim + cfg.act_dim, *cfg.hidden_sizes, 1]
    return init_mlp_params(key, sizes, final_scale=1e-2)


def q_value(q_params, obs: jax.Array, act: jax.Array) -> jax.Array:
    q = mlp_forward(q_params, jnp.concatenate([obs, act], axis=-1))
    return jnp.squeeze(q, axis=-1)

