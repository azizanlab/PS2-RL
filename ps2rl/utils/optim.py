"""Small JAX optimizer utilities (Adam + Polyak averaging)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp


PyTree = Any


def adam_init(params: PyTree) -> Dict[str, PyTree]:
    """Initialize Adam optimizer state."""
    zeros_like = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros_like, "v": zeros_like, "t": jnp.array(0, dtype=jnp.int32)}


def adam_step(
    params: PyTree,
    grads: PyTree,
    state: Dict[str, PyTree],
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[PyTree, Dict[str, PyTree]]:
    """Apply one Adam update."""
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m_i, g_i: beta1 * m_i + (1.0 - beta1) * g_i, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v_i, g_i: beta2 * v_i + (1.0 - beta2) * (g_i * g_i), state["v"], grads)
    t_f = t.astype(jnp.float32)
    m_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta1**t_f), m)
    v_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta2**t_f), v)
    new_params = jax.tree_util.tree_map(
        lambda p_i, m_i, v_i: p_i - lr * m_i / (jnp.sqrt(v_i) + eps),
        params,
        m_hat,
        v_hat,
    )
    new_state = {"m": m, "v": v, "t": t}
    return new_params, new_state


def soft_update(target_params: PyTree, source_params: PyTree, tau: float) -> PyTree:
    """Polyak averaging: target <- (1-tau) target + tau source."""
    return jax.tree_util.tree_map(lambda t, s: (1.0 - tau) * t + tau * s, target_params, source_params)

