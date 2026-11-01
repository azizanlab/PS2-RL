"""Shared JAX replay buffer for the Phase-1 SA and Phase-2 PS2 trainers.

One circular uniform-sampling buffer, generic over its per-transition fields:
Phase 1 stores safe-arrival indicator/mask fields alongside the transition,
Phase 2 stores the plain SAC (obs, act, rew, next_obs, done) tuple. Field
layouts are fixed at init; add/sample use the same pointer arithmetic and
`jax.random.randint` draw as the previous in-module implementations, so
sampling is bit-identical for a given key.
"""

from __future__ import annotations

from typing import Dict, Mapping, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp


class JaxReplayState(NamedTuple):
    data: Dict[str, jax.Array]
    ptr: jax.Array
    size: jax.Array


def jax_replay_init(capacity: int, field_specs: Mapping[str, Sequence[int]]) -> JaxReplayState:
    """Allocate a zeroed buffer.

    ``field_specs`` maps field name -> trailing per-transition shape
    (``()`` for scalar fields). All fields are float32.
    """
    data = {
        name: jnp.zeros((int(capacity), *tuple(int(d) for d in shape)), dtype=jnp.float32)
        for name, shape in field_specs.items()
    }
    return JaxReplayState(data=data, ptr=jnp.int32(0), size=jnp.int32(0))


def jax_replay_add_batch(replay: JaxReplayState, values: Mapping[str, jax.Array]) -> JaxReplayState:
    """Insert a batch of transitions; ``values`` must cover every buffer field."""
    if set(values.keys()) != set(replay.data.keys()):
        raise KeyError(
            f"Replay add_batch fields {sorted(values.keys())} do not match buffer fields {sorted(replay.data.keys())}"
        )
    first = next(iter(values.values()))
    n = first.shape[0]
    cap = replay.data[next(iter(replay.data))].shape[0]
    idx = (replay.ptr + jnp.arange(n, dtype=jnp.int32)) % jnp.int32(cap)
    data = {name: replay.data[name].at[idx].set(values[name]) for name in replay.data}
    return JaxReplayState(
        data=data,
        ptr=(replay.ptr + jnp.int32(n)) % jnp.int32(cap),
        size=jnp.minimum(replay.size + jnp.int32(n), jnp.int32(cap)),
    )


def jax_replay_sample(
    replay: JaxReplayState,
    batch_size: int,
    key: jax.Array,
    field_names: Tuple[str, ...],
) -> Dict[str, jax.Array]:
    """Sample ``batch_size`` transitions uniformly; returns only ``field_names``."""
    upper = jnp.maximum(replay.size, jnp.int32(1))
    idx = jax.random.randint(key, (batch_size,), minval=0, maxval=upper, dtype=jnp.int32)
    return {name: replay.data[name][idx] for name in field_names}


__all__ = [
    "JaxReplayState",
    "jax_replay_add_batch",
    "jax_replay_init",
    "jax_replay_sample",
]
