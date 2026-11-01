"""Pure-JAX batched environment for quadrotor safe-arrival phase-1 training.

The key safe-arrival collector requirement is that replay stores the physical
successor state before any auto-reset occurs. We therefore return both:

  - ``next_obs_true``: successor before reset, used for replay and Bellman targets
  - ``next_obs_out``: collector observation after reset-if-done, used to keep scans running
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.base_controller.quadrotor_dlqr import QuadrotorDLQR
from ps2rl.sets.base_sets import EllipsoidBaseSet
from ps2rl.backup_policy.backup_policy import BackupPolicy
from ps2rl.utils.policy import clip_to_box
from ps2rl.sets.quadrotor_sets import QuadrotorSafeSet
from ps2rl.utils.quaternion import (
    normalize_quaternion_batch,
    quaternion_from_euler_zyx_batch,
    quaternion_multiply_batch,
)
from ps2rl.evaluation.quadrotor_trace_reset_lib import QuadrotorResetLibrary
from ps2rl.cil.quadrotor_backup_cbf import (
    QuadrotorBCBFConfig,
    hard_deck_value,
    quadrotor_step_euler,
)

Array = jax.Array

_POOL_NAMES = ("general_trace", "near_ceiling", "bridge", "base_shell")


@dataclass(frozen=True)
class QuadrotorSAEnvConfig:
    cbf_cfg: QuadrotorBCBFConfig

    terminate_on_crash: bool = True

    @property
    def dt(self) -> float:
        return float(self.cbf_cfg.dt)

    @property
    def horizon_steps(self) -> int:
        return int(self.cbf_cfg.num_steps)

    @property
    def horizon_T(self) -> float:
        return float(self.cbf_cfg.T)


def quadrotor_backup_env_config_from_dict(payload: dict[str, Any]) -> QuadrotorSAEnvConfig:
    cbf_valid = {f.name for f in fields(QuadrotorBCBFConfig) if f.init}
    cbf_kwargs = {k: v for k, v in payload["cbf_cfg"].items() if k in cbf_valid}
    cbf_cfg = QuadrotorBCBFConfig(**cbf_kwargs)
    return QuadrotorSAEnvConfig(
        cbf_cfg=cbf_cfg,
        terminate_on_crash=bool(payload.get("terminate_on_crash", True)),
    )


class QuadrotorSAEnvState(NamedTuple):
    x: Array
    steps: Array
    ep_len: Array
    ep_safe_sum: Array
    ep_capture_sum: Array
    ep_terminal_sum: Array
    ep_min_hard_deck_margin: Array
    ep_safe_rollout: Array
    ep_entered_terminal: Array
    ep_left_after_entry: Array
    ep_entry_step: Array


class QuadrotorSAStepInfo(NamedTuple):
    is_safe: Array
    is_capture: Array
    is_terminal: Array
    goal_next: Array
    fail_next: Array
    is_success: Array
    is_crash: Array
    horizon_reached: Array
    raw_action: Array
    applied_action: Array
    episode_done: Array
    completed_len: Array
    completed_safe_rate: Array
    completed_capture_rate: Array
    completed_terminal_rate: Array
    completed_safe_rollout: Array
    completed_terminal_at_horizon: Array
    completed_entered_terminal: Array
    completed_invariance_after_entry: Array
    completed_entry_step: Array
    completed_min_hard_deck_margin: Array


class QuadrotorSAEnvFns(NamedTuple):
    obs_dim: int
    action_dim: int
    reset: Callable[[Array, Array], tuple[QuadrotorSAEnvState, Array]]
    step: Callable[
        [QuadrotorSAEnvState, Array, Array, Array],
        tuple[QuadrotorSAEnvState, Array, Array, Array, QuadrotorSAStepInfo],
    ]
    reset_batched: Callable[[Array, Array], tuple[QuadrotorSAEnvState, Array]]
    step_batched: Callable[
        [QuadrotorSAEnvState, Array, Array, Array],
        tuple[QuadrotorSAEnvState, Array, Array, Array, QuadrotorSAStepInfo],
    ]


def build_quadrotor_sa_env(
    cfg: QuadrotorSAEnvConfig,
    reset_library: QuadrotorResetLibrary,
    *,
    split: str = "train",
    terminate_on_goal: bool = True,
    dtype=jnp.float32,
) -> QuadrotorSAEnvFns:
    split = str(split)
    safe_set = QuadrotorSafeSet.from_cbf_config(cfg.cbf_cfg)
    base_controller = QuadrotorDLQR.from_config(cfg.cbf_cfg)
    base_set = EllipsoidBaseSet(
        base_controller,
        float(cfg.cbf_cfg.base_set_c),
        smooth_gain=float(cfg.cbf_cfg.base_set_smooth_gain),
    )

    horizon_steps = jnp.int32(cfg.horizon_steps)
    terminate_on_crash = jnp.asarray(cfg.terminate_on_crash, dtype=jnp.bool_)
    terminate_on_goal_j = jnp.asarray(bool(terminate_on_goal), dtype=jnp.bool_)
    inf = jnp.asarray(jnp.inf, dtype=dtype)

    action_low = jnp.asarray(
        [
            cfg.cbf_cfg.a_cmd_min,
            -cfg.cbf_cfg.omega_max,
            -cfg.cbf_cfg.omega_max,
            -cfg.cbf_cfg.omega_max,
        ],
        dtype=dtype,
    )
    action_high = jnp.asarray(
        [
            cfg.cbf_cfg.a_cmd_max,
            cfg.cbf_cfg.omega_max,
            cfg.cbf_cfg.omega_max,
            cfg.cbf_cfg.omega_max,
        ],
        dtype=dtype,
    )

    low_mix = jnp.asarray(
        [
            reset_library.library_cfg.mix_general_low,
            reset_library.library_cfg.mix_near_ceiling_low,
            reset_library.library_cfg.mix_bridge_low,
            reset_library.library_cfg.mix_base_shell_low,
        ],
        dtype=dtype,
    )
    high_mix = jnp.asarray(
        [
            reset_library.library_cfg.mix_general_high,
            reset_library.library_cfg.mix_near_ceiling_high,
            reset_library.library_cfg.mix_bridge_high,
            reset_library.library_cfg.mix_base_shell_high,
        ],
        dtype=dtype,
    )
    region_mult = jnp.asarray(
        [
            reset_library.library_cfg.general_region_multiplier,
            reset_library.library_cfg.near_ceiling_region_multiplier,
            reset_library.library_cfg.bridge_region_multiplier,
            reset_library.library_cfg.base_shell_region_multiplier,
        ],
        dtype=dtype,
    )

    pos_min = float(reset_library.library_cfg.position_perturb_min)
    pos_max = float(reset_library.library_cfg.position_perturb_max)
    vel_min = float(reset_library.library_cfg.velocity_perturb_min)
    vel_max = float(reset_library.library_cfg.velocity_perturb_max)
    tilt_min = float(np.deg2rad(reset_library.library_cfg.tilt_perturb_deg_min))
    tilt_max = float(np.deg2rad(reset_library.library_cfg.tilt_perturb_deg_max))
    yaw_min = float(np.deg2rad(reset_library.library_cfg.yaw_perturb_deg_min))
    yaw_max = float(np.deg2rad(reset_library.library_cfg.yaw_perturb_deg_max))
    max_resample_tries = int(max(1, reset_library.library_cfg.max_resample_tries))

    split_pools = reset_library.split_pools.get(split, {})
    max_pool_size = max(
        1,
        max((int(np.asarray(split_pools.get(pool_name, np.zeros((0, 10)))).shape[0]) for pool_name in _POOL_NAMES), default=1),
    )
    pool_data = np.zeros((len(_POOL_NAMES), max_pool_size, 10), dtype=np.float32)
    pool_sizes = np.zeros((len(_POOL_NAMES),), dtype=np.int32)
    for pool_idx, pool_name in enumerate(_POOL_NAMES):
        states = np.asarray(split_pools.get(pool_name, np.zeros((0, 10), dtype=np.float64)), dtype=np.float32)
        n = int(states.shape[0])
        pool_sizes[pool_idx] = n
        if n > 0:
            pool_data[pool_idx, :n] = states
    pool_states = jnp.asarray(pool_data, dtype=dtype)
    pool_sizes_j = jnp.asarray(pool_sizes, dtype=jnp.int32)

    def _clip_action(action: Array) -> Array:
        return clip_to_box(jnp.asarray(action, dtype=dtype), action_low, action_high)

    def _curriculum_weights(curriculum_scale: Array) -> Array:
        s = jnp.clip(jnp.asarray(curriculum_scale, dtype=dtype), 0.0, 1.0)
        raw = (1.0 - s) * low_mix + s * high_mix
        masked = jnp.where(pool_sizes_j > 0, raw, 0.0)
        norm = jnp.sum(masked)
        fallback = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=dtype)
        return jnp.where(norm > 0.0, masked / norm, fallback)

    def _perturb_ranges(curriculum_scale: Array, pool_idx: Array) -> tuple[Array, Array, Array, Array]:
        s = jnp.clip(jnp.asarray(curriculum_scale, dtype=dtype), 0.0, 1.0)
        mult = region_mult[pool_idx]
        pos = mult * ((1.0 - s) * pos_min + s * pos_max)
        vel = mult * ((1.0 - s) * vel_min + s * vel_max)
        tilt = mult * ((1.0 - s) * tilt_min + s * tilt_max)
        yaw = mult * ((1.0 - s) * yaw_min + s * yaw_max)
        return pos, vel, tilt, yaw

    def _sample_reset_state(key: Array, curriculum_scale: Array) -> Array:
        key_pool, key_index, key_try = jax.random.split(key, 3)
        logits = jnp.log(jnp.maximum(_curriculum_weights(curriculum_scale), jnp.asarray(1e-12, dtype=dtype)))
        pool_idx = jax.random.categorical(key_pool, logits).astype(jnp.int32)
        pool_size = jnp.maximum(pool_sizes_j[pool_idx], jnp.int32(1))
        pool_row = pool_states[pool_idx]
        base_idx = jax.random.randint(key_index, shape=(), minval=0, maxval=pool_size, dtype=jnp.int32)
        base = pool_row[base_idx]
        base = base.at[6:10].set(normalize_quaternion_batch(base[6:10]))
        pos_rng, vel_rng, tilt_rng, yaw_rng = _perturb_ranges(curriculum_scale, pool_idx)

        def body(_, carry):
            found, best_x, try_key = carry
            try_key, key_pos, key_vel, key_att = jax.random.split(try_key, 4)
            pos_delta = jax.random.uniform(key_pos, shape=(3,), minval=-pos_rng, maxval=pos_rng, dtype=dtype)
            vel_delta = jax.random.uniform(key_vel, shape=(3,), minval=-vel_rng, maxval=vel_rng, dtype=dtype)
            att_delta = jax.random.uniform(
                key_att,
                shape=(3,),
                minval=jnp.asarray([-tilt_rng, -tilt_rng, -yaw_rng], dtype=dtype),
                maxval=jnp.asarray([tilt_rng, tilt_rng, yaw_rng], dtype=dtype),
                dtype=dtype,
            )
            q_delta = quaternion_from_euler_zyx_batch(att_delta[0], att_delta[1], att_delta[2])
            candidate = base.at[0:3].add(pos_delta).at[3:6].add(vel_delta)
            candidate = candidate.at[6:10].set(
                normalize_quaternion_batch(quaternion_multiply_batch(q_delta, base[6:10]))
            )
            safe = safe_set.contains(candidate)
            take = (~found) & safe
            best_x = jnp.where(take, candidate, best_x)
            found = found | safe
            return found, best_x, try_key

        found, chosen, _ = jax.lax.fori_loop(
            0,
            max_resample_tries,
            body,
            (
                jnp.asarray(False, dtype=jnp.bool_),
                base,
                key_try,
            ),
        )
        chosen = chosen.at[6:10].set(normalize_quaternion_batch(chosen[6:10]))
        return jax.lax.select(found, chosen, base)

    def env_reset(key: Array, curriculum_scale: Array) -> tuple[QuadrotorSAEnvState, Array]:
        x0 = _sample_reset_state(key, curriculum_scale)
        state = QuadrotorSAEnvState(
            x=x0,
            steps=jnp.int32(0),
            ep_len=jnp.int32(0),
            ep_safe_sum=jnp.asarray(0.0, dtype=dtype),
            ep_capture_sum=jnp.asarray(0.0, dtype=dtype),
            ep_terminal_sum=jnp.asarray(0.0, dtype=dtype),
            ep_min_hard_deck_margin=inf,
            ep_safe_rollout=jnp.asarray(True, dtype=jnp.bool_),
            ep_entered_terminal=jnp.asarray(False, dtype=jnp.bool_),
            ep_left_after_entry=jnp.asarray(False, dtype=jnp.bool_),
            ep_entry_step=jnp.int32(-1),
        )
        return state, state.x

    def env_step(
        state: QuadrotorSAEnvState,
        raw_action: Array,
        key: Array,
        curriculum_scale: Array,
    ):
        raw_action = _clip_action(raw_action)
        applied_action = _clip_action(
            BackupPolicy.select_action(state.x, raw_action, base_set)
        )
        next_obs_true = quadrotor_step_euler(state.x, applied_action, cfg.cbf_cfg)
        next_obs_true = next_obs_true.at[6:10].set(normalize_quaternion_batch(next_obs_true[6:10]))

        safe = safe_set.contains(next_obs_true)
        capture = base_set.contains(next_obs_true)
        terminal = base_set.contains(next_obs_true)
        fail_next = ~safe
        goal_next = terminal

        next_steps = state.steps + jnp.int32(1)
        horizon_reached = next_steps >= horizon_steps
        crash_done = terminate_on_crash & fail_next
        goal_done = terminate_on_goal_j & goal_next
        done_rollout = crash_done | goal_done | horizon_reached

        ep_len = state.ep_len + jnp.int32(1)
        ep_safe_sum = state.ep_safe_sum + safe.astype(dtype)
        ep_capture_sum = state.ep_capture_sum + capture.astype(dtype)
        ep_terminal_sum = state.ep_terminal_sum + terminal.astype(dtype)
        hard_margin = hard_deck_value(next_obs_true, cfg.cbf_cfg)
        ep_min_hard = jnp.minimum(state.ep_min_hard_deck_margin, hard_margin)
        ep_safe_rollout = state.ep_safe_rollout & safe
        ep_entered_terminal = state.ep_entered_terminal | goal_next
        ep_left_after_entry = state.ep_left_after_entry | (state.ep_entered_terminal & (~terminal))
        ep_entry_step = jnp.where((~state.ep_entered_terminal) & goal_next, next_steps, state.ep_entry_step)

        ep_len_f = jnp.maximum(ep_len.astype(dtype), jnp.asarray(1.0, dtype=dtype))
        completed_safe_rate = jnp.where(done_rollout, ep_safe_sum / ep_len_f, jnp.asarray(0.0, dtype=dtype))
        completed_capture_rate = jnp.where(done_rollout, ep_capture_sum / ep_len_f, jnp.asarray(0.0, dtype=dtype))
        completed_terminal_rate = jnp.where(done_rollout, ep_terminal_sum / ep_len_f, jnp.asarray(0.0, dtype=dtype))
        completed_min_hard = jnp.where(done_rollout, ep_min_hard, jnp.asarray(0.0, dtype=dtype))
        completed_safe_rollout = jnp.where(done_rollout, ep_safe_rollout.astype(dtype), jnp.asarray(0.0, dtype=dtype))
        completed_entered_terminal = jnp.where(done_rollout, ep_entered_terminal.astype(dtype), jnp.asarray(0.0, dtype=dtype))
        completed_invariance_after_entry = jnp.where(
            done_rollout,
            (ep_entered_terminal & (~ep_left_after_entry)).astype(dtype),
            jnp.asarray(0.0, dtype=dtype),
        )
        completed_entry_step = jnp.where(done_rollout & ep_entered_terminal, ep_entry_step, jnp.int32(-1))
        completed_terminal_at_horizon = jnp.where(
            done_rollout,
            (horizon_reached & terminal).astype(dtype),
            jnp.asarray(0.0, dtype=dtype),
        )

        reset_x = _sample_reset_state(key, curriculum_scale)
        next_obs_out = jnp.where(done_rollout, reset_x, next_obs_true)

        zero_f = jnp.asarray(0.0, dtype=dtype)
        state_out = QuadrotorSAEnvState(
            x=next_obs_out,
            steps=jnp.where(done_rollout, jnp.int32(0), next_steps),
            ep_len=jnp.where(done_rollout, jnp.int32(0), ep_len),
            ep_safe_sum=jnp.where(done_rollout, zero_f, ep_safe_sum),
            ep_capture_sum=jnp.where(done_rollout, zero_f, ep_capture_sum),
            ep_terminal_sum=jnp.where(done_rollout, zero_f, ep_terminal_sum),
            ep_min_hard_deck_margin=jnp.where(done_rollout, inf, ep_min_hard),
            ep_safe_rollout=jnp.where(done_rollout, jnp.asarray(True, dtype=jnp.bool_), ep_safe_rollout),
            ep_entered_terminal=jnp.where(done_rollout, jnp.asarray(False, dtype=jnp.bool_), ep_entered_terminal),
            ep_left_after_entry=jnp.where(done_rollout, jnp.asarray(False, dtype=jnp.bool_), ep_left_after_entry),
            ep_entry_step=jnp.where(done_rollout, jnp.int32(-1), ep_entry_step),
        )

        info = QuadrotorSAStepInfo(
            is_safe=safe.astype(dtype),
            is_capture=capture.astype(dtype),
            is_terminal=terminal.astype(dtype),
            goal_next=goal_next.astype(dtype),
            fail_next=fail_next.astype(dtype),
            is_success=goal_next.astype(dtype),
            is_crash=fail_next.astype(dtype),
            horizon_reached=horizon_reached.astype(dtype),
            raw_action=raw_action.astype(dtype),
            applied_action=applied_action.astype(dtype),
            episode_done=done_rollout,
            completed_len=jnp.where(done_rollout, ep_len.astype(dtype), zero_f),
            completed_safe_rate=completed_safe_rate.astype(dtype),
            completed_capture_rate=completed_capture_rate.astype(dtype),
            completed_terminal_rate=completed_terminal_rate.astype(dtype),
            completed_safe_rollout=completed_safe_rollout.astype(dtype),
            completed_terminal_at_horizon=completed_terminal_at_horizon.astype(dtype),
            completed_entered_terminal=completed_entered_terminal.astype(dtype),
            completed_invariance_after_entry=completed_invariance_after_entry.astype(dtype),
            completed_entry_step=completed_entry_step,
            completed_min_hard_deck_margin=completed_min_hard.astype(dtype),
        )
        return state_out, next_obs_true, next_obs_out, done_rollout, info

    reset = jax.jit(env_reset)
    step = jax.jit(env_step)
    reset_batched = jax.jit(jax.vmap(reset, in_axes=(0, None)))
    step_batched = jax.jit(jax.vmap(step, in_axes=(0, 0, 0, None)))
    return QuadrotorSAEnvFns(
        obs_dim=10,
        action_dim=4,
        reset=reset,
        step=step,
        reset_batched=reset_batched,
        step_batched=step_batched,
    )


__all__ = [
    "QuadrotorSAEnvFns",
    "QuadrotorSAEnvState",
    "QuadrotorSAStepInfo",
    "build_quadrotor_sa_env",
]
