"""Pure-JAX batched environment for lane safe-arrival phase-1 training.

The collector must keep the physical successor state before any auto-reset.
We therefore return both:

  - ``next_obs_true``: successor before reset, used for replay and Bellman targets
  - ``next_obs_out``: collector observation after reset-if-done, used to keep scans running
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from ps2rl.base_controller.unicycle_dlqr import UnicycleDLQR
from ps2rl.backup_policy.backup_policy import BackupPolicy
from ps2rl.backup_policy.unicycle_analytic_backup import clip_backup_action
from ps2rl.utils.policy import clip_to_box
from ps2rl.sets.base_sets import EllipsoidBaseSet
from ps2rl.sets.unicycle_sets import UnicycleSafeSet
from ps2rl.cil.unicycle_backup_cbf import (
    unicycle_step_euler as _lane_step_euler_shared,
    validate_unicycle_base_config,
)


Array = jax.Array


@dataclass(frozen=True)
class UnicycleSAEnvConfig:
    dt: float = 0.05
    horizon_steps: int = 40

    a_max: float = 5.0
    r_max: float = 1.0
    v_min: float = 0.0
    v_max: float = 12.0
    v_des: float = 5.0

    y_max: float = 1.8
    psi_max: float = pi / 3.0

    # Base set B.
    base_set_c: float = 0.3
    lqr_q_y: float = 1.0
    lqr_q_v: float = 1.0
    lqr_q_psi: float = 1.0
    lqr_r_a: float = 1.0
    lqr_r_r: float = 1.0

    init_y_range_min: float = 0.08
    init_y_range_max: float = 1.5
    init_v_range_min: float = 0.10
    init_v_range_max: float = 3.0
    init_psi_range_min: float = 0.03
    init_psi_range_max: float = 0.50

    terminate_on_crash: bool = True

    # Reset rejection-sampling bound.
    max_resample_tries: int = 10000

    def __post_init__(self) -> None:
        dt = float(self.dt)
        if (not isfinite(dt)) or dt <= 0.0:
            raise ValueError(f"dt must be a positive finite value, got {self.dt}")
        horizon_steps = int(self.horizon_steps)
        if horizon_steps <= 0:
            raise ValueError(f"horizon_steps must be positive, got {self.horizon_steps}")
        max_resample_tries = int(self.max_resample_tries)
        if max_resample_tries <= 0:
            raise ValueError(f"max_resample_tries must be positive, got {self.max_resample_tries}")
        if float(self.a_max) <= 0.0 or float(self.r_max) <= 0.0:
            raise ValueError("a_max and r_max must be positive")
        if float(self.v_max) <= float(self.v_min):
            raise ValueError(f"v_max must exceed v_min, got [{self.v_min}, {self.v_max}]")
        if not (float(self.v_min) <= float(self.v_des) <= float(self.v_max)):
            raise ValueError(f"v_des must lie in [v_min, v_max], got {self.v_des}")

        def _validate_range(name_min: str, lo: float, name_max: str, hi: float, *, upper: float | None = None) -> None:
            lo = float(lo)
            hi = float(hi)
            if (not isfinite(lo)) or lo < 0.0:
                raise ValueError(f"{name_min} must be nonnegative and finite, got {lo}")
            if (not isfinite(hi)) or hi <= 0.0:
                raise ValueError(f"{name_max} must be positive and finite, got {hi}")
            if hi < lo:
                raise ValueError(f"{name_max} ({hi}) must be >= {name_min} ({lo})")
            if upper is not None and hi > float(upper):
                raise ValueError(f"{name_max} ({hi}) must be <= {upper}")

        _validate_range("init_y_range_min", self.init_y_range_min, "init_y_range_max", self.init_y_range_max, upper=self.y_max)
        _validate_range(
            "init_v_range_min",
            self.init_v_range_min,
            "init_v_range_max",
            self.init_v_range_max,
            upper=max(float(self.v_des - self.v_min), float(self.v_max - self.v_des)),
        )
        _validate_range(
            "init_psi_range_min",
            self.init_psi_range_min,
            "init_psi_range_max",
            self.init_psi_range_max,
            upper=self.psi_max,
        )
        validate_unicycle_base_config(
            y_max=self.y_max,
            psi_max=self.psi_max,
            dt=self.dt,
            a_max=self.a_max,
            r_max=self.r_max,
            base_set_c=self.base_set_c,
            v_des=self.v_des,
            lqr_q_y=self.lqr_q_y,
            lqr_q_v=self.lqr_q_v,
            lqr_q_psi=self.lqr_q_psi,
            lqr_r_a=self.lqr_r_a,
            lqr_r_r=self.lqr_r_r,
        )

        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "horizon_steps", horizon_steps)
        object.__setattr__(self, "max_resample_tries", max_resample_tries)

    @property
    def horizon_T(self) -> float:
        return float(self.dt) * float(self.horizon_steps)


def unicycle_sa_env_config_from_dict(payload: dict) -> UnicycleSAEnvConfig:
    valid = UnicycleSAEnvConfig.__dataclass_fields__
    return UnicycleSAEnvConfig(**{k: v for k, v in dict(payload).items() if k in valid})


class UnicycleSAEnvState(NamedTuple):
    x: Array
    steps: Array
    ep_len: Array
    ep_safe_sum: Array
    ep_capture_sum: Array
    ep_terminal_sum: Array
    ep_safe_rollout: Array
    ep_entered_capture: Array
    ep_entered_terminal: Array
    ep_left_after_terminal_entry: Array
    ep_capture_entry_step: Array
    ep_terminal_entry_step: Array


class UnicycleSAStepInfo(NamedTuple):
    is_safe: Array
    is_capture: Array
    is_terminal: Array
    goal_next: Array
    fail_next: Array
    is_crash: Array
    horizon_reached: Array
    raw_action: Array
    applied_action: Array
    episode_done: Array
    completed_len: Array
    completed_safe_rate: Array
    completed_capture_rate: Array
    completed_capture_success: Array
    completed_terminal_rate: Array
    completed_safe_rollout: Array
    completed_terminal_at_horizon: Array
    completed_entered_terminal: Array
    completed_invariance_after_terminal_entry: Array
    completed_capture_entry_step: Array
    completed_terminal_entry_step: Array


class UnicycleSAEnvFns(NamedTuple):
    obs_dim: int
    action_dim: int
    action_scale: Array
    action_low: Array
    action_high: Array
    reset: Callable[[Array, Array], tuple[UnicycleSAEnvState, Array]]
    step: Callable[
        [UnicycleSAEnvState, Array, Array, Array],
        tuple[UnicycleSAEnvState, Array, Array, Array, UnicycleSAStepInfo],
    ]
    reset_batched: Callable[[Array, Array], tuple[UnicycleSAEnvState, Array]]
    step_batched: Callable[
        [UnicycleSAEnvState, Array, Array, Array],
        tuple[UnicycleSAEnvState, Array, Array, Array, UnicycleSAStepInfo],
    ]


def unicycle_step_euler(x: Array, u: Array, cfg: UnicycleSAEnvConfig) -> Array:
    return _lane_step_euler_shared(
        x,
        u,
        dt=cfg.dt,
        v_min=cfg.v_min,
        v_max=cfg.v_max,
    )


def build_unicycle_sa_env(
    cfg: UnicycleSAEnvConfig,
    *,
    terminate_on_goal: bool = True,
    dtype=jnp.float32,
) -> UnicycleSAEnvFns:
    safe_set = UnicycleSafeSet(y_max=cfg.y_max, psi_max=cfg.psi_max)
    lqr_design = UnicycleDLQR.from_config(cfg)
    capture_set = EllipsoidBaseSet(lqr_design, float(cfg.base_set_c))
    terminal_set = capture_set

    action_low = jnp.asarray([-cfg.a_max, -cfg.r_max], dtype=dtype)
    action_high = jnp.asarray([cfg.a_max, cfg.r_max], dtype=dtype)
    action_scale = action_high
    terminate_on_crash = jnp.asarray(bool(cfg.terminate_on_crash), dtype=jnp.bool_)
    terminate_on_goal_j = jnp.asarray(bool(terminate_on_goal), dtype=jnp.bool_)
    horizon_steps = jnp.int32(cfg.horizon_steps)

    def _clip_action(action: Array) -> Array:
        return clip_to_box(jnp.asarray(action, dtype=dtype), action_low, action_high)

    def _curriculum_range(min_value: float, max_value: float, curriculum_scale: Array) -> Array:
        s = jnp.clip(jnp.asarray(curriculum_scale, dtype=dtype), 0.0, 1.0)
        return (1.0 - s) * jnp.asarray(min_value, dtype=dtype) + s * jnp.asarray(max_value, dtype=dtype)

    def _sample_reset_state_raw(key: Array, curriculum_scale: Array) -> Array:
        key_y, key_v, key_psi = jax.random.split(key, 3)
        y_range = _curriculum_range(cfg.init_y_range_min, cfg.init_y_range_max, curriculum_scale)
        v_range = _curriculum_range(cfg.init_v_range_min, cfg.init_v_range_max, curriculum_scale)
        psi_range = _curriculum_range(cfg.init_psi_range_min, cfg.init_psi_range_max, curriculum_scale)
        y = jax.random.uniform(key_y, shape=(), minval=-y_range, maxval=y_range, dtype=dtype)
        v = jnp.asarray(cfg.v_des, dtype=dtype) + jax.random.uniform(
            key_v,
            shape=(),
            minval=-v_range,
            maxval=v_range,
            dtype=dtype,
        )
        v = jnp.clip(v, jnp.asarray(cfg.v_min, dtype=dtype), jnp.asarray(cfg.v_max, dtype=dtype))
        psi = jax.random.uniform(key_psi, shape=(), minval=-psi_range, maxval=psi_range, dtype=dtype)
        return jnp.array([y, v, psi], dtype=dtype)

    max_resample_tries = jnp.int32(cfg.max_resample_tries)

    def _sample_reset_state(key: Array, curriculum_scale: Array) -> Array:
        key_loop, key_sample = jax.random.split(key)
        x0 = _sample_reset_state_raw(key_sample, curriculum_scale)
        inside_capture0 = capture_set.contains(x0)

        # Bounded rejection sampling: resample while the candidate is inside the
        # base set, but stop after max_resample_tries so a degenerate reset region
        # (every candidate inside the base set, e.g. curriculum_start_scale too small)
        # fails via the trainer's post-reset check instead of looping forever.
        def cond(carry):
            _, _, inside_capture, tries = carry
            return inside_capture & (tries < max_resample_tries)

        def body(carry):
            key_i, _, _, tries = carry
            key_i, key_sample_i = jax.random.split(key_i)
            x_i = _sample_reset_state_raw(key_sample_i, curriculum_scale)
            inside_capture_i = capture_set.contains(x_i)
            return key_i, x_i, inside_capture_i, tries + jnp.int32(1)

        _, x0, _, _ = jax.lax.while_loop(
            cond, body, (key_loop, x0, inside_capture0, jnp.int32(0))
        )
        return x0

    def _init_episode_state(x0: Array) -> UnicycleSAEnvState:
        capture0 = capture_set.contains(x0)
        terminal0 = terminal_set.contains(x0)
        return UnicycleSAEnvState(
            x=x0,
            steps=jnp.int32(0),
            ep_len=jnp.int32(0),
            ep_safe_sum=jnp.asarray(0.0, dtype=dtype),
            ep_capture_sum=jnp.asarray(0.0, dtype=dtype),
            ep_terminal_sum=jnp.asarray(0.0, dtype=dtype),
            ep_safe_rollout=jnp.asarray(True, dtype=jnp.bool_),
            ep_entered_capture=capture0,
            ep_entered_terminal=terminal0,
            ep_left_after_terminal_entry=jnp.asarray(False, dtype=jnp.bool_),
            ep_capture_entry_step=jnp.where(capture0, jnp.int32(0), jnp.int32(-1)),
            ep_terminal_entry_step=jnp.where(terminal0, jnp.int32(0), jnp.int32(-1)),
        )

    def env_reset(key: Array, curriculum_scale: Array) -> tuple[UnicycleSAEnvState, Array]:
        x0 = _sample_reset_state(key, curriculum_scale)
        state = _init_episode_state(x0)
        return state, state.x

    def env_step(
        state: UnicycleSAEnvState,
        raw_action: Array,
        key: Array,
        curriculum_scale: Array,
    ):
        raw_action = _clip_action(raw_action)
        applied_action = _clip_action(
            clip_backup_action(
                BackupPolicy.select_action(
                    state.x,
                    clip_backup_action(raw_action, a_max=cfg.a_max, r_max=cfg.r_max),
                    capture_set,
                    controller=lqr_design,
                ),
                a_max=cfg.a_max,
                r_max=cfg.r_max,
            )
        )

        next_obs_true = unicycle_step_euler(state.x, applied_action, cfg)
        safe = safe_set.contains(next_obs_true)
        capture = capture_set.contains(next_obs_true)
        terminal = terminal_set.contains(next_obs_true)
        goal_next = terminal
        fail_next = ~safe

        next_steps = state.steps + jnp.int32(1)
        horizon_reached = next_steps >= horizon_steps
        crash_done = terminate_on_crash & fail_next
        goal_done = terminate_on_goal_j & goal_next
        done_rollout = crash_done | goal_done | horizon_reached

        ep_len = state.ep_len + jnp.int32(1)
        ep_safe_sum = state.ep_safe_sum + safe.astype(dtype)
        ep_capture_sum = state.ep_capture_sum + capture.astype(dtype)
        ep_terminal_sum = state.ep_terminal_sum + terminal.astype(dtype)
        ep_safe_rollout = state.ep_safe_rollout & safe
        ep_entered_capture = state.ep_entered_capture | capture
        ep_entered_terminal = state.ep_entered_terminal | goal_next
        ep_left_after_terminal_entry = state.ep_left_after_terminal_entry | (state.ep_entered_terminal & (~terminal))
        ep_capture_entry_step = jnp.where((~state.ep_entered_capture) & capture, next_steps, state.ep_capture_entry_step)
        ep_terminal_entry_step = jnp.where((~state.ep_entered_terminal) & goal_next, next_steps, state.ep_terminal_entry_step)

        ep_len_f = jnp.maximum(ep_len.astype(dtype), jnp.asarray(1.0, dtype=dtype))
        completed_safe_rate = jnp.where(done_rollout, ep_safe_sum / ep_len_f, jnp.asarray(0.0, dtype=dtype))
        completed_capture_rate = jnp.where(done_rollout, ep_capture_sum / ep_len_f, jnp.asarray(0.0, dtype=dtype))
        completed_terminal_rate = jnp.where(done_rollout, ep_terminal_sum / ep_len_f, jnp.asarray(0.0, dtype=dtype))
        completed_capture_success = jnp.where(done_rollout, ep_entered_capture.astype(dtype), jnp.asarray(0.0, dtype=dtype))
        completed_safe_rollout = jnp.where(done_rollout, ep_safe_rollout.astype(dtype), jnp.asarray(0.0, dtype=dtype))
        completed_entered_terminal = jnp.where(done_rollout, ep_entered_terminal.astype(dtype), jnp.asarray(0.0, dtype=dtype))
        completed_invariance_after_terminal_entry = jnp.where(
            done_rollout,
            (ep_entered_terminal & (~ep_left_after_terminal_entry)).astype(dtype),
            jnp.asarray(0.0, dtype=dtype),
        )
        completed_terminal_at_horizon = jnp.where(
            done_rollout,
            (horizon_reached & ep_safe_rollout & terminal).astype(dtype),
            jnp.asarray(0.0, dtype=dtype),
        )
        completed_capture_entry_step = jnp.where(done_rollout, ep_capture_entry_step, jnp.int32(-1))
        completed_terminal_entry_step = jnp.where(done_rollout, ep_terminal_entry_step, jnp.int32(-1))

        zero_f = jnp.asarray(0.0, dtype=dtype)
        continued_state = UnicycleSAEnvState(
            x=next_obs_true,
            steps=next_steps,
            ep_len=ep_len,
            ep_safe_sum=ep_safe_sum,
            ep_capture_sum=ep_capture_sum,
            ep_terminal_sum=ep_terminal_sum,
            ep_safe_rollout=ep_safe_rollout,
            ep_entered_capture=ep_entered_capture,
            ep_entered_terminal=ep_entered_terminal,
            ep_left_after_terminal_entry=ep_left_after_terminal_entry,
            ep_capture_entry_step=ep_capture_entry_step,
            ep_terminal_entry_step=ep_terminal_entry_step,
        )
        reset_state = _init_episode_state(_sample_reset_state(key, curriculum_scale))
        state_out = jax.tree_util.tree_map(
            lambda reset_v, continue_v: jnp.where(done_rollout, reset_v, continue_v),
            reset_state,
            continued_state,
        )
        next_obs_out = state_out.x

        info = UnicycleSAStepInfo(
            is_safe=safe.astype(dtype),
            is_capture=capture.astype(dtype),
            is_terminal=terminal.astype(dtype),
            goal_next=goal_next.astype(dtype),
            fail_next=fail_next.astype(dtype),
            is_crash=fail_next.astype(dtype),
            horizon_reached=horizon_reached.astype(dtype),
            raw_action=raw_action.astype(dtype),
            applied_action=applied_action.astype(dtype),
            episode_done=done_rollout,
            completed_len=jnp.where(done_rollout, ep_len.astype(dtype), zero_f),
            completed_safe_rate=completed_safe_rate.astype(dtype),
            completed_capture_rate=completed_capture_rate.astype(dtype),
            completed_capture_success=completed_capture_success.astype(dtype),
            completed_terminal_rate=completed_terminal_rate.astype(dtype),
            completed_safe_rollout=completed_safe_rollout.astype(dtype),
            completed_terminal_at_horizon=completed_terminal_at_horizon.astype(dtype),
            completed_entered_terminal=completed_entered_terminal.astype(dtype),
            completed_invariance_after_terminal_entry=completed_invariance_after_terminal_entry.astype(dtype),
            completed_capture_entry_step=completed_capture_entry_step,
            completed_terminal_entry_step=completed_terminal_entry_step,
        )
        return state_out, next_obs_true, next_obs_out, done_rollout, info

    reset = jax.jit(env_reset)
    step = jax.jit(env_step)
    reset_batched = jax.jit(jax.vmap(reset, in_axes=(0, None)))
    step_batched = jax.jit(jax.vmap(step, in_axes=(0, 0, 0, None)))
    return UnicycleSAEnvFns(
        obs_dim=3,
        action_dim=2,
        action_scale=action_scale,
        action_low=action_low,
        action_high=action_high,
        reset=reset,
        step=step,
        reset_batched=reset_batched,
        step_batched=step_batched,
    )


__all__ = [
    "UnicycleSAEnvConfig",
    "UnicycleSAEnvFns",
    "UnicycleSAEnvState",
    "UnicycleSAStepInfo",
    "build_unicycle_sa_env",
    "unicycle_sa_env_config_from_dict",
    "unicycle_step_euler",
]
