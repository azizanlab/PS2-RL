"""Dubin unicycle lane-keeping environment (JAX-native).

The environment config plus the batched JAX env used by Phase-2 training and
evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.utils.angles import wrap_angle
from ps2rl.utils.policy import clip_to_box

Array = jax.Array


def unicycle_control_affine_terms(x: Array) -> tuple[Array, Array]:
    """x=[Y,v,psi], u=[a,r], x_dot=f(x)+g(x)u."""
    _, v, psi = x
    f = jnp.array([v * jnp.sin(psi), 0.0, 0.0], dtype=x.dtype)
    g = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=x.dtype)
    return f, g


def unicycle_dynamics(x: Array, u: Array) -> Array:
    f, g = unicycle_control_affine_terms(x)
    return f + g @ u


def postprocess_unicycle_state(x: Array, *, v_min: float, v_max: float) -> Array:
    x_arr = jnp.asarray(x)
    v_next = jnp.clip(
        x_arr[1],
        jnp.asarray(v_min, dtype=x_arr.dtype),
        jnp.asarray(v_max, dtype=x_arr.dtype),
    )
    psi_next = wrap_angle(x_arr[2])
    return jnp.array([x_arr[0], v_next, psi_next], dtype=x_arr.dtype)


def unicycle_step_euler(x: Array, u: Array, *, dt: float, v_min: float, v_max: float) -> Array:
    x_arr = jnp.asarray(x)
    u_arr = jnp.asarray(u, dtype=x_arr.dtype)
    dx = unicycle_dynamics(x_arr, u_arr)
    x_next = x_arr + jnp.asarray(dt, dtype=x_arr.dtype) * dx
    return postprocess_unicycle_state(x_next, v_min=v_min, v_max=v_max)


@dataclass(frozen=True)
class UnicycleEnvConfig:
    """Environment and reward config."""

    dt: float = 0.05
    max_steps: int | None = None

    # Safety bounds
    y_max: float = 1.8
    psi_max: float = np.pi / 3.0

    # Dynamics/action bounds
    a_max: float = 5.0
    r_max: float = 0.5
    v_min: float = 0.0
    v_max: float = 12.0

    # Nominal objective
    v_des: float = 5.0
    reward_v_des: float | None = None

    # Initial state ranges around [0, v_des, 0]
    init_y_range: float = 0.5
    init_v_range: float = 1.5
    init_psi_range: float = 0.20

    # Optional disturbances (for robustness tests) - NOT used!
    d1_max: float = 0.0
    d2_max: float = 0.0
    d3_max: float = 0.0

    if d1_max > 0.0 or d2_max > 0.0 or d3_max > 0.0:
        raise ValueError("Disturbances are currently not supported (d1_max, d2_max, d3_max must be 0.0)")

    # Reward shaping
    reward_mode: str = "trajectory_following"
    w_v: float = 1.0
    w_lane_y: float = 0.3
    w_lane_psi: float = 0.3
    w_control: float = 0.01
    # Trajectory-following targets
    traj_y_amplitude: float = 2.5
    traj_y_period: float = 10.0
    traj_y_phase: float = 0.0
    traj_v_mean: float = 5.0
    traj_v_amplitude: float = 0.0
    traj_v_period: float = 10.0
    traj_v_phase: float = 0.0
    traj_normalize_reward: bool = False
    traj_speed_err_scale: float = 5.0

    terminate_on_violation: bool = True

    def __post_init__(self) -> None:
        dt = float(self.dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be a positive finite value, got {self.dt}")

        max_steps = self.max_steps
        if max_steps is None:
            # Default to a 20-second episode horizon in wall-clock time.
            max_steps = int(np.ceil(20.0 / dt))
        max_steps = int(max_steps)
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")

        reward_v_des = self.v_des if self.reward_v_des is None else float(self.reward_v_des)
        if not np.isfinite(reward_v_des):
            raise ValueError(f"reward_v_des must be finite, got {self.reward_v_des}")
        traj_y_period = float(self.traj_y_period)
        traj_v_period = float(self.traj_v_period)
        traj_speed_err_scale = float(self.traj_speed_err_scale)
        if not np.isfinite(traj_y_period) or traj_y_period <= 0.0:
            raise ValueError(f"traj_y_period must be a positive finite value, got {self.traj_y_period}")
        if not np.isfinite(traj_v_period) or traj_v_period <= 0.0:
            raise ValueError(f"traj_v_period must be a positive finite value, got {self.traj_v_period}")
        if not np.isfinite(traj_speed_err_scale) or traj_speed_err_scale <= 0.0:
            raise ValueError(
                f"traj_speed_err_scale must be a positive finite value, got {self.traj_speed_err_scale}"
            )
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "max_steps", max_steps)
        object.__setattr__(self, "reward_v_des", reward_v_des)
        object.__setattr__(self, "traj_y_period", traj_y_period)
        object.__setattr__(self, "traj_v_period", traj_v_period)
        object.__setattr__(self, "traj_speed_err_scale", traj_speed_err_scale)


class UnicycleReference(NamedTuple):
    time_sec: Array
    y_ref: Array
    v_ref: Array
    psi_ref: Array
    psi_ref_feasible: Array
    traj_phase_y_sin: Array
    traj_phase_y_cos: Array
    traj_phase_v_sin: Array
    traj_phase_v_cos: Array


class UnicycleEnvState(NamedTuple):
    x: Array
    steps: Array
    ep_return: Array
    ep_len: Array
    ep_safe_sum: Array
    ep_speed_error_sum: Array
    ep_y_error_sum: Array
    ep_psi_error_sum: Array


class UnicycleStepInfo(NamedTuple):
    is_safe: Array
    speed_error: Array
    speed_error_abs: Array
    y_error: Array
    y_error_abs: Array
    psi_error: Array
    psi_error_abs: Array
    ref_time_sec: Array
    y_ref: Array
    v_ref: Array
    psi_ref: Array
    psi_ref_feasible: Array
    traj_phase_y_sin: Array
    traj_phase_y_cos: Array
    traj_phase_v_sin: Array
    traj_phase_v_cos: Array
    lane_margin: Array
    violation_y: Array
    violation_psi: Array
    episode_done: Array
    completed_return: Array
    completed_len: Array
    completed_safe_rate: Array
    completed_speed_error_abs: Array
    completed_y_error_abs: Array
    completed_psi_error_abs: Array


class UnicycleEnvFns(NamedTuple):
    obs_dim: int
    action_dim: int
    action_scale: Array
    action_low: Array
    action_high: Array
    reset: Callable[[Array], tuple[UnicycleEnvState, Array]]
    step: Callable[
        [UnicycleEnvState, Array, Array],
        tuple[UnicycleEnvState, Array, Array, Array, Array, UnicycleStepInfo],
    ]
    reset_batched: Callable[[Array], tuple[UnicycleEnvState, Array]]
    step_batched: Callable[
        [UnicycleEnvState, Array, Array],
        tuple[UnicycleEnvState, Array, Array, Array, Array, UnicycleStepInfo],
    ]


def build_unicycle_env(cfg: UnicycleEnvConfig, dtype=jnp.float32) -> UnicycleEnvFns:
    reward_mode = str(cfg.reward_mode).strip().lower()
    use_trajectory = reward_mode == "trajectory_following"
    obs_dim = 10 if use_trajectory else 3

    dt = jnp.asarray(cfg.dt, dtype=dtype)
    max_steps_i32 = jnp.int32(cfg.max_steps)
    y_max = jnp.asarray(cfg.y_max, dtype=dtype)
    psi_max = jnp.asarray(cfg.psi_max, dtype=dtype)
    a_max = jnp.asarray(cfg.a_max, dtype=dtype)
    r_max = jnp.asarray(cfg.r_max, dtype=dtype)
    v_min = jnp.asarray(cfg.v_min, dtype=dtype)
    v_max = jnp.asarray(cfg.v_max, dtype=dtype)
    v_des = jnp.asarray(cfg.v_des, dtype=dtype)
    reward_v_des = jnp.asarray(cfg.reward_v_des, dtype=dtype)

    init_y_range = jnp.asarray(cfg.init_y_range, dtype=dtype)
    init_v_range = jnp.asarray(cfg.init_v_range, dtype=dtype)
    init_psi_range = jnp.asarray(cfg.init_psi_range, dtype=dtype)

    terminate_on_violation = jnp.asarray(bool(cfg.terminate_on_violation), dtype=jnp.bool_)
    w_v = jnp.asarray(cfg.w_v, dtype=dtype)
    w_lane_y = jnp.asarray(cfg.w_lane_y, dtype=dtype)
    w_lane_psi = jnp.asarray(cfg.w_lane_psi, dtype=dtype)
    w_control = jnp.asarray(cfg.w_control, dtype=dtype)
    traj_y_amplitude = jnp.asarray(cfg.traj_y_amplitude, dtype=dtype)
    traj_y_period = jnp.asarray(cfg.traj_y_period, dtype=dtype)
    traj_y_phase = jnp.asarray(cfg.traj_y_phase, dtype=dtype)
    traj_v_mean = jnp.asarray(cfg.traj_v_mean, dtype=dtype)
    traj_v_amplitude = jnp.asarray(cfg.traj_v_amplitude, dtype=dtype)
    traj_v_period = jnp.asarray(cfg.traj_v_period, dtype=dtype)
    traj_v_phase = jnp.asarray(cfg.traj_v_phase, dtype=dtype)
    traj_normalize_reward = bool(cfg.traj_normalize_reward)
    traj_speed_err_scale = jnp.asarray(cfg.traj_speed_err_scale, dtype=dtype)

    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)

    action_low = jnp.asarray([-cfg.a_max, -cfg.r_max], dtype=dtype)
    action_high = jnp.asarray([cfg.a_max, cfg.r_max], dtype=dtype)
    action_scale = action_high

    def _wrap_angle(theta: Array) -> Array:
        return wrap_angle(jnp.asarray(theta, dtype=dtype))

    def _psi_from_y_dot_over_v(y_dot_ref: Array, v_ref_now: Array) -> tuple[Array, Array]:
        v_abs = jnp.maximum(jnp.abs(v_ref_now), jnp.asarray(1e-6, dtype=dtype))
        psi_arg_raw = y_dot_ref / v_abs
        psi_arg = jnp.clip(psi_arg_raw, -one, one)
        feasible = (jnp.abs(psi_arg_raw) <= one).astype(dtype)
        return jnp.arcsin(psi_arg), feasible

    def _reference(step_idx: Array) -> UnicycleReference:
        step_idx = jnp.asarray(step_idx, dtype=jnp.int32)
        t = step_idx.astype(dtype) * dt
        omega_y = 2.0 * jnp.asarray(jnp.pi, dtype=dtype) / traj_y_period
        y_phase = omega_y * t + traj_y_phase
        omega_v = 2.0 * jnp.asarray(jnp.pi, dtype=dtype) / traj_v_period
        v_phase = omega_v * t + traj_v_phase
        phase_y_sin = jnp.sin(y_phase)
        phase_y_cos = jnp.cos(y_phase)
        phase_v_sin = jnp.sin(v_phase)
        phase_v_cos = jnp.cos(v_phase)

        if not use_trajectory:
            return UnicycleReference(
                time_sec=t,
                y_ref=zero,
                v_ref=reward_v_des,
                psi_ref=zero,
                psi_ref_feasible=one,
                traj_phase_y_sin=phase_y_sin,
                traj_phase_y_cos=phase_y_cos,
                traj_phase_v_sin=phase_v_sin,
                traj_phase_v_cos=phase_v_cos,
            )

        y_ref = traj_y_amplitude * jnp.sin(y_phase)
        y_dot_ref = traj_y_amplitude * omega_y * jnp.cos(y_phase)
        v_ref = traj_v_mean + traj_v_amplitude * jnp.sin(v_phase)
        psi_ref, psi_ref_feasible = _psi_from_y_dot_over_v(y_dot_ref, v_ref)

        return UnicycleReference(
            time_sec=t,
            y_ref=y_ref,
            v_ref=v_ref,
            psi_ref=psi_ref,
            psi_ref_feasible=psi_ref_feasible,
            traj_phase_y_sin=phase_y_sin,
            traj_phase_y_cos=phase_y_cos,
            traj_phase_v_sin=phase_v_sin,
            traj_phase_v_cos=phase_v_cos,
        )

    def _observation(x: Array, ref: UnicycleReference) -> Array:
        x = jnp.asarray(x, dtype=dtype)
        if not use_trajectory:
            return x
        return jnp.asarray(
            [
                x[0],
                x[1],
                x[2],
                ref.y_ref,
                ref.v_ref,
                ref.psi_ref,
                ref.traj_phase_y_sin,
                ref.traj_phase_y_cos,
                ref.traj_phase_v_sin,
                ref.traj_phase_v_cos,
            ],
            dtype=dtype,
        )

    def _sample_reset_state(key: Array) -> Array:
        key_y, key_v, key_psi = jax.random.split(key, 3)
        y0 = jax.random.uniform(key_y, shape=(), minval=-init_y_range, maxval=init_y_range, dtype=dtype)
        v0 = v_des + jax.random.uniform(key_v, shape=(), minval=-init_v_range, maxval=init_v_range, dtype=dtype)
        v0 = jnp.clip(v0, v_min, v_max)
        psi0 = jax.random.uniform(key_psi, shape=(), minval=-init_psi_range, maxval=init_psi_range, dtype=dtype)
        return jnp.asarray([y0, v0, psi0], dtype=dtype)

    def _init_episode_state(x0: Array) -> UnicycleEnvState:
        return UnicycleEnvState(
            x=jnp.asarray(x0, dtype=dtype),
            steps=jnp.int32(0),
            ep_return=zero,
            ep_len=jnp.int32(0),
            ep_safe_sum=zero,
            ep_speed_error_sum=zero,
            ep_y_error_sum=zero,
            ep_psi_error_sum=zero,
        )

    def _reset_impl(key: Array) -> tuple[UnicycleEnvState, Array]:
        x0 = _sample_reset_state(key)
        ref0 = _reference(jnp.int32(0))
        state0 = _init_episode_state(x0)
        return state0, _observation(x0, ref0)

    def env_reset(key: Array) -> tuple[UnicycleEnvState, Array]:
        return _reset_impl(key)

    def _is_safe(x: Array) -> Array:
        return (jnp.abs(x[0]) <= y_max) & (jnp.abs(x[2]) <= psi_max)

    def _reward(
        x_next: Array,
        u: Array,
        *,
        speed_err: Array,
        y_err: Array,
        psi_err: Array,
    ) -> Array:
        control_cost = jnp.sum(jnp.square(u))

        if reward_mode == "trajectory_following":
            if traj_normalize_reward:
                norm_v_err = jnp.square(speed_err / traj_speed_err_scale)
                norm_y_err = jnp.square(y_err / y_max)
                norm_psi_err = jnp.square(psi_err / psi_max)
                norm_control_cost = jnp.square(u[0] / a_max) + jnp.square(u[1] / r_max)
                reward = (
                    -w_v * norm_v_err
                    - w_lane_y * norm_y_err
                    - w_lane_psi * norm_psi_err
                    - w_control * norm_control_cost
                )
            else:
                reward = (
                    -w_v * jnp.square(speed_err)
                    - w_lane_y * jnp.square(y_err)
                    - w_lane_psi * jnp.square(psi_err)
                    - w_control * control_cost
                )
        else:
            raise ValueError(f"Unsupported reward_mode: {cfg.reward_mode}")

        return jnp.asarray(reward, dtype=dtype)

    def env_step(state: UnicycleEnvState, action: Array, key: Array):
        u = clip_to_box(jnp.asarray(action, dtype=dtype), action_low, action_high)
        x = jnp.asarray(state.x, dtype=dtype)

        y_next = x[0] + dt * x[1] * jnp.sin(x[2])
        v_next = jnp.clip(x[1] + dt * u[0], v_min, v_max)
        psi_next = _wrap_angle(x[2] + dt * u[1])
        x_next = jnp.asarray([y_next, v_next, psi_next], dtype=dtype)

        next_steps = state.steps + jnp.int32(1)
        safe = _is_safe(x_next)
        done = (next_steps >= max_steps_i32) | (terminate_on_violation & (~safe))

        ref = _reference(next_steps)
        speed_err = x_next[1] - ref.v_ref
        y_err = x_next[0] - ref.y_ref
        psi_err = _wrap_angle(x_next[2] - ref.psi_ref)
        reward = _reward(
            x_next,
            u,
            speed_err=speed_err,
            y_err=y_err,
            psi_err=psi_err,
        )

        speed_err_abs = jnp.abs(speed_err)
        y_err_abs = jnp.abs(y_err)
        psi_err_abs = jnp.abs(psi_err)
        lane_margin = jnp.minimum(y_max - jnp.abs(x_next[0]), psi_max - jnp.abs(x_next[2]))

        ep_len = state.ep_len + jnp.int32(1)
        ep_return = state.ep_return + reward
        ep_safe_sum = state.ep_safe_sum + safe.astype(dtype)
        ep_speed_error_sum = state.ep_speed_error_sum + speed_err_abs
        ep_y_error_sum = state.ep_y_error_sum + y_err_abs
        ep_psi_error_sum = state.ep_psi_error_sum + psi_err_abs
        ep_len_f = jnp.maximum(ep_len.astype(dtype), one)

        completed_return = jnp.where(done, ep_return, zero)
        completed_len = jnp.where(done, ep_len.astype(dtype), zero)
        completed_safe_rate = jnp.where(done, ep_safe_sum / ep_len_f, zero)
        completed_speed_error_abs = jnp.where(done, ep_speed_error_sum / ep_len_f, zero)
        completed_y_error_abs = jnp.where(done, ep_y_error_sum / ep_len_f, zero)
        completed_psi_error_abs = jnp.where(done, ep_psi_error_sum / ep_len_f, zero)

        continued_state = UnicycleEnvState(
            x=x_next,
            steps=next_steps,
            ep_return=ep_return,
            ep_len=ep_len,
            ep_safe_sum=ep_safe_sum,
            ep_speed_error_sum=ep_speed_error_sum,
            ep_y_error_sum=ep_y_error_sum,
            ep_psi_error_sum=ep_psi_error_sum,
        )
        reset_state, reset_obs = _reset_impl(key)
        state_out = jax.tree_util.tree_map(
            lambda reset_v, continue_v: jnp.where(done, reset_v, continue_v),
            reset_state,
            continued_state,
        )

        next_obs_true = _observation(x_next, ref)
        next_obs_out = jnp.where(done, reset_obs, next_obs_true)
        info = UnicycleStepInfo(
            is_safe=safe.astype(dtype),
            speed_error=speed_err,
            speed_error_abs=speed_err_abs,
            y_error=y_err,
            y_error_abs=y_err_abs,
            psi_error=psi_err,
            psi_error_abs=psi_err_abs,
            ref_time_sec=ref.time_sec,
            y_ref=ref.y_ref,
            v_ref=ref.v_ref,
            psi_ref=ref.psi_ref,
            psi_ref_feasible=ref.psi_ref_feasible,
            traj_phase_y_sin=ref.traj_phase_y_sin,
            traj_phase_y_cos=ref.traj_phase_y_cos,
            traj_phase_v_sin=ref.traj_phase_v_sin,
            traj_phase_v_cos=ref.traj_phase_v_cos,
            lane_margin=lane_margin,
            violation_y=jnp.maximum(zero, jnp.abs(x_next[0]) - y_max),
            violation_psi=jnp.maximum(zero, jnp.abs(x_next[2]) - psi_max),
            episode_done=done,
            completed_return=completed_return,
            completed_len=completed_len,
            completed_safe_rate=completed_safe_rate,
            completed_speed_error_abs=completed_speed_error_abs,
            completed_y_error_abs=completed_y_error_abs,
            completed_psi_error_abs=completed_psi_error_abs,
        )
        return state_out, next_obs_true, next_obs_out, reward, done, info

    reset = jax.jit(env_reset)
    step = jax.jit(env_step)
    reset_batched = jax.jit(jax.vmap(reset))
    step_batched = jax.jit(jax.vmap(step))
    return UnicycleEnvFns(
        obs_dim=obs_dim,
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
    "UnicycleEnvConfig",
    "UnicycleEnvFns",
    "UnicycleEnvState",
    "UnicycleReference",
    "UnicycleStepInfo",
    "build_unicycle_env",
]
