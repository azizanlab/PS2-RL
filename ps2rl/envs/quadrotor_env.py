"""Agile quadrotor powerloop environment (JAX-native).

The environment config, reference-bundle loading, and the batched JAX env used
by Phase-2 training and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.utils.quaternion import (
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_from_euler_zyx,
    quaternion_multiply,
    quaternion_rate_matrix,
    thrust_axis_world,
)

from ps2rl.envs.assets.utils import (
    _default_reference_path,
    _load_reference_bundle,
    _load_reference_states,
)

Array = jax.Array


def quadrotor_clip_action(u: Array, a_cmd_min, a_cmd_max, omega_max) -> Array:
    """Per-element clip to the action box, reassembled with ``jnp.array``."""
    u = jnp.asarray(u)
    u0 = jnp.clip(u[0], a_cmd_min, a_cmd_max)
    u_omega = jnp.clip(u[1:4], -omega_max, omega_max)
    return jnp.array([u0, u_omega[0], u_omega[1], u_omega[2]], dtype=u.dtype)


def quadrotor_control_affine_terms(x: Array, gravity) -> tuple[Array, Array]:
    """x_dot = f(x) + g(x)u for u=[a_cmd, omega_x, omega_y, omega_z]."""
    x = jnp.asarray(x)
    q = normalize_quaternion(x[6:10])
    xi = quaternion_rate_matrix(q)
    thrust_axis = thrust_axis_world(q)

    f = jnp.zeros((10,), dtype=x.dtype).at[0:3].set(x[3:6]).at[5].set(-gravity)
    g = jnp.zeros((10, 4), dtype=x.dtype).at[3:6, 0].set(thrust_axis).at[6:10, 1:4].set(0.5 * xi)
    return f, g


def quadrotor_dynamics(x: Array, u: Array, gravity, a_cmd_min, a_cmd_max, omega_max) -> Array:
    x = jnp.asarray(x)
    u = quadrotor_clip_action(u, a_cmd_min, a_cmd_max, omega_max)
    f, g = quadrotor_control_affine_terms(x, gravity)
    return f + g @ u


def quadrotor_step_euler(x: Array, u: Array, dt, gravity, a_cmd_min, a_cmd_max, omega_max) -> Array:
    x = jnp.asarray(x)
    x_next = x + jnp.asarray(dt, dtype=x.dtype) * quadrotor_dynamics(
        x, u, gravity, a_cmd_min, a_cmd_max, omega_max
    )
    return x_next.at[6:10].set(normalize_quaternion(x_next[6:10]))


@dataclass(frozen=True)
class QuadrotorEnvConfig:
    """Environment and reward config for the 10D quadrotor model."""

    dt: float = 0.02
    max_steps: int | None = None
    max_steps_extra_sec: float = 0.1  # 0.1 is used by only the vanilla tracker.
    backup_horizon_T: float = 0.0

    # Dynamics and action bounds.
    gravity: float = 9.81
    a_cmd_min: float = 0.0
    a_cmd_max: float | None = None  # Defaults to 4g.
    omega_max: float = 18.0

    # Safety constraints.
    z_max: float = 3.0
    terminate_on_violation: bool = False

    # Nominal objective.
    reward_mode: str = "trajectory_following"
    reference_path: str = ""
    reference_dt: float | None = None
    include_time_features: bool = True

    z_des: float = 2.0

    # Reward: -(tracking cost + control cost).
    w_pos_xy: float = 1.0
    w_pos_z: float = 2.0
    w_vel: float = 0.2
    w_att: float = 1.0
    w_ref_omega_x: float = 0.0
    w_ref_omega_y: float = 0.0
    w_ref_omega_z: float = 0.0
    w_control_a: float = 0.01
    w_control_omega: float = 0.01
    # Initial-state sampling around reference.
    init_px_range: float = 0.1
    init_py_range: float = 0.1
    init_pz_range: float = 0.1
    init_v_range: float = 0.0
    init_tilt_deg_range: float = 0.0
    init_yaw_deg_range: float = 0.0

    def __post_init__(self) -> None:
        dt = float(self.dt)
        if (not isfinite(dt)) or dt <= 0.0:
            raise ValueError(f"dt must be a positive finite value, got {self.dt}")

        gravity = float(self.gravity)
        if (not isfinite(gravity)) or gravity <= 0.0:
            raise ValueError(f"gravity must be a positive finite value, got {self.gravity}")

        a_cmd_min = float(self.a_cmd_min)
        a_cmd_max = float(4.0 * gravity) if self.a_cmd_max is None else float(self.a_cmd_max)
        if a_cmd_min < 0.0:
            raise ValueError(f"a_cmd_min must be nonnegative, got {self.a_cmd_min}")
        if a_cmd_max <= a_cmd_min:
            raise ValueError(f"a_cmd_max must be greater than a_cmd_min, got [{a_cmd_min}, {a_cmd_max}]")

        omega_max = float(self.omega_max)
        if (not isfinite(omega_max)) or omega_max <= 0.0:
            raise ValueError(f"omega_max must be a positive finite value, got {self.omega_max}")

        z_max = float(self.z_max)
        z_des = float(self.z_des)
        if (not isfinite(z_max)) or (not isfinite(z_des)):
            raise ValueError("z_max and z_des must be finite values")
        if z_max <= 0.0:
            raise ValueError(f"z_max must be positive for hard-deck mode, got {z_max}")
        if z_des >= z_max:
            raise ValueError(f"z_des must be less than z_max for hard-deck mode, got z_des={z_des}, z_max={z_max}")

        reward_mode = str(self.reward_mode).strip().lower()
        if reward_mode != "trajectory_following":
            raise ValueError(
                f"reward_mode must be 'trajectory_following' (the hover mode was removed in Batch 6b-4), got {self.reward_mode}"
            )

        reference_path = self.reference_path.strip() if self.reference_path.strip() else _default_reference_path()
        ref_dt = dt if self.reference_dt is None else float(self.reference_dt)
        if (not isfinite(ref_dt)) or ref_dt <= 0.0:
            raise ValueError(f"reference_dt must be a positive finite value, got {self.reference_dt}")

        max_steps = self.max_steps
        if max_steps is None:
            ref_states = _load_reference_states(reference_path)
            # -------------------- fixed ref. length for the max steps ------------------- #
            # This should ONLY be used when reference dt == env dt
            ref_steps = int(ref_states.shape[0])
            tail_steps = int(np.ceil(self.max_steps_extra_sec/dt))
            max_steps = ref_steps + tail_steps

        max_steps = int(max_steps)
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")

        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "max_steps", max_steps)
        object.__setattr__(self, "gravity", gravity)
        object.__setattr__(self, "a_cmd_min", a_cmd_min)
        object.__setattr__(self, "a_cmd_max", a_cmd_max)
        object.__setattr__(self, "omega_max", omega_max)
        object.__setattr__(self, "z_max", z_max)
        object.__setattr__(self, "z_des", z_des)
        object.__setattr__(self, "reward_mode", reward_mode)
        object.__setattr__(self, "reference_path", reference_path)
        object.__setattr__(self, "reference_dt", ref_dt)


class QuadrotorEnvState(NamedTuple):
    x: Array
    steps: Array
    ep_return: Array
    ep_len: Array
    ep_safe_sum: Array
    ep_pos_err_sum: Array
    ep_vel_err_sum: Array
    ep_att_err_sum: Array
    ep_hard_deck_margin_min: Array


class QuadrotorStepInfo(NamedTuple):
    is_safe: Array
    pos_error_norm: Array
    vel_error_norm: Array
    att_error_norm: Array
    omega_ref_error_norm: Array
    hard_deck_margin: Array
    ref_progress: Array
    ref_time_sec: Array
    ref_state: Array  # (10,) reference [p, v, q] at the post-step index
    ref_omega: Array  # (3,) reference body rates at the post-step index
    episode_done: Array
    completed_return: Array
    completed_len: Array
    completed_safe_rate: Array
    completed_pos_error_norm: Array
    completed_vel_error_norm: Array
    completed_att_error_norm: Array
    completed_hard_deck_margin_min: Array


class QuadrotorEnvFns(NamedTuple):
    obs_dim: int
    action_dim: int
    reset: Callable[[Array], tuple[QuadrotorEnvState, Array]]
    step: Callable[
        [QuadrotorEnvState, Array, Array],
        tuple[QuadrotorEnvState, Array, Array, Array, Array, QuadrotorStepInfo],
    ]
    reset_batched: Callable[[Array], tuple[QuadrotorEnvState, Array]]
    step_batched: Callable[
        [QuadrotorEnvState, Array, Array],
        tuple[QuadrotorEnvState, Array, Array, Array, Array, QuadrotorStepInfo],
    ]


def build_quadrotor_env(cfg: QuadrotorEnvConfig, dtype=jnp.float32) -> QuadrotorEnvFns:
    include_time_features = bool(cfg.include_time_features)

    ref_bundle = _load_reference_bundle(cfg.reference_path)
    ref_states_np = ref_bundle["states"].astype(np.float32)
    ref_omega_cmd_np = ref_bundle["omega_cmd"].astype(np.float32)

    ref_states = jnp.asarray(ref_states_np, dtype=dtype)
    ref_omega_cmd = jnp.asarray(ref_omega_cmd_np, dtype=dtype)
    ref_last_i32 = jnp.int32(ref_states.shape[0] - 1)
    ref_last_f = jnp.asarray(max(ref_states.shape[0] - 1, 1), dtype=dtype)

    dt = jnp.asarray(cfg.dt, dtype=dtype)
    ref_dt = jnp.asarray(cfg.reference_dt, dtype=dtype)

    gravity = jnp.asarray(cfg.gravity, dtype=dtype)
    a_cmd_min = jnp.asarray(cfg.a_cmd_min, dtype=dtype)
    a_cmd_max = jnp.asarray(cfg.a_cmd_max, dtype=dtype)
    omega_max = jnp.asarray(cfg.omega_max, dtype=dtype)

    max_steps_i32 = jnp.int32(cfg.max_steps)
    terminate_on_violation = jnp.asarray(cfg.terminate_on_violation, dtype=jnp.bool_)

    z_max = jnp.asarray(cfg.z_max, dtype=dtype)


    w_pos_xy = jnp.asarray(cfg.w_pos_xy, dtype=dtype)
    w_pos_z = jnp.asarray(cfg.w_pos_z, dtype=dtype)
    w_vel = jnp.asarray(cfg.w_vel, dtype=dtype)
    w_att = jnp.asarray(cfg.w_att, dtype=dtype)
    w_ref_omega_x = jnp.asarray(cfg.w_ref_omega_x, dtype=dtype)
    w_ref_omega_y = jnp.asarray(cfg.w_ref_omega_y, dtype=dtype)
    w_ref_omega_z = jnp.asarray(cfg.w_ref_omega_z, dtype=dtype)
    w_control_a = jnp.asarray(cfg.w_control_a, dtype=dtype)
    w_control_omega = jnp.asarray(cfg.w_control_omega, dtype=dtype)

    init_px_range = jnp.asarray(cfg.init_px_range, dtype=dtype)
    init_py_range = jnp.asarray(cfg.init_py_range, dtype=dtype)
    init_pz_range = jnp.asarray(cfg.init_pz_range, dtype=dtype)
    init_v_range = jnp.asarray(cfg.init_v_range, dtype=dtype)
    init_tilt_max_rad = jnp.asarray(np.deg2rad(cfg.init_tilt_deg_range), dtype=dtype)
    init_yaw_max_rad = jnp.asarray(np.deg2rad(cfg.init_yaw_deg_range), dtype=dtype)

    obs_dim = 26 if include_time_features else 23

    inf = jnp.asarray(jnp.inf, dtype=dtype)

    _normalize_quaternion = normalize_quaternion
    _quat_conj = quaternion_conjugate
    _quat_mul = quaternion_multiply
    _quat_from_euler_zyx = quaternion_from_euler_zyx

    def _clip_action(u: Array) -> Array:
        u = jnp.asarray(u, dtype=dtype)
        u0 = jnp.clip(u[0], a_cmd_min, a_cmd_max)
        u_omega = jnp.clip(u[1:4], -omega_max, omega_max)
        return jnp.array([u0, u_omega[0], u_omega[1], u_omega[2]], dtype=dtype)

    def _state_derivative(x: Array, u: Array) -> Array:
        u = _clip_action(u)
        f, g = quadrotor_control_affine_terms(x, gravity)
        x_dot = f
        x_dot = x_dot.at[3:6].add(u[0] * g[3:6, 0])
        x_dot = x_dot.at[6:10].add(g[6:10, 1:4] @ u[1:4])
        return x_dot

    def _hard_deck_value(x: Array) -> Array:
        return z_max - x[2]

    def _is_safe_state(x: Array) -> Array:
        return _hard_deck_value(x) >= 0.0

    def _reference(step_idx: Array) -> tuple[Array, Array, Array, Array, Array, Array]:
        t = step_idx.astype(dtype) * dt
        idx_float = t / ref_dt
        idx_clip = jnp.clip(idx_float, 0.0, ref_last_f)
        i0 = jnp.floor(idx_clip).astype(jnp.int32)
        i1 = jnp.minimum(i0 + 1, ref_last_i32)
        w = idx_clip - i0.astype(dtype)

        ref0 = ref_states[i0]
        ref1 = ref_states[i1]
        ref = (1.0 - w) * ref0 + w * ref1
        ref = ref.at[6:10].set(_normalize_quaternion(ref[6:10]))
        omega0 = ref_omega_cmd[i0]
        omega1 = ref_omega_cmd[i1]
        ref_omega = (1.0 - w) * omega0 + w * omega1

        s = jnp.clip(idx_clip / ref_last_f, 0.0, 1.0)
        phase = 2.0 * jnp.asarray(jnp.pi, dtype=dtype) * s
        return ref, ref_omega, s, t, jnp.sin(phase), jnp.cos(phase)

    def _observation(x: Array, step_idx: Array) -> Array:
        ref_state, ref_omega, _, t, phase_sin, phase_cos = _reference(step_idx)
        if include_time_features:
            time_feats = jnp.array([t, phase_sin, phase_cos], dtype=dtype)
            return jnp.concatenate([x, ref_state, ref_omega, time_feats], axis=0)
        return jnp.concatenate([x, ref_state, ref_omega], axis=0)

    def _sample_initial_state(key: Array) -> Array:
        ref0 = ref_states[jnp.int32(0)]

        px0, py0, pz0 = ref0[0], ref0[1], ref0[2]
        vx0, vy0, vz0 = ref0[3], ref0[4], ref0[5]
        q0 = _normalize_quaternion(ref0[6:10])

        k_px, k_py, k_pz, k_vx, k_vy, k_vz, k_roll, k_pitch, k_yaw = jax.random.split(key, 9)

        px = px0 + jax.random.uniform(k_px, (), minval=-init_px_range, maxval=init_px_range, dtype=dtype)
        py = py0 + jax.random.uniform(k_py, (), minval=-init_py_range, maxval=init_py_range, dtype=dtype)
        pz = pz0 + jax.random.uniform(k_pz, (), minval=-init_pz_range, maxval=init_pz_range, dtype=dtype)
        pz = jnp.minimum(pz, z_max - jnp.asarray(0.05, dtype=dtype))

        vx = vx0 + jax.random.uniform(k_vx, (), minval=-init_v_range, maxval=init_v_range, dtype=dtype)
        vy = vy0 + jax.random.uniform(k_vy, (), minval=-init_v_range, maxval=init_v_range, dtype=dtype)
        vz = vz0 + jax.random.uniform(k_vz, (), minval=-init_v_range, maxval=init_v_range, dtype=dtype)

        roll = jax.random.uniform(k_roll, (), minval=-init_tilt_max_rad, maxval=init_tilt_max_rad, dtype=dtype)
        pitch = jax.random.uniform(k_pitch, (), minval=-init_tilt_max_rad, maxval=init_tilt_max_rad, dtype=dtype)
        yaw = jax.random.uniform(k_yaw, (), minval=-init_yaw_max_rad, maxval=init_yaw_max_rad, dtype=dtype)

        q_perturb = _quat_from_euler_zyx(roll, pitch, yaw)
        q = _normalize_quaternion(_quat_mul(q_perturb, q0))

        return jnp.array([px, py, pz, vx, vy, vz, q[0], q[1], q[2], q[3]], dtype=dtype)

    def _reward(
        x_next: Array,
        u: Array,
        ref_state: Array,
        reward_ref_omega_cmd: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        pos_err = x_next[0:3] - ref_state[0:3]
        vel_err = x_next[3:6] - ref_state[3:6]

        q_ref = _normalize_quaternion(ref_state[6:10])
        q_now = _normalize_quaternion(x_next[6:10])
        q_err = _quat_mul(q_ref, _quat_conj(q_now))
        sign_term = jnp.where(q_err[0] >= 0.0, 1.0, -1.0).astype(dtype)
        att_err_vec = sign_term * q_err[1:4]
        omega_ref_err = u[1:4] - reward_ref_omega_cmd

        pos_cost = w_pos_xy * (pos_err[0] ** 2 + pos_err[1] ** 2) + w_pos_z * (pos_err[2] ** 2)
        vel_cost = w_vel * jnp.dot(vel_err, vel_err)
        att_cost = w_att * jnp.dot(att_err_vec, att_err_vec)
        omega_ref_cost = (
            w_ref_omega_x * (omega_ref_err[0] ** 2)
            + w_ref_omega_y * (omega_ref_err[1] ** 2)
            + w_ref_omega_z * (omega_ref_err[2] ** 2)
        )
        control_cost = w_control_a * (u[0] ** 2) + w_control_omega * jnp.dot(u[1:], u[1:])

        reward = -(pos_cost + vel_cost + att_cost + omega_ref_cost + control_cost)

        return (
            reward.astype(dtype),
            jnp.linalg.norm(pos_err),
            jnp.linalg.norm(vel_err),
            jnp.linalg.norm(att_err_vec),
            jnp.linalg.norm(omega_ref_err),
        )

    def env_reset(key: Array) -> tuple[QuadrotorEnvState, Array]:
        x0 = _sample_initial_state(key)
        state = QuadrotorEnvState(
            x=x0,
            steps=jnp.int32(0),
            ep_return=jnp.asarray(0.0, dtype=dtype),
            ep_len=jnp.int32(0),
            ep_safe_sum=jnp.asarray(0.0, dtype=dtype),
            ep_pos_err_sum=jnp.asarray(0.0, dtype=dtype),
            ep_vel_err_sum=jnp.asarray(0.0, dtype=dtype),
            ep_att_err_sum=jnp.asarray(0.0, dtype=dtype),
            ep_hard_deck_margin_min=inf,
        )
        obs = _observation(state.x, state.steps)
        return state, obs

    def env_step(state: QuadrotorEnvState, action: Array, key: Array):
        u = _clip_action(action)
        x_dot = _state_derivative(state.x, u)
        x_next = state.x + dt * x_dot
        x_next = x_next.at[6:10].set(_normalize_quaternion(x_next[6:10]))

        next_steps = state.steps + jnp.int32(1)
        safe = _is_safe_state(x_next)
        hard_deck_margin = _hard_deck_value(x_next)

        done_time = next_steps >= max_steps_i32
        done = done_time | (terminate_on_violation & (~safe))

        _, ref_omega_prev, _, _, _, _ = _reference(state.steps)
        ref_state, ref_omega_next, ref_progress, ref_time_sec, _, _ = _reference(next_steps)
        reward_ref_omega = 0.5 * (ref_omega_prev + ref_omega_next)
        reward, pos_err_norm, vel_err_norm, att_err_norm, omega_ref_err_norm = _reward(
            x_next,
            u,
            ref_state,
            reward_ref_omega,
        )

        ep_return = state.ep_return + reward
        ep_len = state.ep_len + jnp.int32(1)
        ep_safe_sum = state.ep_safe_sum + safe.astype(dtype)
        ep_pos_err_sum = state.ep_pos_err_sum + pos_err_norm
        ep_vel_err_sum = state.ep_vel_err_sum + vel_err_norm
        ep_att_err_sum = state.ep_att_err_sum + att_err_norm
        ep_hard_deck_margin_min = jnp.minimum(state.ep_hard_deck_margin_min, hard_deck_margin)

        done_f = done.astype(dtype)
        ep_len_f = jnp.maximum(ep_len.astype(dtype), jnp.asarray(1.0, dtype=dtype))

        completed_return = done_f * ep_return
        completed_len = done_f * ep_len.astype(dtype)
        completed_safe_rate = done_f * (ep_safe_sum / ep_len_f)
        completed_pos_error_norm = done_f * (ep_pos_err_sum / ep_len_f)
        completed_vel_error_norm = done_f * (ep_vel_err_sum / ep_len_f)
        completed_att_error_norm = done_f * (ep_att_err_sum / ep_len_f)
        completed_hard_deck_margin_min = done_f * ep_hard_deck_margin_min

        reset_x = _sample_initial_state(key)

        x_out = jnp.where(done, reset_x, x_next)
        steps_out = jnp.where(done, jnp.int32(0), next_steps)

        state_out = QuadrotorEnvState(
            x=x_out,
            steps=steps_out,
            ep_return=jnp.where(done, jnp.asarray(0.0, dtype=dtype), ep_return),
            ep_len=jnp.where(done, jnp.int32(0), ep_len),
            ep_safe_sum=jnp.where(done, jnp.asarray(0.0, dtype=dtype), ep_safe_sum),
            ep_pos_err_sum=jnp.where(done, jnp.asarray(0.0, dtype=dtype), ep_pos_err_sum),
            ep_vel_err_sum=jnp.where(done, jnp.asarray(0.0, dtype=dtype), ep_vel_err_sum),
            ep_att_err_sum=jnp.where(done, jnp.asarray(0.0, dtype=dtype), ep_att_err_sum),
            ep_hard_deck_margin_min=jnp.where(done, inf, ep_hard_deck_margin_min),
        )

        obs_out = _observation(state_out.x, state_out.steps)
        next_obs_true = _observation(x_next, next_steps)

        info = QuadrotorStepInfo(
            is_safe=safe.astype(dtype),
            pos_error_norm=pos_err_norm.astype(dtype),
            vel_error_norm=vel_err_norm.astype(dtype),
            att_error_norm=att_err_norm.astype(dtype),
            omega_ref_error_norm=omega_ref_err_norm.astype(dtype),
            hard_deck_margin=hard_deck_margin.astype(dtype),
            ref_progress=ref_progress.astype(dtype),
            ref_time_sec=ref_time_sec.astype(dtype),
            ref_state=ref_state.astype(dtype),
            ref_omega=ref_omega_next.astype(dtype),
            episode_done=done,
            completed_return=completed_return.astype(dtype),
            completed_len=completed_len.astype(dtype),
            completed_safe_rate=completed_safe_rate.astype(dtype),
            completed_pos_error_norm=completed_pos_error_norm.astype(dtype),
            completed_vel_error_norm=completed_vel_error_norm.astype(dtype),
            completed_att_error_norm=completed_att_error_norm.astype(dtype),
            completed_hard_deck_margin_min=completed_hard_deck_margin_min.astype(dtype),
        )

        return state_out, next_obs_true, obs_out, reward.astype(dtype), done, info

    reset = jax.jit(env_reset)
    step = jax.jit(env_step)
    reset_batched = jax.jit(jax.vmap(reset, in_axes=0))
    step_batched = jax.jit(jax.vmap(step, in_axes=(0, 0, 0)))

    return QuadrotorEnvFns(
        obs_dim=obs_dim,
        action_dim=4,
        reset=reset,
        step=step,
        reset_batched=reset_batched,
        step_batched=step_batched,
    )
