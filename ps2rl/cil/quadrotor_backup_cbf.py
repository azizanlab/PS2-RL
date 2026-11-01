"""Quadrotor backup-CBF system: config, dynamics/quaternion math, assembly.

State: x = [p_x, p_y, p_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z]
Action: u = [a_cmd, omega_x, omega_y, omega_z]
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, isclose
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.backup_policy.quadrotor_analytic_backup import (
    QuadrotorABP,
    _aggressive_pid_policy_raw,
)
from ps2rl.utils.quaternion import (
    normalize_quaternion,
    quaternion_conjugate,
)
from ps2rl.backup_policy.quadrotor_learned_backup import (
    QuadrotorLBP,
    load_learned_quadrotor_backup_policy,
)
from ps2rl.base_controller.quadrotor_dlqr import QuadrotorDLQR
from ps2rl.cil.backup_cbf import (
    BCBFSystem,
    BackupCBFConfig,
    BackupCBFProjector,
    make_backup_cbf_facades,
)
from ps2rl.sets.base_sets import EllipsoidBaseSet
from ps2rl.envs.quadrotor_env import (
    quadrotor_control_affine_terms as _env_quadrotor_control_affine_terms,
    quadrotor_dynamics as _env_quadrotor_dynamics,
    quadrotor_step_euler as _env_quadrotor_step_euler,
)


@dataclass(frozen=True)
class QuadrotorBCBFConfig(BackupCBFConfig):
    """Configuration for analytical or learned quadrotor backup-CBF projection.

    The shared backup-CBF fields (horizon/mesh, class-K gains, QP weights,
    solver params, row conditioning) are inherited from ``BackupCBFConfig``.
    """

    # Backup-flow horizon.
    num_steps: int = 100

    # Dynamics and action bounds.
    gravity: float = 9.81
    a_cmd_min: float = 0.0
    a_cmd_max: float | None = None  # If unset, defaults to 4g.
    omega_max: float = 18.0

    # Safety set C (the hard-deck ceiling z <= z_max).
    z_max: float = 3.0

    # Base controller design point (the hover-LQR equilibrium z_des).
    z_des: float = 2.0

    # Analytic backup policy u_b(x): aggressive cascaded PID outside the base
    # set B, discrete-time hover LQR inside it.
    # (same base-set handoff contract as the learned phase-1 policy).
    base_set_smooth_gain: float = 20.0
    pid_kp_z: float = 36.0
    pid_kv_z: float = 24.0
    pid_kv_xy: float = 14.0
    pid_attitude_p_gain: float = 45.0
    pid_yaw_gain_scale: float = 0.35
    pid_ceiling_margin: float = 0.60
    pid_z_safety_gain: float = 32.0
    pid_ceiling_vz_gain: float = 18.0
    pid_lateral_boost: float = 1.50
    pid_min_virtual_accel_z: float = 0.0

    # alpha: the safe-set CBF row's class-K gain.
    alpha: float = 4.0
    # base_alpha: the base-set CBF row's class-K gain.
    base_alpha: float = 2.0
    # Base set B: hover-LQR ellipsoid h_B(x) = base_set_c - x_err^T P x_err. 
    base_set_c: float = 8.0
    # LQR weights are DISCRETE-TIME.
    lqr_q_z: float = 1.0
    lqr_q_vx: float = 0.16
    lqr_q_vy: float = 0.16
    lqr_q_vz: float = 0.4
    lqr_q_thetax: float = 0.8
    lqr_q_thetay: float = 0.8
    lqr_q_thetaz: float = 0.16
    lqr_r_a_cmd: float = 0.02
    lqr_r_omega_x: float = 0.012
    lqr_r_omega_y: float = 0.012
    lqr_r_omega_z: float = 0.004

    # qpax solver params. solver_tol overrides the base default 1e-3.
    solver_tol: float = 5e-4

    # Sensitivity rollout options.
    use_analytic_jacobian: bool = False

    def __post_init__(self) -> None:
        n = int(self.num_steps)
        if n <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")

        g = float(self.gravity)
        if (not isfinite(g)) or g <= 0.0:
            raise ValueError(f"gravity must be a positive finite value, got {self.gravity}")

        t = self.T
        dt = self.dt
        if t is None and dt is None:
            t = 2.0
            dt = t / n
        elif t is None:
            dt = float(dt)
            if (not isfinite(dt)) or dt <= 0.0:
                raise ValueError(f"dt must be a positive finite value, got {self.dt}")
            t = dt * n
        elif dt is None:
            t = float(t)
            if (not isfinite(t)) or t <= 0.0:
                raise ValueError(f"T must be a positive finite value, got {self.T}")
            dt = t / n
        else:
            t = float(t)
            dt = float(dt)
            if (not isfinite(t)) or t <= 0.0:
                raise ValueError(f"T must be a positive finite value, got {self.T}")
            if (not isfinite(dt)) or dt <= 0.0:
                raise ValueError(f"dt must be a positive finite value, got {self.dt}")
            implied_t = dt * n
            if not isclose(t, implied_t, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    f"Inconsistent T, dt, num_steps: T={t}, dt={dt}, num_steps={n}, dt*num_steps={implied_t}"
                )
            t = implied_t

        a_cmd_min = float(self.a_cmd_min)
        a_cmd_max = float(4.0 * g) if self.a_cmd_max is None else float(self.a_cmd_max)
        if (not isfinite(a_cmd_min)) or (not isfinite(a_cmd_max)):
            raise ValueError("a_cmd bounds must be finite")
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
            raise ValueError("z_max and z_des must be finite")
        if z_max <= 0.0:
            raise ValueError(f"z_max must be positive for hard-deck recovery, got z_max={z_max}")
        if z_des >= z_max:
            raise ValueError(
                f"z_des must be less than z_max for hard-deck recovery, got z_des={z_des}, z_max={z_max}"
            )

        mode = str(self.backup_policy_mode).strip().lower()
        learned_backup_policy_path = str(self.learned_backup_policy_path).strip()
        if mode not in {"analytic", "learned"}:
            raise ValueError(
                f"backup_policy_mode must be one of ['analytic', 'learned'], got {self.backup_policy_mode}"
            )
        if mode == "learned" and not learned_backup_policy_path:
            raise ValueError("learned_backup_policy_path must be provided when backup_policy_mode='learned'")

        base_set_smooth_gain = float(self.base_set_smooth_gain)
        if (not isfinite(base_set_smooth_gain)) or base_set_smooth_gain < 0.0:
            raise ValueError(
                f"base_set_smooth_gain must be nonnegative and finite, got {self.base_set_smooth_gain}"
            )

        base_set_c = float(self.base_set_c)
        if (not isfinite(base_set_c)) or base_set_c <= 0.0:
            raise ValueError(f"base_set_c must be positive and finite, got {self.base_set_c}")

        new_policy_positive = {
            "pid_kp_z": self.pid_kp_z,
            "pid_kv_z": self.pid_kv_z,
            "pid_kv_xy": self.pid_kv_xy,
            "pid_attitude_p_gain": self.pid_attitude_p_gain,
            "pid_yaw_gain_scale": self.pid_yaw_gain_scale,
            "pid_ceiling_margin": self.pid_ceiling_margin,
            "pid_z_safety_gain": self.pid_z_safety_gain,
            "pid_ceiling_vz_gain": self.pid_ceiling_vz_gain,
            "pid_lateral_boost": self.pid_lateral_boost,
            "base_set_c": self.base_set_c,
            "lqr_q_z": self.lqr_q_z,
            "lqr_q_vx": self.lqr_q_vx,
            "lqr_q_vy": self.lqr_q_vy,
            "lqr_q_vz": self.lqr_q_vz,
            "lqr_q_thetax": self.lqr_q_thetax,
            "lqr_q_thetay": self.lqr_q_thetay,
            "lqr_q_thetaz": self.lqr_q_thetaz,
            "lqr_r_a_cmd": self.lqr_r_a_cmd,
            "lqr_r_omega_x": self.lqr_r_omega_x,
            "lqr_r_omega_y": self.lqr_r_omega_y,
            "lqr_r_omega_z": self.lqr_r_omega_z,
        }
        for name, value in new_policy_positive.items():
            value_f = float(value)
            if (not isfinite(value_f)) or value_f <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {value}")
        pid_min_virtual_accel_z = float(self.pid_min_virtual_accel_z)
        if not isfinite(pid_min_virtual_accel_z):
            raise ValueError(
                f"pid_min_virtual_accel_z must be finite, got {self.pid_min_virtual_accel_z}"
            )

        object.__setattr__(self, "num_steps", n)
        object.__setattr__(self, "T", float(t))
        object.__setattr__(self, "dt", float(dt))
        object.__setattr__(self, "gravity", g)
        object.__setattr__(self, "a_cmd_min", a_cmd_min)
        object.__setattr__(self, "a_cmd_max", a_cmd_max)
        object.__setattr__(self, "omega_max", omega_max)
        object.__setattr__(self, "z_max", z_max)
        object.__setattr__(self, "z_des", z_des)
        object.__setattr__(self, "backup_policy_mode", mode)
        object.__setattr__(self, "learned_backup_policy_path", learned_backup_policy_path)
        object.__setattr__(self, "base_set_smooth_gain", base_set_smooth_gain)
        object.__setattr__(self, "base_set_c", base_set_c)
        object.__setattr__(self, "pid_min_virtual_accel_z", pid_min_virtual_accel_z)

    @property
    def action_scale(self) -> jax.Array:
        return jnp.array(
            [self.a_cmd_max, self.omega_max, self.omega_max, self.omega_max],
            dtype=jnp.float32,
        )

    @property
    def lqr_q_diag(self) -> tuple[float, ...]:
        return (
            float(self.lqr_q_z),
            float(self.lqr_q_vx),
            float(self.lqr_q_vy),
            float(self.lqr_q_vz),
            float(self.lqr_q_thetax),
            float(self.lqr_q_thetay),
            float(self.lqr_q_thetaz),
        )

    @property
    def lqr_r_diag(self) -> tuple[float, ...]:
        return (
            float(self.lqr_r_a_cmd),
            float(self.lqr_r_omega_x),
            float(self.lqr_r_omega_y),
            float(self.lqr_r_omega_z),
        )

    @property
    def num_safe_constraints(self) -> int:
        return 1

    @property
    def num_base_set_constraints(self) -> int:
        return 1

    @property
    def num_backup_inequalities(self) -> int:
        return self.num_safe_constraints * (self.num_steps + 1) + self.num_base_set_constraints

    @property
    def num_qp_inequalities(self) -> int:
        # Eight actuator bounds: thrust and three body rates, upper and lower.
        return self.num_backup_inequalities + 8


_RUNTIME_CACHE: dict[tuple, BCBFSystem] = {}


def _runtime_cache_key(cfg: QuadrotorBCBFConfig) -> tuple:
    return (
        cfg.backup_policy_mode,
        cfg.learned_backup_policy_path,
        cfg.dt,
        cfg.a_cmd_min,
        cfg.a_cmd_max,
        cfg.omega_max,
        cfg.gravity,
        cfg.z_max,
        cfg.z_des,
        cfg.base_set_smooth_gain,
        cfg.pid_kp_z,
        cfg.pid_kv_z,
        cfg.pid_kv_xy,
        cfg.pid_attitude_p_gain,
        cfg.pid_yaw_gain_scale,
        cfg.pid_ceiling_margin,
        cfg.pid_z_safety_gain,
        cfg.pid_ceiling_vz_gain,
        cfg.pid_lateral_boost,
        cfg.pid_min_virtual_accel_z,
        cfg.base_set_c,
        cfg.lqr_q_z,
        cfg.lqr_q_vx,
        cfg.lqr_q_vy,
        cfg.lqr_q_vz,
        cfg.lqr_q_thetax,
        cfg.lqr_q_thetay,
        cfg.lqr_q_thetaz,
        cfg.lqr_r_a_cmd,
        cfg.lqr_r_omega_x,
        cfg.lqr_r_omega_y,
        cfg.lqr_r_omega_z,
        cfg.use_analytic_jacobian,
    )


def quadrotor_hover_lqr_config_from_cbf_cfg(cfg: QuadrotorBCBFConfig) -> dict[str, float]:
    return {
        "dt": float(cfg.dt),
        "gravity": float(cfg.gravity),
        "a_cmd_min": float(cfg.a_cmd_min),
        "a_cmd_max": float(cfg.a_cmd_max),
        "omega_max": float(cfg.omega_max),
        "z_des": float(cfg.z_des),
        "lqr_q_z": float(cfg.lqr_q_z),
        "lqr_q_vx": float(cfg.lqr_q_vx),
        "lqr_q_vy": float(cfg.lqr_q_vy),
        "lqr_q_vz": float(cfg.lqr_q_vz),
        "lqr_q_thetax": float(cfg.lqr_q_thetax),
        "lqr_q_thetay": float(cfg.lqr_q_thetay),
        "lqr_q_thetaz": float(cfg.lqr_q_thetaz),
        "lqr_r_a_cmd": float(cfg.lqr_r_a_cmd),
        "lqr_r_omega_x": float(cfg.lqr_r_omega_x),
        "lqr_r_omega_y": float(cfg.lqr_r_omega_y),
        "lqr_r_omega_z": float(cfg.lqr_r_omega_z),
    }


def quadrotor_step_euler(
    x: jax.Array,
    u: jax.Array,
    cfg: QuadrotorBCBFConfig,
    dt: float | None = None,
) -> jax.Array:
    dt_use = cfg.dt if dt is None else float(dt)
    return _env_quadrotor_step_euler(
        x, u, dt_use, cfg.gravity, cfg.a_cmd_min, cfg.a_cmd_max, cfg.omega_max
    )


def hard_deck_value(x: jax.Array, cfg: QuadrotorBCBFConfig) -> jax.Array:
    x = jnp.asarray(x)
    return jnp.asarray(cfg.z_max, dtype=x.dtype) - x[2]


def is_safe_state(x: jax.Array, cfg: QuadrotorBCBFConfig) -> jax.Array:
    x = jnp.asarray(x)
    safe = jnp.asarray(True)
    safe = safe & (hard_deck_value(x, cfg) >= 0.0)
    return safe


def _safe_set_values_and_grads_default(
    x: jax.Array, cfg: QuadrotorBCBFConfig
) -> Tuple[jax.Array, jax.Array]:
    val = hard_deck_value(x, cfg)
    dh = jnp.zeros((10,), dtype=x.dtype).at[2].set(-1.0)
    return jnp.stack([val]), jnp.stack([dh])


def reduced_hover_error_state(x: jax.Array, cfg: QuadrotorBCBFConfig) -> jax.Array:
    x = jnp.asarray(x)
    q = normalize_quaternion(x[6:10])
    q_err = quaternion_conjugate(q)
    sign_term = jnp.where(q_err[0] >= 0.0, 1.0, -1.0).astype(x.dtype)
    theta_err = 2.0 * sign_term * q_err[1:4]
    return jnp.array(
        [x[2] - cfg.z_des, x[3], x[4], x[5], theta_err[0], theta_err[1], theta_err[2]],
        dtype=x.dtype,
    )


def _base_set_values_impl(x: jax.Array, cfg: QuadrotorBCBFConfig, p_matrix: jax.Array) -> jax.Array:
    x = jnp.asarray(x)
    x_err = reduced_hover_error_state(x, cfg)
    P = jnp.asarray(p_matrix, dtype=x_err.dtype)
    quad = x_err @ (P @ x_err)
    return jnp.asarray([cfg.base_set_c - quad], dtype=x_err.dtype)


def base_set_values(
    x: jax.Array,
    cfg: QuadrotorBCBFConfig,
    runtime: BCBFSystem | None = None,
) -> jax.Array:
    rt = _resolve_runtime(cfg, runtime)
    return rt.base_set_values_fn(x)


def _base_set_values_and_grads_default(
    x: jax.Array,
    cfg: QuadrotorBCBFConfig,
    runtime: BCBFSystem | None = None,
) -> Tuple[jax.Array, jax.Array]:
    rt = _resolve_runtime(cfg, runtime)
    vals = rt.base_set_values_fn(x)
    grads = jax.jacfwd(rt.base_set_values_fn)(x)
    return vals, grads


def base_margin(
    x: jax.Array,
    cfg: QuadrotorBCBFConfig,
    runtime: BCBFSystem | None = None,
) -> jax.Array:
    return jnp.min(base_set_values(x, cfg, runtime=runtime))


def is_in_base_set(
    x: jax.Array,
    cfg: QuadrotorBCBFConfig,
    atol: float = 0.0,
    runtime: BCBFSystem | None = None,
) -> jax.Array:
    vals = base_set_values(x, cfg, runtime=runtime)
    return jnp.all(vals >= -jnp.asarray(atol, dtype=vals.dtype))


def analytic_backup_policy(
    x: jax.Array,
    cfg: QuadrotorBCBFConfig,
    runtime: BCBFSystem | None = None,
) -> jax.Array:
    rt = _resolve_runtime(cfg, runtime)
    return rt.backup_policy_fn(x)


def analytic_backup_policy_batch(
    x_batch: jax.Array,
    cfg: QuadrotorBCBFConfig,
    runtime: BCBFSystem | None = None,
) -> jax.Array:
    x_batch = jnp.asarray(x_batch)
    rt = _resolve_runtime(cfg, runtime)
    return jax.vmap(rt.backup_policy_fn, in_axes=0)(x_batch)


def _sanitize_quadrotor_solve_state(x: jax.Array) -> jax.Array:
    x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.at[6:10].set(normalize_quaternion(x[6:10]))


def make_backup_runtime(cfg: QuadrotorBCBFConfig) -> BCBFSystem:
    controller = QuadrotorDLQR.from_config(cfg)
    p_matrix = controller.p_matrix
    analytic_base_set = EllipsoidBaseSet(
        controller,
        float(cfg.base_set_c),
        smooth_gain=float(cfg.base_set_smooth_gain),
    )

    def terminal_values_fn(x: jax.Array) -> jax.Array:
        return _base_set_values_impl(x, cfg, p_matrix)

    mode = cfg.backup_policy_mode.strip().lower()
    if mode == "analytic":
        abp = QuadrotorABP(
            base_controller=controller,
            base_set=analytic_base_set,
            pid_cfg=cfg,
        )
        backup_policy_fn = abp.action
    elif mode == "learned":
        learned = load_learned_quadrotor_backup_policy(cfg.learned_backup_policy_path)
        expected_action_scale = np.asarray([cfg.a_cmd_max, cfg.omega_max, cfg.omega_max, cfg.omega_max], dtype=np.float32)
        expected_action_low = np.asarray([cfg.a_cmd_min, -cfg.omega_max, -cfg.omega_max, -cfg.omega_max], dtype=np.float32)
        expected_action_high = np.asarray([cfg.a_cmd_max, cfg.omega_max, cfg.omega_max, cfg.omega_max], dtype=np.float32)
        learned_action_scale = np.asarray(jax.device_get(learned.action_scale), dtype=np.float32)
        learned_action_low = np.asarray(jax.device_get(learned.action_low), dtype=np.float32)
        learned_action_high = np.asarray(jax.device_get(learned.action_high), dtype=np.float32)
        if not np.allclose(learned_action_scale, expected_action_scale, atol=1e-6):
            raise ValueError(
                "Learned quadrotor backup policy action_scale does not match the current backup-CBF bounds: "
                f"got {learned_action_scale.tolist()}, expected {expected_action_scale.tolist()}."
            )
        if not np.allclose(learned_action_low, expected_action_low, atol=1e-6):
            raise ValueError(
                "Learned quadrotor backup policy action_low does not match the current backup-CBF bounds: "
                f"got {learned_action_low.tolist()}, expected {expected_action_low.tolist()}."
            )
        if not np.allclose(learned_action_high, expected_action_high, atol=1e-6):
            raise ValueError(
                "Learned quadrotor backup policy action_high does not match the current backup-CBF bounds: "
                f"got {learned_action_high.tolist()}, expected {expected_action_high.tolist()}."
            )
        lqr_cfg_payload = learned.metadata.get("lqr_config")
        if not isinstance(lqr_cfg_payload, dict):
            raise KeyError(
                "Learned quadrotor backup policy is missing metadata['lqr_config'] for the base-set LQR."
            )
        # The hover-LQR design (P/K provenance, z_des) comes from the saved
        # payload; the input box and level come from the runtime config.
        learned_controller = QuadrotorDLQR.from_payload(lqr_cfg_payload, fallback=cfg)
        base_set = EllipsoidBaseSet(
            learned_controller,
            float(cfg.base_set_c),
            smooth_gain=float(cfg.base_set_smooth_gain),
        )
        lbp = QuadrotorLBP(
            base_controller=learned_controller,
            base_set=base_set,
            learned=learned,
            a_cmd_min=float(cfg.a_cmd_min),
            a_cmd_max=float(cfg.a_cmd_max),
            omega_max=float(cfg.omega_max),
        )
        backup_policy_fn = lbp.action
    else:
        raise ValueError(f"Unsupported backup_policy_mode: {cfg.backup_policy_mode}")

    return BCBFSystem(
        state_dim=10,
        action_dim=4,
        action_low=(
            float(cfg.a_cmd_min),
            -float(cfg.omega_max),
            -float(cfg.omega_max),
            -float(cfg.omega_max),
        ),
        action_high=(
            float(cfg.a_cmd_max),
            float(cfg.omega_max),
            float(cfg.omega_max),
            float(cfg.omega_max),
        ),
        backup_policy_fn=backup_policy_fn,
        safe_set_values_and_grads_fn=lambda x: _safe_set_values_and_grads_default(x, cfg),
        base_set_values_and_grads_fn=lambda x: _base_set_values_and_grads_default(x, cfg),
        dynamics_fn=lambda x, u: _env_quadrotor_dynamics(
            x, u, cfg.gravity, cfg.a_cmd_min, cfg.a_cmd_max, cfg.omega_max
        ),
        control_affine_terms_fn=lambda x: _env_quadrotor_control_affine_terms(x, cfg.gravity),
        postprocess_rollout_state_fn=lambda z: z.at[6:10].set(normalize_quaternion(z[6:10])),
        sanitize_solve_state_fn=_sanitize_quadrotor_solve_state,
        solve_state_slice_fn=lambda xb: xb,
        base_set_values_fn=terminal_values_fn,
        use_analytic_jacobian=False,
        analytic_closed_loop_and_jacobian_fn=None,
    )


# ---------------------------------- Facades --------------------------------- #
_facades = make_backup_cbf_facades(make_backup_runtime, _runtime_cache_key, _RUNTIME_CACHE)
get_cached_runtime = _facades["get_cached_runtime"]
_resolve_runtime = _facades["_resolve_runtime"]
backup_policy = _facades["backup_policy"]
backup_policy_batch = _facades["backup_policy_batch"]
closed_loop_backup_dynamics = _facades["closed_loop_backup_dynamics"]
safe_set_values_and_grads = _facades["safe_set_values_and_grads"]
base_set_values_and_grads = _facades["base_set_values_and_grads"]
rollout_backup_flow_and_sensitivity = _facades["rollout_backup_flow_and_sensitivity"]
rollout_backup_flow_and_sensitivity_with_info = _facades["rollout_backup_flow_and_sensitivity_with_info"]
build_discretized_backup_cbf_rows = _facades["build_discretized_backup_cbf_rows"]
build_discretized_backup_cbf_rows_with_info = _facades["build_discretized_backup_cbf_rows_with_info"]
build_backup_cbf_qp = _facades["build_backup_cbf_qp"]
solve_backup_cbf_qp_single = _facades["solve_backup_cbf_qp_single"]
solve_backup_cbf_qp_single_with_info = _facades["solve_backup_cbf_qp_single_with_info"]
solve_backup_cbf_qp_batch = _facades["solve_backup_cbf_qp_batch"]
solve_backup_cbf_qp_batch_with_info = _facades["solve_backup_cbf_qp_batch_with_info"]
constraint_residuals = _facades["constraint_residuals"]


class QuadrotorBackupCBFProjector(BackupCBFProjector):
    """Convenience wrapper with jitted single/batch projection calls."""

    def __init__(self, cfg: QuadrotorBCBFConfig, runtime: BCBFSystem | None = None):
        super().__init__(cfg, _resolve_runtime(cfg, runtime))


# Explicit aliases to mirror lane-keeping naming style where needed.
analytic_backup_policy_jax = analytic_backup_policy
analytic_backup_policy_batch_jax = analytic_backup_policy_batch
quadrotor_step_euler_jax = quadrotor_step_euler
is_in_base_set_jax = is_in_base_set
base_set_values_jax = base_set_values
base_margin_jax = base_margin
