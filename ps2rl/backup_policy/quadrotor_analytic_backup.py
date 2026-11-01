"""Quadrotor analytic backup policy.

The analytic quadrotor backup is a HYBRID: 
  - pi_SA: an aggressive cascaded PID safe-arrival policy outside the base set B, 
  - pi_B: the discrete-time hover LQR inside it, switched by the base-set 
    membership check (one-way handoff).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ps2rl.backup_policy.backup_policy import ABP
from ps2rl.base_controller.quadrotor_dlqr import QuadrotorDLQR
from ps2rl.sets.base_sets import EllipsoidBaseSet
from ps2rl.utils.quaternion import (
    desired_quaternion_from_virtual_accel,
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_from_two_vectors,
    quaternion_multiply,
)


def clip_action(u: jax.Array, cfg: Any) -> jax.Array:
    """Clip to the quadrotor action box (duck cfg: a_cmd_min/a_cmd_max/omega_max).

    Kept as the per-element clip. This form is differentiated inside the BCBF
    sensitivity (``build_discretized_backup_cbf_rows`` takes ``jacfwd`` of the
    backup flow through ``quadrotor_dynamics`` -> ``clip_action``).
    """
    u = jnp.asarray(u)
    u0 = jnp.clip(u[0], cfg.a_cmd_min, cfg.a_cmd_max)
    u_omega = jnp.clip(u[1:4], -cfg.omega_max, cfg.omega_max)
    return jnp.array([u0, u_omega[0], u_omega[1], u_omega[2]], dtype=u.dtype)


def _aggressive_pid_policy_raw(
    x: jax.Array,
    cfg: Any,
) -> jax.Array:
    x = jnp.asarray(x)
    q = normalize_quaternion(x[6:10])
    vx, vy, vz = x[3], x[4], x[5]
    pz = x[2]

    margin_nom = float(max(cfg.pid_ceiling_margin, 1e-6))
    margin_nom_j = jnp.asarray(margin_nom, dtype=x.dtype)
    ceiling_margin = jnp.asarray(cfg.z_max, dtype=x.dtype) - pz
    near_ceiling = jnp.clip((margin_nom_j - ceiling_margin) / margin_nom_j, 0.0, 1.0)

    upward_speed = jnp.maximum(vz, 0.0)
    lateral_gain = jnp.asarray(cfg.pid_kv_xy, dtype=x.dtype) * (
        1.0 + jnp.asarray(cfg.pid_lateral_boost, dtype=x.dtype) * near_ceiling
    )
    nu = jnp.array(
        [
            -lateral_gain * vx,
            -lateral_gain * vy,
            jnp.asarray(cfg.gravity, dtype=x.dtype)
            - jnp.asarray(cfg.pid_kp_z, dtype=x.dtype) * (pz - cfg.z_des)
            - jnp.asarray(cfg.pid_kv_z, dtype=x.dtype) * vz
            - jnp.asarray(cfg.pid_z_safety_gain, dtype=x.dtype) * near_ceiling
            - jnp.asarray(cfg.pid_ceiling_vz_gain, dtype=x.dtype) * near_ceiling * upward_speed,
        ],
        dtype=x.dtype,
    )
    nu = nu.at[2].set(jnp.maximum(nu[2], jnp.asarray(cfg.pid_min_virtual_accel_z, dtype=x.dtype)))

    q_des = desired_quaternion_from_virtual_accel(nu)
    q_err = quaternion_multiply(q_des, quaternion_conjugate(q))
    sign_term = jnp.where(q_err[0] >= 0.0, 1.0, -1.0).astype(x.dtype)
    omega_gain = jnp.array(
        [
            cfg.pid_attitude_p_gain,
            cfg.pid_attitude_p_gain,
            cfg.pid_attitude_p_gain * cfg.pid_yaw_gain_scale,
        ],
        dtype=x.dtype,
    )
    omega_cmd = omega_gain * sign_term * q_err[1:4]
    a_cmd = jnp.linalg.norm(nu)
    return jnp.array([a_cmd, omega_cmd[0], omega_cmd[1], omega_cmd[2]], dtype=x.dtype)


@dataclass(frozen=True)
class QuadrotorABP(ABP):
    """Analytic quadrotor backup: cascaded PID outside B, hover LQR inside."""

    pid_cfg: Any = None

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        *,
        base_controller: QuadrotorDLQR | None = None,
        base_set: EllipsoidBaseSet | None = None,
    ) -> "QuadrotorABP":
        controller = base_controller if base_controller is not None else QuadrotorDLQR.from_config(cfg)
        if base_set is None:
            base_set = EllipsoidBaseSet(
                controller,
                float(cfg.base_set_c),
                smooth_gain=float(cfg.base_set_smooth_gain),
            )
        return cls(base_controller=controller, base_set=base_set, pid_cfg=cfg)

    def sa_action(self, x: jax.Array) -> jax.Array:
        return _aggressive_pid_policy_raw(jnp.asarray(x), self.pid_cfg)

    def action(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        pid_raw = _aggressive_pid_policy_raw(x, self.pid_cfg)
        u_raw = self.select_action(x, pid_raw, self.base_set)
        return clip_action(u_raw, self.pid_cfg)


__all__ = [
    "QuadrotorABP",
    "_aggressive_pid_policy_raw",
    "clip_action",
]
