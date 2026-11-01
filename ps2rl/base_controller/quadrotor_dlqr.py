"""Quadrotor discrete-time hover LQR base controller.

Owns the reduced hover-error coordinates e(x) in R^7 (altitude error, inertial
velocity, first-order quaternion-error angles) and the reduced linearization
about hover used for the DARE design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.base_controller.base_controller import DiscreteLQR, euler_discretize
from ps2rl.utils.quaternion import normalize_quaternion_batch, quaternion_conjugate_batch

# --- Hover-LQR weight fields, in the order they map onto (Q diag, R diag). -- #
_LQR_Q_KEYS = ("lqr_q_z", "lqr_q_vx", "lqr_q_vy", "lqr_q_vz", "lqr_q_thetax", "lqr_q_thetay", "lqr_q_thetaz")
_LQR_R_KEYS = ("lqr_r_a_cmd", "lqr_r_omega_x", "lqr_r_omega_y", "lqr_r_omega_z")


@dataclass(frozen=True)
class QuadrotorDLQR(DiscreteLQR):
    """Discrete-time hover LQR in the reduced 7D error coordinates.

      - ``from_config`` is the canonical constructor for the DLQR.
      - ``from_payload`` builds from a saved learned-policy hover-LQR payload. 
      - The certificate (P, K, c-bar) is produced by the shared
        ``DiscreteLQR._compute_certificate`` f64 pipeline.
    """

    z_des: float = 0.0  # overwritten by from_config/from_payload; default only for dataclass ordering

    @staticmethod
    def _reduced_linearization_error_coords(
        gravity: float,
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
        """Reduced hover linearization in the quaternion-error coordinates used by the LQR base set."""
        g = float(gravity)
        a_mat = (
            (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, -g, 0.0),
            (0.0, 0.0, 0.0, 0.0, g, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        b_mat = (
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 0.0, 0.0, -1.0),
        )
        return a_mat, b_mat

    @classmethod
    def _discrete_hover_lqr_matrices(cls, *, dt: float, gravity: float) -> tuple[np.ndarray, np.ndarray]:
        """Reduced-hover continuous linearization + the shared forward-Euler discretization 
        (identical pipeline to the unicycle; see ``euler_discretize``). 
        """
        a_raw, b_raw = cls._reduced_linearization_error_coords(gravity)
        return euler_discretize(a_raw, b_raw, dt)

    @classmethod
    def _build(
        cls,
        *,
        dt: float,
        design_gravity: float,
        z_des: float,
        q_diag: tuple[float, ...],
        r_diag: tuple[float, ...],
        box_gravity: float,
        a_cmd_min: float,
        a_cmd_max: float,
        omega_max: float,
    ) -> "QuadrotorDLQR":
        """Assemble the generic DLQR tuples."""
        a_d, b_d = cls._discrete_hover_lqr_matrices(dt=dt, gravity=design_gravity)
        omega_max = float(omega_max)
        return cls(
            a_d=tuple(tuple(row) for row in a_d.tolist()),
            b_d=tuple(tuple(row) for row in b_d.tolist()),
            q_diag=tuple(float(v) for v in q_diag),
            r_diag=tuple(float(v) for v in r_diag),
            u_star=(float(box_gravity), 0.0, 0.0, 0.0),
            u_low=(float(a_cmd_min), -omega_max, -omega_max, -omega_max),
            u_high=(float(a_cmd_max), omega_max, omega_max, omega_max),
            z_des=float(z_des),
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "QuadrotorDLQR":
        """Canonical constructor for the DLQR from a config."""
        gravity = float(getattr(cfg, "gravity"))
        return cls._build(
            dt=float(getattr(cfg, "dt")),
            design_gravity=gravity,
            z_des=float(getattr(cfg, "z_des")),
            q_diag=tuple(float(getattr(cfg, k)) for k in _LQR_Q_KEYS),
            r_diag=tuple(float(getattr(cfg, k)) for k in _LQR_R_KEYS),
            box_gravity=gravity,
            a_cmd_min=float(getattr(cfg, "a_cmd_min")),
            a_cmd_max=float(getattr(cfg, "a_cmd_max")),
            omega_max=float(getattr(cfg, "omega_max")),
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, fallback: Any) -> "QuadrotorDLQR":
        """Build from a saved hover-LQR payload (learned-policy metadata)."""
        def _value(key: str) -> float:
            if key in payload:
                return float(payload[key])
            return float(getattr(fallback, key))

        return cls._build(
            dt=_value("dt"),
            design_gravity=_value("gravity"),
            z_des=_value("z_des"),
            q_diag=tuple(_value(k) for k in _LQR_Q_KEYS),
            r_diag=tuple(_value(k) for k in _LQR_R_KEYS),
            box_gravity=float(getattr(fallback, "gravity")),
            a_cmd_min=float(getattr(fallback, "a_cmd_min")),
            a_cmd_max=float(getattr(fallback, "a_cmd_max")),
            omega_max=float(getattr(fallback, "omega_max")),
        )

    def error_state(self, x: jax.Array) -> jax.Array:
        """Reduced hover error e(x): altitude error, inertial velocity, and the
        first-order quaternion-error angles (2 * sign-corrected q_err vector)."""
        x_arr = jnp.asarray(x)
        q = normalize_quaternion_batch(x_arr[..., 6:10])
        q_err = quaternion_conjugate_batch(q)
        sign_term = jnp.where(q_err[..., 0] >= 0.0, 1.0, -1.0).astype(x_arr.dtype)
        theta_err = 2.0 * sign_term[..., None] * q_err[..., 1:4]
        z_des = jnp.asarray(self.z_des, dtype=x_arr.dtype)
        return jnp.concatenate(
            [
                (x_arr[..., 2:3] - z_des),
                x_arr[..., 3:6],
                theta_err,
            ],
            axis=-1,
        )
    
    def action(self, x: jax.Array) -> jax.Array:
        """Hover LQR u = u* - K e(x), clipped to the action box (verbatim)."""
        x_arr = jnp.asarray(x)
        u_eq = jnp.asarray(list(self.u_star), dtype=x_arr.dtype)
        u = u_eq - jnp.asarray(self.k_matrix, dtype=x_arr.dtype) @ self.error_state(x_arr)
        action_low = jnp.asarray(list(self.u_low), dtype=x_arr.dtype)
        action_high = jnp.asarray(list(self.u_high), dtype=x_arr.dtype)
        return jnp.clip(u, action_low, action_high)


__all__ = ["QuadrotorDLQR"]
