"""Unicycle discrete-time LQR base controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.base_controller.base_controller import DiscreteLQR, euler_discretize


@dataclass(frozen=True)
class UnicycleDLQR(DiscreteLQR):
    """Discrete-time LQR about the lane-cruise equilibrium x* = [0, v_des, 0].

    The lane error map e(x) = [y, v - v_des, psi] is linear with identity
    Jacobian, so the paired EllipsoidBaseSet uses the explicit -2 P e gradient.
    """

    v_des: float = 0.0  # overwritten by from_params; default only for dataclass ordering

    error_jacobian_is_identity = True

    @classmethod
    def from_params(
        cls,
        *,
        v_des: float,
        dt: float,
        a_max: float,
        r_max: float,
        q_y: float,
        q_v: float,
        q_psi: float,
        r_a: float,
        r_r: float,
    ) -> "UnicycleDLQR":
        v_des = float(v_des)
        dt = float(dt)
        a_max = float(a_max)
        r_max = float(r_max)
        if v_des <= 0.0:
            raise ValueError(f"v_des must be positive, got {v_des}")
        if (not np.isfinite(dt)) or dt <= 0.0:
            raise ValueError(f"dt must be positive and finite, got {dt}")
        if (not np.isfinite(a_max)) or a_max <= 0.0:
            raise ValueError(f"a_max must be positive and finite, got {a_max}")
        if (not np.isfinite(r_max)) or r_max <= 0.0:
            raise ValueError(f"r_max must be positive and finite, got {r_max}")


        # Continuous lane linearization about the cruise equilibrium, then a
        # shared forward-Euler discretization (see ``euler_discretize``).
        a_mat = np.array(
            [
                [0.0, 0.0, v_des],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        b_mat = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        a_d, b_d = euler_discretize(a_mat, b_mat, dt)
        return cls(
            a_d=tuple(tuple(row) for row in a_d.tolist()),
            b_d=tuple(tuple(row) for row in b_d.tolist()),
            q_diag=(float(q_y), float(q_v), float(q_psi)),
            r_diag=(float(r_a), float(r_r)),
            u_star=(0.0, 0.0),
            u_low=(-a_max, -r_max),
            u_high=(a_max, r_max),
            v_des=v_des,
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "UnicycleDLQR":
        """Canonical constructor for the DLQR from a config."""
        return cls.from_params(
            v_des=float(getattr(cfg, "v_des")),
            dt=float(getattr(cfg, "dt")),
            a_max=float(getattr(cfg, "a_max")),
            r_max=float(getattr(cfg, "r_max")),
            q_y=float(getattr(cfg, "lqr_q_y")),
            q_v=float(getattr(cfg, "lqr_q_v")),
            q_psi=float(getattr(cfg, "lqr_q_psi")),
            r_a=float(getattr(cfg, "lqr_r_a")),
            r_r=float(getattr(cfg, "lqr_r_r")),
        )

    def error_state(self, x: jax.Array) -> jax.Array:
        x_arr = jnp.asarray(x)
        v_des = jnp.asarray(self.v_des, dtype=x_arr.dtype)
        return jnp.stack(
            [
                x_arr[..., 0],
                x_arr[..., 1] - v_des,
                x_arr[..., 2],
            ],
            axis=-1,
        )
    
    def action(self, x: jax.Array) -> jax.Array:
        err = self.error_state(x)
        k_mat = jnp.asarray(self.k_matrix, dtype=err.dtype)
        u = -jnp.einsum("ij,...j->...i", k_mat, err)
        action_low = jnp.asarray(self.u_low, dtype=err.dtype)
        action_high = jnp.asarray(self.u_high, dtype=err.dtype)
        return jnp.clip(u, action_low, action_high)

    # Live: base-set containment checks (unicycle_backup_cbf base-set radii).
    def coordinate_abs_max(self, *, level: float, index: int) -> float:
        """Largest |e_index| on the level-c ellipsoid (base-set containment checks)."""
        p_inv = np.linalg.inv(np.asarray(self.p_matrix, dtype=np.float64))
        return float(np.sqrt(max(float(level) * float(p_inv[index, index]), 0.0)))


__all__ = ["UnicycleDLQR"]
