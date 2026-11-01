"""Generic certified base controller.

``BaseController`` is the abstract pairing partner of a certified base set B:
it supplies the feedback law pi_B(x), the error coordinates e(x) about the
equilibrium, and the largest input-feasible certificate level. 
``DiscreteLQR`` is a concrete instance — a discrete-time LQR designed about 
an equilibrium via the DARE.

Numerical note: ``_compute_certificate`` is an overridable hook so a subclass may
supply a system-specific gain pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_discrete_are as _solve_discrete_are_scipy


def euler_discretize(
    a_cont: Any, b_cont: Any, dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward-Euler discretization of a continuous LTI pair (numpy f64).

    Returns ``(A_d, B_d) = (I + dt * A, dt * B)``. 
    """
    a = np.asarray(a_cont, dtype=np.float64)
    b = np.asarray(b_cont, dtype=np.float64)
    dt = float(dt)
    a_d = np.eye(a.shape[0], dtype=np.float64) + dt * a
    b_d = dt * b
    return a_d, b_d


class BaseController(ABC):
    """Certified base controller pi_B paired with a base set B."""

    #: Largest base-set level c for which pi_B stays inside the input box on B.
    max_certified_level: float

    #: True iff d e(x)/dx is the identity map, so the base-set gradient can be
    #: written explicitly as -2 P e (see EllipsoidBaseSet).
    error_jacobian_is_identity: ClassVar[bool] = False

    @abstractmethod
    def action(self, x: jax.Array) -> jax.Array:
        """pi_B(x), clipped to the input box."""

    @abstractmethod
    def error_state(self, x: jax.Array) -> jax.Array:
        """Map the state to error coordinates e(x) about the equilibrium."""


@dataclass(frozen=True)
class DiscreteLQR(BaseController):
    """Discrete-time LQR base controller about an equilibrium (u*, e=0).

    System subclasses provide ``from_config`` constructors that map the named
    ground-truth config fields onto the generic tuples below and supply the
    linearization + error map. The certificate matrices P (DARE solution) and
    K (feedback gain) are stored at f32; ``max_certified_level`` is the
    largest ellipsoid level {e : e^T P e <= c} on which the *unclipped* LQR
    action stays inside [u_low, u_high].
    """

    a_d: Tuple[Tuple[float, ...], ...]
    b_d: Tuple[Tuple[float, ...], ...]
    q_diag: Tuple[float, ...]
    r_diag: Tuple[float, ...]
    u_star: Tuple[float, ...]
    u_low: Tuple[float, ...]
    u_high: Tuple[float, ...]
    p_matrix: jax.Array = field(init=False, repr=False)
    k_matrix: jax.Array = field(init=False, repr=False)
    max_certified_level: float = field(init=False)

    def __post_init__(self) -> None:
        a_d = np.asarray(self.a_d, dtype=np.float64)
        b_d = np.asarray(self.b_d, dtype=np.float64)
        q_diag = np.asarray(self.q_diag, dtype=np.float64)
        r_diag = np.asarray(self.r_diag, dtype=np.float64)
        u_star = np.asarray(self.u_star, dtype=np.float64)
        u_low = np.asarray(self.u_low, dtype=np.float64)
        u_high = np.asarray(self.u_high, dtype=np.float64)

        n, m = b_d.shape
        if a_d.shape != (n, n):
            raise ValueError(f"a_d must be ({n}, {n}) to match b_d {b_d.shape}, got {a_d.shape}")
        if q_diag.shape != (n,) or r_diag.shape != (m,):
            raise ValueError(
                f"q_diag/r_diag must have shapes ({n},)/({m},), got {q_diag.shape}/{r_diag.shape}"
            )
        if u_star.shape != (m,) or u_low.shape != (m,) or u_high.shape != (m,):
            raise ValueError(f"u_star/u_low/u_high must have shape ({m},)")
        if np.any(~np.isfinite(q_diag)) or np.any(q_diag <= 0.0):
            raise ValueError(f"LQR Q diagonal must be positive and finite, got {q_diag.tolist()}")
        if np.any(~np.isfinite(r_diag)) or np.any(r_diag <= 0.0):
            raise ValueError(f"LQR R diagonal must be positive and finite, got {r_diag.tolist()}")

        p_matrix, k_matrix, max_certified_level = self._compute_certificate()
        object.__setattr__(self, "p_matrix", p_matrix)
        object.__setattr__(self, "k_matrix", k_matrix)
        object.__setattr__(self, "max_certified_level", float(max_certified_level))

    # Per-input one-sided margins around the equilibrium input u*.
    def _input_margins(self) -> np.ndarray:
        u_star = np.asarray(self.u_star, dtype=np.float64)
        u_low = np.asarray(self.u_low, dtype=np.float64)
        u_high = np.asarray(self.u_high, dtype=np.float64)
        return np.minimum(u_star - u_low, u_high - u_star)

    def _compute_certificate(self) -> tuple[jax.Array, jax.Array, float]:
        """Solve the DARE and the input-feasibility bound at f64, store f32."""
        a_d = np.asarray(self.a_d, dtype=np.float64)
        b_d = np.asarray(self.b_d, dtype=np.float64)
        q_mat = np.diag(np.asarray(self.q_diag, dtype=np.float64))
        r_mat = np.diag(np.asarray(self.r_diag, dtype=np.float64))
        p_raw = _solve_discrete_are_scipy(a_d, b_d, q_mat, r_mat)
        p_raw = 0.5 * (p_raw.real + p_raw.real.T)
        k_raw = np.linalg.solve(r_mat + b_d.T @ p_raw @ b_d, b_d.T @ p_raw @ a_d)
        p_inv = np.linalg.inv(p_raw)

        margins = self._input_margins()
        if np.any(margins <= 0.0):
            raise ValueError(
                "The equilibrium input u_star must lie strictly inside [u_low, u_high]; "
                f"got one-sided margins {margins.tolist()}."
            )
        max_levels = []
        for row_idx in range(k_raw.shape[0]):
            k_row = k_raw[row_idx : row_idx + 1, :]
            denom = float((k_row @ p_inv @ k_row.T).item())
            if denom <= 0.0:
                max_levels.append(np.inf)
            else:
                max_levels.append(float(margins[row_idx]) ** 2 / denom)
        max_certified_level = float(min(max_levels))
        return (
            jnp.asarray(p_raw, dtype=jnp.float32),
            jnp.asarray(k_raw, dtype=jnp.float32),
            max_certified_level,
        )

    def action(self, x: jax.Array) -> jax.Array:
        """pi_B(x) = u* - K e(x), clipped to the input box.

        Generic default for user extensions.
        """
        err = self.error_state(jnp.asarray(x))
        k_mat = jnp.asarray(self.k_matrix, dtype=err.dtype)
        u_star = jnp.asarray(self.u_star, dtype=err.dtype)
        u = u_star - jnp.einsum("ij,...j->...i", k_mat, err)
        action_low = jnp.asarray(self.u_low, dtype=err.dtype)
        action_high = jnp.asarray(self.u_high, dtype=err.dtype)
        return jnp.clip(u, action_low, action_high)

    def quadratic_form(self, x: jax.Array) -> jax.Array:
        """e(x)^T P e(x) — the base-set certificate value."""
        err = self.error_state(jnp.asarray(x))
        p_mat = jnp.asarray(self.p_matrix, dtype=err.dtype)
        return jnp.einsum("...i,ij,...j->...", err, p_mat, err)


__all__ = ["BaseController", "DiscreteLQR"]
