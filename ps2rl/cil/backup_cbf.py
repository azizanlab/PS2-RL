"""Shared backup-CBF engine.

Everything system-specific enters through a ``BCBFSystem`` bundle (dynamics,
backup policy, set rows, dims, action box, state sanitizers); everything
scalar (dt, num_steps, alpha, weights, solver params, row conditioning) is
read duck-typed off the system's backup-CBF config, whose field names are
shared across the two systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import qpax


@dataclass(frozen=True)
class BCBFSystem:
    """System bundle consumed by the shared engine."""

    state_dim: int
    action_dim: int
    action_low: Tuple[float, ...]
    action_high: Tuple[float, ...]
    backup_policy_fn: Callable[[jax.Array], jax.Array]
    safe_set_values_and_grads_fn: Callable[[jax.Array], Tuple[jax.Array, jax.Array]]
    base_set_values_and_grads_fn: Callable[[jax.Array], Tuple[jax.Array, jax.Array]]
    #: x_dot = f(x) + g(x) u with the system's exact historical clip placement.
    dynamics_fn: Callable[[jax.Array, jax.Array], jax.Array]
    control_affine_terms_fn: Callable[[jax.Array], Tuple[jax.Array, jax.Array]]
    #: Applied to x + dt * f_pi inside the backup rollout scan.
    postprocess_rollout_state_fn: Callable[[jax.Array], jax.Array]
    #: Entry-state preparation for one QP solve (slice/nan_to_num/renorm).
    sanitize_solve_state_fn: Callable[[jax.Array], jax.Array]
    #: Batch-level state slice applied before vmapping the single solve.
    solve_state_slice_fn: Callable[[jax.Array], jax.Array]
    #: Terminal-set values-only callable (quadrotor public API); optional.
    base_set_values_fn: Callable[[jax.Array], jax.Array] | None = None
    use_analytic_jacobian: bool = False
    analytic_closed_loop_and_jacobian_fn: Callable[[jax.Array], Tuple[jax.Array, jax.Array]] | None = None


@dataclass(frozen=True)
class BackupCBFConfig:
    """Shared backup-CBF configuration base.

    Holds the fields the shared engine in this module reads duck-typed off the
    config: the backup-flow horizon/mesh, class-K gains, the QP objective
    weights, the qpax solver params, and the constraint-row conditioning knobs.
    ``Unicycle/QuadrotorBCBFConfig`` subclass this, add their own state /
    actuator / LQR / PID fields, and keep their own ``__post_init__`` (the
    T/dt/num_steps resolution differs).

    Defaults are the unicycle values; ``QuadrotorBCBFConfig`` re-declares the
    six that differ (``num_steps``, ``alpha``, ``base_alpha``, ``base_set_c``,
    ``solver_tol``, ``use_analytic_jacobian``) to override them.
    """

    # Backup policy backend: "analytic" or "learned".
    backup_policy_mode: str = "analytic"
    learned_backup_policy_path: str = ""

    # Backup-CBF discretization: t = tau_0 < ... < tau_N = t + T.
    T: float | None = None
    num_steps: int = 40
    dt: float | None = None
    # alpha is the safe-set CBF row's class-K gain.
    alpha: float = 1.0
    # d/dt h_C(Phi(x(t), tau-t)), which introduces the
    # relative-time correction term -f_pi(Phi(...)).
    include_relative_time_term: bool = True

    # Base set B: the LQR ellipsoid that is both the BCBF terminal set 
    # and the backup policy's hand-off region; always enforced.
    # base_alpha is the base-set CBF row's class-K gain.
    base_set_c: float = 0.3
    base_alpha: float = 1.0

    # QP objective.
    control_weight: float = 1.0

    # qpax solver params.
    solver_tol: float = 1e-3
    target_kappa: float = 1e-2

    # Sensitivity rollout / row conditioning. 
    use_analytic_jacobian: bool = True
    sensitivity_clip: float = 1e6
    constraint_row_normalize: bool = True
    constraint_row_scale_floor: float = 1.0
    max_abs_constraint_value: float = 1e6

    @property
    def horizon(self) -> float:
        return float(self.T)


def closed_loop_backup_dynamics(x: jax.Array, system: BCBFSystem) -> jax.Array:
    return system.dynamics_fn(x, system.backup_policy_fn(x))


def rollout_backup_flow_and_sensitivity_with_info(
    x0: jax.Array,
    cfg: Any,
    system: BCBFSystem,
) -> Tuple[jax.Array, jax.Array, Dict[str, jax.Array]]:
    """Rollout backup flow and Q-sensitivity with diagnostics.

    Sensitivity propagation is Q_{k+1} = exp(dt * J_k) Q_k with optional
    Frobenius-norm clipping.
    """
    dt_j = jnp.array(cfg.dt, dtype=x0.dtype)
    clip_enabled = bool(cfg.sensitivity_clip > 0.0)
    clip_j = jnp.array(cfg.sensitivity_clip if clip_enabled else 0.0, dtype=x0.dtype)
    use_analytic = bool(
        cfg.use_analytic_jacobian and system.use_analytic_jacobian and system.analytic_closed_loop_and_jacobian_fn
    )
    jac_backup = jax.jacfwd(lambda z: closed_loop_backup_dynamics(z, system))

    one_j = jnp.array(1.0, dtype=x0.dtype)
    eps_j = jnp.array(1e-12, dtype=x0.dtype)

    def propagate_q(q: jax.Array, j_pi: jax.Array) -> jax.Array:
        return jsp.linalg.expm(dt_j * j_pi) @ q

    def clip_q(q_pred: jax.Array) -> jax.Array:
        fro = jnp.linalg.norm(q_pred, ord="fro")
        scale = jnp.minimum(one_j, clip_j / (fro + eps_j))
        return q_pred * scale

    def step(carry, _):
        x, q, max_abs_q, q_saturated = carry
        if use_analytic:
            f_pi, j_pi = system.analytic_closed_loop_and_jacobian_fn(x)
        else:
            f_pi = closed_loop_backup_dynamics(x, system)
            j_pi = jac_backup(x)
        x_next = system.postprocess_rollout_state_fn(x + dt_j * f_pi)

        q_pred = propagate_q(q, j_pi)
        step_max_abs_q = jnp.max(jnp.abs(q_pred))
        max_abs_q_next = jnp.maximum(max_abs_q, step_max_abs_q)
        if clip_enabled:
            q_saturated_next = q_saturated | (step_max_abs_q >= clip_j)
            q_next = clip_q(q_pred)
        else:
            q_saturated_next = q_saturated
            q_next = q_pred

        return (x_next, q_next, max_abs_q_next, q_saturated_next), (x_next, q_next)

    q0 = jnp.eye(system.state_dim, dtype=x0.dtype)
    q0_abs = jnp.max(jnp.abs(q0))
    q_saturated0 = jnp.asarray((q0_abs >= clip_j) if clip_enabled else False, dtype=jnp.bool_)
    (_, _, max_abs_q_final, q_saturated_final), (xs_tail, qs_tail) = jax.lax.scan(
        step, (x0, q0, q0_abs, q_saturated0), jnp.arange(cfg.num_steps)
    )
    xs = jnp.concatenate([x0[None, :], xs_tail], axis=0)
    qs = jnp.concatenate([q0[None, :, :], qs_tail], axis=0)
    diag = {
        "max_abs_q": max_abs_q_final.astype(x0.dtype),
        "q_saturated": jnp.asarray(q_saturated_final, dtype=jnp.bool_),
    }
    return xs, qs, diag


def build_discretized_backup_cbf_rows_with_info(
    x: jax.Array,
    cfg: Any,
    system: BCBFSystem,
) -> Tuple[jax.Array, jax.Array, Dict[str, jax.Array]]:
    """Build A u <= b from discretized backup-CBF constraints with diagnostics."""
    xs, qs, rollout_diag = rollout_backup_flow_and_sensitivity_with_info(x, cfg, system)
    f0, g0 = system.control_affine_terms_fn(x)

    def per_node_rows(xi, qi):
        h_c, dh_c = system.safe_set_values_and_grads_fn(xi)
        qi_g0 = qi @ g0
        qi_f0 = qi @ f0
        a = -(dh_c @ qi_g0)
        f_pi_i = closed_loop_backup_dynamics(xi, system)
        flow_term = qi_f0 - f_pi_i if cfg.include_relative_time_term else qi_f0
        b = cfg.alpha * h_c + dh_c @ flow_term
        return a, b

    a_seq, b_seq = jax.vmap(per_node_rows, in_axes=(0, 0))(xs, qs)
    a_rows = a_seq.reshape((-1, system.action_dim))
    b_rows = b_seq.reshape((-1,))

    x_t = xs[-1]
    q_t = qs[-1]
    h_s, dh_s = system.base_set_values_and_grads_fn(x_t)
    q_t_g0 = q_t @ g0
    q_t_f0 = q_t @ f0
    a_term = -(dh_s @ q_t_g0)
    b_term = cfg.base_alpha * h_s + dh_s @ q_t_f0
    a_rows = jnp.concatenate([a_rows, a_term], axis=0)
    b_rows = jnp.concatenate([b_rows, b_term], axis=0)

    diag = {
        "max_abs_q": rollout_diag["max_abs_q"],
        "q_saturated": rollout_diag["q_saturated"],
        "max_abs_b": jnp.max(jnp.abs(b_rows)).astype(x.dtype),
    }
    return a_rows, b_rows, diag


def _build_backup_cbf_qp_from_rows(
    a_rows: jax.Array,
    b_rows: jax.Array,
    u_ref: jax.Array,
    cfg: Any,
    system: BCBFSystem,
    *,
    dtype: jnp.dtype,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build qpax matrices from precomputed backup-CBF rows."""
    m = system.action_dim
    g_cbf = a_rows
    h_cbf = b_rows

    if cfg.constraint_row_normalize:
        row_scale = jnp.maximum(
            jnp.array(cfg.constraint_row_scale_floor, dtype=dtype),
            jnp.max(jnp.abs(g_cbf), axis=1, keepdims=True),
        )
        g_cbf = g_cbf / row_scale
        h_cbf = h_cbf / row_scale[:, 0]
    if cfg.max_abs_constraint_value > 0.0:
        max_abs = jnp.array(cfg.max_abs_constraint_value, dtype=dtype)
        g_cbf = jnp.clip(g_cbf, -max_abs, max_abs)
        h_cbf = jnp.clip(h_cbf, -max_abs, max_abs)

    # Box rows.
    g_box_rows: list[list[float]] = []
    h_box_vals: list[float] = []
    for i in range(m):
        row_hi = [0.0] * m
        row_hi[i] = 1.0
        g_box_rows.append(row_hi)
        h_box_vals.append(float(system.action_high[i]))
        row_lo = [0.0] * m
        row_lo[i] = -1.0
        g_box_rows.append(row_lo)
        h_box_vals.append(-float(system.action_low[i]))
    g_box = jnp.array(g_box_rows, dtype=dtype)
    h_box = jnp.array(h_box_vals, dtype=dtype)

    g = jnp.concatenate([g_cbf, g_box], axis=0)
    h = jnp.concatenate([h_cbf, h_box], axis=0)

    q_mat = jnp.diag(jnp.array([cfg.control_weight] * m, dtype=dtype))
    q_vec = -jnp.array([cfg.control_weight * u_ref[i] for i in range(m)], dtype=dtype)
    a_eq = jnp.zeros((0, m), dtype=dtype)
    b_eq = jnp.zeros((0,), dtype=dtype)
    return q_mat, q_vec, a_eq, b_eq, g, h


def _default_cbf_finite_info(
    *,
    dtype: jnp.dtype,
    q_saturated: jax.Array,
    max_abs_q: jax.Array,
    max_abs_b: jax.Array,
    delta_min_u_ref: jax.Array,
) -> Dict[str, jax.Array]:
    ones = jnp.asarray(True, dtype=jnp.bool_)
    return {
        "q_mat_finite": ones,
        "q_vec_finite": ones,
        "g_finite": ones,
        "h_finite": ones,
        "inputs_finite": ones,
        "z_finite": ones,
        "q_saturated": jnp.asarray(q_saturated, dtype=jnp.bool_),
        "max_abs_q": jnp.asarray(max_abs_q, dtype=dtype),
        "max_abs_b": jnp.asarray(max_abs_b, dtype=dtype),
        "delta_min_u_ref": jnp.asarray(delta_min_u_ref, dtype=dtype),
    }


def solve_backup_cbf_qp_single_with_info(
    x: jax.Array,
    u_ref: jax.Array,
    cfg: Any,
    system: BCBFSystem,
) -> Tuple[jax.Array, jax.Array, Dict[str, jax.Array]]:
    """Solve one backup-CBF QP and report solver usage + finiteness diagnostics."""
    x = system.sanitize_solve_state_fn(x)
    u_ref = jnp.nan_to_num(u_ref, nan=0.0, posinf=0.0, neginf=0.0)
    u_ref = jnp.clip(
        u_ref,
        jnp.array([float(v) for v in system.action_low], dtype=x.dtype),
        jnp.array([float(v) for v in system.action_high], dtype=x.dtype),
    )
    fallback_u = system.backup_policy_fn(x)

    a_rows, b_rows, cbf_diag = build_discretized_backup_cbf_rows_with_info(x, cfg, system)
    max_abs_q = cbf_diag["max_abs_q"]
    q_saturated = cbf_diag["q_saturated"]
    max_abs_b = cbf_diag["max_abs_b"]
    residual_u_ref = a_rows @ u_ref - b_rows
    delta_min_u_ref = jnp.maximum(0.0, jnp.max(residual_u_ref)).astype(x.dtype)

    def _skip_solver(_: None):
        use_solver_skip = jnp.asarray(False, dtype=jnp.bool_)
        finite_info_skip = _default_cbf_finite_info(
            dtype=x.dtype,
            q_saturated=q_saturated,
            max_abs_q=max_abs_q,
            max_abs_b=max_abs_b,
            delta_min_u_ref=delta_min_u_ref,
        )
        return fallback_u, use_solver_skip, finite_info_skip

    def _solve_or_fallback(_: None):
        q_mat, q_vec, a_eq, b_eq, g, h = _build_backup_cbf_qp_from_rows(
            a_rows,
            b_rows,
            u_ref,
            cfg,
            system,
            dtype=x.dtype,
        )
        q_mat_finite = jnp.all(jnp.isfinite(q_mat))
        q_vec_finite = jnp.all(jnp.isfinite(q_vec))
        g_finite = jnp.all(jnp.isfinite(g))
        h_finite = jnp.all(jnp.isfinite(h))
        inputs_finite = q_mat_finite & q_vec_finite & g_finite & h_finite

        def _solve_qp(_: None) -> jax.Array:
            return qpax.solve_qp_primal(
                q_mat,
                q_vec,
                a_eq,
                b_eq,
                g,
                h,
                solver_tol=cfg.solver_tol,
                target_kappa=cfg.target_kappa,
            )

        z_candidate = jax.lax.cond(inputs_finite, _solve_qp, lambda _: fallback_u, operand=None)
        z_finite = inputs_finite & jnp.all(jnp.isfinite(z_candidate))
        use_solver = inputs_finite & z_finite
        z_out = jnp.where(use_solver, z_candidate, fallback_u)
        finite_info = {
            "q_mat_finite": jnp.asarray(q_mat_finite, dtype=jnp.bool_),
            "q_vec_finite": jnp.asarray(q_vec_finite, dtype=jnp.bool_),
            "g_finite": jnp.asarray(g_finite, dtype=jnp.bool_),
            "h_finite": jnp.asarray(h_finite, dtype=jnp.bool_),
            "inputs_finite": jnp.asarray(inputs_finite, dtype=jnp.bool_),
            "z_finite": jnp.asarray(z_finite, dtype=jnp.bool_),
            "q_saturated": jnp.asarray(q_saturated, dtype=jnp.bool_),
            "max_abs_q": jnp.asarray(max_abs_q, dtype=x.dtype),
            "max_abs_b": jnp.asarray(max_abs_b, dtype=x.dtype),
            "delta_min_u_ref": jnp.asarray(delta_min_u_ref, dtype=x.dtype),
        }
        return z_out, jnp.asarray(use_solver, dtype=jnp.bool_), finite_info

    z, use_solver, finite_info = jax.lax.cond(q_saturated, _skip_solver, _solve_or_fallback, operand=None)
    z = jnp.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    u = jnp.clip(
        z,
        jnp.array([float(v) for v in system.action_low], dtype=x.dtype),
        jnp.array([float(v) for v in system.action_high], dtype=x.dtype),
    )
    use_solver = jnp.asarray(use_solver, dtype=jnp.bool_)
    u_ref_minus_u_safe = u_ref - u
    u_ref_minus_u_safe_norm = jnp.linalg.norm(u_ref_minus_u_safe, ord=2).astype(x.dtype)
    a_ref_minus_a_safe = jnp.asarray(u_ref_minus_u_safe[0], dtype=x.dtype)
    r_ref_minus_r_safe = jnp.asarray(u_ref_minus_u_safe[1], dtype=x.dtype)
    finite_info = {
        **finite_info,
        "u_ref_minus_u_safe_norm": jnp.asarray(u_ref_minus_u_safe_norm, dtype=x.dtype),
        "a_ref_minus_a_safe": jnp.asarray(a_ref_minus_a_safe, dtype=x.dtype),
        "r_ref_minus_r_safe": jnp.asarray(r_ref_minus_r_safe, dtype=x.dtype),
    }
    return u, use_solver, finite_info


def solve_backup_cbf_qp_batch_with_info(
    x_batch: jax.Array,
    u_ref_batch: jax.Array,
    cfg: Any,
    system: BCBFSystem,
) -> Tuple[jax.Array, jax.Array, Dict[str, jax.Array]]:
    """Batch backup-CBF solve with per-sample solver usage and finiteness info."""
    x_batch = system.solve_state_slice_fn(x_batch)
    fn = lambda x, u: solve_backup_cbf_qp_single_with_info(x, u, cfg, system)
    u_safe, use_solver, finite_info = jax.vmap(fn, in_axes=(0, 0))(x_batch, u_ref_batch)
    return u_safe, use_solver, finite_info


def constraint_residuals(
    x: jax.Array,
    u: jax.Array,
    cfg: Any,
    system: BCBFSystem,
) -> jax.Array:
    """Residual A u - b for backup-CBF rows (positive means violation)."""
    a_rows, b_rows, _ = build_discretized_backup_cbf_rows_with_info(x, cfg, system)
    return a_rows @ u - b_rows


def make_backup_cbf_facades(
    make_runtime: Callable[[Any], BCBFSystem],
    runtime_cache_key: Callable[[Any], tuple],
    cache: Dict[tuple, BCBFSystem],
) -> Dict[str, Callable]:
    """Build the shared cfg-keyed facade layer over the engine functions above."""

    def _f_get_cached_runtime(cfg):
        key = runtime_cache_key(cfg)
        if key not in cache:
            cache[key] = make_runtime(cfg)
        return cache[key]

    def _f_resolve_runtime(cfg, runtime=None):
        if runtime is not None:
            return runtime
        return _f_get_cached_runtime(cfg)

    def _f_backup_policy(x, cfg, runtime=None):
        rt = _f_resolve_runtime(cfg, runtime)
        return rt.backup_policy_fn(x)

    def _f_backup_policy_batch(x_batch, cfg, runtime=None):
        fn = lambda x: _f_backup_policy(x, cfg, runtime=runtime)
        return jax.vmap(fn, in_axes=0)(x_batch)

    def _f_closed_loop_backup_dynamics(x, cfg, runtime=None):
        return closed_loop_backup_dynamics(x, _f_resolve_runtime(cfg, runtime))

    def _f_safe_set_values_and_grads(x, cfg, runtime=None):
        rt = _f_resolve_runtime(cfg, runtime)
        return rt.safe_set_values_and_grads_fn(x)

    def _f_base_set_values_and_grads(x, cfg, runtime=None):
        rt = _f_resolve_runtime(cfg, runtime)
        return rt.base_set_values_and_grads_fn(x)

    def _f_rollout_backup_flow_and_sensitivity_with_info(x0, cfg, runtime=None):
        return rollout_backup_flow_and_sensitivity_with_info(x0, cfg, _f_resolve_runtime(cfg, runtime))

    def _f_rollout_backup_flow_and_sensitivity(x0, cfg, runtime=None):
        xs, qs, _ = _f_rollout_backup_flow_and_sensitivity_with_info(x0, cfg, runtime=runtime)
        return xs, qs

    def _f_build_discretized_backup_cbf_rows_with_info(x, cfg, runtime=None):
        return build_discretized_backup_cbf_rows_with_info(x, cfg, _f_resolve_runtime(cfg, runtime))

    def _f_build_discretized_backup_cbf_rows(x, cfg, runtime=None):
        a_rows, b_rows, _ = _f_build_discretized_backup_cbf_rows_with_info(x, cfg, runtime=runtime)
        return a_rows, b_rows

    def _f_build_backup_cbf_qp(x, u_ref, cfg, runtime=None):
        rt = _f_resolve_runtime(cfg, runtime)
        dtype = x.dtype
        a_rows, b_rows = _f_build_discretized_backup_cbf_rows(x, cfg, runtime=rt)
        return _build_backup_cbf_qp_from_rows(a_rows, b_rows, u_ref, cfg, rt, dtype=dtype)

    def _f_solve_backup_cbf_qp_single_with_info(x, u_ref, cfg, runtime=None):
        return solve_backup_cbf_qp_single_with_info(x, u_ref, cfg, _f_resolve_runtime(cfg, runtime))

    def _f_solve_backup_cbf_qp_single(x, u_ref, cfg, runtime=None):
        u, _, _ = _f_solve_backup_cbf_qp_single_with_info(x, u_ref, cfg, runtime=runtime)
        return u

    def _f_solve_backup_cbf_qp_batch_with_info(x_batch, u_ref_batch, cfg, runtime=None):
        return solve_backup_cbf_qp_batch_with_info(x_batch, u_ref_batch, cfg, _f_resolve_runtime(cfg, runtime))

    def _f_solve_backup_cbf_qp_batch(x_batch, u_ref_batch, cfg, runtime=None):
        u_safe, _, _ = _f_solve_backup_cbf_qp_batch_with_info(x_batch, u_ref_batch, cfg, runtime=runtime)
        return u_safe

    def _f_constraint_residuals(x, u, cfg, runtime=None):
        return constraint_residuals(x, u, cfg, _f_resolve_runtime(cfg, runtime))

    return {
        "get_cached_runtime": _f_get_cached_runtime,
        "_resolve_runtime": _f_resolve_runtime,
        "backup_policy": _f_backup_policy,
        "backup_policy_batch": _f_backup_policy_batch,
        "closed_loop_backup_dynamics": _f_closed_loop_backup_dynamics,
        "safe_set_values_and_grads": _f_safe_set_values_and_grads,
        "base_set_values_and_grads": _f_base_set_values_and_grads,
        "rollout_backup_flow_and_sensitivity": _f_rollout_backup_flow_and_sensitivity,
        "rollout_backup_flow_and_sensitivity_with_info": _f_rollout_backup_flow_and_sensitivity_with_info,
        "build_discretized_backup_cbf_rows": _f_build_discretized_backup_cbf_rows,
        "build_discretized_backup_cbf_rows_with_info": _f_build_discretized_backup_cbf_rows_with_info,
        "build_backup_cbf_qp": _f_build_backup_cbf_qp,
        "solve_backup_cbf_qp_single": _f_solve_backup_cbf_qp_single,
        "solve_backup_cbf_qp_single_with_info": _f_solve_backup_cbf_qp_single_with_info,
        "solve_backup_cbf_qp_batch": _f_solve_backup_cbf_qp_batch,
        "solve_backup_cbf_qp_batch_with_info": _f_solve_backup_cbf_qp_batch_with_info,
        "constraint_residuals": _f_constraint_residuals,
    }


class BackupCBFProjector:
    """Convenience wrapper with jitted single/batch projection calls."""

    def __init__(self, cfg: Any, system: BCBFSystem):
        self.cfg = cfg
        self.runtime = system
        self._solve_single = jax.jit(
            lambda x, u_ref: solve_backup_cbf_qp_single_with_info(x, u_ref, self.cfg, self.runtime)[0]
        )
        self._solve_single_with_info = jax.jit(
            lambda x, u_ref: solve_backup_cbf_qp_single_with_info(x, u_ref, self.cfg, self.runtime)
        )
        self._solve_batch = jax.jit(
            lambda x_b, u_b: solve_backup_cbf_qp_batch_with_info(x_b, u_b, self.cfg, self.runtime)[0]
        )
        self._solve_batch_with_info = jax.jit(
            lambda x_b, u_b: solve_backup_cbf_qp_batch_with_info(x_b, u_b, self.cfg, self.runtime)
        )
        self._residual = jax.jit(
            lambda x, u: constraint_residuals(x, u, self.cfg, self.runtime)
        )

    def solve_single(self, x: jax.Array, u_ref: jax.Array) -> jax.Array:
        return self._solve_single(x, u_ref)

    def solve_single_with_info(
        self, x: jax.Array, u_ref: jax.Array
    ) -> Tuple[jax.Array, jax.Array, Dict[str, jax.Array]]:
        return self._solve_single_with_info(x, u_ref)

    def solve_batch(self, x_batch: jax.Array, u_ref_batch: jax.Array) -> jax.Array:
        return self._solve_batch(x_batch, u_ref_batch)

    def solve_batch_with_info(
        self, x_batch: jax.Array, u_ref_batch: jax.Array
    ) -> Tuple[jax.Array, jax.Array, Dict[str, jax.Array]]:
        return self._solve_batch_with_info(x_batch, u_ref_batch)

    def residuals(self, x: jax.Array, u: jax.Array) -> jax.Array:
        return self._residual(x, u)

    @property
    def num_backup_inequalities(self) -> int:
        return self.cfg.num_backup_inequalities

    @property
    def num_qp_inequalities(self) -> int:
        return self.cfg.num_qp_inequalities


__all__ = [
    "BCBFSystem",
    "BackupCBFConfig",
    "BackupCBFProjector",
    "build_discretized_backup_cbf_rows_with_info",
    "closed_loop_backup_dynamics",
    "constraint_residuals",
    "make_backup_cbf_facades",
    "rollout_backup_flow_and_sensitivity_with_info",
    "solve_backup_cbf_qp_batch_with_info",
    "solve_backup_cbf_qp_single_with_info",
]
