"""Unicycle backup-CBF system: config, dynamics, payload translators, assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.backup_policy.unicycle_analytic_backup import (
    UnicycleABP,
    clip_backup_action,
)
from ps2rl.backup_policy.unicycle_learned_backup import (
    UnicycleLBP,
    load_learned_unicycle_backup_policy,
)
from ps2rl.base_controller.unicycle_dlqr import UnicycleDLQR
from ps2rl.cil.backup_cbf import (
    BCBFSystem,
    BackupCBFConfig,
    BackupCBFProjector,
    make_backup_cbf_facades,
)
from ps2rl.sets.base_sets import EllipsoidBaseSet
from ps2rl.sets.unicycle_sets import UnicycleSafeSet
from ps2rl.envs.unicycle_env import (
    unicycle_control_affine_terms as control_affine_terms,
    unicycle_dynamics,
    postprocess_unicycle_state,
)
from ps2rl.envs.unicycle_env import unicycle_step_euler  # noqa: F401


@dataclass(frozen=True)
class UnicycleBCBFConfig(BackupCBFConfig):
    """Configuration for backup-CBF projection."""

    # Safety set C: |Y| <= y_max, |psi| <= psi_max
    y_max: float = 1.8
    psi_max: float = pi / 3.0

    # Input bounds U
    a_max: float = 5.0
    r_max: float = 1.0
    v_min: float = 0.0
    v_max: float = 12.0

    # Equilibrium speed of the LQR base controller / analytic backup policy.
    v_des: float = 5.0

    # LQR base-set weights (base set B is the LQR ellipsoid
    # {x : e(x)^T P e(x) <= base_set_c}).
    lqr_q_y: float = 1.0
    lqr_q_v: float = 1.0
    lqr_q_psi: float = 1.0
    lqr_r_a: float = 1.0
    lqr_r_r: float = 1.0

    def __post_init__(self) -> None:
        n = int(self.num_steps)
        if n <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")

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
            # Dynamics use (dt, num_steps); keep T consistent for reporting/logging.
            t = dt * n
        if (not isfinite(float(self.v_min))) or (not isfinite(float(self.v_max))) or float(self.v_min) > float(self.v_max):
            raise ValueError(f"Expected finite speed bounds with v_min <= v_max, got ({self.v_min}, {self.v_max})")
        object.__setattr__(self, "num_steps", n)
        object.__setattr__(self, "T", float(t))
        object.__setattr__(self, "dt", float(dt))

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

    @property
    def num_backup_inequalities(self) -> int:
        # + 1: the base set contributes a single ellipsoid constraint row.
        return 4 * (self.num_steps + 1) + 1

    @property
    def num_qp_inequalities(self) -> int:
        # Four actuator bounds: acceleration and yaw rate, upper and lower.
        return self.num_backup_inequalities + 4

    @property
    def lqr_q_diag(self) -> tuple[float, float, float]:
        return (float(self.lqr_q_y), float(self.lqr_q_v), float(self.lqr_q_psi))

    @property
    def lqr_r_diag(self) -> tuple[float, float]:
        return (float(self.lqr_r_a), float(self.lqr_r_r))


_RUNTIME_CACHE: dict[tuple, BCBFSystem] = {}


def _unicycle_lqr_design_from_cbf(cfg: UnicycleBCBFConfig) -> UnicycleDLQR:
    return UnicycleDLQR.from_config(cfg)


def _unicycle_lqr_config_from_cbf(cfg: UnicycleBCBFConfig) -> dict[str, float]:
    return {
        "v_des": float(cfg.v_des),
        "dt": float(cfg.dt),
        "a_max": float(cfg.a_max),
        "r_max": float(cfg.r_max),
        "r_matrix_diag_a": float(cfg.lqr_r_a),
        "r_matrix_diag_r": float(cfg.lqr_r_r),
        "lqr_q_y": float(cfg.lqr_q_y),
        "lqr_q_v": float(cfg.lqr_q_v),
        "lqr_q_psi": float(cfg.lqr_q_psi),
        "lqr_r_a": float(cfg.lqr_r_a),
        "lqr_r_r": float(cfg.lqr_r_r),
    }


def _unicycle_base_set_from_cbf(
    cfg: UnicycleBCBFConfig,
    *,
    lqr_design: UnicycleDLQR | None = None,
) -> EllipsoidBaseSet:
    design = lqr_design if lqr_design is not None else UnicycleDLQR.from_config(cfg)
    return EllipsoidBaseSet(design, float(cfg.base_set_c))


def _runtime_cache_key(cfg: UnicycleBCBFConfig) -> tuple:
    return (
        cfg.backup_policy_mode,
        cfg.learned_backup_policy_path,
        cfg.v_des,
        cfg.a_max,
        cfg.r_max,
        cfg.y_max,
        cfg.psi_max,
        cfg.base_set_c,
        cfg.lqr_q_y,
        cfg.lqr_q_v,
        cfg.lqr_q_psi,
        cfg.lqr_r_a,
        cfg.lqr_r_r,
        cfg.use_analytic_jacobian,
    )

def validate_unicycle_base_config(
    *,
    y_max: float,
    psi_max: float,
    dt: float,
    a_max: float,
    r_max: float,
    base_set_c: float,
    v_des: float,
    lqr_q_y: float,
    lqr_q_v: float,
    lqr_q_psi: float,
    lqr_r_a: float,
    lqr_r_r: float,
) -> None:
    """Validate the LQR base set B: positive level, actuator-admissible, inside C."""
    base_set_c = float(base_set_c)
    if (not isfinite(base_set_c)) or base_set_c <= 0.0:
        raise ValueError(f"base_set_c must be positive and finite, got {base_set_c}")

    lqr_design = UnicycleDLQR.from_params(
        v_des=v_des,
        dt=dt,
        a_max=a_max,
        r_max=r_max,
        q_y=lqr_q_y,
        q_v=lqr_q_v,
        q_psi=lqr_q_psi,
        r_a=lqr_r_a,
        r_r=lqr_r_r,
    )
    if base_set_c > lqr_design.max_certified_level + 1e-12:
        raise ValueError(
            "base_set_c exceeds the discrete-time LQR actuator-admissible bound: "
            f"got {base_set_c}, max {lqr_design.max_certified_level}."
        )
    y_radius = lqr_design.coordinate_abs_max(level=base_set_c, index=0)
    psi_radius = lqr_design.coordinate_abs_max(level=base_set_c, index=2)
    if y_radius > float(y_max) + 1e-12:
        raise ValueError(
            f"LQR base set exceeds safe y bound: radius={y_radius}, y_max={y_max}"
        )
    if psi_radius > float(psi_max) + 1e-12:
        raise ValueError(
            f"LQR base set exceeds safe psi bound: radius={psi_radius}, psi_max={psi_max}"
        )

def _validate_unicycle_base_set_payload(cfg: UnicycleBCBFConfig, payload: dict[str, Any]) -> None:
    """Check the checkpoint's base_set_config level matches the current backup-CBF config."""
    expected = float(cfg.base_set_c)
    if "base_set_c" in payload and abs(float(payload["base_set_c"]) - expected) > 1e-6:
        raise ValueError(
            "Learned lane backup policy base_set_config does not match the current backup-CBF "
            f"config: got base_set_c={payload['base_set_c']}, expected {expected}."
        )


def _validate_unicycle_lqr_payload(cfg: UnicycleBCBFConfig, payload: dict[str, Any]) -> None:
    expected = _unicycle_lqr_config_from_cbf(cfg)
    for key, expected_value in expected.items():
        if key in payload and abs(float(payload[key]) - float(expected_value)) > 1e-6:
            raise ValueError(
                "Learned lane backup policy lqr_config does not match the current backup-CBF "
                f"config for {key}: got {payload[key]}, expected {expected_value}."
            )


def _unicycle_base_set_from_payload(
    cfg: UnicycleBCBFConfig,
    payload: dict[str, Any],
) -> EllipsoidBaseSet:
    base_set_c = float(payload.get("base_set_c", cfg.base_set_c))
    lqr_design = _unicycle_lqr_design_from_cbf(cfg)
    validate_unicycle_base_config(
        y_max=cfg.y_max,
        psi_max=cfg.psi_max,
        dt=cfg.dt,
        a_max=cfg.a_max,
        r_max=cfg.r_max,
        base_set_c=base_set_c,
        v_des=cfg.v_des,
        lqr_q_y=cfg.lqr_q_y,
        lqr_q_v=cfg.lqr_q_v,
        lqr_q_psi=cfg.lqr_q_psi,
        lqr_r_a=cfg.lqr_r_a,
        lqr_r_r=cfg.lqr_r_r,
    )
    return EllipsoidBaseSet(lqr_design, base_set_c)


def analytic_backup_policy(
    x: jax.Array,
    cfg: UnicycleBCBFConfig,
    *,
    lqr_design: UnicycleDLQR | None = None,
) -> jax.Array:
    """Analytic backup policy: the saturated discrete-time LQR base controller."""
    design = lqr_design if lqr_design is not None else _unicycle_lqr_design_from_cbf(cfg)
    u = design.action(jnp.asarray(x))
    return clip_backup_action(u, a_max=cfg.a_max, r_max=cfg.r_max)


def _sanitize_unicycle_solve_state(x: jax.Array) -> jax.Array:
    # Projection layer only depends on physical state [y, v, psi].
    x = x[:3]
    return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def make_backup_runtime(cfg: UnicycleBCBFConfig) -> BCBFSystem:
    safe_set = UnicycleSafeSet(y_max=cfg.y_max, psi_max=cfg.psi_max)
    lqr_design = _unicycle_lqr_design_from_cbf(cfg)
    capture_set = _unicycle_base_set_from_cbf(cfg, lqr_design=lqr_design)
    terminal_set = capture_set

    def system_with_backup_policy(backup_policy_fn) -> BCBFSystem:
        return BCBFSystem(
            state_dim=3,
            action_dim=2,
            action_low=(-float(cfg.a_max), -float(cfg.r_max)),
            action_high=(float(cfg.a_max), float(cfg.r_max)),
            backup_policy_fn=backup_policy_fn,
            safe_set_values_and_grads_fn=safe_set.values_and_grads,
            base_set_values_and_grads_fn=terminal_set.values_and_grads,
            dynamics_fn=unicycle_dynamics,
            control_affine_terms_fn=control_affine_terms,
            postprocess_rollout_state_fn=lambda z: postprocess_unicycle_state(
                z, v_min=cfg.v_min, v_max=cfg.v_max
            ),
            sanitize_solve_state_fn=_sanitize_unicycle_solve_state,
            solve_state_slice_fn=lambda xb: xb[..., :3],
            use_analytic_jacobian=False,
            analytic_closed_loop_and_jacobian_fn=None,
        )

    mode = cfg.backup_policy_mode.strip().lower()
    if mode == "analytic":
        abp = UnicycleABP(base_controller=lqr_design, base_set=capture_set)
        return system_with_backup_policy(abp.action)
    if mode == "learned":
        def _validate_learned_bound_match(
            name: str,
            actual: np.ndarray | jax.Array,
            expected: np.ndarray | jax.Array,
            *,
            atol: float = 1e-6,
        ) -> None:
            actual_np = np.asarray(jax.device_get(actual), dtype=np.float64).reshape(-1)
            expected_np = np.asarray(jax.device_get(expected), dtype=np.float64).reshape(-1)
            if actual_np.shape != expected_np.shape:
                raise ValueError(
                    f"Learned lane backup policy {name} has shape {actual_np.shape}, "
                    f"expected {expected_np.shape}."
                )
            if not np.all(np.isfinite(actual_np)):
                raise ValueError(
                    f"Learned lane backup policy {name} contains non-finite values: {actual_np.tolist()}."
                )
            if not np.allclose(actual_np, expected_np, atol=atol):
                raise ValueError(
                    f"Learned lane backup policy {name} does not match the current backup-CBF bounds: "
                    f"got {actual_np.tolist()}, expected {expected_np.tolist()}."
                )

        if not cfg.learned_backup_policy_path.strip():
            raise ValueError("learned_backup_policy_path must be provided when backup_policy_mode='learned'")
        learned = load_learned_unicycle_backup_policy(cfg.learned_backup_policy_path)
        clip_low = jnp.asarray([-cfg.a_max, -cfg.r_max], dtype=jnp.float32)
        clip_high = jnp.asarray([cfg.a_max, cfg.r_max], dtype=jnp.float32)
        expected_action_scale = jnp.asarray([cfg.a_max, cfg.r_max], dtype=jnp.float32)
        learned_action_scale = jnp.asarray(learned.action_scale, dtype=jnp.float32)
        _validate_learned_bound_match("action_scale", learned_action_scale, expected_action_scale)
        if learned.action_low is not None and learned.action_high is not None:
            learned_action_low = jnp.asarray(learned.action_low, dtype=jnp.float32)
            learned_action_high = jnp.asarray(learned.action_high, dtype=jnp.float32)
            _validate_learned_bound_match("action_low", learned_action_low, clip_low)
            _validate_learned_bound_match("action_high", learned_action_high, clip_high)

        metadata = dict(learned.metadata)
        base_set_handoff_enabled = bool(metadata.get("base_set_handoff_enabled", False))
        capture_set_from_payload = None
        if base_set_handoff_enabled:
            base_set_payload = metadata.get("base_set_config")
            if not isinstance(base_set_payload, dict):
                raise KeyError(
                    "Learned lane backup policy enables the base-set handoff but is missing "
                    "metadata['base_set_config']."
                )
            _validate_unicycle_base_set_payload(cfg, base_set_payload)

            tail_controller = str(metadata.get("tail_controller", "analytic_pid")).strip().lower()
            if tail_controller != "lqr":
                raise ValueError(
                    "Learned lane backup policy requests the removed "
                    f"'{tail_controller}' tail controller; only 'lqr' tail "
                    "checkpoints are supported after the unicycle PID removal."
                )
            lqr_cfg_payload = metadata.get("lqr_config")
            if isinstance(lqr_cfg_payload, dict):
                _validate_unicycle_lqr_payload(cfg, lqr_cfg_payload)
            else:
                raise KeyError(
                    "Learned lane backup policy requires metadata['lqr_config'] for the base-set handoff."
                )
            capture_set_from_payload = _unicycle_base_set_from_payload(cfg, base_set_payload)

        lbp = UnicycleLBP(
            base_controller=lqr_design,
            base_set=terminal_set,
            learned=learned,
            a_max=float(cfg.a_max),
            r_max=float(cfg.r_max),
            capture_set=capture_set_from_payload,
        )
        return system_with_backup_policy(lbp.action)
    raise ValueError(f"Unsupported backup_policy_mode: {cfg.backup_policy_mode}")


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


class UnicycleBackupCBFProjector(BackupCBFProjector):
    """Convenience wrapper with jitted single/batch projection calls."""

    def __init__(self, cfg: UnicycleBCBFConfig, runtime: BCBFSystem | None = None):
        super().__init__(cfg, _resolve_runtime(cfg, runtime))
