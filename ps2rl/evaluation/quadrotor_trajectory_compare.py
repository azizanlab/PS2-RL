"""Trajectory-conditioned ABP-vs-LBP benchmark near the quadrotor powerloop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import json
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    _HAS_MATPLOTLIB = False

from ps2rl.backup_policy.quadrotor_learned_backup import LearnedQuadrotorBackupPolicy
from ps2rl.base_controller.quadrotor_dlqr import QuadrotorDLQR
from ps2rl.sets.base_sets import EllipsoidBaseSet
from ps2rl.backup_policy.backup_policy import BackupPolicy
from ps2rl.backup_policy.quadrotor_analytic_backup import _aggressive_pid_policy_raw
from ps2rl.evaluation.quadrotor_trace_reset_lib import QuadrotorResetLibrary
from ps2rl.phase1_sa.quadrotor_sa_trainer import (
    QuadrotorRecoverabilityWeights,
)
from ps2rl.cil.quadrotor_backup_cbf import (
    QuadrotorBCBFConfig,
    QuadrotorBackupCBFProjector,
    BCBFSystem,
    build_discretized_backup_cbf_rows,
    hard_deck_value,
    is_safe_state,
    make_backup_runtime,
    quadrotor_step_euler,
)


_REGION_KEYS = ("general_trace", "near_ceiling", "bridge", "base_shell")
_POLICY_ORDER = ("analytic_backup_policy", "learned_backup_policy")
_POLICY_LABELS = {
    "analytic_backup_policy": "ABP",
    "learned_backup_policy": "LBP",
}
_POLICY_COLORS = {
    "analytic_backup_policy": "#8C3B00",
    "learned_backup_policy": "#003B67",
}


def _sanitize_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_jsonable(payload), f, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_sanitize_jsonable(row))


def _string_dtype(max_len: int) -> np.dtype[np.str_]:
    return np.dtype(f"<U{max(1, int(max_len))}")


def _string_full(length: int, value: str) -> np.ndarray:
    return np.full((int(length),), str(value), dtype=_string_dtype(len(str(value))))


def _summary_stats(x: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _rate(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


@dataclass(frozen=True)
class QuadrotorTrajectoryCompareConfig:
    perturbed_split: str = "test"
    max_exact_points_per_region: int = 0
    num_perturbed_general: int = 512
    num_perturbed_near_ceiling: int = 512
    num_perturbed_bridge: int = 512
    num_perturbed_base_shell: int = 256
    batch_size: int = 512

    enable_qp_screening: bool = False
    qp_batch_size: int = 1024
    qp_max_points: int = 0
    qp_postsolve_feas_tol: float | None = None

    benchmark_seed: int = 0


@dataclass(frozen=True)
class _PolicyRolloutSpec:
    name: str
    runtime: BCBFSystem
    raw_action_fn: Callable[[jax.Array], jax.Array]
    used_lqr_fn: Callable[[jax.Array], jax.Array]


def quadrotor_trajectory_compare_config_from_dict(payload: dict[str, Any]) -> QuadrotorTrajectoryCompareConfig:
    return QuadrotorTrajectoryCompareConfig(
        perturbed_split=str(payload.get("perturbed_split", "test")),
        max_exact_points_per_region=int(payload.get("max_exact_points_per_region", 0)),
        num_perturbed_general=int(payload.get("num_perturbed_general", 512)),
        num_perturbed_near_ceiling=int(payload.get("num_perturbed_near_ceiling", 512)),
        num_perturbed_bridge=int(payload.get("num_perturbed_bridge", 512)),
        num_perturbed_base_shell=int(payload.get("num_perturbed_base_shell", payload.get("num_perturbed_capture_shell", 256))),
        batch_size=int(payload.get("batch_size", 512)),
        enable_qp_screening=bool(payload.get("enable_qp_screening", False)),
        qp_batch_size=int(payload.get("qp_batch_size", 1024)),
        qp_max_points=int(payload.get("qp_max_points", 0)),
        qp_postsolve_feas_tol=payload.get("qp_postsolve_feas_tol"),
        benchmark_seed=int(payload.get("benchmark_seed", 0)),
    )


def _sample_exact_points(states: np.ndarray, *, max_points: int, seed: int) -> np.ndarray:
    x = np.asarray(states, dtype=np.float64)
    if max_points <= 0 or x.shape[0] <= max_points:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=int(max_points), replace=False)
    return x[np.asarray(idx, dtype=np.int64)]


def _build_benchmark_dataset(
    reset_library: QuadrotorResetLibrary,
    cfg: QuadrotorTrajectoryCompareConfig,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(cfg.benchmark_seed))
    states: list[np.ndarray] = []
    subset_name: list[np.ndarray] = []
    region_name: list[np.ndarray] = []

    exact_seed = int(rng.integers(0, 2**31 - 1))
    for region_idx, region in enumerate(_REGION_KEYS):
        sampled = _sample_exact_points(
            reset_library.all_pools.get(region, np.zeros((0, 10), dtype=np.float64)),
            max_points=int(cfg.max_exact_points_per_region),
            seed=exact_seed + region_idx,
        )
        if sampled.size == 0:
            continue
        states.append(sampled)
        subset_name.append(_string_full(sampled.shape[0], f"exact_{region}"))
        region_name.append(_string_full(sampled.shape[0], region))

    perturbed_specs = [
        ("general_trace", int(cfg.num_perturbed_general)),
        ("near_ceiling", int(cfg.num_perturbed_near_ceiling)),
        ("bridge", int(cfg.num_perturbed_bridge)),
        ("base_shell", int(cfg.num_perturbed_base_shell)),
    ]
    for region_idx, (region, count) in enumerate(perturbed_specs):
        sampled = reset_library.sample_perturbed_region_states(
            region,
            count=count,
            seed=int(cfg.benchmark_seed + 10_000 * (region_idx + 1)),
            curriculum_scale=1.0,
            split=cfg.perturbed_split,
        )
        if sampled.size == 0:
            continue
        states.append(sampled)
        subset_name.append(_string_full(sampled.shape[0], f"perturbed_{region}"))
        region_name.append(_string_full(sampled.shape[0], region))

    if not states:
        raise ValueError("Benchmark dataset is empty. Reset-library pools did not produce any samples.")

    return {
        "states": np.concatenate(states, axis=0),
        "subset_name": np.concatenate(subset_name, axis=0),
        "region_name": np.concatenate(region_name, axis=0),
    }


def _build_rollout_batch_fn(
    cbf_cfg: QuadrotorBCBFConfig,
    runtime: BCBFSystem,
    *,
    base_set: EllipsoidBaseSet,
) -> Callable[[jax.Array], tuple[jax.Array, ...]]:
    terminal_values_fn = runtime.base_set_values_fn

    def terminal_margin(x: jax.Array) -> jax.Array:
        return jnp.min(terminal_values_fn(x))

    def rollout_single(x0: jax.Array):
        x0 = jnp.asarray(x0)
        safe0 = is_safe_state(x0, cbf_cfg)
        entered_capture0 = base_set.contains(x0)
        capture_entry0 = jnp.where(entered_capture0, jnp.int32(0), jnp.int32(-1))
        entered0 = terminal_margin(x0) >= 0.0
        terminal_after_capture0 = entered_capture0 & entered0
        min_hard0 = hard_deck_value(x0, cbf_cfg)
        entry0 = jnp.where(entered0, jnp.int32(0), jnp.int32(-1))

        def step(carry, k):
            (
                x,
                safe_rollout,
                entered_capture,
                capture_entry_step,
                left_capture_after_entry,
                entered_terminal,
                terminal_after_capture,
                terminal_entry_step,
                left_terminal_after_entry,
                min_hard_rollout,
            ) = carry
            u = runtime.backup_policy_fn(x)
            x_next = quadrotor_step_euler(x, u, cbf_cfg)
            safe_next = is_safe_state(x_next, cbf_cfg)
            in_capture_next = base_set.contains(x_next)
            term_next = terminal_margin(x_next)
            in_term_next = term_next >= 0.0
            entered_capture_next = entered_capture | in_capture_next
            capture_entry_step_next = jnp.where(
                (~entered_capture) & in_capture_next,
                k + jnp.int32(1),
                capture_entry_step,
            )
            left_capture_after_entry_next = left_capture_after_entry | (entered_capture & (~in_capture_next))
            entered_terminal_next = entered_terminal | in_term_next
            terminal_after_capture_next = terminal_after_capture | (entered_capture_next & in_term_next)
            terminal_entry_step_next = jnp.where(
                (~entered_terminal) & in_term_next,
                k + jnp.int32(1),
                terminal_entry_step,
            )
            left_terminal_after_entry_next = left_terminal_after_entry | (entered_terminal & (~in_term_next))
            min_hard_next = jnp.minimum(
                min_hard_rollout,
                hard_deck_value(x_next, cbf_cfg),
            )
            return (
                x_next,
                safe_rollout & safe_next,
                entered_capture_next,
                capture_entry_step_next,
                left_capture_after_entry_next,
                entered_terminal_next,
                terminal_after_capture_next,
                terminal_entry_step_next,
                left_terminal_after_entry_next,
                min_hard_next,
            ), ()

        final_carry, _ = jax.lax.scan(
            step,
            (
                x0,
                safe0,
                entered_capture0,
                capture_entry0,
                jnp.asarray(False),
                entered0,
                terminal_after_capture0,
                entry0,
                jnp.asarray(False),
                min_hard0,
            ),
            jnp.arange(cbf_cfg.num_steps, dtype=jnp.int32),
        )
        (
            x_f,
            safe_rollout_f,
            entered_capture_f,
            capture_entry_f,
            left_capture_after_entry_f,
            entered_f,
            terminal_after_capture_f,
            entry_f,
            left_after_entry_f,
            min_hard_f,
        ) = final_carry
        capture_invariance_after_entry_f = entered_capture_f & (~left_capture_after_entry_f)
        terminal_at_end_f = terminal_margin(x_f) >= 0.0
        invariance_after_entry_f = entered_f & (~left_after_entry_f)
        success_f = safe_rollout_f & entered_f & invariance_after_entry_f
        return (
            success_f,
            safe_rollout_f,
            entered_capture_f,
            capture_invariance_after_entry_f,
            terminal_after_capture_f,
            capture_entry_f,
            entered_f,
            invariance_after_entry_f,
            terminal_at_end_f,
            entry_f,
            min_hard_f,
        )

    return jax.jit(jax.vmap(rollout_single, in_axes=0))


def _build_rollout_trace_batch_fn(
    cbf_cfg: QuadrotorBCBFConfig,
    *,
    policy_spec: _PolicyRolloutSpec,
    base_set: EllipsoidBaseSet,
) -> Callable[[jax.Array], dict[str, jax.Array]]:
    terminal_values_fn = policy_spec.runtime.base_set_values_fn

    def terminal_margin(x: jax.Array) -> jax.Array:
        return jnp.min(terminal_values_fn(x))

    def rollout_trace_single(x0: jax.Array) -> dict[str, jax.Array]:
        def step(x: jax.Array, _k: jax.Array):
            x = jnp.asarray(x)
            raw_action = policy_spec.raw_action_fn(x)
            used_lqr = policy_spec.used_lqr_fn(x)
            act = policy_spec.runtime.backup_policy_fn(x)
            x_next = quadrotor_step_euler(x, act, cbf_cfg)
            record = {
                "obs": x,
                "next_obs": x_next,
                "act": act,
                "raw_action": raw_action,
                "safe": is_safe_state(x_next, cbf_cfg),
                "capture": base_set.contains(x_next),
                "terminal": terminal_margin(x_next) >= 0.0,
                "used_lqr": used_lqr,
            }
            return x_next, record

        _, trace = jax.lax.scan(
            step,
            jnp.asarray(x0, dtype=jnp.float32),
            jnp.arange(cbf_cfg.num_steps, dtype=jnp.int32),
        )
        return trace

    return jax.jit(jax.vmap(rollout_trace_single, in_axes=0))


def _build_primal_residual_batch_fn(cbf_cfg: QuadrotorBCBFConfig, runtime: BCBFSystem):
    def residual_single(x: jax.Array, u: jax.Array) -> jax.Array:
        a_rows, b_rows = build_discretized_backup_cbf_rows(x, cbf_cfg, runtime=runtime)
        cbf_residual = a_rows @ u - b_rows
        box_residual = jnp.array(
            [
                u[0] - cbf_cfg.a_cmd_max,
                cbf_cfg.a_cmd_min - u[0],
                u[1] - cbf_cfg.omega_max,
                -u[1] - cbf_cfg.omega_max,
                u[2] - cbf_cfg.omega_max,
                -u[2] - cbf_cfg.omega_max,
                u[3] - cbf_cfg.omega_max,
                -u[3] - cbf_cfg.omega_max,
            ],
            dtype=x.dtype,
        )
        return jnp.max(jnp.concatenate([cbf_residual, box_residual], axis=0))

    return jax.jit(jax.vmap(residual_single, in_axes=(0, 0)))


def _aggregate_rollout_metrics(
    success: np.ndarray,
    safe_rollout: np.ndarray,
    entered_capture: np.ndarray,
    capture_invariance_after_entry: np.ndarray,
    terminal_after_capture: np.ndarray,
    capture_entry_step: np.ndarray,
    capture_entry_time_sec: np.ndarray,
    entered: np.ndarray,
    invariance_after_entry: np.ndarray,
    terminal_at_end: np.ndarray,
    entry_time_sec: np.ndarray,
    min_hard_deck_margin: np.ndarray,
) -> dict[str, Any]:
    entered_capture_mask = np.asarray(entered_capture, dtype=bool)
    entered_mask = np.asarray(entered, dtype=bool)
    success_mask = np.asarray(success, dtype=bool)
    return {
        "count": int(success.shape[0]),
        "recoverability_rate": _rate(success),
        "safe_rollout_rate": _rate(safe_rollout),
        "entered_capture_rate": _rate(entered_capture),
        "post_capture_terminal_entry_rate": _rate(terminal_after_capture),
        "post_capture_terminal_entry_rate_given_capture_entry": _rate(
            np.asarray(terminal_after_capture, dtype=np.float64)[entered_capture_mask]
        ),
        "capture_invariance_after_entry_rate": _rate(capture_invariance_after_entry),
        "capture_invariance_after_entry_rate_given_entry": _rate(
            np.asarray(capture_invariance_after_entry, dtype=np.float64)[entered_capture_mask]
        ),
        "capture_entry_step_entered": _summary_stats(np.asarray(capture_entry_step, dtype=np.float64)[entered_capture_mask]),
        "capture_entry_time_sec_entered": _summary_stats(np.asarray(capture_entry_time_sec, dtype=np.float64)[entered_capture_mask]),
        "entered_terminal_rate": _rate(entered),
        "invariance_after_entry_rate": _rate(invariance_after_entry),
        "invariance_after_entry_rate_given_entry": _rate(np.asarray(invariance_after_entry, dtype=np.float64)[entered_mask]),
        "terminal_invariance_after_entry_rate": _rate(invariance_after_entry),
        "terminal_invariance_after_entry_rate_given_entry": _rate(
            np.asarray(invariance_after_entry, dtype=np.float64)[entered_mask]
        ),
        "terminal_at_horizon_rate": _rate(terminal_at_end),
        "terminal_entry_time_sec_success": _summary_stats(np.asarray(entry_time_sec, dtype=np.float64)[success_mask]),
        "terminal_entry_time_sec_entered": _summary_stats(np.asarray(entry_time_sec, dtype=np.float64)[entered_mask]),
        "minimum_hard_deck_margin": _summary_stats(min_hard_deck_margin),
    }


def _weighted_recoverability(
    by_region: dict[str, dict[str, Any]],
    weights: QuadrotorRecoverabilityWeights,
) -> float:
    weight_map = {
        "general_trace": float(weights.general_trace),
        "near_ceiling": float(weights.near_ceiling),
        "bridge": float(weights.bridge),
        "base_shell": float(weights.base_shell),
    }
    numer = 0.0
    denom = 0.0
    for region, weight in weight_map.items():
        metrics = by_region.get(region)
        if metrics is None or int(metrics.get("count", 0)) <= 0:
            continue
        numer += weight * float(metrics.get("recoverability_rate", 0.0))
        denom += weight
    return float(numer / denom) if denom > 0.0 else 0.0


def _policy_runtime_table(
    *,
    cbf_cfg: QuadrotorBCBFConfig,
    learned_policy: LearnedQuadrotorBackupPolicy,
) -> dict[str, _PolicyRolloutSpec]:
    analytic_runtime = make_backup_runtime(cbf_cfg)
    base_set = EllipsoidBaseSet(
        QuadrotorDLQR.from_config(cbf_cfg),
        float(cbf_cfg.base_set_c),
        smooth_gain=float(cbf_cfg.base_set_smooth_gain),
    )

    def learned_raw_action_fn(x: jax.Array) -> jax.Array:
        x_arr = jnp.asarray(x, dtype=jnp.float32)
        raw = learned_policy.action_single(x_arr)
        return jnp.asarray(raw, dtype=jnp.asarray(x).dtype)

    analytic_used_lqr_fn = lambda x: base_set.contains(jnp.asarray(x))

    def learned_backup_policy_fn(x: jax.Array) -> jax.Array:
        raw = learned_raw_action_fn(x)
        return BackupPolicy.select_action(x, raw, base_set)

    learned_runtime = replace(analytic_runtime, backup_policy_fn=learned_backup_policy_fn)
    return {
        "analytic_backup_policy": _PolicyRolloutSpec(
            name="analytic_backup_policy",
            runtime=analytic_runtime,
            raw_action_fn=lambda x: _aggressive_pid_policy_raw(jnp.asarray(x), cbf_cfg),
            used_lqr_fn=analytic_used_lqr_fn,
        ),
        "learned_backup_policy": _PolicyRolloutSpec(
            name="learned_backup_policy",
            runtime=learned_runtime,
            raw_action_fn=learned_raw_action_fn,
            used_lqr_fn=lambda x: base_set.contains(jnp.asarray(x)),
        ),
    }


def _resolve_policy_names(
    available_policy_names: list[str],
    selected_policies: list[str] | tuple[str, ...] | None,
) -> list[str]:
    if selected_policies is None:
        return [name for name in _POLICY_ORDER if name in available_policy_names]

    requested = []
    unknown = []
    for raw in selected_policies:
        name = str(raw).strip()
        if name in available_policy_names:
            if name not in requested:
                requested.append(name)
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"Unknown policy names requested: {unknown}. "
            f"Available policies: {available_policy_names}"
        )
    if not requested:
        raise ValueError("No policies were selected for comparison.")
    return requested


def _build_compare_array_payload(
    *,
    states: np.ndarray,
    subset_name: np.ndarray,
    region_name: np.ndarray,
    policy_names: list[str],
    rollout_arrays: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "states": np.asarray(states, dtype=np.float64),
        "subset_name": np.asarray(subset_name).astype(str),
        "region_name": np.asarray(region_name).astype(str),
        "policy_names": np.asarray(policy_names, dtype=_string_dtype(max(len(name) for name in policy_names))),
        "recoverable": np.stack([np.asarray(rollout_arrays[name]["recoverable"], dtype=bool) for name in policy_names], axis=0),
        "safe_rollout": np.stack([np.asarray(rollout_arrays[name]["safe_rollout"], dtype=bool) for name in policy_names], axis=0),
        "entered_capture": np.stack([np.asarray(rollout_arrays[name]["entered_capture"], dtype=bool) for name in policy_names], axis=0),
        "capture_invariance_after_entry": np.stack([np.asarray(rollout_arrays[name]["capture_invariance_after_entry"], dtype=bool) for name in policy_names], axis=0),
        "terminal_after_capture_entry": np.stack([np.asarray(rollout_arrays[name]["terminal_after_capture_entry"], dtype=bool) for name in policy_names], axis=0),
        "capture_entry_time_sec": np.stack([np.asarray(rollout_arrays[name]["capture_entry_time_sec"], dtype=np.float64) for name in policy_names], axis=0),
        "capture_entry_step": np.stack([np.asarray(rollout_arrays[name]["capture_entry_step"], dtype=np.int32) for name in policy_names], axis=0),
        "entered_terminal": np.stack([np.asarray(rollout_arrays[name]["entered_terminal"], dtype=bool) for name in policy_names], axis=0),
        "invariance_after_entry": np.stack([np.asarray(rollout_arrays[name]["invariance_after_entry"], dtype=bool) for name in policy_names], axis=0),
        "terminal_at_end": np.stack([np.asarray(rollout_arrays[name]["terminal_at_end"], dtype=bool) for name in policy_names], axis=0),
        "entry_time_sec": np.stack([np.asarray(rollout_arrays[name]["entry_time_sec"], dtype=np.float64) for name in policy_names], axis=0),
        "entry_step": np.stack([np.asarray(rollout_arrays[name]["entry_step"], dtype=np.int32) for name in policy_names], axis=0),
        "min_hard_deck_margin": np.stack([np.asarray(rollout_arrays[name]["min_hard_deck_margin"], dtype=np.float64) for name in policy_names], axis=0),
    }

    legacy_name_map = {
        "analytic_backup_policy": "analytic",
        "learned_backup_policy": "learned",
    }
    for policy_name, prefix in legacy_name_map.items():
        if policy_name not in rollout_arrays:
            continue
        payload[f"{prefix}_recoverable"] = np.asarray(rollout_arrays[policy_name]["recoverable"], dtype=bool)
        payload[f"{prefix}_terminal_at_end"] = np.asarray(rollout_arrays[policy_name]["terminal_at_end"], dtype=bool)
        payload[f"{prefix}_min_hard_deck_margin"] = np.asarray(rollout_arrays[policy_name]["min_hard_deck_margin"], dtype=np.float64)

    return payload


def _write_rollout_trace_bundle(
    *,
    out_dir: Path,
    states: np.ndarray,
    subset_name: np.ndarray,
    region_name: np.ndarray,
    policy_names: list[str],
    rollout_arrays: dict[str, dict[str, np.ndarray]],
    trace_arrays: dict[str, dict[str, np.ndarray]],
    cbf_cfg: QuadrotorBCBFConfig,
) -> None:
    np.savez(
        out_dir / "trajectory_compare_rollouts.npz",
        policy_names=np.asarray(policy_names, dtype=_string_dtype(max(len(name) for name in policy_names))),
        benchmark_idx=np.arange(states.shape[0], dtype=np.int32),
        initial_state=np.asarray(states, dtype=np.float32),
        subset_name=np.asarray(subset_name).astype(str),
        region_name=np.asarray(region_name).astype(str),
        obs=np.stack([np.asarray(trace_arrays[name]["obs"], dtype=np.float32) for name in policy_names], axis=0),
        next_obs=np.stack([np.asarray(trace_arrays[name]["next_obs"], dtype=np.float32) for name in policy_names], axis=0),
        act=np.stack([np.asarray(trace_arrays[name]["act"], dtype=np.float32) for name in policy_names], axis=0),
        raw_action=np.stack([np.asarray(trace_arrays[name]["raw_action"], dtype=np.float32) for name in policy_names], axis=0),
        safe=np.stack([np.asarray(trace_arrays[name]["safe"], dtype=bool) for name in policy_names], axis=0),
        capture=np.stack([np.asarray(trace_arrays[name]["capture"], dtype=bool) for name in policy_names], axis=0),
        terminal=np.stack([np.asarray(trace_arrays[name]["terminal"], dtype=bool) for name in policy_names], axis=0),
        used_lqr=np.stack([np.asarray(trace_arrays[name]["used_lqr"], dtype=bool) for name in policy_names], axis=0),
        recoverable=np.stack([np.asarray(rollout_arrays[name]["recoverable"], dtype=bool) for name in policy_names], axis=0),
        safe_rollout=np.stack([np.asarray(rollout_arrays[name]["safe_rollout"], dtype=bool) for name in policy_names], axis=0),
        entered_capture=np.stack([np.asarray(rollout_arrays[name]["entered_capture"], dtype=bool) for name in policy_names], axis=0),
        capture_invariance_after_entry=np.stack([np.asarray(rollout_arrays[name]["capture_invariance_after_entry"], dtype=bool) for name in policy_names], axis=0),
        terminal_after_capture_entry=np.stack([np.asarray(rollout_arrays[name]["terminal_after_capture_entry"], dtype=bool) for name in policy_names], axis=0),
        capture_entry_time_sec=np.stack([np.asarray(rollout_arrays[name]["capture_entry_time_sec"], dtype=np.float32) for name in policy_names], axis=0),
        capture_entry_step=np.stack([np.asarray(rollout_arrays[name]["capture_entry_step"], dtype=np.int32) for name in policy_names], axis=0),
        entered_terminal=np.stack([np.asarray(rollout_arrays[name]["entered_terminal"], dtype=bool) for name in policy_names], axis=0),
        invariance_after_entry=np.stack([np.asarray(rollout_arrays[name]["invariance_after_entry"], dtype=bool) for name in policy_names], axis=0),
        terminal_at_end=np.stack([np.asarray(rollout_arrays[name]["terminal_at_end"], dtype=bool) for name in policy_names], axis=0),
        entry_time_sec=np.stack([np.asarray(rollout_arrays[name]["entry_time_sec"], dtype=np.float32) for name in policy_names], axis=0),
        entry_step=np.stack([np.asarray(rollout_arrays[name]["entry_step"], dtype=np.int32) for name in policy_names], axis=0),
        min_hard_deck_margin=np.stack([np.asarray(rollout_arrays[name]["min_hard_deck_margin"], dtype=np.float32) for name in policy_names], axis=0),
        dt=np.asarray(float(cbf_cfg.dt), dtype=np.float32),
        num_steps=np.asarray(int(cbf_cfg.num_steps), dtype=np.int32),
    )


def compare_quadrotor_backup_policies(
    *,
    cbf_cfg: QuadrotorBCBFConfig,
    reset_library: QuadrotorResetLibrary,
    learned_policy: LearnedQuadrotorBackupPolicy,
    compare_cfg: QuadrotorTrajectoryCompareConfig,
    recoverability_weights: QuadrotorRecoverabilityWeights,
    output_dir: str | Path,
    selected_policies: list[str] | tuple[str, ...] | None = None,
    save_rollout_trajectories: bool = False,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark = _build_benchmark_dataset(reset_library, compare_cfg)
    policy_specs = _policy_runtime_table(
        cbf_cfg=cbf_cfg,
        learned_policy=learned_policy,
    )
    policy_names = _resolve_policy_names(list(policy_specs.keys()), selected_policies)
    base_set = EllipsoidBaseSet(
        QuadrotorDLQR.from_config(cbf_cfg),
        float(cbf_cfg.base_set_c),
        smooth_gain=float(cbf_cfg.base_set_smooth_gain),
    )

    states = np.asarray(benchmark["states"], dtype=np.float64)
    subset_name = np.asarray(benchmark["subset_name"]).astype(str)
    region_name = np.asarray(benchmark["region_name"]).astype(str)

    rollout_arrays: dict[str, dict[str, np.ndarray]] = {}
    trace_arrays: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, Any] = {}

    for policy_name in policy_names:
        policy_spec = policy_specs[policy_name]
        runtime = policy_spec.runtime
        rollout_batch_fn = _build_rollout_batch_fn(cbf_cfg, runtime, base_set=base_set)
        rollout_trace_batch_fn = (
            _build_rollout_trace_batch_fn(
                cbf_cfg,
                policy_spec=policy_spec,
                base_set=base_set,
            )
            if save_rollout_trajectories
            else None
        )

        success_all: list[np.ndarray] = []
        safe_rollout_all: list[np.ndarray] = []
        entered_capture_all: list[np.ndarray] = []
        capture_invariance_all: list[np.ndarray] = []
        terminal_after_capture_all: list[np.ndarray] = []
        capture_entry_time_all: list[np.ndarray] = []
        capture_entry_step_all: list[np.ndarray] = []
        entered_all: list[np.ndarray] = []
        invariance_all: list[np.ndarray] = []
        terminal_at_end_all: list[np.ndarray] = []
        entry_time_all: list[np.ndarray] = []
        entry_step_all: list[np.ndarray] = []
        min_hard_all: list[np.ndarray] = []
        trace_all: dict[str, list[np.ndarray]] = {
            "obs": [],
            "next_obs": [],
            "act": [],
            "raw_action": [],
            "safe": [],
            "capture": [],
            "terminal": [],
            "used_lqr": [],
        }

        for start in range(0, states.shape[0], max(1, int(compare_cfg.batch_size))):
            end = min(start + int(compare_cfg.batch_size), states.shape[0])
            batch = jnp.asarray(states[start:end], dtype=jnp.float32)
            (
                success_j,
                safe_j,
                entered_capture_j,
                capture_invariance_j,
                terminal_after_capture_j,
                capture_entry_step_j,
                entered_j,
                invariance_j,
                term_end_j,
                entry_step_j,
                min_hard_j,
            ) = rollout_batch_fn(batch)
            success_all.append(np.asarray(success_j, dtype=bool))
            safe_rollout_all.append(np.asarray(safe_j, dtype=bool))
            entered_capture_all.append(np.asarray(entered_capture_j, dtype=bool))
            capture_invariance_all.append(np.asarray(capture_invariance_j, dtype=bool))
            terminal_after_capture_all.append(np.asarray(terminal_after_capture_j, dtype=bool))
            capture_entry_step_np = np.asarray(capture_entry_step_j, dtype=np.int32)
            capture_entry_step_all.append(capture_entry_step_np)
            capture_entry_time_all.append(
                np.where(
                    capture_entry_step_np >= 0,
                    capture_entry_step_np.astype(np.float64) * float(cbf_cfg.dt),
                    np.nan,
                )
            )
            entered_all.append(np.asarray(entered_j, dtype=bool))
            invariance_all.append(np.asarray(invariance_j, dtype=bool))
            terminal_at_end_all.append(np.asarray(term_end_j, dtype=bool))
            entry_step_np = np.asarray(entry_step_j, dtype=np.int32)
            entry_step_all.append(entry_step_np)
            entry_time_all.append(
                np.where(entry_step_np >= 0, entry_step_np.astype(np.float64) * float(cbf_cfg.dt), np.nan)
            )
            min_hard_all.append(np.asarray(min_hard_j, dtype=np.float64))
            if rollout_trace_batch_fn is not None:
                trace_batch = jax.device_get(rollout_trace_batch_fn(batch))
                for key in trace_all:
                    trace_all[key].append(np.asarray(trace_batch[key]))

        arr = {
            "recoverable": np.concatenate(success_all, axis=0),
            "safe_rollout": np.concatenate(safe_rollout_all, axis=0),
            "entered_capture": np.concatenate(entered_capture_all, axis=0),
            "capture_invariance_after_entry": np.concatenate(capture_invariance_all, axis=0),
            "terminal_after_capture_entry": np.concatenate(terminal_after_capture_all, axis=0),
            "capture_entry_time_sec": np.concatenate(capture_entry_time_all, axis=0),
            "capture_entry_step": np.concatenate(capture_entry_step_all, axis=0),
            "entered_terminal": np.concatenate(entered_all, axis=0),
            "invariance_after_entry": np.concatenate(invariance_all, axis=0),
            "terminal_at_end": np.concatenate(terminal_at_end_all, axis=0),
            "entry_time_sec": np.concatenate(entry_time_all, axis=0),
            "entry_step": np.concatenate(entry_step_all, axis=0),
            "min_hard_deck_margin": np.concatenate(min_hard_all, axis=0),
        }
        rollout_arrays[policy_name] = arr
        if rollout_trace_batch_fn is not None:
            trace_arrays[policy_name] = {key: np.concatenate(parts, axis=0) for key, parts in trace_all.items()}

        by_subset: dict[str, Any] = {}
        for subset in np.unique(subset_name).tolist():
            mask = subset_name == subset
            by_subset[str(subset)] = _aggregate_rollout_metrics(
                arr["recoverable"][mask].astype(np.float64),
                arr["safe_rollout"][mask].astype(np.float64),
                arr["entered_capture"][mask].astype(np.float64),
                arr["capture_invariance_after_entry"][mask].astype(np.float64),
                arr["terminal_after_capture_entry"][mask].astype(np.float64),
                arr["capture_entry_step"][mask],
                arr["capture_entry_time_sec"][mask],
                arr["entered_terminal"][mask].astype(np.float64),
                arr["invariance_after_entry"][mask].astype(np.float64),
                arr["terminal_at_end"][mask].astype(np.float64),
                arr["entry_time_sec"][mask],
                arr["min_hard_deck_margin"][mask],
            )

        by_region: dict[str, Any] = {}
        for region in _REGION_KEYS:
            mask = region_name == region
            by_region[region] = _aggregate_rollout_metrics(
                arr["recoverable"][mask].astype(np.float64),
                arr["safe_rollout"][mask].astype(np.float64),
                arr["entered_capture"][mask].astype(np.float64),
                arr["capture_invariance_after_entry"][mask].astype(np.float64),
                arr["terminal_after_capture_entry"][mask].astype(np.float64),
                arr["capture_entry_step"][mask],
                arr["capture_entry_time_sec"][mask],
                arr["entered_terminal"][mask].astype(np.float64),
                arr["invariance_after_entry"][mask].astype(np.float64),
                arr["terminal_at_end"][mask].astype(np.float64),
                arr["entry_time_sec"][mask],
                arr["min_hard_deck_margin"][mask],
            )

        overall = _aggregate_rollout_metrics(
            arr["recoverable"].astype(np.float64),
            arr["safe_rollout"].astype(np.float64),
            arr["entered_capture"].astype(np.float64),
            arr["capture_invariance_after_entry"].astype(np.float64),
            arr["terminal_after_capture_entry"].astype(np.float64),
            arr["capture_entry_step"],
            arr["capture_entry_time_sec"],
            arr["entered_terminal"].astype(np.float64),
            arr["invariance_after_entry"].astype(np.float64),
            arr["terminal_at_end"].astype(np.float64),
            arr["entry_time_sec"],
            arr["min_hard_deck_margin"],
        )
        overall["weighted_recoverability_score"] = _weighted_recoverability(by_region, recoverability_weights)
        summaries[policy_name] = {
            "overall": overall,
            "by_region": by_region,
            "by_subset": by_subset,
        }

    qp_screening = None
    if compare_cfg.enable_qp_screening:
        qp_screening = _run_qp_compatibility_screen(
            cbf_cfg=cbf_cfg,
            runtimes={name: policy_specs[name].runtime for name in policy_names},
            nominal_dataset=reset_library.nominal_dataset,
            compare_cfg=compare_cfg,
        )

    report = {
        "benchmark_config": asdict(compare_cfg),
        "evaluated_policy_names": policy_names,
        "saved_rollout_trajectories": bool(save_rollout_trajectories),
        "recoverability_weights": asdict(recoverability_weights),
        "cbf_config": asdict(cbf_cfg),
        "benchmark_dataset": {
            "num_states": int(states.shape[0]),
            "subset_counts": {str(subset): int(np.sum(subset_name == subset)) for subset in np.unique(subset_name).tolist()},
            "region_counts": {str(region): int(np.sum(region_name == region)) for region in np.unique(region_name).tolist()},
        },
        "policies": summaries,
    }
    if qp_screening is not None:
        report["qp_compatibility"] = qp_screening

    _write_json(out_dir / "trajectory_compare_summary.json", report)
    _write_csv(
        out_dir / "benchmark_dataset_index.csv",
        [
            {
                "idx": int(i),
                "subset_name": str(subset_name[i]),
                "region_name": str(region_name[i]),
                "px": float(states[i, 0]),
                "py": float(states[i, 1]),
                "pz": float(states[i, 2]),
            }
            for i in range(states.shape[0])
        ],
    )
    np.savez(
        out_dir / "trajectory_compare_arrays.npz",
        **_build_compare_array_payload(
            states=states,
            subset_name=subset_name,
            region_name=region_name,
            policy_names=policy_names,
            rollout_arrays=rollout_arrays,
        ),
    )
    if save_rollout_trajectories:
        _write_rollout_trace_bundle(
            out_dir=out_dir,
            states=states,
            subset_name=subset_name,
            region_name=region_name,
            policy_names=policy_names,
            rollout_arrays=rollout_arrays,
            trace_arrays=trace_arrays,
            cbf_cfg=cbf_cfg,
        )

    if _HAS_MATPLOTLIB:
        _plot_compare_summary(report, out_dir)

    return report


def _run_qp_compatibility_screen(
    *,
    cbf_cfg: QuadrotorBCBFConfig,
    runtimes: dict[str, BCBFSystem],
    nominal_dataset: dict[str, np.ndarray],
    compare_cfg: QuadrotorTrajectoryCompareConfig,
) -> dict[str, Any]:
    states = np.asarray(nominal_dataset.get("states", np.zeros((0, 10))), dtype=np.float64)
    nom_act = np.asarray(nominal_dataset.get("nom_act", np.zeros((0, 4))), dtype=np.float64)
    region = np.asarray(nominal_dataset.get("region", np.zeros((0,), dtype=_string_dtype(1)))).astype(str)
    if states.shape[0] == 0:
        return {"note": "Nominal dataset is empty; QP screening was skipped."}
    if compare_cfg.qp_max_points > 0 and states.shape[0] > compare_cfg.qp_max_points:
        rng = np.random.default_rng(int(compare_cfg.benchmark_seed) + 314_159)
        idx = rng.choice(states.shape[0], size=int(compare_cfg.qp_max_points), replace=False)
        states = states[np.asarray(idx, dtype=np.int64)]
        nom_act = nom_act[np.asarray(idx, dtype=np.int64)]
        region = region[np.asarray(idx, dtype=np.int64)]

    postsolve_tol = float(
        compare_cfg.qp_postsolve_feas_tol if compare_cfg.qp_postsolve_feas_tol is not None else cbf_cfg.solver_tol
    )
    step_batch_fn = jax.jit(jax.vmap(lambda x, u: quadrotor_step_euler(x, u, cbf_cfg), in_axes=(0, 0)))
    out: dict[str, Any] = {}

    qp_base_set = EllipsoidBaseSet(
        QuadrotorDLQR.from_config(cbf_cfg),
        float(cbf_cfg.base_set_c),
        smooth_gain=float(cbf_cfg.base_set_smooth_gain),
    )
    for policy_name, runtime in runtimes.items():
        projector = QuadrotorBackupCBFProjector(cbf_cfg, runtime=runtime)
        rollout_batch_fn = _build_rollout_batch_fn(cbf_cfg, runtime, base_set=qp_base_set)
        primal_residual_batch_fn = _build_primal_residual_batch_fn(cbf_cfg, runtime)

        delta_u_norm_all: list[np.ndarray] = []
        use_solver_all: list[np.ndarray] = []
        fallback_all: list[np.ndarray] = []
        one_step_admissible_all: list[np.ndarray] = []
        min_rollout_margin_all: list[np.ndarray] = []
        postsolve_feasible_all: list[np.ndarray] = []

        for start in range(0, states.shape[0], max(1, int(compare_cfg.qp_batch_size))):
            end = min(start + int(compare_cfg.qp_batch_size), states.shape[0])
            x = jnp.asarray(states[start:end], dtype=jnp.float32)
            u_nom = jnp.asarray(nom_act[start:end], dtype=jnp.float32)
            u_safe_j, use_solver_j, finite_info_j = projector.solve_batch_with_info(x, u_nom)
            x_safe_plus_j = step_batch_fn(x, u_safe_j)
            one_step_j = rollout_batch_fn(x_safe_plus_j)
            residual_j = primal_residual_batch_fn(x, u_safe_j)

            u_safe = np.asarray(u_safe_j, dtype=np.float64)
            use_solver = np.asarray(use_solver_j, dtype=bool)
            finite_info = jax.tree_util.tree_map(lambda v: np.asarray(v), finite_info_j)
            one_step_success = np.asarray(one_step_j[0], dtype=bool)
            min_hard = np.asarray(one_step_j[6], dtype=np.float64)
            residual = np.asarray(residual_j, dtype=np.float64)
            solver_attempted = ~np.asarray(finite_info["q_saturated"], dtype=bool)
            solver_finite_output = solver_attempted & np.asarray(finite_info["z_finite"], dtype=bool)
            postsolve_feasible = solver_attempted & solver_finite_output & np.isfinite(residual) & (residual <= postsolve_tol)

            delta_u_norm_all.append(np.linalg.norm(np.asarray(nom_act[start:end], dtype=np.float64) - u_safe, axis=1))
            use_solver_all.append(use_solver)
            fallback_all.append(~use_solver)
            one_step_admissible_all.append(one_step_success)
            min_rollout_margin_all.append(min_hard)
            postsolve_feasible_all.append(postsolve_feasible)

        delta_u_norm = np.concatenate(delta_u_norm_all, axis=0)
        use_solver = np.concatenate(use_solver_all, axis=0)
        fallback = np.concatenate(fallback_all, axis=0)
        one_step_admissible = np.concatenate(one_step_admissible_all, axis=0)
        min_rollout_margin = np.concatenate(min_rollout_margin_all, axis=0)
        postsolve_feasible = np.concatenate(postsolve_feasible_all, axis=0)

        by_region = {}
        for region_key in np.unique(region).tolist():
            mask = region == region_key
            by_region[str(region_key)] = {
                "count": int(np.sum(mask)),
                "delta_u_norm": _summary_stats(delta_u_norm[mask]),
                "use_solver_rate": _rate(use_solver[mask].astype(np.float64)),
                "fallback_rate": _rate(fallback[mask].astype(np.float64)),
                "postsolve_feasible_rate": _rate(postsolve_feasible[mask].astype(np.float64)),
                "one_step_admissibility_rate": _rate(one_step_admissible[mask].astype(np.float64)),
                "minimum_rollout_safety_margin": _summary_stats(min_rollout_margin[mask]),
            }

        out[policy_name] = {
            "overall": {
                "count": int(states.shape[0]),
                "delta_u_norm": _summary_stats(delta_u_norm),
                "use_solver_rate": _rate(use_solver.astype(np.float64)),
                "fallback_rate": _rate(fallback.astype(np.float64)),
                "postsolve_feasible_rate": _rate(postsolve_feasible.astype(np.float64)),
                "one_step_admissibility_rate": _rate(one_step_admissible.astype(np.float64)),
                "minimum_rollout_safety_margin": _summary_stats(min_rollout_margin),
            },
            "by_region": by_region,
        }

    return out


def _plot_compare_summary(report: dict[str, Any], out_dir: Path) -> None:
    if not _HAS_MATPLOTLIB:
        return
    policy_names = list(report.get("evaluated_policy_names", report["policies"].keys()))
    if not policy_names:
        return
    by_policy = {name: report["policies"][name]["by_region"] for name in policy_names}
    regions = [
        r
        for r in _REGION_KEYS
        if any(int(by_policy[name].get(r, {}).get("count", 0)) > 0 for name in policy_names)
    ]
    if not regions:
        return

    x = np.arange(len(regions), dtype=np.float64)
    width = 0.8 / max(len(policy_names), 1)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    center_offset = 0.5 * (len(policy_names) - 1)
    for idx, policy_name in enumerate(policy_names):
        offset = (idx - center_offset) * width
        ax.bar(
            x + offset,
            [float(by_policy[policy_name].get(r, {}).get("recoverability_rate", 0.0)) for r in regions],
            width=width,
            label=_POLICY_LABELS.get(policy_name, policy_name),
            color=_POLICY_COLORS.get(policy_name),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=15)
    ax.set_ylabel("Recoverability Rate")
    ax.set_title("Recoverability by Region")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "recoverability_by_region.png", dpi=160)
    plt.close(fig)

    if not {"analytic_backup_policy", "learned_backup_policy"}.issubset(report["policies"].keys()):
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    with np.load(out_dir / "trajectory_compare_arrays.npz") as payload:
        if "analytic_recoverable" not in payload or "learned_recoverable" not in payload:
            plt.close(fig)
            return
        dataset = np.asarray(payload["states"], dtype=np.float64)
        analytic_rec = np.asarray(payload["analytic_recoverable"], dtype=bool)
        learned_rec = np.asarray(payload["learned_recoverable"], dtype=bool)
    lbp_only = learned_rec & (~analytic_rec)
    abp_only = analytic_rec & (~learned_rec)
    both = analytic_rec & learned_rec
    neither = (~analytic_rec) & (~learned_rec)
    if np.any(neither):
        ax.scatter(dataset[neither, 0], dataset[neither, 2], s=8, alpha=0.35, label="neither")
    if np.any(abp_only):
        ax.scatter(dataset[abp_only, 0], dataset[abp_only, 2], s=10, alpha=0.55, label="ABP only")
    if np.any(lbp_only):
        ax.scatter(dataset[lbp_only, 0], dataset[lbp_only, 2], s=10, alpha=0.55, label="LBP only")
    if np.any(both):
        ax.scatter(dataset[both, 0], dataset[both, 2], s=8, alpha=0.35, label="both")
    ax.axhline(float(report["cbf_config"]["z_max"]), color="r", linestyle="--", linewidth=1.0)
    ax.set_xlabel("p_x [m]")
    ax.set_ylabel("p_z [m]")
    ax.set_title("XZ Projection of Recoverability Agreement")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "xz_projection_compare.png", dpi=160)
    plt.close(fig)


__all__ = [
    "QuadrotorTrajectoryCompareConfig",
    "compare_quadrotor_backup_policies",
    "quadrotor_trajectory_compare_config_from_dict",
]
