#!/usr/bin/env python
"""Evaluate saved safe quadrotor PS2 policies.

The saved actor is rebuilt with the exact training configuration and evaluated
through the CIL safety projection specified in the saved run config.

Each requested run/checkpoint is evaluated on a shared seed bank and its
artifacts are saved.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any
import warnings

# Normalize ambiguous GPU backend aliases before importing JAX.
if os.environ.get("JAX_PLATFORMS", "").strip().lower() in {"gpu", "rocm"}:
    os.environ["JAX_PLATFORMS"] = "cuda"
if os.environ.get("JAX_PLATFORM_NAME", "").strip().lower() in {"gpu", "rocm"}:
    os.environ["JAX_PLATFORM_NAME"] = "cuda"

import jax
import jax.numpy as jnp
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    _HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover
    _HAS_MATPLOTLIB = False

from ps2rl.evaluation import quadrotor_vanilla_eval as track_eval_utils

from ps2rl.cil.quadrotor_backup_cbf import (
    QuadrotorBCBFConfig,
    QuadrotorBackupCBFProjector,
    quadrotor_step_euler_jax,
)
from ps2rl.envs.quadrotor_env import (
    QuadrotorEnvConfig,
    _load_reference_bundle,
    build_quadrotor_env,
)
from ps2rl.cil.cil_policy import ActorConfig
from ps2rl.plotting.plots import plot_quad_trajectory
from ps2rl.phase2_ps2.quadrotor_ps2_trainer import (
    SACConfig,
    _build_action_fns,
    _eval_selection_key_return,
    _validate_action_scale,
)

DEFAULT_RUN_GLOB = "*"
DEFAULT_SEED_GROUP_STRIDE = track_eval_utils.DEFAULT_SEED_GROUP_STRIDE
DEFAULT_ROLLOUT_BATCH_SIZE = 256


def _resolve_config_path(run_dir: Path) -> Path:
    for name in ("configs.json", "config.json"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing configs.json/config.json under run directory: {run_dir}")


def _iter_run_dirs(outputs_dir: Path, run_glob: str, experiment: str = "") -> list[Path]:
    if experiment.strip():
        exp = experiment.strip()
        p = Path(exp)
        run_dir = p if p.exists() and p.is_dir() else outputs_dir / exp
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {exp}")
        _resolve_config_path(run_dir)
        return [run_dir]

    run_dirs: list[Path] = []
    for run_dir in sorted(outputs_dir.glob(run_glob)):
        if not run_dir.is_dir():
            continue
        try:
            _resolve_config_path(run_dir)
        except FileNotFoundError:
            continue
        run_dirs.append(run_dir)
    return run_dirs


def _metric_value(metrics: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(metrics.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _selection_formula_payload() -> dict[str, Any]:
    return {
        "selection_priority": [
            "maximize violation_free_episode_rate",
            "maximize safe_rate",
            "minimize tracking_score_mean",
            "minimize tracking_score_p95",
            "minimize pos_xz_rmse_mean",
            "minimize vel_xz_rmse_mean",
            "minimize pitch_rmse_deg_mean",
            "minimize y_rmse_mean",
            "minimize vy_rmse_mean",
        ],
    }


def _summary_stats(x: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    arr = np.asarray(x, dtype=np.float64)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _safe_mean_1d(x: np.ndarray | list[float]) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def _safe_distribution_fields(prefix: str, x: np.ndarray | list[float]) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_average": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_minimum": 0.0,
            f"{prefix}_maximum": 0.0,
        }
    return {
        f"{prefix}_average": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_minimum": float(np.min(arr)),
        f"{prefix}_maximum": float(np.max(arr)),
    }


def _block_until_ready_tree(tree: Any) -> Any:
    for leaf in jax.tree_util.tree_leaves(tree):
        block = getattr(leaf, "block_until_ready", None)
        if callable(block):
            block()
    return tree


def _benchmark_batched_inference_stats(
    eval_action_batch,
    actor_params: Any,
    obs: np.ndarray,
    *,
    actor_obs_dim: int,
    batch_size: int,
) -> dict[str, float]:
    obs_np = np.asarray(obs, dtype=np.float32)
    if obs_np.size == 0:
        return {
            "inference_freq": 0.0,
            "inference_total_decisions": 0.0,
            "inference_total_time_sec": 0.0,
        }
    if obs_np.ndim == 1:
        obs_np = obs_np.reshape(1, -1)
    obs_flat = obs_np.reshape(-1, obs_np.shape[-1])
    obs_actor = jnp.asarray(obs_flat[:, : int(actor_obs_dim)], dtype=jnp.float32)
    chunk_size = max(1, int(batch_size))
    total_decisions = int(obs_actor.shape[0])
    unique_chunk_sizes = sorted(
        {min(chunk_size, total_decisions - start) for start in range(0, total_decisions, chunk_size)}
    )
    for chunk_n in unique_chunk_sizes:
        _block_until_ready_tree(eval_action_batch(actor_params, obs_actor[:chunk_n]))

    total_time_sec = 0.0
    for start in range(0, total_decisions, chunk_size):
        stop = min(start + chunk_size, total_decisions)
        t_start = time.perf_counter()
        outputs = eval_action_batch(actor_params, obs_actor[start:stop])
        _block_until_ready_tree(outputs)
        total_time_sec += float(time.perf_counter() - t_start)

    inference_freq = float(total_decisions / total_time_sec) if total_time_sec > 0.0 else 0.0
    return {
        "inference_freq": inference_freq,
        "inference_total_decisions": float(total_decisions),
        "inference_total_time_sec": float(total_time_sec),
    }


def _select_inference_benchmark_obs(arrays: dict[str, np.ndarray]) -> np.ndarray:
    obs_actor = np.asarray(arrays.get("obs_actor", []), dtype=np.float32)
    if obs_actor.size > 0:
        return obs_actor
    return np.asarray(arrays.get("obs", []), dtype=np.float32)


def _reduce_step_metric_by_episode(
    episode_idx: np.ndarray,
    values: np.ndarray,
    num_episodes: int,
    reducer,
    default: float = 0.0,
) -> np.ndarray:
    out = np.full((int(max(0, num_episodes)),), float(default), dtype=np.float64)
    if out.size == 0:
        return out
    ep = np.asarray(episode_idx, dtype=np.int32).reshape(-1)
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    limit = min(ep.size, vals.size)
    if limit == 0:
        return out
    ep = ep[:limit]
    vals = vals[:limit]
    finite_mask = np.isfinite(vals)
    ep = ep[finite_mask]
    vals = vals[finite_mask]
    for idx in range(out.size):
        mask = ep == idx
        if np.any(mask):
            out[idx] = float(reducer(vals[mask]))
    return out


def _add_public_summary_metric_blocks(summary: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    num_episodes = int(np.asarray(arrays.get("episode_return", []), dtype=np.float64).size)
    episode_idx = np.asarray(arrays.get("episode_idx", []), dtype=np.int32)
    theta_att_rad = 2.0 * np.arcsin(
        np.clip(np.asarray(arrays.get("att_error_norm", []), dtype=np.float64), 0.0, 1.0)
    )
    episode_theta_att_rmse_rad = _reduce_step_metric_by_episode(
        episode_idx,
        theta_att_rad,
        num_episodes,
        lambda x: np.sqrt(np.mean(np.square(x))),
        default=0.0,
    )
    violation_z = np.maximum(-np.asarray(arrays.get("hard_deck_margin", []), dtype=np.float64), 0.0)
    episode_worst_violation_z = _reduce_step_metric_by_episode(
        episode_idx,
        violation_z,
        num_episodes,
        np.max,
        default=0.0,
    )
    episode_cumulative_violation_z = _reduce_step_metric_by_episode(
        episode_idx,
        violation_z,
        num_episodes,
        np.sum,
        default=0.0,
    )

    arrays["episode_theta_att_rmse_rad"] = episode_theta_att_rmse_rad
    arrays["episode_worst_violation_z"] = episode_worst_violation_z
    arrays["episode_cumulative_violation_z"] = episode_cumulative_violation_z

    summary["performance_metrics"] = {
        "x_rmse": _safe_mean_1d(arrays.get("episode_x_rmse", [])),
        "y_rmse": _safe_mean_1d(arrays.get("episode_y_rmse", [])),
        "z_rmse": _safe_mean_1d(arrays.get("episode_z_rmse", [])),
        "pos_rmse": _safe_mean_1d(arrays.get("episode_pos_xyz_rmse", [])),
        "vx_rmse": _safe_mean_1d(arrays.get("episode_vx_rmse", [])),
        "vy_rmse": _safe_mean_1d(arrays.get("episode_vy_rmse", [])),
        "vz_rmse": _safe_mean_1d(arrays.get("episode_vz_rmse", [])),
        "vel_rmse": _safe_mean_1d(arrays.get("episode_vel_xyz_rmse", [])),
        "roll_rmse_rad": _safe_mean_1d(np.deg2rad(np.asarray(arrays.get("episode_roll_rmse_deg", []), dtype=np.float64))),
        "pitch_rmse_rad": _safe_mean_1d(np.deg2rad(np.asarray(arrays.get("episode_pitch_rmse_deg", []), dtype=np.float64))),
        "yaw_rmse_rad": _safe_mean_1d(np.deg2rad(np.asarray(arrays.get("episode_yaw_rmse_deg", []), dtype=np.float64))),
        "theta_att_rmse_rad": _safe_mean_1d(episode_theta_att_rmse_rad),
    }
    summary["safety_metrics"] = {
        "episode_safety_rate": float(summary.get("violation_free_episode_rate", 0.0)),
        **_safe_distribution_fields("worst_violation_z", episode_worst_violation_z),
        **_safe_distribution_fields("cumulative_violation_z", episode_cumulative_violation_z),
    }
    return summary


def _build_eval_dir_name(
    tag: str,
    checkpoint: str,
    num_eval_seeds: int,
    episodes_per_seed: int,
    label: str,
) -> str:
    total_episodes = int(num_eval_seeds) * int(episodes_per_seed)
    parts = [
        f"quadSAC_eval-{tag}",
        f"ckpt_{track_eval_utils._sanitize_token(checkpoint)}",
        f"seeds{int(num_eval_seeds)}",
        f"epPerSeed{int(episodes_per_seed)}",
        f"epTotal{int(total_episodes)}",
    ]
    if label.strip():
        parts.append(track_eval_utils._sanitize_token(label.strip()))
    return "-".join(parts)


def _resolve_checkpoint_weights_path(run_dir: Path, checkpoint: str) -> Path:
    checkpoint_key = str(checkpoint).strip()
    filename_map = {
        "best": "best_weights.pkl",
        "final": "final_weights.pkl",
    }
    candidate = run_dir / filename_map.get(checkpoint_key, f"{checkpoint_key}_weights.pkl")
    if not candidate.exists():
        raise FileNotFoundError(
            f"Checkpoint weights not found for checkpoint='{checkpoint_key}': {candidate}"
        )
    return candidate


def _build_reference_tables(env_cfg: QuadrotorEnvConfig, max_steps: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    step_idx = np.arange(int(max_steps) + 1, dtype=np.float64)
    t = step_idx * float(env_cfg.dt)

    bundle = _load_reference_bundle(env_cfg.reference_path)
    ref_states = np.asarray(bundle["states"], dtype=np.float64)
    ref_omega = np.asarray(bundle["omega_cmd"], dtype=np.float64)
    ref_dt = float(env_cfg.reference_dt if env_cfg.reference_dt is not None else env_cfg.dt)
    last = max(int(ref_states.shape[0] - 1), 1)
    idx_float = np.clip(t / ref_dt, 0.0, float(last))
    i0 = np.floor(idx_float).astype(np.int32)
    i1 = np.minimum(i0 + 1, int(ref_states.shape[0] - 1))
    w = (idx_float - i0.astype(np.float64))[:, None]
    states = (1.0 - w) * ref_states[i0] + w * ref_states[i1]
    states[:, 6:10] = track_eval_utils._normalize_quaternion_np(states[:, 6:10])
    omega = (1.0 - w) * ref_omega[i0] + w * ref_omega[i1]
    progress = np.clip(idx_float / float(last), 0.0, 1.0)
    phase = 2.0 * np.pi * progress
    final_ref_state = np.asarray(ref_states[-1], dtype=np.float64)
    ref_idx_floor = i0.astype(np.float32)
    ref_idx_ceil = i1.astype(np.float32)
    ref_interp = (idx_float - np.floor(idx_float)).astype(np.float32)

    tables = {
        "ref_state": states.astype(np.float32),
        "ref_omega": omega.astype(np.float32),
        "ref_time_sec": t.astype(np.float32),
        "ref_progress": progress.astype(np.float32),
        "ref_phase_sin": np.sin(phase).astype(np.float32),
        "ref_phase_cos": np.cos(phase).astype(np.float32),
        "ref_idx_floor": ref_idx_floor,
        "ref_idx_ceil": ref_idx_ceil,
        "ref_interp": ref_interp,
    }
    return tables, final_ref_state.astype(np.float64)


def _save_best_episode_artifacts(
    arrays: dict[str, np.ndarray],
    *,
    best_episode_idx: int,
    checkpoint: str,
    out_dir: Path,
    env_cfg: QuadrotorEnvConfig,
    gif_slowdown: float,
    gif_trail_length: int,
    gif_print_every: int,
    artifact_prefix: str = "best",
    title_label: str = "best evaluation",
    score_key: str = "episode_tracking_score",
    score_label: str = "score",
    save_gif: bool = True,
) -> dict[str, Any]:
    trace = track_eval_utils._extract_episode_trace(arrays, best_episode_idx)
    if not trace:
        return {
            "available": False,
            "message": f"Episode trace missing for best episode {best_episode_idx}.",
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"{artifact_prefix}_episode_trace.npz"
    np.savez(trace_path, **trace)

    score_array = np.asarray(arrays.get(score_key, []), dtype=np.float64)
    score_value = float(score_array[best_episode_idx]) if best_episode_idx < score_array.size else float("nan")

    title = (
        f"{checkpoint} checkpoint {title_label} episode | "
        f"ep={best_episode_idx} | "
        f"safe={float(np.asarray(arrays['episode_safe_rate'])[best_episode_idx]):.3f} | "
        f"{score_label}={score_value:.4f}"
    )

    plot_path = out_dir / f"{artifact_prefix}_trajectory.png"
    if _HAS_MATPLOTLIB:
        try:
            plot_quad_trajectory(
                trace,
                z_max=float(env_cfg.z_max),
                output_path=str(plot_path),
                dt=float(env_cfg.dt),
            )
        except Exception:  # pragma: no cover
            pass

    gif_path: str | None = None
    gif_error: str | None = None
    if save_gif:
        try:
            gif_path = track_eval_utils._save_best_episode_gif(
                trace,
                out_path=out_dir / f"{artifact_prefix}_trajectory.gif",
                title=title,
                slowdown=float(gif_slowdown),
                trail_length=max(1, int(gif_trail_length)),
                print_every=max(1, int(gif_print_every)),
            )
        except Exception as exc:  # noqa: BLE001
            gif_error = str(exc)

    return {
        "available": True,
        "trace_path": str(trace_path),
        "plot_path": str(plot_path) if plot_path.exists() else None,
        "gif_path": gif_path,
        "gif_error": gif_error,
    }


def _evaluate_checkpoint(
    actor_params: Any,
    sac_cfg: SACConfig,
    env_cfg: QuadrotorEnvConfig,
    cbf_cfg: QuadrotorBCBFConfig,
    *,
    num_eval_seeds: int,
    episodes_per_seed: int,
    eval_seed_base: int,
    seed_group_stride: int,
    rollout_batch_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    env_fns = build_quadrotor_env(env_cfg)
    projector = QuadrotorBackupCBFProjector(cbf_cfg)
    backup_runtime = projector.runtime

    action_scale = np.array(
        [cbf_cfg.a_cmd_max, cbf_cfg.omega_max, cbf_cfg.omega_max, cbf_cfg.omega_max],
        dtype=np.float64,
    )
    expected_action_scale = np.asarray(jax.device_get(cbf_cfg.action_scale), dtype=np.float64)
    _validate_action_scale("eval.action_scale", action_scale, expected_action_scale)
    _validate_action_scale(
        "env_vs_cbf_action_scale",
        np.array(
            [env_cfg.a_cmd_max, env_cfg.omega_max, env_cfg.omega_max, env_cfg.omega_max],
            dtype=np.float64,
        ),
        expected_action_scale,
    )

    actor_obs_dim = track_eval_utils._infer_actor_obs_dim(actor_params)
    if actor_obs_dim is None:
        actor_obs_dim = int(env_fns.obs_dim)
    if actor_obs_dim > int(env_fns.obs_dim):
        raise ValueError(
            f"Actor expects obs_dim={actor_obs_dim}, but env observation dim is only {env_fns.obs_dim}."
        )

    actor_cfg = ActorConfig(
        obs_dim=int(actor_obs_dim),
        action_dim=env_fns.action_dim,
        hidden_sizes=(sac_cfg.hidden_size, sac_cfg.hidden_size),
    )
    _, eval_action_single = _build_action_fns(
        sac_cfg,
        actor_cfg,
        cbf_cfg,
        jnp.asarray(action_scale, dtype=jnp.float32),
        backup_runtime=backup_runtime,
        return_solver_info=True,
    )
    batched_eval_action = jax.jit(jax.vmap(eval_action_single, in_axes=(None, 0)))
    batched_next_state = jax.jit(
        jax.vmap(lambda x, u: quadrotor_step_euler_jax(x, u, cbf_cfg, dt=float(env_cfg.dt)), in_axes=(0, 0))
    )
    batched_residuals = jax.jit(jax.vmap(projector.residuals, in_axes=(0, 0)))

    ref_tables_np, _ = _build_reference_tables(env_cfg, int(env_cfg.max_steps))
    ref_tables_jax = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in ref_tables_np.items()}

    max_steps = int(env_cfg.max_steps)
    total_episodes = int(num_eval_seeds) * int(episodes_per_seed)
    projection_active = bool(sac_cfg.use_projection and sac_cfg.project_actor_actions)
    qp_success_tol = max(float(cbf_cfg.solver_tol) * 10.0, 1e-6)

    def _broadcast_mask(mask: jax.Array, ndim: int) -> jax.Array:
        out = mask
        while out.ndim < ndim:
            out = out[..., None]
        return out

    def _mask_like(arr: jax.Array, active: jax.Array, fill_value: float = 0.0) -> jax.Array:
        mask = _broadcast_mask(active, arr.ndim)
        fill = jnp.full_like(arr, fill_value)
        return jnp.where(mask, arr, fill)

    @jax.jit
    def rollout_chunk(actor_params_in: Any, reset_keys: jax.Array) -> dict[str, jax.Array]:
        batch_size = reset_keys.shape[0]
        state, obs = env_fns.reset_batched(reset_keys)
        done = jnp.zeros((batch_size,), dtype=jnp.bool_)
        zero_action = jnp.zeros((batch_size, env_fns.action_dim), dtype=jnp.float32)

        def scan_step(carry: tuple[Any, jax.Array, jax.Array], step_idx: jax.Array):
            state_now, obs_now, done_now = carry
            active = jnp.logical_not(done_now)
            obs_actor = obs_now[:, : actor_obs_dim]
            safe_action, raw_action, use_solver, finite_info = batched_eval_action(actor_params_in, obs_actor)

            action_for_env = jnp.where(active[:, None], safe_action, zero_action)
            step_keys = jax.vmap(lambda k: jax.random.fold_in(k, step_idx + jnp.int32(1)))(reset_keys)
            state_raw, _, obs_raw, rew_raw, done_raw, info_raw = env_fns.step_batched(state_now, action_for_env, step_keys)

            next_x = batched_next_state(state_now.x, action_for_env)
            if projection_active:
                residuals = batched_residuals(state_now.x, action_for_env)
                residuals_finite = jnp.all(jnp.isfinite(residuals), axis=-1)
                residual_max = jnp.max(
                    jnp.maximum(jnp.nan_to_num(residuals, nan=jnp.inf, posinf=jnp.inf, neginf=0.0), 0.0),
                    axis=-1,
                )
                successful_qp = (
                    active
                    & use_solver
                    & finite_info["inputs_finite"]
                    & finite_info["z_finite"]
                    & residuals_finite
                    & (residual_max <= jnp.asarray(qp_success_tol, dtype=jnp.float32))
                )
                fallback = active & ~use_solver
            else:
                successful_qp = jnp.zeros((batch_size,), dtype=jnp.bool_)
                fallback = jnp.zeros((batch_size,), dtype=jnp.bool_)
                residual_max = jnp.zeros((batch_size,), dtype=jnp.float32)

            ref_state_step = jnp.broadcast_to(ref_tables_jax["ref_state"][step_idx + 1], next_x.shape)
            ref_omega_next = jnp.broadcast_to(ref_tables_jax["ref_omega"][step_idx + 1], (batch_size, 3))
            reward_ref_omega = 0.5 * (
                jnp.broadcast_to(ref_tables_jax["ref_omega"][step_idx], (batch_size, 3)) + ref_omega_next
            )
            ref_time_sec = jnp.broadcast_to(ref_tables_jax["ref_time_sec"][step_idx + 1], (batch_size,))
            ref_progress = jnp.broadcast_to(ref_tables_jax["ref_progress"][step_idx + 1], (batch_size,))
            ref_phase_sin = jnp.broadcast_to(ref_tables_jax["ref_phase_sin"][step_idx + 1], (batch_size,))
            ref_phase_cos = jnp.broadcast_to(ref_tables_jax["ref_phase_cos"][step_idx + 1], (batch_size,))
            ref_idx_floor = jnp.broadcast_to(ref_tables_jax["ref_idx_floor"][step_idx + 1], (batch_size,))
            ref_idx_ceil = jnp.broadcast_to(ref_tables_jax["ref_idx_ceil"][step_idx + 1], (batch_size,))
            ref_interp = jnp.broadcast_to(ref_tables_jax["ref_interp"][step_idx + 1], (batch_size,))


            done_step = active & done_raw
            done_out = done_now | done_step

            record = {
                "valid_mask": active,
                "episode_done": done_step,
                "obs": _mask_like(state_now.x, active),
                "obs_actor": _mask_like(obs_actor, active),
                "next_obs": _mask_like(next_x, active),
                "act": _mask_like(safe_action, active),
                "act_raw": _mask_like(raw_action, active),
                "rew": _mask_like(rew_raw.astype(jnp.float32), active),
                "safe": _mask_like(info_raw.is_safe.astype(jnp.float32), active),
                "use_solver": _mask_like(use_solver.astype(jnp.float32), active),
                "fallback": _mask_like(fallback.astype(jnp.float32), active),
                "successful_qp": _mask_like(successful_qp.astype(jnp.float32), active),
                "qp_residual_max": _mask_like(residual_max.astype(jnp.float32), active),
                "pos_error_norm": _mask_like(info_raw.pos_error_norm.astype(jnp.float32), active),
                "vel_error_norm": _mask_like(info_raw.vel_error_norm.astype(jnp.float32), active),
                "att_error_norm": _mask_like(info_raw.att_error_norm.astype(jnp.float32), active),
                "omega_ref_error_norm": _mask_like(info_raw.omega_ref_error_norm.astype(jnp.float32), active),
                "hard_deck_margin": _mask_like(info_raw.hard_deck_margin.astype(jnp.float32), active),
                "ref_time_sec": _mask_like(ref_time_sec.astype(jnp.float32), active),
                "ref_progress": _mask_like(ref_progress.astype(jnp.float32), active),
                "ref_phase_sin": _mask_like(ref_phase_sin.astype(jnp.float32), active),
                "ref_phase_cos": _mask_like(ref_phase_cos.astype(jnp.float32), active),
                "ref_idx_floor": _mask_like(ref_idx_floor.astype(jnp.float32), active),
                "ref_idx_ceil": _mask_like(ref_idx_ceil.astype(jnp.float32), active),
                "ref_interp": _mask_like(ref_interp.astype(jnp.float32), active),
                "ref_state": _mask_like(ref_state_step.astype(jnp.float32), active),
                "ref_omega_x": _mask_like(reward_ref_omega[:, 0].astype(jnp.float32), active),
                "ref_omega_y": _mask_like(reward_ref_omega[:, 1].astype(jnp.float32), active),
                "ref_omega_z": _mask_like(reward_ref_omega[:, 2].astype(jnp.float32), active),
                "obs_ref_omega_x": _mask_like(ref_omega_next[:, 0].astype(jnp.float32), active),
                "obs_ref_omega_y": _mask_like(ref_omega_next[:, 1].astype(jnp.float32), active),
                "obs_ref_omega_z": _mask_like(ref_omega_next[:, 2].astype(jnp.float32), active),
            }

            state_out = jax.tree_util.tree_map(
                lambda new, old: jnp.where(_broadcast_mask(active, new.ndim), new, old),
                state_raw,
                state_now,
            )
            obs_out = jnp.where(active[:, None], obs_raw, obs_now)
            return (state_out, obs_out, done_out), record

        (_, _, _), records = jax.lax.scan(
            scan_step,
            (state, obs, done),
            jnp.arange(max_steps, dtype=jnp.int32),
        )
        return records

    warm_seed = jnp.asarray([eval_seed_base + 999_999], dtype=jnp.uint32)
    warm_keys = jax.vmap(jax.random.PRNGKey)(warm_seed)
    warm_obs_state, warm_obs = env_fns.reset_batched(warm_keys)
    _ = warm_obs_state
    warm_obs_actor = warm_obs[:, : actor_obs_dim]
    _block_until_ready_tree(batched_eval_action(actor_params, warm_obs_actor))

    episode_seed_group_all = np.repeat(np.arange(int(num_eval_seeds), dtype=np.int32), int(episodes_per_seed))
    episode_in_seed_all = np.tile(np.arange(int(episodes_per_seed), dtype=np.int32), int(num_eval_seeds))
    episode_seed_all = (
        int(eval_seed_base)
        + episode_seed_group_all.astype(np.int64) * int(seed_group_stride)
        + episode_in_seed_all.astype(np.int64)
    )

    episode_acc: dict[str, list[np.ndarray]] = {}
    step_acc: dict[str, list[np.ndarray]] = {}

    def append_episode(name: str, value: np.ndarray) -> None:
        episode_acc.setdefault(name, []).append(np.asarray(value))

    def append_step(name: str, value: np.ndarray) -> None:
        step_acc.setdefault(name, []).append(np.asarray(value))

    def masked_mean(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        masked = np.where(mask, arr, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out = np.nanmean(masked, axis=0)
        return np.nan_to_num(out, nan=0.0)

    def masked_rms(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        masked_sq = np.where(mask, np.square(arr), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out = np.sqrt(np.nanmean(masked_sq, axis=0))
        return np.nan_to_num(out, nan=0.0)

    def masked_percentile(arr: np.ndarray, mask: np.ndarray, q: float) -> np.ndarray:
        masked = np.where(mask, arr, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out = np.nanpercentile(masked, q, axis=0)
        return np.nan_to_num(out, nan=0.0)

    def masked_max(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        masked = np.where(mask, arr, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out = np.nanmax(masked, axis=0)
        return np.nan_to_num(out, nan=0.0)

    for start in range(0, total_episodes, int(rollout_batch_size)):
        end = min(start + int(rollout_batch_size), total_episodes)
        seeds_chunk = episode_seed_all[start:end]
        seed_groups_chunk = episode_seed_group_all[start:end]
        seed_in_chunk = episode_in_seed_all[start:end]
        batch_size = int(end - start)

        reset_keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds_chunk, dtype=jnp.uint32))
        records = jax.device_get(rollout_chunk(actor_params, reset_keys))

        valid = np.asarray(records["valid_mask"], dtype=bool)
        done_step = np.asarray(records["episode_done"], dtype=bool)
        obs = np.asarray(records["obs"], dtype=np.float64)
        obs_actor = np.asarray(records["obs_actor"], dtype=np.float64)
        next_obs = np.asarray(records["next_obs"], dtype=np.float64)
        act = np.asarray(records["act"], dtype=np.float64)
        act_raw = np.asarray(records["act_raw"], dtype=np.float64)
        rew = np.asarray(records["rew"], dtype=np.float64)
        safe = np.asarray(records["safe"], dtype=np.float64)
        use_solver = np.asarray(records["use_solver"], dtype=np.float64)
        fallback = np.asarray(records["fallback"], dtype=np.float64)
        successful_qp = np.asarray(records["successful_qp"], dtype=np.float64)
        qp_residual_max = np.asarray(records["qp_residual_max"], dtype=np.float64)
        pos_error_norm = np.asarray(records["pos_error_norm"], dtype=np.float64)
        vel_error_norm = np.asarray(records["vel_error_norm"], dtype=np.float64)
        att_error_norm = np.asarray(records["att_error_norm"], dtype=np.float64)
        omega_ref_error_norm = np.asarray(records["omega_ref_error_norm"], dtype=np.float64)
        hard_deck_margin = np.asarray(records["hard_deck_margin"], dtype=np.float64)
        ref_time_sec = np.asarray(records["ref_time_sec"], dtype=np.float64)
        ref_progress = np.asarray(records["ref_progress"], dtype=np.float64)
        ref_phase_sin = np.asarray(records["ref_phase_sin"], dtype=np.float64)
        ref_phase_cos = np.asarray(records["ref_phase_cos"], dtype=np.float64)
        ref_idx_floor = np.asarray(records["ref_idx_floor"], dtype=np.float64)
        ref_idx_ceil = np.asarray(records["ref_idx_ceil"], dtype=np.float64)
        ref_interp = np.asarray(records["ref_interp"], dtype=np.float64)
        ref_state = np.asarray(records["ref_state"], dtype=np.float64)
        ref_omega_x = np.asarray(records["ref_omega_x"], dtype=np.float64)
        ref_omega_y = np.asarray(records["ref_omega_y"], dtype=np.float64)
        ref_omega_z = np.asarray(records["ref_omega_z"], dtype=np.float64)
        obs_ref_omega_x = np.asarray(records["obs_ref_omega_x"], dtype=np.float64)
        obs_ref_omega_y = np.asarray(records["obs_ref_omega_y"], dtype=np.float64)
        obs_ref_omega_z = np.asarray(records["obs_ref_omega_z"], dtype=np.float64)

        x_error = next_obs[..., 0] - ref_state[..., 0]
        y_error = next_obs[..., 1] - ref_state[..., 1]
        z_error = next_obs[..., 2] - ref_state[..., 2]
        vx_error = next_obs[..., 3] - ref_state[..., 3]
        vy_error = next_obs[..., 4] - ref_state[..., 4]
        vz_error = next_obs[..., 5] - ref_state[..., 5]
        pos_xyz_error = np.sqrt(np.square(x_error) + np.square(y_error) + np.square(z_error))
        pos_xz_error = np.sqrt(np.square(x_error) + np.square(z_error))
        vel_xz_error = np.sqrt(np.square(vx_error) + np.square(vz_error))

        roll_deg_flat, pitch_deg_flat, yaw_deg_flat = track_eval_utils._quaternion_to_euler_deg_batch_np(
            next_obs[..., 6:10].reshape(-1, 4)
        )
        ref_roll_deg_flat, ref_pitch_deg_flat, ref_yaw_deg_flat = track_eval_utils._quaternion_to_euler_deg_batch_np(
            ref_state[..., 6:10].reshape(-1, 4)
        )
        roll_deg = roll_deg_flat.reshape(max_steps, batch_size)
        pitch_deg = pitch_deg_flat.reshape(max_steps, batch_size)
        yaw_deg = yaw_deg_flat.reshape(max_steps, batch_size)
        ref_roll_deg = ref_roll_deg_flat.reshape(max_steps, batch_size)
        ref_pitch_deg = ref_pitch_deg_flat.reshape(max_steps, batch_size)
        ref_yaw_deg = ref_yaw_deg_flat.reshape(max_steps, batch_size)

        roll_error_deg = np.abs(
            np.rad2deg(track_eval_utils._wrap_angle_rad(np.deg2rad(roll_deg - ref_roll_deg)))
        )
        pitch_error_deg = np.abs(
            np.rad2deg(track_eval_utils._wrap_angle_rad(np.deg2rad(pitch_deg - ref_pitch_deg)))
        )
        yaw_error_deg = np.abs(
            np.rad2deg(track_eval_utils._wrap_angle_rad(np.deg2rad(yaw_deg - ref_yaw_deg)))
        )

        episode_length = np.sum(valid, axis=0).astype(np.int32)
        episode_return = np.sum(rew * valid.astype(np.float64), axis=0)
        episode_safe_rate = masked_mean(safe, valid)
        episode_violation_free = np.all(~valid | (safe >= 0.5), axis=0).astype(np.float64)

        episode_x_rmse = masked_rms(x_error, valid)
        episode_y_rmse = masked_rms(y_error, valid)
        episode_z_rmse = masked_rms(z_error, valid)
        episode_vx_rmse = masked_rms(vx_error, valid)
        episode_vy_rmse = masked_rms(vy_error, valid)
        episode_vz_rmse = masked_rms(vz_error, valid)
        episode_pos_xyz_rmse = masked_rms(pos_xyz_error, valid)
        episode_vel_xyz_rmse = masked_rms(
            np.sqrt(np.square(vx_error) + np.square(vy_error) + np.square(vz_error)),
            valid,
        )
        episode_pos_xz_rmse = masked_rms(pos_xz_error, valid)
        episode_vel_xz_rmse = masked_rms(vel_xz_error, valid)
        episode_p95_pos_xyz = masked_percentile(pos_xyz_error, valid, 95.0)
        episode_max_pos_xyz = masked_max(pos_xyz_error, valid)
        episode_p95_pos_xz = masked_percentile(pos_xz_error, valid, 95.0)
        episode_max_pos_xz = masked_max(pos_xz_error, valid)
        episode_p95_vel_xz = masked_percentile(vel_xz_error, valid, 95.0)
        episode_max_vel_xz = masked_max(vel_xz_error, valid)
        episode_roll_rmse_deg = masked_rms(roll_error_deg, valid)
        episode_pitch_rmse_deg = masked_rms(pitch_error_deg, valid)
        episode_yaw_rmse_deg = masked_rms(yaw_error_deg, valid)
        episode_pos_error_norm_mean = masked_mean(pos_error_norm, valid)
        episode_vel_error_norm_mean = masked_mean(vel_error_norm, valid)
        episode_att_error_norm_mean = masked_mean(att_error_norm, valid)
        episode_omega_ref_error_norm_mean = masked_mean(omega_ref_error_norm, valid)

        episode_tracking_score = np.zeros((batch_size,), dtype=np.float64)
        finite_episode_score = np.ones((batch_size,), dtype=bool)
        tracking_metric_map = {
            "pos_xz_rmse": episode_pos_xz_rmse,
            "vel_xz_rmse": episode_vel_xz_rmse,
            "pitch_rmse_deg": episode_pitch_rmse_deg,
            "p95_pos_xz": episode_p95_pos_xz,
            "max_pos_xz": episode_max_pos_xz,
            "y_rmse": episode_y_rmse,
            "vy_rmse": episode_vy_rmse,
            "roll_rmse_deg": episode_roll_rmse_deg,
            "yaw_rmse_deg": episode_yaw_rmse_deg,
        }
        for key, term in track_eval_utils.TRACKING_SCORE_TERMS.items():
            vals = np.asarray(tracking_metric_map[key], dtype=np.float64)
            finite_episode_score &= np.isfinite(vals)
            episode_tracking_score += float(term.weight) * vals / float(term.scale)
        episode_tracking_score[~finite_episode_score] = np.inf

        global_episode_idx = np.arange(start, end, dtype=np.int32)
        append_episode("episode_idx_unique", global_episode_idx)
        append_episode("episode_seed_group", seed_groups_chunk.astype(np.int32))
        append_episode("episode_in_seed_group", seed_in_chunk.astype(np.int32))
        append_episode("episode_seed", seeds_chunk.astype(np.int64))
        append_episode("episode_return", episode_return)
        append_episode("episode_length", episode_length)
        append_episode("episode_tracking_score", episode_tracking_score)
        append_episode("episode_x_rmse", episode_x_rmse)
        append_episode("episode_y_rmse", episode_y_rmse)
        append_episode("episode_z_rmse", episode_z_rmse)
        append_episode("episode_vx_rmse", episode_vx_rmse)
        append_episode("episode_vy_rmse", episode_vy_rmse)
        append_episode("episode_vz_rmse", episode_vz_rmse)
        append_episode("episode_pos_xyz_rmse", episode_pos_xyz_rmse)
        append_episode("episode_vel_xyz_rmse", episode_vel_xyz_rmse)
        append_episode("episode_pos_xz_rmse", episode_pos_xz_rmse)
        append_episode("episode_vel_xz_rmse", episode_vel_xz_rmse)
        append_episode("episode_p95_pos_xyz", episode_p95_pos_xyz)
        append_episode("episode_max_pos_xyz", episode_max_pos_xyz)
        append_episode("episode_p95_pos_xz", episode_p95_pos_xz)
        append_episode("episode_max_pos_xz", episode_max_pos_xz)
        append_episode("episode_p95_vel_xz", episode_p95_vel_xz)
        append_episode("episode_max_vel_xz", episode_max_vel_xz)
        append_episode("episode_roll_rmse_deg", episode_roll_rmse_deg)
        append_episode("episode_pitch_rmse_deg", episode_pitch_rmse_deg)
        append_episode("episode_yaw_rmse_deg", episode_yaw_rmse_deg)
        append_episode("episode_pos_error_norm_mean", episode_pos_error_norm_mean)
        append_episode("episode_vel_error_norm_mean", episode_vel_error_norm_mean)
        append_episode("episode_att_error_norm_mean", episode_att_error_norm_mean)
        append_episode("episode_omega_ref_error_norm_mean", episode_omega_ref_error_norm_mean)
        append_episode("episode_safe_rate", episode_safe_rate)
        append_episode("episode_use_solver_rate", masked_mean(use_solver, valid))
        append_episode("episode_fallback_rate", masked_mean(fallback, valid))
        append_episode("episode_successful_qp_rate", masked_mean(successful_qp, valid))
        append_episode("episode_violation_free", episode_violation_free)

        valid_ep = valid.T
        flat_mask = valid_ep.reshape(-1)
        ep_idx_grid = np.broadcast_to(global_episode_idx[:, None], (batch_size, max_steps))
        seed_group_grid = np.broadcast_to(seed_groups_chunk[:, None], (batch_size, max_steps))
        seed_in_grid = np.broadcast_to(seed_in_chunk[:, None], (batch_size, max_steps))
        seed_value_grid = np.broadcast_to(seeds_chunk[:, None], (batch_size, max_steps))
        step_grid = np.broadcast_to(np.arange(max_steps, dtype=np.int32)[None, :], (batch_size, max_steps))

        def flatten_step(arr: np.ndarray) -> np.ndarray:
            arr_ep = np.swapaxes(arr, 0, 1)
            return arr_ep.reshape((-1,) + arr_ep.shape[2:])[flat_mask]

        append_step("episode_idx", ep_idx_grid.reshape(-1)[flat_mask].astype(np.int32))
        append_step("seed_group_idx", seed_group_grid.reshape(-1)[flat_mask].astype(np.int32))
        append_step("episode_in_seed_group_step", seed_in_grid.reshape(-1)[flat_mask].astype(np.int32))
        append_step("episode_seed_step", seed_value_grid.reshape(-1)[flat_mask].astype(np.int64))
        append_step("step_in_episode", step_grid.reshape(-1)[flat_mask].astype(np.int32))
        for name, arr in (
            ("ref_time_sec", ref_time_sec),
            ("ref_progress", ref_progress),
            ("ref_phase_sin", ref_phase_sin),
            ("ref_phase_cos", ref_phase_cos),
            ("ref_idx_floor", ref_idx_floor),
            ("ref_idx_ceil", ref_idx_ceil),
            ("ref_interp", ref_interp),
            ("obs", obs),
            ("obs_actor", obs_actor),
            ("next_obs", next_obs),
            ("act", act),
            ("act_raw", act_raw),
            ("rew", rew),
            ("safe", safe),
            ("use_solver", use_solver),
            ("fallback", fallback),
            ("successful_qp", successful_qp),
            ("qp_residual_max", qp_residual_max),
            ("pos_error_norm", pos_error_norm),
            ("vel_error_norm", vel_error_norm),
            ("att_error_norm", att_error_norm),
            ("omega_ref_error_norm", omega_ref_error_norm),
            ("hard_deck_margin", hard_deck_margin),
            ("x_error", x_error),
            ("y_error", y_error),
            ("z_error", z_error),
            ("vx_error", vx_error),
            ("vy_error", vy_error),
            ("vz_error", vz_error),
            ("pos_xz_error", pos_xz_error),
            ("vel_xz_error", vel_xz_error),
            ("roll_deg", roll_deg),
            ("pitch_deg", pitch_deg),
            ("yaw_deg", yaw_deg),
            ("ref_roll_deg", ref_roll_deg),
            ("ref_pitch_deg", ref_pitch_deg),
            ("ref_yaw_deg", ref_yaw_deg),
            ("roll_error_deg", roll_error_deg),
            ("pitch_error_deg", pitch_error_deg),
            ("yaw_error_deg", yaw_error_deg),
            ("ref_state", ref_state),
            ("ref_omega_x", ref_omega_x),
            ("ref_omega_y", ref_omega_y),
            ("ref_omega_z", ref_omega_z),
            ("obs_ref_omega_x", obs_ref_omega_x),
            ("obs_ref_omega_y", obs_ref_omega_y),
            ("obs_ref_omega_z", obs_ref_omega_z),
            ("episode_done_step", done_step.astype(np.float64)),
        ):
            append_step(name, flatten_step(arr))

    arrays: dict[str, np.ndarray] = {
        key: np.concatenate(vals, axis=0) if vals else np.asarray([], dtype=np.float64)
        for key, vals in episode_acc.items()
    }
    arrays.update(
        {
            key: np.concatenate(vals, axis=0) if vals else np.asarray([], dtype=np.float64)
            for key, vals in step_acc.items()
        }
    )

    ep_x_rmse = np.asarray(arrays["episode_x_rmse"], dtype=np.float64)
    ep_y_rmse = np.asarray(arrays["episode_y_rmse"], dtype=np.float64)
    ep_z_rmse = np.asarray(arrays["episode_z_rmse"], dtype=np.float64)
    ep_pos_xyz_rmse = np.asarray(arrays["episode_pos_xyz_rmse"], dtype=np.float64)
    ep_p95_pos_xyz = np.asarray(arrays["episode_p95_pos_xyz"], dtype=np.float64)
    ep_max_pos_xyz = np.asarray(arrays["episode_max_pos_xyz"], dtype=np.float64)
    ep_pos_xz_rmse = np.asarray(arrays["episode_pos_xz_rmse"], dtype=np.float64)
    ep_vel_xz_rmse = np.asarray(arrays["episode_vel_xz_rmse"], dtype=np.float64)
    ep_pitch_rmse = np.asarray(arrays["episode_pitch_rmse_deg"], dtype=np.float64)
    ep_y_rmse = np.asarray(arrays["episode_y_rmse"], dtype=np.float64)
    ep_vy_rmse = np.asarray(arrays["episode_vy_rmse"], dtype=np.float64)
    ep_roll_rmse = np.asarray(arrays["episode_roll_rmse_deg"], dtype=np.float64)
    ep_yaw_rmse = np.asarray(arrays["episode_yaw_rmse_deg"], dtype=np.float64)
    ep_p95_pos_xz = np.asarray(arrays["episode_p95_pos_xz"], dtype=np.float64)
    ep_max_pos_xz = np.asarray(arrays["episode_max_pos_xz"], dtype=np.float64)
    ep_return = np.asarray(arrays["episode_return"], dtype=np.float64)
    ep_length = np.asarray(arrays["episode_length"], dtype=np.int32)
    ep_safe_rate = np.asarray(arrays["episode_safe_rate"], dtype=np.float64)
    ep_violation_free = np.asarray(arrays["episode_violation_free"], dtype=np.float64)
    ep_seed_group = np.asarray(arrays["episode_seed_group"], dtype=np.int32)

    best_episode_idx = 0
    if total_episodes > 0:
        return_episode_keys = [
            _eval_selection_key_return(
                {
                    "violation_free_episode_rate": float(ep_violation_free[idx]),
                    "safe_rate": float(ep_safe_rate[idx]),
                    "return_mean": float(ep_return[idx]),
                }
            )
            for idx in range(total_episodes)
        ]
        best_episode_idx = int(min(range(total_episodes), key=lambda i: return_episode_keys[i]))

    step_safe = np.asarray(arrays["safe"], dtype=np.float64)

    seed_group_stats: list[dict[str, Any]] = []
    for seed_group_idx in range(int(num_eval_seeds)):
        mask = ep_seed_group == int(seed_group_idx)
        if not np.any(mask):
            continue
        seed_group_stats.append(
            {
                "seed_group_idx": int(seed_group_idx),
                "seed_group_base": int(eval_seed_base + seed_group_idx * seed_group_stride),
                "episodes": int(np.sum(mask)),
                "violation_free_episode_rate": track_eval_utils._safe_mean(ep_violation_free[mask]),
                "safe_rate": track_eval_utils._safe_mean(ep_safe_rate[mask]),
                "pos_xz_rmse_mean": track_eval_utils._safe_mean(ep_pos_xz_rmse[mask]),
                "vel_xz_rmse_mean": track_eval_utils._safe_mean(ep_vel_xz_rmse[mask]),
                "pitch_rmse_deg_mean": track_eval_utils._safe_mean(ep_pitch_rmse[mask]),
                "return_mean": track_eval_utils._safe_mean(ep_return[mask]),
            }
        )

    summary: dict[str, Any] = {
        "episodes_total": int(total_episodes),
        "num_eval_seeds": int(num_eval_seeds),
        "episodes_per_seed": int(episodes_per_seed),
        "eval_seed_base": int(eval_seed_base),
        "seed_group_stride": int(seed_group_stride),
        "total_steps": int(step_safe.size),
        "return_mean": track_eval_utils._safe_mean(ep_return),
        "return_std": track_eval_utils._safe_std(ep_return),
        "safe_rate": track_eval_utils._safe_mean(step_safe),
        "safe_rate_step_mean": track_eval_utils._safe_mean(step_safe),
        "unsafe_rate": 1.0 - track_eval_utils._safe_mean(step_safe),
        "violation_free_episode_rate": track_eval_utils._safe_mean(ep_violation_free),
        "violation_episode_rate": 1.0 - track_eval_utils._safe_mean(ep_violation_free),
        "pos_error_norm_mean": track_eval_utils._safe_mean(np.asarray(arrays["pos_error_norm"], dtype=np.float64)),
        "vel_error_norm_mean": track_eval_utils._safe_mean(np.asarray(arrays["vel_error_norm"], dtype=np.float64)),
        "att_error_norm_mean": track_eval_utils._safe_mean(np.asarray(arrays["att_error_norm"], dtype=np.float64)),
        "omega_ref_error_norm_mean": track_eval_utils._safe_mean(
            np.asarray(arrays["omega_ref_error_norm"], dtype=np.float64)
        ),
        "pos_xz_rmse_mean": track_eval_utils._safe_mean(ep_pos_xz_rmse),
        "pos_xz_rmse_p95": track_eval_utils._safe_percentile(ep_pos_xz_rmse, 95.0),
        "vel_xz_rmse_mean": track_eval_utils._safe_mean(ep_vel_xz_rmse),
        "vel_xz_rmse_p95": track_eval_utils._safe_percentile(ep_vel_xz_rmse, 95.0),
        "pitch_rmse_deg_mean": track_eval_utils._safe_mean(ep_pitch_rmse),
        "pitch_rmse_deg_p95": track_eval_utils._safe_percentile(ep_pitch_rmse, 95.0),
        "y_rmse_mean": track_eval_utils._safe_mean(ep_y_rmse),
        "vy_rmse_mean": track_eval_utils._safe_mean(ep_vy_rmse),
        "roll_rmse_deg_mean": track_eval_utils._safe_mean(ep_roll_rmse),
        "yaw_rmse_deg_mean": track_eval_utils._safe_mean(ep_yaw_rmse),
        "p95_pos_xyz_mean": track_eval_utils._safe_mean(ep_p95_pos_xyz),
        "max_pos_xyz_mean": track_eval_utils._safe_mean(ep_max_pos_xyz),
        "p95_pos_xz_mean": track_eval_utils._safe_mean(ep_p95_pos_xz),
        "max_pos_xz_mean": track_eval_utils._safe_mean(ep_max_pos_xz),
        "x_rmse_mean": track_eval_utils._safe_mean(ep_x_rmse),
        "z_rmse_mean": track_eval_utils._safe_mean(ep_z_rmse),
        "vx_rmse_mean": track_eval_utils._safe_mean(np.asarray(arrays["episode_vx_rmse"], dtype=np.float64)),
        "vz_rmse_mean": track_eval_utils._safe_mean(np.asarray(arrays["episode_vz_rmse"], dtype=np.float64)),
        "pos_xyz_rmse_mean": track_eval_utils._safe_mean(ep_pos_xyz_rmse),
        "vel_xyz_rmse_mean": track_eval_utils._safe_mean(np.asarray(arrays["episode_vel_xyz_rmse"], dtype=np.float64)),
        "episode_length_mean": track_eval_utils._safe_mean(ep_length.astype(np.float64)),
        "episode_length_std": track_eval_utils._safe_std(ep_length.astype(np.float64)),
        "use_solver_rate": track_eval_utils._safe_mean(arrays["use_solver"]),
        "fallback_rate": track_eval_utils._safe_mean(arrays["fallback"]),
        "successful_qp_rate": track_eval_utils._safe_mean(arrays["successful_qp"]),
        "qp_residual_max_mean": track_eval_utils._safe_mean(arrays["qp_residual_max"]),
        "qp_residual_max_p95": track_eval_utils._safe_percentile(arrays["qp_residual_max"], 95.0),
        "best_episode_idx": int(best_episode_idx),
        "seed_group_stats": seed_group_stats,
    }

    summary = _add_public_summary_metric_blocks(summary, arrays)

    inference_stats = _benchmark_batched_inference_stats(
        batched_eval_action,
        actor_params,
        _select_inference_benchmark_obs(arrays),
        actor_obs_dim=int(actor_obs_dim),
        batch_size=int(rollout_batch_size),
    )
    summary["inference_freq"] = float(inference_stats["inference_freq"])

    aux = {
        "actor_obs_dim": int(actor_obs_dim),
        "projection_enabled": bool(sac_cfg.use_projection and sac_cfg.project_actor_actions),
        "num_qp_inequalities": int(projector.num_qp_inequalities),
        "num_backup_inequalities": int(projector.num_backup_inequalities),
        "backend": str(jax.default_backend()),
        "devices": [str(device) for device in jax.devices()],
        "rollout_batch_size": int(rollout_batch_size),
        "qp_success_residual_tol": float(qp_success_tol),
        "inference_freq": float(inference_stats["inference_freq"]),
        "inference_total_decisions": int(inference_stats["inference_total_decisions"]),
        "inference_total_time_sec": float(inference_stats["inference_total_time_sec"]),
    }
    return summary, arrays, aux


def _evaluate_and_save_task(task: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(task["run_dir"])
    checkpoint = str(task["checkpoint"])
    outputs_dir = Path(task["outputs_dir"])
    num_eval_seeds = int(task["num_eval_seeds"])
    episodes_per_seed = int(task["episodes_per_seed"])
    eval_seed_base = int(task["eval_seed_base"])
    seed_group_stride = int(task["seed_group_stride"])
    eval_label = str(task.get("eval_label", ""))
    tag = str(task["tag"])
    gif_slowdown = float(task.get("gif_slowdown", 0.25))
    gif_trail_length = int(task.get("gif_trail_length", 150))
    gif_print_every = int(task.get("gif_print_every", 25))
    rollout_batch_size = int(task.get("rollout_batch_size", DEFAULT_ROLLOUT_BATCH_SIZE))

    config_path = _resolve_config_path(run_dir)
    cfg_json = track_eval_utils._load_json(config_path)
    sac_cfg = track_eval_utils._dataclass_from_dict(SACConfig, cfg_json.get("sac", {}))
    env_cfg = track_eval_utils._dataclass_from_dict(QuadrotorEnvConfig, cfg_json.get("env", {}))
    cbf_cfg = track_eval_utils._dataclass_from_dict(
        QuadrotorBCBFConfig,
        cfg_json.get("cbf", {}),
    )

    weights_path = _resolve_checkpoint_weights_path(run_dir, checkpoint)
    actor_params = track_eval_utils._load_actor_params(weights_path)

    eval_root = run_dir / "evaluation"
    eval_root.mkdir(parents=True, exist_ok=True)
    eval_dir = eval_root / _build_eval_dir_name(
        tag=tag,
        checkpoint=checkpoint,
        num_eval_seeds=num_eval_seeds,
        episodes_per_seed=episodes_per_seed,
        label=eval_label,
    )
    eval_dir.mkdir(parents=True, exist_ok=True)

    summary, arrays, aux = _evaluate_checkpoint(
        actor_params=actor_params,
        sac_cfg=sac_cfg,
        env_cfg=env_cfg,
        cbf_cfg=cbf_cfg,
        num_eval_seeds=num_eval_seeds,
        episodes_per_seed=episodes_per_seed,
        eval_seed_base=eval_seed_base,
        seed_group_stride=seed_group_stride,
        rollout_batch_size=rollout_batch_size,
    )

    best_episode_idx = int(summary["best_episode_idx"])
    best_episode_artifacts = _save_best_episode_artifacts(
        arrays,
        best_episode_idx=best_episode_idx,
        checkpoint=checkpoint,
        out_dir=eval_dir,
        env_cfg=env_cfg,
        gif_slowdown=gif_slowdown,
        gif_trail_length=gif_trail_length,
        gif_print_every=gif_print_every,
        artifact_prefix="best",
        title_label="best evaluation (return)",
        score_key="episode_return",
        score_label="return",
    )

    results_npz = eval_dir / "evaluation_data.npz"
    metadata = {
        "outputs_dir": str(outputs_dir),
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "checkpoint": checkpoint,
        "weights_path": str(weights_path),
        "config_path": str(config_path),
        "eval_dir": str(eval_dir),
        "results_npz": str(results_npz),
        "eval_label": eval_label,
        "num_eval_seeds": int(num_eval_seeds),
        "episodes_per_seed": int(episodes_per_seed),
        "episodes_total": int(num_eval_seeds * episodes_per_seed),
        "eval_seed_base": int(eval_seed_base),
        "seed_group_stride": int(seed_group_stride),
        "selection_formula": _selection_formula_payload(),
        "saved_config": {
            "env_dt": float(env_cfg.dt),
            "env_max_steps": int(env_cfg.max_steps),
            "terminate_on_violation": bool(env_cfg.terminate_on_violation),
            "reward_mode": str(env_cfg.reward_mode),
            "reference_path": str(env_cfg.reference_path),
            "reference_dt": (None if env_cfg.reference_dt is None else float(env_cfg.reference_dt)),
            "z_max": float(env_cfg.z_max),
            "backup_policy_mode": str(cbf_cfg.backup_policy_mode),
            "cbf_dt": float(cbf_cfg.dt),
            "cbf_horizon_T": float(cbf_cfg.horizon),
            "cbf_num_steps": int(cbf_cfg.num_steps),
            "base_set_c": float(cbf_cfg.base_set_c),
            "terminal_alpha": float(cbf_cfg.base_alpha),
            "use_projection": bool(sac_cfg.use_projection),
            "project_actor_actions": bool(sac_cfg.project_actor_actions),
            "project_target_actions": bool(sac_cfg.project_target_actions),
        },
        "hyperparams": {
            "w_pos_xy": float(getattr(env_cfg, "w_pos_xy", np.nan)),
            "w_pos_z": float(getattr(env_cfg, "w_pos_z", np.nan)),
            "w_vel": float(getattr(env_cfg, "w_vel", np.nan)),
            "w_att": float(getattr(env_cfg, "w_att", np.nan)),
            "w_control_a": float(getattr(env_cfg, "w_control_a", np.nan)),
            "w_control_omega": float(getattr(env_cfg, "w_control_omega", np.nan)),
            "w_ref_omega_x": float(getattr(env_cfg, "w_ref_omega_x", np.nan)),
            "w_ref_omega_y": float(getattr(env_cfg, "w_ref_omega_y", np.nan)),
            "w_ref_omega_z": float(getattr(env_cfg, "w_ref_omega_z", np.nan)),
            "z_max": float(getattr(env_cfg, "z_max", np.nan)),
            "base_set_c": float(getattr(cbf_cfg, "base_set_c", np.nan)),
            "hc_alpha": float(getattr(cbf_cfg, "alpha", np.nan)),
            "term_alpha": float(getattr(cbf_cfg, "base_alpha", np.nan)),
        },
        "evaluation_runtime": {
            "actor_obs_dim": int(aux["actor_obs_dim"]),
            "projection_enabled": bool(aux["projection_enabled"]),
            "num_qp_inequalities": int(aux["num_qp_inequalities"]),
            "num_backup_inequalities": int(aux["num_backup_inequalities"]),
            "backend": aux["backend"],
            "devices": aux["devices"],
            "rollout_batch_size": int(aux["rollout_batch_size"]),
            "qp_success_residual_tol": float(aux["qp_success_residual_tol"]),
            "inference_freq": float(aux["inference_freq"]),
            "inference_total_decisions": int(aux["inference_total_decisions"]),
            "inference_total_time_sec": float(aux["inference_total_time_sec"]),
            "inference_measurement": "timed on evaluation observations with projection included, env stepping excluded, and JIT warmup excluded",
        },
        "best_episode_artifacts": best_episode_artifacts,
    }

    summary_payload = {
        "summary": summary,
        "metadata": metadata,
        "best_episode": {
            "episode_idx": int(best_episode_idx),
            "seed_group_idx": int(np.asarray(arrays["episode_seed_group"], dtype=np.int32)[best_episode_idx]),
            "episode_in_seed_group": int(
                np.asarray(arrays["episode_in_seed_group"], dtype=np.int32)[best_episode_idx]
            ),
            "episode_seed": int(np.asarray(arrays["episode_seed"], dtype=np.int64)[best_episode_idx]),
            "violation_free": float(np.asarray(arrays["episode_violation_free"], dtype=np.float64)[best_episode_idx]),
            "safe_rate": float(np.asarray(arrays["episode_safe_rate"], dtype=np.float64)[best_episode_idx]),
            "tracking_score": float(np.asarray(arrays["episode_tracking_score"], dtype=np.float64)[best_episode_idx]),
            "pos_xz_rmse": float(np.asarray(arrays["episode_pos_xz_rmse"], dtype=np.float64)[best_episode_idx]),
            "vel_xz_rmse": float(np.asarray(arrays["episode_vel_xz_rmse"], dtype=np.float64)[best_episode_idx]),
            "pitch_rmse_deg": float(
                np.asarray(arrays["episode_pitch_rmse_deg"], dtype=np.float64)[best_episode_idx]
            ),
            "artifacts": best_episode_artifacts,
        },
    }

    np.savez(
        results_npz,
        **arrays,
        summary_json=np.asarray(json.dumps(summary_payload["summary"], sort_keys=True), dtype=np.str_),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    track_eval_utils._save_json(eval_dir / "summary.json", summary_payload)

    return {
        "status": "ok",
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "eval_dir": str(eval_dir),
        "results_npz": str(results_npz),
        "summary": summary,
        "best_episode": summary_payload["best_episode"],
        "hyperparams": metadata["hyperparams"],
    }


def _run_parallel_evaluation(tasks: list[dict[str, Any]], workers: int) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    if not tasks:
        return results, skipped

    if workers <= 1 or len(tasks) <= 1:
        for idx, task in enumerate(tasks, start=1):
            try:
                results.append(_evaluate_and_save_task(task))
            except Exception as exc:  # noqa: BLE001
                skipped.append(
                    {
                        "run_name": Path(task["run_dir"]).name,
                        "checkpoint": str(task["checkpoint"]),
                        "reason": str(exc),
                    }
                )
            if idx % 10 == 0 or idx == len(tasks):
                print(f"Evaluated {idx}/{len(tasks)} tasks...")
        return results, skipped

    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        fut_to_task = {pool.submit(_evaluate_and_save_task, task): task for task in tasks}
        done_count = 0
        for fut in cf.as_completed(fut_to_task):
            task = fut_to_task[fut]
            done_count += 1
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                skipped.append(
                    {
                        "run_name": Path(task["run_dir"]).name,
                        "checkpoint": str(task["checkpoint"]),
                        "reason": str(exc),
                    }
                )
            if done_count % 10 == 0 or done_count == len(tasks):
                print(f"Evaluated {done_count}/{len(tasks)} tasks...")
    return results, skipped


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved safe quadrotor SAC policies with the full HardNet-CVX "
            "backup-CBF-QP projection stack and rank them safety-first."
        )
    )
    parser.add_argument("--outputs_dir", type=str, required=True, help="Outputs root containing run directories.")
    parser.add_argument("--run_glob", type=str, default=DEFAULT_RUN_GLOB, help="Glob for run directory names.")
    parser.add_argument("--experiment", type=str, default="", help="Optional single run directory name/path.")
    parser.add_argument("--num_eval_seeds", type=int, default=10, help="Number of seed groups to evaluate.")
    parser.add_argument(
        "--episodes_per_seed",
        type=int,
        default=10,
        help="Episodes to evaluate within each seed group.",
    )
    parser.add_argument(
        "--eval_seed_base",
        type=int,
        default=3_500_000,
        help="Base seed used to build the shared evaluation seed bank.",
    )
    parser.add_argument(
        "--seed_group_stride",
        type=int,
        default=DEFAULT_SEED_GROUP_STRIDE,
        help="Stride between seed-group bases in the shared evaluation seed bank.",
    )
    parser.add_argument(
        "--weight_preference",
        type=str,
        default="both",
        choices=["both", "best_only", "final_only"],
        help=(
            "Which checkpoints to evaluate for each run. "
            "'best_only' targets best_weights.pkl (the return-selected best, matching the unicycle)."
        ),
    )
    parser.add_argument(
        "--parallel_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for the evaluation stage.",
    )
    parser.add_argument(
        "--rollout_batch_size",
        type=int,
        default=DEFAULT_ROLLOUT_BATCH_SIZE,
        help="Maximum number of episodes evaluated in one batched JAX rollout.",
    )
    parser.add_argument(
        "--eval_label",
        type=str,
        default="",
        help="Optional label embedded in saved evaluation directory names and metadata.",
    )
    parser.add_argument(
        "--gif_slowdown",
        type=float,
        default=0.25,
        help="Slowdown factor used for best-episode GIF animations.",
    )
    parser.add_argument(
        "--gif_trail_length",
        type=int,
        default=150,
        help="Trail length used for best-episode GIF animations.",
    )
    parser.add_argument(
        "--gif_print_every",
        type=int,
        default=25,
        help="Print progress every N frames when building GIF animations.",
    )
    args = parser.parse_args(argv)

    if args.num_eval_seeds <= 0:
        raise ValueError(f"--num_eval_seeds must be positive, got {args.num_eval_seeds}")
    if args.episodes_per_seed <= 0:
        raise ValueError(f"--episodes_per_seed must be positive, got {args.episodes_per_seed}")
    if args.seed_group_stride <= 0:
        raise ValueError(f"--seed_group_stride must be positive, got {args.seed_group_stride}")
    if args.parallel_workers <= 0:
        raise ValueError(f"--parallel_workers must be positive, got {args.parallel_workers}")
    if args.rollout_batch_size <= 0:
        raise ValueError(f"--rollout_batch_size must be positive, got {args.rollout_batch_size}")
    if args.gif_slowdown <= 0.0:
        raise ValueError(f"--gif_slowdown must be positive, got {args.gif_slowdown}")
    if args.gif_trail_length <= 0:
        raise ValueError(f"--gif_trail_length must be positive, got {args.gif_trail_length}")
    if args.gif_print_every <= 0:
        raise ValueError(f"--gif_print_every must be positive, got {args.gif_print_every}")

    outputs_dir = track_eval_utils._resolve_outputs_dir(args.outputs_dir)
    run_dirs = _iter_run_dirs(outputs_dir, args.run_glob, experiment=args.experiment)
    if not run_dirs:
        raise RuntimeError(f"No run directories found under {outputs_dir} matching '{args.run_glob}'")

    checkpoints: set[str]
    if args.weight_preference == "both":
        checkpoints = {"best", "final"}
    elif args.weight_preference == "best_only":
        checkpoints = {"best"}
    else:
        checkpoints = {"final"}

    eval_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    tasks: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        for checkpoint in sorted(checkpoints):
            tasks.append(
                {
                    "outputs_dir": str(outputs_dir),
                    "run_dir": str(run_dir),
                    "checkpoint": checkpoint,
                    "num_eval_seeds": int(args.num_eval_seeds),
                    "episodes_per_seed": int(args.episodes_per_seed),
                    "eval_seed_base": int(args.eval_seed_base),
                    "seed_group_stride": int(args.seed_group_stride),
                    "eval_label": str(args.eval_label),
                    "gif_slowdown": float(args.gif_slowdown),
                    "gif_trail_length": int(args.gif_trail_length),
                    "gif_print_every": int(args.gif_print_every),
                    "rollout_batch_size": int(args.rollout_batch_size),
                    "tag": eval_tag,
                }
            )

    total_episode_evals = int(args.num_eval_seeds) * int(args.episodes_per_seed) * len(tasks)
    print(
        f"Starting quad SAC evaluation for {len(tasks)} tasks "
        f"({len(run_dirs)} runs x {len(checkpoints)} checkpoint(s)). "
        f"Each task uses {args.num_eval_seeds} seed groups x {args.episodes_per_seed} episodes "
        f"= {args.num_eval_seeds * args.episodes_per_seed} episodes. "
        f"Total episode evaluations scheduled: {total_episode_evals}. "
        f"workers={args.parallel_workers}, rollout_batch_size={args.rollout_batch_size}."
    )

    eval_results, eval_skipped = _run_parallel_evaluation(tasks, workers=int(args.parallel_workers))
    print("Evaluation artifacts saved under each run's 'evaluation/' folder.")
    print(f"Successful eval tasks: {len(eval_results)} / {len(tasks)}")
    if eval_skipped:
        print(f"Failed eval tasks: {len(eval_skipped)}")
        print(json.dumps(eval_skipped, indent=2))


if __name__ == "__main__":
    main()
