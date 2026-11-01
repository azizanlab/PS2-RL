#!/usr/bin/env python
"""Parallel trajectory-tracking policy evaluation.

Evaluates runs/checkpoints and saves results.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
from dataclasses import fields
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import re
import time
from typing import Any

# Normalize ambiguous/incorrect GPU backend aliases before importing JAX.
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
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    _HAS_MATPLOTLIB = False

from ps2rl.utils.paths import PROJECT_ROOT

from ps2rl.cil.unicycle_backup_cbf import (
    UnicycleBCBFConfig,
    constraint_residuals,
    get_cached_runtime,
)
from ps2rl.backup_policy.unicycle_learned_backup import load_learned_unicycle_backup_policy
from ps2rl.envs.unicycle_env import UnicycleEnvConfig, build_unicycle_env
from ps2rl.cil.cil_policy import ActorConfig, actor_mean_action
from ps2rl.phase2_ps2.unicycle_ps2_trainer import SACConfig, _build_action_fns, _validate_action_scale


def _sanitize_token(x: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(x))
    token = token.strip("_.-")
    return token or "na"


def _dt_token(x: float) -> str:
    return _sanitize_token(f"{float(x):.6g}".replace(".", "p"))


def _resolve_outputs_dir(raw: str) -> Path:
    direct = Path(raw)
    if direct.exists():
        return direct
    candidate = PROJECT_ROOT / raw
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Outputs directory not found: {raw} (checked {direct} and {candidate})")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dataclass_from_dict(cls, payload: dict[str, Any], overrides: dict[str, Any] | None = None):
    valid = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in payload.items() if k in valid}
    if overrides:
        kwargs.update(overrides)
    return cls(**kwargs)


def _coerce_finite_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _infer_saved_v_ref(env_cfg: UnicycleEnvConfig) -> float:
    reward_v = _coerce_finite_float(getattr(env_cfg, "reward_v_des", np.nan), np.nan)
    if np.isfinite(reward_v):
        return reward_v
    traj_v = _coerce_finite_float(getattr(env_cfg, "traj_v_mean", np.nan), np.nan)
    if np.isfinite(traj_v):
        return traj_v
    return _coerce_finite_float(getattr(env_cfg, "v_des", 0.0), 0.0)


def _iter_run_dirs(outputs_dir: Path, run_glob: str, experiment: str = "") -> list[Path]:
    if experiment.strip():
        exp = experiment.strip()
        p = Path(exp)
        if p.exists() and p.is_dir():
            run_dir = p
        else:
            run_dir = outputs_dir / exp
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {exp}")
        if not (run_dir / "configs.json").exists():
            raise FileNotFoundError(f"Missing configs.json under run directory: {run_dir}")
        return [run_dir]

    run_dirs = []
    for run_dir in sorted(outputs_dir.glob(run_glob)):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "configs.json").exists():
            continue
        run_dirs.append(run_dir)
    return run_dirs


def _load_actor_params(weights_path: Path):
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")
    with open(weights_path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict):
        if "actor_params" in payload:
            return payload["actor_params"]
        raise KeyError(
            f"'actor_params' not found in {weights_path}. "
            f"Available keys: {sorted(payload.keys())}"
        )
    raise TypeError(f"Unexpected weights format in {weights_path}: {type(payload)}")


def _infer_actor_obs_dim(actor_params: Any) -> int | None:
    if isinstance(actor_params, dict):
        layers = actor_params.get("layers")
        if isinstance(layers, (list, tuple)) and len(layers) > 0:
            first = layers[0]
            if isinstance(first, dict) and "w" in first:
                w = np.asarray(first["w"])
                if w.ndim == 2:
                    return int(w.shape[0])
    return None


def _adapt_obs(obs: np.ndarray, expected_dim: int) -> np.ndarray:
    obs_flat = np.asarray(obs, dtype=np.float64).reshape(-1)
    if expected_dim <= obs_flat.shape[0]:
        return obs_flat[:expected_dim]
    raise ValueError(
        f"Actor expects obs_dim={expected_dim}, but env observation has shape {obs_flat.shape}. "
        "Cannot pad safely."
    )


def _wrap_angle(x: float | np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def _build_eval_env_cfg(
    env_cfg: UnicycleEnvConfig,
    *,
    env_dt: float,
    max_steps: int,
    eval_v_ref: float | None,
) -> UnicycleEnvConfig:
    env_overrides: dict[str, Any] = {
        "dt": float(env_dt),
        "max_steps": int(max_steps),
    }
    if eval_v_ref is not None:
        env_overrides["reward_v_des"] = float(eval_v_ref)
        env_overrides["traj_v_mean"] = float(eval_v_ref)
    return _dataclass_from_dict(UnicycleEnvConfig, vars(env_cfg), overrides=env_overrides)


def _finite_argmin(values: np.ndarray) -> int:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    idx = np.arange(vals.size, dtype=np.int64)
    vals = np.where(np.isfinite(vals), vals, np.inf)
    order = np.lexsort((idx, vals))
    return int(order[0])


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
    episode_y_rmse = _reduce_step_metric_by_episode(
        episode_idx,
        np.asarray(arrays.get("step_y_error_abs", []), dtype=np.float64),
        num_episodes,
        lambda x: np.sqrt(np.mean(np.square(x))),
        default=0.0,
    )
    episode_psi_rmse = _reduce_step_metric_by_episode(
        episode_idx,
        np.asarray(arrays.get("step_psi_error_abs", []), dtype=np.float64),
        num_episodes,
        lambda x: np.sqrt(np.mean(np.square(x))),
        default=0.0,
    )
    episode_v_rmse = _reduce_step_metric_by_episode(
        episode_idx,
        np.asarray(arrays.get("step_speed_error_abs", []), dtype=np.float64),
        num_episodes,
        lambda x: np.sqrt(np.mean(np.square(x))),
        default=0.0,
    )
    episode_worst_violation_y = _reduce_step_metric_by_episode(
        episode_idx,
        np.asarray(arrays.get("step_violation_y", []), dtype=np.float64),
        num_episodes,
        np.max,
        default=0.0,
    )
    episode_worst_violation_psi = _reduce_step_metric_by_episode(
        episode_idx,
        np.asarray(arrays.get("step_violation_psi", []), dtype=np.float64),
        num_episodes,
        np.max,
        default=0.0,
    )
    episode_cumulative_violation_y = _reduce_step_metric_by_episode(
        episode_idx,
        np.asarray(arrays.get("step_violation_y", []), dtype=np.float64),
        num_episodes,
        np.sum,
        default=0.0,
    )
    episode_cumulative_violation_psi = _reduce_step_metric_by_episode(
        episode_idx,
        np.asarray(arrays.get("step_violation_psi", []), dtype=np.float64),
        num_episodes,
        np.sum,
        default=0.0,
    )
    episode_violation_free = (
        (episode_worst_violation_y <= 0.0) & (episode_worst_violation_psi <= 0.0)
    ).astype(np.float64)

    arrays["episode_y_rmse"] = episode_y_rmse
    arrays["episode_psi_rmse"] = episode_psi_rmse
    arrays["episode_v_rmse"] = episode_v_rmse
    arrays["episode_worst_violation_y"] = episode_worst_violation_y
    arrays["episode_worst_violation_psi"] = episode_worst_violation_psi
    arrays["episode_cumulative_violation_y"] = episode_cumulative_violation_y
    arrays["episode_cumulative_violation_psi"] = episode_cumulative_violation_psi
    arrays["episode_violation_free"] = episode_violation_free

    episode_safety_rate = _safe_mean_1d(episode_violation_free)
    summary["violation_free_episode_rate"] = episode_safety_rate
    summary["safe_episode_rate"] = episode_safety_rate
    summary["performance_metrics"] = {
        "y_rmse": _safe_mean_1d(episode_y_rmse),
        "psi_rmse": _safe_mean_1d(episode_psi_rmse),
        "v_rmse": _safe_mean_1d(episode_v_rmse),
    }
    summary["safety_metrics"] = {
        "episode_safety_rate": episode_safety_rate,
        **_safe_distribution_fields("worst_violation_y", episode_worst_violation_y),
        **_safe_distribution_fields("worst_violation_psi", episode_worst_violation_psi),
        **_safe_distribution_fields("cumulative_violation_y", episode_cumulative_violation_y),
        **_safe_distribution_fields("cumulative_violation_psi", episode_cumulative_violation_psi),
    }
    return summary


def _episode_trace(arrays: dict[str, np.ndarray], ep_idx: int) -> dict[str, np.ndarray]:
    ep_all = np.asarray(arrays.get("episode_idx", []), dtype=np.int32).reshape(-1)
    mask = ep_all == int(ep_idx)
    if not np.any(mask):
        return {}

    step_all = np.asarray(arrays.get("step_in_episode", np.arange(ep_all.size)), dtype=np.int32).reshape(-1)
    order = np.argsort(step_all[mask])

    def pick(name: str, default: float = np.nan) -> np.ndarray:
        if name not in arrays:
            return np.full((int(np.sum(mask)),), float(default), dtype=np.float64)
        arr = np.asarray(arrays[name], dtype=np.float64).reshape(-1)
        return arr[mask][order]

    return {
        "t": pick("time_sec", default=0.0),
        "y": pick("y"),
        "v": pick("v"),
        "psi": pick("psi"),
        "y_ref": pick("y_ref"),
        "v_ref": pick("v_ref"),
        "psi_ref": pick("psi_ref"),
        "y_err": pick("step_y_error_abs"),
        "v_err": pick("step_speed_error_abs"),
        "psi_err": pick("step_psi_error_abs"),
        "safe": pick("step_safe", default=1.0),
        "reward": pick("step_reward", default=0.0),
    }


def _plot_episode_diagnostics(
    trace: dict[str, np.ndarray],
    *,
    out_path: Path,
    title: str,
    y_max: float,
    psi_max: float,
) -> None:
    if (not trace) or (not _HAS_MATPLOTLIB):
        return

    t = np.asarray(trace["t"], dtype=np.float64)
    y = np.asarray(trace["y"], dtype=np.float64)
    v = np.asarray(trace["v"], dtype=np.float64)
    psi = np.asarray(trace["psi"], dtype=np.float64)
    y_ref = np.asarray(trace["y_ref"], dtype=np.float64)
    v_ref = np.asarray(trace["v_ref"], dtype=np.float64)
    psi_ref = np.asarray(trace["psi_ref"], dtype=np.float64)
    y_err = np.asarray(trace["y_err"], dtype=np.float64)
    v_err = np.asarray(trace["v_err"], dtype=np.float64)
    psi_err = np.asarray(trace["psi_err"], dtype=np.float64)
    safe = np.asarray(trace["safe"], dtype=np.float64)
    reward = np.asarray(trace["reward"], dtype=np.float64)

    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True)

    axes[0].plot(t, y, label="y", linewidth=1.2)
    axes[0].plot(t, y_ref, label="y_ref", linewidth=1.0, linestyle="--")
    axes[0].axhline(float(y_max), color="r", linestyle="--", linewidth=1.0)
    axes[0].axhline(-float(y_max), color="r", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Y")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(t, psi, label="psi", linewidth=1.2)
    axes[1].plot(t, psi_ref, label="psi_ref", linewidth=1.0, linestyle="--")
    axes[1].axhline(float(psi_max), color="r", linestyle="--", linewidth=1.0)
    axes[1].axhline(-float(psi_max), color="r", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("psi")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(t, v, label="v", linewidth=1.2)
    axes[2].plot(t, v_ref, label="v_ref", linewidth=1.0, linestyle="--")
    axes[2].set_ylabel("v")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    axes[3].plot(t, y_err, label="|y-y_ref|", linewidth=1.1)
    axes[3].plot(t, psi_err, label="|psi-psi_ref|", linewidth=1.1)
    axes[3].plot(t, v_err, label="|v-v_ref|", linewidth=1.1)
    axes[3].set_ylabel("Tracking Err")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")

    l1 = axes[4].plot(t, safe, color="tab:blue", linewidth=1.2, label="safe")
    axes[4].set_ylim(-0.05, 1.05)
    axes[4].set_ylabel("Safe")
    ax4b = axes[4].twinx()
    l2 = ax4b.plot(t, reward, color="tab:orange", linewidth=1.1, label="reward")
    ax4b.set_ylabel("Reward")
    axes[4].set_xlabel("Time (s)")
    axes[4].grid(True, alpha=0.3)
    lines = l1 + l2
    axes[4].legend(lines, [ln.get_label() for ln in lines], loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _make_representative_episode_plots(
    arrays: dict[str, np.ndarray],
    *,
    out_prefix: Path,
    env_cfg_eval: UnicycleEnvConfig,
) -> dict[str, dict[str, Any]]:
    if not _HAS_MATPLOTLIB:
        return {
            "warning": {
                "message": "matplotlib is not available; diagnostic episode plots were skipped."
            }
        }
    ep_y = np.asarray(arrays.get("episode_y_error_abs_mean", []), dtype=np.float64).reshape(-1)
    ep_psi = np.asarray(arrays.get("episode_psi_error_abs_mean", []), dtype=np.float64).reshape(-1)
    ep_v = np.asarray(arrays.get("episode_speed_error_abs_mean", []), dtype=np.float64).reshape(-1)
    n = int(min(len(ep_y), len(ep_psi), len(ep_v)))
    if n <= 0:
        return {}

    ep_y = ep_y[:n]
    ep_psi = ep_psi[:n]
    ep_v = ep_v[:n]

    picks = [
        ("best_y_tracking", "Best Y-Tracking", _finite_argmin(ep_y)),
    ]

    manifest: dict[str, dict[str, Any]] = {}
    for key, label, ep_idx in picks:
        trace = _episode_trace(arrays, int(ep_idx))
        if not trace:
            continue
        out_path = out_prefix.parent / f"{out_prefix.name}-diagnostic_{key}-ep{int(ep_idx):03d}.png"
        title = (
            f"{label} | ep={int(ep_idx)} | "
            f"y_err={float(ep_y[ep_idx]):.4f}, psi_err={float(ep_psi[ep_idx]):.4f}, "
            f"v_err={float(ep_v[ep_idx]):.4f}"
        )
        _plot_episode_diagnostics(
            trace,
            out_path=out_path,
            title=title,
            y_max=float(env_cfg_eval.y_max),
            psi_max=float(env_cfg_eval.psi_max),
        )
        manifest[key] = {
            "episode_idx": int(ep_idx),
            "plot_path": str(out_path),
            "y_error_abs_mean": float(ep_y[ep_idx]),
            "psi_error_abs_mean": float(ep_psi[ep_idx]),
            "speed_error_abs_mean": float(ep_v[ep_idx]),
        }
    return manifest


def _evaluate_checkpoint(
    actor_params: Any,
    sac_cfg: SACConfig,
    env_eval_cfg: UnicycleEnvConfig,
    cbf_cfg: UnicycleBCBFConfig,
    episodes: int,
    eval_seed_base: int,
    rollout_batch_size: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    env_fns = build_unicycle_env(env_eval_cfg)
    action_scale_np = np.asarray(jax.device_get(env_fns.action_scale), dtype=np.float64)
    action_scale = jnp.asarray(action_scale_np, dtype=jnp.float32)
    expected_action_scale = np.array([cbf_cfg.a_max, cbf_cfg.r_max], dtype=np.float64)
    _validate_action_scale("jax_env.action_scale", action_scale_np, expected_action_scale)
    if cbf_cfg.backup_policy_mode.strip().lower() == "learned":
        learned = load_learned_unicycle_backup_policy(cbf_cfg.learned_backup_policy_path)
        _validate_action_scale("learned_backup_policy.action_scale", learned.action_scale, expected_action_scale)

    actor_obs_dim = _infer_actor_obs_dim(actor_params)
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

    projection_active = bool(sac_cfg.use_projection and sac_cfg.project_actor_actions)
    if projection_active:
        backup_runtime = get_cached_runtime(cbf_cfg)
        batched_residuals = jax.jit(
            jax.vmap(
                lambda x, u: constraint_residuals(x, u, cbf_cfg, runtime=backup_runtime),
                in_axes=(0, 0),
            )
        )
        _, eval_action_single = _build_action_fns(
            sac_cfg,
            actor_cfg,
            cbf_cfg,
            action_scale,
            backup_runtime=backup_runtime,
            return_solver_info=True,
        )
    else:
        # Projection-disabled runs should evaluate as plain SAC actors and must not
        # depend on reconstructing the backup-CBF runtime from legacy configs.
        @jax.jit
        def eval_action_single(params, obs: jax.Array):
            raw = actor_mean_action(params, obs[None, :], action_scale, actor_cfg)
            safe = jnp.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            safe = jnp.clip(safe, -action_scale, action_scale)
            # Empty solver-info dict keeps the tuple arity of the projection-enabled action fn.
            return safe[0], raw[0], jnp.asarray(False, dtype=jnp.bool_), {}

    batched_eval_action = jax.jit(jax.vmap(eval_action_single, in_axes=(None, 0)))
    qp_success_tol = max(float(cbf_cfg.solver_tol) * 10.0, 1e-6)

    warm_seed = jnp.asarray([eval_seed_base + 999_999], dtype=jnp.uint32)
    warm_keys = jax.vmap(jax.random.PRNGKey)(warm_seed)
    _, warm_obs = env_fns.reset_batched(warm_keys)
    warm_obs_actor = warm_obs[:, : actor_obs_dim]
    _block_until_ready_tree(batched_eval_action(actor_params, warm_obs_actor))

    max_steps = int(env_eval_cfg.max_steps)
    inference_total_time_sec = 0.0
    inference_total_decisions = 0.0
    episode_acc: dict[str, list[np.ndarray]] = {}
    step_acc: dict[str, list[np.ndarray]] = {}

    def append_episode(name: str, value: np.ndarray) -> None:
        episode_acc.setdefault(name, []).append(np.asarray(value))

    def append_step(name: str, value: np.ndarray) -> None:
        step_acc.setdefault(name, []).append(np.asarray(value))

    def _broadcast_mask(mask: jax.Array, ndim: int) -> jax.Array:
        out = mask
        while out.ndim < ndim:
            out = out[..., None]
        return out

    def _mask_like(arr: jax.Array, active: jax.Array, fill_value: float = 0.0) -> jax.Array:
        mask = _broadcast_mask(active, arr.ndim)
        fill = jnp.full_like(arr, fill_value)
        return jnp.where(mask, arr, fill)

    def masked_mean(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        masked = np.where(mask, arr, np.nan)
        with np.errstate(invalid="ignore"):
            out = np.nanmean(masked, axis=0)
        return np.nan_to_num(out, nan=0.0)

    def masked_value_at_end(arr: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        idx = np.maximum(lengths - 1, 0).astype(np.int64)
        return arr[idx, np.arange(arr.shape[1])]

    @jax.jit
    def rollout_chunk(actor_params_in: Any, reset_keys: jax.Array):
        batch_size = reset_keys.shape[0]
        env_state, obs = env_fns.reset_batched(reset_keys)
        done = jnp.zeros((batch_size,), dtype=jnp.bool_)
        zero_action = jnp.zeros((batch_size, env_fns.action_dim), dtype=jnp.float32)

        def scan_step(carry, step_idx):
            env_state_i, obs_i, done_i = carry
            active = jnp.logical_not(done_i)
            obs_actor = obs_i[:, : actor_obs_dim]
            safe_action, raw_action, use_solver, finite_info = batched_eval_action(actor_params_in, obs_actor)
            if projection_active:
                residuals = batched_residuals(obs_i[:, :3], safe_action)
                residuals_finite = jnp.all(jnp.isfinite(residuals), axis=-1)
                qp_residual_max = jnp.max(
                    jnp.maximum(jnp.nan_to_num(residuals, nan=jnp.inf, posinf=jnp.inf, neginf=0.0), 0.0),
                    axis=-1,
                )
                successful_qp = (
                    active
                    & use_solver
                    & finite_info["inputs_finite"]
                    & finite_info["z_finite"]
                    & residuals_finite
                    & (qp_residual_max <= jnp.asarray(qp_success_tol, dtype=jnp.float32))
                )
                fallback = active & ~use_solver
            else:
                successful_qp = jnp.zeros((batch_size,), dtype=jnp.bool_)
                fallback = jnp.zeros((batch_size,), dtype=jnp.bool_)
                qp_residual_max = jnp.zeros((batch_size,), dtype=jnp.float32)
            action_for_env = jnp.where(active[:, None], safe_action, zero_action)
            step_keys = jax.vmap(lambda k: jax.random.fold_in(k, step_idx + jnp.int32(1)))(reset_keys)
            env_state_raw, next_obs_true_i, next_obs_out_i, rew_i, done_raw_i, info_i = env_fns.step_batched(
                env_state_i,
                action_for_env,
                step_keys,
            )
            done_step = active & done_raw_i
            done_out = done_i | done_step

            record = {
                "valid_mask": active,
                "episode_done": done_step,
                "obs": _mask_like(obs_i, active),
                "next_obs": _mask_like(next_obs_true_i, active),
                "rew": _mask_like(rew_i.astype(jnp.float32), active),
                "safe": _mask_like(info_i.is_safe.astype(jnp.float32), active),
                "violation_y": _mask_like(info_i.violation_y.astype(jnp.float32), active),
                "violation_psi": _mask_like(info_i.violation_psi.astype(jnp.float32), active),
                "speed_error_abs": _mask_like(info_i.speed_error_abs.astype(jnp.float32), active),
                "y_error_abs": _mask_like(info_i.y_error_abs.astype(jnp.float32), active),
                "psi_error_abs": _mask_like(info_i.psi_error_abs.astype(jnp.float32), active),
                "y_ref": _mask_like(info_i.y_ref.astype(jnp.float32), active),
                "v_ref": _mask_like(info_i.v_ref.astype(jnp.float32), active),
                "psi_ref": _mask_like(info_i.psi_ref.astype(jnp.float32), active),
                "use_solver": _mask_like(use_solver.astype(jnp.float32), active),
                "fallback": _mask_like(fallback.astype(jnp.float32), active),
                "successful_qp": _mask_like(successful_qp.astype(jnp.float32), active),
                "qp_residual_max": _mask_like(qp_residual_max.astype(jnp.float32), active),
            }

            env_state_out = jax.tree_util.tree_map(
                lambda new, old: jnp.where(_broadcast_mask(active, new.ndim), new, old),
                env_state_raw,
                env_state_i,
            )
            obs_out = jnp.where(active[:, None], next_obs_out_i, obs_i)
            return (env_state_out, obs_out, done_out), record

        (_, _, _), records = jax.lax.scan(
            scan_step,
            (env_state, obs, done),
            xs=jnp.arange(max_steps, dtype=jnp.int32),
        )
        return records

    total_episodes = int(episodes)
    for start in range(0, total_episodes, int(max(1, rollout_batch_size))):
        batch_size = min(int(max(1, rollout_batch_size)), total_episodes - start)
        seeds = np.arange(start, start + batch_size, dtype=np.int64) + int(eval_seed_base)
        reset_keys = jnp.asarray(np.stack([np.asarray(jax.random.PRNGKey(int(seed))) for seed in seeds], axis=0))
        rollout = jax.device_get(rollout_chunk(actor_params, reset_keys))

        valid = np.asarray(rollout["valid_mask"], dtype=bool)
        next_obs = np.asarray(rollout["next_obs"], dtype=np.float64)
        rew = np.asarray(rollout["rew"], dtype=np.float64)
        safe = np.asarray(rollout["safe"], dtype=np.float64)
        violation_y = np.asarray(rollout["violation_y"], dtype=np.float64)
        violation_psi = np.asarray(rollout["violation_psi"], dtype=np.float64)
        speed_error_abs = np.asarray(rollout["speed_error_abs"], dtype=np.float64)
        y_error_abs = np.asarray(rollout["y_error_abs"], dtype=np.float64)
        psi_error_abs = np.asarray(rollout["psi_error_abs"], dtype=np.float64)
        y_ref = np.asarray(rollout["y_ref"], dtype=np.float64)
        v_ref = np.asarray(rollout["v_ref"], dtype=np.float64)
        psi_ref = np.asarray(rollout["psi_ref"], dtype=np.float64)
        use_solver = np.asarray(rollout["use_solver"], dtype=np.float64)
        fallback = np.asarray(rollout["fallback"], dtype=np.float64)
        successful_qp = np.asarray(rollout["successful_qp"], dtype=np.float64)
        qp_residual_max = np.asarray(rollout["qp_residual_max"], dtype=np.float64)

        valid_ep = valid.T
        flat_mask = valid_ep.reshape(-1)
        obs_roll = np.asarray(rollout["obs"], dtype=np.float32)
        obs_bench = np.swapaxes(obs_roll, 0, 1).reshape((-1, obs_roll.shape[-1]))[flat_mask]
        inference_stats = _benchmark_batched_inference_stats(
            batched_eval_action,
            actor_params,
            obs_bench,
            actor_obs_dim=int(actor_obs_dim),
            batch_size=int(batch_size),
        )
        inference_total_time_sec += float(inference_stats["inference_total_time_sec"])
        inference_total_decisions += float(inference_stats["inference_total_decisions"])

        episode_length = np.sum(valid, axis=0).astype(np.int32)
        episode_return = np.sum(rew * valid.astype(np.float64), axis=0)
        episode_speed_error_abs_mean = masked_mean(speed_error_abs, valid)
        episode_y_error_abs_mean = masked_mean(y_error_abs, valid)
        episode_psi_error_abs_mean = masked_mean(psi_error_abs, valid)
        episode_safe_rate = masked_mean(safe, valid)
        episode_y_violation_step_rate = masked_mean((violation_y > 0.0).astype(np.float64), valid)
        episode_psi_violation_step_rate = masked_mean((violation_psi > 0.0).astype(np.float64), valid)
        last_safe = masked_value_at_end(safe, episode_length)
        episode_violation_terminated = (
            bool(env_eval_cfg.terminate_on_violation)
            & (episode_length < int(max_steps))
            & (last_safe < 0.5)
        ).astype(np.float64)

        global_episode_idx = np.arange(start, start + batch_size, dtype=np.int32)
        append_episode("episode_return", episode_return)
        append_episode("episode_length", episode_length)
        append_episode("episode_speed_error_abs_mean", episode_speed_error_abs_mean)
        append_episode("episode_y_error_abs_mean", episode_y_error_abs_mean)
        append_episode("episode_psi_error_abs_mean", episode_psi_error_abs_mean)
        append_episode("episode_safe_rate", episode_safe_rate)
        append_episode("episode_use_solver_rate", masked_mean(use_solver, valid))
        append_episode("episode_fallback_rate", masked_mean(fallback, valid))
        append_episode("episode_successful_qp_rate", masked_mean(successful_qp, valid))
        append_episode("episode_y_violation_step_rate", episode_y_violation_step_rate)
        append_episode("episode_psi_violation_step_rate", episode_psi_violation_step_rate)
        append_episode("episode_violation_terminated", episode_violation_terminated)

        ep_idx_grid = np.broadcast_to(global_episode_idx[:, None], (batch_size, max_steps))
        step_grid = np.broadcast_to(np.arange(max_steps, dtype=np.int32)[None, :], (batch_size, max_steps))

        def flatten_step(arr: np.ndarray) -> np.ndarray:
            arr_ep = np.swapaxes(arr, 0, 1)
            return arr_ep.reshape((-1,) + arr_ep.shape[2:])[flat_mask]

        append_step("episode_idx", ep_idx_grid.reshape(-1)[flat_mask].astype(np.int32))
        append_step("step_in_episode", step_grid.reshape(-1)[flat_mask].astype(np.int32))
        append_step("time_sec", (step_grid.reshape(-1)[flat_mask].astype(np.float64) * float(env_eval_cfg.dt)))
        append_step("y", flatten_step(next_obs[..., 0]))
        append_step("v", flatten_step(next_obs[..., 1]))
        append_step("psi", flatten_step(next_obs[..., 2]))
        append_step("y_ref", flatten_step(y_ref))
        append_step("v_ref", flatten_step(v_ref))
        append_step("psi_ref", flatten_step(psi_ref))
        append_step("step_y_error_abs", flatten_step(y_error_abs))
        append_step("step_speed_error_abs", flatten_step(speed_error_abs))
        append_step("step_psi_error_abs", flatten_step(psi_error_abs))
        append_step("step_violation_y", flatten_step(violation_y))
        append_step("step_violation_psi", flatten_step(violation_psi))
        append_step("step_safe", flatten_step(safe))
        append_step("step_reward", flatten_step(rew))
        append_step("step_use_solver", flatten_step(use_solver))
        append_step("step_fallback", flatten_step(fallback))
        append_step("step_successful_qp", flatten_step(successful_qp))
        append_step("step_qp_residual_max", flatten_step(qp_residual_max))

    arrays = {
        key: np.concatenate(vals, axis=0) if vals else np.asarray([], dtype=np.float64)
        for key, vals in episode_acc.items()
    }
    arrays.update(
        {
            key: np.concatenate(vals, axis=0) if vals else np.asarray([], dtype=np.float64)
            for key, vals in step_acc.items()
        }
    )
    step_safe = np.asarray(arrays.get("step_safe", []), dtype=np.float64)
    step_qp_residual_max = np.asarray(arrays["step_qp_residual_max"], dtype=np.float64)
    finite_qp_residual_max = step_qp_residual_max[np.isfinite(step_qp_residual_max)]

    total_steps = int(step_safe.size)
    summary = {
        "episodes": int(episodes),
        "total_steps": int(total_steps),
        "inference_freq": (
            float(inference_total_decisions / inference_total_time_sec) if inference_total_time_sec > 0.0 else 0.0
        ),
        "safe_rate": _safe_mean_1d(step_safe),
        "use_solver_rate": _safe_mean_1d(arrays["step_use_solver"]),
        "fallback_rate": _safe_mean_1d(arrays["step_fallback"]),
        "successful_qp_rate": _safe_mean_1d(arrays["step_successful_qp"]),
        "qp_residual_max_mean": _safe_mean_1d(step_qp_residual_max),
        "qp_residual_max_p95": (
            float(np.percentile(finite_qp_residual_max, 95.0)) if finite_qp_residual_max.size else 0.0
        ),
        "qp_success_residual_tol": float(qp_success_tol),
        "env_terminate_on_violation": bool(env_eval_cfg.terminate_on_violation),
    }
    summary = _add_public_summary_metric_blocks(summary, arrays)
    return summary, arrays


def _build_eval_filename(
    tag: str,
    checkpoint: str,
    episodes: int,
    max_steps: int,
    env_dt: float,
    label: str,
) -> str:
    parts = [
        f"trajTrack_eval-{tag}",
        f"ckpt_{_sanitize_token(checkpoint)}",
        f"ep{int(episodes)}",
        f"ms{int(max_steps)}",
        f"dt{_dt_token(env_dt)}",
    ]
    if label.strip():
        parts.append(_sanitize_token(label.strip()))
    return "-".join(parts) + ".npz"


def _evaluate_and_save_task(task: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(task["run_dir"])
    checkpoint = str(task["checkpoint"])
    outputs_dir = Path(task["outputs_dir"])
    episodes = int(task["episodes"])
    max_steps = int(task["max_steps"])
    env_dt = float(task["env_dt"])
    eval_seed_base = int(task["eval_seed_base"])
    rollout_batch_size = int(task.get("rollout_batch_size", 256))
    eval_v_ref_mode = str(task.get("eval_v_ref_mode", "saved")).strip().lower()
    eval_v_ref_arg = task.get("eval_v_ref", None)
    eval_label = str(task.get("eval_label", ""))
    tag = str(task["tag"])

    cfg_json = _load_json(run_dir / "configs.json")
    sac_cfg = _dataclass_from_dict(SACConfig, cfg_json.get("sac", {}))
    env_cfg = _dataclass_from_dict(UnicycleEnvConfig, cfg_json.get("env", {}))
    cbf_cfg = _dataclass_from_dict(
        UnicycleBCBFConfig,
        cfg_json.get("cbf", {}),
    )
    v_ref_train = _infer_saved_v_ref(env_cfg)
    if eval_v_ref_mode == "saved":
        v_ref_eval = float(v_ref_train)
    elif eval_v_ref_mode == "override":
        if eval_v_ref_arg is None:
            raise ValueError("eval_v_ref_mode='override' requires eval_v_ref to be provided.")
        v_ref_eval = _coerce_finite_float(eval_v_ref_arg, np.nan)
        if not np.isfinite(v_ref_eval):
            raise ValueError(f"Invalid eval_v_ref={eval_v_ref_arg}; expected a finite float.")
    else:
        raise ValueError(f"Unsupported eval_v_ref_mode: {eval_v_ref_mode}")

    weights_path = run_dir / f"{checkpoint}_weights.pkl"
    actor_params = _load_actor_params(weights_path)

    env_eval_cfg = _build_eval_env_cfg(
        env_cfg,
        env_dt=float(env_dt),
        max_steps=int(max_steps),
        eval_v_ref=float(v_ref_eval),
    )

    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_name = _build_eval_filename(
        tag=tag,
        checkpoint=checkpoint,
        episodes=episodes,
        max_steps=max_steps,
        env_dt=env_dt,
        label=eval_label,
    )
    out_path = eval_dir / out_name

    summary, arrays = _evaluate_checkpoint(
        actor_params=actor_params,
        sac_cfg=sac_cfg,
        env_eval_cfg=env_eval_cfg,
        cbf_cfg=cbf_cfg,
        episodes=episodes,
        eval_seed_base=eval_seed_base,
        rollout_batch_size=rollout_batch_size,
    )
    diagnostic_plots = _make_representative_episode_plots(
        arrays,
        out_prefix=out_path.with_suffix(""),
        env_cfg_eval=env_eval_cfg,
    )

    metadata = {
        "outputs_dir": str(outputs_dir),
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "checkpoint": checkpoint,
        "weights_path": str(weights_path),
        "episodes": episodes,
        "max_steps": max_steps,
        "env_dt": env_dt,
        "eval_seed_base": eval_seed_base,
        "eval_v_ref_mode": eval_v_ref_mode,
        "v_ref_train": float(v_ref_train),
        "v_ref_eval": float(v_ref_eval),
        "v_ref": float(v_ref_eval),
        "eval_label": eval_label,
        "diagnostic_episode_plots": diagnostic_plots,
        "summary": summary,
        "evaluation_runtime": {
            "inference_freq": float(summary.get("inference_freq", 0.0)),
            "inference_total_decisions": int(summary.get("total_steps", 0)),
            "rollout_batch_size": int(rollout_batch_size),
            "inference_measurement": "timed on evaluation observations with env stepping excluded and JIT warmup excluded",
        },
        "hyperparams": {
            "w_v": float(getattr(env_cfg, "w_v", np.nan)),
            "w_lane_y": float(getattr(env_cfg, "w_lane_y", np.nan)),
            "w_lane_psi": float(getattr(env_cfg, "w_lane_psi", np.nan)),
            "w_control": float(getattr(env_cfg, "w_control", np.nan)),
            "traj_y_amplitude": float(getattr(env_cfg, "traj_y_amplitude", np.nan)),
            "traj_y_period": float(getattr(env_cfg, "traj_y_period", np.nan)),
        },
    }

    np.savez(
        out_path,
        **arrays,
        summary_json=np.asarray(json.dumps(summary, sort_keys=True), dtype=np.str_),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )

    with open(out_path.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "metadata": metadata}, f, indent=2)

    return {
        "status": "ok",
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "v_ref": float(v_ref_eval),
        "v_ref_train": float(v_ref_train),
        "v_ref_eval": float(v_ref_eval),
        "out_path": str(out_path),
        "summary": summary,
        "diagnostic_episode_plots": diagnostic_plots,
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
                rec = _evaluate_and_save_task(task)
                results.append(rec)
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
                rec = fut.result()
                results.append(rec)
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
            "Parallel trajectory policy evaluation + analysis. "
            "Evaluations are saved under each run folder in run_dir/evaluation."
        )
    )
    parser.add_argument("--outputs_dir", type=str, required=True, help="Outputs root containing run directories.")
    parser.add_argument("--run_glob", type=str, default="sac_lane-*", help="Glob for run directory names.")
    parser.add_argument("--experiment", type=str, default="", help="Optional single run directory name/path.")
    parser.add_argument("--episodes", type=int, default=200, help="Evaluation episodes per checkpoint.")
    parser.add_argument("--env_dt", type=float, default=0.05, help="Evaluation env dt.")
    parser.add_argument("--max_steps", type=int, default=400, help="Evaluation max steps per episode.")
    parser.add_argument(
        "--eval_v_ref_mode",
        type=str,
        default="saved",
        choices=["saved", "override"],
        help="saved: evaluate each run at its training v_ref from configs.json; override: evaluate all runs at --eval_v_ref.",
    )
    parser.add_argument(
        "--eval_v_ref",
        type=float,
        default=None,
        help="Required when --eval_v_ref_mode=override. Sets reward_v_des and traj_v_mean during evaluation.",
    )
    parser.add_argument("--eval_seed_base", type=int, default=2_500_000, help="Base seed for evaluation episodes.")
    parser.add_argument(
        "--weight_preference",
        type=str,
        default="both",
        choices=["both", "best_only", "final_only"],
        help="Which checkpoints to evaluate for each run.",
    )
    parser.add_argument(
        "--parallel_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for evaluation stage.",
    )
    parser.add_argument(
        "--rollout_batch_size",
        type=int,
        default=256,
        help="Number of evaluation episodes to roll out together inside the JAX/GPU batched evaluator.",
    )
    parser.add_argument(
        "--eval_label",
        type=str,
        default="",
        help="Optional label embedded in saved evaluation filenames/metadata.",
    )
    args = parser.parse_args(argv)

    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}")
    if args.max_steps <= 0:
        raise ValueError(f"--max_steps must be positive, got {args.max_steps}")
    if args.parallel_workers <= 0:
        raise ValueError(f"--parallel_workers must be positive, got {args.parallel_workers}")
    if args.rollout_batch_size <= 0:
        raise ValueError(f"--rollout_batch_size must be positive, got {args.rollout_batch_size}")
    if args.parallel_workers != 1:
        print(
            "parallel_workers>1 launches multiple JAX evaluator processes; "
            "the evaluator is already GPU-batched internally."
        )
    if args.eval_v_ref_mode == "override":
        if args.eval_v_ref is None:
            raise ValueError("--eval_v_ref is required when --eval_v_ref_mode=override")
        if not np.isfinite(float(args.eval_v_ref)):
            raise ValueError(f"--eval_v_ref must be finite, got {args.eval_v_ref}")

    outputs_dir = _resolve_outputs_dir(args.outputs_dir)
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
        for ckpt in sorted(checkpoints):
            tasks.append(
                {
                    "outputs_dir": str(outputs_dir),
                    "run_dir": str(run_dir),
                    "checkpoint": ckpt,
                    "episodes": int(args.episodes),
                    "max_steps": int(args.max_steps),
                    "env_dt": float(args.env_dt),
                    "eval_seed_base": int(args.eval_seed_base),
                    "rollout_batch_size": int(args.rollout_batch_size),
                    "eval_v_ref_mode": str(args.eval_v_ref_mode),
                    "eval_v_ref": (float(args.eval_v_ref) if args.eval_v_ref is not None else None),
                    "eval_label": str(args.eval_label),
                    "tag": eval_tag,
                }
            )
    print(
        f"Starting evaluation for {len(tasks)} tasks "
        f"({len(run_dirs)} runs x {len(checkpoints)} checkpoint(s)) "
        f"with workers={args.parallel_workers} and rollout_batch_size={args.rollout_batch_size}."
    )
    eval_results, eval_skipped = _run_parallel_evaluation(tasks, workers=int(args.parallel_workers))
    print(f"Evaluation artifacts saved under each run's 'evaluation/' folder.")
    print(f"Successful eval tasks: {len(eval_results)} / {len(tasks)}")
    if eval_skipped:
        print(f"Failed eval tasks: {len(eval_skipped)}")
        for item in eval_skipped[:10]:
            print(
                f"  - {item['run_name']} [ckpt={item['checkpoint']}]: {item['reason']}"
            )
        if len(eval_skipped) > 10:
            print(f"  ... and {len(eval_skipped) - 10} more")


if __name__ == "__main__":
    main()
