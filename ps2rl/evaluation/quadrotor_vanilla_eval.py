#!/usr/bin/env python
"""Evaluate and analyze saved quadrotor tracking policies.
These policies are trained to only track a powerloop reference.
They do not care for safety, and this script does not report safety metrics.

The ranking score intentionally ignores safety. During evaluation we force
``terminate_on_violation=False`` so every episode rolls out across the full
reference horizon.

Per-episode tracking score (lower is better):
    4.00 * (pos_xz_rmse / 0.25 m)
  + 3.00 * (vel_xz_rmse / 0.50 m/s)
  + 2.50 * (pitch_rmse_deg / 5 deg)
  + 1.50 * (p95_pos_xz / 0.50 m)
  + 1.00 * (max_pos_xz / 1.00 m)
  + 0.75 * (y_rmse / 0.10 m)
  + 0.50 * (vy_rmse / 0.25 m/s)
  + 0.25 * (roll_rmse_deg / 5 deg)
  + 0.25 * (yaw_rmse_deg / 5 deg)

Run-level ranking is lexicographic on:
  1) tracking_score_mean
  2) tracking_score_p95
  3) pos_xz_rmse_mean
  4) vel_xz_rmse_mean
  5) pitch_rmse_deg_mean
  6) y_rmse_mean
  7) vy_rmse_mean
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
from dataclasses import dataclass, fields
from datetime import datetime
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import re
from typing import Any

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

from ps2rl.utils.paths import PROJECT_ROOT

from ps2rl.cil.quadrotor_backup_cbf import (
    QuadrotorBCBFConfig,
    QuadrotorBackupCBFProjector,
)
from ps2rl.envs.quadrotor_env import (
    QuadrotorEnvConfig,
    _load_reference_bundle,
    build_quadrotor_env,
)
from ps2rl.utils.quaternion import (
    normalize_quaternion_np,
    quaternion_from_euler_zyx_np,
    quaternion_multiply_np,
)
from ps2rl.cil.cil_policy import ActorConfig
from ps2rl.plotting.plots import plot_quad_trajectory
from ps2rl.phase2_ps2.quadrotor_ps2_trainer import SACConfig, _build_action_fns

try:
    from ps2rl.envs.assets.generate_quadrotor_powerloop_reference import _save_animation as _save_reference_animation

    _HAS_REFERENCE_ANIMATION = True
except Exception:  # pragma: no cover
    _save_reference_animation = None  # type: ignore[assignment]
    _HAS_REFERENCE_ANIMATION = False


DEFAULT_RUN_GLOB = "sac_quadTrack-*"
DEFAULT_SEED_GROUP_STRIDE = 10_000


@dataclass(frozen=True)
class ScoreTerm:
    weight: float
    scale: float
    description: str


TRACKING_SCORE_TERMS: dict[str, ScoreTerm] = {
    "pos_xz_rmse": ScoreTerm(4.00, 0.25, "Primary geometric tracking in the power-loop plane."),
    "vel_xz_rmse": ScoreTerm(3.00, 0.50, "Primary dynamic tracking in the power-loop plane."),
    "pitch_rmse_deg": ScoreTerm(2.50, 5.00, "Pitch drives the loop geometry, so emphasize it."),
    "p95_pos_xz": ScoreTerm(1.50, 0.50, "Penalize episodes that lose the track for a segment."),
    "max_pos_xz": ScoreTerm(1.00, 1.00, "Penalize worst-case blowups on the loop."),
    "y_rmse": ScoreTerm(0.75, 0.10, "Secondary penalty for leaving the loop plane."),
    "vy_rmse": ScoreTerm(0.50, 0.25, "Secondary penalty for lateral velocity drift."),
    "roll_rmse_deg": ScoreTerm(0.25, 5.00, "Roll is diagnostic, but less important than pitch."),
    "yaw_rmse_deg": ScoreTerm(0.25, 5.00, "Yaw is diagnostic, but less important than pitch."),
}


def _sanitize_token(x: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(x))
    token = token.strip("_.-")
    return token or "na"


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


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _dataclass_from_dict(cls, payload: dict[str, Any], overrides: dict[str, Any] | None = None):
    valid = {f.name for f in fields(cls) if f.init}
    kwargs = {k: v for k, v in payload.items() if k in valid}
    if overrides:
        kwargs.update(overrides)
    return cls(**kwargs)


def _iter_run_dirs(outputs_dir: Path, run_glob: str, experiment: str = "") -> list[Path]:
    if experiment.strip():
        exp = experiment.strip()
        p = Path(exp)
        run_dir = p if p.exists() and p.is_dir() else outputs_dir / exp
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {exp}")
        if not (run_dir / "configs.json").exists():
            raise FileNotFoundError(f"Missing configs.json under run directory: {run_dir}")
        return [run_dir]

    run_dirs: list[Path] = []
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
            f"'actor_params' not found in {weights_path}. Available keys: {sorted(payload.keys())}"
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
        f"Actor expects obs_dim={expected_dim}, but env observation has shape {obs_flat.shape}. Cannot pad safely."
    )


def _normalize_quaternion_np(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    nrm = np.linalg.norm(q, axis=-1, keepdims=True)
    q_norm = q / np.maximum(nrm, eps)
    q_id = np.zeros_like(q_norm)
    q_id[..., 0] = 1.0
    valid = nrm > eps
    return np.where(valid, q_norm, q_id)


# --- NumPy reproduction of the QuadrotorEnv reset path ------------------------
# The ported evaluation (``_evaluate_checkpoint``) drives the JAX
# ``build_quadrotor_env`` for dynamics, but reproduces the ground-truth NumPy
# init sampler BITWISE so the rank-key metrics (``pos_xz_rmse_mean`` etc.) are not
# shifted by the JAX-random init that the sibling PS2 evaluator uses. These
# helpers are self-contained (they do NOT depend on the NumPy ``QuadrotorEnv``,
# now removed) and use the exact shared f64 quaternion ops that env used, so they
# stayed bitwise-identical across that deletion (7b-10, re-verified by job_7b8).


def _initial_reference_np(
    ref_states: np.ndarray, ref_omega_cmd: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reference state / body-rate at step 0 (mirrors ``QuadrotorEnv._reference(0)``).

    At ``t = 0`` the interpolation weight ``w`` is 0, so the reference is
    ``ref_states[0]`` with its quaternion re-normalized (matching the env's
    ``(1 - w) * ref0 + w * ref1`` arithmetic exactly, including any signed-zero).
    """
    ref_state = (1.0 - 0.0) * np.asarray(ref_states[0], dtype=np.float64) + 0.0 * np.asarray(
        ref_states[1], dtype=np.float64
    )
    ref_state[6:10] = normalize_quaternion_np(ref_state[6:10])
    ref_omega = (1.0 - 0.0) * np.asarray(ref_omega_cmd[0], dtype=np.float64) + 0.0 * np.asarray(
        ref_omega_cmd[1], dtype=np.float64
    )
    return ref_state, ref_omega


def sample_initial_state_np(
    cfg: QuadrotorEnvConfig, ref_states: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Bitwise reproduction of ``QuadrotorEnv._sample_initial_state`` (NumPy, f64)."""
    ref0 = (1.0 - 0.0) * np.asarray(ref_states[0], dtype=np.float64) + 0.0 * np.asarray(
        ref_states[1], dtype=np.float64
    )
    ref0[6:10] = normalize_quaternion_np(ref0[6:10])
    px0, py0, pz0 = ref0[0], ref0[1], ref0[2]
    vx0, vy0, vz0 = ref0[3], ref0[4], ref0[5]
    q0 = normalize_quaternion_np(ref0[6:10])

    px = px0 + rng.uniform(-cfg.init_px_range, cfg.init_px_range)
    py = py0 + rng.uniform(-cfg.init_py_range, cfg.init_py_range)
    pz = pz0 + rng.uniform(-cfg.init_pz_range, cfg.init_pz_range)
    pz = min(pz, cfg.z_max - 0.05)

    vx = vx0 + rng.uniform(-cfg.init_v_range, cfg.init_v_range)
    vy = vy0 + rng.uniform(-cfg.init_v_range, cfg.init_v_range)
    vz = vz0 + rng.uniform(-cfg.init_v_range, cfg.init_v_range)

    tilt_max = np.deg2rad(cfg.init_tilt_deg_range)
    yaw_max = np.deg2rad(cfg.init_yaw_deg_range)
    roll = rng.uniform(-tilt_max, tilt_max)
    pitch = rng.uniform(-tilt_max, tilt_max)
    yaw = rng.uniform(-yaw_max, yaw_max)
    q_perturb = quaternion_from_euler_zyx_np(roll, pitch, yaw)
    q = normalize_quaternion_np(quaternion_multiply_np(q_perturb, q0))

    return np.array([px, py, pz, vx, vy, vz, q[0], q[1], q[2], q[3]], dtype=np.float64)


def _observation_np(
    x: np.ndarray,
    ref_state: np.ndarray,
    ref_omega: np.ndarray,
    *,
    include_time_features: bool,
) -> np.ndarray:
    """Step-0 observation (mirrors ``QuadrotorEnv._observation`` at ``steps = 0``).

    At ``t = 0`` the time features are ``[time_sec, sin(phase), cos(phase)] =
    [0, 0, 1]``. Only the initial observation is built here; every subsequent
    observation is returned by the JAX ``env.step``.
    """
    x = np.asarray(x, dtype=np.float64)
    ref_state = np.asarray(ref_state, dtype=np.float64)
    ref_omega = np.asarray(ref_omega, dtype=np.float64)
    if include_time_features:
        time_feats = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return np.concatenate([x, ref_state, ref_omega, time_feats], axis=0)
    return np.concatenate([x, ref_state, ref_omega], axis=0)


def _wrap_angle_rad(x: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return np.arctan2(np.sin(arr), np.cos(arr))


def _quaternion_to_euler_deg_batch_np(q_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = _normalize_quaternion_np(np.asarray(q_batch, dtype=np.float64))
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)


def _rms(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def _safe_mean(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def _safe_std(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.std(arr))


def _safe_percentile(x: np.ndarray, q: float) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def _score_formula_payload() -> dict[str, Any]:
    return {
        "definition": "Lower is better. Each term contributes weight * metric / scale.",
        "ranking_priority": [
            "tracking_score_mean",
            "tracking_score_p95",
            "pos_xz_rmse_mean",
            "vel_xz_rmse_mean",
            "pitch_rmse_deg_mean",
            "y_rmse_mean",
            "vy_rmse_mean",
        ],
        "terms": {
            name: {
                "weight": float(term.weight),
                "scale": float(term.scale),
                "description": term.description,
            }
            for name, term in TRACKING_SCORE_TERMS.items()
        },
    }


def _tracking_score_from_metrics(metrics: dict[str, float]) -> float:
    score = 0.0
    for key, term in TRACKING_SCORE_TERMS.items():
        value = float(metrics.get(key, np.inf))
        if not np.isfinite(value):
            return float(np.inf)
        score += float(term.weight) * float(value) / float(term.scale)
    return float(score)


def _metric_value(metrics: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(metrics.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _tracking_rank_key(metrics: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    return (
        _metric_value(metrics, "tracking_score_mean", float(np.inf)),
        _metric_value(metrics, "tracking_score_p95", float(np.inf)),
        _metric_value(metrics, "pos_xz_rmse_mean", float(np.inf)),
        _metric_value(metrics, "vel_xz_rmse_mean", float(np.inf)),
        _metric_value(metrics, "pitch_rmse_deg_mean", float(np.inf)),
        _metric_value(metrics, "y_rmse_mean", float(np.inf)),
        _metric_value(metrics, "vy_rmse_mean", float(np.inf)),
    )


def _build_eval_env_cfg(env_cfg: QuadrotorEnvConfig, *, force_full_episode: bool) -> QuadrotorEnvConfig:
    overrides: dict[str, Any] = {}
    if force_full_episode:
        overrides["terminate_on_violation"] = False
    return _dataclass_from_dict(QuadrotorEnvConfig, vars(env_cfg), overrides=overrides)


def _build_eval_dir_name(
    tag: str,
    checkpoint: str,
    num_eval_seeds: int,
    episodes_per_seed: int,
    label: str,
) -> str:
    total_episodes = int(num_eval_seeds) * int(episodes_per_seed)
    parts = [
        f"quadTrack_eval-{tag}",
        f"ckpt_{_sanitize_token(checkpoint)}",
        f"seeds{int(num_eval_seeds)}",
        f"epPerSeed{int(episodes_per_seed)}",
        f"epTotal{int(total_episodes)}",
    ]
    if label.strip():
        parts.append(_sanitize_token(label.strip()))
    return "-".join(parts)


def _build_ref_state(info: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(info.get("ref_px", 0.0)),
            float(info.get("ref_py", 0.0)),
            float(info.get("ref_pz", 0.0)),
            float(info.get("ref_vx", 0.0)),
            float(info.get("ref_vy", 0.0)),
            float(info.get("ref_vz", 0.0)),
            float(info.get("ref_qw", 1.0)),
            float(info.get("ref_qx", 0.0)),
            float(info.get("ref_qy", 0.0)),
            float(info.get("ref_qz", 0.0)),
        ],
        dtype=np.float64,
    )


def _extract_episode_trace(arrays: dict[str, np.ndarray], ep_idx: int) -> dict[str, np.ndarray]:
    ep_all = np.asarray(arrays.get("episode_idx", []), dtype=np.int32).reshape(-1)
    mask = ep_all == int(ep_idx)
    if not np.any(mask):
        return {}

    step_all = np.asarray(arrays.get("step_in_episode", np.arange(ep_all.size)), dtype=np.int32).reshape(-1)
    order = np.argsort(step_all[mask])

    trace: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            continue
        if arr.shape[0] != ep_all.shape[0]:
            continue
        trace[key] = arr[mask][order]

    if "ref_state" in trace:
        ref = np.asarray(trace["ref_state"], dtype=np.float64)
        trace["ref_px"] = ref[:, 0]
        trace["ref_py"] = ref[:, 1]
        trace["ref_pz"] = ref[:, 2]
        trace["ref_vx"] = ref[:, 3]
        trace["ref_vy"] = ref[:, 4]
        trace["ref_vz"] = ref[:, 5]
        trace["ref_qw"] = ref[:, 6]
        trace["ref_qx"] = ref[:, 7]
        trace["ref_qy"] = ref[:, 8]
        trace["ref_qz"] = ref[:, 9]
    return trace


def _save_best_episode_gif(
    trace: dict[str, np.ndarray],
    *,
    out_path: Path,
    title: str,
    slowdown: float,
    trail_length: int,
    print_every: int,
) -> str | None:
    if (not _HAS_REFERENCE_ANIMATION) or (_save_reference_animation is None):
        return None

    next_states = np.asarray(trace.get("next_obs", []), dtype=np.float64)
    if next_states.ndim != 2 or next_states.shape[1] != 10 or next_states.shape[0] == 0:
        return None

    obs_states = np.asarray(trace.get("obs", []), dtype=np.float64)
    ref_t = np.asarray(trace.get("ref_time_sec", []), dtype=np.float64).reshape(-1)

    if obs_states.ndim == 2 and obs_states.shape[1] == 10 and obs_states.shape[0] == next_states.shape[0]:
        full_states = np.vstack([obs_states[0:1], next_states])
    else:
        full_states = next_states

    if ref_t.size == next_states.shape[0]:
        if full_states.shape[0] == next_states.shape[0] + 1:
            dt = float(ref_t[1] - ref_t[0]) if ref_t.size > 1 else 0.02
            full_t = np.concatenate([np.asarray([ref_t[0] - dt], dtype=np.float64), ref_t])
        else:
            full_t = ref_t
    else:
        full_t = np.arange(full_states.shape[0], dtype=np.float64) * 0.02

    _, pitch_deg, _ = _quaternion_to_euler_deg_batch_np(full_states[:, 6:10])
    payload = {
        "t": np.asarray(full_t, dtype=np.float64),
        "states": np.asarray(full_states, dtype=np.float64),
        "pitch": np.deg2rad(np.asarray(pitch_deg, dtype=np.float64)),
    }
    _save_reference_animation(
        payload,
        out_path,
        z_max=float(np.max(full_states[:, 2]) + 0.5),
        slowdown=float(slowdown),
        trail_length=max(1, int(trail_length)),
        print_every=max(1, int(print_every)),
        title=title,
    )
    return str(out_path)


def _episode_metrics_from_lists(local: dict[str, list[float]], episode_return: float, episode_len: int) -> dict[str, float]:
    x_err = np.asarray(local["x_error"], dtype=np.float64)
    y_err = np.asarray(local["y_error"], dtype=np.float64)
    z_err = np.asarray(local["z_error"], dtype=np.float64)
    vx_err = np.asarray(local["vx_error"], dtype=np.float64)
    vy_err = np.asarray(local["vy_error"], dtype=np.float64)
    vz_err = np.asarray(local["vz_error"], dtype=np.float64)
    pos_xz = np.asarray(local["pos_xz_error"], dtype=np.float64)
    vel_xz = np.asarray(local["vel_xz_error"], dtype=np.float64)
    pos_norm = np.asarray(local["pos_error_norm"], dtype=np.float64)
    vel_norm = np.asarray(local["vel_error_norm"], dtype=np.float64)
    att_norm = np.asarray(local["att_error_norm"], dtype=np.float64)
    roll_err = np.asarray(local["roll_error_deg"], dtype=np.float64)
    pitch_err = np.asarray(local["pitch_error_deg"], dtype=np.float64)
    yaw_err = np.asarray(local["yaw_error_deg"], dtype=np.float64)
    safe = np.asarray(local["safe"], dtype=np.float64)
    ref_progress = np.asarray(local["ref_progress"], dtype=np.float64)

    metrics = {
        "return": float(episode_return),
        "episode_length": int(episode_len),
        "x_rmse": _rms(x_err),
        "y_rmse": _rms(y_err),
        "z_rmse": _rms(z_err),
        "vx_rmse": _rms(vx_err),
        "vy_rmse": _rms(vy_err),
        "vz_rmse": _rms(vz_err),
        "pos_xyz_rmse": _rms(np.sqrt(np.square(x_err) + np.square(y_err) + np.square(z_err))),
        "vel_xyz_rmse": _rms(np.sqrt(np.square(vx_err) + np.square(vy_err) + np.square(vz_err))),
        "pos_xz_rmse": _rms(pos_xz),
        "vel_xz_rmse": _rms(vel_xz),
        "p95_pos_xz": _safe_percentile(pos_xz, 95.0),
        "max_pos_xz": float(np.max(pos_xz)) if pos_xz.size else 0.0,
        "p95_vel_xz": _safe_percentile(vel_xz, 95.0),
        "max_vel_xz": float(np.max(vel_xz)) if vel_xz.size else 0.0,
        "roll_rmse_deg": _rms(roll_err),
        "pitch_rmse_deg": _rms(pitch_err),
        "yaw_rmse_deg": _rms(yaw_err),
        "pos_error_norm_mean": _safe_mean(pos_norm),
        "vel_error_norm_mean": _safe_mean(vel_norm),
        "att_error_norm_mean": _safe_mean(att_norm),
        "safe_rate": _safe_mean(safe),
        "ref_progress_end": float(ref_progress[-1]) if ref_progress.size else 0.0,
    }
    metrics["tracking_score"] = _tracking_score_from_metrics(metrics)
    return metrics


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
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    env_fns = build_quadrotor_env(env_cfg)
    projector = QuadrotorBackupCBFProjector(cbf_cfg)
    ref_bundle = _load_reference_bundle(env_cfg.reference_path)
    ref_states = ref_bundle["states"]
    ref_omega_cmd = ref_bundle["omega_cmd"]
    ref0_state, ref0_omega = _initial_reference_np(ref_states, ref_omega_cmd)
    include_time_features = bool(env_cfg.include_time_features)

    action_scale = jnp.asarray(
        np.array(
            [env_cfg.a_cmd_max, env_cfg.omega_max, env_cfg.omega_max, env_cfg.omega_max],
            dtype=np.float64,
        ),
        dtype=jnp.float32,
    )
    actor_obs_dim = _infer_actor_obs_dim(actor_params)
    if actor_obs_dim is None:
        actor_obs_dim = int(env_fns.obs_dim)

    actor_cfg = ActorConfig(
        obs_dim=int(actor_obs_dim),
        action_dim=env_fns.action_dim,
        hidden_sizes=(sac_cfg.hidden_size, sac_cfg.hidden_size),
    )
    _, eval_action_fn = _build_action_fns(
        sac_cfg,
        actor_cfg,
        cbf_cfg,
        action_scale,
        backup_runtime=projector.runtime,
        return_solver_info=True,
    )

    template_state, _ = env_fns.reset(jax.random.PRNGKey(0))
    step_key = jax.random.PRNGKey(0)

    def _reset_np(episode_seed: int) -> tuple[Any, np.ndarray]:
        rng = np.random.default_rng(int(episode_seed))
        x0 = sample_initial_state_np(env_cfg, ref_states, rng)
        state = template_state._replace(x=jnp.asarray(x0, dtype=jnp.float32))
        obs0 = _observation_np(
            x0, ref0_state, ref0_omega, include_time_features=include_time_features
        )
        return state, obs0

    warm_state, warm_obs = _reset_np(eval_seed_base + 999_999)
    warm_obs_j = jnp.asarray(_adapt_obs(warm_obs, actor_obs_dim), dtype=jnp.float32)
    warm_safe_action, _, _, _ = eval_action_fn(actor_params, warm_obs_j)
    warm_step_out = env_fns.step(
        warm_state, jnp.asarray(warm_safe_action, dtype=jnp.float32), step_key
    )
    try:
        jax.block_until_ready((warm_safe_action, warm_step_out))
    except Exception:  # noqa: BLE001
        pass

    total_episodes = int(num_eval_seeds) * int(episodes_per_seed)

    episode_idx_list: list[int] = []
    episode_seed_group_list: list[int] = []
    episode_in_seed_list: list[int] = []
    episode_seed_list: list[int] = []
    episode_return_list: list[float] = []
    episode_length_list: list[int] = []
    episode_tracking_score_list: list[float] = []
    episode_x_rmse_list: list[float] = []
    episode_y_rmse_list: list[float] = []
    episode_z_rmse_list: list[float] = []
    episode_vx_rmse_list: list[float] = []
    episode_vy_rmse_list: list[float] = []
    episode_vz_rmse_list: list[float] = []
    episode_pos_xyz_rmse_list: list[float] = []
    episode_vel_xyz_rmse_list: list[float] = []
    episode_pos_xz_rmse_list: list[float] = []
    episode_vel_xz_rmse_list: list[float] = []
    episode_p95_pos_xz_list: list[float] = []
    episode_max_pos_xz_list: list[float] = []
    episode_p95_vel_xz_list: list[float] = []
    episode_max_vel_xz_list: list[float] = []
    episode_roll_rmse_deg_list: list[float] = []
    episode_pitch_rmse_deg_list: list[float] = []
    episode_yaw_rmse_deg_list: list[float] = []
    episode_pos_error_norm_mean_list: list[float] = []
    episode_vel_error_norm_mean_list: list[float] = []
    episode_att_error_norm_mean_list: list[float] = []
    episode_safe_rate_list: list[float] = []
    episode_ref_progress_end_list: list[float] = []

    flat_episode_idx: list[int] = []
    flat_seed_group_idx: list[int] = []
    flat_episode_in_seed: list[int] = []
    flat_episode_seed: list[int] = []
    flat_step_in_episode: list[int] = []
    flat_ref_time_sec: list[float] = []
    flat_ref_progress: list[float] = []
    flat_obs: list[np.ndarray] = []
    flat_next_obs: list[np.ndarray] = []
    flat_act_safe: list[np.ndarray] = []
    flat_act_raw: list[np.ndarray] = []
    flat_rew: list[float] = []
    flat_safe: list[float] = []
    flat_pos_error_norm: list[float] = []
    flat_vel_error_norm: list[float] = []
    flat_att_error_norm: list[float] = []
    flat_x_error: list[float] = []
    flat_y_error: list[float] = []
    flat_z_error: list[float] = []
    flat_vx_error: list[float] = []
    flat_vy_error: list[float] = []
    flat_vz_error: list[float] = []
    flat_pos_xz_error: list[float] = []
    flat_vel_xz_error: list[float] = []
    flat_roll_deg: list[float] = []
    flat_pitch_deg: list[float] = []
    flat_yaw_deg: list[float] = []
    flat_ref_roll_deg: list[float] = []
    flat_ref_pitch_deg: list[float] = []
    flat_ref_yaw_deg: list[float] = []
    flat_roll_error_deg: list[float] = []
    flat_pitch_error_deg: list[float] = []
    flat_yaw_error_deg: list[float] = []
    flat_hard_deck_margin: list[float] = []
    flat_ref_state: list[np.ndarray] = []

    episode_counter = 0
    for seed_group_idx in range(int(num_eval_seeds)):
        seed_group_base = int(eval_seed_base) + int(seed_group_idx) * int(seed_group_stride)
        for ep_in_seed in range(int(episodes_per_seed)):
            episode_seed = int(seed_group_base + ep_in_seed)
            env_state, obs = _reset_np(episode_seed)
            done = False
            ep_return = 0.0
            ep_len = 0
            local: dict[str, list[float]] = {
                "x_error": [],
                "y_error": [],
                "z_error": [],
                "vx_error": [],
                "vy_error": [],
                "vz_error": [],
                "pos_xz_error": [],
                "vel_xz_error": [],
                "roll_error_deg": [],
                "pitch_error_deg": [],
                "yaw_error_deg": [],
                "pos_error_norm": [],
                "vel_error_norm": [],
                "att_error_norm": [],
                "safe": [],
                "ref_progress": [],
            }

            while not done:
                obs_phys = np.asarray(obs[:10], dtype=np.float64)
                obs_actor = _adapt_obs(obs, actor_obs_dim)
                obs_j = jnp.asarray(obs_actor, dtype=jnp.float32)
                safe_action, raw_action, _, _ = eval_action_fn(actor_params, obs_j)

                act_safe_np = np.asarray(safe_action, dtype=np.float64)
                act_raw_np = np.asarray(raw_action, dtype=np.float64)

                env_state, next_obs_j, _obs_out_j, rew_j, done_j, info = env_fns.step(
                    env_state, jnp.asarray(safe_action, dtype=jnp.float32), step_key
                )
                next_obs = np.asarray(next_obs_j, dtype=np.float64)
                rew = float(rew_j)
                done = bool(done_j)
                next_obs_phys = np.asarray(next_obs[:10], dtype=np.float64)
                ref_state = np.asarray(info.ref_state, dtype=np.float64)

                pos_error_norm = float(info.pos_error_norm)
                vel_error_norm = float(info.vel_error_norm)
                att_error_norm = float(info.att_error_norm)
                hard_deck_margin = float(info.hard_deck_margin)
                is_safe = float(info.is_safe)
                ref_progress = float(info.ref_progress)
                ref_time_sec = float(info.ref_time_sec)

                x_err = float(next_obs_phys[0] - ref_state[0])
                y_err = float(next_obs_phys[1] - ref_state[1])
                z_err = float(next_obs_phys[2] - ref_state[2])
                vx_err = float(next_obs_phys[3] - ref_state[3])
                vy_err = float(next_obs_phys[4] - ref_state[4])
                vz_err = float(next_obs_phys[5] - ref_state[5])
                pos_xz_error = float(np.sqrt(x_err * x_err + z_err * z_err))
                vel_xz_error = float(np.sqrt(vx_err * vx_err + vz_err * vz_err))

                roll_now_deg, pitch_now_deg, yaw_now_deg = _quaternion_to_euler_deg_batch_np(
                    next_obs_phys[6:10][None, :]
                )
                roll_ref_deg, pitch_ref_deg, yaw_ref_deg = _quaternion_to_euler_deg_batch_np(
                    ref_state[6:10][None, :]
                )
                roll_err_deg = float(np.abs(np.rad2deg(_wrap_angle_rad(np.deg2rad(roll_now_deg[0] - roll_ref_deg[0])))))
                pitch_err_deg = float(
                    np.abs(np.rad2deg(_wrap_angle_rad(np.deg2rad(pitch_now_deg[0] - pitch_ref_deg[0]))))
                )
                yaw_err_deg = float(np.abs(np.rad2deg(_wrap_angle_rad(np.deg2rad(yaw_now_deg[0] - yaw_ref_deg[0])))))

                ep_return += float(rew)
                ep_len += 1

                local["x_error"].append(x_err)
                local["y_error"].append(y_err)
                local["z_error"].append(z_err)
                local["vx_error"].append(vx_err)
                local["vy_error"].append(vy_err)
                local["vz_error"].append(vz_err)
                local["pos_xz_error"].append(pos_xz_error)
                local["vel_xz_error"].append(vel_xz_error)
                local["roll_error_deg"].append(roll_err_deg)
                local["pitch_error_deg"].append(pitch_err_deg)
                local["yaw_error_deg"].append(yaw_err_deg)
                local["pos_error_norm"].append(pos_error_norm)
                local["vel_error_norm"].append(vel_error_norm)
                local["att_error_norm"].append(att_error_norm)
                local["safe"].append(is_safe)
                local["ref_progress"].append(ref_progress)

                flat_episode_idx.append(int(episode_counter))
                flat_seed_group_idx.append(int(seed_group_idx))
                flat_episode_in_seed.append(int(ep_in_seed))
                flat_episode_seed.append(int(episode_seed))
                flat_step_in_episode.append(int(ep_len - 1))
                flat_ref_time_sec.append(ref_time_sec)
                flat_ref_progress.append(ref_progress)
                flat_obs.append(obs_phys.copy())
                flat_next_obs.append(next_obs_phys.copy())
                flat_act_safe.append(act_safe_np.copy())
                flat_act_raw.append(act_raw_np.copy())
                flat_rew.append(float(rew))
                flat_safe.append(is_safe)
                flat_pos_error_norm.append(pos_error_norm)
                flat_vel_error_norm.append(vel_error_norm)
                flat_att_error_norm.append(att_error_norm)
                flat_x_error.append(x_err)
                flat_y_error.append(y_err)
                flat_z_error.append(z_err)
                flat_vx_error.append(vx_err)
                flat_vy_error.append(vy_err)
                flat_vz_error.append(vz_err)
                flat_pos_xz_error.append(pos_xz_error)
                flat_vel_xz_error.append(vel_xz_error)
                flat_roll_deg.append(float(roll_now_deg[0]))
                flat_pitch_deg.append(float(pitch_now_deg[0]))
                flat_yaw_deg.append(float(yaw_now_deg[0]))
                flat_ref_roll_deg.append(float(roll_ref_deg[0]))
                flat_ref_pitch_deg.append(float(pitch_ref_deg[0]))
                flat_ref_yaw_deg.append(float(yaw_ref_deg[0]))
                flat_roll_error_deg.append(roll_err_deg)
                flat_pitch_error_deg.append(pitch_err_deg)
                flat_yaw_error_deg.append(yaw_err_deg)
                flat_hard_deck_margin.append(hard_deck_margin)
                flat_ref_state.append(ref_state.copy())

                obs = next_obs

            ep_metrics = _episode_metrics_from_lists(local, episode_return=ep_return, episode_len=ep_len)
            episode_idx_list.append(int(episode_counter))
            episode_seed_group_list.append(int(seed_group_idx))
            episode_in_seed_list.append(int(ep_in_seed))
            episode_seed_list.append(int(episode_seed))
            episode_return_list.append(float(ep_metrics["return"]))
            episode_length_list.append(int(ep_metrics["episode_length"]))
            episode_tracking_score_list.append(float(ep_metrics["tracking_score"]))
            episode_x_rmse_list.append(float(ep_metrics["x_rmse"]))
            episode_y_rmse_list.append(float(ep_metrics["y_rmse"]))
            episode_z_rmse_list.append(float(ep_metrics["z_rmse"]))
            episode_vx_rmse_list.append(float(ep_metrics["vx_rmse"]))
            episode_vy_rmse_list.append(float(ep_metrics["vy_rmse"]))
            episode_vz_rmse_list.append(float(ep_metrics["vz_rmse"]))
            episode_pos_xyz_rmse_list.append(float(ep_metrics["pos_xyz_rmse"]))
            episode_vel_xyz_rmse_list.append(float(ep_metrics["vel_xyz_rmse"]))
            episode_pos_xz_rmse_list.append(float(ep_metrics["pos_xz_rmse"]))
            episode_vel_xz_rmse_list.append(float(ep_metrics["vel_xz_rmse"]))
            episode_p95_pos_xz_list.append(float(ep_metrics["p95_pos_xz"]))
            episode_max_pos_xz_list.append(float(ep_metrics["max_pos_xz"]))
            episode_p95_vel_xz_list.append(float(ep_metrics["p95_vel_xz"]))
            episode_max_vel_xz_list.append(float(ep_metrics["max_vel_xz"]))
            episode_roll_rmse_deg_list.append(float(ep_metrics["roll_rmse_deg"]))
            episode_pitch_rmse_deg_list.append(float(ep_metrics["pitch_rmse_deg"]))
            episode_yaw_rmse_deg_list.append(float(ep_metrics["yaw_rmse_deg"]))
            episode_pos_error_norm_mean_list.append(float(ep_metrics["pos_error_norm_mean"]))
            episode_vel_error_norm_mean_list.append(float(ep_metrics["vel_error_norm_mean"]))
            episode_att_error_norm_mean_list.append(float(ep_metrics["att_error_norm_mean"]))
            episode_safe_rate_list.append(float(ep_metrics["safe_rate"]))
            episode_ref_progress_end_list.append(float(ep_metrics["ref_progress_end"]))
            episode_counter += 1

    assert episode_counter == total_episodes, (episode_counter, total_episodes)

    ep_tracking_score = np.asarray(episode_tracking_score_list, dtype=np.float64)
    ep_pos_xz_rmse = np.asarray(episode_pos_xz_rmse_list, dtype=np.float64)
    ep_vel_xz_rmse = np.asarray(episode_vel_xz_rmse_list, dtype=np.float64)
    ep_pitch_rmse = np.asarray(episode_pitch_rmse_deg_list, dtype=np.float64)
    ep_y_rmse = np.asarray(episode_y_rmse_list, dtype=np.float64)
    ep_vy_rmse = np.asarray(episode_vy_rmse_list, dtype=np.float64)
    ep_roll_rmse = np.asarray(episode_roll_rmse_deg_list, dtype=np.float64)
    ep_yaw_rmse = np.asarray(episode_yaw_rmse_deg_list, dtype=np.float64)
    ep_p95_pos_xz = np.asarray(episode_p95_pos_xz_list, dtype=np.float64)
    ep_max_pos_xz = np.asarray(episode_max_pos_xz_list, dtype=np.float64)
    ep_return = np.asarray(episode_return_list, dtype=np.float64)
    ep_length = np.asarray(episode_length_list, dtype=np.int32)
    ep_seed_group = np.asarray(episode_seed_group_list, dtype=np.int32)
    ep_seed_value = np.asarray(episode_seed_list, dtype=np.int64)
    ep_in_seed = np.asarray(episode_in_seed_list, dtype=np.int32)

    best_episode_idx = int(np.argmin(np.where(np.isfinite(ep_tracking_score), ep_tracking_score, np.inf))) if total_episodes > 0 else 0
    worst_episode_idx = int(np.argmax(np.where(np.isfinite(ep_tracking_score), ep_tracking_score, -np.inf))) if total_episodes > 0 else 0

    step_pos_error_norm = np.asarray(flat_pos_error_norm, dtype=np.float64)
    step_vel_error_norm = np.asarray(flat_vel_error_norm, dtype=np.float64)
    step_att_error_norm = np.asarray(flat_att_error_norm, dtype=np.float64)
    step_safe = np.asarray(flat_safe, dtype=np.float64)

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
                "tracking_score_mean": _safe_mean(ep_tracking_score[mask]),
                "tracking_score_std": _safe_std(ep_tracking_score[mask]),
                "pos_xz_rmse_mean": _safe_mean(ep_pos_xz_rmse[mask]),
                "vel_xz_rmse_mean": _safe_mean(ep_vel_xz_rmse[mask]),
                "pitch_rmse_deg_mean": _safe_mean(ep_pitch_rmse[mask]),
                "return_mean": _safe_mean(ep_return[mask]),
            }
        )

    summary: dict[str, Any] = {
        "episodes_total": int(total_episodes),
        "num_eval_seeds": int(num_eval_seeds),
        "episodes_per_seed": int(episodes_per_seed),
        "eval_seed_base": int(eval_seed_base),
        "seed_group_stride": int(seed_group_stride),
        "total_steps": int(step_pos_error_norm.size),
        "tracking_score_mean": _safe_mean(ep_tracking_score),
        "tracking_score_std": _safe_std(ep_tracking_score),
        "tracking_score_p50": _safe_percentile(ep_tracking_score, 50.0),
        "tracking_score_p95": _safe_percentile(ep_tracking_score, 95.0),
        "tracking_score_best": float(ep_tracking_score[best_episode_idx]) if ep_tracking_score.size else 0.0,
        "tracking_score_worst": float(ep_tracking_score[worst_episode_idx]) if ep_tracking_score.size else 0.0,
        "pos_xz_rmse_mean": _safe_mean(ep_pos_xz_rmse),
        "pos_xz_rmse_p95": _safe_percentile(ep_pos_xz_rmse, 95.0),
        "vel_xz_rmse_mean": _safe_mean(ep_vel_xz_rmse),
        "vel_xz_rmse_p95": _safe_percentile(ep_vel_xz_rmse, 95.0),
        "pitch_rmse_deg_mean": _safe_mean(ep_pitch_rmse),
        "pitch_rmse_deg_p95": _safe_percentile(ep_pitch_rmse, 95.0),
        "y_rmse_mean": _safe_mean(ep_y_rmse),
        "vy_rmse_mean": _safe_mean(ep_vy_rmse),
        "roll_rmse_deg_mean": _safe_mean(ep_roll_rmse),
        "yaw_rmse_deg_mean": _safe_mean(ep_yaw_rmse),
        "p95_pos_xz_mean": _safe_mean(ep_p95_pos_xz),
        "max_pos_xz_mean": _safe_mean(ep_max_pos_xz),
        "x_rmse_mean": _safe_mean(np.asarray(episode_x_rmse_list, dtype=np.float64)),
        "z_rmse_mean": _safe_mean(np.asarray(episode_z_rmse_list, dtype=np.float64)),
        "vx_rmse_mean": _safe_mean(np.asarray(episode_vx_rmse_list, dtype=np.float64)),
        "vz_rmse_mean": _safe_mean(np.asarray(episode_vz_rmse_list, dtype=np.float64)),
        "pos_xyz_rmse_mean": _safe_mean(np.asarray(episode_pos_xyz_rmse_list, dtype=np.float64)),
        "vel_xyz_rmse_mean": _safe_mean(np.asarray(episode_vel_xyz_rmse_list, dtype=np.float64)),
        "return_mean": _safe_mean(ep_return),
        "return_std": _safe_std(ep_return),
        "episode_length_mean": _safe_mean(ep_length.astype(np.float64)),
        "episode_length_std": _safe_std(ep_length.astype(np.float64)),
        "pos_error_norm_step_mean": _safe_mean(step_pos_error_norm),
        "vel_error_norm_step_mean": _safe_mean(step_vel_error_norm),
        "att_error_norm_step_mean": _safe_mean(step_att_error_norm),
        "safe_rate_step_mean": _safe_mean(step_safe),
        "ref_progress_end_mean": _safe_mean(np.asarray(episode_ref_progress_end_list, dtype=np.float64)),
        "ref_progress_end_min": float(np.min(np.asarray(episode_ref_progress_end_list, dtype=np.float64)))
        if episode_ref_progress_end_list
        else 0.0,
        "best_episode_idx": int(best_episode_idx),
        "worst_episode_idx": int(worst_episode_idx),
        "seed_group_stats": seed_group_stats,
    }

    arrays: dict[str, np.ndarray] = {
        "episode_idx_unique": np.asarray(episode_idx_list, dtype=np.int32),
        "episode_seed_group": np.asarray(episode_seed_group_list, dtype=np.int32),
        "episode_in_seed_group": np.asarray(episode_in_seed_list, dtype=np.int32),
        "episode_seed": np.asarray(episode_seed_list, dtype=np.int64),
        "episode_return": ep_return,
        "episode_length": ep_length,
        "episode_tracking_score": ep_tracking_score,
        "episode_x_rmse": np.asarray(episode_x_rmse_list, dtype=np.float64),
        "episode_y_rmse": np.asarray(episode_y_rmse_list, dtype=np.float64),
        "episode_z_rmse": np.asarray(episode_z_rmse_list, dtype=np.float64),
        "episode_vx_rmse": np.asarray(episode_vx_rmse_list, dtype=np.float64),
        "episode_vy_rmse": np.asarray(episode_vy_rmse_list, dtype=np.float64),
        "episode_vz_rmse": np.asarray(episode_vz_rmse_list, dtype=np.float64),
        "episode_pos_xyz_rmse": np.asarray(episode_pos_xyz_rmse_list, dtype=np.float64),
        "episode_vel_xyz_rmse": np.asarray(episode_vel_xyz_rmse_list, dtype=np.float64),
        "episode_pos_xz_rmse": ep_pos_xz_rmse,
        "episode_vel_xz_rmse": ep_vel_xz_rmse,
        "episode_p95_pos_xz": ep_p95_pos_xz,
        "episode_max_pos_xz": ep_max_pos_xz,
        "episode_p95_vel_xz": np.asarray(episode_p95_vel_xz_list, dtype=np.float64),
        "episode_max_vel_xz": np.asarray(episode_max_vel_xz_list, dtype=np.float64),
        "episode_roll_rmse_deg": ep_roll_rmse,
        "episode_pitch_rmse_deg": ep_pitch_rmse,
        "episode_yaw_rmse_deg": ep_yaw_rmse,
        "episode_pos_error_norm_mean": np.asarray(episode_pos_error_norm_mean_list, dtype=np.float64),
        "episode_vel_error_norm_mean": np.asarray(episode_vel_error_norm_mean_list, dtype=np.float64),
        "episode_att_error_norm_mean": np.asarray(episode_att_error_norm_mean_list, dtype=np.float64),
        "episode_safe_rate": np.asarray(episode_safe_rate_list, dtype=np.float64),
        "episode_ref_progress_end": np.asarray(episode_ref_progress_end_list, dtype=np.float64),
        "episode_idx": np.asarray(flat_episode_idx, dtype=np.int32),
        "seed_group_idx": np.asarray(flat_seed_group_idx, dtype=np.int32),
        "episode_in_seed_group_step": np.asarray(flat_episode_in_seed, dtype=np.int32),
        "episode_seed_step": np.asarray(flat_episode_seed, dtype=np.int64),
        "step_in_episode": np.asarray(flat_step_in_episode, dtype=np.int32),
        "ref_time_sec": np.asarray(flat_ref_time_sec, dtype=np.float64),
        "ref_progress": np.asarray(flat_ref_progress, dtype=np.float64),
        "obs": np.asarray(flat_obs, dtype=np.float64),
        "next_obs": np.asarray(flat_next_obs, dtype=np.float64),
        "act": np.asarray(flat_act_safe, dtype=np.float64),
        "act_raw": np.asarray(flat_act_raw, dtype=np.float64),
        "rew": np.asarray(flat_rew, dtype=np.float64),
        "safe": np.asarray(flat_safe, dtype=np.float64),
        "pos_error_norm": step_pos_error_norm,
        "vel_error_norm": step_vel_error_norm,
        "att_error_norm": step_att_error_norm,
        "x_error": np.asarray(flat_x_error, dtype=np.float64),
        "y_error": np.asarray(flat_y_error, dtype=np.float64),
        "z_error": np.asarray(flat_z_error, dtype=np.float64),
        "vx_error": np.asarray(flat_vx_error, dtype=np.float64),
        "vy_error": np.asarray(flat_vy_error, dtype=np.float64),
        "vz_error": np.asarray(flat_vz_error, dtype=np.float64),
        "pos_xz_error": np.asarray(flat_pos_xz_error, dtype=np.float64),
        "vel_xz_error": np.asarray(flat_vel_xz_error, dtype=np.float64),
        "roll_deg": np.asarray(flat_roll_deg, dtype=np.float64),
        "pitch_deg": np.asarray(flat_pitch_deg, dtype=np.float64),
        "yaw_deg": np.asarray(flat_yaw_deg, dtype=np.float64),
        "ref_roll_deg": np.asarray(flat_ref_roll_deg, dtype=np.float64),
        "ref_pitch_deg": np.asarray(flat_ref_pitch_deg, dtype=np.float64),
        "ref_yaw_deg": np.asarray(flat_ref_yaw_deg, dtype=np.float64),
        "roll_error_deg": np.asarray(flat_roll_error_deg, dtype=np.float64),
        "pitch_error_deg": np.asarray(flat_pitch_error_deg, dtype=np.float64),
        "yaw_error_deg": np.asarray(flat_yaw_error_deg, dtype=np.float64),
        "hard_deck_margin": np.asarray(flat_hard_deck_margin, dtype=np.float64),
        "ref_state": np.asarray(flat_ref_state, dtype=np.float64),
    }

    aux = {
        "actor_obs_dim": int(actor_obs_dim),
        "projection_enabled": bool(sac_cfg.use_projection and sac_cfg.project_actor_actions),
        "num_qp_inequalities": int(projector.num_qp_inequalities),
        "num_backup_inequalities": int(projector.num_backup_inequalities),
        "best_episode_idx": int(best_episode_idx),
    }
    return summary, arrays, aux


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
) -> dict[str, Any]:
    trace = _extract_episode_trace(arrays, best_episode_idx)
    if not trace:
        return {
            "available": False,
            "message": f"Episode trace missing for best episode {best_episode_idx}.",
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "best_episode_trace.npz"
    np.savez(trace_path, **trace)

    title = (
        f"{checkpoint} checkpoint best evaluation episode | "
        f"ep={best_episode_idx} | "
        f"score={float(np.asarray(arrays['episode_tracking_score'])[best_episode_idx]):.4f}"
    )

    plot_path = out_dir / "best_trajectory.png"
    if _HAS_MATPLOTLIB:
        plot_quad_trajectory(
            trace,
            z_max=float(env_cfg.z_max),
            output_path=str(plot_path),
            dt=float(env_cfg.dt),
        )

    gif_path: str | None = None
    try:
        gif_path = _save_best_episode_gif(
            trace,
            out_path=out_dir / "best_trajectory.gif",
            title=title,
            slowdown=float(gif_slowdown),
            trail_length=max(1, int(gif_trail_length)),
            print_every=max(1, int(gif_print_every)),
        )
    except Exception as exc:  # noqa: BLE001
        gif_path = None
        return {
            "available": True,
            "trace_path": str(trace_path),
            "plot_path": str(plot_path) if plot_path.exists() else None,
            "gif_path": None,
            "gif_error": str(exc),
        }

    return {
        "available": True,
        "trace_path": str(trace_path),
        "plot_path": str(plot_path) if plot_path.exists() else None,
        "gif_path": gif_path,
    }


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
    force_full_episode = bool(task.get("force_full_episode", True))

    cfg_json = _load_json(run_dir / "configs.json")
    sac_cfg = _dataclass_from_dict(SACConfig, cfg_json.get("sac", {}))
    env_cfg_train = _dataclass_from_dict(QuadrotorEnvConfig, cfg_json.get("env", {}))
    env_cfg_eval = _build_eval_env_cfg(env_cfg_train, force_full_episode=force_full_episode)
    cbf_cfg = _dataclass_from_dict(
        QuadrotorBCBFConfig,
        cfg_json.get("cbf", {}),
    )

    weights_path = run_dir / f"{checkpoint}_weights.pkl"
    actor_params = _load_actor_params(weights_path)

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
        env_cfg=env_cfg_eval,
        cbf_cfg=cbf_cfg,
        num_eval_seeds=num_eval_seeds,
        episodes_per_seed=episodes_per_seed,
        eval_seed_base=eval_seed_base,
        seed_group_stride=seed_group_stride,
    )

    best_episode_idx = int(summary["best_episode_idx"])
    best_episode_artifacts = _save_best_episode_artifacts(
        arrays,
        best_episode_idx=best_episode_idx,
        checkpoint=checkpoint,
        out_dir=eval_dir,
        env_cfg=env_cfg_eval,
        gif_slowdown=gif_slowdown,
        gif_trail_length=gif_trail_length,
        gif_print_every=gif_print_every,
    )

    results_npz = eval_dir / "evaluation_data.npz"
    metadata = {
        "outputs_dir": str(outputs_dir),
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "checkpoint": checkpoint,
        "weights_path": str(weights_path),
        "config_path": str(run_dir / "configs.json"),
        "eval_dir": str(eval_dir),
        "results_npz": str(results_npz),
        "eval_label": eval_label,
        "force_full_episode": bool(force_full_episode),
        "num_eval_seeds": int(num_eval_seeds),
        "episodes_per_seed": int(episodes_per_seed),
        "episodes_total": int(num_eval_seeds * episodes_per_seed),
        "eval_seed_base": int(eval_seed_base),
        "seed_group_stride": int(seed_group_stride),
        "score_formula": _score_formula_payload(),
        "saved_config": {
            "env_dt_train": float(env_cfg_train.dt),
            "env_max_steps_train": int(env_cfg_train.max_steps),
            "env_dt_eval": float(env_cfg_eval.dt),
            "env_max_steps_eval": int(env_cfg_eval.max_steps),
            "terminate_on_violation_train": bool(env_cfg_train.terminate_on_violation),
            "terminate_on_violation_eval": bool(env_cfg_eval.terminate_on_violation),
            "reward_mode": str(env_cfg_eval.reward_mode),
            "reference_path": str(env_cfg_eval.reference_path),
            "reference_dt": (None if env_cfg_eval.reference_dt is None else float(env_cfg_eval.reference_dt)),
            "z_max": float(env_cfg_eval.z_max),
            "backup_policy_mode": str(cbf_cfg.backup_policy_mode),
            "cbf_dt": float(cbf_cfg.dt),
            "cbf_horizon_T": float(cbf_cfg.horizon),
            "cbf_num_steps": int(cbf_cfg.num_steps),
            "use_projection": bool(sac_cfg.use_projection),
            "project_actor_actions": bool(sac_cfg.project_actor_actions),
            "project_target_actions": bool(sac_cfg.project_target_actions),
        },
        "hyperparams": {
            "w_pos_xy": float(getattr(env_cfg_train, "w_pos_xy", np.nan)),
            "w_pos_z": float(getattr(env_cfg_train, "w_pos_z", np.nan)),
            "w_vel": float(getattr(env_cfg_train, "w_vel", np.nan)),
            "w_att": float(getattr(env_cfg_train, "w_att", np.nan)),
            "w_control_a": float(getattr(env_cfg_train, "w_control_a", np.nan)),
            "w_control_omega": float(getattr(env_cfg_train, "w_control_omega", np.nan)),
            "z_max": float(getattr(env_cfg_train, "z_max", np.nan)),
        },
        "evaluation_runtime": {
            "actor_obs_dim": int(aux["actor_obs_dim"]),
            "projection_enabled": bool(aux["projection_enabled"]),
            "num_qp_inequalities": int(aux["num_qp_inequalities"]),
            "num_backup_inequalities": int(aux["num_backup_inequalities"]),
        },
        "best_episode_artifacts": best_episode_artifacts,
    }

    summary_payload = {
        "summary": summary,
        "metadata": metadata,
        "best_episode": {
            "episode_idx": int(best_episode_idx),
            "seed_group_idx": int(np.asarray(arrays["episode_seed_group"], dtype=np.int32)[best_episode_idx]),
            "episode_in_seed_group": int(np.asarray(arrays["episode_in_seed_group"], dtype=np.int32)[best_episode_idx]),
            "episode_seed": int(np.asarray(arrays["episode_seed"], dtype=np.int64)[best_episode_idx]),
            "tracking_score": float(np.asarray(arrays["episode_tracking_score"], dtype=np.float64)[best_episode_idx]),
            "pos_xz_rmse": float(np.asarray(arrays["episode_pos_xz_rmse"], dtype=np.float64)[best_episode_idx]),
            "vel_xz_rmse": float(np.asarray(arrays["episode_vel_xz_rmse"], dtype=np.float64)[best_episode_idx]),
            "pitch_rmse_deg": float(np.asarray(arrays["episode_pitch_rmse_deg"], dtype=np.float64)[best_episode_idx]),
            "y_rmse": float(np.asarray(arrays["episode_y_rmse"], dtype=np.float64)[best_episode_idx]),
            "vy_rmse": float(np.asarray(arrays["episode_vy_rmse"], dtype=np.float64)[best_episode_idx]),
            "artifacts": best_episode_artifacts,
        },
    }

    np.savez(
        results_npz,
        **arrays,
        summary_json=np.asarray(json.dumps(summary_payload["summary"], sort_keys=True), dtype=np.str_),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    _save_json(eval_dir / "summary.json", summary_payload)

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


def _load_eval_summary(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if "summary" not in payload or "metadata" not in payload:
        raise ValueError(f"Malformed evaluation summary: {path}")
    return payload


def _collect_latest_eval_by_run_checkpoint(
    run_dir: Path,
    *,
    checkpoints: set[str],
    analysis_label: str,
    num_eval_seeds: int,
    episodes_per_seed: int,
    eval_seed_base: int | None,
) -> dict[str, dict[str, Any]]:
    eval_root = run_dir / "evaluation"
    if not eval_root.exists():
        return {}

    chosen: dict[str, tuple[float, Path, dict[str, Any]]] = {}
    for eval_dir in sorted(eval_root.glob("quadTrack_eval-*")):
        summary_path = eval_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            payload = _load_eval_summary(summary_path)
        except Exception:  # noqa: BLE001
            continue

        metadata = payload["metadata"]
        checkpoint = str(metadata.get("checkpoint", ""))
        if checkpoint not in checkpoints:
            continue
        if int(metadata.get("num_eval_seeds", -1)) != int(num_eval_seeds):
            continue
        if int(metadata.get("episodes_per_seed", -1)) != int(episodes_per_seed):
            continue
        if eval_seed_base is not None and int(metadata.get("eval_seed_base", -1)) != int(eval_seed_base):
            continue
        if analysis_label.strip() and str(metadata.get("eval_label", "")).strip() != analysis_label.strip():
            continue

        mtime = summary_path.stat().st_mtime
        prev = chosen.get(checkpoint)
        if prev is None or mtime > prev[0]:
            chosen[checkpoint] = (mtime, summary_path, payload)

    out: dict[str, dict[str, Any]] = {}
    for checkpoint, (_, summary_path, payload) in chosen.items():
        out[checkpoint] = {
            "summary_path": str(summary_path),
            "payload": payload,
            "summary": payload["summary"],
            "metadata": payload["metadata"],
        }
    return out


def _write_analysis_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "run_name",
        "selected_checkpoint",
        "tracking_score_mean",
        "tracking_score_p95",
        "pos_xz_rmse_mean",
        "vel_xz_rmse_mean",
        "pitch_rmse_deg_mean",
        "y_rmse_mean",
        "vy_rmse_mean",
        "roll_rmse_deg_mean",
        "yaw_rmse_deg_mean",
        "p95_pos_xz_mean",
        "max_pos_xz_mean",
        "x_rmse_mean",
        "z_rmse_mean",
        "vx_rmse_mean",
        "vz_rmse_mean",
        "pos_xyz_rmse_mean",
        "vel_xyz_rmse_mean",
        "return_mean",
        "episodes_total",
        "w_pos_xy",
        "w_pos_z",
        "w_vel",
        "w_att",
        "w_control_a",
        "w_control_omega",
        "summary_path",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            metrics = row["selected_metrics"]
            hyper = row["hyperparams"]
            writer.writerow(
                {
                    "rank": rank,
                    "run_name": row["run_name"],
                    "selected_checkpoint": row["selected_checkpoint"],
                    "tracking_score_mean": _metric_value(metrics, "tracking_score_mean", np.nan),
                    "tracking_score_p95": _metric_value(metrics, "tracking_score_p95", np.nan),
                    "pos_xz_rmse_mean": _metric_value(metrics, "pos_xz_rmse_mean", np.nan),
                    "vel_xz_rmse_mean": _metric_value(metrics, "vel_xz_rmse_mean", np.nan),
                    "pitch_rmse_deg_mean": _metric_value(metrics, "pitch_rmse_deg_mean", np.nan),
                    "y_rmse_mean": _metric_value(metrics, "y_rmse_mean", np.nan),
                    "vy_rmse_mean": _metric_value(metrics, "vy_rmse_mean", np.nan),
                    "roll_rmse_deg_mean": _metric_value(metrics, "roll_rmse_deg_mean", np.nan),
                    "yaw_rmse_deg_mean": _metric_value(metrics, "yaw_rmse_deg_mean", np.nan),
                    "p95_pos_xz_mean": _metric_value(metrics, "p95_pos_xz_mean", np.nan),
                    "max_pos_xz_mean": _metric_value(metrics, "max_pos_xz_mean", np.nan),
                    "x_rmse_mean": _metric_value(metrics, "x_rmse_mean", np.nan),
                    "z_rmse_mean": _metric_value(metrics, "z_rmse_mean", np.nan),
                    "vx_rmse_mean": _metric_value(metrics, "vx_rmse_mean", np.nan),
                    "vz_rmse_mean": _metric_value(metrics, "vz_rmse_mean", np.nan),
                    "pos_xyz_rmse_mean": _metric_value(metrics, "pos_xyz_rmse_mean", np.nan),
                    "vel_xyz_rmse_mean": _metric_value(metrics, "vel_xyz_rmse_mean", np.nan),
                    "return_mean": _metric_value(metrics, "return_mean", np.nan),
                    "episodes_total": int(_metric_value(metrics, "episodes_total", 0.0)),
                    "w_pos_xy": hyper["w_pos_xy"],
                    "w_pos_z": hyper["w_pos_z"],
                    "w_vel": hyper["w_vel"],
                    "w_att": hyper["w_att"],
                    "w_control_a": hyper["w_control_a"],
                    "w_control_omega": hyper["w_control_omega"],
                    "summary_path": row["selected_summary_path"],
                }
            )


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (*_tracking_rank_key(row["selected_metrics"]), str(row["run_name"])))


def _analyze_saved_evaluations(
    outputs_dir: Path,
    run_dirs: list[Path],
    *,
    checkpoints: set[str],
    analysis_label: str,
    num_eval_seeds: int,
    episodes_per_seed: int,
    eval_seed_base: int | None,
    top_k: int,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]]]:
    ranked_candidates: list[dict[str, Any]] = []
    skipped_runs: list[dict[str, str]] = []

    for run_dir in run_dirs:
        run_name = run_dir.name
        try:
            cfg_json = _load_json(run_dir / "configs.json")
            env_cfg = _dataclass_from_dict(QuadrotorEnvConfig, cfg_json.get("env", {}))
            evals = _collect_latest_eval_by_run_checkpoint(
                run_dir,
                checkpoints=checkpoints,
                analysis_label=analysis_label,
                num_eval_seeds=num_eval_seeds,
                episodes_per_seed=episodes_per_seed,
                eval_seed_base=eval_seed_base,
            )
            if not evals:
                skipped_runs.append(
                    {
                        "run_name": run_name,
                        "reason": (
                            "No matching evaluation summaries found under run_dir/evaluation "
                            f"(label='{analysis_label}', num_eval_seeds={num_eval_seeds}, "
                            f"episodes_per_seed={episodes_per_seed}"
                            + (f", eval_seed_base={eval_seed_base}" if eval_seed_base is not None else "")
                            + ")."
                        ),
                    }
                )
                continue

            selected_checkpoint = min(evals.keys(), key=lambda ckpt: _tracking_rank_key(evals[ckpt]["summary"]))
            payload = evals[selected_checkpoint]["payload"]
            ranked_candidates.append(
                {
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "selected_checkpoint": selected_checkpoint,
                    "selected_metrics": payload["summary"],
                    "selected_summary_path": evals[selected_checkpoint]["summary_path"],
                    "checkpoint_metrics": {ck: rec["summary"] for ck, rec in evals.items()},
                    "checkpoint_summary_paths": {ck: rec["summary_path"] for ck, rec in evals.items()},
                    "hyperparams": {
                        "w_pos_xy": float(getattr(env_cfg, "w_pos_xy", np.nan)),
                        "w_pos_z": float(getattr(env_cfg, "w_pos_z", np.nan)),
                        "w_vel": float(getattr(env_cfg, "w_vel", np.nan)),
                        "w_att": float(getattr(env_cfg, "w_att", np.nan)),
                        "w_control_a": float(getattr(env_cfg, "w_control_a", np.nan)),
                        "w_control_omega": float(getattr(env_cfg, "w_control_omega", np.nan)),
                        "z_max": float(getattr(env_cfg, "z_max", np.nan)),
                    },
                    "metadata": payload["metadata"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            skipped_runs.append({"run_name": run_name, "reason": str(exc)})

    ranked = _rank_rows(ranked_candidates)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = outputs_dir / f"quadTrack_eval_analysis_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "outputs_dir": str(outputs_dir),
            "analysis_label": analysis_label,
            "num_eval_seeds": int(num_eval_seeds),
            "episodes_per_seed": int(episodes_per_seed),
            "episodes_total": int(num_eval_seeds * episodes_per_seed),
            "eval_seed_base": (None if eval_seed_base is None else int(eval_seed_base)),
            "checkpoints_considered": sorted(checkpoints),
            "score_formula": _score_formula_payload(),
        },
        "ranked_runs": ranked,
        "skipped_runs": skipped_runs,
    }
    _save_json(out_dir / "all_results.json", payload)
    _save_json(out_dir / "ranked_runs.json", ranked)
    _save_json(out_dir / "skipped_runs.json", {"skipped_runs": skipped_runs})
    _write_analysis_csv(out_dir / "ranked_runs.csv", ranked)

    best_model = ranked[0] if ranked else None
    if best_model is not None:
        _save_json(out_dir / "best_model.json", best_model)

    if top_k > 0:
        _save_json(out_dir / "top_k.json", {"top_k": ranked[: int(top_k)]})

    return out_dir, ranked_candidates, skipped_runs, ranked


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved quadrotor tracking policies on a shared seed bank and rank them by "
            "tracking-only performance."
        )
    )
    parser.add_argument("--outputs_dir", type=str, required=True, help="Outputs root containing run directories.")
    parser.add_argument("--run_glob", type=str, default=DEFAULT_RUN_GLOB, help="Glob for run directory names.")
    parser.add_argument("--experiment", type=str, default="", help="Optional single run directory name/path.")
    parser.add_argument(
        "--mode",
        type=str,
        default="evaluate_analyze",
        choices=["evaluate", "analyze", "evaluate_analyze"],
        help="evaluate: save per-run artifacts; analyze: rank existing artifacts; evaluate_analyze: do both.",
    )
    parser.add_argument("--num_eval_seeds", type=int, default=10, help="Number of seed groups to evaluate.")
    parser.add_argument(
        "--episodes_per_seed",
        type=int,
        default=100,
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
        help="Which checkpoints to evaluate/analyze for each run.",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Top-k runs to report in analysis mode.")
    parser.add_argument(
        "--parallel_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for the evaluation stage.",
    )
    parser.add_argument(
        "--eval_label",
        type=str,
        default="",
        help="Optional label embedded in saved evaluation directory names and metadata.",
    )
    parser.add_argument(
        "--analysis_label",
        type=str,
        default="",
        help="Optional label filter when analyzing saved evaluation artifacts.",
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
    parser.add_argument(
        "--respect_saved_termination",
        action="store_true",
        default=False,
        help="If set, keep terminate_on_violation exactly as saved. Default is tracking-only full episodes.",
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
    if args.top_k <= 0:
        raise ValueError(f"--top_k must be positive, got {args.top_k}")
    if args.gif_slowdown <= 0.0:
        raise ValueError(f"--gif_slowdown must be positive, got {args.gif_slowdown}")
    if args.gif_trail_length <= 0:
        raise ValueError(f"--gif_trail_length must be positive, got {args.gif_trail_length}")
    if args.gif_print_every <= 0:
        raise ValueError(f"--gif_print_every must be positive, got {args.gif_print_every}")

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

    eval_results: list[dict[str, Any]] = []
    eval_skipped: list[dict[str, str]] = []

    if args.mode in {"evaluate", "evaluate_analyze"}:
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
                        "force_full_episode": bool(not args.respect_saved_termination),
                        "tag": eval_tag,
                    }
                )

        total_episode_evals = int(args.num_eval_seeds) * int(args.episodes_per_seed) * len(tasks)
        print(
            f"Starting quad-track evaluation for {len(tasks)} tasks "
            f"({len(run_dirs)} runs x {len(checkpoints)} checkpoint(s)). "
            f"Each task uses {args.num_eval_seeds} seed groups x {args.episodes_per_seed} episodes "
            f"= {args.num_eval_seeds * args.episodes_per_seed} episodes. "
            f"Total episode evaluations scheduled: {total_episode_evals}. "
            f"workers={args.parallel_workers}."
        )

        eval_results, eval_skipped = _run_parallel_evaluation(tasks, workers=int(args.parallel_workers))
        print("Evaluation artifacts saved under each run's 'evaluation/' folder.")
        print(f"Successful eval tasks: {len(eval_results)} / {len(tasks)}")
        if eval_skipped:
            print(f"Failed eval tasks: {len(eval_skipped)}")
            print(json.dumps(eval_skipped, indent=2))

    if args.mode in {"analyze", "evaluate_analyze"}:
        analysis_label = args.analysis_label.strip() or args.eval_label.strip()
        out_dir, _, skipped_runs, ranked = _analyze_saved_evaluations(
            outputs_dir=outputs_dir,
            run_dirs=run_dirs,
            checkpoints=checkpoints,
            analysis_label=analysis_label,
            num_eval_seeds=int(args.num_eval_seeds),
            episodes_per_seed=int(args.episodes_per_seed),
            eval_seed_base=int(args.eval_seed_base),
            top_k=int(args.top_k),
        )

        print("\nTop quad-track models by tracking-only score:")
        top_rows = ranked[: int(args.top_k)]
        if not top_rows:
            print("No matching evaluated runs were found.")
        else:
            for rank, row in enumerate(top_rows, start=1):
                metrics = row["selected_metrics"]
                hyper = row["hyperparams"]
                print(
                    f"{rank:>2}. {row['run_name']} "
                    f"[ckpt={row['selected_checkpoint']}] "
                    f"score={_metric_value(metrics, 'tracking_score_mean', np.nan):.4f}, "
                    f"score_p95={_metric_value(metrics, 'tracking_score_p95', np.nan):.4f}, "
                    f"pos_xz={_metric_value(metrics, 'pos_xz_rmse_mean', np.nan):.4f}, "
                    f"vel_xz={_metric_value(metrics, 'vel_xz_rmse_mean', np.nan):.4f}, "
                    f"pitch_deg={_metric_value(metrics, 'pitch_rmse_deg_mean', np.nan):.4f}, "
                    f"w_xy={hyper['w_pos_xy']:.3g}, w_z={hyper['w_pos_z']:.3g}, "
                    f"w_vel={hyper['w_vel']:.3g}, w_att={hyper['w_att']:.3g}"
                )
            print(f"\nBest model summary saved to: {out_dir / 'best_model.json'}")
        if skipped_runs:
            print(f"Skipped runs during analysis: {len(skipped_runs)}")


if __name__ == "__main__":
    main()
