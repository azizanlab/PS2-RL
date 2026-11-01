"""PS2 policy trainer for quadrotor with CIL projection.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import pickle
import time
from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.backup_policy.quadrotor_learned_backup import load_learned_quadrotor_backup_policy
from ps2rl.cil.quadrotor_backup_cbf import (
    QuadrotorBCBFConfig,
    QuadrotorBackupCBFProjector,
    BCBFSystem,
    backup_policy_batch,
    solve_backup_cbf_qp_batch,
    solve_backup_cbf_qp_batch_with_info,
)
from ps2rl.envs.quadrotor_env import QuadrotorEnvConfig, build_quadrotor_env
from ps2rl.utils.angles import wrap_angle_np
from ps2rl.utils.policy import ActorConfig
from ps2rl.utils.networks import CriticConfig
from ps2rl.utils.quaternion import quaternion_to_euler_deg_batch_np
from ps2rl.cil.cil_policy import BCBFProjectionOps
from ps2rl.phase2_ps2.ps2_trainer_core import (
    PS2LoopState,
    PS2SystemBinding,
    SACConfig,
    build_ps2_action_fns_for,
    build_ps2_batched_action_fn_for,
    build_ps2_one_vec_step,
    build_ps2_update_fn_for,
    init_sac_state,
    make_ps2_chunk_fn_getter,
    ps2_replay_init,
    snapshot_sac_state as _snapshot_state,
    to_jnp_batch as _to_jnp_batch,
    validate_action_bounds as _validate_action_bounds,
    validate_action_scale as _validate_action_scale,
)
from ps2rl.utils.optim import adam_init
from ps2rl.utils.seed import make_prng_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_PHYS_DIM = 10


def _action_bounds_from_cbf_cfg(cbf_cfg: QuadrotorBCBFConfig) -> Tuple[jax.Array, jax.Array]:
    action_low = jnp.array(
        [cbf_cfg.a_cmd_min, -cbf_cfg.omega_max, -cbf_cfg.omega_max, -cbf_cfg.omega_max],
        dtype=jnp.float32,
    )
    action_high = jnp.array(
        [cbf_cfg.a_cmd_max, cbf_cfg.omega_max, cbf_cfg.omega_max, cbf_cfg.omega_max],
        dtype=jnp.float32,
    )
    return action_low, action_high


def _disable_backup_fallback(sac_cfg: SACConfig) -> bool:
    return (not sac_cfg.use_projection) and (not sac_cfg.project_actor_actions)


_BINDING = PS2SystemBinding(
    phys_dim=_PHYS_DIM,
    extended_qp_diagnostics=True,
    project_with_info_fn=solve_backup_cbf_qp_batch_with_info,
    project_fn=solve_backup_cbf_qp_batch,
    backup_policy_fn=backup_policy_batch,
    action_bounds_fn=lambda action_scale, cbf_cfg: _action_bounds_from_cbf_cfg(cbf_cfg),
    disable_backup_fallback_fn=_disable_backup_fallback,
)


def _make_projection_ops(cbf_cfg: QuadrotorBCBFConfig, backup_runtime: BCBFSystem | None) -> BCBFProjectionOps:
    return _BINDING.projection_ops(cbf_cfg, backup_runtime)


def _validate_learned_backup_policy_compatibility(
    cbf_cfg: QuadrotorBCBFConfig,
    *,
    expected_action_scale: np.ndarray,
    expected_action_low: np.ndarray,
    expected_action_high: np.ndarray,
) -> None:
    if cbf_cfg.backup_policy_mode.strip().lower() != "learned":
        return

    learned = load_learned_quadrotor_backup_policy(cbf_cfg.learned_backup_policy_path)
    if int(learned.actor_cfg.obs_dim) != 10:
        raise ValueError(
            "Learned quadrotor backup policy must use the raw 10D physical state; "
            f"got actor_cfg.obs_dim={learned.actor_cfg.obs_dim}."
        )
    if int(learned.actor_cfg.action_dim) != 4:
        raise ValueError(
            "Learned quadrotor backup policy must output 4D actions [a_cmd, omega_x, omega_y, omega_z]; "
            f"got actor_cfg.action_dim={learned.actor_cfg.action_dim}."
        )
    _validate_action_scale("learned_backup_policy.action_scale", learned.action_scale, expected_action_scale)
    _validate_action_bounds(
        "learned_backup_policy",
        learned.action_low,
        learned.action_high,
        expected_action_low,
        expected_action_high,
    )


def _resolve_warm_start_path(raw_path: str | Path) -> Path:
    ckpt_path = Path(raw_path).expanduser()
    if ckpt_path.is_absolute():
        candidates = [ckpt_path]
    else:
        candidates = []
        seen: set[str] = set()
        for base in (Path.cwd(), PROJECT_ROOT, PROJECT_ROOT.parent):
            candidate = (base / ckpt_path).resolve()
            candidate_key = str(candidate)
            if candidate_key not in seen:
                seen.add(candidate_key)
                candidates.append(candidate)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n".join(f"  - {candidate}" for candidate in candidates) or f"  - {ckpt_path}"
    raise FileNotFoundError(f"Warm-start checkpoint not found: {raw_path}\nTried:\n{tried}")


def _coerce_tree_like(name: str, loaded: Any, expected: Any):
    loaded_def = jax.tree_util.tree_structure(loaded)
    expected_def = jax.tree_util.tree_structure(expected)
    if loaded_def != expected_def:
        raise ValueError(f"{name} tree structure does not match the current network definition.")

    loaded_leaves = jax.tree_util.tree_leaves(loaded)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    coerced_leaves = []
    for idx, (loaded_leaf, expected_leaf) in enumerate(zip(loaded_leaves, expected_leaves)):
        loaded_arr = np.asarray(loaded_leaf)
        expected_arr = np.asarray(expected_leaf)
        if loaded_arr.shape != expected_arr.shape:
            raise ValueError(
                f"{name} leaf {idx} has shape {loaded_arr.shape}, expected {expected_arr.shape}."
            )
        coerced_leaves.append(jnp.asarray(loaded_arr, dtype=expected_arr.dtype))
    return jax.tree_util.tree_unflatten(expected_def, coerced_leaves)


def _coerce_array_like(name: str, loaded: Any, expected: jax.Array) -> jax.Array:
    loaded_arr = np.asarray(loaded)
    expected_arr = np.asarray(expected)
    if loaded_arr.shape != expected_arr.shape:
        raise ValueError(f"{name} has shape {loaded_arr.shape}, expected {expected_arr.shape}.")
    return jnp.asarray(loaded_arr, dtype=expected_arr.dtype)


def _maybe_warm_start_state(sac_cfg: SACConfig, init_state: Dict[str, Any]) -> Dict[str, Any]:
    if not sac_cfg.warm_start:
        return init_state
    if not sac_cfg.warm_start_weights.strip():
        raise ValueError("SACConfig.warm_start=True requires SACConfig.warm_start_weights to be set.")

    ckpt_path = _resolve_warm_start_path(sac_cfg.warm_start_weights)
    with open(ckpt_path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected warm-start payload type in {ckpt_path}: {type(payload)}")

    required_keys = ("actor_params", "q1_params", "q2_params")
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise KeyError(f"Warm-start checkpoint missing required keys {missing_keys}: {ckpt_path}")

    if "target_q1_params" not in payload or "target_q2_params" not in payload:
        print("Warm-start checkpoint missing target critics; initializing targets from loaded critics.")
    if "log_alpha" not in payload:
        print("Warm-start checkpoint missing log_alpha; using configured initial alpha.")

    actor_params = _coerce_tree_like("actor_params", payload["actor_params"], init_state["actor_params"])
    q1_params = _coerce_tree_like("q1_params", payload["q1_params"], init_state["q1_params"])
    q2_params = _coerce_tree_like("q2_params", payload["q2_params"], init_state["q2_params"])
    target_q1_params = _coerce_tree_like(
        "target_q1_params",
        payload.get("target_q1_params", payload["q1_params"]),
        init_state["target_q1_params"],
    )
    target_q2_params = _coerce_tree_like(
        "target_q2_params",
        payload.get("target_q2_params", payload["q2_params"]),
        init_state["target_q2_params"],
    )
    log_alpha = _coerce_array_like("log_alpha", payload.get("log_alpha", init_state["log_alpha"]), init_state["log_alpha"])

    print(f"Warm-starting SAC state from: {ckpt_path}")
    return {
        "actor_params": actor_params,
        "q1_params": q1_params,
        "q2_params": q2_params,
        "target_q1_params": target_q1_params,
        "target_q2_params": target_q2_params,
        "log_alpha": log_alpha,
        "actor_opt": adam_init(actor_params),
        "q1_opt": adam_init(q1_params),
        "q2_opt": adam_init(q2_params),
        "alpha_opt": adam_init(log_alpha),
    }


def _init_state(
    key: jax.Array,
    actor_cfg: ActorConfig,
    critic_cfg: CriticConfig,
    sac_cfg: SACConfig,
) -> Dict[str, Any]:
    state = init_sac_state(key, actor_cfg, critic_cfg, sac_cfg)
    return _maybe_warm_start_state(sac_cfg, state)


def _build_update_fn(
    sac_cfg: SACConfig,
    actor_cfg: ActorConfig,
    cbf_cfg: QuadrotorBCBFConfig,
    action_scale: jax.Array,
    backup_runtime: BCBFSystem | None = None,
):
    """Create one JITed SAC update step."""
    return build_ps2_update_fn_for(_BINDING, sac_cfg, actor_cfg, cbf_cfg, action_scale, backup_runtime)


def _build_action_fns(
    sac_cfg: SACConfig,
    actor_cfg: ActorConfig,
    cbf_cfg: QuadrotorBCBFConfig,
    action_scale: jax.Array,
    backup_runtime: BCBFSystem | None = None,
    return_solver_info: bool = False,
):
    """Build jitted training/eval policy calls."""
    return build_ps2_action_fns_for(
        _BINDING, sac_cfg, actor_cfg, cbf_cfg, action_scale, backup_runtime, return_solver_info=return_solver_info
    )


def _build_batched_action_fn(
    sac_cfg: SACConfig,
    actor_cfg: ActorConfig,
    cbf_cfg: QuadrotorBCBFConfig,
    action_scale: jax.Array,
    backup_runtime: BCBFSystem | None = None,
):
    return build_ps2_batched_action_fn_for(_BINDING, sac_cfg, actor_cfg, cbf_cfg, action_scale, backup_runtime)


_TRACKING_SCORE_TERMS: Dict[str, Tuple[float, float]] = {
    "pos_xz_rmse": (4.00, 0.25),
    "vel_xz_rmse": (3.00, 0.50),
    "pitch_rmse_deg": (2.50, 5.00),
    "p95_pos_xz": (1.50, 0.50),
    "max_pos_xz": (1.00, 1.00),
    "y_rmse": (0.75, 0.10),
    "vy_rmse": (0.50, 0.25),
    "roll_rmse_deg": (0.25, 5.00),
    "yaw_rmse_deg": (0.25, 5.00),
}


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


def _safe_percentile(x: np.ndarray, q: float) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def _tracking_score_from_metrics(metrics: Dict[str, float]) -> float:
    score = 0.0
    for key, (weight, scale) in _TRACKING_SCORE_TERMS.items():
        value = float(metrics.get(key, float("inf")))
        if not np.isfinite(value):
            return float("inf")
        score += weight * value / scale
    return float(score)


_EVAL_SUMMARY_KEYS: Tuple[str, ...] = (
    "return_mean",
    "return_std",
    "safe_rate",
    "violation_free_episode_rate",
    "pos_error_norm_mean",
    "vel_error_norm_mean",
    "att_error_norm_mean",
    "omega_ref_error_norm_mean",
    "hard_deck_margin_min",
    "tracking_score_mean",
    "tracking_score_p95",
    "x_rmse_mean",
    "pos_xz_rmse_mean",
    "z_rmse_mean",
    "pos_xyz_rmse_mean",
    "vel_xz_rmse_mean",
    "pitch_rmse_deg_mean",
    "y_rmse_mean",
    "vy_rmse_mean",
    "p95_pos_xyz_mean",
    "max_pos_xyz_mean",
    "p95_pos_xz_mean",
    "max_pos_xz_mean",
)


def _compact_eval_stats(eval_stats: Dict[str, Any]) -> Dict[str, float]:
    return {key: float(eval_stats[key]) for key in _EVAL_SUMMARY_KEYS if key in eval_stats}


def _add_eval_summary(summary: Dict[str, Any], prefix: str, eval_stats: Dict[str, Any]) -> None:
    for key in _EVAL_SUMMARY_KEYS:
        if key in eval_stats:
            summary[f"{prefix}_{key}"] = float(eval_stats[key])

def _evaluate_policy(
    env_cfg: QuadrotorEnvConfig,
    eval_action_fn,
    actor_params,
    seed: int,
    episodes: int,
) -> Dict[str, Any]:
    """Roll out eval episodes on the JAX env and aggregate tracking metrics."""
    env_fns = build_quadrotor_env(env_cfg)
    base_key = make_prng_key(seed + 777)
    returns = []
    pos_errors = []
    vel_errors = []
    att_errors = []
    omega_ref_errors = []
    safes = []
    episode_violation_free = []
    hard_deck_margins = []
    episode_tracking_scores = []
    episode_x_rmse = []
    episode_pos_xz_rmse = []
    episode_z_rmse = []
    episode_pos_xyz_rmse = []
    episode_vel_xz_rmse = []
    episode_pitch_rmse_deg = []
    episode_y_rmse = []
    episode_vy_rmse = []
    episode_p95_pos_xyz = []
    episode_max_pos_xyz = []
    episode_p95_pos_xz = []
    episode_max_pos_xz = []
    trajectory = {
        "obs": [],
        "next_obs": [],
        "act": [],
        "rew": [],
        "safe": [],
        "pos_error_norm": [],
        "vel_error_norm": [],
        "att_error_norm": [],
        "omega_ref_error_norm": [],
        "hard_deck_margin": [],
        "ref_time_sec": [],
        "ref_progress": [],
        "ref_px": [],
        "ref_py": [],
        "ref_pz": [],
        "ref_vx": [],
        "ref_vy": [],
        "ref_vz": [],
        "ref_qw": [],
        "ref_qx": [],
        "ref_qy": [],
        "ref_qz": [],
        "ref_omega_x": [],
        "ref_omega_y": [],
        "ref_omega_z": [],
    }
    for ep in range(episodes):
        key_ep = jax.random.fold_in(base_key, 10_000 + ep)
        state, obs = env_fns.reset(key_ep)
        done = False
        ep_ret = 0.0
        step_idx = 0
        ep_all_safe = True
        ep_pos_xz_err = []
        ep_pos_xyz_err = []
        ep_vel_xz_err = []
        ep_pitch_err_deg = []
        ep_roll_err_deg = []
        ep_yaw_err_deg = []
        ep_x_err = []
        ep_y_err = []
        ep_z_err = []
        ep_vy_err = []
        ep_pos_x = []
        ep_pos_z = []
        ep_quat = []
        ep_ref_x = []
        ep_ref_z = []
        while not done:
            obs_j = jnp.asarray(obs, dtype=jnp.float32)
            safe_action, raw_action = eval_action_fn(actor_params, obs_j)
            _ = raw_action
            # The auto-reset key is unused: the loop breaks on done, and
            # next_obs_true (pre-reset) is what the metrics consume.
            key_step = jax.random.fold_in(key_ep, step_idx)
            state, next_obs_true, _, rew, done_j, info = env_fns.step(state, safe_action, key_step)
            done = bool(done_j)
            step_idx += 1
            act_np = np.asarray(safe_action, dtype=np.float64)
            next_obs = np.asarray(next_obs_true, dtype=np.float64)
            rew = float(rew)
            ep_ret += rew
            pos_err = float(info.pos_error_norm)
            vel_err = float(info.vel_error_norm)
            att_err = float(info.att_error_norm)
            omega_ref_err = float(info.omega_ref_error_norm)
            margin = float(info.hard_deck_margin)
            is_safe = float(info.is_safe)
            pos_errors.append(pos_err)
            vel_errors.append(vel_err)
            att_errors.append(att_err)
            omega_ref_errors.append(omega_ref_err)
            safes.append(is_safe)
            ep_all_safe = ep_all_safe and (is_safe >= 0.5)
            hard_deck_margins.append(margin)

            x_next = np.asarray(next_obs[:10], dtype=np.float64)
            ref_state = np.asarray(info.ref_state, dtype=np.float64)
            ref_omega = np.asarray(info.ref_omega, dtype=np.float64)
            x_err = float(x_next[0] - ref_state[0])
            y_err = float(x_next[1] - ref_state[1])
            z_err = float(x_next[2] - ref_state[2])
            vx_err = float(x_next[3] - ref_state[3])
            vy_err = float(x_next[4] - ref_state[4])
            vz_err = float(x_next[5] - ref_state[5])
            pos_xz_err = float(np.sqrt(x_err * x_err + z_err * z_err))
            pos_xyz_err = float(np.sqrt(x_err * x_err + y_err * y_err + z_err * z_err))
            vel_xz_err = float(np.sqrt(vx_err * vx_err + vz_err * vz_err))
            roll_now, pitch_now, yaw_now = quaternion_to_euler_deg_batch_np(x_next[6:10][None, :])
            roll_ref, pitch_ref, yaw_ref = quaternion_to_euler_deg_batch_np(ref_state[6:10][None, :])
            roll_err_deg = float(np.rad2deg(wrap_angle_np(np.deg2rad(roll_now[0] - roll_ref[0]))))
            pitch_err_deg = float(np.rad2deg(wrap_angle_np(np.deg2rad(pitch_now[0] - pitch_ref[0]))))
            yaw_err_deg = float(np.rad2deg(wrap_angle_np(np.deg2rad(yaw_now[0] - yaw_ref[0]))))

            ep_pos_xz_err.append(pos_xz_err)
            ep_pos_xyz_err.append(pos_xyz_err)
            ep_vel_xz_err.append(vel_xz_err)
            ep_pitch_err_deg.append(pitch_err_deg)
            ep_roll_err_deg.append(roll_err_deg)
            ep_yaw_err_deg.append(yaw_err_deg)
            ep_x_err.append(x_err)
            ep_y_err.append(y_err)
            ep_z_err.append(z_err)
            ep_vy_err.append(vy_err)
            ep_pos_x.append(float(x_next[0]))
            ep_pos_z.append(float(x_next[2]))
            ep_quat.append(np.asarray(x_next[6:10], dtype=np.float64))
            ep_ref_x.append(float(ref_state[0]))
            ep_ref_z.append(float(ref_state[2]))

            if ep == 0:
                trajectory["obs"].append(np.asarray(obs, dtype=np.float64))
                trajectory["next_obs"].append(next_obs)
                trajectory["act"].append(act_np)
                trajectory["rew"].append(rew)
                trajectory["safe"].append(is_safe)
                trajectory["pos_error_norm"].append(pos_err)
                trajectory["vel_error_norm"].append(vel_err)
                trajectory["att_error_norm"].append(att_err)
                trajectory["omega_ref_error_norm"].append(omega_ref_err)
                trajectory["hard_deck_margin"].append(margin)
                trajectory["ref_time_sec"].append(float(info.ref_time_sec))
                trajectory["ref_progress"].append(float(info.ref_progress))
                trajectory["ref_px"].append(float(ref_state[0]))
                trajectory["ref_py"].append(float(ref_state[1]))
                trajectory["ref_pz"].append(float(ref_state[2]))
                trajectory["ref_vx"].append(float(ref_state[3]))
                trajectory["ref_vy"].append(float(ref_state[4]))
                trajectory["ref_vz"].append(float(ref_state[5]))
                trajectory["ref_qw"].append(float(ref_state[6]))
                trajectory["ref_qx"].append(float(ref_state[7]))
                trajectory["ref_qy"].append(float(ref_state[8]))
                trajectory["ref_qz"].append(float(ref_state[9]))
                trajectory["ref_omega_x"].append(float(ref_omega[0]))
                trajectory["ref_omega_y"].append(float(ref_omega[1]))
                trajectory["ref_omega_z"].append(float(ref_omega[2]))
            obs = next_obs_true
        returns.append(ep_ret)
        episode_violation_free.append(1.0 if ep_all_safe else 0.0)

        ep_metrics = {
            "x_rmse": _rms(np.asarray(ep_x_err, dtype=np.float64)),
            "pos_xz_rmse": _rms(np.asarray(ep_pos_xz_err, dtype=np.float64)),
            "z_rmse": _rms(np.asarray(ep_z_err, dtype=np.float64)),
            "pos_xyz_rmse": _rms(np.asarray(ep_pos_xyz_err, dtype=np.float64)),
            "vel_xz_rmse": _rms(np.asarray(ep_vel_xz_err, dtype=np.float64)),
            "pitch_rmse_deg": _rms(np.asarray(ep_pitch_err_deg, dtype=np.float64)),
            "p95_pos_xyz": _safe_percentile(np.asarray(ep_pos_xyz_err, dtype=np.float64), 95.0),
            "max_pos_xyz": float(np.max(ep_pos_xyz_err)) if ep_pos_xyz_err else 0.0,
            "p95_pos_xz": _safe_percentile(np.asarray(ep_pos_xz_err, dtype=np.float64), 95.0),
            "max_pos_xz": float(np.max(ep_pos_xz_err)) if ep_pos_xz_err else 0.0,
            "y_rmse": _rms(np.asarray(ep_y_err, dtype=np.float64)),
            "vy_rmse": _rms(np.asarray(ep_vy_err, dtype=np.float64)),
            "roll_rmse_deg": _rms(np.asarray(ep_roll_err_deg, dtype=np.float64)),
            "yaw_rmse_deg": _rms(np.asarray(ep_yaw_err_deg, dtype=np.float64)),
        }
        ep_metrics["tracking_score"] = _tracking_score_from_metrics(ep_metrics)
        episode_tracking_scores.append(float(ep_metrics["tracking_score"]))
        episode_x_rmse.append(float(ep_metrics["x_rmse"]))
        episode_pos_xz_rmse.append(float(ep_metrics["pos_xz_rmse"]))
        episode_z_rmse.append(float(ep_metrics["z_rmse"]))
        episode_pos_xyz_rmse.append(float(ep_metrics["pos_xyz_rmse"]))
        episode_vel_xz_rmse.append(float(ep_metrics["vel_xz_rmse"]))
        episode_pitch_rmse_deg.append(float(ep_metrics["pitch_rmse_deg"]))
        episode_y_rmse.append(float(ep_metrics["y_rmse"]))
        episode_vy_rmse.append(float(ep_metrics["vy_rmse"]))
        episode_p95_pos_xyz.append(float(ep_metrics["p95_pos_xyz"]))
        episode_max_pos_xyz.append(float(ep_metrics["max_pos_xyz"]))
        episode_p95_pos_xz.append(float(ep_metrics["p95_pos_xz"]))
        episode_max_pos_xz.append(float(ep_metrics["max_pos_xz"]))
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "safe_rate": float(np.mean(safes)) if safes else 1.0,
        "violation_free_episode_rate": float(np.mean(episode_violation_free)) if episode_violation_free else 1.0,
        "pos_error_norm_mean": float(np.mean(pos_errors)) if pos_errors else 0.0,
        "vel_error_norm_mean": float(np.mean(vel_errors)) if vel_errors else 0.0,
        "att_error_norm_mean": float(np.mean(att_errors)) if att_errors else 0.0,
        "omega_ref_error_norm_mean": float(np.mean(omega_ref_errors)) if omega_ref_errors else 0.0,
        "hard_deck_margin_min": float(np.min(hard_deck_margins)) if hard_deck_margins else 0.0,
        "tracking_score_mean": _safe_mean(np.asarray(episode_tracking_scores, dtype=np.float64)),
        "tracking_score_p95": _safe_percentile(np.asarray(episode_tracking_scores, dtype=np.float64), 95.0),
        "x_rmse_mean": _safe_mean(np.asarray(episode_x_rmse, dtype=np.float64)),
        "pos_xz_rmse_mean": _safe_mean(np.asarray(episode_pos_xz_rmse, dtype=np.float64)),
        "z_rmse_mean": _safe_mean(np.asarray(episode_z_rmse, dtype=np.float64)),
        "pos_xyz_rmse_mean": _safe_mean(np.asarray(episode_pos_xyz_rmse, dtype=np.float64)),
        "vel_xz_rmse_mean": _safe_mean(np.asarray(episode_vel_xz_rmse, dtype=np.float64)),
        "pitch_rmse_deg_mean": _safe_mean(np.asarray(episode_pitch_rmse_deg, dtype=np.float64)),
        "y_rmse_mean": _safe_mean(np.asarray(episode_y_rmse, dtype=np.float64)),
        "vy_rmse_mean": _safe_mean(np.asarray(episode_vy_rmse, dtype=np.float64)),
        "p95_pos_xyz_mean": _safe_mean(np.asarray(episode_p95_pos_xyz, dtype=np.float64)),
        "max_pos_xyz_mean": _safe_mean(np.asarray(episode_max_pos_xyz, dtype=np.float64)),
        "p95_pos_xz_mean": _safe_mean(np.asarray(episode_p95_pos_xz, dtype=np.float64)),
        "max_pos_xz_mean": _safe_mean(np.asarray(episode_max_pos_xz, dtype=np.float64)),
        "trajectory": {k: np.asarray(v) for k, v in trajectory.items()},
    }


def _eval_selection_key_return(eval_stats: Dict[str, Any]) -> Tuple[float, ...]:
    """Safety-first, then maximize evaluation return; lower is better."""
    return (
        -float(eval_stats.get("violation_free_episode_rate", 0.0)),
        -float(eval_stats.get("safe_rate", 0.0)),
        -float(eval_stats.get("return_mean", -float("inf"))),
    )


def _save_best_weight_snapshots(
    output_dir: str | None,
    *,
    best_state_return: Dict[str, Any] | None,
) -> None:
    """Save the return-selected best weights as ``best_weights.pkl`` (the only
    checkpoint the quad Phase-2 evaluator loads; matches the unicycle)."""
    if output_dir is None or best_state_return is None:
        return
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "best_weights.pkl", "wb") as f:
        pickle.dump(best_state_return, f, protocol=pickle.HIGHEST_PROTOCOL)


def _maybe_save_periodic_best_weights(
    output_dir: str | None,
    *,
    step: int,
    next_save_step: int,
    save_period: int,
    best_state_return: Dict[str, Any] | None,
) -> int:
    if output_dir is None:
        return next_save_step

    period = int(save_period)
    if period <= 0:
        return next_save_step

    step_i = int(step)
    next_step_i = int(next_save_step)
    if next_step_i <= 0:
        next_step_i = period
    if step_i < next_step_i:
        return next_step_i

    if best_state_return is None:
        return next_step_i

    _save_best_weight_snapshots(output_dir, best_state_return=best_state_return)

    while next_step_i <= step_i:
        next_step_i += period
    return next_step_i


_UPDATE_METRIC_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("critic_loss_sum", "critic_loss"),
    ("actor_loss_sum", "actor_loss"),
    ("alpha_sum", "alpha"),
    ("q_pi_sum", "q_pi_mean"),
    ("target_use_solver_sum", "target_use_solver_rate"),
    ("actor_use_solver_sum", "actor_use_solver_rate"),
    ("target_q_mat_finite_sum", "target_q_mat_finite_rate"),
    ("target_q_vec_finite_sum", "target_q_vec_finite_rate"),
    ("target_g_finite_sum", "target_g_finite_rate"),
    ("target_h_finite_sum", "target_h_finite_rate"),
    ("target_inputs_finite_sum", "target_inputs_finite_rate"),
    ("target_z_finite_sum", "target_z_finite_rate"),
    ("target_q_saturated_sum", "target_q_saturated_rate"),
    ("target_max_abs_q_sum", "target_max_abs_q_mean"),
    ("target_max_abs_b_sum", "target_max_abs_b_mean"),
    ("target_delta_min_u_ref_sum", "target_delta_min_u_ref_mean"),
    ("actor_q_mat_finite_sum", "actor_q_mat_finite_rate"),
    ("actor_q_vec_finite_sum", "actor_q_vec_finite_rate"),
    ("actor_g_finite_sum", "actor_g_finite_rate"),
    ("actor_h_finite_sum", "actor_h_finite_rate"),
    ("actor_inputs_finite_sum", "actor_inputs_finite_rate"),
    ("actor_z_finite_sum", "actor_z_finite_rate"),
    ("actor_q_saturated_sum", "actor_q_saturated_rate"),
    ("actor_max_abs_q_sum", "actor_max_abs_q_mean"),
    ("actor_max_abs_b_sum", "actor_max_abs_b_mean"),
    ("actor_delta_min_u_ref_sum", "actor_delta_min_u_ref_mean"),
)

_EPISODE_FIELD_NAMES: Tuple[str, ...] = (
    "episode_return_sum",
    "episode_len_sum",
    "episode_safe_rate_sum",
    "episode_pos_error_norm_sum",
    "episode_vel_error_norm_sum",
    "episode_att_error_norm_sum",
    "episode_hard_deck_margin_min_sum",
)


def _episode_fields(info: Any) -> Dict[str, jax.Array]:
    return {
        "episode_return_sum": jnp.sum(info.completed_return),
        "episode_len_sum": jnp.sum(info.completed_len),
        "episode_safe_rate_sum": jnp.sum(info.completed_safe_rate),
        "episode_pos_error_norm_sum": jnp.sum(info.completed_pos_error_norm),
        "episode_vel_error_norm_sum": jnp.sum(info.completed_vel_error_norm),
        "episode_att_error_norm_sum": jnp.sum(info.completed_att_error_norm),
        "episode_hard_deck_margin_min_sum": jnp.sum(info.completed_hard_deck_margin_min),
    }


def _run_training_jax(
    sac_cfg: SACConfig,
    env_cfg: QuadrotorEnvConfig,
    cbf_cfg: QuadrotorBCBFConfig,
    output_dir: str | None = None,
    metric_logger: Callable[[int, Dict[str, float]], None] | None = None,
) -> Dict[str, Any]:
    """Train safe SAC using pure-JAX batched environment stepping."""
    if int(sac_cfg.num_envs) <= 0:
        raise ValueError(f"num_envs must be positive, got {sac_cfg.num_envs}")
    if int(sac_cfg.steps_per_jit) <= 0:
        raise ValueError(f"steps_per_jit must be positive, got {sac_cfg.steps_per_jit}")

    num_envs = int(sac_cfg.num_envs)
    steps_per_jit = int(sac_cfg.steps_per_jit)
    total_steps = int(sac_cfg.total_steps)
    total_vec_steps = (total_steps + num_envs - 1) // num_envs

    jax_key = make_prng_key(sac_cfg.seed)
    env_fns = build_quadrotor_env(env_cfg)

    action_scale_np = np.array(
        [cbf_cfg.a_cmd_max, cbf_cfg.omega_max, cbf_cfg.omega_max, cbf_cfg.omega_max],
        dtype=np.float64,
    )
    action_scale = jnp.asarray(action_scale_np, dtype=jnp.float32)
    action_low_np = np.array(
        [cbf_cfg.a_cmd_min, -cbf_cfg.omega_max, -cbf_cfg.omega_max, -cbf_cfg.omega_max],
        dtype=np.float64,
    )
    action_high_np = np.array(
        [cbf_cfg.a_cmd_max, cbf_cfg.omega_max, cbf_cfg.omega_max, cbf_cfg.omega_max],
        dtype=np.float64,
    )
    expected_action_scale = np.asarray(jax.device_get(cbf_cfg.action_scale), dtype=np.float64)
    _validate_action_scale("jax_env.action_scale", action_scale_np, expected_action_scale)
    _validate_learned_backup_policy_compatibility(
        cbf_cfg,
        expected_action_scale=expected_action_scale,
        expected_action_low=action_low_np,
        expected_action_high=action_high_np,
    )

    actor_cfg = ActorConfig(
        obs_dim=env_fns.obs_dim,
        action_dim=env_fns.action_dim,
        hidden_sizes=(sac_cfg.hidden_size, sac_cfg.hidden_size),
    )
    critic_cfg = CriticConfig(
        obs_dim=env_fns.obs_dim,
        act_dim=env_fns.action_dim,
        hidden_sizes=(sac_cfg.hidden_size, sac_cfg.hidden_size),
    )

    jax_key, key_state, key_env = jax.random.split(jax_key, 3)
    state = _init_state(key_state, actor_cfg, critic_cfg, sac_cfg)
    replay = ps2_replay_init(sac_cfg.replay_size, env_fns.obs_dim, env_fns.action_dim)
    projector = QuadrotorBackupCBFProjector(cbf_cfg)
    backup_runtime = projector.runtime
    proj_ops = _make_projection_ops(cbf_cfg, backup_runtime)

    update_fn = _build_update_fn(sac_cfg, actor_cfg, cbf_cfg, action_scale, backup_runtime=backup_runtime)
    sample_action_batch_fn, action_low, action_high = _build_batched_action_fn(
        sac_cfg,
        actor_cfg,
        cbf_cfg,
        action_scale,
        backup_runtime=backup_runtime,
    )
    _, eval_action_fn = _build_action_fns(
        sac_cfg,
        actor_cfg,
        cbf_cfg,
        action_scale,
        backup_runtime=backup_runtime,
    )

    env_keys = jax.random.split(key_env, num_envs)
    env_state, obs = env_fns.reset_batched(env_keys)

    loop_state = PS2LoopState(
        state=state,
        replay=replay,
        env_state=env_state,
        obs=obs,
        key=jax_key,
        env_keys=env_keys,
        global_step=jnp.int32(0),
        updates=jnp.int32(0),
    )

    one_vec_step = build_ps2_one_vec_step(
        env_fns=env_fns,
        sample_action_batch_fn=sample_action_batch_fn,
        update_fn=update_fn,
        sac_cfg=sac_cfg,
        action_low=action_low,
        action_high=action_high,
        proj=proj_ops,
        phys_dim=_PHYS_DIM,
        # The quad collector does NOT backup-filter random warm-up actions
        # (GT behavior; the uni collector differs).
        sanitize_random_actions=False,
        update_metric_pairs=_UPDATE_METRIC_PAIRS,
        episode_fields_fn=_episode_fields,
        episode_field_names=_EPISODE_FIELD_NAMES,
    )
    get_chunk_fn = make_ps2_chunk_fn_getter(one_vec_step)

    history: Dict[str, List[float]] = {
        "step": [],
        "ep_return": [],
        "ep_len": [],
        "ep_safe_rate": [],
        "ep_pos_error_norm": [],
        "ep_vel_error_norm": [],
        "ep_att_error_norm": [],
        "ep_hard_deck_margin_min": [],
        "critic_loss": [],
        "actor_loss": [],
        "alpha": [],
        "q1_grad_norm": [],
        "q2_grad_norm": [],
        "actor_grad_norm": [],
        "target_action_bad_rate": [],
        "actor_action_bad_rate": [],
        "target_use_solver_rate": [],
        "actor_use_solver_rate": [],
        "target_fallback_rate": [],
        "actor_fallback_rate": [],
        "target_q_mat_finite_rate": [],
        "target_q_vec_finite_rate": [],
        "target_g_finite_rate": [],
        "target_h_finite_rate": [],
        "target_z_finite_rate": [],
        "actor_q_mat_finite_rate": [],
        "actor_q_vec_finite_rate": [],
        "actor_g_finite_rate": [],
        "actor_h_finite_rate": [],
        "actor_z_finite_rate": [],
        "eval_return_mean": [],
        "eval_safe_rate": [],
        "eval_violation_free_episode_rate": [],
        "eval_pos_error_norm_mean": [],
        "eval_vel_error_norm_mean": [],
        "eval_att_error_norm_mean": [],
        "eval_omega_ref_error_norm_mean": [],
        "eval_hard_deck_margin_min": [],
        "eval_tracking_score_mean": [],
        "eval_tracking_score_p95": [],
        "eval_x_rmse_mean": [],
        "eval_pos_xz_rmse_mean": [],
        "eval_z_rmse_mean": [],
        "eval_pos_xyz_rmse_mean": [],
        "eval_vel_xz_rmse_mean": [],
        "eval_pitch_rmse_deg_mean": [],
        "eval_y_rmse_mean": [],
        "eval_vy_rmse_mean": [],
        "eval_p95_pos_xyz_mean": [],
        "eval_max_pos_xyz_mean": [],
    }

    latest_episode_metrics: Dict[str, float] | None = None
    latest_eval_metrics: Dict[str, float] | None = None
    best_eval_stats_return: Dict[str, float] | None = None
    best_eval_step_return = 0
    best_eval_key_return: Tuple[float, ...] | None = None
    best_state_return: Dict[str, Any] | None = None

    t0 = time.time()
    last_log_t = t0
    next_eval_step = int(sac_cfg.eval_every)
    next_log_step = int(sac_cfg.log_every)
    next_best_weights_save_step = int(sac_cfg.best_weights_save_period)

    vec_done = 0
    while vec_done < total_vec_steps:
        this_vec = min(steps_per_jit, total_vec_steps - vec_done)
        loop_state, chunk = get_chunk_fn(this_vec)(loop_state)
        vec_done += this_vec

        chunk_host = jax.device_get(chunk)
        step_now = int(min(total_steps, int(chunk_host["global_step"])))
        upd_count = float(chunk_host["update_count"])
        ep_count = float(chunk_host["episode_count"])

        if upd_count > 0.0:
            inv_upd = 1.0 / max(1.0, upd_count)
            history["critic_loss"].append(float(chunk_host["critic_loss_sum"] * inv_upd))
            history["actor_loss"].append(float(chunk_host["actor_loss_sum"] * inv_upd))
            history["alpha"].append(float(chunk_host["alpha_sum"] * inv_upd))
            history["target_use_solver_rate"].append(float(chunk_host["target_use_solver_sum"] * inv_upd))
            history["actor_use_solver_rate"].append(float(chunk_host["actor_use_solver_sum"] * inv_upd))

        if ep_count > 0.0:
            inv_ep = 1.0 / max(1.0, ep_count)
            ep_return_mean = float(chunk_host["episode_return_sum"] * inv_ep)
            ep_len_mean = float(chunk_host["episode_len_sum"] * inv_ep)
            ep_safe_rate = float(chunk_host["episode_safe_rate_sum"] * inv_ep)
            ep_pos_error = float(chunk_host["episode_pos_error_norm_sum"] * inv_ep)
            ep_vel_error = float(chunk_host["episode_vel_error_norm_sum"] * inv_ep)
            ep_att_error = float(chunk_host["episode_att_error_norm_sum"] * inv_ep)
            ep_hard_deck_margin_min = float(chunk_host["episode_hard_deck_margin_min_sum"] * inv_ep)
            history["step"].append(float(step_now))
            history["ep_return"].append(ep_return_mean)
            history["ep_len"].append(ep_len_mean)
            history["ep_safe_rate"].append(ep_safe_rate)
            history["ep_pos_error_norm"].append(ep_pos_error)
            history["ep_vel_error_norm"].append(ep_vel_error)
            history["ep_att_error_norm"].append(ep_att_error)
            history["ep_hard_deck_margin_min"].append(ep_hard_deck_margin_min)
            latest_episode_metrics = {
                "last_step": float(step_now),
                "return": ep_return_mean,
                "len": ep_len_mean,
                "safe_rate": ep_safe_rate,
                "pos_error_norm": ep_pos_error,
                "vel_error_norm": ep_vel_error,
                "att_error_norm": ep_att_error,
                "hard_deck_margin_min": ep_hard_deck_margin_min,
            }

        while step_now >= next_eval_step and next_eval_step <= total_steps:
            eval_stats = _evaluate_policy(
                env_cfg,
                eval_action_fn,
                loop_state.state["actor_params"],
                sac_cfg.seed + next_eval_step,
                sac_cfg.eval_episodes,
            )
            history["eval_return_mean"].append(eval_stats["return_mean"])
            history["eval_safe_rate"].append(eval_stats["safe_rate"])
            history["eval_violation_free_episode_rate"].append(eval_stats["violation_free_episode_rate"])
            history["eval_pos_error_norm_mean"].append(eval_stats["pos_error_norm_mean"])
            history["eval_vel_error_norm_mean"].append(eval_stats["vel_error_norm_mean"])
            history["eval_att_error_norm_mean"].append(eval_stats["att_error_norm_mean"])
            history["eval_omega_ref_error_norm_mean"].append(eval_stats["omega_ref_error_norm_mean"])
            history["eval_hard_deck_margin_min"].append(eval_stats["hard_deck_margin_min"])
            history["eval_tracking_score_mean"].append(eval_stats["tracking_score_mean"])
            history["eval_tracking_score_p95"].append(eval_stats["tracking_score_p95"])
            history["eval_x_rmse_mean"].append(eval_stats["x_rmse_mean"])
            history["eval_pos_xz_rmse_mean"].append(eval_stats["pos_xz_rmse_mean"])
            history["eval_z_rmse_mean"].append(eval_stats["z_rmse_mean"])
            history["eval_pos_xyz_rmse_mean"].append(eval_stats["pos_xyz_rmse_mean"])
            history["eval_vel_xz_rmse_mean"].append(eval_stats["vel_xz_rmse_mean"])
            history["eval_pitch_rmse_deg_mean"].append(eval_stats["pitch_rmse_deg_mean"])
            history["eval_y_rmse_mean"].append(eval_stats["y_rmse_mean"])
            history["eval_vy_rmse_mean"].append(eval_stats["vy_rmse_mean"])
            history["eval_p95_pos_xyz_mean"].append(eval_stats["p95_pos_xyz_mean"])
            history["eval_max_pos_xyz_mean"].append(eval_stats["max_pos_xyz_mean"])
            latest_eval_metrics = {
                "last_step": float(next_eval_step),
                "return_mean": float(eval_stats["return_mean"]),
                "return_std": float(eval_stats["return_std"]),
                "safe_rate": float(eval_stats["safe_rate"]),
                "violation_free_episode_rate": float(eval_stats["violation_free_episode_rate"]),
                "pos_error_norm_mean": float(eval_stats["pos_error_norm_mean"]),
                "vel_error_norm_mean": float(eval_stats["vel_error_norm_mean"]),
                "att_error_norm_mean": float(eval_stats["att_error_norm_mean"]),
                "omega_ref_error_norm_mean": float(eval_stats["omega_ref_error_norm_mean"]),
                "hard_deck_margin_min": float(eval_stats["hard_deck_margin_min"]),
                "tracking_score_mean": float(eval_stats["tracking_score_mean"]),
                "tracking_score_p95": float(eval_stats["tracking_score_p95"]),
                "x_rmse_mean": float(eval_stats["x_rmse_mean"]),
                "pos_xz_rmse_mean": float(eval_stats["pos_xz_rmse_mean"]),
                "z_rmse_mean": float(eval_stats["z_rmse_mean"]),
                "pos_xyz_rmse_mean": float(eval_stats["pos_xyz_rmse_mean"]),
                "vel_xz_rmse_mean": float(eval_stats["vel_xz_rmse_mean"]),
                "pitch_rmse_deg_mean": float(eval_stats["pitch_rmse_deg_mean"]),
                "y_rmse_mean": float(eval_stats["y_rmse_mean"]),
                "vy_rmse_mean": float(eval_stats["vy_rmse_mean"]),
                "p95_pos_xyz_mean": float(eval_stats["p95_pos_xyz_mean"]),
                "max_pos_xyz_mean": float(eval_stats["max_pos_xyz_mean"]),
            }
            eval_key_return = _eval_selection_key_return(eval_stats)
            improve_return = (best_eval_key_return is None) or (eval_key_return < best_eval_key_return)
            if improve_return:
                best_eval_key_return = eval_key_return
                best_eval_step_return = next_eval_step
                best_eval_stats_return = _compact_eval_stats(eval_stats)
                best_state_return = _snapshot_state(loop_state.state)
            next_eval_step += int(sac_cfg.eval_every)

        next_best_weights_save_step = _maybe_save_periodic_best_weights(
            output_dir,
            step=step_now,
            next_save_step=next_best_weights_save_step,
            save_period=sac_cfg.best_weights_save_period,
            best_state_return=best_state_return,
        )

        while step_now >= next_log_step and next_log_step <= total_steps:
            now = time.time()
            step_rate = float(max(1, sac_cfg.log_every) / max(1e-6, now - last_log_t))
            last_log_t = now
            live_metrics: Dict[str, float] = {
                "perf/steps_per_sec": step_rate,
                "train/replay_size": float(jax.device_get(loop_state.replay.size)),
                "train/updates": float(jax.device_get(loop_state.updates)),
                "cbf/num_qp_inequalities": float(projector.num_qp_inequalities),
                "cbf/num_backup_inequalities": float(projector.num_backup_inequalities),
                "env/dt": float(env_cfg.dt),
                "cbf/dt": float(cbf_cfg.dt),
                "cbf/horizon_T": float(cbf_cfg.horizon),
                "cbf/N_steps": float(cbf_cfg.num_steps),
                "train/num_envs": float(num_envs),
            }
            if upd_count > 0.0:
                inv_upd = 1.0 / max(1.0, upd_count)
                live_metrics.update(
                    {
                        "train/critic_loss": float(chunk_host["critic_loss_sum"] * inv_upd),
                        "train/actor_loss": float(chunk_host["actor_loss_sum"] * inv_upd),
                        "train/alpha": float(chunk_host["alpha_sum"] * inv_upd),
                        "train/q_pi_mean": float(chunk_host["q_pi_sum"] * inv_upd),
                        "train/target_use_solver_rate": float(chunk_host["target_use_solver_sum"] * inv_upd),
                        "train/actor_use_solver_rate": float(chunk_host["actor_use_solver_sum"] * inv_upd),
                        "train/target_q_mat_finite_rate": float(chunk_host["target_q_mat_finite_sum"] * inv_upd),
                        "train/target_q_vec_finite_rate": float(chunk_host["target_q_vec_finite_sum"] * inv_upd),
                        "train/target_g_finite_rate": float(chunk_host["target_g_finite_sum"] * inv_upd),
                        "train/target_h_finite_rate": float(chunk_host["target_h_finite_sum"] * inv_upd),
                        "train/target_inputs_finite_rate": float(chunk_host["target_inputs_finite_sum"] * inv_upd),
                        "train/target_z_finite_rate": float(chunk_host["target_z_finite_sum"] * inv_upd),
                        "train/target_q_saturated_rate": float(chunk_host["target_q_saturated_sum"] * inv_upd),
                        "train/target_max_abs_q_mean": float(chunk_host["target_max_abs_q_sum"] * inv_upd),
                        "train/target_max_abs_b_mean": float(chunk_host["target_max_abs_b_sum"] * inv_upd),
                        "train/target_delta_min_u_ref_mean": float(chunk_host["target_delta_min_u_ref_sum"] * inv_upd),
                        "train/actor_q_mat_finite_rate": float(chunk_host["actor_q_mat_finite_sum"] * inv_upd),
                        "train/actor_q_vec_finite_rate": float(chunk_host["actor_q_vec_finite_sum"] * inv_upd),
                        "train/actor_g_finite_rate": float(chunk_host["actor_g_finite_sum"] * inv_upd),
                        "train/actor_h_finite_rate": float(chunk_host["actor_h_finite_sum"] * inv_upd),
                        "train/actor_inputs_finite_rate": float(chunk_host["actor_inputs_finite_sum"] * inv_upd),
                        "train/actor_z_finite_rate": float(chunk_host["actor_z_finite_sum"] * inv_upd),
                        "train/actor_q_saturated_rate": float(chunk_host["actor_q_saturated_sum"] * inv_upd),
                        "train/actor_max_abs_q_mean": float(chunk_host["actor_max_abs_q_sum"] * inv_upd),
                        "train/actor_max_abs_b_mean": float(chunk_host["actor_max_abs_b_sum"] * inv_upd),
                        "train/actor_delta_min_u_ref_mean": float(chunk_host["actor_delta_min_u_ref_sum"] * inv_upd),
                        "train/q_mat_finite": float(
                            0.5 * (chunk_host["target_q_mat_finite_sum"] + chunk_host["actor_q_mat_finite_sum"]) * inv_upd
                        ),
                        "train/q_vec_finite": float(
                            0.5 * (chunk_host["target_q_vec_finite_sum"] + chunk_host["actor_q_vec_finite_sum"]) * inv_upd
                        ),
                        "train/g_finite": float(
                            0.5 * (chunk_host["target_g_finite_sum"] + chunk_host["actor_g_finite_sum"]) * inv_upd
                        ),
                        "train/h_finite": float(
                            0.5 * (chunk_host["target_h_finite_sum"] + chunk_host["actor_h_finite_sum"]) * inv_upd
                        ),
                        "train/inputs_finite": float(
                            0.5 * (chunk_host["target_inputs_finite_sum"] + chunk_host["actor_inputs_finite_sum"]) * inv_upd
                        ),
                        "train/z_finite": float(
                            0.5 * (chunk_host["target_z_finite_sum"] + chunk_host["actor_z_finite_sum"]) * inv_upd
                        ),
                        "train/q_saturated": float(
                            0.5 * (chunk_host["target_q_saturated_sum"] + chunk_host["actor_q_saturated_sum"]) * inv_upd
                        ),
                        "train/max_abs_q": float(
                            0.5 * (chunk_host["target_max_abs_q_sum"] + chunk_host["actor_max_abs_q_sum"]) * inv_upd
                        ),
                        "train/max_abs_b": float(
                            0.5 * (chunk_host["target_max_abs_b_sum"] + chunk_host["actor_max_abs_b_sum"]) * inv_upd
                        ),
                        "train/delta_min_u_ref": float(
                            0.5
                            * (chunk_host["target_delta_min_u_ref_sum"] + chunk_host["actor_delta_min_u_ref_sum"])
                            * inv_upd
                        ),
                    }
                )
            if latest_episode_metrics is not None:
                for key, val in latest_episode_metrics.items():
                    live_metrics[f"episode/{key}"] = float(val)
            if latest_eval_metrics is not None:
                for key, val in latest_eval_metrics.items():
                    live_metrics[f"eval/{key}"] = float(val)
            print(
                f"step={next_log_step} replay={int(jax.device_get(loop_state.replay.size))} "
                f"updates={int(jax.device_get(loop_state.updates))} eps/sec={step_rate:.1f} "
                f"num_envs={num_envs}"
            )
            if metric_logger is not None:
                try:
                    metric_logger(int(next_log_step), live_metrics)
                except Exception as exc:
                    print(f"Warning: metric_logger failed at step={next_log_step}: {exc}")
            next_log_step += int(sac_cfg.log_every)

    total_time = time.time() - t0
    eval_stats = _evaluate_policy(
        env_cfg,
        eval_action_fn,
        loop_state.state["actor_params"],
        sac_cfg.seed + 99_999,
        sac_cfg.eval_episodes,
    )
    final_eval_key_return = _eval_selection_key_return(eval_stats)
    improve_return = (best_eval_key_return is None) or (final_eval_key_return < best_eval_key_return)
    if improve_return:
        best_eval_key_return = final_eval_key_return
        best_eval_step_return = total_steps
        best_eval_stats_return = _compact_eval_stats(eval_stats)
        best_state_return = _snapshot_state(loop_state.state)

    updates_final = int(jax.device_get(loop_state.updates))
    summary = {
        "total_steps": total_steps,
        "updates": updates_final,
        "wall_time_sec": float(total_time),
        "steps_per_sec": float(total_steps / max(total_time, 1e-6)),
        "num_qp_inequalities": projector.num_qp_inequalities,
        "num_backup_inequalities": projector.num_backup_inequalities,
        "env_dt": float(env_cfg.dt),
        "cbf_dt": float(cbf_cfg.dt),
        "horizon_T": cbf_cfg.horizon,
        "N_steps": cbf_cfg.num_steps,
        "backup_policy_mode": cbf_cfg.backup_policy_mode,
        "num_envs": num_envs,
        "steps_per_jit": steps_per_jit,
    }
    final_compact_eval = _compact_eval_stats(eval_stats)
    best_eval_return = best_eval_stats_return if best_eval_stats_return is not None else final_compact_eval
    _add_eval_summary(summary, "final_eval", final_compact_eval)
    summary["best_eval_step"] = int(best_eval_step_return)
    _add_eval_summary(summary, "best_eval", best_eval_return)

    result = {
        "summary": summary,
        "history": history,
        "eval": eval_stats,
        "best_eval": best_eval_return,
        "configs": {
            "sac": asdict(sac_cfg),
            "env": asdict(env_cfg),
            "cbf": asdict(cbf_cfg),
        },
    }
    result["final_state"] = _snapshot_state(loop_state.state)
    result["best_state"] = best_state_return if best_state_return is not None else result["final_state"]

    if output_dir is not None:
        with open(f"{output_dir}/summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(f"{output_dir}/configs.json", "w", encoding="utf-8") as f:
            json.dump(result["configs"], f, indent=2)
        with open(f"{output_dir}/final_weights.pkl", "wb") as f:
            pickle.dump(result["final_state"], f, protocol=pickle.HIGHEST_PROTOCOL)
        _save_best_weight_snapshots(output_dir, best_state_return=result["best_state"])

    return result


def run_training(
    sac_cfg: SACConfig,
    env_cfg: QuadrotorEnvConfig,
    cbf_cfg: QuadrotorBCBFConfig,
    output_dir: str | None = None,
    metric_logger: Callable[[int, Dict[str, float]], None] | None = None,
) -> Dict[str, Any]:
    return _run_training_jax(
        sac_cfg,
        env_cfg,
        cbf_cfg,
        output_dir=output_dir,
        metric_logger=metric_logger,
    )
