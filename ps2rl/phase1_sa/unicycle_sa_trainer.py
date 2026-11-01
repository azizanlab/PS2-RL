"""Discounted safe-arrival trainer for the lane phase-1 backup policy.

The TD3-style training backbone lives in ``sa_trainer_core``; this module
supplies the unicycle-specific construction (env, sets, action bounds, PRNG
layout), episode bookkeeping, dense-grid evaluation, and output schemas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.base_controller.unicycle_dlqr import UnicycleDLQR
from ps2rl.sets.base_sets import EllipsoidBaseSet
from ps2rl.sets.unicycle_sets import UnicycleSafeSet
from ps2rl.backup_policy.backup_policy import BackupPolicy, save_learned_backup_policy
from ps2rl.backup_policy.unicycle_analytic_backup import clip_backup_action
from ps2rl.utils.policy import ActorConfig, actor_mean_action
from ps2rl.phase1_sa.unicycle_sa_env import (
    UnicycleSAEnvConfig,
    build_unicycle_sa_env,
    unicycle_step_euler,
)
from ps2rl.phase1_sa.sa_critic import SafeArrivalCriticConfig
from ps2rl.phase1_sa.sa_trainer_core import (
    SALoopState,
    SASystemHooks,
    build_sa_action_fns,
    build_sa_one_vec_step,
    build_sa_update_fn,
    batch_safe_contains as _batch_safe_contains,
    init_sa_state,
    rate_mean as _rate,
    run_sa_training_loop,
    sa_replay_init,
    snapshot_sa_state as _snapshot_state,
    summary_stats as _summary_stats,
)
from ps2rl.utils.seed import make_prng_key


@dataclass(frozen=True)
class UnicycleSAConfig:
    seed: int = 0
    total_steps: int = 2_000_000
    start_steps: int = 5_000
    update_after: int = 2_000
    update_every: int = 8
    gradient_steps: int = 1
    batch_size: int = 128
    replay_size: int = 400_000

    beta: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    max_grad_norm: float = 5.0
    critic_huber_delta: float = 1.0
    action_smoothness_weight: float = 0.0

    hidden_size: int = 128
    actor_log_std_min: float = -8.0
    actor_log_std_max: float = -3.8

    exploration_std: float = 0.10
    exploration_clip: float = 0.25
    target_policy_noise_std: float = 0.0
    target_policy_noise_clip: float = 0.0

    eval_every: int = 5_000
    log_every: int = 1_000
    record_update_metrics: bool = True
    update_metric_every: int = 200
    num_envs: int = 32
    steps_per_jit: int = 128

    curriculum_start_scale: float = 0.0
    curriculum_increment: float = 0.005
    curriculum_success_threshold: float = 0.90
    curriculum_window_episodes: int = 50
    curriculum_min_episodes: int = 50

    val_reset_count: int = 128
    test_reset_count: int = 128

    use_handoff: bool = True
    goal_mode: str = "terminal"
    collector_terminate_on_goal: bool = True


def unicycle_sa_config_from_dict(payload: dict[str, Any]) -> UnicycleSAConfig:
    valid = UnicycleSAConfig.__dataclass_fields__
    return UnicycleSAConfig(**{k: v for k, v in payload.items() if k in valid})


def _aggregate_episode_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        empty_stats = _summary_stats(np.zeros((0,), dtype=np.float64))
        return {
            "count": 0,
            "success_rate": 0.0,
            "capture_success_rate": 0.0,
            "crash_rate": 0.0,
            "safe_rate": 0.0,
            "safe_rollout_rate": 0.0,
            "terminal_at_horizon_rate": 0.0,
            "post_capture_terminal_success_rate": 0.0,
            "invariance_after_terminal_entry_rate": 0.0,
            "mean_discounted_ra_score": 0.0,
            "mean_capture_entry_step": None,
            "mean_terminal_entry_step": None,
            "capture_entry_step": empty_stats,
            "terminal_entry_step": empty_stats,
        }

    success = np.asarray([rec["success"] for rec in records], dtype=np.float64)
    capture_success = np.asarray([rec["capture_success"] for rec in records], dtype=np.float64)
    crash = np.asarray([rec["crash"] for rec in records], dtype=np.float64)
    safe_rate = np.asarray([rec["safe_rate"] for rec in records], dtype=np.float64)
    safe_rollout = np.asarray([rec["safe_rollout"] for rec in records], dtype=np.float64)
    terminal_at_horizon = np.asarray([rec["terminal_at_horizon"] for rec in records], dtype=np.float64)
    post_capture_terminal_success = np.asarray(
        [rec["post_capture_terminal_success"] for rec in records],
        dtype=np.float64,
    )
    invariance_after_terminal_entry = np.asarray(
        [rec["invariance_after_terminal_entry"] for rec in records],
        dtype=np.float64,
    )
    discounted_score = np.asarray([rec["discounted_ra_score"] for rec in records], dtype=np.float64)
    capture_entry_steps = np.asarray([rec["capture_entry_step"] for rec in records], dtype=np.float64)
    terminal_entry_steps = np.asarray([rec["terminal_entry_step"] for rec in records], dtype=np.float64)

    capture_mask = capture_success > 0.5
    terminal_mask = np.isfinite(terminal_entry_steps)
    post_capture_terminal_success_rate = (
        float(np.mean(post_capture_terminal_success[capture_mask])) if np.any(capture_mask) else 0.0
    )
    invariance_after_terminal_entry_rate = (
        float(np.mean(invariance_after_terminal_entry[terminal_mask])) if np.any(terminal_mask) else 0.0
    )

    capture_step_stats = _summary_stats(capture_entry_steps[capture_mask])
    terminal_step_stats = _summary_stats(terminal_entry_steps[terminal_mask])

    return {
        "count": int(len(records)),
        "success_rate": _rate(success),
        "capture_success_rate": _rate(capture_success),
        "crash_rate": _rate(crash),
        "safe_rate": _rate(safe_rate),
        "safe_rollout_rate": _rate(safe_rollout),
        "terminal_at_horizon_rate": _rate(terminal_at_horizon),
        "post_capture_terminal_success_rate": post_capture_terminal_success_rate,
        "invariance_after_terminal_entry_rate": invariance_after_terminal_entry_rate,
        "mean_discounted_ra_score": float(np.mean(discounted_score)),
        "mean_capture_entry_step": capture_step_stats["mean"],
        "mean_terminal_entry_step": terminal_step_stats["mean"],
        "capture_entry_step": capture_step_stats,
        "terminal_entry_step": terminal_step_stats,
    }


def _evaluate_policy_jax(
    *,
    env_cfg: UnicycleSAEnvConfig,
    actor_params,
    actor_cfg: ActorConfig,
    action_scale: jax.Array,
    action_low: jax.Array,
    action_high: jax.Array,
    eval_states: np.ndarray,
    beta: float,
    use_handoff: bool,
) -> dict[str, Any]:
    reset_states = np.asarray(eval_states, dtype=np.float32)
    if reset_states.shape[0] == 0:
        overall = _aggregate_episode_records([])
        overall["trajectory"] = {
            k: np.zeros((0,), dtype=np.float32)
            for k in ("obs", "next_obs", "act", "raw_action", "rew", "safe", "capture", "terminal")
        }
        return overall

    safe_set = UnicycleSafeSet(y_max=env_cfg.y_max, psi_max=env_cfg.psi_max)
    lqr_design = UnicycleDLQR.from_config(env_cfg)
    # One base set plays both roles: safe-arrival goal and LQR hand-off region.
    capture_set = EllipsoidBaseSet(lqr_design, float(env_cfg.base_set_c))
    terminal_set = capture_set
    beta_arr = jnp.asarray(float(beta), dtype=jnp.float32)
    horizon_steps = int(env_cfg.horizon_steps)

    def eval_raw_action(params, x: jax.Array) -> jax.Array:
        raw = actor_mean_action(
            params,
            x[None, :],
            action_scale,
            actor_cfg,
            action_low=action_low,
            action_high=action_high,
        )[0]
        return jnp.clip(jnp.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), action_low, action_high)

    def transition(x: jax.Array, raw_action: jax.Array):
        raw = jnp.clip(raw_action, action_low, action_high)
        if use_handoff:
            applied = jnp.clip(
                clip_backup_action(
                    BackupPolicy.select_action(
                        x,
                        clip_backup_action(raw, a_max=env_cfg.a_max, r_max=env_cfg.r_max),
                        capture_set,
                        controller=lqr_design,
                    ),
                    a_max=env_cfg.a_max,
                    r_max=env_cfg.r_max,
                ),
                action_low,
                action_high,
            )
        else:
            applied = raw
        x_next = unicycle_step_euler(x, applied, env_cfg)
        safe = safe_set.contains(x_next)
        capture = capture_set.contains(x_next)
        terminal = terminal_set.contains(x_next)
        return x_next, safe, capture, terminal, raw, applied

    def rollout_single(params, x0: jax.Array):
        safe0 = safe_set.contains(x0)
        capture0 = capture_set.contains(x0)
        terminal0 = terminal_set.contains(x0)

        def step(carry, k):
            (
                x,
                done,
                step_count,
                safe_sum,
                safe_rollout,
                entered_capture,
                capture_entry_step,
                entered_terminal,
                terminal_entry_step,
                left_after_terminal,
                crash_flag,
            ) = carry
            active = ~done
            raw_action = eval_raw_action(params, x)
            x_next, safe, capture, terminal, raw, applied = transition(x, raw_action)
            done_now = active & (~safe)
            x_next = jnp.where(active, x_next, x)
            step_count = step_count + active.astype(jnp.int32)
            safe_sum = safe_sum + active.astype(jnp.float32) * safe.astype(jnp.float32)
            capture_entry_step = jnp.where(active & (~entered_capture) & capture, k + jnp.int32(1), capture_entry_step)
            terminal_entry_step = jnp.where(active & (~entered_terminal) & terminal, k + jnp.int32(1), terminal_entry_step)
            entered_capture = entered_capture | (active & capture)
            left_after_terminal = left_after_terminal | (active & entered_terminal & (~terminal))
            entered_terminal = entered_terminal | (active & terminal)
            safe_rollout = safe_rollout & jnp.where(active, safe, jnp.asarray(True, dtype=jnp.bool_))
            crash_flag = crash_flag | done_now
            record = {
                "obs": x,
                "next_obs": x_next,
                "act": jnp.where(active, applied, jnp.zeros_like(applied)),
                "raw_action": jnp.where(active, raw, jnp.zeros_like(raw)),
                "rew": jnp.asarray(0.0, dtype=jnp.float32),
                "safe": active.astype(jnp.float32) * safe.astype(jnp.float32),
                "capture": active.astype(jnp.float32) * capture.astype(jnp.float32),
                "terminal": active.astype(jnp.float32) * terminal.astype(jnp.float32),
                "done": done_now,
            }
            return (
                x_next,
                done | done_now,
                step_count,
                safe_sum,
                safe_rollout,
                entered_capture,
                capture_entry_step,
                entered_terminal,
                terminal_entry_step,
                left_after_terminal,
                crash_flag,
            ), record

        init = (
            jnp.asarray(x0, dtype=jnp.float32),
            jnp.asarray(False, dtype=jnp.bool_),
            jnp.int32(0),
            jnp.asarray(0.0, dtype=jnp.float32),
            safe0,
            capture0,
            jnp.where(capture0, jnp.int32(0), jnp.int32(-1)),
            terminal0,
            jnp.where(terminal0, jnp.int32(0), jnp.int32(-1)),
            jnp.asarray(False, dtype=jnp.bool_),
            jnp.asarray(False, dtype=jnp.bool_),
        )
        final, records = jax.lax.scan(step, init, jnp.arange(horizon_steps, dtype=jnp.int32))
        eval_len = jnp.maximum(final[2].astype(jnp.float32), jnp.asarray(1.0, dtype=jnp.float32))
        terminal_at_horizon = ((~final[10]) & safe_set.contains(final[0]) & terminal_set.contains(final[0])).astype(jnp.float32)
        success = (final[4] & final[7] & (~final[9]) & (terminal_at_horizon > 0.5)).astype(jnp.float32)
        capture_success = final[5].astype(jnp.float32)
        post_capture_terminal_success = (final[5] & final[7]).astype(jnp.float32)
        invariance_after_terminal_entry = jnp.where(final[7], (~final[9]).astype(jnp.float32), jnp.asarray(0.0, dtype=jnp.float32))
        discounted_score = jnp.where(
            final[5],
            beta_arr ** jnp.maximum(final[6].astype(jnp.float32), jnp.asarray(0.0, dtype=jnp.float32)),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        capture_entry_step = jnp.where(final[6] >= 0, final[6].astype(jnp.float32), jnp.asarray(jnp.nan, dtype=jnp.float32))
        terminal_entry_step = jnp.where(final[8] >= 0, final[8].astype(jnp.float32), jnp.asarray(jnp.nan, dtype=jnp.float32))
        metrics = (
            success,
            capture_success,
            final[10].astype(jnp.float32),
            final[3] / eval_len,
            final[4].astype(jnp.float32),
            terminal_at_horizon,
            post_capture_terminal_success,
            invariance_after_terminal_entry,
            discounted_score,
            capture_entry_step,
            terminal_entry_step,
        )
        return metrics, records

    @jax.jit
    def rollout_batch(x0_batch: jax.Array):
        return jax.vmap(lambda x0: rollout_single(actor_params, x0)[0])(x0_batch)

    @jax.jit
    def rollout_trace(x0: jax.Array):
        return rollout_single(actor_params, x0)[1]

    metrics = rollout_batch(jnp.asarray(reset_states, dtype=jnp.float32))
    metrics_np = [np.asarray(v) for v in jax.device_get(metrics)]
    episode_records = []
    for idx in range(reset_states.shape[0]):
        episode_records.append(
            {
                "success": float(metrics_np[0][idx]),
                "capture_success": float(metrics_np[1][idx]),
                "crash": float(metrics_np[2][idx]),
                "safe_rate": float(metrics_np[3][idx]),
                "safe_rollout": float(metrics_np[4][idx]),
                "terminal_at_horizon": float(metrics_np[5][idx]),
                "post_capture_terminal_success": float(metrics_np[6][idx]),
                "invariance_after_terminal_entry": float(metrics_np[7][idx]),
                "discounted_ra_score": float(metrics_np[8][idx]),
                "capture_entry_step": float(metrics_np[9][idx]),
                "terminal_entry_step": float(metrics_np[10][idx]),
            }
        )

    overall = _aggregate_episode_records(episode_records)
    trace_raw = jax.device_get(rollout_trace(jnp.asarray(reset_states[0], dtype=jnp.float32)))
    done_steps = np.asarray(trace_raw["done"], dtype=bool)
    valid_steps = int(np.argmax(done_steps) + 1) if np.any(done_steps) else horizon_steps
    overall["trajectory"] = {
        "obs": np.asarray(trace_raw["obs"][:valid_steps]),
        "next_obs": np.asarray(trace_raw["next_obs"][:valid_steps]),
        "act": np.asarray(trace_raw["act"][:valid_steps]),
        "raw_action": np.asarray(trace_raw["raw_action"][:valid_steps]),
        "rew": np.asarray(trace_raw["rew"][:valid_steps]),
        "safe": np.asarray(trace_raw["safe"][:valid_steps]),
        "capture": np.asarray(trace_raw["capture"][:valid_steps]),
        "terminal": np.asarray(trace_raw["terminal"][:valid_steps]),
    }
    return overall


def _eval_rank_key(eval_stats: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    capture_step_mean = eval_stats.get("mean_capture_entry_step")
    capture_step_rank = -float(capture_step_mean) if capture_step_mean is not None else -1e9
    return (
        float(eval_stats.get("success_rate", 0.0)),
        -float(eval_stats.get("crash_rate", 1.0)),
        float(eval_stats.get("terminal_at_horizon_rate", 0.0)),
        float(eval_stats.get("post_capture_terminal_success_rate", 0.0)),
        float(eval_stats.get("invariance_after_terminal_entry_rate", 0.0)),
        float(eval_stats.get("mean_discounted_ra_score", 0.0)),
        capture_step_rank,
    )


def _sample_eval_states(
    env_fns,
    *,
    key: jax.Array,
    count: int,
) -> np.ndarray:
    if int(count) <= 0:
        return np.zeros((0, env_fns.obs_dim), dtype=np.float32)
    keys = jax.random.split(key, int(count))
    _, obs = env_fns.reset_batched(keys, jnp.asarray(1.0, dtype=jnp.float32))
    return np.asarray(jax.device_get(obs), dtype=np.float32)


def _validate_eval_env_compatibility(
    train_env_cfg: UnicycleSAEnvConfig,
    eval_env_cfg: UnicycleSAEnvConfig,
) -> None:
    train_payload = asdict(train_env_cfg)
    eval_payload = asdict(eval_env_cfg)
    allowed_differences = {"dt", "horizon_steps"}
    mismatches = [
        key
        for key in train_payload
        if key not in allowed_differences and train_payload[key] != eval_payload[key]
    ]
    if mismatches:
        detail = ", ".join(
            f"{key}: train={train_payload[key]!r}, eval={eval_payload[key]!r}"
            for key in mismatches[:5]
        )
        if len(mismatches) > 5:
            detail += ", ..."
        raise ValueError(
            "eval_env_cfg must match env_cfg on every field except dt and horizon_steps for lane safe-arrival training. "
            f"Mismatches: {detail}"
        )


def _checkpoint_metadata(
    ra_cfg: UnicycleSAConfig,
    env_cfg: UnicycleSAEnvConfig,
    *,
    training_env_cfg: UnicycleSAEnvConfig | None = None,
) -> dict[str, Any]:
    training_env_cfg = env_cfg if training_env_cfg is None else training_env_cfg
    return {
        "training_objective": "discounted_reach_avoid",
        "goal_mode": str(ra_cfg.goal_mode),
        "base_set_handoff_enabled": bool(ra_cfg.use_handoff),
        "base_set_config": {"base_set_c": float(env_cfg.base_set_c)},
        "lqr_config": {
            "v_des": float(env_cfg.v_des),
            "dt": float(env_cfg.dt),
            "a_max": float(env_cfg.a_max),
            "r_max": float(env_cfg.r_max),
            "r_matrix_diag_a": float(env_cfg.lqr_r_a),
            "r_matrix_diag_r": float(env_cfg.lqr_r_r),
            "lqr_q_y": float(env_cfg.lqr_q_y),
            "lqr_q_v": float(env_cfg.lqr_q_v),
            "lqr_q_psi": float(env_cfg.lqr_q_psi),
            "lqr_r_a": float(env_cfg.lqr_r_a),
            "lqr_r_r": float(env_cfg.lqr_r_r),
        },
        "tail_controller": "lqr",
        "horizon_T": float(env_cfg.horizon_T),
        "num_steps": int(env_cfg.horizon_steps),
        "training_horizon_T": float(training_env_cfg.horizon_T),
        "training_num_steps": int(training_env_cfg.horizon_steps),
        "v_des": float(env_cfg.v_des),
    }


_EPISODE_METRIC_FIELDS: tuple[str, ...] = (
    "episode_len_sum",
    "episode_safe_rate_sum",
    "episode_capture_rate_sum",
    "episode_terminal_rate_sum",
    "episode_capture_success_sum",
    "episode_success_sum",
    "episode_crash_sum",
    "episode_terminal_at_horizon_sum",
    "episode_discounted_ra_score_sum",
)

_EPISODE_HISTORY_PAIRS: tuple[tuple[str, str], ...] = (
    ("ep_len", "episode_len_sum"),
    ("ep_safe_rate", "episode_safe_rate_sum"),
    ("ep_capture_rate", "episode_capture_rate_sum"),
    ("ep_terminal_rate", "episode_terminal_rate_sum"),
    ("ep_capture_success", "episode_capture_success_sum"),
    ("ep_success", "episode_success_sum"),
    ("ep_crash", "episode_crash_sum"),
    ("ep_terminal_at_horizon", "episode_terminal_at_horizon_sum"),
    ("ep_discounted_ra_score", "episode_discounted_ra_score_sum"),
)

_HISTORY_KEYS: tuple[str, ...] = (
    "step",
    "episode_idx",
    "curriculum_scale",
    "ep_len",
    "ep_safe_rate",
    "ep_capture_rate",
    "ep_terminal_rate",
    "ep_capture_success",
    "ep_success",
    "ep_crash",
    "ep_terminal_at_horizon",
    "ep_discounted_ra_score",
    "critic_loss",
    "actor_loss",
    "action_penalty",
    "target_mean",
    "q_pi_mean",
    "q1_grad_norm",
    "q2_grad_norm",
    "actor_grad_norm",
    "eval_success_rate",
    "eval_capture_success_rate",
    "eval_crash_rate",
    "eval_safe_rate",
    "eval_terminal_at_horizon_rate",
    "eval_post_capture_terminal_success_rate",
    "eval_invariance_after_terminal_entry_rate",
    "eval_mean_discounted_ra_score",
    "eval_mean_capture_entry_step",
    "eval_mean_terminal_entry_step",
)


def _episode_bookkeeping(info: Any, beta_arr: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    strict_success = info.completed_entered_terminal * info.completed_safe_rollout

    capture_entry_step_f = jnp.where(
        info.completed_capture_entry_step >= 0,
        info.completed_capture_entry_step.astype(jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    discounted_ra_score = jnp.where(
        info.completed_capture_success > 0.5,
        beta_arr ** capture_entry_step_f,
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    fields = {
        "episode_len_sum": jnp.sum(info.completed_len),
        "episode_safe_rate_sum": jnp.sum(info.completed_safe_rate),
        "episode_capture_rate_sum": jnp.sum(info.completed_capture_rate),
        "episode_terminal_rate_sum": jnp.sum(info.completed_terminal_rate),
        "episode_capture_success_sum": jnp.sum(info.completed_capture_success),
        "episode_success_sum": jnp.sum(strict_success),
        "episode_crash_sum": jnp.sum(info.is_crash * info.episode_done.astype(jnp.float32)),
        "episode_terminal_at_horizon_sum": jnp.sum(info.completed_terminal_at_horizon),
        "episode_discounted_ra_score_sum": jnp.sum(discounted_ra_score),
    }
    return strict_success, fields


def _append_eval_history(history: dict[str, list[float]], eval_stats: dict[str, Any]) -> None:
    history["eval_success_rate"].append(float(eval_stats["success_rate"]))
    history["eval_capture_success_rate"].append(float(eval_stats["capture_success_rate"]))
    history["eval_crash_rate"].append(float(eval_stats["crash_rate"]))
    history["eval_safe_rate"].append(float(eval_stats["safe_rate"]))
    history["eval_terminal_at_horizon_rate"].append(float(eval_stats["terminal_at_horizon_rate"]))
    history["eval_post_capture_terminal_success_rate"].append(float(eval_stats["post_capture_terminal_success_rate"]))
    history["eval_invariance_after_terminal_entry_rate"].append(float(eval_stats["invariance_after_terminal_entry_rate"]))
    history["eval_mean_discounted_ra_score"].append(float(eval_stats["mean_discounted_ra_score"]))
    history["eval_mean_capture_entry_step"].append(
        float(eval_stats["mean_capture_entry_step"]) if eval_stats["mean_capture_entry_step"] is not None else np.nan
    )
    history["eval_mean_terminal_entry_step"].append(
        float(eval_stats["mean_terminal_entry_step"]) if eval_stats["mean_terminal_entry_step"] is not None else np.nan
    )


def run_unicycle_sa_training(
    ra_cfg: UnicycleSAConfig,
    env_cfg: UnicycleSAEnvConfig,
    *,
    eval_env_cfg: UnicycleSAEnvConfig | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    eval_env_cfg = env_cfg if eval_env_cfg is None else eval_env_cfg
    _validate_eval_env_compatibility(env_cfg, eval_env_cfg)

    if int(ra_cfg.num_envs) <= 0:
        raise ValueError(f"num_envs must be positive, got {ra_cfg.num_envs}")
    if int(ra_cfg.steps_per_jit) <= 0:
        raise ValueError(f"steps_per_jit must be positive, got {ra_cfg.steps_per_jit}")
    if int(ra_cfg.total_steps) < 0:
        raise ValueError(f"total_steps must be nonnegative, got {ra_cfg.total_steps}")
    if not (0.0 < float(ra_cfg.beta) < 1.0):
        raise ValueError(f"beta must lie in (0, 1), got {ra_cfg.beta}")

    out_dir = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    num_envs = int(ra_cfg.num_envs)

    jax_key = make_prng_key(ra_cfg.seed)
    env_fns = build_unicycle_sa_env(
        env_cfg,
        terminate_on_goal=bool(ra_cfg.collector_terminate_on_goal),
    )

    action_scale = jnp.asarray(env_fns.action_scale, dtype=jnp.float32)
    action_low = jnp.asarray(env_fns.action_low, dtype=jnp.float32)
    action_high = jnp.asarray(env_fns.action_high, dtype=jnp.float32)

    actor_cfg = ActorConfig(
        obs_dim=env_fns.obs_dim,
        action_dim=env_fns.action_dim,
        hidden_sizes=(ra_cfg.hidden_size, ra_cfg.hidden_size),
        log_std_min=ra_cfg.actor_log_std_min,
        log_std_max=ra_cfg.actor_log_std_max,
    )
    critic_cfg = SafeArrivalCriticConfig(
        obs_dim=env_fns.obs_dim,
        act_dim=env_fns.action_dim,
        hidden_sizes=(ra_cfg.hidden_size, ra_cfg.hidden_size),
    )

    safe_set = UnicycleSafeSet(y_max=env_cfg.y_max, psi_max=env_cfg.psi_max)
    lqr_design = UnicycleDLQR.from_config(env_cfg)
    capture_set = EllipsoidBaseSet(lqr_design, float(env_cfg.base_set_c))
    terminal_set = capture_set

    if ra_cfg.goal_mode == "terminal":
        goal_contains_fn = _batch_safe_contains(lambda x: terminal_set.contains(x))
    else:
        raise ValueError(f"unsupported goal_mode {ra_cfg.goal_mode} for lane safe-arrival")
    fail_contains_fn = _batch_safe_contains(lambda x: jnp.logical_not(safe_set.contains(x)))
    handoff_contains_fn = _batch_safe_contains(lambda x: capture_set.contains(x))

    jax_key, key_state, key_env, key_val, key_test = jax.random.split(jax_key, 5)
    state = init_sa_state(key_state, actor_cfg, critic_cfg)
    replay = sa_replay_init(ra_cfg.replay_size, env_fns.obs_dim, env_fns.action_dim)
    update_fn = build_sa_update_fn(
        ra_cfg,
        actor_cfg,
        action_scale,
        action_low,
        action_high,
        goal_contains_fn,
        fail_contains_fn,
        handoff_contains_fn,
    )
    _, collect_action_batch_fn = build_sa_action_fns(
        ra_cfg,
        actor_cfg,
        action_scale,
        action_low,
        action_high,
    )

    env_keys = jax.random.split(key_env, num_envs)
    curriculum_scale0 = jnp.asarray(float(ra_cfg.curriculum_start_scale), dtype=jnp.float32)
    env_state, obs = env_fns.reset_batched(env_keys, curriculum_scale0)

    if bool(jnp.all(env_state.ep_entered_capture)):
        raise ValueError(
            "Unicycle Phase-1 reset sampler: every initial reset state is inside "
            "the base set (the rejection sampler hit max_resample_tries). This "
            "usually means --curriculum_start_scale is too small (the reset region "
            "collapses inside the base set); recommended value is 0.2. Increase it so "
            "reset states start outside the base set."
        )

    val_reset_states = _sample_eval_states(env_fns, key=key_val, count=ra_cfg.val_reset_count)
    test_reset_states = _sample_eval_states(env_fns, key=key_test, count=ra_cfg.test_reset_count)

    window = int(max(1, ra_cfg.curriculum_window_episodes))
    loop_state = SALoopState(
        state=state,
        replay=replay,
        env_state=env_state,
        obs=obs,
        key=jax_key,
        env_keys=env_keys,
        global_step=jnp.int32(0),
        updates=jnp.int32(0),
        curriculum_scale=curriculum_scale0,
        episode_count=jnp.int32(0),
        success_window=jnp.zeros((window,), dtype=jnp.float32),
        success_window_size=jnp.int32(0),
        success_window_ptr=jnp.int32(0),
    )

    def run_eval(actor_params, split: str) -> dict[str, Any]:
        eval_states = val_reset_states if split == "val" else test_reset_states
        return _evaluate_policy_jax(
            env_cfg=eval_env_cfg,
            actor_params=actor_params,
            actor_cfg=actor_cfg,
            action_scale=action_scale,
            action_low=action_low,
            action_high=action_high,
            eval_states=eval_states,
            beta=ra_cfg.beta,
            use_handoff=ra_cfg.use_handoff,
        )

    hooks = SASystemHooks(
        episode_metric_fields=_EPISODE_METRIC_FIELDS,
        episode_bookkeeping=_episode_bookkeeping,
        run_eval=run_eval,
        eval_rank_key=_eval_rank_key,
        append_eval_history=_append_eval_history,
        episode_history_pairs=_EPISODE_HISTORY_PAIRS,
        history_keys=_HISTORY_KEYS,
    )
    one_vec_step = build_sa_one_vec_step(
        env_fns=env_fns,
        collect_action_batch_fn=collect_action_batch_fn,
        update_fn=update_fn,
        ra_cfg=ra_cfg,
        hooks=hooks,
    )
    loop_result = run_sa_training_loop(
        ra_cfg=ra_cfg,
        loop_state=loop_state,
        one_vec_step=one_vec_step,
        hooks=hooks,
    )

    loop_state = loop_result.loop_state
    history = loop_result.history
    best_eval_stats = loop_result.best_eval_stats
    best_eval_step = loop_result.best_eval_step
    best_state = loop_result.best_state
    val_eval = loop_result.val_eval
    total_time = loop_result.total_time

    summary = {
        "training_objective": "discounted_reach_avoid",
        "total_steps": int(ra_cfg.total_steps),
        "updates": int(jax.device_get(loop_state.updates)),
        "wall_time_sec": float(total_time),
        "steps_per_sec": float(ra_cfg.total_steps / max(total_time, 1e-6)) if total_time > 0.0 else 0.0,
        "horizon_T": float(env_cfg.horizon_T),
        "N_steps": int(env_cfg.horizon_steps),
        "eval_horizon_T": float(eval_env_cfg.horizon_T),
        "eval_N_steps": int(eval_env_cfg.horizon_steps),
        "beta": float(ra_cfg.beta),
        "goal_mode": str(ra_cfg.goal_mode),
        "use_handoff": bool(ra_cfg.use_handoff),
        "collector_terminate_on_goal": bool(ra_cfg.collector_terminate_on_goal),
        "final_curriculum_scale": float(jax.device_get(loop_state.curriculum_scale)),
        "final_eval_success_rate": float(val_eval["success_rate"]),
        "final_eval_capture_success_rate": float(val_eval["capture_success_rate"]),
        "final_eval_crash_rate": float(val_eval["crash_rate"]),
        "final_eval_safe_rate": float(val_eval["safe_rate"]),
        "final_eval_terminal_at_horizon_rate": float(val_eval["terminal_at_horizon_rate"]),
        "final_eval_post_capture_terminal_success_rate": float(val_eval["post_capture_terminal_success_rate"]),
        "final_eval_invariance_after_terminal_entry_rate": float(val_eval["invariance_after_terminal_entry_rate"]),
        "final_eval_mean_discounted_ra_score": float(val_eval["mean_discounted_ra_score"]),
        "best_eval_step": int(best_eval_step),
        "best_eval_success_rate": float(best_eval_stats["success_rate"] if best_eval_stats is not None else val_eval["success_rate"]),
        "best_eval_capture_success_rate": float(
            best_eval_stats["capture_success_rate"] if best_eval_stats is not None else val_eval["capture_success_rate"]
        ),
        "best_eval_crash_rate": float(best_eval_stats["crash_rate"] if best_eval_stats is not None else val_eval["crash_rate"]),
        "best_eval_terminal_at_horizon_rate": float(
            best_eval_stats["terminal_at_horizon_rate"] if best_eval_stats is not None else val_eval["terminal_at_horizon_rate"]
        ),
        "best_eval_post_capture_terminal_success_rate": float(
            best_eval_stats["post_capture_terminal_success_rate"]
            if best_eval_stats is not None
            else val_eval["post_capture_terminal_success_rate"]
        ),
        "best_eval_invariance_after_terminal_entry_rate": float(
            best_eval_stats["invariance_after_terminal_entry_rate"]
            if best_eval_stats is not None
            else val_eval["invariance_after_terminal_entry_rate"]
        ),
        "best_eval_mean_discounted_ra_score": float(
            best_eval_stats["mean_discounted_ra_score"] if best_eval_stats is not None else val_eval["mean_discounted_ra_score"]
        ),
    }

    result = {
        "summary": summary,
        "history": history,
        "eval": val_eval,
        "best_eval": best_eval_stats if best_eval_stats is not None else {k: v for k, v in val_eval.items() if k != "trajectory"},
        "configs": {
            "backup_ra": asdict(ra_cfg),
            "backup_env": asdict(env_cfg),
            "backup_eval_env": asdict(eval_env_cfg),
            "actor": asdict(actor_cfg),
        },
    }
    result["final_state"] = _snapshot_state(loop_state.state)
    result["best_state"] = best_state if best_state is not None else result["final_state"]

    metadata = _checkpoint_metadata(ra_cfg, eval_env_cfg, training_env_cfg=env_cfg)
    if out_dir is not None:
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(out_dir / "configs.json", "w", encoding="utf-8") as f:
            json.dump(result["configs"], f, indent=2)
        save_learned_backup_policy(
            out_dir / "final_weights.pkl",
            actor_params=result["final_state"]["actor_params"],
            actor_cfg=actor_cfg,
            action_scale=np.asarray(jax.device_get(action_scale), dtype=np.float32),
            action_low=np.asarray(jax.device_get(action_low), dtype=np.float32),
            action_high=np.asarray(jax.device_get(action_high), dtype=np.float32),
            metadata={**metadata, "checkpoint_kind": "final"},
            metadata_before_bounds=True,
        )
        save_learned_backup_policy(
            out_dir / "best_weights.pkl",
            actor_params=result["best_state"]["actor_params"],
            actor_cfg=actor_cfg,
            action_scale=np.asarray(jax.device_get(action_scale), dtype=np.float32),
            action_low=np.asarray(jax.device_get(action_low), dtype=np.float32),
            action_high=np.asarray(jax.device_get(action_high), dtype=np.float32),
            metadata={**metadata, "checkpoint_kind": "best", "best_eval_step": int(best_eval_step)},
            metadata_before_bounds=True,
        )
        result["final_weights_path"] = str(out_dir / "final_weights.pkl")
        result["best_weights_path"] = str(out_dir / "best_weights.pkl")
    else:
        result["final_weights_path"] = None
        result["best_weights_path"] = None

    return result


__all__ = [
    "UnicycleSAConfig",
    "_aggregate_episode_records",
    "_evaluate_policy_jax",
    "unicycle_sa_config_from_dict",
    "run_unicycle_sa_training",
]
