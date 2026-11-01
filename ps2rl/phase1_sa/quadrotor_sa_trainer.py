"""Discounted safe-arrival trainer for the quadrotor phase-1 backup policy.

The TD3-style training backbone lives in ``sa_trainer_core``; this module
supplies the quadrotor-specific construction (reset-library env, cbf-config
action bounds, PRNG layout), episode bookkeeping, region-stratified held-out
evaluation, and output schemas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.base_controller.quadrotor_dlqr import QuadrotorDLQR
from ps2rl.sets.base_sets import EllipsoidBaseSet
from ps2rl.backup_policy.backup_policy import BackupPolicy
from ps2rl.sets.quadrotor_sets import QuadrotorSafeSet
from ps2rl.phase1_sa.quadrotor_sa_env import QuadrotorSAEnvConfig
from ps2rl.evaluation.quadrotor_trace_reset_lib import QuadrotorResetLibrary
from ps2rl.cil.quadrotor_backup_cbf import hard_deck_value, quadrotor_step_euler
from ps2rl.utils.policy import ActorConfig, actor_mean_action
from ps2rl.phase1_sa.quadrotor_sa_env import build_quadrotor_sa_env
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
class QuadrotorRecoverabilityWeights:
    general_trace: float = 1.0
    near_ceiling: float = 2.0
    bridge: float = 2.5
    base_shell: float = 1.0


@dataclass(frozen=True)
class QuadrotorSAConfig:
    seed: int = 0
    total_steps: int = 200_000
    start_steps: int = 5_000
    update_after: int = 2_000
    update_every: int = 8
    gradient_steps: int = 1
    batch_size: int = 128
    replay_size: int = 400_000

    beta: float = 0.984035
    tau: float = 0.005
    policy_delay: int = 2
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    max_grad_norm: float = 5.0
    critic_huber_delta: float = 1.0
    action_smoothness_weight: float = 0.0

    hidden_size: int = 128
    actor_log_std_min: float = -7.0
    actor_log_std_max: float = -2.5

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
    curriculum_increment: float = 0.10
    curriculum_success_threshold: float = 0.80
    curriculum_window_episodes: int = 20
    curriculum_min_episodes: int = 10

    use_handoff: bool = True
    goal_mode: str = "terminal"
    collector_terminate_on_goal: bool = True

    smoke_test: bool = False


def quadrotor_sa_config_from_dict(payload: dict[str, Any]) -> QuadrotorSAConfig:
    return QuadrotorSAConfig(**{k: v for k, v in payload.items() if k in QuadrotorSAConfig.__dataclass_fields__})


def quadrotor_recoverability_weights_from_dict(payload: dict[str, Any]) -> QuadrotorRecoverabilityWeights:
    return QuadrotorRecoverabilityWeights(
        general_trace=float(payload.get("general_trace", 1.0)),
        near_ceiling=float(payload.get("near_ceiling", 2.0)),
        bridge=float(payload.get("bridge", 2.5)),
        base_shell=float(payload.get("base_shell", payload.get("capture_shell", 1.0))),
    )


def _aggregate_episode_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        empty_stats = _summary_stats(np.zeros((0,), dtype=np.float64))
        return {
            "count": 0,
            "success_rate": 0.0,
            "crash_rate": 0.0,
            "safe_rate": 0.0,
            "capture_rate": 0.0,
            "terminal_rate": 0.0,
            "safe_rollout_rate": 0.0,
            "terminal_at_horizon_rate": 0.0,
            "entered_terminal_rate": 0.0,
            "invariance_after_entry_rate": 0.0,
            "post_entry_terminal_step_rate": 0.0,
            "post_entry_terminal_steps": 0,
            "post_entry_total_steps": 0,
            "mean_discounted_ra_score": 0.0,
            "entry_time_sec": empty_stats,
            "minimum_hard_deck_margin": empty_stats,
        }

    success = np.asarray([rec["success"] for rec in records], dtype=np.float64)
    crash = np.asarray([rec["crash"] for rec in records], dtype=np.float64)
    safe_rate = np.asarray([rec["safe_rate"] for rec in records], dtype=np.float64)
    capture_rate = np.asarray([rec["capture_rate"] for rec in records], dtype=np.float64)
    terminal_rate = np.asarray([rec["terminal_rate"] for rec in records], dtype=np.float64)
    safe_rollout = np.asarray([rec["safe_rollout"] for rec in records], dtype=np.float64)
    terminal_at_horizon = np.asarray([rec["terminal_at_horizon"] for rec in records], dtype=np.float64)
    entered_terminal = np.asarray([rec["entered_terminal"] for rec in records], dtype=np.float64)
    invariance_after_entry = np.asarray([rec["invariance_after_entry"] for rec in records], dtype=np.float64)
    discounted_score = np.asarray([rec["discounted_ra_score"] for rec in records], dtype=np.float64)
    entry_times = np.asarray([rec["entry_time_sec"] for rec in records], dtype=np.float64)
    min_hard_deck_margin = np.asarray([rec["min_hard_deck_margin"] for rec in records], dtype=np.float64)
    post_entry_terminal_steps = np.asarray([rec.get("post_entry_terminal_steps", 0.0) for rec in records], dtype=np.float64)
    post_entry_total_steps = np.asarray([rec.get("post_entry_total_steps", 0.0) for rec in records], dtype=np.float64)
    total_post_entry_steps = float(np.sum(post_entry_total_steps))
    post_entry_terminal_step_rate = float(np.sum(post_entry_terminal_steps) / total_post_entry_steps) if total_post_entry_steps > 0.0 else 0.0

    return {
        "count": int(len(records)),
        "success_rate": _rate(success),
        "crash_rate": _rate(crash),
        "safe_rate": _rate(safe_rate),
        "capture_rate": _rate(capture_rate),
        "terminal_rate": _rate(terminal_rate),
        "safe_rollout_rate": _rate(safe_rollout),
        "terminal_at_horizon_rate": _rate(terminal_at_horizon),
        "entered_terminal_rate": _rate(entered_terminal),
        "invariance_after_entry_rate": _rate(invariance_after_entry),
        "post_entry_terminal_step_rate": post_entry_terminal_step_rate,
        "post_entry_terminal_steps": int(np.sum(post_entry_terminal_steps)),
        "post_entry_total_steps": int(np.sum(post_entry_total_steps)),
        "mean_discounted_ra_score": float(np.mean(discounted_score)),
        "entry_time_sec": _summary_stats(entry_times),
        "minimum_hard_deck_margin": _summary_stats(min_hard_deck_margin),
    }


def _weighted_recoverability_score(
    subset_metrics: dict[str, dict[str, Any]],
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
        region_metrics = subset_metrics.get(region)
        if region_metrics is None or int(region_metrics.get("count", 0)) <= 0:
            continue
        numer += weight * float(region_metrics.get("success_rate", 0.0))
        denom += weight
    if denom <= 0.0:
        return 0.0
    return float(numer / denom)


def _evaluate_policy_jax(
    *,
    env_cfg: QuadrotorSAEnvConfig,
    reset_library: QuadrotorResetLibrary,
    actor_params,
    actor_cfg: ActorConfig,
    action_scale: jax.Array,
    action_low: jax.Array,
    action_high: jax.Array,
    split: str,
    recoverability_weights: QuadrotorRecoverabilityWeights,
    beta: float,
) -> dict[str, Any]:
    heldout = reset_library.heldout_reset_sets.get(split)
    if heldout is None:
        raise KeyError(f"Unknown held-out reset split: {split}")

    reset_states = np.asarray(heldout["states"], dtype=np.float32)
    reset_regions = np.asarray(heldout["region"]).astype(str)
    if reset_states.shape[0] == 0:
        overall = _aggregate_episode_records([])
        overall["weighted_recoverability_score"] = 0.0
        overall["subset_metrics"] = {
            region: _aggregate_episode_records([])
            for region in ("general_trace", "near_ceiling", "bridge", "base_shell")
        }
        overall["trajectory"] = {
            k: np.zeros((0,), dtype=np.float32)
            for k in ("obs", "next_obs", "act", "raw_action", "rew", "safe", "capture", "terminal", "hard_deck_margin")
        }
        return overall

    cbf_cfg = env_cfg.cbf_cfg
    safe_set = QuadrotorSafeSet.from_cbf_config(cbf_cfg)
    base_controller = QuadrotorDLQR.from_config(cbf_cfg)
    base_set = EllipsoidBaseSet(
        base_controller,
        float(cbf_cfg.base_set_c),
        smooth_gain=float(cbf_cfg.base_set_smooth_gain),
    )
    dt = jnp.asarray(env_cfg.dt, dtype=jnp.float32)
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
        applied = jnp.clip(
            BackupPolicy.select_action(x, raw, base_set),
            action_low,
            action_high,
        )
        x_next = quadrotor_step_euler(x, applied, cbf_cfg)
        safe = safe_set.contains(x_next)
        capture = base_set.contains(x_next)
        terminal = base_set.contains(x_next)
        goal = safe & terminal
        hard_margin = hard_deck_value(x_next, cbf_cfg)
        return x_next, safe, capture, terminal, goal, hard_margin, raw, applied

    def rollout_single(params, x0: jax.Array):
        safe0 = safe_set.contains(x0)
        terminal0 = base_set.contains(x0)
        goal0 = safe0 & terminal0

        def step(carry, k):
            (
                x,
                done,
                step_count,
                safe_sum,
                capture_sum,
                terminal_sum,
                safe_rollout,
                entered_terminal,
                left_after_entry,
                entry_step,
                crash_flag,
                min_hard_margin,
                post_entry_terminal_steps,
                post_entry_total_steps,
            ) = carry
            active = ~done
            raw_action = eval_raw_action(params, x)
            x_next, safe, capture, terminal, goal, hard_margin, raw, applied = transition(x, raw_action)
            done_now = active & (~safe)
            x_next = jnp.where(active, x_next, x)
            step_count = step_count + active.astype(jnp.int32)
            safe_sum = safe_sum + active.astype(jnp.float32) * safe.astype(jnp.float32)
            capture_sum = capture_sum + active.astype(jnp.float32) * capture.astype(jnp.float32)
            terminal_sum = terminal_sum + active.astype(jnp.float32) * terminal.astype(jnp.float32)
            safe_rollout = safe_rollout & jnp.where(active, safe, jnp.asarray(True, dtype=jnp.bool_))
            post_entry_active = active & (entered_terminal | goal)
            post_entry_terminal_steps = post_entry_terminal_steps + post_entry_active.astype(jnp.float32) * terminal.astype(jnp.float32)
            post_entry_total_steps = post_entry_total_steps + post_entry_active.astype(jnp.float32)
            entry_step = jnp.where(active & (~entered_terminal) & goal, k + jnp.int32(1), entry_step)
            left_after_entry = left_after_entry | (active & entered_terminal & (~goal))
            entered_terminal = entered_terminal | (active & goal)
            crash_flag = crash_flag | done_now
            min_hard_margin = jnp.minimum(min_hard_margin, jnp.where(active, hard_margin, min_hard_margin))
            record = {
                "obs": x,
                "next_obs": x_next,
                "act": jnp.where(active, applied, jnp.zeros_like(applied)),
                "raw_action": jnp.where(active, raw, jnp.zeros_like(raw)),
                "rew": jnp.asarray(0.0, dtype=jnp.float32),
                "safe": active.astype(jnp.float32) * safe.astype(jnp.float32),
                "capture": active.astype(jnp.float32) * capture.astype(jnp.float32),
                "terminal": active.astype(jnp.float32) * terminal.astype(jnp.float32),
                "hard_deck_margin": jnp.where(active, hard_margin, jnp.asarray(0.0, dtype=jnp.float32)),
                "done": done_now,
            }
            return (
                x_next,
                done | done_now,
                step_count,
                safe_sum,
                capture_sum,
                terminal_sum,
                safe_rollout,
                entered_terminal,
                left_after_entry,
                entry_step,
                crash_flag,
                min_hard_margin,
                post_entry_terminal_steps,
                post_entry_total_steps,
            ), record

        init = (
            jnp.asarray(x0, dtype=jnp.float32),
            jnp.asarray(False, dtype=jnp.bool_),
            jnp.int32(0),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(True, dtype=jnp.bool_),
            goal0,
            jnp.asarray(False, dtype=jnp.bool_),
            jnp.where(goal0, jnp.int32(0), jnp.int32(-1)),
            jnp.asarray(False, dtype=jnp.bool_),
            hard_deck_value(x0, cbf_cfg),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        final, records = jax.lax.scan(step, init, jnp.arange(horizon_steps, dtype=jnp.int32))
        entry_time_sec = jnp.where(final[9] >= 0, final[9].astype(jnp.float32) * dt, jnp.asarray(jnp.nan, dtype=jnp.float32))
        eval_len = jnp.maximum(final[2].astype(jnp.float32), jnp.asarray(1.0, dtype=jnp.float32))
        terminal_at_horizon = ((~final[10]) & safe_set.contains(final[0]) & base_set.contains(final[0])).astype(jnp.float32)
        invariance_after_entry = (final[7] & (~final[8])).astype(jnp.float32)
        strict_success = (final[6] & final[7] & (~final[8]) & (terminal_at_horizon > 0.5)).astype(jnp.float32)
        discounted_score = jnp.where(strict_success > 0.5, beta_arr ** final[9].astype(jnp.float32), jnp.asarray(0.0, dtype=jnp.float32))
        metrics = (
            strict_success,
            final[10].astype(jnp.float32),
            final[3] / eval_len,
            final[4] / eval_len,
            final[5] / eval_len,
            final[6].astype(jnp.float32),
            terminal_at_horizon,
            final[7].astype(jnp.float32),
            invariance_after_entry,
            discounted_score,
            entry_time_sec,
            final[11],
            final[12],
            final[13],
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
    for idx, region in enumerate(reset_regions.tolist()):
        episode_records.append(
            {
                "region": str(region),
                "success": float(metrics_np[0][idx]),
                "crash": float(metrics_np[1][idx]),
                "safe_rate": float(metrics_np[2][idx]),
                "capture_rate": float(metrics_np[3][idx]),
                "terminal_rate": float(metrics_np[4][idx]),
                "safe_rollout": float(metrics_np[5][idx]),
                "terminal_at_horizon": float(metrics_np[6][idx]),
                "entered_terminal": float(metrics_np[7][idx]),
                "invariance_after_entry": float(metrics_np[8][idx]),
                "discounted_ra_score": float(metrics_np[9][idx]),
                "entry_time_sec": float(metrics_np[10][idx]),
                "min_hard_deck_margin": float(metrics_np[11][idx]),
                "post_entry_terminal_steps": float(metrics_np[12][idx]),
                "post_entry_total_steps": float(metrics_np[13][idx]),
            }
        )

    overall = _aggregate_episode_records(episode_records)
    subset_metrics = {
        region: _aggregate_episode_records([rec for rec in episode_records if rec["region"] == region])
        for region in ("general_trace", "near_ceiling", "bridge", "base_shell")
    }
    overall["weighted_recoverability_score"] = float(_weighted_recoverability_score(subset_metrics, recoverability_weights))

    trace_raw = jax.device_get(rollout_trace(jnp.asarray(reset_states[0], dtype=jnp.float32)))
    done_steps = np.asarray(trace_raw["done"], dtype=bool)
    valid_steps = int(np.argmax(done_steps) + 1) if np.any(done_steps) else horizon_steps
    overall["subset_metrics"] = subset_metrics
    overall["trajectory"] = {
        "obs": np.asarray(trace_raw["obs"][:valid_steps]),
        "next_obs": np.asarray(trace_raw["next_obs"][:valid_steps]),
        "act": np.asarray(trace_raw["act"][:valid_steps]),
        "raw_action": np.asarray(trace_raw["raw_action"][:valid_steps]),
        "rew": np.asarray(trace_raw["rew"][:valid_steps]),
        "safe": np.asarray(trace_raw["safe"][:valid_steps]),
        "capture": np.asarray(trace_raw["capture"][:valid_steps]),
        "terminal": np.asarray(trace_raw["terminal"][:valid_steps]),
        "hard_deck_margin": np.asarray(trace_raw["hard_deck_margin"][:valid_steps]),
    }
    return overall


def _eval_rank_key(eval_stats: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float, float]:
    subset = eval_stats.get("subset_metrics", {})
    near_ceiling = float(subset.get("near_ceiling", {}).get("success_rate", 0.0))
    bridge = float(subset.get("bridge", {}).get("success_rate", 0.0))
    entry_median = eval_stats.get("entry_time_sec", {}).get("median")
    if entry_median is None:
        entry_rank = -1e9
    else:
        entry_rank = -float(entry_median)
    return (
        float(eval_stats.get("weighted_recoverability_score", 0.0)),
        -float(eval_stats.get("crash_rate", 1.0)),
        near_ceiling,
        bridge,
        float(eval_stats.get("terminal_at_horizon_rate", 0.0)),
        float(eval_stats.get("safe_rollout_rate", 0.0)),
        float(eval_stats.get("post_entry_terminal_step_rate", 0.0)),
        float(eval_stats.get("mean_discounted_ra_score", 0.0)),
        entry_rank,
    )


_EPISODE_METRIC_FIELDS: tuple[str, ...] = (
    "episode_len_sum",
    "episode_safe_rate_sum",
    "episode_capture_rate_sum",
    "episode_terminal_rate_sum",
    "episode_safe_rollout_sum",
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
    "eval_crash_rate",
    "eval_safe_rate",
    "eval_safe_rollout_rate",
    "eval_terminal_rate",
    "eval_entered_terminal_rate",
    "eval_post_entry_terminal_step_rate",
    "eval_mean_discounted_ra_score",
    "eval_weighted_recoverability_score",
)


def _episode_bookkeeping(info: Any, beta_arr: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    success = info.completed_entered_terminal

    entry_step_f = jnp.where(
        info.completed_entry_step >= 0,
        info.completed_entry_step.astype(jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    discounted_ra_score = jnp.where(
        info.completed_entered_terminal > 0.5,
        beta_arr ** entry_step_f,
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    fields = {
        "episode_len_sum": jnp.sum(info.completed_len),
        "episode_safe_rate_sum": jnp.sum(info.completed_safe_rate),
        "episode_capture_rate_sum": jnp.sum(info.completed_capture_rate),
        "episode_terminal_rate_sum": jnp.sum(info.completed_terminal_rate),
        "episode_safe_rollout_sum": jnp.sum(info.completed_safe_rollout),
        "episode_success_sum": jnp.sum(info.completed_entered_terminal),
        "episode_crash_sum": jnp.sum(info.is_crash * info.episode_done.astype(jnp.float32)),
        "episode_terminal_at_horizon_sum": jnp.sum(info.completed_terminal_at_horizon),
        "episode_discounted_ra_score_sum": jnp.sum(discounted_ra_score),
    }
    return success, fields


def _append_eval_history(history: dict[str, list[float]], eval_stats: dict[str, Any]) -> None:
    history["eval_success_rate"].append(float(eval_stats["success_rate"]))
    history["eval_crash_rate"].append(float(eval_stats["crash_rate"]))
    history["eval_safe_rate"].append(float(eval_stats["safe_rate"]))
    history["eval_safe_rollout_rate"].append(float(eval_stats["safe_rollout_rate"]))
    history["eval_terminal_rate"].append(float(eval_stats["terminal_at_horizon_rate"]))
    history["eval_entered_terminal_rate"].append(float(eval_stats["entered_terminal_rate"]))
    history["eval_post_entry_terminal_step_rate"].append(float(eval_stats["post_entry_terminal_step_rate"]))
    history["eval_mean_discounted_ra_score"].append(float(eval_stats["mean_discounted_ra_score"]))
    history["eval_weighted_recoverability_score"].append(float(eval_stats["weighted_recoverability_score"]))


def run_quadrotor_sa_training(
    ra_cfg: QuadrotorSAConfig,
    env_cfg: QuadrotorSAEnvConfig,
    reset_library: QuadrotorResetLibrary,
    *,
    recoverability_weights: QuadrotorRecoverabilityWeights,
    output_dir: str | None = None,
) -> dict[str, Any]:
    if int(ra_cfg.num_envs) <= 0:
        raise ValueError(f"num_envs must be positive, got {ra_cfg.num_envs}")
    if int(ra_cfg.steps_per_jit) <= 0:
        raise ValueError(f"steps_per_jit must be positive, got {ra_cfg.steps_per_jit}")
    if not (0.0 < float(ra_cfg.beta) < 1.0):
        raise ValueError(f"beta must lie in (0, 1), got {ra_cfg.beta}")

    num_envs = int(ra_cfg.num_envs)

    jax_key = make_prng_key(ra_cfg.seed)
    env_fns = build_quadrotor_sa_env(
        env_cfg,
        reset_library,
        split="train",
        terminate_on_goal=bool(ra_cfg.collector_terminate_on_goal),
    )

    action_scale_np = np.array(
        [env_cfg.cbf_cfg.a_cmd_max, env_cfg.cbf_cfg.omega_max, env_cfg.cbf_cfg.omega_max, env_cfg.cbf_cfg.omega_max],
        dtype=np.float32,
    )
    action_low_np = np.array(
        [env_cfg.cbf_cfg.a_cmd_min, -env_cfg.cbf_cfg.omega_max, -env_cfg.cbf_cfg.omega_max, -env_cfg.cbf_cfg.omega_max],
        dtype=np.float32,
    )
    action_high_np = np.array(
        [env_cfg.cbf_cfg.a_cmd_max, env_cfg.cbf_cfg.omega_max, env_cfg.cbf_cfg.omega_max, env_cfg.cbf_cfg.omega_max],
        dtype=np.float32,
    )
    action_scale = jnp.asarray(action_scale_np, dtype=jnp.float32)
    action_low = jnp.asarray(action_low_np, dtype=jnp.float32)
    action_high = jnp.asarray(action_high_np, dtype=jnp.float32)

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

    safe_set = QuadrotorSafeSet.from_cbf_config(env_cfg.cbf_cfg)
    base_controller = QuadrotorDLQR.from_config(env_cfg.cbf_cfg)
    base_set = EllipsoidBaseSet(
        base_controller,
        float(env_cfg.cbf_cfg.base_set_c),
        smooth_gain=float(env_cfg.cbf_cfg.base_set_smooth_gain),
    )

    # Goal region and hand-off region are both the base set B (the legacy
    # terminal and capture sets coincide after the unification).
    goal_contains_fn = _batch_safe_contains(lambda x: base_set.contains(x))
    fail_contains_fn = _batch_safe_contains(lambda x: jnp.logical_not(safe_set.contains(x)))
    handoff_contains_fn = _batch_safe_contains(lambda x: base_set.contains(x))

    jax_key, key_state, key_env = jax.random.split(jax_key, 3)
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
        return _evaluate_policy_jax(
            env_cfg=env_cfg,
            reset_library=reset_library,
            actor_params=actor_params,
            actor_cfg=actor_cfg,
            action_scale=action_scale,
            action_low=action_low,
            action_high=action_high,
            split=split,
            recoverability_weights=recoverability_weights,
            beta=ra_cfg.beta,
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
    test_eval = loop_result.test_eval
    total_time = loop_result.total_time

    summary = {
        "training_objective": "discounted_reach_avoid",
        "total_steps": int(ra_cfg.total_steps),
        "updates": int(jax.device_get(loop_state.updates)),
        "wall_time_sec": float(total_time),
        "steps_per_sec": float(ra_cfg.total_steps / max(total_time, 1e-6)),
        "horizon_T": float(env_cfg.horizon_T),
        "N_steps": int(env_cfg.horizon_steps),
        "beta": float(ra_cfg.beta),
        "goal_mode": str(ra_cfg.goal_mode),
        "use_handoff": bool(ra_cfg.use_handoff),
        "collector_terminate_on_goal": bool(ra_cfg.collector_terminate_on_goal),
        "final_curriculum_scale": float(jax.device_get(loop_state.curriculum_scale)),
        "final_eval_success_rate": float(val_eval["success_rate"]),
        "final_eval_crash_rate": float(val_eval["crash_rate"]),
        "final_eval_safe_rate": float(val_eval["safe_rate"]),
        "final_eval_safe_rollout_rate": float(val_eval["safe_rollout_rate"]),
        "final_eval_terminal_at_horizon_rate": float(val_eval["terminal_at_horizon_rate"]),
        "final_eval_entered_terminal_rate": float(val_eval["entered_terminal_rate"]),
        "final_eval_post_entry_terminal_step_rate": float(val_eval["post_entry_terminal_step_rate"]),
        "final_eval_mean_discounted_ra_score": float(val_eval["mean_discounted_ra_score"]),
        "final_eval_weighted_recoverability_score": float(val_eval["weighted_recoverability_score"]),
        "best_eval_step": int(best_eval_step),
        "best_eval_weighted_recoverability_score": float(
            best_eval_stats["weighted_recoverability_score"] if best_eval_stats is not None else val_eval["weighted_recoverability_score"]
        ),
        "best_eval_success_rate": float(best_eval_stats["success_rate"] if best_eval_stats is not None else val_eval["success_rate"]),
        "best_eval_crash_rate": float(best_eval_stats["crash_rate"] if best_eval_stats is not None else val_eval["crash_rate"]),
        "best_eval_safe_rollout_rate": float(best_eval_stats["safe_rollout_rate"] if best_eval_stats is not None else val_eval["safe_rollout_rate"]),
        "best_eval_terminal_at_horizon_rate": float(
            best_eval_stats["terminal_at_horizon_rate"] if best_eval_stats is not None else val_eval["terminal_at_horizon_rate"]
        ),
        "best_eval_entered_terminal_rate": float(
            best_eval_stats["entered_terminal_rate"] if best_eval_stats is not None else val_eval["entered_terminal_rate"]
        ),
        "best_eval_post_entry_terminal_step_rate": float(
            best_eval_stats["post_entry_terminal_step_rate"] if best_eval_stats is not None else val_eval["post_entry_terminal_step_rate"]
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
        "test_eval": {k: v for k, v in test_eval.items() if k != "trajectory"},
        "configs": {
            "backup_ra": asdict(ra_cfg),
            "backup_env": asdict(env_cfg),
            "actor": asdict(actor_cfg),
            "recoverability_weights": asdict(recoverability_weights),
        },
    }
    result["final_state"] = _snapshot_state(loop_state.state)
    result["best_state"] = best_state if best_state is not None else result["final_state"]

    if output_dir is not None:
        with open(f"{output_dir}/summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(f"{output_dir}/configs.json", "w", encoding="utf-8") as f:
            json.dump(result["configs"], f, indent=2)
    return result


__all__ = [
    "QuadrotorSAConfig",
    "QuadrotorRecoverabilityWeights",
    "_evaluate_policy_jax",
    "quadrotor_sa_config_from_dict",
    "quadrotor_recoverability_weights_from_dict",
    "run_quadrotor_sa_training",
]
