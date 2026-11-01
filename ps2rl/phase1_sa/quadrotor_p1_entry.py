#!/usr/bin/env python
"""Quadrotor Phase-1 learned backup-policy training (discounted safe-arrival).

The reusable arg-spec + reset-library build + SA-training + save-learned body for
the quadrotor Phase-1 trainer. The public entrypoint ``scripts/train_phase1.py
--system quadrotor`` calls ``main(argv)`` here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from math import exp, log
from pathlib import Path
import pickle

import numpy as np

try:
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    _HAS_MATPLOTLIB = False

from ps2rl.utils.paths import PROJECT_ROOT  # = the PS2-RL repo root

from ps2rl.backup_policy.quadrotor_learned_backup import load_learned_quadrotor_backup_policy
from ps2rl.backup_policy.backup_policy import save_learned_backup_policy
from ps2rl.phase1_sa.quadrotor_sa_env import QuadrotorSAEnvConfig
from ps2rl.evaluation.quadrotor_trace_reset_lib import (
    QuadrotorResetLibraryConfig,
    QuadrotorTraceSourceConfig,
    build_quadrotor_reset_library,
    export_quadrotor_reset_library_metadata,
)
from ps2rl.cil.quadrotor_backup_cbf import (
    QuadrotorBCBFConfig,
    quadrotor_hover_lqr_config_from_cbf_cfg,
)
from ps2rl.cil.cil_policy import ActorConfig
from ps2rl.phase1_sa.quadrotor_sa_trainer import (
    QuadrotorSAConfig,
    QuadrotorRecoverabilityWeights,
    run_quadrotor_sa_training,
)
try:
    from ps2rl.plotting.plots import plot_quad_trajectory
except Exception:  # pragma: no cover
    plot_quad_trajectory = None  # type: ignore[assignment]


def _sanitize_output_name(name: str) -> str:
    token = name.strip().replace("/", "_").replace("\\", "_")
    return token.strip("_-")


def _plot_quad_backup_training(history: dict[str, list[float]], output_path: Path, eval_every: int) -> None:
    if not _HAS_MATPLOTLIB:
        return
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    ax = axes.ravel()
    step = np.asarray(history.get("step", []), dtype=np.float64)

    ax[0].plot(step, np.asarray(history.get("ep_success", []), dtype=np.float64), label="Entered terminal", lw=1.0)
    ax[0].plot(step, np.asarray(history.get("ep_crash", []), dtype=np.float64), label="Crash", lw=1.0)
    ax[0].plot(step, np.asarray(history.get("ep_safe_rate", []), dtype=np.float64), label="Safe", lw=1.0)
    ax[0].set_title("Collector Outcomes")
    ax[0].set_xlabel("Step")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    ax[1].plot(step, np.asarray(history.get("ep_discounted_ra_score", []), dtype=np.float64), lw=1.0)
    ax[1].plot(step, np.asarray(history.get("ep_terminal_at_horizon", []), dtype=np.float64), lw=1.0, label="Terminal@H")
    ax[1].set_title("Collector Safe-Arrival Metrics")
    ax[1].set_xlabel("Step")
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    ax[2].plot(step, np.asarray(history.get("curriculum_scale", []), dtype=np.float64), lw=1.0)
    ax[2].plot(step, np.asarray(history.get("ep_terminal_rate", []), dtype=np.float64), lw=1.0, label="Terminal occupancy")
    ax[2].set_title("Curriculum / Occupancy")
    ax[2].set_xlabel("Step")
    ax[2].grid(alpha=0.3)
    ax[2].legend()

    ax[3].plot(np.asarray(history.get("critic_loss", []), dtype=np.float64), label="Critic", lw=1.0)
    ax[3].plot(np.asarray(history.get("actor_loss", []), dtype=np.float64), label="Actor", lw=1.0)
    ax[3].plot(np.asarray(history.get("action_penalty", []), dtype=np.float64), label="Act penalty", lw=1.0)
    ax[3].plot(np.asarray(history.get("target_mean", []), dtype=np.float64), label="Target", lw=1.0)
    ax[3].set_title("Update Metrics")
    ax[3].set_xlabel("Update metric index")
    ax[3].grid(alpha=0.3)
    ax[3].legend()

    eval_x = (np.arange(len(history.get("eval_success_rate", [])), dtype=np.float64) + 1.0) * float(eval_every)
    ax[4].plot(eval_x, np.asarray(history.get("eval_weighted_recoverability_score", []), dtype=np.float64), label="Weighted recoverability", lw=1.0)
    ax[4].plot(eval_x, np.asarray(history.get("eval_mean_discounted_ra_score", []), dtype=np.float64), label="Mean discounted RA", lw=1.0)
    ax[4].plot(
        eval_x,
        np.asarray(history.get("eval_post_entry_terminal_step_rate", []), dtype=np.float64),
        label="Post-entry terminal steps",
        lw=1.0,
    )
    ax[4].set_title("Held-out Safe-Arrival Metrics")
    ax[4].set_xlabel("Step")
    ax[4].grid(alpha=0.3)
    ax[4].legend()

    ax[5].plot(eval_x, np.asarray(history.get("eval_success_rate", []), dtype=np.float64), label="Strict success", lw=1.0)
    ax[5].plot(eval_x, np.asarray(history.get("eval_entered_terminal_rate", []), dtype=np.float64), label="Entered terminal", lw=1.0)
    ax[5].plot(eval_x, np.asarray(history.get("eval_crash_rate", []), dtype=np.float64), label="Crash", lw=1.0)
    ax[5].plot(eval_x, np.asarray(history.get("eval_safe_rollout_rate", []), dtype=np.float64), label="Safe rollout", lw=1.0)
    ax[5].plot(eval_x, np.asarray(history.get("eval_terminal_rate", []), dtype=np.float64), label="Terminal@H", lw=1.0)
    ax[5].set_title("Held-out Validation Rates")
    ax[5].set_xlabel("Step")
    ax[5].grid(alpha=0.3)
    ax[5].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the quadrotor phase-1 learned backup policy with discounted safe-arrival.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=2000000)
    parser.add_argument("--start_steps", type=int, default=5000)
    parser.add_argument("--update_after", type=int, default=2000)
    parser.add_argument("--update_every", type=int, default=8)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--replay_size", type=int, default=400000)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--actor_log_std_min", type=float, default=-7.0)
    parser.add_argument("--actor_log_std_max", type=float, default=-2.5)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--record_update_metrics", action="store_true", dest="record_update_metrics")
    parser.add_argument("--no_record_update_metrics", action="store_false", dest="record_update_metrics")
    parser.set_defaults(record_update_metrics=True)
    parser.add_argument("--update_metric_every", type=int, default=200)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--steps_per_jit", type=int, default=128)

    parser.add_argument("--beta", type=float, default=0.0, help="If <= 0, beta is derived from --beta_horizon_value and --num_steps.")
    parser.add_argument("--beta_horizon_value", type=float, default=0.2, help="Choose beta so beta^num_steps ~= beta_horizon_value.")
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_delay", type=int, default=2)
    parser.add_argument("--action_smoothness_weight", type=float, default=0.0)
    parser.add_argument("--exploration_std", type=float, default=0.10)
    parser.add_argument("--exploration_clip", type=float, default=0.25)
    parser.add_argument("--target_policy_noise_std", type=float, default=0.0)
    parser.add_argument("--target_policy_noise_clip", type=float, default=0.0)
    parser.add_argument("--critic_huber_delta", type=float, default=1.0)
    parser.add_argument("--use_handoff", action="store_true", default=True)
    parser.add_argument("--no_use_handoff", action="store_false", dest="use_handoff")
    parser.add_argument("--goal_mode", type=str, default="terminal", choices=["terminal"])
    parser.add_argument("--collector_terminate_on_goal", action="store_true", default=True)
    parser.add_argument("--no_collector_terminate_on_goal", action="store_false", dest="collector_terminate_on_goal")

    parser.add_argument("--curriculum_start_scale", type=float, default=0.0)
    parser.add_argument("--curriculum_increment", type=float, default=0.10)
    parser.add_argument("--curriculum_success_threshold", type=float, default=0.90)
    parser.add_argument("--curriculum_window_episodes", type=int, default=100)
    parser.add_argument("--curriculum_min_episodes", type=int, default=100)

    parser.add_argument("--horizon_T", type=float, default=2.0)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--a_cmd_min", type=float, default=0.0)
    parser.add_argument("--a_cmd_max_g", type=float, default=4.0)
    parser.add_argument("--omega_max", type=float, default=18.0)
    parser.add_argument("--z_max", type=float, default=3.0)
    parser.add_argument("--z_des", type=float, default=2.0)

    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--base_alpha", type=float, default=2.0)
    parser.add_argument(
        "--base_set_c",
        type=float,
        default=None,
        help=(
            "Base-set level c in h_B(x)=c-x_err^T P x_err (paper's base set B; "
            "the unified terminal set == capture set)."
        ),
    )
    parser.add_argument("--lqr_q_z", type=float, default=1.0)
    parser.add_argument("--lqr_q_vx", type=float, default=0.16)
    parser.add_argument("--lqr_q_vy", type=float, default=0.16)
    parser.add_argument("--lqr_q_vz", type=float, default=0.4)
    parser.add_argument("--lqr_q_thetax", type=float, default=0.8)
    parser.add_argument("--lqr_q_thetay", type=float, default=0.8)
    parser.add_argument("--lqr_q_thetaz", type=float, default=0.16)
    parser.add_argument("--lqr_r_a_cmd", type=float, default=0.02)
    parser.add_argument("--lqr_r_omega_x", type=float, default=0.012)
    parser.add_argument("--lqr_r_omega_y", type=float, default=0.012)
    parser.add_argument("--lqr_r_omega_z", type=float, default=0.004)
    parser.add_argument("--control_weight", type=float, default=1.0)
    parser.add_argument("--solver_tol", type=float, default=5e-4)
    parser.add_argument("--target_kappa", type=float, default=1e-2)

    parser.add_argument("--vanilla_quadtrack_outputs_dir", type=str, default="outputs_objE_quadTrack_vanilla_withOmega_longRuns")
    parser.add_argument("--reference_path", type=str, default="ps2rl/envs/assets/quadrotor_powerloop_reference.npz")
    parser.add_argument("--reference_run_name", action="append", default=[])
    parser.add_argument("--reference_run_glob", type=str, default="sac_quadTrackOmega-*")
    parser.add_argument("--reference_glob", type=str, default="evaluation/quadTrack_eval-*/best_episode_trace.npz")
    parser.add_argument("--max_traces", type=int, default=20)
    parser.add_argument("--trace_set_label", type=str, default="omega_runIdx_0to19")
    parser.add_argument("--staged_trace_dir", type=str, default="checkpoints/quadrotor_vanilla/vanilla_traces")
    parser.add_argument("--staged_trace_glob", type=str, default="trace_seed*_runIdx*.npz")
    parser.add_argument("--select_seed", type=int, default=0)
    parser.add_argument("--select_run_idx_min", type=int, default=0)
    parser.add_argument("--select_run_idx_max", type=int, default=19)

    parser.add_argument("--near_ceiling_margin", type=float, default=0.25)
    parser.add_argument("--bridge_num_interp", type=int, default=41)
    parser.add_argument("--base_shell_distance", type=float, default=0.10)
    parser.add_argument("--base_shell_terminal_margin", type=float, default=0.10)
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--position_perturb_min", type=float, default=0.00)
    parser.add_argument("--position_perturb_max", type=float, default=0.40)
    parser.add_argument("--velocity_perturb_min", type=float, default=0.00)
    parser.add_argument("--velocity_perturb_max", type=float, default=2.50)
    parser.add_argument("--tilt_perturb_deg_min", type=float, default=0.00)
    parser.add_argument("--tilt_perturb_deg_max", type=float, default=35.0)
    parser.add_argument("--yaw_perturb_deg_min", type=float, default=0.00)
    parser.add_argument("--yaw_perturb_deg_max", type=float, default=12.0)
    parser.add_argument("--general_region_multiplier", type=float, default=1.0)
    parser.add_argument("--near_ceiling_region_multiplier", type=float, default=1.5)
    parser.add_argument("--bridge_region_multiplier", type=float, default=1.8)
    parser.add_argument("--base_shell_region_multiplier", type=float, default=0.8)
    parser.add_argument("--mix_general_low", type=float, default=0.45)
    parser.add_argument("--mix_general_high", type=float, default=0.20)
    parser.add_argument("--mix_near_ceiling_low", type=float, default=0.15)
    parser.add_argument("--mix_near_ceiling_high", type=float, default=0.35)
    parser.add_argument("--mix_bridge_low", type=float, default=0.05)
    parser.add_argument("--mix_bridge_high", type=float, default=0.30)
    parser.add_argument("--mix_base_shell_low", type=float, default=0.35)
    parser.add_argument("--mix_base_shell_high", type=float, default=0.15)
    parser.add_argument("--max_resample_tries", type=int, default=80)
    parser.add_argument("--heldout_val_per_region", type=int, default=64)
    parser.add_argument("--heldout_test_per_region", type=int, default=64)
    parser.add_argument("--heldout_seed", type=int, default=1234)

    parser.add_argument("--weight_general_trace", type=float, default=1.0)
    parser.add_argument("--weight_near_ceiling", type=float, default=2.0)
    parser.add_argument("--weight_bridge", type=float, default=2.5)
    parser.add_argument("--weight_base_shell", type=float, default=1.0)

    parser.add_argument("--output_root", type=str, default="outputs_objE_quadBackup_policy_ra")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--smoke_test", action="store_true", default=False)
    parser.add_argument("--build_reset_library_only", action="store_true", default=False,
                        help="Build + save the reset library, then exit (skip SA-training).")
    args = parser.parse_args(argv)
    args.base_set_c = _resolve_base_set_c(parser, args)
    return args


def _resolve_base_set_c(parser: argparse.ArgumentParser, args: argparse.Namespace) -> float:
    """Resolve the unified LQR base-set level from --base_set_c."""
    if args.base_set_c is None:
        parser.error("--base_set_c is required.")
    return float(args.base_set_c)


def _apply_smoke_test_defaults(args: argparse.Namespace) -> None:
    args.total_steps = min(int(args.total_steps), 800)
    args.start_steps = min(int(args.start_steps), 128)
    args.update_after = min(int(args.update_after), 64)
    args.update_every = min(int(args.update_every), 4)
    args.batch_size = min(int(args.batch_size), 64)
    args.replay_size = min(int(args.replay_size), 20000)
    args.eval_every = min(int(args.eval_every), 200)
    args.log_every = min(int(args.log_every), 100)
    args.num_envs = min(int(args.num_envs), 8)
    args.steps_per_jit = min(int(args.steps_per_jit), 16)
    args.max_traces = min(int(args.max_traces), 2)
    args.heldout_val_per_region = min(int(args.heldout_val_per_region), 6)
    args.heldout_test_per_region = min(int(args.heldout_test_per_region), 6)


def _resolve_beta(beta: float, *, beta_horizon_value: float, num_steps: int) -> float:
    if beta > 0.0:
        return float(beta)
    if not (0.0 < beta_horizon_value < 1.0):
        raise ValueError(f"beta_horizon_value must lie in (0, 1), got {beta_horizon_value}")
    if int(num_steps) <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    return float(exp(log(float(beta_horizon_value)) / float(num_steps)))


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.smoke_test:
        _apply_smoke_test_defaults(args)

    beta = _resolve_beta(args.beta, beta_horizon_value=args.beta_horizon_value, num_steps=args.num_steps)

    tag = args.run_tag.strip() if args.run_tag.strip() else datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = _sanitize_output_name(args.output_dir) if args.output_dir else ""
    run_name = f"{tag}-{suffix}" if suffix else tag
    run_dir = (PROJECT_ROOT / "outputs" / args.output_root / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    a_cmd_max = float(args.a_cmd_max_g) * float(args.gravity)
    cbf_cfg = QuadrotorBCBFConfig(
        T=float(args.horizon_T),
        num_steps=int(args.num_steps),
        alpha=float(args.alpha),
        gravity=float(args.gravity),
        a_cmd_min=float(args.a_cmd_min),
        a_cmd_max=float(a_cmd_max),
        omega_max=float(args.omega_max),
        z_max=float(args.z_max),
        z_des=float(args.z_des),
        base_alpha=float(args.base_alpha),
        base_set_c=float(args.base_set_c),
        lqr_q_z=float(args.lqr_q_z),
        lqr_q_vx=float(args.lqr_q_vx),
        lqr_q_vy=float(args.lqr_q_vy),
        lqr_q_vz=float(args.lqr_q_vz),
        lqr_q_thetax=float(args.lqr_q_thetax),
        lqr_q_thetay=float(args.lqr_q_thetay),
        lqr_q_thetaz=float(args.lqr_q_thetaz),
        lqr_r_a_cmd=float(args.lqr_r_a_cmd),
        lqr_r_omega_x=float(args.lqr_r_omega_x),
        lqr_r_omega_y=float(args.lqr_r_omega_y),
        lqr_r_omega_z=float(args.lqr_r_omega_z),
        control_weight=float(args.control_weight),
        solver_tol=float(args.solver_tol),
        target_kappa=float(args.target_kappa),
        backup_policy_mode="analytic",
    )
    env_cfg = QuadrotorSAEnvConfig(cbf_cfg=cbf_cfg)
    ra_cfg = QuadrotorSAConfig(
        seed=int(args.seed),
        total_steps=int(args.total_steps),
        start_steps=int(args.start_steps),
        update_after=int(args.update_after),
        update_every=int(args.update_every),
        gradient_steps=int(args.gradient_steps),
        batch_size=int(args.batch_size),
        replay_size=int(args.replay_size),
        beta=float(beta),
        tau=float(args.tau),
        policy_delay=int(args.policy_delay),
        action_smoothness_weight=float(args.action_smoothness_weight),
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        max_grad_norm=float(args.max_grad_norm),
        critic_huber_delta=float(args.critic_huber_delta),
        hidden_size=int(args.hidden_size),
        actor_log_std_min=float(args.actor_log_std_min),
        actor_log_std_max=float(args.actor_log_std_max),
        exploration_std=float(args.exploration_std),
        exploration_clip=float(args.exploration_clip),
        target_policy_noise_std=float(args.target_policy_noise_std),
        target_policy_noise_clip=float(args.target_policy_noise_clip),
        eval_every=int(args.eval_every),
        log_every=int(args.log_every),
        record_update_metrics=bool(args.record_update_metrics),
        update_metric_every=int(args.update_metric_every),
        num_envs=int(args.num_envs),
        steps_per_jit=int(args.steps_per_jit),
        curriculum_start_scale=float(args.curriculum_start_scale),
        curriculum_increment=float(args.curriculum_increment),
        curriculum_success_threshold=float(args.curriculum_success_threshold),
        curriculum_window_episodes=int(args.curriculum_window_episodes),
        curriculum_min_episodes=int(args.curriculum_min_episodes),
        use_handoff=bool(args.use_handoff),
        goal_mode=str(args.goal_mode),
        collector_terminate_on_goal=bool(args.collector_terminate_on_goal),
        smoke_test=bool(args.smoke_test),
    )
    trace_source_cfg = QuadrotorTraceSourceConfig(
        vanilla_quadtrack_outputs_dir=str(args.vanilla_quadtrack_outputs_dir),
        reference_path=str(args.reference_path),
        reference_run_name=tuple(args.reference_run_name),
        reference_run_glob=str(args.reference_run_glob),
        reference_glob=str(args.reference_glob),
        max_traces=int(args.max_traces),
        trace_set_label=str(args.trace_set_label),
        staged_trace_dir=str(args.staged_trace_dir),
        staged_trace_glob=str(args.staged_trace_glob),
        select_seed=int(args.select_seed),
        select_run_idx_min=int(args.select_run_idx_min),
        select_run_idx_max=int(args.select_run_idx_max),
    )
    reset_cfg = QuadrotorResetLibraryConfig(
        near_ceiling_margin=float(args.near_ceiling_margin),
        bridge_num_interp=int(args.bridge_num_interp),
        base_shell_distance=float(args.base_shell_distance),
        base_shell_terminal_margin=float(args.base_shell_terminal_margin),
        train_fraction=float(args.train_fraction),
        val_fraction=float(args.val_fraction),
        test_fraction=float(args.test_fraction),
        split_seed=int(args.split_seed),
        position_perturb_min=float(args.position_perturb_min),
        position_perturb_max=float(args.position_perturb_max),
        velocity_perturb_min=float(args.velocity_perturb_min),
        velocity_perturb_max=float(args.velocity_perturb_max),
        tilt_perturb_deg_min=float(args.tilt_perturb_deg_min),
        tilt_perturb_deg_max=float(args.tilt_perturb_deg_max),
        yaw_perturb_deg_min=float(args.yaw_perturb_deg_min),
        yaw_perturb_deg_max=float(args.yaw_perturb_deg_max),
        general_region_multiplier=float(args.general_region_multiplier),
        near_ceiling_region_multiplier=float(args.near_ceiling_region_multiplier),
        bridge_region_multiplier=float(args.bridge_region_multiplier),
        base_shell_region_multiplier=float(args.base_shell_region_multiplier),
        mix_general_low=float(args.mix_general_low),
        mix_general_high=float(args.mix_general_high),
        mix_near_ceiling_low=float(args.mix_near_ceiling_low),
        mix_near_ceiling_high=float(args.mix_near_ceiling_high),
        mix_bridge_low=float(args.mix_bridge_low),
        mix_bridge_high=float(args.mix_bridge_high),
        mix_base_shell_low=float(args.mix_base_shell_low),
        mix_base_shell_high=float(args.mix_base_shell_high),
        max_resample_tries=int(args.max_resample_tries),
        heldout_val_per_region=int(args.heldout_val_per_region),
        heldout_test_per_region=int(args.heldout_test_per_region),
        heldout_seed=int(args.heldout_seed),
    )
    recoverability_weights = QuadrotorRecoverabilityWeights(
        general_trace=float(args.weight_general_trace),
        near_ceiling=float(args.weight_near_ceiling),
        bridge=float(args.weight_bridge),
        base_shell=float(args.weight_base_shell),
    )

    print(
        f"Building reset library from {trace_source_cfg.max_traces} trace files "
        f"with T={cbf_cfg.T:.3f}, N={cbf_cfg.num_steps}, z_max={cbf_cfg.z_max:.2f}, "
        f"base_set_c={cbf_cfg.base_set_c:.3f}"
    )
    reset_library = build_quadrotor_reset_library(
        trace_source_cfg=trace_source_cfg,
        library_cfg=reset_cfg,
        cbf_cfg=cbf_cfg,
    )
    reset_library.save(run_dir / "reset_library.pkl")
    reset_metadata = export_quadrotor_reset_library_metadata(reset_library, output_dir=run_dir)

    if args.build_reset_library_only:
        print(f"Reset library built and saved to {run_dir / 'reset_library.pkl'}; "
              "skipping SA-training (--build_reset_library_only).")
        return

    print(
        f"Training discounted safe-arrival policy with beta={beta:.6f}, "
        f"goal_mode={args.goal_mode}, use_handoff={args.use_handoff}"
    )
    result = run_quadrotor_sa_training(
        ra_cfg,
        env_cfg,
        reset_library,
        recoverability_weights=recoverability_weights,
        output_dir=str(run_dir),
    )

    history = result["history"]
    summary = result["summary"]
    val_eval = result["eval"]
    best_eval = result["best_eval"]
    test_eval = result["test_eval"]
    configs = {
        **result["configs"],
        "trace_source": asdict(trace_source_cfg),
        "reset_library": asdict(reset_cfg),
    }
    final_state = result["final_state"]
    best_state = result.get("best_state", final_state)

    with open(run_dir / "final_weights.pkl", "wb") as f:
        pickle.dump(final_state, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(run_dir / "best_weights.pkl", "wb") as f:
        pickle.dump(best_state, f, protocol=pickle.HIGHEST_PROTOCOL)

    actor_cfg_payload = dict(configs.get("actor", {}))
    if "hidden_sizes" in actor_cfg_payload:
        actor_cfg_payload["hidden_sizes"] = tuple(actor_cfg_payload["hidden_sizes"])
    policy_ckpt = run_dir / "quad_backup_policy_actor.pkl"
    save_learned_backup_policy(
        path=policy_ckpt,
        actor_params=best_state["actor_params"],
        actor_cfg=ActorConfig(**actor_cfg_payload),
        action_scale=np.asarray(
            [cbf_cfg.a_cmd_max, cbf_cfg.omega_max, cbf_cfg.omega_max, cbf_cfg.omega_max],
            dtype=np.float32,
        ),
        action_low=np.asarray([cbf_cfg.a_cmd_min, -cbf_cfg.omega_max, -cbf_cfg.omega_max, -cbf_cfg.omega_max], dtype=np.float32),
        action_high=np.asarray([cbf_cfg.a_cmd_max, cbf_cfg.omega_max, cbf_cfg.omega_max, cbf_cfg.omega_max], dtype=np.float32),
        metadata={
            "seed": int(args.seed),
            "training_objective": "discounted_reach_avoid",
            "goal_mode": str(args.goal_mode),
            "beta": float(beta),
            "use_handoff": bool(args.use_handoff),
            "collector_terminate_on_goal": bool(args.collector_terminate_on_goal),
            "observation_feature_mode": "raw_10d_physical_state",
            "horizon_T": float(args.horizon_T),
            "num_steps": int(args.num_steps),
            "z_max": float(args.z_max),
            "base_set_c": float(cbf_cfg.base_set_c),
            "lqr_config": quadrotor_hover_lqr_config_from_cbf_cfg(cbf_cfg),
            "best_eval_step": int(summary.get("best_eval_step", args.total_steps)),
            "curriculum_config": {
                "start": float(args.curriculum_start_scale),
                "increment": float(args.curriculum_increment),
                "success_threshold": float(args.curriculum_success_threshold),
                "window_episodes": int(args.curriculum_window_episodes),
                "min_episodes": int(args.curriculum_min_episodes),
            },
            "reset_library_metadata": {
                "trace_set_label": str(args.trace_set_label),
                "num_trace_files_loaded": int(reset_metadata["num_trace_files_loaded"]),
                "pool_counts_total": reset_metadata["pool_counts_total"],
            },
        },
    )

    loaded_policy = load_learned_quadrotor_backup_policy(policy_ckpt)
    policy_probe_states = np.asarray(reset_library.heldout_reset_sets["val"]["states"][:8], dtype=np.float32)
    probe_action_a = np.asarray(loaded_policy.action_batch(policy_probe_states), dtype=np.float64)
    probe_action_b = np.asarray(loaded_policy.action_batch(policy_probe_states), dtype=np.float64)
    policy_load_check = {
        "num_probe_states": int(policy_probe_states.shape[0]),
        "deterministic": bool(np.allclose(probe_action_a, probe_action_b)),
        "within_bounds": bool(
            np.all(probe_action_a[:, 0] >= cbf_cfg.a_cmd_min - 1e-6)
            and np.all(probe_action_a[:, 0] <= cbf_cfg.a_cmd_max + 1e-6)
            and np.all(np.abs(probe_action_a[:, 1:]) <= cbf_cfg.omega_max + 1e-6)
        ),
        "action_min": probe_action_a.min(axis=0).tolist() if probe_action_a.size else [],
        "action_max": probe_action_a.max(axis=0).tolist() if probe_action_a.size else [],
    }

    np.savez(run_dir / "history.npz", **{k: np.asarray(v, dtype=np.float64) for k, v in history.items()})
    np.savez(run_dir / "val_eval_trajectory.npz", **{k: np.asarray(v) for k, v in val_eval["trajectory"].items()})
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(run_dir / "configs.json", "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2)
    with open(run_dir / "best_eval.json", "w", encoding="utf-8") as f:
        json.dump(best_eval, f, indent=2)
    with open(run_dir / "val_eval.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in val_eval.items() if k != "trajectory"}, f, indent=2)
    with open(run_dir / "test_eval.json", "w", encoding="utf-8") as f:
        json.dump(test_eval, f, indent=2)
    with open(run_dir / "policy_load_check.json", "w", encoding="utf-8") as f:
        json.dump(policy_load_check, f, indent=2)
    _plot_quad_backup_training(history, run_dir / "training_metrics.png", eval_every=ra_cfg.eval_every)
    if plot_quad_trajectory is not None:
        plot_quad_trajectory(
            val_eval["trajectory"],
            z_max=cbf_cfg.z_max,
            output_path=str(run_dir / "val_eval_trajectory.png"),
            dt=cbf_cfg.dt,
        )

    print("Done.")
    print(f"Saved outputs to: {run_dir}")
    print(f"Best checkpoint step: {summary.get('best_eval_step', args.total_steps)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
