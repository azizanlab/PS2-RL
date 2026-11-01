#!/usr/bin/env python
"""Unicycle Phase-1 safe-arrival backup-policy training + invariant-set compare.

The reusable arg-spec + SA-training + inline invariant-compare body for the
unicycle Phase-1 trainer. The public entrypoint ``scripts/train_phase1.py
--system unicycle`` calls ``main(argv)`` here.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parents[2] / "matplotlib-cache")))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ps2rl.utils.paths import PROJECT_ROOT  # = the PS2-RL repo root

from ps2rl.evaluation.invariant_compare import InvariantGridConfig, compare_invariant_sets
from ps2rl.cil.unicycle_backup_cbf import UnicycleBCBFConfig
from ps2rl.phase1_sa.unicycle_sa_env import UnicycleSAEnvConfig
from ps2rl.phase1_sa.unicycle_sa_trainer import (
    UnicycleSAConfig,
    run_unicycle_sa_training,
)


def _sanitize_output_name(name: str) -> str:
    s = name.strip()
    s = s.replace("/", "_").replace("\\", "_")
    return s.strip("_-")


def _to_bool_flag(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean flag value, got: {value}")


def _resolve_base_set_c(parser: argparse.ArgumentParser, args: argparse.Namespace) -> float:
    """Resolve the unified LQR base-set level from --base_set_c."""
    if args.base_set_c is None:
        parser.error("--base_set_c is required.")
    return float(args.base_set_c)


def _compute_dt(horizon_T: float, num_steps: int, *, label: str) -> float:
    horizon = float(horizon_T)
    steps = int(num_steps)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError(f"{label}_horizon_T must be positive and finite, got {horizon_T}")
    if steps <= 0:
        raise ValueError(f"{label}_num_steps must be positive, got {num_steps}")
    return horizon / float(steps)


def _plot_backup_training(history: dict[str, list[float]], output_path: Path, eval_every: int) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    ax = axes.ravel()

    step = np.asarray(history.get("step", []), dtype=np.float64)
    eval_x = np.arange(len(history.get("eval_success_rate", [])), dtype=np.float64) * float(eval_every)

    ax[0].plot(step, np.asarray(history.get("ep_success", []), dtype=np.float64), label="Success", lw=1.0)
    ax[0].plot(step, np.asarray(history.get("ep_capture_success", []), dtype=np.float64), label="Capture", lw=1.0)
    ax[0].plot(step, np.asarray(history.get("ep_crash", []), dtype=np.float64), label="Crash", lw=1.0)
    ax[0].set_title("Episode Outcomes")
    ax[0].set_xlabel("Step")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    ax[1].plot(step, np.asarray(history.get("curriculum_scale", []), dtype=np.float64), lw=1.0)
    ax[1].set_title("Curriculum Scale")
    ax[1].set_xlabel("Step")
    ax[1].grid(alpha=0.3)

    ax[2].plot(np.asarray(history.get("critic_loss", []), dtype=np.float64), label="Critic", lw=1.0)
    ax[2].plot(np.asarray(history.get("actor_loss", []), dtype=np.float64), label="Actor", lw=1.0)
    ax[2].plot(np.asarray(history.get("action_penalty", []), dtype=np.float64), label="Action penalty", lw=1.0)
    ax[2].set_title("Update Metrics")
    ax[2].set_xlabel("Update metric index")
    ax[2].grid(alpha=0.3)
    ax[2].legend()

    ax[3].plot(eval_x, np.asarray(history.get("eval_success_rate", []), dtype=np.float64), label="Success", lw=1.0)
    ax[3].plot(eval_x, np.asarray(history.get("eval_capture_success_rate", []), dtype=np.float64), label="Capture", lw=1.0)
    ax[3].plot(eval_x, np.asarray(history.get("eval_crash_rate", []), dtype=np.float64), label="Crash", lw=1.0)
    ax[3].set_title("Validation Outcomes")
    ax[3].set_xlabel("Step")
    ax[3].grid(alpha=0.3)
    ax[3].legend()

    ax[4].plot(
        eval_x,
        np.asarray(history.get("eval_terminal_at_horizon_rate", []), dtype=np.float64),
        label="Terminal@H",
        lw=1.0,
    )
    ax[4].plot(
        eval_x,
        np.asarray(history.get("eval_post_capture_terminal_success_rate", []), dtype=np.float64),
        label="Capture->Terminal",
        lw=1.0,
    )
    ax[4].plot(
        eval_x,
        np.asarray(history.get("eval_invariance_after_terminal_entry_rate", []), dtype=np.float64),
        label="Post-terminal invariance",
        lw=1.0,
    )
    ax[4].set_title("Validation Hybrid Metrics")
    ax[4].set_xlabel("Step")
    ax[4].grid(alpha=0.3)
    ax[4].legend()

    ax[5].plot(
        eval_x,
        np.asarray(history.get("eval_mean_discounted_ra_score", []), dtype=np.float64),
        label="Discounted RA score",
        lw=1.0,
    )
    ax[5].set_title("Validation Discounted Score")
    ax[5].set_xlabel("Step")
    ax[5].grid(alpha=0.3)
    ax[5].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train lane discounted safe-arrival policy.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=2000000)
    parser.add_argument("--start_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--update_every", type=int, default=8)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--update_after", type=int, default=2000)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--record_update_metrics", type=_to_bool_flag, default=True)
    parser.add_argument("--update_metric_every", type=int, default=200)

    parser.add_argument("--beta", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_delay", type=int, default=2)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--critic_huber_delta", type=float, default=1.0)
    parser.add_argument("--action_smoothness_weight", type=float, default=0.1)
    parser.add_argument("--actor_log_std_min", type=float, default=-8.0)
    parser.add_argument("--actor_log_std_max", type=float, default=-3.8)
    parser.add_argument("--exploration_std", type=float, default=0.10)
    parser.add_argument("--exploration_clip", type=float, default=0.25)
    parser.add_argument("--target_policy_noise_std", type=float, default=0.0)
    parser.add_argument("--target_policy_noise_clip", type=float, default=0.0)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--steps_per_jit", type=int, default=128)
    parser.add_argument("--val_reset_count", type=int, default=128)
    parser.add_argument("--test_reset_count", type=int, default=128)
    parser.add_argument("--collector_terminate_on_goal", type=_to_bool_flag, default=True)
    parser.add_argument("--terminate_on_crash", type=_to_bool_flag, default=True)

    parser.add_argument(
        "--train_num_steps",
        "--num_steps",
        dest="train_num_steps",
        type=int,
        default=40,
        help="Training horizon steps N. --num_steps is kept as a backward-compatible alias.",
    )
    parser.add_argument(
        "--train_horizon_T",
        "--horizon_T",
        dest="train_horizon_T",
        type=float,
        default=2.0,
        help="Training horizon length T. --horizon_T is kept as a backward-compatible alias.",
    )
    parser.add_argument("--eval_num_steps", type=int, default=20, help="Evaluation/invariant-comparison horizon steps N.")
    parser.add_argument("--eval_horizon_T", type=float, default=1.0, help="Evaluation/invariant-comparison horizon length T.")

    parser.add_argument("--y_max", type=float, default=1.8)
    parser.add_argument("--psi_max", type=float, default=float(np.pi / 3.0))
    parser.add_argument("--a_max", type=float, default=5.0)
    parser.add_argument("--r_max", type=float, default=1.0)
    parser.add_argument("--v_min", type=float, default=0.0)
    parser.add_argument("--v_max", type=float, default=12.0)
    parser.add_argument("--v_des", type=float, default=5.0)

    parser.add_argument(
        "--base_set_c",
        type=float,
        default=None,
        help="Level of the unified LQR base set B.",
    )
    parser.add_argument("--lqr_q_y", type=float, default=1.0)
    parser.add_argument("--lqr_q_v", type=float, default=1.0)
    parser.add_argument("--lqr_q_psi", type=float, default=1.0)
    parser.add_argument("--lqr_r_a", type=float, default=1.0)
    parser.add_argument("--lqr_r_r", type=float, default=1.0)

    parser.add_argument("--curriculum_start_scale", type=float, default=0.2)
    parser.add_argument("--curriculum_increment", type=float, default=0.005)
    parser.add_argument("--curriculum_success_threshold", type=float, default=0.90)
    parser.add_argument("--curriculum_window_episodes", type=int, default=50)
    parser.add_argument("--curriculum_min_episodes", type=int, default=50)

    parser.add_argument("--init_y_range_min", type=float, default=0.08)
    parser.add_argument("--init_y_range_max", type=float, default=1.5)
    parser.add_argument("--init_v_range_min", type=float, default=0.10)
    parser.add_argument("--init_v_range_max", type=float, default=3.0)
    parser.add_argument("--init_psi_range_min", type=float, default=0.03)
    parser.add_argument("--init_psi_range_max", type=float, default=0.50)

    parser.add_argument("--compare_num_y", type=int, default=121)
    parser.add_argument("--compare_num_psi", type=int, default=121)
    parser.add_argument("--compare_num_v", type=int, default=25)
    parser.add_argument("--compare_v_min", type=float, default=2.0)
    parser.add_argument("--compare_v_max", type=float, default=8.0)
    parser.add_argument("--compare_max_scatter_points", type=int, default=20000)
    parser.add_argument("--skip_invariant_compare", type=_to_bool_flag, default=False)
    parser.add_argument(
        "--invariant_compare_checkpoint",
        type=str,
        choices=("best", "final"),
        default="best",
        help="Which checkpoint to use for invariant comparison.",
    )

    parser.add_argument("--output_root", type=str, default="outputs_objC_RA_run")
    parser.add_argument("--output_dir", type=str, default="", help="Run-name suffix.")
    parser.add_argument("--run_tag", type=str, default="", help="Optional fixed timestamp tag.")
    args = parser.parse_args(argv)
    args.base_set_c = _resolve_base_set_c(parser, args)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_dt = _compute_dt(args.train_horizon_T, args.train_num_steps, label="train")
    eval_dt = _compute_dt(args.eval_horizon_T, args.eval_num_steps, label="eval")
    if not np.isclose(train_dt, eval_dt, atol=1e-12, rtol=1e-12):
        raise ValueError(
            "train_dt must equal eval_dt for lane safe-arrival policy training, "
            f"but got train_dt={train_dt:.12g} from T/N={args.train_horizon_T}/{args.train_num_steps} "
            f"and eval_dt={eval_dt:.12g} from T/N={args.eval_horizon_T}/{args.eval_num_steps}."
        )

    ra_cfg = UnicycleSAConfig(
        seed=args.seed,
        total_steps=args.total_steps,
        start_steps=args.start_steps,
        update_after=args.update_after,
        update_every=args.update_every,
        gradient_steps=args.gradient_steps,
        batch_size=args.batch_size,
        beta=args.beta,
        tau=args.tau,
        policy_delay=args.policy_delay,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        max_grad_norm=args.max_grad_norm,
        critic_huber_delta=args.critic_huber_delta,
        action_smoothness_weight=args.action_smoothness_weight,
        hidden_size=args.hidden_size,
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_max=args.actor_log_std_max,
        exploration_std=args.exploration_std,
        exploration_clip=args.exploration_clip,
        target_policy_noise_std=args.target_policy_noise_std,
        target_policy_noise_clip=args.target_policy_noise_clip,
        eval_every=args.eval_every,
        log_every=args.log_every,
        record_update_metrics=args.record_update_metrics,
        update_metric_every=args.update_metric_every,
        num_envs=args.num_envs,
        steps_per_jit=args.steps_per_jit,
        curriculum_start_scale=args.curriculum_start_scale,
        curriculum_increment=args.curriculum_increment,
        curriculum_success_threshold=args.curriculum_success_threshold,
        curriculum_window_episodes=args.curriculum_window_episodes,
        curriculum_min_episodes=args.curriculum_min_episodes,
        val_reset_count=args.val_reset_count,
        test_reset_count=args.test_reset_count,
        collector_terminate_on_goal=args.collector_terminate_on_goal,
    )
    env_cfg = UnicycleSAEnvConfig(
        dt=train_dt,
        horizon_steps=args.train_num_steps,
        a_max=args.a_max,
        r_max=args.r_max,
        v_min=args.v_min,
        v_max=args.v_max,
        v_des=args.v_des,
        y_max=args.y_max,
        psi_max=args.psi_max,
        base_set_c=args.base_set_c,
        lqr_q_y=args.lqr_q_y,
        lqr_q_v=args.lqr_q_v,
        lqr_q_psi=args.lqr_q_psi,
        lqr_r_a=args.lqr_r_a,
        lqr_r_r=args.lqr_r_r,
        init_y_range_min=args.init_y_range_min,
        init_y_range_max=args.init_y_range_max,
        init_v_range_min=args.init_v_range_min,
        init_v_range_max=args.init_v_range_max,
        init_psi_range_min=args.init_psi_range_min,
        init_psi_range_max=args.init_psi_range_max,
        terminate_on_crash=args.terminate_on_crash,
    )
    eval_env_cfg = replace(
        env_cfg,
        dt=eval_dt,
        horizon_steps=args.eval_num_steps,
    )
    compare_grid_cfg = InvariantGridConfig(
        y_min=-args.y_max,
        y_max=args.y_max,
        num_y=args.compare_num_y,
        psi_min=-args.psi_max,
        psi_max=args.psi_max,
        num_psi=args.compare_num_psi,
        v_min=args.compare_v_min,
        v_max=args.compare_v_max,
        num_v=args.compare_num_v,
        max_scatter_points=args.compare_max_scatter_points,
    )

    tag = args.run_tag.strip() if args.run_tag.strip() else datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = _sanitize_output_name(args.output_dir) if args.output_dir else ""
    run_name = f"{tag}-{suffix}" if suffix else tag
    run_dir = PROJECT_ROOT / "outputs" / args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Training lane safe-arrival policy with train(N={args.train_num_steps}, T={args.train_horizon_T:.3f}, dt={train_dt:.4f}) "
        f"and eval(N={args.eval_num_steps}, T={args.eval_horizon_T:.3f}, dt={eval_dt:.4f}), "
        f"base_set_c={args.base_set_c}"
    )
    result = run_unicycle_sa_training(
        ra_cfg,
        env_cfg,
        eval_env_cfg=eval_env_cfg,
        output_dir=str(run_dir),
    )

    history = result["history"]
    summary = result["summary"]
    eval_stats = result["eval"]
    best_eval = result.get("best_eval", {})

    best_weights_path = Path(result["best_weights_path"]) if result.get("best_weights_path") else run_dir / "best_weights.pkl"
    final_weights_path = Path(result["final_weights_path"]) if result.get("final_weights_path") else run_dir / "final_weights.pkl"
    policy_ckpt = run_dir / "backup_policy_actor.pkl"
    compare_weights_path = best_weights_path if args.invariant_compare_checkpoint == "best" else final_weights_path
    shutil.copyfile(compare_weights_path, policy_ckpt)

    np.savez(run_dir / "history.npz", **{k: np.asarray(v, dtype=np.float64) for k, v in history.items()})
    if isinstance(eval_stats.get("trajectory"), dict):
        np.savez(run_dir / "eval_trajectory.npz", **{k: np.asarray(v) for k, v in eval_stats["trajectory"].items()})
    with open(run_dir / "best_eval.json", "w", encoding="utf-8") as f:
        json.dump(best_eval, f, indent=2)
    _plot_backup_training(history, run_dir / "training_metrics.png", eval_every=max(1, ra_cfg.eval_every))

    invariant_metrics = None
    if not args.skip_invariant_compare:
        cbf_cfg = UnicycleBCBFConfig(
            y_max=args.y_max,
            psi_max=args.psi_max,
            a_max=args.a_max,
            r_max=args.r_max,
            v_min=args.v_min,
            v_max=args.v_max,
            v_des=args.v_des,
            dt=eval_dt,
            num_steps=args.eval_num_steps,
            base_set_c=args.base_set_c,
            lqr_q_y=args.lqr_q_y,
            lqr_q_v=args.lqr_q_v,
            lqr_q_psi=args.lqr_q_psi,
            lqr_r_a=args.lqr_r_a,
            lqr_r_r=args.lqr_r_r,
        )
        compare_dir = run_dir / "invariant_compare"
        invariant_metrics = compare_invariant_sets(
            cbf_cfg=cbf_cfg,
            learned_backup_policy_path=str(policy_ckpt),
            output_dir=compare_dir,
            grid_cfg=compare_grid_cfg,
        )
        with open(run_dir / "invariant_compare_summary.json", "w", encoding="utf-8") as f:
            json.dump(invariant_metrics, f, indent=2)

    print("Done.")
    print(f"Saved outputs to: {run_dir}")
    print(f"Best checkpoint step: {summary.get('best_eval_step', args.total_steps)}")
    if invariant_metrics is not None:
        print(
            "Invariant volume ratio (learned/analytic): "
            f"{invariant_metrics['volume_ratio_learned_over_analytic']:.4f}"
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
