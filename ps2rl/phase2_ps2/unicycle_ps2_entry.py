#!/usr/bin/env python
"""Unicycle Phase-2 PS2 SAC training orchestration.

The reusable arg-spec + config-build + train/save/eval body for the unicycle
SAC trainer (HardNet-CVX backup-CBF QP projection). The public entrypoint
``scripts/train_phase2.py --system unicycle`` calls ``main(argv)`` here.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pickle

from ps2rl.utils.paths import PROJECT_ROOT  # = the PS2-RL repo root

from ps2rl.cil.unicycle_backup_cbf import UnicycleBCBFConfig
from ps2rl.envs.unicycle_env import UnicycleEnvConfig
from ps2rl.plotting.plots import plot_training_metrics, plot_trajectory
from ps2rl.phase2_ps2.unicycle_ps2_trainer import SACConfig, run_training


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
    """Resolve the level of the unified LQR base set B from --base_set_c."""
    if args.base_set_c is None:
        parser.error("--base_set_c is required (level of the unified LQR base set B).")
    return float(args.base_set_c)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PS2 policy with CIL projection")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--start_steps", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--update_every", type=int, default=8)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--update_after", type=int, default=2000)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--min_alpha", type=float, default=1e-4, help="Lower floor for SAC entropy temperature alpha.")
    parser.add_argument(
        "--q_clip_abs",
        type=float,
        default=1e3,
        help="Absolute clipping bound for critic targets/values. Set <=0 to disable value clipping.",
    )
    parser.add_argument("--use_projection", action="store_true", default=False)
    parser.add_argument(
        "--project_target_actions",
        action="store_true",
        default=False,
        help="Project next-state actions in critic target (more accurate, slower).",
    )
    parser.add_argument(
        "--no_project_actor_actions",
        action="store_true",
        default=False,
        help="Disable projection in actor/data path (for ablation).",
    )
    parser.add_argument("--record_update_metrics", action="store_true", default=False)
    parser.add_argument("--update_metric_every", type=int, default=200)
    parser.add_argument("--num_envs", type=int, default=32, help="Number of batched environments for the JAX collector.")
    parser.add_argument("--steps_per_jit", type=int, default=128, help="Number of vectorized environment steps per JIT chunk.")

    parser.add_argument("--reward_mode", type=str, default="trajectory_following", choices=["trajectory_following"])
    parser.add_argument("--w_v", type=float, default=1.0)
    parser.add_argument("--w_lane_y", type=float, default=0.3)
    parser.add_argument("--w_lane_psi", type=float, default=0.3)
    parser.add_argument("--w_control", type=float, default=0.01)
    parser.add_argument("--env_v_des", type=float, default=5.0, help="Environment nominal velocity for non-trajectory initial-state sampling.")
    parser.add_argument("--reward_v_des", type=float, default=None, help="Velocity tracking target used in reward modes that track speed.")
    parser.add_argument("--traj_y_amplitude", type=float, default=2.5, help="Sine-wave amplitude for trajectory_following y reference.")
    parser.add_argument("--traj_y_period", type=float, default=10.0, help="Sine-wave period (s) for trajectory_following y reference.")
    parser.add_argument("--traj_y_phase", type=float, default=0.0, help="Sine-wave phase (rad) for trajectory_following y reference.")
    parser.add_argument("--traj_v_mean", type=float, default=5.0, help="Mean velocity reference for trajectory_following.")
    parser.add_argument("--traj_v_amplitude", type=float, default=0.0, help="Velocity-reference sine amplitude for trajectory_following.")
    parser.add_argument("--traj_v_period", type=float, default=10.0, help="Velocity-reference sine period (s) for trajectory_following.")
    parser.add_argument("--traj_v_phase", type=float, default=0.0, help="Velocity-reference sine phase (rad) for trajectory_following.")
    parser.add_argument(
        "--traj_normalize_reward",
        action="store_true",
        default=False,
        help="Normalize trajectory-following reward terms using lane/heading/speed scales.",
    )
    parser.add_argument(
        "--traj_speed_err_scale",
        type=float,
        default=5.0,
        help="Speed-error normalization scale for trajectory reward when --traj_normalize_reward is enabled.",
    )

    parser.add_argument("--env_dt", type=float, default=0.05, help="Environment timestep (s)")
    parser.add_argument("--env_max_steps", type=int, default=None, help="Environment episode horizon in steps. If unset, defaults to 20 seconds via dt.")
    parser.add_argument(
        "--r_max",
        type=float,
        default=0.5,
        help="Yaw-rate bound applied to both environment action bounds and backup-CBF input bounds.",
    )
    parser.add_argument("--num_steps", type=int, default=40, help="N in tau_0 < ... < tau_N")
    parser.add_argument("--horizon_T", type=float, default=2.0, help="Backup-CBF horizon")
    parser.add_argument("--not_terminate_on_violation", action="store_true", default=False, help="Whether to terminate episode on safety violation (for training only).")
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--base_alpha", type=float, default=2.0)
    parser.add_argument("--solver_tol", type=float, default=1e-3)
    parser.add_argument("--target_kappa", type=float, default=1e-2)
    parser.add_argument("--cbf_v_des", type=float, default=5.0, help="Backup-CBF equilibrium velocity target.")
    parser.add_argument(
        "--backup_policy_mode",
        type=str,
        default="analytic",
        choices=["analytic", "learned"],
        help="Backup policy used in backup-flow rollout for CBF constraints.",
    )
    parser.add_argument(
        "--learned_backup_policy_path",
        type=str,
        default="",
        help="Checkpoint path for learned backup policy when --backup_policy_mode=learned.",
    )
    parser.add_argument(
        "--base_set_c",
        type=float,
        default=None,
        help="Level of the unified LQR base set B (terminal set == capture set).",
    )
    parser.add_argument("--lqr_q_y", type=float, default=1.0)
    parser.add_argument("--lqr_q_v", type=float, default=1.0)
    parser.add_argument("--lqr_q_psi", type=float, default=1.0)
    parser.add_argument("--lqr_r_a", type=float, default=1.0)
    parser.add_argument("--lqr_r_r", type=float, default=1.0)
    parser.add_argument("--use_autodiff_jacobian", action="store_true", default=False)

    parser.add_argument("--save_final_weights", action="store_true", default=False, help="Whether to save final policy weights to disk.")

    parser.add_argument("--output_root", type=str, default="uni_phase2_ps2", help="Experiment dir under PROJECT_ROOT/outputs.")
    parser.add_argument("--output_dir", type=str, default="", help="Run-name suffix appended under PROJECT_ROOT/outputs/<output_root>.")
    parser.add_argument("--run_tag", type=str, default="", help="Optional fixed timestamp tag (default: current time).")
    args = parser.parse_args(argv)
    args.base_set_c = _resolve_base_set_c(parser, args)
    return args


def _sanitize_output_name(name: str) -> str:
    s = name.strip()
    s = s.replace("/", "_").replace("\\", "_")
    return s.strip("_-")


def main(argv=None):
    args = parse_args(argv)
    if args.backup_policy_mode == "learned" and not args.learned_backup_policy_path.strip():
        raise ValueError("--learned_backup_policy_path is required when --backup_policy_mode=learned")

    sac_cfg = SACConfig(
        seed=args.seed,
        total_steps=args.total_steps,
        start_steps=args.start_steps,
        batch_size=args.batch_size,
        update_every=args.update_every,
        gradient_steps=args.gradient_steps,
        update_after=args.update_after,
        hidden_size=args.hidden_size,
        max_grad_norm=args.max_grad_norm,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        min_alpha=args.min_alpha,
        q_clip_abs=args.q_clip_abs,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        log_every=args.log_every,
        use_projection=args.use_projection,
        project_target_actions=args.project_target_actions,
        project_actor_actions=not args.no_project_actor_actions,
        record_update_metrics=args.record_update_metrics,
        update_metric_every=args.update_metric_every,
        num_envs=args.num_envs,
        steps_per_jit=args.steps_per_jit,
    )
    env_cfg = UnicycleEnvConfig(
        dt=args.env_dt,
        max_steps=args.env_max_steps,
        r_max=args.r_max,
        v_des=args.env_v_des,
        reward_v_des=args.reward_v_des,
        reward_mode=args.reward_mode,
        w_v=args.w_v,
        w_lane_y=args.w_lane_y,
        w_lane_psi=args.w_lane_psi,
        w_control=args.w_control,
        traj_y_amplitude=args.traj_y_amplitude,
        traj_y_period=args.traj_y_period,
        traj_y_phase=args.traj_y_phase,
        traj_v_mean=args.traj_v_mean,
        traj_v_amplitude=args.traj_v_amplitude,
        traj_v_period=args.traj_v_period,
        traj_v_phase=args.traj_v_phase,
        traj_normalize_reward=args.traj_normalize_reward,
        traj_speed_err_scale=args.traj_speed_err_scale,
        terminate_on_violation=not args.not_terminate_on_violation,
    )
    cbf_cfg = UnicycleBCBFConfig(
        r_max=args.r_max,
        v_des=args.cbf_v_des,
        T=args.horizon_T,
        num_steps=args.num_steps,
        alpha=args.alpha,
        base_alpha=args.base_alpha,
        backup_policy_mode=args.backup_policy_mode,
        learned_backup_policy_path=args.learned_backup_policy_path,
        base_set_c=args.base_set_c,
        lqr_q_y=args.lqr_q_y,
        lqr_q_v=args.lqr_q_v,
        lqr_q_psi=args.lqr_q_psi,
        lqr_r_a=args.lqr_r_a,
        lqr_r_r=args.lqr_r_r,
        solver_tol=args.solver_tol,
        target_kappa=args.target_kappa,
        use_analytic_jacobian=not args.use_autodiff_jacobian,
    )

    tag = args.run_tag.strip() if args.run_tag.strip() else datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = _sanitize_output_name(args.output_dir) if args.output_dir else ""
    run_name = f"{tag}-{suffix}" if suffix else tag
    run_dir = PROJECT_ROOT / "outputs" / args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Training with projection={sac_cfg.use_projection}, "
        f"project_target={sac_cfg.project_target_actions}, "
        f"backup_policy_mode={cbf_cfg.backup_policy_mode}, "
        f"num_envs={sac_cfg.num_envs}, "
        f"steps_per_jit={sac_cfg.steps_per_jit}, "
        f"base_set_c={cbf_cfg.base_set_c}, "
        f"analytic_jac={cbf_cfg.use_analytic_jacobian}, "
        f"q_clip_abs={sac_cfg.q_clip_abs:.3g}, "
        f"min_alpha={sac_cfg.min_alpha:.3g}, "
        f"env_dt={env_cfg.dt:.4f}, "
        f"cbf_dt={cbf_cfg.dt:.4f}, "
        f"N={cbf_cfg.num_steps}, "
        f"T={cbf_cfg.horizon:.3f}, #ineq={cbf_cfg.num_qp_inequalities}"
    )
    print(
        f"Reward mode={env_cfg.reward_mode}, env_v_des={env_cfg.v_des:.3f}, "
        f"reward_v_des={float(env_cfg.reward_v_des):.3f}, cbf_v_des={cbf_cfg.v_des:.3f}"
    )
    if env_cfg.reward_mode == "trajectory_following":
        print(
            f"Trajectory refs: y=A*sin(2pi t/P + phi) with A={env_cfg.traj_y_amplitude:.3f}, "
            f"P={env_cfg.traj_y_period:.3f}, phi={env_cfg.traj_y_phase:.3f}; "
            f"v(t)=v0 + Av*sin(2pi t/Pv + phiv) with v0={env_cfg.traj_v_mean:.3f}, "
            f"Av={env_cfg.traj_v_amplitude:.3f}, Pv={env_cfg.traj_v_period:.3f}, phiv={env_cfg.traj_v_phase:.3f}; "
            f"traj_norm_reward={env_cfg.traj_normalize_reward}, traj_speed_err_scale={env_cfg.traj_speed_err_scale:.3f}"
        )

    result = run_training(sac_cfg, env_cfg, cbf_cfg, output_dir=str(run_dir))

    if args.save_final_weights:
        final_state = result.get("final_state")
        best_state = result.get("best_state")
        if final_state is None:
            print("Warning: final_state missing from trainer result; skipping checkpoint save.")
        else:
            with open(run_dir / "final_weights.pkl", "wb") as f:
                pickle.dump(final_state, f, protocol=pickle.HIGHEST_PROTOCOL)
            if best_state is None:
                print("Warning: best_state missing from trainer result; skipping best_weights.pkl.")
            else:
                with open(run_dir / "best_weights.pkl", "wb") as f:
                    pickle.dump(best_state, f, protocol=pickle.HIGHEST_PROTOCOL)

    history = result["history"]
    summary = result["summary"]
    eval_stats = result["eval"]
    configs = result["configs"]

    np.savez(
        run_dir / "history.npz",
        **{k: np.asarray(v, dtype=np.float64) for k, v in history.items()},
    )
    np.savez(
        run_dir / "eval_trajectory.npz",
        **{k: np.asarray(v) for k, v in eval_stats["trajectory"].items()},
    )
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(run_dir / "configs.json", "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2)

    plot_training_metrics(history, str(run_dir / "training_metrics.png"), eval_every=sac_cfg.eval_every)
    plot_trajectory(
        eval_stats["trajectory"],
        y_max=env_cfg.y_max,
        psi_max=env_cfg.psi_max,
        v_des=float(env_cfg.reward_v_des),
        output_path=str(run_dir / "trajectory.png"),
        reward_mode=env_cfg.reward_mode,
        dt=env_cfg.dt,
    )

    print("Done.")
    print(f"Saved outputs to: {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
