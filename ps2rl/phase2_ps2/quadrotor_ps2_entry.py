#!/usr/bin/env python
"""Quadrotor Phase-2 / vanilla-tracker SAC training orchestration.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pickle

from ps2rl.utils.paths import PROJECT_ROOT  # = the PS2-RL repo root

from ps2rl.cil.quadrotor_backup_cbf import QuadrotorBCBFConfig, QuadrotorBackupCBFProjector
from ps2rl.envs.quadrotor_env import QuadrotorEnvConfig, build_quadrotor_env
from ps2rl.cil.cil_policy import ActorConfig
from ps2rl.plotting.plots import plot_quad_trajectory
from ps2rl.phase2_ps2.quadrotor_ps2_trainer import SACConfig, _build_action_fns, _evaluate_policy, run_training


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Safe PS2 policy for quadrotor with CIL projection")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=120000)
    parser.add_argument("--start_steps", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--update_every", type=int, default=8)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--update_after", type=int, default=2000)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument(
        "--best_weights_save_period",
        type=int,
        default=100000,
        help="Period in training steps for refreshing best_weights*.pkl during training; <=0 disables periodic saves.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=1e-4)
    parser.add_argument("--min_alpha", type=float, default=1e-4, help="Lower floor for SAC entropy temperature alpha.")
    parser.add_argument("--target_entropy", type=float, default=-4.0, help="SAC target entropy (default -action_dim).")
    parser.add_argument(
        "--q_clip_abs",
        type=float,
        default=5e6,
        help="Absolute clipping bound for critic targets/values. Set <=0 to disable clipping.",
    )
    parser.add_argument(
        "--disable_projection",
        action="store_true",
        default=False,
        help="Disable backup-CBF projection in actor/data path and random-start projection.",
    )
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
    parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel environments for JAX training.")
    parser.add_argument(
        "--steps_per_jit",
        type=int,
        default=128,
        help="Vectorized env steps per compiled JAX scan chunk.",
    )
    parser.add_argument(
        "--warm_start",
        action="store_true",
        default=False,
        help="Initialize SAC actor/critics from an existing checkpoint before training.",
    )
    parser.add_argument(
        "--warm_start_weights",
        type=str,
        default="",
        help="Path to a saved SAC checkpoint (for example best_weights.pkl) used for warm start.",
    )

    parser.add_argument("--env_dt", type=float, default=0.02, help="Environment timestep (s)")
    parser.add_argument("--env_max_steps", type=int, default=None, help="Environment episode horizon in steps.")
    parser.add_argument(
        "--env_max_steps_extra_sec",
        type=float,
        default=0.1,
        help="Extra wall-clock time added beyond reference+horizon when env_max_steps is unset.",
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        default="",
        help=(
            "Reference trajectory file for nominal tracking. "
            "Leave empty to use the env default, which prefers the richer .npz bundle when available."
        ),
    )
    parser.add_argument(
        "--reference_dt",
        type=float,
        default=None,
        help="Reference sample timestep in seconds (default: env_dt).",
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="trajectory_following",
        choices=["trajectory_following"],
    )
    parser.add_argument("--w_pos_xy", type=float, default=1.0)
    parser.add_argument("--w_pos_z", type=float, default=2.0)
    parser.add_argument("--w_vel", type=float, default=0.2)
    parser.add_argument("--w_att", type=float, default=1.0)
    parser.add_argument("--w_ref_omega_x", type=float, default=0.0)
    parser.add_argument("--w_ref_omega_y", type=float, default=0.0)
    parser.add_argument("--w_ref_omega_z", type=float, default=0.0)
    parser.add_argument("--w_control_a", type=float, default=0.01)
    parser.add_argument("--w_control_omega", type=float, default=0.01)

    parser.add_argument("--init_px_range", type=float, default=0.1)
    parser.add_argument("--init_py_range", type=float, default=0.1)
    parser.add_argument("--init_pz_range", type=float, default=0.1)
    parser.add_argument("--init_v_range", type=float, default=0.0)
    parser.add_argument("--init_tilt_deg_range", type=float, default=0.0)
    parser.add_argument("--init_yaw_deg_range", type=float, default=0.0)

    parser.add_argument("--a_cmd_min", type=float, default=0.0)
    parser.add_argument(
        "--a_cmd_max_g",
        type=float,
        default=4.0,
        help="Thrust/acceleration upper bound in g units.",
    )
    parser.add_argument("--omega_max", type=float, default=18.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument(
        "--z_max",
        "--z_min",
        dest="z_max",
        type=float,
        default=3.0,
        help="Maximum safe altitude [m] (legacy alias: --z_min).",
    )
    parser.add_argument(
        "--not_terminate_on_violation",
        action="store_true",
        default=False,
        help="Whether to continue episode after safety violation.",
    )

    parser.add_argument("--num_steps", type=int, default=100, help="N in tau_0 < ... < tau_N")
    parser.add_argument("--horizon_T", type=float, default=2.0, help="Backup-CBF horizon")
    parser.add_argument(
        "--backup_policy_mode",
        type=str,
        default="analytic",
        choices=["analytic", "learned"],
        help="Backup policy mode for quadrotor backup-CBF.",
    )
    parser.add_argument(
        "--learned_backup_policy_path",
        type=str,
        default="",
        help=(
            "Path to the learned phase-1 quadrotor backup policy artifact when "
            "--backup_policy_mode=learned. This may point to a run directory, "
            "quad_backup_policy_actor.pkl, or best_weights.pkl with sibling configs.json."
        ),
    )
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--base_alpha", type=float, default=2.0)
    parser.add_argument("--z_des", type=float, default=2.0)
    parser.add_argument("--pid-kp-z", type=float, default=36.0, help="PID proportional gain on altitude error.")
    parser.add_argument("--pid-kv-z", type=float, default=24.0, help="PID damping gain on vertical velocity.")
    parser.add_argument("--pid-kv-xy", type=float, default=14.0, help="PID damping gain on lateral velocities.")
    parser.add_argument(
        "--pid-attitude-p-gain",
        type=float,
        default=45.0,
        help="PID proportional gain on quaternion attitude error during the aggressive phase.",
    )
    parser.add_argument(
        "--pid-yaw-gain-scale",
        type=float,
        default=0.35,
        help="Relative yaw-rate gain scale during the aggressive PID phase.",
    )
    parser.add_argument(
        "--pid-ceiling-margin",
        type=float,
        default=0.60,
        help="Altitude margin below z_max where anti-ceiling behavior ramps in.",
    )
    parser.add_argument(
        "--pid-z-safety-gain",
        type=float,
        default=32.0,
        help="Extra downward virtual-acceleration gain activated near the ceiling.",
    )
    parser.add_argument(
        "--pid-ceiling-vz-gain",
        type=float,
        default=18.0,
        help="Additional damping on upward velocity near the ceiling.",
    )
    parser.add_argument(
        "--pid-lateral-boost",
        type=float,
        default=1.50,
        help="Multiplier that increases lateral braking/tilt urgency near the ceiling.",
    )
    parser.add_argument(
        "--pid-min-virtual-accel-z",
        type=float,
        default=0.0,
        help="Lower bound on the PID virtual-acceleration z component.",
    )
    parser.add_argument(
        "--base_set_c",
        type=float,
        default=None,
        help=(
            "Base-set level c in h_B(x)=c-x_err^T P x_err (paper's base set B; "
            "plays both legacy terminal-set and capture-set roles). "
            "Defaults to the --terminal-c/--capture_c aliases when unset."
        ),
    )
    parser.add_argument("--lqr-q-z", type=float, default=1.0, help="LQR Q weight on z error.")
    parser.add_argument("--lqr-q-vx", type=float, default=0.16, help="LQR Q weight on v_x.")
    parser.add_argument("--lqr-q-vy", type=float, default=0.16, help="LQR Q weight on v_y.")
    parser.add_argument("--lqr-q-vz", type=float, default=0.4, help="LQR Q weight on v_z.")
    parser.add_argument("--lqr-q-thetax", type=float, default=0.8, help="LQR Q weight on theta_x error.")
    parser.add_argument("--lqr-q-thetay", type=float, default=0.8, help="LQR Q weight on theta_y error.")
    parser.add_argument("--lqr-q-thetaz", type=float, default=0.16, help="LQR Q weight on theta_z error.")
    parser.add_argument("--lqr-r-a-cmd", type=float, default=0.02, help="LQR R weight on a_cmd.")
    parser.add_argument("--lqr-r-omega-x", type=float, default=0.012, help="LQR R weight on omega_x.")
    parser.add_argument("--lqr-r-omega-y", type=float, default=0.012, help="LQR R weight on omega_y.")
    parser.add_argument("--lqr-r-omega-z", type=float, default=0.004, help="LQR R weight on omega_z.")
    parser.add_argument("--solver_tol", type=float, default=1e-4)
    parser.add_argument("--target_kappa", type=float, default=1e-2)
    parser.add_argument("--sensitivity_clip", type=float, default=1e6)

    parser.add_argument(
        "--save_final_weights",
        action="store_true",
        default=True,
        help="Whether to save final policy weights to disk.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="quad_phase2_ps2",
        help="Experiment dir under PROJECT_ROOT/outputs (relative names nest under outputs/; absolute paths used as-is).",
    )
    parser.add_argument("--output_dir", type=str, default="", help="Run-name suffix appended under PROJECT_ROOT/outputs/<output_root>.")
    parser.add_argument("--run_tag", type=str, default="", help="Optional fixed timestamp tag (default: current time).")
    args = parser.parse_args(argv)
    args.base_set_c = _resolve_base_set_c(parser, args)
    return args


def _resolve_base_set_c(parser, args) -> float:
    """Resolve the base-set level c (paper's base set B) from --base_set_c."""
    if args.base_set_c is None:
        parser.error("--base_set_c is required (level of the base set B).")
    return float(args.base_set_c)


def _sanitize_output_name(name: str) -> str:
    s = name.strip()
    s = s.replace("/", "_").replace("\\", "_")
    return s.strip("_-")


def _resolve_output_root(raw: str) -> Path:
    token = raw.strip()
    if not token:
        return (PROJECT_ROOT / "outputs").resolve()
    p = Path(token).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / "outputs" / p
    return p.resolve()


def _build_metric_logger(run_dir: Path):
    """Return a metric-logging callback that echoes to stdout and appends JSONL."""
    metrics_path = run_dir / "metrics.jsonl"

    def _metric_logger(step: int, metrics: dict[str, float]) -> None:
        if not metrics:
            return
        record: dict[str, object] = {"step": int(step)}
        for key, value in metrics.items():
            try:
                record[key] = float(value)
            except (TypeError, ValueError):
                record[key] = value
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        compact = " ".join(
            f"{key}={record[key]:.4g}"
            for key in metrics
            if isinstance(record.get(key), float)
        )
        print(f"[step {int(step)}] {compact}", flush=True)

    return _metric_logger


def _evaluate_saved_policy(
    sac_cfg: SACConfig,
    env_cfg: QuadrotorEnvConfig,
    cbf_cfg: QuadrotorBCBFConfig,
    actor_state: dict[str, object],
    seed: int,
    episodes: int,
):
    env_fns = build_quadrotor_env(env_cfg)
    actor_cfg = ActorConfig(
        obs_dim=env_fns.obs_dim,
        action_dim=env_fns.action_dim,
        hidden_sizes=(sac_cfg.hidden_size, sac_cfg.hidden_size),
    )
    action_scale = jnp.asarray(
        [cbf_cfg.a_cmd_max, cbf_cfg.omega_max, cbf_cfg.omega_max, cbf_cfg.omega_max],
        dtype=jnp.float32,
    )
    projector = QuadrotorBackupCBFProjector(cbf_cfg)
    _, eval_action_fn = _build_action_fns(
        sac_cfg,
        actor_cfg,
        cbf_cfg,
        action_scale,
        backup_runtime=projector.runtime,
    )
    return _evaluate_policy(
        env_cfg,
        eval_action_fn,
        actor_state["actor_params"],
        seed=seed,
        episodes=episodes,
    )


def main(argv=None):
    args = parse_args(argv)
    a_cmd_max = float(args.a_cmd_max_g) * float(args.gravity)
    warm_start_weights = args.warm_start_weights.strip()
    warm_start = bool(args.warm_start or warm_start_weights)
    learned_backup_policy_path = args.learned_backup_policy_path.strip()
    if args.backup_policy_mode.strip().lower() == "learned" and not learned_backup_policy_path:
        raise SystemExit("--learned_backup_policy_path is required when --backup_policy_mode=learned.")
    if warm_start and not warm_start_weights:
        raise SystemExit("--warm_start requires --warm_start_weights to point to a saved SAC checkpoint.")

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
        target_entropy=args.target_entropy,
        q_clip_abs=args.q_clip_abs,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        log_every=args.log_every,
        best_weights_save_period=args.best_weights_save_period,
        use_projection=not args.disable_projection,
        project_target_actions=args.project_target_actions,
        project_actor_actions=not args.no_project_actor_actions,
        record_update_metrics=args.record_update_metrics,
        update_metric_every=args.update_metric_every,
        num_envs=args.num_envs,
        steps_per_jit=args.steps_per_jit,
        warm_start=warm_start,
        warm_start_weights=warm_start_weights,
    )
    env_cfg = QuadrotorEnvConfig(
        dt=args.env_dt,
        max_steps=args.env_max_steps,
        max_steps_extra_sec=args.env_max_steps_extra_sec,
        backup_horizon_T=args.horizon_T,
        gravity=args.gravity,
        a_cmd_min=args.a_cmd_min,
        a_cmd_max=a_cmd_max,
        omega_max=args.omega_max,
        z_max=args.z_max,
        terminate_on_violation=not args.not_terminate_on_violation,
        reward_mode=args.reward_mode,
        reference_path=args.reference_path,
        reference_dt=args.reference_dt,
        z_des=args.z_des,
        w_pos_xy=args.w_pos_xy,
        w_pos_z=args.w_pos_z,
        w_vel=args.w_vel,
        w_att=args.w_att,
        w_ref_omega_x=args.w_ref_omega_x,
        w_ref_omega_y=args.w_ref_omega_y,
        w_ref_omega_z=args.w_ref_omega_z,
        w_control_a=args.w_control_a,
        w_control_omega=args.w_control_omega,
        init_px_range=args.init_px_range,
        init_py_range=args.init_py_range,
        init_pz_range=args.init_pz_range,
        init_v_range=args.init_v_range,
        init_tilt_deg_range=args.init_tilt_deg_range,
        init_yaw_deg_range=args.init_yaw_deg_range,
    )
    cbf_cfg = QuadrotorBCBFConfig(
        T=args.horizon_T,
        num_steps=args.num_steps,
        backup_policy_mode=args.backup_policy_mode,
        learned_backup_policy_path=learned_backup_policy_path,
        alpha=args.alpha,
        gravity=args.gravity,
        a_cmd_min=args.a_cmd_min,
        a_cmd_max=a_cmd_max,
        omega_max=args.omega_max,
        z_max=args.z_max,
        z_des=args.z_des,
        pid_kp_z=args.pid_kp_z,
        pid_kv_z=args.pid_kv_z,
        pid_kv_xy=args.pid_kv_xy,
        pid_attitude_p_gain=args.pid_attitude_p_gain,
        pid_yaw_gain_scale=args.pid_yaw_gain_scale,
        pid_ceiling_margin=args.pid_ceiling_margin,
        pid_z_safety_gain=args.pid_z_safety_gain,
        pid_ceiling_vz_gain=args.pid_ceiling_vz_gain,
        pid_lateral_boost=args.pid_lateral_boost,
        pid_min_virtual_accel_z=args.pid_min_virtual_accel_z,
        base_alpha=args.base_alpha,
        base_set_c=args.base_set_c,
        lqr_q_z=args.lqr_q_z,
        lqr_q_vx=args.lqr_q_vx,
        lqr_q_vy=args.lqr_q_vy,
        lqr_q_vz=args.lqr_q_vz,
        lqr_q_thetax=args.lqr_q_thetax,
        lqr_q_thetay=args.lqr_q_thetay,
        lqr_q_thetaz=args.lqr_q_thetaz,
        lqr_r_a_cmd=args.lqr_r_a_cmd,
        lqr_r_omega_x=args.lqr_r_omega_x,
        lqr_r_omega_y=args.lqr_r_omega_y,
        lqr_r_omega_z=args.lqr_r_omega_z,
        solver_tol=args.solver_tol,
        target_kappa=args.target_kappa,
        sensitivity_clip=args.sensitivity_clip,
    )

    tag = args.run_tag.strip() if args.run_tag.strip() else datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = _sanitize_output_name(args.output_dir) if args.output_dir else ""
    run_name = f"{tag}-{suffix}" if suffix else tag
    output_root = _resolve_output_root(args.output_root)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metric_logger = _build_metric_logger(run_dir)

    print(
        f"Training quadrotor with projection={sac_cfg.use_projection}, "
        f"project_target={sac_cfg.project_target_actions}, "
        f"reward_mode={env_cfg.reward_mode}, "
        f"ref={env_cfg.reference_path}, "
        f"a_cmd in [{env_cfg.a_cmd_min:.3f}, {env_cfg.a_cmd_max:.3f}], "
        f"omega_max={env_cfg.omega_max:.3f}, "
        f"z_max={env_cfg.z_max:.3f}, "
        f"env_dt={env_cfg.dt:.4f}, "
        f"max_steps={env_cfg.max_steps}, "
        f"cbf_dt={cbf_cfg.dt:.4f}, "
        f"N={cbf_cfg.num_steps}, "
        f"T={cbf_cfg.horizon:.3f}, #ineq={cbf_cfg.num_qp_inequalities}, "
        f"base_set_c={cbf_cfg.base_set_c:.3f}, "
        f"backup_mode={cbf_cfg.backup_policy_mode}, "
        f"hidden={sac_cfg.hidden_size}x{sac_cfg.hidden_size}, "
        f"num_envs={sac_cfg.num_envs}, "
        f"steps_per_jit={sac_cfg.steps_per_jit}"
    )
    if cbf_cfg.backup_policy_mode == "learned":
        print(f"Learned backup policy artifact: {cbf_cfg.learned_backup_policy_path}")
    if sac_cfg.warm_start:
        print(f"Warm-start checkpoint: {sac_cfg.warm_start_weights}")

    result = run_training(sac_cfg, env_cfg, cbf_cfg, output_dir=str(run_dir), metric_logger=metric_logger)

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
    best_state = result.get("best_state")

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

    if best_state is not None:
        best_eval_seed = int(sac_cfg.seed + int(summary.get("best_eval_step", sac_cfg.total_steps)))
        best_eval_stats = _evaluate_saved_policy(
            sac_cfg,
            env_cfg,
            cbf_cfg,
            best_state,
            seed=best_eval_seed,
            episodes=sac_cfg.eval_episodes,
        )
        np.savez(
            run_dir / "best_eval_trajectory.npz",
            **{k: np.asarray(v) for k, v in best_eval_stats["trajectory"].items()},
        )
        plot_quad_trajectory(
            best_eval_stats["trajectory"],
            z_max=env_cfg.z_max,
            output_path=str(run_dir / "best_trajectory.png"),
            dt=env_cfg.dt,
        )

    print("Done.")
    print(f"Saved outputs to: {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
