#!/usr/bin/env python
"""Compare the analytic vs. learned quadrotor backup policy near the powerloop.

This is the quadrotor Phase-1 *learned-vs-analytic* backup-policy comparison: 
it measures how many task-relevant near-powerloop states each backup policy 
can safely recover to the base set B, reporting the recoverability rate overall 
and per sub-region (general / near-ceiling / bridge / base-shell). 

Inputs come from a completed quadrotor Phase-1 run directory (``--run_dir``): its
``configs.json`` (backup-CBF config, recoverability weights, compare config), the
saved learned backup actor (``quad_backup_policy_actor.pkl``), and the reset
library (``reset_library.pkl``). CLI flags override individual config fields. The
analytic backup is the aggressive cascaded-PID + LQR hybrid; its PID gains can be
overridden with the ``--pid_*`` flags.

Outputs (a JSON summary plus arrays / optional rollouts) are written under
``<run_dir>/comparison/quadBackup_compare-<timestamp>[-<suffix>]/``.

Example::

    python scripts/compare_backup_policies.py \\
        --run_dir <quad Phase-1 run dir> \\
        --policies analytic_backup_policy learned_backup_policy

This is the quadrotor half of the unified Phase-1 comparison: it is equivalently
reachable as ``scripts/evaluate_phase1.py --system quadrotor --mode compare`` (the
unicycle counterpart is ``--system unicycle --mode compare``). Driven by
``slurm_batch/slurm_eval_quad_phase1.sh``.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ps2rl.backup_policy.quadrotor_learned_backup import load_learned_quadrotor_backup_policy
from ps2rl.evaluation.quadrotor_trace_reset_lib import QuadrotorResetLibrary
from ps2rl.evaluation.quadrotor_trajectory_compare import (
    QuadrotorTrajectoryCompareConfig,
    compare_quadrotor_backup_policies,
)
from ps2rl.phase1_sa.quadrotor_sa_trainer import (
    QuadrotorRecoverabilityWeights,
    quadrotor_recoverability_weights_from_dict,
)
from ps2rl.cil.quadrotor_backup_cbf import QuadrotorBCBFConfig
from ps2rl.utils.paths import resolve_existing_path, resolve_output_root


def _resolve_existing_path(raw: str | None) -> Path | None:
    return resolve_existing_path(raw, bases=(Path.cwd(), PROJECT_ROOT), allow_none=True)


def _resolve_output_root(raw: str | None, default_root: Path) -> Path:
    return resolve_output_root(raw, default_root)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the analytic vs. learned quadrotor backup policy near the powerloop "
        "(Phase-1 recoverability comparison)."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Completed quadrotor Phase-1 run directory (holds configs.json, the learned backup "
        "actor, and the reset library). Required.",
    )
    parser.add_argument(
        "--policy_path",
        type=str,
        default="",
        help="Override path to the learned backup actor .pkl "
        "(default: <run_dir>/quad_backup_policy_actor.pkl).",
    )
    parser.add_argument(
        "--reset_library_path",
        type=str,
        default="",
        help="Override path to the reset library .pkl (default: <run_dir>/reset_library.pkl).",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=("analytic_backup_policy", "learned_backup_policy"),
        default=None,
        help="Subset of backup policies to evaluate. Defaults to both.",
    )
    parser.add_argument(
        "--max_exact_points_per_region",
        type=int,
        default=None,
        help="Cap on exact (unperturbed) trace states evaluated per sub-region.",
    )
    parser.add_argument(
        "--num_perturbed_general", type=int, default=None,
        help="Perturbed initial states sampled from the general-trace sub-region.",
    )
    parser.add_argument(
        "--num_perturbed_near_ceiling", type=int, default=None,
        help="Perturbed initial states sampled from the near-ceiling sub-region.",
    )
    parser.add_argument(
        "--num_perturbed_bridge", type=int, default=None,
        help="Perturbed initial states sampled from the bridge (below-apex) sub-region.",
    )
    parser.add_argument(
        "--num_perturbed_base_shell", "--num_perturbed_capture_shell", type=int, default=None,
        help="Perturbed initial states sampled from the base-shell sub-region (near B). "
        "(--num_perturbed_capture_shell is a legacy alias.)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Rollout batch size for the recoverability sweep.",
    )
    parser.add_argument(
        "--save-rollout-trajectories",
        action="store_true",
        default=False,
        help="Save fixed-horizon per-start rollout traces under trajectory_compare_rollouts.npz.",
    )
    parser.add_argument(
        "--enable_qp_screening", action="store_true", default=False,
        help="Additionally screen each start with a BCBF-QP feasibility check.",
    )
    parser.add_argument("--qp_batch_size", type=int, default=None, help="Batch size for QP screening.")
    parser.add_argument("--qp_max_points", type=int, default=None, help="Max points to QP-screen.")
    parser.add_argument(
        "--qp_postsolve_feas_tol", type=float, default=None,
        help="Post-solve feasibility tolerance for QP screening.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Benchmark RNG seed for perturbation sampling.")
    parser.add_argument(
        "--output_root", type=str, default="",
        help="Root directory for the comparison output (default: <run_dir>/comparison).",
    )
    parser.add_argument(
        "--output_suffix", type=str, default="",
        help="Optional suffix appended to the output run name.",
    )
    # Analytic-backup cascaded-PID gain overrides (default: values from the run's configs.json).
    parser.add_argument("--pid_kp_z", type=float, default=None, help="Analytic-PID: altitude P gain.")
    parser.add_argument("--pid_kv_z", type=float, default=None, help="Analytic-PID: vertical velocity D gain.")
    parser.add_argument("--pid_kv_xy", type=float, default=None, help="Analytic-PID: lateral velocity D gain.")
    parser.add_argument("--pid_attitude_p_gain", type=float, default=None, help="Analytic-PID: attitude P gain.")
    parser.add_argument("--pid_yaw_gain_scale", type=float, default=None, help="Analytic-PID: yaw-rate gain scale.")
    parser.add_argument("--pid_ceiling_margin", type=float, default=None, help="Analytic-PID: ceiling safety margin.")
    parser.add_argument("--pid_z_safety_gain", type=float, default=None, help="Analytic-PID: near-ceiling altitude safety gain.")
    parser.add_argument("--pid_ceiling_vz_gain", type=float, default=None, help="Analytic-PID: near-ceiling vertical-damping gain.")
    parser.add_argument("--pid_lateral_boost", type=float, default=None, help="Analytic-PID: lateral acceleration boost.")
    parser.add_argument("--pid_min_virtual_accel_z", type=float, default=None, help="Analytic-PID: floor on the virtual vertical acceleration.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run_dir = _resolve_existing_path(args.run_dir)
    if run_dir is None:
        raise ValueError("--run_dir is required so the compare script can load configs and default artifacts.")

    policy_path = _resolve_existing_path(args.policy_path) or (run_dir / "quad_backup_policy_actor.pkl")
    reset_library_path = _resolve_existing_path(args.reset_library_path) or (run_dir / "reset_library.pkl")
    config_path = run_dir / "configs.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing configs.json under run directory: {run_dir}")
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    compare_cfg_payload = dict(configs.get("compare", {}))
    overrides = {
        "max_exact_points_per_region": args.max_exact_points_per_region,
        "num_perturbed_general": args.num_perturbed_general,
        "num_perturbed_near_ceiling": args.num_perturbed_near_ceiling,
        "num_perturbed_bridge": args.num_perturbed_bridge,
        "num_perturbed_base_shell": args.num_perturbed_base_shell,
        "batch_size": args.batch_size,
        "qp_batch_size": args.qp_batch_size,
        "qp_max_points": args.qp_max_points,
        "qp_postsolve_feas_tol": args.qp_postsolve_feas_tol,
    }
    for key, value in overrides.items():
        if value is not None:
            compare_cfg_payload[key] = value
    if args.enable_qp_screening:
        compare_cfg_payload["enable_qp_screening"] = True
    compare_cfg_payload["benchmark_seed"] = int(args.seed)
    compare_cfg = QuadrotorTrajectoryCompareConfig(**{
        **QuadrotorTrajectoryCompareConfig().__dict__,
        **compare_cfg_payload,
    })

    cbf_cfg_payload = dict(configs["backup_env"]["cbf_cfg"])
    capture_cfg_payload = dict(configs["backup_env"].get("capture_set_cfg", {}))
    capture_to_cbf = {
        "capture_set_mode": "capture_set_mode",
        "capture_c": "capture_c",
        "smooth_gain": "base_set_smooth_gain",
    }
    for src_key, dst_key in capture_to_cbf.items():
        if src_key in capture_cfg_payload:
            cbf_cfg_payload[dst_key] = capture_cfg_payload[src_key]
    pid_overrides = {
        "pid_kp_z": args.pid_kp_z,
        "pid_kv_z": args.pid_kv_z,
        "pid_kv_xy": args.pid_kv_xy,
        "pid_attitude_p_gain": args.pid_attitude_p_gain,
        "pid_yaw_gain_scale": args.pid_yaw_gain_scale,
        "pid_ceiling_margin": args.pid_ceiling_margin,
        "pid_z_safety_gain": args.pid_z_safety_gain,
        "pid_ceiling_vz_gain": args.pid_ceiling_vz_gain,
        "pid_lateral_boost": args.pid_lateral_boost,
        "pid_min_virtual_accel_z": args.pid_min_virtual_accel_z,
    }
    for key, value in pid_overrides.items():
        if value is not None:
            cbf_cfg_payload[key] = float(value)
    cbf_valid = {f.name for f in fields(QuadrotorBCBFConfig) if f.init}
    cbf_cfg = QuadrotorBCBFConfig(**{k: v for k, v in cbf_cfg_payload.items() if k in cbf_valid})
    weights = quadrotor_recoverability_weights_from_dict(configs.get("recoverability_weights", {}))
    reset_library = QuadrotorResetLibrary.load(reset_library_path)
    policy = load_learned_quadrotor_backup_policy(policy_path)

    out_root = _resolve_output_root(args.output_root, run_dir / "comparison")
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = args.output_suffix.strip()
    name = f"quadBackup_compare-{tag}-{suffix}" if suffix else f"quadBackup_compare-{tag}"
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = compare_quadrotor_backup_policies(
        cbf_cfg=cbf_cfg,
        reset_library=reset_library,
        learned_policy=policy,
        compare_cfg=compare_cfg,
        recoverability_weights=weights,
        output_dir=out_dir,
        selected_policies=args.policies,
        save_rollout_trajectories=bool(args.save_rollout_trajectories),
    )

    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts saved under: {out_dir}")


if __name__ == "__main__":
    main()
