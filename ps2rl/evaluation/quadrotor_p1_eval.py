#!/usr/bin/env python
"""Quadrotor Phase-1 evaluation: score a saved learned backup policy on held-out reset sets.

Entry orchestration for the quadrotor Phase-1 eval. The public entrypoint
``scripts/evaluate_phase1.py --system quadrotor`` calls ``main(argv)`` here.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from ps2rl.utils.paths import PROJECT_ROOT  # = the PS2-RL repo root

from ps2rl.backup_policy.quadrotor_learned_backup import load_learned_quadrotor_backup_policy
from ps2rl.phase1_sa.quadrotor_sa_env import quadrotor_backup_env_config_from_dict
from ps2rl.evaluation.quadrotor_trace_reset_lib import QuadrotorResetLibrary
from ps2rl.phase1_sa.quadrotor_sa_trainer import (
    _evaluate_policy_jax as _evaluate_policy_jax_ra,
    quadrotor_recoverability_weights_from_dict as quadrotor_recoverability_weights_from_dict_ra,
)
from ps2rl.utils.paths import resolve_existing_path, resolve_output_root


def _resolve_existing_path(raw: str | None) -> Path | None:
    return resolve_existing_path(raw, bases=(Path.cwd(), PROJECT_ROOT), allow_none=True)


def _resolve_output_root(raw: str | None, default_root: Path) -> Path:
    return resolve_output_root(raw, default_root)


def _load_run_artifacts(run_dir: Path | None, policy_path: Path | None, reset_library_path: Path | None):
    if policy_path is None:
        if run_dir is None:
            raise ValueError("Provide --policy_path or --run_dir.")
        policy_path = run_dir / "quad_backup_policy_actor.pkl"
    if reset_library_path is None:
        if run_dir is None:
            raise ValueError("Provide --reset_library_path or --run_dir.")
        reset_library_path = run_dir / "reset_library.pkl"
    if run_dir is None:
        run_dir = policy_path.parent
    config_path = run_dir / "configs.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing configs.json under run directory: {run_dir}")
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    return run_dir, policy_path.resolve(), reset_library_path.resolve(), configs


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved learned quadrotor backup policy.")
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--policy_path", type=str, default="")
    parser.add_argument("--reset_library_path", type=str, default="")
    parser.add_argument("--split", type=str, default="both", choices=["val", "test", "both"])
    parser.add_argument("--max_resets", type=int, default=0, help="Optional cap per split. 0 means use all held-out resets.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--output_suffix", type=str, default="")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run_dir = _resolve_existing_path(args.run_dir)
    policy_path = _resolve_existing_path(args.policy_path)
    reset_library_path = _resolve_existing_path(args.reset_library_path)
    run_dir, policy_path, reset_library_path, configs = _load_run_artifacts(run_dir, policy_path, reset_library_path)

    env_cfg = quadrotor_backup_env_config_from_dict(configs["backup_env"])
    weights = quadrotor_recoverability_weights_from_dict_ra(configs.get("recoverability_weights", {}))
    beta = float(configs.get("backup_ra", {}).get("beta", 0.0))
    if not (0.0 < beta < 1.0):
        raise ValueError(f"Expected a valid backup_ra.beta in configs.json, got {beta}")
    policy = load_learned_quadrotor_backup_policy(policy_path)
    reset_library = QuadrotorResetLibrary.load(reset_library_path)

    splits = ["val", "test"] if args.split == "both" else [args.split]
    out_root = _resolve_output_root(args.output_root, run_dir / "evaluation")
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = args.output_suffix.strip()
    name = f"quadBackup_eval-{tag}-{suffix}" if suffix else f"quadBackup_eval-{tag}"
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "policy_path": str(policy_path),
        "reset_library_path": str(reset_library_path),
        "splits": {},
    }
    split_rows: list[dict[str, Any]] = []

    for split_idx, split in enumerate(splits):
        if args.max_resets > 0:
            heldout = reset_library.heldout_reset_sets[split]
            trimmed = {
                k: np.asarray(v[: int(args.max_resets)]) if np.asarray(v).ndim > 0 else np.asarray(v)
                for k, v in heldout.items()
            }
            reset_library.heldout_reset_sets[split] = trimmed

        split_eval = _evaluate_policy_jax_ra(
            env_cfg=env_cfg,
            reset_library=reset_library,
            actor_params=policy.actor_params,
            actor_cfg=policy.actor_cfg,
            action_scale=policy.action_scale,
            action_low=policy.action_low,
            action_high=policy.action_high,
            split=split,
            recoverability_weights=weights,
            beta=beta,
        )
        summary["splits"][split] = {k: v for k, v in split_eval.items() if k != "trajectory"}
        np.savez(out_dir / f"{split}_trajectory.npz", **{k: np.asarray(v) for k, v in split_eval["trajectory"].items()})

        for region, metrics in summary["splits"][split]["subset_metrics"].items():
            split_rows.append(
                {
                    "split": split,
                    "region": region,
                    "count": int(metrics["count"]),
                    "success_rate": float(metrics["success_rate"]),
                    "crash_rate": float(metrics["crash_rate"]),
                    "entered_terminal_rate": float(metrics["entered_terminal_rate"]),
                    "safe_rollout_rate": float(metrics["safe_rollout_rate"]),
                    "terminal_at_horizon_rate": float(metrics["terminal_at_horizon_rate"]),
                    "post_entry_terminal_step_rate": float(metrics.get("post_entry_terminal_step_rate", 0.0)),
                    "mean_discounted_ra_score": float(metrics.get("mean_discounted_ra_score", 0.0)),
                    "entry_time_mean_sec": metrics["entry_time_sec"]["mean"],
                    "min_hard_deck_margin_mean": metrics["minimum_hard_deck_margin"]["mean"],
                }
            )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "subset_metrics.csv", "w", encoding="utf-8") as f:
        f.write("split,region,count,success_rate,crash_rate,entered_terminal_rate,safe_rollout_rate,terminal_at_horizon_rate,post_entry_terminal_step_rate,mean_discounted_ra_score,entry_time_mean_sec,min_hard_deck_margin_mean\n")
        for row in split_rows:
            f.write(",".join(str(row[k]) for k in row.keys()) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts saved under: {out_dir}")


if __name__ == "__main__":
    main()
