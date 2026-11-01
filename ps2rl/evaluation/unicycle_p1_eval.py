#!/usr/bin/env python
"""Unicycle Phase-1 evaluation: rerun the lane invariant comparison on a saved checkpoint.

Entry orchestration for the unicycle Phase-1 eval. The public entrypoint
``scripts/evaluate_phase1.py --system unicycle`` calls ``main(argv)`` here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from functools import lru_cache
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parents[2] / "matplotlib-cache")))

from ps2rl.utils.paths import PROJECT_ROOT  # = the PS2-RL repo root

@lru_cache(maxsize=1)
def _load_compare_symbols():
    from ps2rl.evaluation.invariant_compare import InvariantGridConfig, compare_invariant_sets
    from ps2rl.cil.unicycle_backup_cbf import UnicycleBCBFConfig

    return InvariantGridConfig, compare_invariant_sets, UnicycleBCBFConfig


def _cbf_field_names() -> set[str]:
    _, _, lane_cfg_cls = _load_compare_symbols()
    return {field.name for field in fields(lane_cfg_cls)}


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _build_cbf_cfg(saved_cfgs: dict, args: argparse.Namespace):
    _, _, lane_cfg_cls = _load_compare_symbols()
    env_cfg = saved_cfgs.get("backup_eval_env") or saved_cfgs.get("backup_env")
    if not isinstance(env_cfg, dict):
        raise KeyError("configs.json is missing both 'backup_eval_env' and 'backup_env'")
    if "horizon_steps" not in env_cfg:
        raise KeyError("Saved environment config is missing 'horizon_steps'")
    if "dt" not in env_cfg:
        raise KeyError("Saved environment config is missing 'dt'")

    cbf_kwargs = {key: value for key, value in env_cfg.items() if key in _cbf_field_names()}
    cbf_kwargs["dt"] = env_cfg["dt"]
    cbf_kwargs["num_steps"] = env_cfg["horizon_steps"]
    cbf_cfg = lane_cfg_cls(**cbf_kwargs)
    return cbf_cfg


def _build_grid_cfg(saved_cfgs: dict, args: argparse.Namespace):
    inv_grid_cls, _, _ = _load_compare_symbols()
    env_cfg = saved_cfgs.get("backup_eval_env") or saved_cfgs.get("backup_env")
    if not isinstance(env_cfg, dict):
        raise KeyError("configs.json is missing both 'backup_eval_env' and 'backup_env'")
    if "y_max" not in env_cfg or "psi_max" not in env_cfg:
        raise KeyError("Saved environment config must include 'y_max' and 'psi_max'")

    return inv_grid_cls(
        y_min=-float(env_cfg["y_max"]),
        y_max=float(env_cfg["y_max"]),
        num_y=int(args.compare_num_y),
        psi_min=-float(env_cfg["psi_max"]),
        psi_max=float(env_cfg["psi_max"]),
        num_psi=int(args.compare_num_psi),
        v_min=float(args.compare_v_min),
        v_max=float(args.compare_v_max),
        num_v=int(args.compare_num_v),
        max_scatter_points=int(args.compare_max_scatter_points),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun lane invariant comparison for one saved run using final_weights.pkl "
            "and write results to a per-run output subdirectory."
        )
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to one saved experiment directory containing the requested checkpoint and configs.json.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="final_weights.pkl",
        help="Checkpoint filename to compare, for example final_weights.pkl or best_weights.pkl.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="invariant_compare_final",
        help="Per-run subdirectory where the rerun outputs are written.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help=(
            "Save reusable boolean masks and plotted point clouds to "
            "invariant_compare_results.npz so the compare plots can be rebuilt later "
            "without recomputing the invariant check."
        ),
    )
    parser.add_argument("--skip_existing", action="store_true", help="Skip the rerun if the target metrics file already exists.")
    parser.add_argument("--dry_run", action="store_true", help="Print the resolved paths without running compare.")

    parser.add_argument("--compare_num_y", type=int, default=121)
    parser.add_argument("--compare_num_psi", type=int, default=121)
    parser.add_argument("--compare_num_v", type=int, default=25)
    parser.add_argument("--compare_v_min", type=float, default=2.0)
    parser.add_argument("--compare_v_max", type=float, default=8.0)
    parser.add_argument("--compare_max_scatter_points", type=int, default=20000)
    parser.add_argument(
        "--analytic_kv",
        type=float,
        default=None,
        help="DEPRECATED alias; must match the LQR gain K[0,1] if provided (then ignored).",
    )
    parser.add_argument(
        "--analytic_ky_y",
        type=float,
        default=None,
        help="DEPRECATED alias; must match the LQR gain K[1,0] if provided (then ignored).",
    )
    parser.add_argument(
        "--analytic_ky_psi",
        type=float,
        default=None,
        help="DEPRECATED alias; must match the LQR gain K[1,2] if provided (then ignored).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    run_dir = run_dir.resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run path is not a directory: {run_dir}")
    if not args.output_subdir.strip():
        raise ValueError("--output_subdir must be non-empty")

    metrics_filename = "invariant_metrics.json"
    results_filename = "invariant_compare_results.npz"
    checkpoint_name = str(args.checkpoint_name).strip()
    if not checkpoint_name:
        raise ValueError("--checkpoint_name must be non-empty")
    weights_path = run_dir / checkpoint_name
    configs_path = run_dir / "configs.json"
    out_dir = run_dir / args.output_subdir
    metrics_path = out_dir / metrics_filename
    results_path = out_dir / results_filename

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing final checkpoint: {weights_path}")
    if not configs_path.exists():
        raise FileNotFoundError(f"Missing saved config: {configs_path}")

    if args.skip_existing and metrics_path.exists() and (not args.save_results or results_path.exists()):
        print(f"Skipping existing {metrics_path}")
        return 0

    if args.dry_run:
        print(f"run_dir={run_dir}")
        print(f"weights_path={weights_path}")
        print(f"checkpoint_name={checkpoint_name}")
        print(f"configs_path={configs_path}")
        print(f"output_dir={out_dir}")
        print(f"save_results={args.save_results}")
        print(f"results_path={results_path}")
        print(f"analytic_kv={args.analytic_kv}")
        print(f"analytic_ky_y={args.analytic_ky_y}")
        print(f"analytic_ky_psi={args.analytic_ky_psi}")
        return 0

    print(f"Rerunning invariant compare for {run_dir.name}")
    _, compare_invariant_sets, _ = _load_compare_symbols()
    saved_cfgs = _load_json(configs_path)
    cbf_cfg = _build_cbf_cfg(saved_cfgs, args)
    grid_cfg = _build_grid_cfg(saved_cfgs, args)
    metrics = compare_invariant_sets(
        cbf_cfg=cbf_cfg,
        learned_backup_policy_path=str(weights_path),
        output_dir=out_dir,
        grid_cfg=grid_cfg,
        save_results_path=results_path if args.save_results else None,
    )

    metadata = {
        "source_run_dir": str(run_dir),
        "source_weights_path": str(weights_path),
        "source_configs_path": str(configs_path),
        "checkpoint_used": checkpoint_name,
        "deprecated_analytic_gain_args": {
            "kv": args.analytic_kv,
            "ky_y": args.analytic_ky_y,
            "ky_psi": args.analytic_ky_psi,
        },
        "grid": asdict(grid_cfg),
        "cbf": asdict(cbf_cfg),
        "metrics_path": str(out_dir / metrics_filename),
        "saved_results_path": str(results_path) if args.save_results else None,
        "volume_ratio_learned_over_analytic": metrics["volume_ratio_learned_over_analytic"],
    }
    with open(out_dir / "rerun_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        "Done. "
        f"learned/analytic volume ratio={metrics['volume_ratio_learned_over_analytic']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
