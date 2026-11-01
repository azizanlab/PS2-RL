"""Invariant-set comparison utilities for learned vs analytical backup policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "matplotlib-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from ps2rl.base_controller.unicycle_dlqr import UnicycleDLQR
from ps2rl.sets.base_sets import EllipsoidBaseSet
from ps2rl.sets.unicycle_sets import UnicycleSafeSet
from ps2rl.cil.unicycle_backup_cbf import (
    UnicycleBCBFConfig,
    unicycle_step_euler,
    make_backup_runtime,
)


@dataclass(frozen=True)
class InvariantGridConfig:
    """Grid configuration for invariant-set approximation."""

    y_min: float = -1.8
    y_max: float = 1.8
    num_y: int = 121

    psi_min: float = -np.pi/3
    psi_max: float = np.pi/3
    num_psi: int = 121

    v_min: float = 2.0
    v_max: float = 8.0
    num_v: int = 25

    max_scatter_points: int = 20_000


def _build_checker(
    cfg: UnicycleBCBFConfig,
    *,
    policy_fn,
):
    """Return a batched checker for the strict lane invariant condition."""
    safe_set = UnicycleSafeSet(y_max=cfg.y_max, psi_max=cfg.psi_max)
    base_set = EllipsoidBaseSet(UnicycleDLQR.from_config(cfg), float(cfg.base_set_c))

    def single(x0: jax.Array):
        safe0 = safe_set.contains(x0)
        base0 = base_set.contains(x0)

        def step(carry, _):
            x, safe_ok, entered_terminal, left_after_terminal_entry = carry
            u = policy_fn(x)
            x_next = unicycle_step_euler(
                x,
                u,
                dt=cfg.dt,
                v_min=cfg.v_min,
                v_max=cfg.v_max,
            )
            terminal_next = base_set.contains(x_next)
            safe_next = safe_ok & safe_set.contains(x_next)
            entered_terminal_next = entered_terminal | terminal_next
            left_after_terminal_entry_next = left_after_terminal_entry | (entered_terminal & (~terminal_next))
            return (
                x_next,
                safe_next,
                entered_terminal_next,
                left_after_terminal_entry_next,
            ), None

        (_, safe_ok, entered_terminal, left_after_terminal_entry), _ = jax.lax.scan(
            step,
            (
                x0,
                safe0,
                base0,
                jnp.asarray(False, dtype=jnp.bool_),
            ),
            jnp.arange(cfg.num_steps),
        )
        return safe_ok & entered_terminal & (~left_after_terminal_entry)

    return jax.jit(jax.vmap(single, in_axes=0))


def _sample_points(mask: np.ndarray, points: np.ndarray, max_count: int, seed: int = 0) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if len(idx) <= max_count:
        return points[idx]
    rng = np.random.default_rng(seed)
    sel = rng.choice(idx, size=max_count, replace=False)
    return points[sel]


def _save_results_bundle(
    *,
    y_vals: np.ndarray,
    psi_vals: np.ndarray,
    v_vals: np.ndarray,
    analytic_masks: np.ndarray,
    learned_masks: np.ndarray,
    analytic_points_plot: np.ndarray,
    learned_points_plot: np.ndarray,
    max_scatter_points: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        y_vals=np.asarray(y_vals, dtype=np.float64),
        psi_vals=np.asarray(psi_vals, dtype=np.float64),
        v_vals=np.asarray(v_vals, dtype=np.float64),
        analytic_success_mask=np.asarray(analytic_masks, dtype=bool),
        learned_success_mask=np.asarray(learned_masks, dtype=bool),
        analytic_points_plot=np.asarray(analytic_points_plot, dtype=np.float64),
        learned_points_plot=np.asarray(learned_points_plot, dtype=np.float64),
        max_scatter_points=np.asarray(max_scatter_points, dtype=np.int32),
        mask_axis_order=np.asarray(["v", "y", "psi"]),
        scatter_columns=np.asarray(["y", "v", "psi"]),
    )


def _plot_slice_overlays(
    y_vals: np.ndarray,
    psi_vals: np.ndarray,
    v_vals: np.ndarray,
    analytic_masks: np.ndarray,
    learned_masks: np.ndarray,
    output_path: Path,
) -> None:
    cmap = ListedColormap(["#f2f2f2", "#1f77b4", "#ff7f0e", "#2ca02c"])
    labels = {0: "outside both", 1: "analytic only", 2: "learned only", 3: "both"}

    num_slices = len(v_vals)
    cols = min(3, num_slices)
    rows = int(np.ceil(num_slices / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.0 * rows), squeeze=False)
    y_grid, psi_grid = np.meshgrid(y_vals, psi_vals, indexing="xy")

    for i, v0 in enumerate(v_vals):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        a = analytic_masks[i]
        l = learned_masks[i]
        cat = a.astype(np.int32) + 2 * l.astype(np.int32)
        im = ax.pcolormesh(y_grid, psi_grid, cat.T, cmap=cmap, shading="auto", vmin=0, vmax=3)
        _ = im
        ax.set_title(f"v={v0:.2f}")
        ax.set_xlabel("y")
        ax.set_ylabel("psi")
        ax.grid(alpha=0.2)

    for j in range(num_slices, rows * cols):
        r = j // cols
        c = j % cols
        axes[r][c].axis("off")

    handles = [
        plt.Line2D([0], [0], color=cmap.colors[k], lw=8, label=labels[k]) for k in [0, 1, 2, 3]
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Invariant-Set Slice Overlay (y, psi) per v", y=0.99)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_3d_scatter(
    analytic_points: np.ndarray,
    learned_points: np.ndarray,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    if len(analytic_points):
        ax1.scatter(analytic_points[:, 0], analytic_points[:, 2], analytic_points[:, 1], s=2, alpha=0.6, c="#1f77b4")
    ax1.set_title("Analytical Backup: Safe Initial Points")
    ax1.set_xlabel("y")
    ax1.set_ylabel("psi")
    ax1.set_zlabel("v")

    if len(learned_points):
        ax2.scatter(learned_points[:, 0], learned_points[:, 2], learned_points[:, 1], s=2, alpha=0.6, c="#ff7f0e")
    ax2.set_title("Learned Backup: Safe Initial Points")
    ax2.set_xlabel("y")
    ax2.set_ylabel("psi")
    ax2.set_zlabel("v")

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def compare_invariant_sets(
    *,
    cbf_cfg: UnicycleBCBFConfig,
    learned_backup_policy_path: str,
    output_dir: str | Path,
    grid_cfg: InvariantGridConfig,
    save_results_path: str | Path | None = None,
) -> Dict[str, object]:
    """Compare invariant set approximations for analytic and learned backup policies."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    analytic_cfg = UnicycleBCBFConfig(**{**asdict(cbf_cfg), "backup_policy_mode": "analytic", "learned_backup_policy_path": ""})
    learned_cfg = UnicycleBCBFConfig(
        **{
            **asdict(cbf_cfg),
            "backup_policy_mode": "learned",
            "learned_backup_policy_path": learned_backup_policy_path,
            "use_analytic_jacobian": False,
        }
    )
    analytic_runtime = make_backup_runtime(analytic_cfg)
    learned_runtime = make_backup_runtime(learned_cfg)

    analytic_checker = _build_checker(analytic_cfg, policy_fn=analytic_runtime.backup_policy_fn)
    learned_checker = _build_checker(learned_cfg, policy_fn=learned_runtime.backup_policy_fn)

    y_vals = np.linspace(grid_cfg.y_min, grid_cfg.y_max, grid_cfg.num_y, dtype=np.float64)
    psi_vals = np.linspace(grid_cfg.psi_min, grid_cfg.psi_max, grid_cfg.num_psi, dtype=np.float64)
    v_vals = np.linspace(grid_cfg.v_min, grid_cfg.v_max, grid_cfg.num_v, dtype=np.float64)
    dy = float(y_vals[1] - y_vals[0]) if len(y_vals) > 1 else 1.0
    dpsi = float(psi_vals[1] - psi_vals[0]) if len(psi_vals) > 1 else 1.0
    dv = float(v_vals[1] - v_vals[0]) if len(v_vals) > 1 else 1.0

    slice_masks_analytic = []
    slice_masks_learned = []
    slice_records = []

    for i, v0 in enumerate(v_vals):
        yy, pp = np.meshgrid(y_vals, psi_vals, indexing="xy")
        points = np.stack([yy.reshape(-1), np.full(yy.size, v0), pp.reshape(-1)], axis=1).astype(np.float32)
        p_j = jnp.asarray(points, dtype=jnp.float32)
        a_ok = analytic_checker(p_j)
        l_ok = learned_checker(p_j)
        # meshgrid(indexing="xy") creates arrays with shape [num_psi, num_y].
        # Flattening therefore iterates psi-major then y-minor.
        # Rebuild in that order, then transpose to keep mask convention [num_y, num_psi].
        a_ok_np = np.asarray(a_ok, dtype=bool).reshape(len(psi_vals), len(y_vals)).T
        l_ok_np = np.asarray(l_ok, dtype=bool).reshape(len(psi_vals), len(y_vals)).T
        slice_masks_analytic.append(a_ok_np)
        slice_masks_learned.append(l_ok_np)

        a_count = int(np.sum(a_ok_np))
        l_count = int(np.sum(l_ok_np))
        total = int(a_ok_np.size)
        record = {
            "slice_index": i,
            "v": float(v0),
            "area_analytic": float(a_count * dy * dpsi),
            "area_learned": float(l_count * dy * dpsi),
            "area_ratio_learned_over_analytic": float((l_count + 1e-9) / (a_count + 1e-9)),
            "analytic_fraction": float(a_count / total),
            "learned_fraction": float(l_count / total),
        }
        slice_records.append(record)

    analytic_masks = np.asarray(slice_masks_analytic, dtype=bool)
    learned_masks = np.asarray(slice_masks_learned, dtype=bool)

    yy3, vv3, pp3 = np.meshgrid(y_vals, v_vals, psi_vals, indexing="xy")
    points3 = np.stack([yy3.reshape(-1), vv3.reshape(-1), pp3.reshape(-1)], axis=1).astype(np.float32)
    points3_j = jnp.asarray(points3, dtype=jnp.float32)
    a3_ok = analytic_checker(points3_j)
    l3_ok = learned_checker(points3_j)
    a3_ok_np = np.asarray(a3_ok, dtype=bool)
    l3_ok_np = np.asarray(l3_ok, dtype=bool)

    total3 = int(points3.shape[0])
    a_count3 = int(np.sum(a3_ok_np))
    l_count3 = int(np.sum(l3_ok_np))
    learned_only_count3 = int(np.sum((~a3_ok_np) & l3_ok_np))
    analytic_only_count3 = int(np.sum(a3_ok_np & (~l3_ok_np)))
    voxel_volume = dy * dv * dpsi

    analytic_points_plot = _sample_points(a3_ok_np, points3.astype(np.float64), grid_cfg.max_scatter_points, seed=0)
    learned_points_plot = _sample_points(l3_ok_np, points3.astype(np.float64), grid_cfg.max_scatter_points, seed=1)

    if save_results_path is not None:
        _save_results_bundle(
            y_vals=y_vals,
            psi_vals=psi_vals,
            v_vals=v_vals,
            analytic_masks=analytic_masks,
            learned_masks=learned_masks,
            analytic_points_plot=analytic_points_plot,
            learned_points_plot=learned_points_plot,
            max_scatter_points=grid_cfg.max_scatter_points,
            output_path=Path(save_results_path),
        )

    _plot_slice_overlays(
        y_vals=y_vals,
        psi_vals=psi_vals,
        v_vals=v_vals,
        analytic_masks=analytic_masks,
        learned_masks=learned_masks,
        output_path=out_dir / "slice_overlay.png",
    )
    _plot_3d_scatter(
        analytic_points=analytic_points_plot,
        learned_points=learned_points_plot,
        output_path=out_dir / "voxel_pointcloud_compare.png",
    )

    slice_csv_path = out_dir / "slice_area_comparison.csv"
    with open(slice_csv_path, "w", encoding="utf-8") as f:
        header = [
            "slice_index",
            "v",
            "area_analytic",
            "area_learned",
            "area_ratio_learned_over_analytic",
            "analytic_fraction",
            "learned_fraction",
        ]
        f.write(",".join(header) + "\n")
        for row in slice_records:
            f.write(",".join(str(row[k]) for k in header) + "\n")

    metrics = {
        "grid": asdict(grid_cfg),
        "cbf": asdict(cbf_cfg),
        "total_grid_points_3d": total3,
        "voxel_volume": float(voxel_volume),
        "analytic_safe_count": a_count3,
        "learned_safe_count": l_count3,
        "analytic_safe_fraction": float(a_count3 / total3),
        "learned_safe_fraction": float(l_count3 / total3),
        "volume_ratio_learned_over_analytic": float((l_count3 + 1e-9) / (a_count3 + 1e-9)),
        "learned_only_count": learned_only_count3,
        "analytic_only_count": analytic_only_count3,
        "learned_only_fraction": float(learned_only_count3 / total3),
        "analytic_only_fraction": float(analytic_only_count3 / total3),
        "slice_records": slice_records,
    }
    with open(out_dir / "invariant_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics
