"""Matplotlib plotting for training and benchmark results."""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _as_np(x):
    return np.asarray(x, dtype=np.float64)


def _normalize_quaternion_batch(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    nrm = np.linalg.norm(q, axis=1, keepdims=True)
    qn = q / np.maximum(nrm, eps)
    q_id = np.zeros_like(qn)
    q_id[:, 0] = 1.0
    return np.where(nrm > eps, qn, q_id)


def _pitch_from_quaternion_batch(q: np.ndarray) -> np.ndarray:
    qn = _normalize_quaternion_batch(q)
    qw, qx, qy, qz = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
    sin_pitch = 2.0 * (qw * qy - qx * qz)
    return np.arcsin(np.clip(sin_pitch, -1.0, 1.0))


def plot_training_metrics(history: Dict[str, List[float]], output_path: str, eval_every: int = 1) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    ax = axes.ravel()

    step = _as_np(history.get("step", []))
    ep_return = _as_np(history.get("ep_return", []))
    ep_safe = _as_np(history.get("ep_safe_rate", []))
    ep_speed = _as_np(history.get("ep_speed_error_abs", []))
    ep_y = _as_np(history.get("ep_y_error_abs", []))
    ep_psi = _as_np(history.get("ep_psi_error_abs", []))

    ax[0].plot(step, ep_return, linewidth=1.0)
    ax[0].set_title("Episode Return")
    ax[0].set_xlabel("Step")
    ax[0].set_ylabel("Return")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(step, ep_safe, label="Safe Rate", linewidth=1.0)
    ax[1].plot(step, ep_speed, label="Speed Err |v-v_ref|", linewidth=1.0)
    if ep_y.size > 0:
        ax[1].plot(step, ep_y, label="Y Err |y-y_ref|", linewidth=1.0)
    if ep_psi.size > 0:
        ax[1].plot(step, ep_psi, label="Psi Err |psi-psi_ref|", linewidth=1.0)
    ax[1].set_title("Episode Safety/Tracking")
    ax[1].set_xlabel("Step")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    critic_loss = _as_np(history.get("critic_loss", []))
    actor_loss = _as_np(history.get("actor_loss", []))
    ax[2].plot(critic_loss, label="Critic Loss", linewidth=1.0)
    ax[2].plot(actor_loss, label="Actor Loss", linewidth=1.0)
    ax[2].set_title("Update Losses")
    ax[2].set_xlabel("Update Step")
    ax[2].grid(True, alpha=0.3)
    ax[2].legend()

    alpha = _as_np(history.get("alpha", []))
    ax[3].plot(alpha, label="Alpha", linewidth=1.0)
    ax[3].set_title("Entropy Temperature")
    ax[3].set_xlabel("Update Step")
    ax[3].grid(True, alpha=0.3)
    ax[3].legend()

    eval_ret = _as_np(history.get("eval_return_mean", []))
    eval_safe = _as_np(history.get("eval_safe_rate", []))
    eval_speed = _as_np(history.get("eval_speed_error_abs_mean", []))
    eval_y = _as_np(history.get("eval_y_error_abs_mean", []))
    eval_psi = _as_np(history.get("eval_psi_error_abs_mean", []))
    eval_x = np.arange(len(eval_ret)) * eval_every
    ax[4].plot(eval_x, eval_ret, label="Eval Return", linewidth=1.0)
    ax[4].set_title("Evaluation Return")
    ax[4].set_xlabel("Step")
    ax[4].grid(True, alpha=0.3)
    ax[4].legend()

    ax[5].plot(eval_x, eval_safe, label="Eval Safe Rate", linewidth=1.0)
    ax[5].plot(eval_x, eval_speed, label="Eval Speed Err", linewidth=1.0)
    if eval_y.size > 0:
        ax[5].plot(eval_x, eval_y, label="Eval Y Err", linewidth=1.0)
    if eval_psi.size > 0:
        ax[5].plot(eval_x, eval_psi, label="Eval Psi Err", linewidth=1.0)
    ax[5].set_title("Evaluation Safety/Tracking")
    ax[5].set_xlabel("Step")
    ax[5].grid(True, alpha=0.3)
    ax[5].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_trajectory(
    trajectory: Dict[str, np.ndarray],
    y_max: float,
    psi_max: float,
    v_des: float,
    output_path: str,
    reward_mode: str = "",
    dt: float = 1.0,
) -> None:
    obs = np.asarray(trajectory.get("next_obs", trajectory["obs"]), dtype=np.float64)
    act = np.asarray(trajectory["act"], dtype=np.float64)
    rew = np.asarray(trajectory["rew"], dtype=np.float64)
    safe = np.asarray(trajectory["safe"], dtype=np.float64)
    y_ref = np.asarray(trajectory.get("y_ref", []), dtype=np.float64)
    v_ref = np.asarray(trajectory.get("v_ref", []), dtype=np.float64)
    psi_ref = np.asarray(trajectory.get("psi_ref", []), dtype=np.float64)
    speed_err = np.asarray(trajectory.get("speed_error_abs", []), dtype=np.float64)
    y_err = np.asarray(trajectory.get("y_error_abs", []), dtype=np.float64)
    psi_err = np.asarray(trajectory.get("psi_error_abs", []), dtype=np.float64)
    t = np.asarray(trajectory.get("ref_time_sec", []), dtype=np.float64)

    if len(obs) == 0:
        return

    if t.size != len(obs):
        t = np.arange(len(obs), dtype=np.float64) * float(dt)
    use_ref_overlay = (
        reward_mode == "trajectory_following"
        and y_ref.size == len(obs)
        and v_ref.size == len(obs)
        and psi_ref.size == len(obs)
    )
    fig, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)

    axes[0].plot(t, obs[:, 0], label="Y", linewidth=1.2)
    if use_ref_overlay:
        axes[0].plot(t, y_ref, label="Y_ref", linewidth=1.0, linestyle="--")
    axes[0].axhline(y_max, color="r", linestyle="--", linewidth=1.0)
    axes[0].axhline(-y_max, color="r", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Lateral Y")
    axes[0].set_title("Lane Keeping Trajectory")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(t, obs[:, 2], label="psi", linewidth=1.2)
    if use_ref_overlay:
        axes[1].plot(t, psi_ref, label="psi_ref", linewidth=1.0, linestyle="--")
    axes[1].axhline(psi_max, color="r", linestyle="--", linewidth=1.0)
    axes[1].axhline(-psi_max, color="r", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Heading psi")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(t, obs[:, 1], label="v", linewidth=1.2)
    if use_ref_overlay:
        axes[2].plot(t, v_ref, color="g", linestyle="--", linewidth=1.0, label="v_ref")
    else:
        axes[2].axhline(v_des, color="g", linestyle="--", linewidth=1.0, label="v_des")
    axes[2].set_ylabel("Velocity")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    if speed_err.size == len(obs):
        axes[3].plot(t, speed_err, label="|v-v_ref|", linewidth=1.1)
    if y_err.size == len(obs):
        axes[3].plot(t, y_err, label="|y-y_ref|", linewidth=1.1)
    if psi_err.size == len(obs):
        axes[3].plot(t, psi_err, label="|psi-psi_ref|", linewidth=1.1)
    if speed_err.size != len(obs) and y_err.size != len(obs) and psi_err.size != len(obs):
        axes[3].plot(t, np.abs(obs[:, 0]), label="|y|", linewidth=1.1)
        axes[3].plot(t, np.abs(obs[:, 2]), label="|psi|", linewidth=1.1)
    axes[3].set_ylabel("Tracking Err")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")

    axes[4].plot(t, safe, label="safe flag", linewidth=1.2)
    axes[4].plot(t, rew, label="reward", linewidth=1.2)
    axes[4].set_ylabel("Safety/Reward")
    axes[4].set_xlabel("Time (s)")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    # Action overlays (secondary axis)
    ax2 = axes[2].twinx()
    ax2.plot(t, act[:, 0], color="tab:orange", alpha=0.5, linewidth=1.0, label="a")
    ax2.plot(t, act[:, 1], color="tab:purple", alpha=0.5, linewidth=1.0, label="r")
    ax2.set_ylabel("Actions")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_benchmark_runtime(records: List[Dict[str, float]], output_path: str) -> None:
    if len(records) == 0:
        return
    ineq = np.asarray([r["num_qp_inequalities"] for r in records], dtype=np.float64)
    sps = np.asarray([r["steps_per_sec"] for r in records], dtype=np.float64)
    safe = np.asarray([r["final_eval_safe_rate"] for r in records], dtype=np.float64)
    speed_err = np.asarray([r["final_eval_speed_error_abs_mean"] for r in records], dtype=np.float64)
    horizon = np.asarray([r["horizon_T"] for r in records], dtype=np.float64)

    order = np.argsort(ineq)
    ineq = ineq[order]
    sps = sps[order]
    safe = safe[order]
    speed_err = speed_err[order]
    horizon = horizon[order]

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    axes[0].plot(ineq, sps, marker="o")
    axes[0].set_ylabel("Env Steps / Sec")
    axes[0].set_title("Training Throughput vs QP Inequality Count")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ineq, safe, marker="o")
    axes[1].set_ylabel("Final Eval Safe Rate")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ineq, speed_err, marker="o", label="|v-v_des|")
    axes[2].plot(ineq, horizon, marker="x", label="T")
    axes[2].set_xlabel("Number of QP Inequalities")
    axes[2].set_ylabel("Tracking/Horizon")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_quad_trajectory(
    trajectory: Dict[str, np.ndarray],
    z_max: float,
    output_path: str,
    dt: float = 1.0,
) -> None:
    obs_full = np.asarray(trajectory.get("next_obs", trajectory.get("obs", [])), dtype=np.float64)
    if obs_full.size == 0:
        return
    x = obs_full[:, :10]
    act = np.asarray(trajectory.get("act", []), dtype=np.float64)
    rew = np.asarray(trajectory.get("rew", []), dtype=np.float64)
    safe = np.asarray(trajectory.get("safe", []), dtype=np.float64)
    pos_err = np.asarray(trajectory.get("pos_error_norm", []), dtype=np.float64)
    vel_err = np.asarray(trajectory.get("vel_error_norm", []), dtype=np.float64)
    att_err = np.asarray(trajectory.get("att_error_norm", []), dtype=np.float64)
    hard_deck_margin = np.asarray(trajectory.get("hard_deck_margin", []), dtype=np.float64)
    ref_t = np.asarray(trajectory.get("ref_time_sec", []), dtype=np.float64)
    ref_px = np.asarray(trajectory.get("ref_px", []), dtype=np.float64)
    ref_py = np.asarray(trajectory.get("ref_py", []), dtype=np.float64)
    ref_pz = np.asarray(trajectory.get("ref_pz", []), dtype=np.float64)
    ref_vx = np.asarray(trajectory.get("ref_vx", []), dtype=np.float64)
    ref_vy = np.asarray(trajectory.get("ref_vy", []), dtype=np.float64)
    ref_vz = np.asarray(trajectory.get("ref_vz", []), dtype=np.float64)
    ref_qw = np.asarray(trajectory.get("ref_qw", []), dtype=np.float64)
    ref_qx = np.asarray(trajectory.get("ref_qx", []), dtype=np.float64)
    ref_qy = np.asarray(trajectory.get("ref_qy", []), dtype=np.float64)
    ref_qz = np.asarray(trajectory.get("ref_qz", []), dtype=np.float64)

    n = x.shape[0]
    if ref_t.size != n:
        t = np.arange(n, dtype=np.float64) * float(dt)
    else:
        t = ref_t

    q_now = x[:, 6:10]
    pitch_now = np.rad2deg(_pitch_from_quaternion_batch(q_now))
    has_ref_q = ref_qw.size == n and ref_qx.size == n and ref_qy.size == n and ref_qz.size == n
    if has_ref_q:
        q_ref = np.column_stack([ref_qw, ref_qx, ref_qy, ref_qz])
        pitch_ref = np.rad2deg(_pitch_from_quaternion_batch(q_ref))
    else:
        pitch_ref = np.zeros((n,), dtype=np.float64)

    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True)

    axes[0].plot(t, x[:, 2], label="z", linewidth=1.2)
    if ref_pz.size == n:
        axes[0].plot(t, ref_pz, label="z_ref", linewidth=1.0, linestyle="--")
    axes[0].axhline(float(z_max), color="r", linestyle="--", linewidth=1.0, label="hard deck (z_max)")
    axes[0].set_ylabel("Altitude [m]")
    axes[0].set_title("Quadrotor Trajectory Tracking")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(t, x[:, 0], label="x", linewidth=1.2)
    axes[1].plot(t, x[:, 1], label="y", linewidth=1.2)
    if ref_px.size == n:
        axes[1].plot(t, ref_px, label="x_ref", linewidth=1.0, linestyle="--")
    if ref_py.size == n:
        axes[1].plot(t, ref_py, label="y_ref", linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("Position [m]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(t, x[:, 3], label="vx", linewidth=1.2)
    axes[2].plot(t, x[:, 4], label="vy", linewidth=1.2)
    axes[2].plot(t, x[:, 5], label="vz", linewidth=1.2)
    if ref_vx.size == n:
        axes[2].plot(t, ref_vx, label="vx_ref", linewidth=1.0, linestyle="--")
    if ref_vy.size == n:
        axes[2].plot(t, ref_vy, label="vy_ref", linewidth=1.0, linestyle="--")
    if ref_vz.size == n:
        axes[2].plot(t, ref_vz, label="vz_ref", linewidth=1.0, linestyle="--")
    axes[2].set_ylabel("Velocity [m/s]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    axes[3].plot(t, pitch_now, label="pitch [deg]", linewidth=1.2)
    if has_ref_q:
        axes[3].plot(t, pitch_ref, label="pitch_ref [deg]", linewidth=1.0, linestyle="--")
    if pos_err.size == n:
        axes[3].plot(t, pos_err, label="||pos_err||", linewidth=1.1)
    if vel_err.size == n:
        axes[3].plot(t, vel_err, label="||vel_err||", linewidth=1.1)
    if att_err.size == n:
        axes[3].plot(t, att_err, label="||att_err||", linewidth=1.1)
    axes[3].set_ylabel("Att/Errors")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")

    axes[4].plot(t, safe, label="safe flag", linewidth=1.2)
    axes[4].plot(t, rew, label="reward", linewidth=1.2)
    if hard_deck_margin.size == n:
        axes[4].plot(t, hard_deck_margin, label="hard_deck_margin", linewidth=1.1)
    axes[4].set_ylabel("Safety/Reward")
    axes[4].set_xlabel("Time [s]")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc="best")

    if act.ndim == 2 and act.shape[0] == n and act.shape[1] == 4:
        ax2 = axes[2].twinx()
        ax2.plot(t, act[:, 0], color="tab:orange", alpha=0.5, linewidth=1.0, label="a_cmd")
        ax2.plot(t, act[:, 1], color="tab:purple", alpha=0.45, linewidth=1.0, label="omega_x")
        ax2.plot(t, act[:, 2], color="tab:brown", alpha=0.45, linewidth=1.0, label="omega_y")
        ax2.plot(t, act[:, 3], color="tab:gray", alpha=0.45, linewidth=1.0, label="omega_z")
        ax2.set_ylabel("Actions")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
