#!/usr/bin/env python3
"""Generate a feasible bottom-start power-loop reference for QuadrotorEnv.

State ordering (10D):
    [p_x, p_y, p_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z]

This script builds a single vertical circular loop that starts at the bottom
(z = 0.5 m), reaches the top (z = 3.5 m), and uses constant tangential
speed (default 4.5 m/s, radius 1.5 m), then saves:
1) a legacy state-only trajectory array (.npy),
2) a richer reference bundle (.npz) containing states, time, thrust, and omega_cmd,
3) a diagnostic figure of position/velocity/attitude/quaternion-continuity traces (.png),
4) a 3D animation (.gif), and
5) a JSON file containing all generation/output configuration values.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PowerLoopConfig:
    dt: float = 0.02
    radius: float = 1.5
    speed: float = 4.5
    z_top: float = 3.5
    z_bottom: float = 0.5
    x_center: float = 0.0
    y_center: float = 0.0
    gravity: float = 9.81
    a_cmd_max_g: float = 4.0
    omega_max: float = 18.0
    num_loops: int = 1
    speed_margin_eps: float = 1.1
    z_max: float = 3.0

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.radius <= 0.0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        if self.speed <= 0.0:
            raise ValueError(f"speed must be positive, got {self.speed}")
        if self.gravity <= 0.0:
            raise ValueError(f"gravity must be positive, got {self.gravity}")
        if self.a_cmd_max_g <= 0.0:
            raise ValueError(f"a_cmd_max_g must be positive, got {self.a_cmd_max_g}")
        if self.omega_max <= 0.0:
            raise ValueError(f"omega_max must be positive, got {self.omega_max}")
        if int(self.num_loops) <= 0:
            raise ValueError(f"num_loops must be >= 1, got {self.num_loops}")

        z_span = self.z_top - self.z_bottom
        expected_span = 2.0 * self.radius
        if abs(z_span - expected_span) > 1e-6:
            raise ValueError(
                "Inconsistent geometry: expected z_top - z_bottom == 2 * radius, "
                f"but got {z_span:.6f} vs {expected_span:.6f}."
            )
        
        critical_speed = self.speed_margin_eps * np.sqrt(self.radius * self.gravity)
        if self.speed <= critical_speed:
            raise ValueError(
                "Paper condition violated: requires ||v_l|| > eps*sqrt(r*g). "
                f"speed={self.speed:.6f}, eps*sqrt(r*g)={critical_speed:.6f} "
                f"(eps={self.speed_margin_eps}, r={self.radius}, g={self.gravity})."
            )



def _build_time_grid(total_duration: float, dt: float) -> np.ndarray:
    n_full = int(np.floor(total_duration / dt))
    t = dt * np.arange(n_full + 1, dtype=np.float64)
    if total_duration - t[-1] > 1e-12:
        t = np.append(t, total_duration)
    return t


def _generate_powerloop(cfg: PowerLoopConfig) -> dict[str, np.ndarray]:
    cfg.validate()

    z_center = 0.5 * (cfg.z_top + cfg.z_bottom)
    omega_loop = cfg.speed / cfg.radius
    loop_duration = (2.0 * np.pi * cfg.radius) / cfg.speed
    total_duration = float(cfg.num_loops) * loop_duration
    t = _build_time_grid(total_duration, cfg.dt)

    theta = omega_loop * t + np.pi
    y = np.full_like(t, cfg.y_center)
    vy = np.zeros_like(t)
    ay = np.zeros_like(t)

    x = cfg.x_center + cfg.radius * np.sin(theta)
    z = z_center + cfg.radius * np.cos(theta)

    vx = cfg.speed * np.cos(theta)
    vz = -cfg.speed * np.sin(theta)

    a_c = (cfg.speed * cfg.speed) / cfg.radius
    ax = -a_c * np.sin(theta)
    az = -a_c * np.cos(theta)

    nu = np.stack([ax, ay, az + cfg.gravity], axis=1)
    a_cmd = np.linalg.norm(nu, axis=1)
    if np.any(a_cmd < 1e-12):
        raise RuntimeError("Encountered near-zero required thrust; orientation becomes undefined.")

    b3 = nu / a_cmd[:, None]
    pitch = np.unwrap(np.arctan2(b3[:, 0], b3[:, 2]))
    roll = np.zeros_like(pitch)
    yaw = np.zeros_like(pitch)

    half_pitch = 0.5 * pitch
    qw = np.cos(half_pitch)
    qx = np.zeros_like(qw)
    qy = np.sin(half_pitch)
    qz = np.zeros_like(qw)

    states = np.column_stack([x, y, z, vx, vy, vz, qw, qx, qy, qz]).astype(np.float64)
    quat = states[:, 6:10]
    quat_norm = np.linalg.norm(quat, axis=1)
    quat_step_norm = np.zeros_like(t)
    if quat.shape[0] > 1:
        quat_step_norm[1:] = np.linalg.norm(np.diff(quat, axis=0), axis=1)

    pitch_rate = np.gradient(pitch, t, edge_order=2 if t.size >= 3 else 1)
    omega_cmd = np.column_stack([np.zeros_like(pitch_rate), pitch_rate, np.zeros_like(pitch_rate)])

    a_cmd_limit = cfg.a_cmd_max_g * cfg.gravity
    max_a_cmd = float(np.max(a_cmd))
    max_abs_omega = float(np.max(np.abs(omega_cmd)))
    if max_a_cmd > a_cmd_limit + 1e-6:
        raise RuntimeError(
            f"Trajectory violates thrust limit: max a_cmd={max_a_cmd:.3f} > {a_cmd_limit:.3f} m/s^2."
        )
    if max_abs_omega > cfg.omega_max + 1e-6:
        raise RuntimeError(
            f"Trajectory violates body-rate limit: max |omega|={max_abs_omega:.3f} > {cfg.omega_max:.3f} rad/s."
        )

    return {
        "t": t,
        "states": states,
        "a_cmd": a_cmd,
        "omega_cmd": omega_cmd,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "quat_norm": quat_norm,
        "quat_step_norm": quat_step_norm,
        "max_a_cmd": np.array([max_a_cmd], dtype=np.float64),
        "max_abs_omega": np.array([max_abs_omega], dtype=np.float64),
        "max_quat_step_norm": np.array([float(np.max(quat_step_norm))], dtype=np.float64),
        "max_quat_norm_error": np.array([float(np.max(np.abs(quat_norm - 1.0)))], dtype=np.float64),
    }


def _save_plot(data: dict[str, np.ndarray], output_path: Path) -> None:
    t = data["t"]
    states = data["states"]
    roll_deg = np.rad2deg(data["roll"])
    pitch_deg = np.rad2deg(data["pitch"])
    yaw_deg = np.rad2deg(data["yaw"])
    quat_norm = data["quat_norm"]
    quat_step_norm = data["quat_step_norm"]

    fig, axes = plt.subplots(5, 3, figsize=(15, 12), sharex=True)
    series = [
        (states[:, 0], "x [m]"),
        (states[:, 1], "y [m]"),
        (states[:, 2], "z [m]"),
        (states[:, 3], "xdot [m/s]"),
        (states[:, 4], "ydot [m/s]"),
        (states[:, 5], "zdot [m/s]"),
        (roll_deg, "roll [deg]"),
        (pitch_deg, "pitch [deg]"),
        (yaw_deg, "yaw [deg]"),
        (states[:, 6], "q_w"),
        (states[:, 7], "q_x"),
        (states[:, 8], "q_y"),
        (states[:, 9], "q_z"),
        (quat_step_norm, "||Delta q||"),
        (quat_norm, "||q||"),
    ]

    for ax, (y, ylabel) in zip(axes.ravel(), series):
        ax.plot(t, y, linewidth=2.0)
        ax.set_ylabel(ylabel)
        if ylabel in {"q_w", "q_x", "q_y", "q_z"}:
            ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.35)
        if ylabel == "||q||":
            ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.35)
        ax.grid(True, alpha=0.35)

    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")

    fig.suptitle(
        "Bottom-Start Power-Loop Reference (Quadrotor 10D State)\n"
        f"max ||Delta q|| = {float(np.max(quat_step_norm)):.4f}, "
        f"max | ||q|| - 1 | = {float(np.max(np.abs(quat_norm - 1.0))):.2e}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _normalize_quaternion(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    nrm = float(np.linalg.norm(q))
    if nrm <= eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / nrm


def _rotation_matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = _normalize_quaternion(q)
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qw * qz),
                2.0 * (qx * qz + qw * qy),
            ],
            [
                2.0 * (qx * qy + qw * qz),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qw * qx),
            ],
            [
                2.0 * (qx * qz - qw * qy),
                2.0 * (qy * qz + qw * qx),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def _quad_world_frame(state: np.ndarray, arm_length: float = 0.258, height: float = 0.15) -> np.ndarray:
    """Return 3x6 world-frame quad points [m1,m2,m3,m4,origin,head]."""
    pos = np.asarray(state[0:3], dtype=np.float64)
    q = np.asarray(state[6:10], dtype=np.float64)
    rot = _rotation_matrix_from_quaternion(q)
    body_points = np.array(
        [
            [arm_length, 0.0, 0.0],
            [0.0, arm_length, 0.0],
            [-arm_length, 0.0, 0.0],
            [0.0, -arm_length, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, height],
        ],
        dtype=np.float64,
    ).T  # 3x6
    return (rot @ body_points) + pos[:, None]


def _save_animation(
    data: dict[str, np.ndarray],
    output_path: Path,
    *,
    z_max: float,
    slowdown: float,
    trail_length: int,
    print_every: int,
    title: str = "",
) -> None:
    if slowdown <= 0.0:
        raise ValueError(f"animation slowdown must be positive, got {slowdown}")
    if trail_length <= 0:
        raise ValueError(f"animation trail length must be positive, got {trail_length}")
    if print_every <= 0:
        raise ValueError(f"animation print_every must be positive, got {print_every}")

    t = np.asarray(data["t"], dtype=np.float64)
    states = np.asarray(data["states"], dtype=np.float64)
    pitch_deg = np.rad2deg(np.asarray(data["pitch"], dtype=np.float64))

    n = int(states.shape[0])
    if n == 0:
        raise ValueError("Cannot animate empty trajectory.")
    dt_nom = float(np.mean(np.diff(t))) if n > 1 else 0.02
    traj_duration = float(t[-1] - t[0]) if n > 1 else 0.0
    anim_duration = traj_duration * slowdown
    fps_save = max(1, int(round(1.0 / max(1e-9, dt_nom * slowdown))))

    print(
        "[anim] configuring animation: "
        f"trajectory_duration={traj_duration:.3f}s, slowdown={slowdown:.3f}x, "
        f"animation_duration={anim_duration:.3f}s, save_fps={fps_save}"
    )

    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    pos = states[:, 0:3]
    xyz_min = np.min(pos, axis=0)
    xyz_max = np.max(pos, axis=0)
    xyz_center = 0.5 * (xyz_min + xyz_max)
    span = float(np.max(xyz_max - xyz_min))
    half = 0.6 * max(span, 1e-3) + 0.2
    x_lo, x_hi = xyz_center[0] - half, xyz_center[0] + half
    y_lo, y_hi = xyz_center[1] - half, xyz_center[1] + half
    z_pad = 0.05 * max(span, 1.0)
    z_lo = min(max(0.0, xyz_center[2] - half), float(z_max) - z_pad)
    z_hi = max(xyz_center[2] + half, float(z_max) + z_pad)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(z_lo, z_hi)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.grid(True, alpha=0.35)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=22.0, azim=-70.0)
    if title.strip():
        ax.set_title(title.strip())

    # Hard-deck/safety plane at z_max.
    x_plane = np.array([x_lo, x_hi], dtype=np.float64)
    y_plane = np.array([y_lo, y_hi], dtype=np.float64)
    xx, yy = np.meshgrid(x_plane, y_plane)
    zz = np.full_like(xx, float(z_max))
    ax.plot_surface(xx, yy, zz, color="red", alpha=0.2, linewidth=0.0, shade=False)
    print(f"[anim] safety plane: z_max={float(z_max):.3f} m, color=red, alpha=0.2")

    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], ":", c="darkgreen", alpha=0.35, linewidth=1.0, zorder=1)

    (line_x_arm,) = ax.plot([], [], [], "-", c="tab:blue", linewidth=2.0, zorder=10)
    (line_y_arm,) = ax.plot([], [], [], "-", c="tab:green", linewidth=2.0, zorder=10)
    (line_trail,) = ax.plot([], [], [], ".", c="green", markersize=2.5, zorder=8)
    info_text = ax.text2D(0.02, 0.97, "", transform=ax.transAxes)
    last_printed_frame = {"idx": -1}
    body_up_arrow = {"obj": None}

    def _update(frame_idx: int):
        frame = _quad_world_frame(states[frame_idx])
        x_arm = frame[:, [0, 2]]
        y_arm = frame[:, [1, 3]]
        body_z = frame[:, [4, 5]]

        line_x_arm.set_data(x_arm[0], x_arm[1])
        line_x_arm.set_3d_properties(x_arm[2])

        line_y_arm.set_data(y_arm[0], y_arm[1])
        line_y_arm.set_3d_properties(y_arm[2])

        if body_up_arrow["obj"] is not None:
            body_up_arrow["obj"].remove()
        origin = body_z[:, 0]
        body_up_vec = body_z[:, 1] - body_z[:, 0]
        body_up_arrow["obj"] = ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            body_up_vec[0],
            body_up_vec[1],
            body_up_vec[2],
            color="red",
            linewidth=2.0,
            arrow_length_ratio=0.45,
            normalize=False,
        )

        lo = max(0, frame_idx - trail_length + 1)
        trail = pos[lo : frame_idx + 1]
        line_trail.set_data(trail[:, 0], trail[:, 1])
        line_trail.set_3d_properties(trail[:, 2])

        if frame_idx == 0:
            dx = dy = dz = dp = 0.0
        else:
            dx = float(pos[frame_idx, 0] - pos[frame_idx - 1, 0])
            dy = float(pos[frame_idx, 1] - pos[frame_idx - 1, 1])
            dz = float(pos[frame_idx, 2] - pos[frame_idx - 1, 2])
            dp = float(pitch_deg[frame_idx] - pitch_deg[frame_idx - 1])

        if frame_idx % print_every == 0 and frame_idx != last_printed_frame["idx"]:
            print(
                "[anim] "
                f"frame={frame_idx + 1:04d}/{n:04d} "
                f"t={t[frame_idx]:.3f}s "
                f"x={pos[frame_idx,0]: .4f} (dx={dx:+.4f}) "
                f"y={pos[frame_idx,1]: .4f} (dy={dy:+.4f}) "
                f"z={pos[frame_idx,2]: .4f} (dz={dz:+.4f}) "
                f"pitch={pitch_deg[frame_idx]: .3f}deg (dPitch={dp:+.3f}) "
                f"slowdown={slowdown:.3f}x"
            )
            last_printed_frame["idx"] = frame_idx

        info_text.set_text(
            f"t={t[frame_idx]:.2f}s | pitch={pitch_deg[frame_idx]:.1f} deg | slowdown={slowdown:.2f}x"
        )

        return line_x_arm, line_y_arm, line_trail, info_text, body_up_arrow["obj"]

    interval_ms = 1000.0 * dt_nom * slowdown
    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=n,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = animation.PillowWriter(fps=fps_save)
        anim.save(output_path, writer=writer)
    finally:
        plt.close(fig)

    print(f"[ok] saved animation: {output_path}")


def parse_args() -> argparse.Namespace:
    env_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate a feasible bottom-start power-loop reference trajectory for QuadrotorEnv."
    )
    parser.add_argument("--dt", type=float, default=0.02, help="Sampling timestep [s].")
    parser.add_argument("--radius", type=float, default=1.5, help="Loop radius [m].")
    parser.add_argument("--speed", type=float, default=4.5, help="Constant tangential speed [m/s].")
    parser.add_argument("--z-top", type=float, default=3.5, help="Top altitude [m].")
    parser.add_argument("--z-bottom", type=float, default=0.5, help="Bottom altitude [m].")
    parser.add_argument("--z-max", type=float, default=3.0, help="Safety hard-deck altitude [m] for animation plane.")
    parser.add_argument("--x-center", type=float, default=0.0, help="Loop center x-coordinate [m].")
    parser.add_argument("--y-center", type=float, default=0.0, help="Loop center y-coordinate [m].")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravity [m/s^2].")
    parser.add_argument("--a-cmd-max-g", type=float, default=4.0, help="Mass-normalized thrust limit in g.")
    parser.add_argument("--omega-max", type=float, default=18.0, help="Body-rate limit [rad/s].")
    parser.add_argument("--num-loops", type=int, default=1, help="Number of full loops.")
    parser.add_argument(
        "--output-npy",
        type=str,
        default=str(env_dir / "quadrotor_powerloop_reference_legacy.npy"),
        help="Legacy output path for the state-only 10D reference trajectory array.",
    )
    parser.add_argument(
        "--output-bundle",
        type=str,
        default=str(env_dir / "quadrotor_powerloop_reference.npz"),
        help="Output path for the richer reference bundle with states, omega_cmd, t, and a_cmd.",
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default=str(env_dir / "quadrotor_powerloop_reference.png"),
        help="Output path for trajectory diagnostic figure.",
    )
    parser.add_argument(
        "--output-animation",
        type=str,
        default=str(env_dir / "quadrotor_powerloop_reference.gif"),
        help="Output path for 3D trajectory animation (GIF).",
    )
    parser.add_argument(
        "--output-config-json",
        type=str,
        default=str(env_dir / "quadrotor_powerloop_reference_config.json"),
        help="Output path for JSON containing all generation and output configurations.",
    )
    parser.add_argument(
        "--animation-slowdown",
        type=float,
        default=2.0,
        help=(
            "Animation slowdown factor. "
            "Example: 2.5 makes animation 2.5x slower than trajectory time."
        ),
    )
    parser.add_argument(
        "--animation-trail-length",
        type=int,
        default=150,
        help="Number of recent points shown in moving dotted trail.",
    )
    parser.add_argument(
        "--animation-print-every",
        type=int,
        default=1,
        help="Print x/y/z/pitch changes every N animation frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PowerLoopConfig(
        dt=float(args.dt),
        radius=float(args.radius),
        speed=float(args.speed),
        z_top=float(args.z_top),
        z_bottom=float(args.z_bottom),
        z_max=float(args.z_max),
        x_center=float(args.x_center),
        y_center=float(args.y_center),
        gravity=float(args.gravity),
        a_cmd_max_g=float(args.a_cmd_max_g),
        omega_max=float(args.omega_max),
        num_loops=int(args.num_loops),
    )
    data = _generate_powerloop(cfg)

    out_npy = Path(args.output_npy).resolve()
    out_bundle = Path(args.output_bundle).resolve()
    out_fig = Path(args.output_figure).resolve()
    out_anim = Path(args.output_animation).resolve()
    out_cfg_json = Path(args.output_config_json).resolve()
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    out_bundle.parent.mkdir(parents=True, exist_ok=True)
    out_cfg_json.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, data["states"])
    np.savez(out_bundle, **data)
    _save_plot(data, out_fig)
    _save_animation(
        data,
        out_anim,
        z_max=float(cfg.z_max),
        slowdown=float(args.animation_slowdown),
        trail_length=int(args.animation_trail_length),
        print_every=int(args.animation_print_every),
    )

    cfg_json_payload = {
        "powerloop_config": asdict(cfg),
        "powerloop_trajectory_stats": {
            "num_points": int(data["t"].shape[0]),
            "total_duration": float(data["t"][-1] - data["t"][0]) if data["t"].shape[0] > 1 else 0.0,
            "max_a_cmd": float(data["max_a_cmd"][0]),
            "max_abs_omega": float(data["max_abs_omega"][0]),
            "max_quat_step_norm": float(data["max_quat_step_norm"][0]),
            "max_quat_norm_error": float(data["max_quat_norm_error"][0]),
        },
        "animation_config": {
            "slowdown": float(args.animation_slowdown),
            "trail_length": int(args.animation_trail_length),
            "print_every": int(args.animation_print_every),
        },
        "output_paths": {
            "trajectory_npy": str(out_npy),
            "reference_bundle_npz": str(out_bundle),
            "diagnostic_figure": str(out_fig),
            "animation_gif": str(out_anim),
            "config_json": str(out_cfg_json),
        },
    }
    with out_cfg_json.open("w", encoding="utf-8") as f:
        json.dump(cfg_json_payload, f, indent=2, sort_keys=True)

    print(f"[ok] saved state trajectory: {out_npy}")
    print(f"[ok] saved reference bundle: {out_bundle}")
    print(f"[ok] saved diagnostic figure: {out_fig}")
    print(f"[ok] saved config json: {out_cfg_json}")
    print(f"[info] trajectory shape: {data['states'].shape}")
    print(f"[info] max required a_cmd: {float(data['max_a_cmd'][0]):.3f} m/s^2")
    print(f"[info] max required |omega|: {float(data['max_abs_omega'][0]):.3f} rad/s")
    print(f"[info] max quaternion step norm: {float(data['max_quat_step_norm'][0]):.4f}")
    print(f"[info] max quaternion norm error: {float(data['max_quat_norm_error'][0]):.2e}")
    print(f"[info] thrust limit (4g default): {(cfg.a_cmd_max_g * cfg.gravity):.3f} m/s^2")
    print(f"[info] body-rate limit: {cfg.omega_max:.3f} rad/s")


if __name__ == "__main__":
    main()
