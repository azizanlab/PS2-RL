"""Quaternion / attitude math, one home for the whole package.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# --------------------------------------------------------------------------- #
# Single-state JAX family (shape (4,) / (3,))                                  #
# --------------------------------------------------------------------------- #


def normalize_quaternion(q: jax.Array, eps: float = 1e-9) -> jax.Array:
    q = jnp.asarray(q)
    sq_norm = jnp.sum(jnp.square(q))
    nrm = jnp.sqrt(jnp.maximum(sq_norm, eps**2))
    q_id = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=q.dtype)
    q_norm = q / nrm
    return jnp.where(sq_norm > eps**2, q_norm, q_id)


def quaternion_conjugate(q: jax.Array) -> jax.Array:
    q = jnp.asarray(q)
    return jnp.array([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype)


def quaternion_multiply(q1: jax.Array, q2: jax.Array) -> jax.Array:
    q1 = jnp.asarray(q1)
    q2 = jnp.asarray(q2, dtype=q1.dtype)
    q1w, q1x, q1y, q1z = q1[0], q1[1], q1[2], q1[3]
    q2w, q2x, q2y, q2z = q2[0], q2[1], q2[2], q2[3]
    return jnp.array(
        [
            q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z,
            q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y,
            q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x,
            q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w,
        ],
        dtype=q1.dtype,
    )


def quaternion_from_two_vectors(v_from: jax.Array, v_to: jax.Array) -> jax.Array:
    """Shortest-rotation quaternion mapping unit vector v_from to v_to."""
    a = jnp.asarray(v_from)
    b = jnp.asarray(v_to, dtype=a.dtype)
    eps = jnp.asarray(1e-9, dtype=a.dtype)

    a_n = jnp.linalg.norm(a)
    b_n = jnp.linalg.norm(b)
    a_u = a / jnp.maximum(a_n, eps)
    b_u = b / jnp.maximum(b_n, eps)

    dot_ab = jnp.clip(jnp.dot(a_u, b_u), -1.0, 1.0)
    cross_ab = jnp.cross(a_u, b_u)
    q_general = jnp.array([1.0 + dot_ab, cross_ab[0], cross_ab[1], cross_ab[2]], dtype=a.dtype)

    helper = jnp.where(
        jnp.abs(a_u[0]) > 0.9,
        jnp.array([0.0, 1.0, 0.0], dtype=a.dtype),
        jnp.array([1.0, 0.0, 0.0], dtype=a.dtype),
    )
    axis_fb = jnp.cross(a_u, helper)
    q_fallback = jnp.array([0.0, axis_fb[0], axis_fb[1], axis_fb[2]], dtype=a.dtype)
    q_raw = jnp.where(dot_ab < (-1.0 + 1e-6), q_fallback, q_general)

    q_out = normalize_quaternion(q_raw)
    q_id = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=a.dtype)
    return jnp.where((a_n > eps) & (b_n > eps), q_out, q_id)


def desired_quaternion_from_virtual_accel(nu: jax.Array) -> jax.Array:
    nu = jnp.asarray(nu)
    nu_nrm = jnp.linalg.norm(nu)
    e_des = nu / jnp.maximum(nu_nrm, 1e-9)
    e3 = jnp.array([0.0, 0.0, 1.0], dtype=nu.dtype)
    q_des = quaternion_from_two_vectors(e3, e_des)
    q_id = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=nu.dtype)
    return jnp.where(nu_nrm > 1e-9, q_des, q_id)


def quaternion_rate_matrix(q: jax.Array) -> jax.Array:
    """Xi(q) in q_dot = 0.5 * Xi(q) * omega."""
    q = jnp.asarray(q)
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return jnp.array(
        [
            [-qx, -qy, -qz],
            [qw, -qz, qy],
            [qz, qw, -qx],
            [-qy, qx, qw],
        ],
        dtype=q.dtype,
    )


def rotation_matrix_from_quaternion(q: jax.Array) -> jax.Array:
    q = normalize_quaternion(q)
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return jnp.array(
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
        dtype=q.dtype,
    )


def thrust_axis_world(q: jax.Array) -> jax.Array:
    return rotation_matrix_from_quaternion(q)[:, 2]


def quaternion_from_euler_zyx(roll: jax.Array, pitch: jax.Array, yaw: jax.Array) -> jax.Array:
    cr = jnp.cos(0.5 * roll)
    sr = jnp.sin(0.5 * roll)
    cp = jnp.cos(0.5 * pitch)
    sp = jnp.sin(0.5 * pitch)
    cy = jnp.cos(0.5 * yaw)
    sy = jnp.sin(0.5 * yaw)
    q = jnp.array(
        [
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            sy * cp * sr + cy * sp * cr,
            sy * cp * cr - cy * sp * sr,
        ]
    )
    return normalize_quaternion(q)


# --------------------------------------------------------------------------- #
# Batched JAX family (shape (..., 4))                                          #
# --------------------------------------------------------------------------- #


def normalize_quaternion_batch(q_batch: jax.Array, eps: float = 1e-9) -> jax.Array:
    q = jnp.asarray(q_batch)
    sq_norm = jnp.sum(jnp.square(q), axis=-1, keepdims=True)
    nrm = jnp.sqrt(jnp.maximum(sq_norm, eps**2))
    q_norm = q / nrm
    ident = jnp.zeros_like(q_norm).at[..., 0].set(1.0)
    return jnp.where(sq_norm > eps**2, q_norm, ident)


def quaternion_conjugate_batch(q_batch: jax.Array) -> jax.Array:
    q = jnp.asarray(q_batch)
    return jnp.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def quaternion_multiply_batch(q1: jax.Array, q2: jax.Array) -> jax.Array:
    q1 = jnp.asarray(q1)
    q2 = jnp.asarray(q2)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return jnp.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quaternion_from_euler_zyx_batch(roll: jax.Array, pitch: jax.Array, yaw: jax.Array) -> jax.Array:
    cr = jnp.cos(0.5 * roll)
    sr = jnp.sin(0.5 * roll)
    cp = jnp.cos(0.5 * pitch)
    sp = jnp.sin(0.5 * pitch)
    cy = jnp.cos(0.5 * yaw)
    sy = jnp.sin(0.5 * yaw)
    q = jnp.stack(
        [
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ],
        axis=-1,
    )
    return normalize_quaternion_batch(q)


# --------------------------------------------------------------------------- #
# NumPy float64 family                                                        #
# --------------------------------------------------------------------------- #


def normalize_quaternion_np(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize a quaternion while preserving the incoming sign convention."""
    q_arr = np.asarray(q, dtype=np.float64)
    if q_arr.shape[-1] != 4:
        raise ValueError(f"Expected quaternion with last dimension 4, got shape {q_arr.shape}")
    if q_arr.ndim == 1:
        nrm = float(np.linalg.norm(q_arr))
        if nrm <= eps:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return q_arr / nrm
    nrm = np.linalg.norm(q_arr, axis=-1, keepdims=True)
    out = q_arr / np.maximum(nrm, eps)
    ident = np.zeros_like(out)
    ident[..., 0] = 1.0
    return np.where(nrm > eps, out, ident)


def quaternion_conjugate_np(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(q, dtype=np.float64)
    return np.array([qw, -qx, -qy, -qz], dtype=np.float64)


def quaternion_multiply_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1w, q1x, q1y, q1z = np.asarray(q1, dtype=np.float64)
    q2w, q2x, q2y, q2z = np.asarray(q2, dtype=np.float64)
    return np.array(
        [
            q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z,
            q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y,
            q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x,
            q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w,
        ],
        dtype=np.float64,
    )


def quaternion_rate_matrix_np(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(q, dtype=np.float64)
    return np.array(
        [
            [-qx, -qy, -qz],
            [qw, -qz, qy],
            [qz, qw, -qx],
            [-qy, qx, qw],
        ],
        dtype=np.float64,
    )


def rotation_matrix_from_quaternion_np(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = normalize_quaternion_np(q)
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


def thrust_axis_world_np(q: np.ndarray) -> np.ndarray:
    return rotation_matrix_from_quaternion_np(q)[:, 2]


def quaternion_to_euler_deg_batch_np(
    q_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll/pitch/yaw (degrees) from a batch ``(N, 4)`` of quaternions (host f64)."""
    q = normalize_quaternion_np(np.asarray(q_batch, dtype=np.float64))
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)


def quaternion_from_euler_zyx_np(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Quaternion from ZYX Euler angles, host-side float64."""
    cr = np.cos(0.5 * roll)
    sr = np.sin(0.5 * roll)
    cp = np.cos(0.5 * pitch)
    sp = np.sin(0.5 * pitch)
    cy = np.cos(0.5 * yaw)
    sy = np.sin(0.5 * yaw)
    q = np.array(
        [
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ],
        dtype=np.float64,
    )
    return normalize_quaternion_np(q)


__all__ = [
    # single-state jnp
    "normalize_quaternion",
    "quaternion_conjugate",
    "quaternion_multiply",
    "quaternion_from_two_vectors",
    "desired_quaternion_from_virtual_accel",
    "quaternion_rate_matrix",
    "rotation_matrix_from_quaternion",
    "thrust_axis_world",
    "quaternion_from_euler_zyx",
    # batched jnp
    "normalize_quaternion_batch",
    "quaternion_conjugate_batch",
    "quaternion_multiply_batch",
    "quaternion_from_euler_zyx_batch",
    # numpy f64
    "normalize_quaternion_np",
    "quaternion_conjugate_np",
    "quaternion_multiply_np",
    "quaternion_rate_matrix_np",
    "rotation_matrix_from_quaternion_np",
    "thrust_axis_world_np",
    "quaternion_to_euler_deg_batch_np",
    "quaternion_from_euler_zyx_np",
]
