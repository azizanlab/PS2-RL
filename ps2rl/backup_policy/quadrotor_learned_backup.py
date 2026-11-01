"""Quadrotor learned backup policy: checkpoint wrapper + LBP composition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.backup_policy.backup_policy import LBP
from ps2rl.backup_policy.quadrotor_analytic_backup import clip_action
from ps2rl.utils.paths import load_json_payload, load_pickle_payload, resolve_existing_path
from ps2rl.utils.policy import ActorConfig, _coerce_actor_cfg, actor_mean_action


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _lqr_metadata_from_cbf_cfg_payload(cbf_cfg_payload: dict[str, Any]) -> dict[str, float]:
    keys = (
        "dt",
        "gravity",
        "a_cmd_min",
        "a_cmd_max",
        "omega_max",
        "z_des",
        "lqr_q_z",
        "lqr_q_vx",
        "lqr_q_vy",
        "lqr_q_vz",
        "lqr_q_thetax",
        "lqr_q_thetay",
        "lqr_q_thetaz",
        "lqr_r_a_cmd",
        "lqr_r_omega_x",
        "lqr_r_omega_y",
        "lqr_r_omega_z",
    )
    missing = [key for key in keys if key not in cbf_cfg_payload]
    if missing:
        raise KeyError(f"Missing required hover-LQR fields {missing} in synthesized quadrotor backup metadata.")
    out = {key: float(cbf_cfg_payload[key]) for key in keys}
    return out


@dataclass(frozen=True)
class LearnedQuadrotorBackupPolicy:
    """Deterministic saved-policy wrapper for the phase-1 quadrotor backup actor."""

    actor_params: Any
    actor_cfg: ActorConfig
    action_scale: jax.Array
    action_low: jax.Array
    action_high: jax.Array
    metadata: dict[str, Any] = field(default_factory=dict)

    def action_single(self, x: jax.Array) -> jax.Array:
        act = actor_mean_action(
            self.actor_params,
            jnp.asarray(x)[None, :],
            self.action_scale,
            self.actor_cfg,
            action_low=self.action_low,
            action_high=self.action_high,
        )[0]
        return jnp.clip(act, self.action_low, self.action_high)

    def action_batch(self, x_batch: jax.Array) -> jax.Array:
        act = actor_mean_action(
            self.actor_params,
            jnp.asarray(x_batch),
            self.action_scale,
            self.actor_cfg,
            action_low=self.action_low,
            action_high=self.action_high,
        )
        return jnp.clip(act, self.action_low, self.action_high)


def _policy_payload_from_training_checkpoint(
    weights_path: Path,
    *,
    config_path: Path,
) -> dict[str, Any]:
    weights_payload = load_pickle_payload(weights_path)
    if "actor_params" not in weights_payload:
        raise KeyError(f"Missing 'actor_params' in learned quadrotor checkpoint: {weights_path}")

    cfg_payload = load_json_payload(config_path)
    actor_cfg_payload = cfg_payload.get("actor")
    if not isinstance(actor_cfg_payload, dict):
        raise KeyError(f"Missing/invalid 'actor' section in {config_path}")

    backup_env_payload = cfg_payload.get("backup_env")
    if not isinstance(backup_env_payload, dict):
        raise KeyError(f"Missing/invalid 'backup_env' section in {config_path}")
    cbf_cfg_payload = backup_env_payload.get("cbf_cfg")
    if not isinstance(cbf_cfg_payload, dict):
        raise KeyError(f"Missing/invalid 'backup_env.cbf_cfg' section in {config_path}")

    required_cbf = ("a_cmd_min", "a_cmd_max", "omega_max")
    missing_cbf = [key for key in required_cbf if key not in cbf_cfg_payload]
    if missing_cbf:
        raise KeyError(f"Missing required CBF fields {missing_cbf} in {config_path}")

    a_cmd_min = float(cbf_cfg_payload["a_cmd_min"])
    a_cmd_max = float(cbf_cfg_payload["a_cmd_max"])
    omega_max = float(cbf_cfg_payload["omega_max"])
    metadata = dict(weights_payload.get("metadata", {}))
    # The base set B is always the hover-LQR ellipsoid at cbf_cfg.base_set_c; the
    # quad backup loader consumes only metadata['lqr_config'].
    if "lqr_config" not in metadata:
        metadata["lqr_config"] = _lqr_metadata_from_cbf_cfg_payload(cbf_cfg_payload)
    metadata.setdefault("source_weights_path", str(weights_path))
    metadata.setdefault("source_config_path", str(config_path))

    return {
        "actor_params": weights_payload["actor_params"],
        "actor_cfg": actor_cfg_payload,
        "action_scale": np.asarray([a_cmd_max, omega_max, omega_max, omega_max], dtype=np.float32),
        "action_low": np.asarray([a_cmd_min, -omega_max, -omega_max, -omega_max], dtype=np.float32),
        "action_high": np.asarray([a_cmd_max, omega_max, omega_max, omega_max], dtype=np.float32),
        "metadata": metadata,
    }


def _resolve_policy_payload(path: str | Path) -> tuple[Path, dict[str, Any]]:
    artifact_path = resolve_existing_path(path, bases=(Path.cwd(), PROJECT_ROOT, PROJECT_ROOT.parent))

    if artifact_path.is_dir():
        actor_ckpt = artifact_path / "quad_backup_policy_actor.pkl"
        if actor_ckpt.exists():
            return actor_ckpt, load_pickle_payload(actor_ckpt)

        config_path = artifact_path / "configs.json"
        for weights_name in ("best_weights.pkl", "final_weights.pkl"):
            weights_path = artifact_path / weights_name
            if weights_path.exists() and config_path.exists():
                return weights_path, _policy_payload_from_training_checkpoint(weights_path, config_path=config_path)

        raise FileNotFoundError(
            "Expected one of:\n"
            f"  - {actor_ckpt}\n"
            f"  - {artifact_path / 'best_weights.pkl'} with {config_path}\n"
            f"  - {artifact_path / 'final_weights.pkl'} with {config_path}"
        )

    payload = load_pickle_payload(artifact_path)
    required = {"actor_params", "actor_cfg", "action_scale", "action_low", "action_high"}
    if required.issubset(payload.keys()):
        return artifact_path, payload

    config_path = artifact_path.with_name("configs.json")
    if "actor_params" in payload and config_path.exists():
        return artifact_path, _policy_payload_from_training_checkpoint(artifact_path, config_path=config_path)

    missing = sorted(required.difference(payload.keys()))
    raise KeyError(
        f"Missing fields in learned quadrotor backup policy file {artifact_path}: {missing}. "
        "If this is a raw SAC checkpoint, keep configs.json next to it or pass the run directory instead."
    )


def load_learned_quadrotor_backup_policy(path: str | Path) -> LearnedQuadrotorBackupPolicy:
    ckpt_path, payload = _resolve_policy_payload(path)

    actor_cfg = _coerce_actor_cfg(payload["actor_cfg"])
    actor_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), payload["actor_params"])
    return LearnedQuadrotorBackupPolicy(
        actor_params=actor_params,
        actor_cfg=actor_cfg,
        action_scale=jnp.asarray(payload["action_scale"], dtype=jnp.float32),
        action_low=jnp.asarray(payload["action_low"], dtype=jnp.float32),
        action_high=jnp.asarray(payload["action_high"], dtype=jnp.float32),
        metadata=dict(payload.get("metadata", {})),
    )


@dataclass(frozen=True)
class QuadrotorLBP(LBP):
    """Learned quadrotor backup: Phase-1 safe-arrival actor + LQR-on-B handoff.

    ``base_controller``/``base_set`` carry the hover-LQR design from the saved
    checkpoint payload (validated against the runtime CBF config by the CBF
    system module before construction). ``a_cmd_min``/``a_cmd_max``/
    ``omega_max`` are the runtime action box.
    """

    learned: LearnedQuadrotorBackupPolicy = None  # type: ignore[assignment]
    a_cmd_min: float = 0.0
    a_cmd_max: float = 0.0
    omega_max: float = 0.0

    def sa_action(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        action_low = jnp.asarray(
            [self.a_cmd_min, -self.omega_max, -self.omega_max, -self.omega_max],
            dtype=jnp.float32,
        )
        action_high = jnp.asarray(
            [self.a_cmd_max, self.omega_max, self.omega_max, self.omega_max],
            dtype=jnp.float32,
        )
        raw = self.learned.action_single(jnp.asarray(x, dtype=jnp.float32))
        raw = jnp.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw = jnp.clip(raw, action_low, action_high)
        return jnp.asarray(raw, dtype=x.dtype)

    def action(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        raw = self.sa_action(x)
        u_hybrid = self.select_action(x, raw, self.base_set)
        return clip_action(u_hybrid, self)


__all__ = [
    "LearnedQuadrotorBackupPolicy",
    "QuadrotorLBP",
    "load_learned_quadrotor_backup_policy",
]
