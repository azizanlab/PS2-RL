"""System-agnostic core of the Phase-2 PS2 policy trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.cil.cil_policy import BCBFProjectionOps
from ps2rl.utils.policy import (
    ActorConfig,
    actor_mean_action,
    init_actor_params,
    sample_actor_action,
)
from ps2rl.utils.networks import CriticConfig, init_q_params, q_value
from ps2rl.utils.optim import adam_init, adam_step, soft_update
from ps2rl.utils.replay_buffer import (
    JaxReplayState,
    jax_replay_add_batch,
    jax_replay_init,
    jax_replay_sample,
)

PS2_REPLAY_FIELDS: Tuple[str, ...] = ("obs", "act", "rew", "next_obs", "done")


@dataclass(frozen=True)
class SACConfig:
    """Shared Phase-2 PS2 (SAC + control-invariant layer) training config."""

    seed: int = 0
    total_steps: int = 120_000
    start_steps: int = 4_000
    update_after: int = 2_000
    update_every: int = 8
    gradient_steps: int = 1
    batch_size: int = 64
    replay_size: int = 300_000

    gamma: float = 0.99
    tau: float = 0.005

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.2
    min_alpha: float = 1e-4
    target_entropy: float = -2.0
    max_grad_norm: float = 5.0
    q_clip_abs: float = 1e3

    hidden_size: int = 128

    use_projection: bool = True
    project_target_actions: bool = False
    project_actor_actions: bool = True
    eval_every: int = 5_000
    eval_episodes: int = 3
    log_every: int = 1_000
    best_weights_save_period: int = 100_000
    record_update_metrics: bool = False
    update_metric_every: int = 200
    num_envs: int = 32
    steps_per_jit: int = 128
    warm_start: bool = False
    warm_start_weights: str = ""


def ps2_replay_init(capacity: int, obs_dim: int, act_dim: int) -> JaxReplayState:
    return jax_replay_init(
        capacity,
        {
            "obs": (obs_dim,),
            "act": (act_dim,),
            "rew": (),
            "next_obs": (obs_dim,),
            "done": (),
        },
    )


def validate_action_scale(name: str, actual: np.ndarray | jax.Array, expected: np.ndarray, atol: float = 1e-6) -> None:
    actual_np = np.asarray(jax.device_get(actual), dtype=np.float64).reshape(-1)
    expected_np = np.asarray(expected, dtype=np.float64).reshape(-1)
    if actual_np.shape != expected_np.shape:
        raise ValueError(
            f"{name} has shape {actual_np.shape}, expected {expected_np.shape} for action bound consistency."
        )
    if not np.all(np.isfinite(actual_np)):
        raise ValueError(f"{name} contains non-finite values: {actual_np.tolist()}")
    if not np.allclose(actual_np, expected_np, atol=atol):
        raise ValueError(f"{name}={actual_np.tolist()} does not match expected={expected_np.tolist()} (atol={atol}).")


def validate_action_bounds(
    name: str,
    actual_low: np.ndarray | jax.Array,
    actual_high: np.ndarray | jax.Array,
    expected_low: np.ndarray,
    expected_high: np.ndarray,
    atol: float = 1e-6,
) -> None:
    actual_low_np = np.asarray(jax.device_get(actual_low), dtype=np.float64).reshape(-1)
    actual_high_np = np.asarray(jax.device_get(actual_high), dtype=np.float64).reshape(-1)
    expected_low_np = np.asarray(expected_low, dtype=np.float64).reshape(-1)
    expected_high_np = np.asarray(expected_high, dtype=np.float64).reshape(-1)

    if actual_low_np.shape != expected_low_np.shape or actual_high_np.shape != expected_high_np.shape:
        raise ValueError(
            f"{name} bounds have shapes low={actual_low_np.shape}, high={actual_high_np.shape}; "
            f"expected low={expected_low_np.shape}, high={expected_high_np.shape}."
        )
    if not np.all(np.isfinite(actual_low_np)) or not np.all(np.isfinite(actual_high_np)):
        raise ValueError(f"{name} contains non-finite bounds: low={actual_low_np.tolist()}, high={actual_high_np.tolist()}")
    if not np.allclose(actual_low_np, expected_low_np, atol=atol):
        raise ValueError(
            f"{name}.action_low={actual_low_np.tolist()} does not match expected={expected_low_np.tolist()} (atol={atol})."
        )
    if not np.allclose(actual_high_np, expected_high_np, atol=atol):
        raise ValueError(
            f"{name}.action_high={actual_high_np.tolist()} does not match expected={expected_high_np.tolist()} (atol={atol})."
        )


def to_jnp_batch(batch_np: Dict[str, np.ndarray]) -> Dict[str, jax.Array]:
    return {k: jnp.asarray(v) for k, v in batch_np.items()}


def init_sac_state(
    key: jax.Array,
    actor_cfg: ActorConfig,
    critic_cfg: CriticConfig,
    sac_cfg: Any,
) -> Dict[str, Any]:
    k_actor, k_q1, k_q2 = jax.random.split(key, 3)
    actor_params = init_actor_params(k_actor, actor_cfg)
    q1_params = init_q_params(k_q1, critic_cfg)
    q2_params = init_q_params(k_q2, critic_cfg)
    min_alpha = max(float(sac_cfg.min_alpha), 1e-8)
    init_alpha = max(float(sac_cfg.init_alpha), min_alpha)
    log_alpha = jnp.array(np.log(init_alpha), dtype=jnp.float32)

    return {
        "actor_params": actor_params,
        "q1_params": q1_params,
        "q2_params": q2_params,
        "target_q1_params": q1_params,
        "target_q2_params": q2_params,
        "log_alpha": log_alpha,
        "actor_opt": adam_init(actor_params),
        "q1_opt": adam_init(q1_params),
        "q2_opt": adam_init(q2_params),
        "alpha_opt": adam_init(log_alpha),
    }


def snapshot_sac_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return jax.device_get(
        {
            "actor_params": state["actor_params"],
            "q1_params": state["q1_params"],
            "q2_params": state["q2_params"],
            "target_q1_params": state["target_q1_params"],
            "target_q2_params": state["target_q2_params"],
            "log_alpha": state["log_alpha"],
        }
    )


@jax.jit
def split_env_keys(env_keys: jax.Array) -> Tuple[jax.Array, jax.Array]:
    split = jax.vmap(lambda k: jax.random.split(k, 2))(env_keys)
    return split[:, 0], split[:, 1]


class PS2LoopState(NamedTuple):
    state: Dict[str, Any]
    replay: JaxReplayState
    env_state: Any
    obs: jax.Array
    key: jax.Array
    env_keys: jax.Array
    global_step: jax.Array
    updates: jax.Array


def default_qp_finite_info(batch_n: int) -> Dict[str, jax.Array]:
    ones = jnp.ones((batch_n,), dtype=jnp.bool_)
    zeros = jnp.zeros((batch_n,), dtype=jnp.float32)
    qsat_false = jnp.zeros((batch_n,), dtype=jnp.bool_)
    return {
        "q_mat_finite": ones,
        "q_vec_finite": ones,
        "g_finite": ones,
        "h_finite": ones,
        "inputs_finite": ones,
        "z_finite": ones,
        "q_saturated": qsat_false,
        "max_abs_q": zeros,
        "max_abs_b": zeros,
        "delta_min_u_ref": zeros,
        "u_ref_minus_u_safe_norm": zeros,
        "a_ref_minus_a_safe": zeros,
        "r_ref_minus_r_safe": zeros,
    }


def make_projection_ops(
    cbf_cfg: Any,
    backup_runtime: Any,
    *,
    project_with_info_fn: Callable[..., Any],
    project_fn: Callable[..., Any],
    backup_policy_fn: Callable[..., Any],
) -> BCBFProjectionOps:
    """Assemble the trainer-facing projection ops from a system's backup-CBF QP callables."""
    return BCBFProjectionOps(
        project_with_info=lambda obs_phys, act: project_with_info_fn(obs_phys, act, cbf_cfg, runtime=backup_runtime),
        project=lambda obs_phys, act: project_fn(obs_phys, act, cbf_cfg, runtime=backup_runtime),
        backup_policy=lambda obs_phys: backup_policy_fn(obs_phys, cbf_cfg, runtime=backup_runtime),
    )


class PS2SystemBinding(NamedTuple):
    """Per-system knobs that specialize the shared PS2 build helpers."""

    phys_dim: int
    extended_qp_diagnostics: bool
    project_with_info_fn: Callable[..., Any]
    project_fn: Callable[..., Any]
    backup_policy_fn: Callable[..., Any]
    action_bounds_fn: Callable[[jax.Array, Any], Tuple[jax.Array, jax.Array]]
    disable_backup_fallback_fn: Callable[[Any], bool]

    def projection_ops(self, cbf_cfg: Any, backup_runtime: Any) -> BCBFProjectionOps:
        return make_projection_ops(
            cbf_cfg,
            backup_runtime,
            project_with_info_fn=self.project_with_info_fn,
            project_fn=self.project_fn,
            backup_policy_fn=self.backup_policy_fn,
        )


def build_ps2_update_fn(
    sac_cfg: Any,
    actor_cfg: ActorConfig,
    action_scale: jax.Array,
    action_low: jax.Array,
    action_high: jax.Array,
    proj: BCBFProjectionOps,
    *,
    phys_dim: int,
    disable_backup_fallback: bool = False,
    extended_qp_diagnostics: bool = False,
):
    """Create one JITed SAC update step."""

    def physical_obs(obs_b: jax.Array) -> jax.Array:
        # Safety modules (backup policy / CIL) only use the physical state.
        return obs_b[..., :phys_dim]

    log_alpha_min = float(np.log(max(float(sac_cfg.min_alpha), 1e-8)))
    q_clip_abs = float(sac_cfg.q_clip_abs)
    q_clip_enabled = q_clip_abs > 0.0
    q_cap = q_clip_abs if q_clip_enabled else 1e12

    def _q_nan_to_num(x: jax.Array) -> jax.Array:
        return jnp.nan_to_num(x, nan=0.0, posinf=q_cap, neginf=-q_cap)

    def _maybe_clip_q(x: jax.Array) -> jax.Array:
        if q_clip_enabled:
            return jnp.clip(x, -q_clip_abs, q_clip_abs)
        return x

    def tree_l2_norm(tree) -> jax.Array:
        leaves = jax.tree_util.tree_leaves(tree)
        sq = jnp.array(0.0, dtype=jnp.float32)
        for x in leaves:
            sq = sq + jnp.sum(jnp.square(x))
        return jnp.sqrt(sq + 1e-12)

    def sanitize_grads(grads):
        grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)
        grad_norm = tree_l2_norm(grads)
        if sac_cfg.max_grad_norm > 0.0:
            scale = jnp.minimum(1.0, sac_cfg.max_grad_norm / (grad_norm + 1e-6))
            grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
        return grads, grad_norm

    def maybe_project_target(
        obs_b: jax.Array, act_b: jax.Array
    ) -> Tuple[jax.Array, jax.Array, Dict[str, jax.Array]]:
        if sac_cfg.use_projection and sac_cfg.project_target_actions:
            return proj.project_with_info(physical_obs(obs_b), act_b)
        return (
            act_b,
            jnp.ones((act_b.shape[0],), dtype=jnp.bool_),
            default_qp_finite_info(act_b.shape[0]),
        )

    def maybe_project_actor(
        obs_b: jax.Array, act_b: jax.Array
    ) -> Tuple[jax.Array, jax.Array, Dict[str, jax.Array]]:
        if sac_cfg.use_projection and sac_cfg.project_actor_actions:
            return proj.project_with_info(physical_obs(obs_b), act_b)
        return (
            act_b,
            jnp.ones((act_b.shape[0],), dtype=jnp.bool_),
            default_qp_finite_info(act_b.shape[0]),
        )

    def _clipped_diag_mean(x: jax.Array) -> jax.Array:
        return jnp.mean(
            jnp.clip(
                jnp.nan_to_num(x, nan=0.0, posinf=1e6, neginf=0.0),
                0.0,
                1e6,
            )
        )

    @jax.jit
    def update(state: Dict[str, Any], batch: Dict[str, jax.Array], key: jax.Array):
        key_next, key_c, key_a = jax.random.split(key, 3)
        alpha = jnp.exp(jnp.clip(state["log_alpha"], log_alpha_min, 5.0))

        def backup_action_batch(obs_b: jax.Array):
            return proj.backup_policy(physical_obs(obs_b))

        def sanitize_action(action: jax.Array, obs_b: jax.Array):
            finite_mask = jnp.all(jnp.isfinite(action), axis=-1, keepdims=True)
            if disable_backup_fallback:
                cleaned = action
            else:
                backup_act = backup_action_batch(obs_b)
                backup_clip = jnp.clip(backup_act, action_low, action_high)
                cleaned = jnp.where(finite_mask, action, backup_clip)
            cleaned = jnp.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)
            cleaned = jnp.clip(cleaned, action_low, action_high)
            bad_rate = jnp.mean((~finite_mask.squeeze(-1)).astype(jnp.float32))
            return cleaned, bad_rate

        # ---------------------- Critic update ----------------------
        def critic_loss_fn(q1_params, q2_params):
            next_raw, next_logp, _ = sample_actor_action(
                state["actor_params"],
                batch["next_obs"],
                key_c,
                action_scale,
                actor_cfg,
                action_low=action_low,
                action_high=action_high,
            )
            next_raw = jnp.nan_to_num(next_raw, nan=0.0, posinf=0.0, neginf=0.0)
            next_raw = jnp.clip(next_raw, action_low, action_high)
            next_logp = jnp.nan_to_num(next_logp, nan=0.0, posinf=10.0, neginf=-10.0)
            next_logp = jnp.clip(next_logp, -20.0, 20.0)
            next_safe, target_use_solver, target_finite_info = maybe_project_target(batch["next_obs"], next_raw)
            next_safe, target_bad_rate = sanitize_action(next_safe, batch["next_obs"])
            target_use_solver_rate = jnp.mean(target_use_solver.astype(jnp.float32))
            target_fallback_rate = 1.0 - target_use_solver_rate
            target_q_mat_finite_rate = jnp.mean(target_finite_info["q_mat_finite"].astype(jnp.float32))
            target_q_vec_finite_rate = jnp.mean(target_finite_info["q_vec_finite"].astype(jnp.float32))
            target_g_finite_rate = jnp.mean(target_finite_info["g_finite"].astype(jnp.float32))
            target_h_finite_rate = jnp.mean(target_finite_info["h_finite"].astype(jnp.float32))
            target_z_finite_rate = jnp.mean(target_finite_info["z_finite"].astype(jnp.float32))
            aux = {
                "target_action_bad_rate": target_bad_rate,
                "target_use_solver_rate": target_use_solver_rate,
                "target_fallback_rate": target_fallback_rate,
                "target_q_mat_finite_rate": target_q_mat_finite_rate,
                "target_q_vec_finite_rate": target_q_vec_finite_rate,
                "target_g_finite_rate": target_g_finite_rate,
                "target_h_finite_rate": target_h_finite_rate,
                "target_z_finite_rate": target_z_finite_rate,
            }
            if extended_qp_diagnostics:
                aux["target_inputs_finite_rate"] = jnp.mean(target_finite_info["inputs_finite"].astype(jnp.float32))
                aux["target_q_saturated_rate"] = jnp.mean(target_finite_info["q_saturated"].astype(jnp.float32))
                aux["target_max_abs_q_mean"] = _clipped_diag_mean(target_finite_info["max_abs_q"])
                aux["target_max_abs_b_mean"] = _clipped_diag_mean(target_finite_info["max_abs_b"])
                aux["target_delta_min_u_ref_mean"] = _clipped_diag_mean(target_finite_info["delta_min_u_ref"])
            target_q1 = q_value(state["target_q1_params"], batch["next_obs"], next_safe)
            target_q2 = q_value(state["target_q2_params"], batch["next_obs"], next_safe)
            target_q = jnp.minimum(target_q1, target_q2) - alpha * next_logp
            target_q = _q_nan_to_num(target_q)
            target_q = _maybe_clip_q(target_q)
            backup = batch["rew"] + sac_cfg.gamma * (1.0 - batch["done"]) * target_q
            backup = _q_nan_to_num(backup)
            backup = _maybe_clip_q(backup)

            q1_pred = q_value(q1_params, batch["obs"], batch["act"])
            q2_pred = q_value(q2_params, batch["obs"], batch["act"])
            q1_pred = _q_nan_to_num(q1_pred)
            q2_pred = _q_nan_to_num(q2_pred)
            q1_pred = _maybe_clip_q(q1_pred)
            q2_pred = _maybe_clip_q(q2_pred)
            q1_loss = jnp.mean((q1_pred - backup) ** 2)
            q2_loss = jnp.mean((q2_pred - backup) ** 2)
            aux["q_backup_mean"] = jnp.mean(backup)
            return q1_loss + q2_loss, aux

        (critic_loss, critic_aux), (q1_grads, q2_grads) = jax.value_and_grad(
            critic_loss_fn, argnums=(0, 1), has_aux=True
        )(state["q1_params"], state["q2_params"])
        q1_grads, q1_grad_norm = sanitize_grads(q1_grads)
        q2_grads, q2_grad_norm = sanitize_grads(q2_grads)

        q1_params, q1_opt = adam_step(state["q1_params"], q1_grads, state["q1_opt"], sac_cfg.critic_lr)
        q2_params, q2_opt = adam_step(state["q2_params"], q2_grads, state["q2_opt"], sac_cfg.critic_lr)

        # ---------------------- Actor update -----------------------
        def actor_loss_fn(actor_params):
            raw_action, logp, _ = sample_actor_action(
                actor_params,
                batch["obs"],
                key_a,
                action_scale,
                actor_cfg,
                action_low=action_low,
                action_high=action_high,
            )
            raw_action = jnp.nan_to_num(raw_action, nan=0.0, posinf=0.0, neginf=0.0)
            raw_action = jnp.clip(raw_action, action_low, action_high)
            logp = jnp.nan_to_num(logp, nan=0.0, posinf=10.0, neginf=-10.0)
            logp = jnp.clip(logp, -20.0, 20.0)
            safe_action, actor_use_solver, actor_finite_info = maybe_project_actor(batch["obs"], raw_action)
            safe_action, actor_bad_rate = sanitize_action(safe_action, batch["obs"])
            actor_use_solver_rate = jnp.mean(actor_use_solver.astype(jnp.float32))
            actor_fallback_rate = 1.0 - actor_use_solver_rate
            actor_q_mat_finite_rate = jnp.mean(actor_finite_info["q_mat_finite"].astype(jnp.float32))
            actor_q_vec_finite_rate = jnp.mean(actor_finite_info["q_vec_finite"].astype(jnp.float32))
            actor_g_finite_rate = jnp.mean(actor_finite_info["g_finite"].astype(jnp.float32))
            actor_h_finite_rate = jnp.mean(actor_finite_info["h_finite"].astype(jnp.float32))
            actor_z_finite_rate = jnp.mean(actor_finite_info["z_finite"].astype(jnp.float32))
            q1_pi = q_value(q1_params, batch["obs"], safe_action)
            q2_pi = q_value(q2_params, batch["obs"], safe_action)
            q_pi = jnp.minimum(q1_pi, q2_pi)
            q_pi = _q_nan_to_num(q_pi)
            q_pi = _maybe_clip_q(q_pi)
            loss = jnp.mean(alpha * logp - q_pi)
            aux = {
                "logp": logp,
                "logp_mean": jnp.mean(logp),
                "q_pi_mean": jnp.mean(q_pi),
                "actor_action_bad_rate": actor_bad_rate,
                "actor_use_solver_rate": actor_use_solver_rate,
                "actor_fallback_rate": actor_fallback_rate,
                "actor_q_mat_finite_rate": actor_q_mat_finite_rate,
                "actor_q_vec_finite_rate": actor_q_vec_finite_rate,
                "actor_g_finite_rate": actor_g_finite_rate,
                "actor_h_finite_rate": actor_h_finite_rate,
                "actor_z_finite_rate": actor_z_finite_rate,
            }
            if extended_qp_diagnostics:
                aux["actor_inputs_finite_rate"] = jnp.mean(actor_finite_info["inputs_finite"].astype(jnp.float32))
                aux["actor_q_saturated_rate"] = jnp.mean(actor_finite_info["q_saturated"].astype(jnp.float32))
                aux["actor_max_abs_q_mean"] = _clipped_diag_mean(actor_finite_info["max_abs_q"])
                aux["actor_max_abs_b_mean"] = _clipped_diag_mean(actor_finite_info["max_abs_b"])
                aux["actor_delta_min_u_ref_mean"] = _clipped_diag_mean(actor_finite_info["delta_min_u_ref"])
            return loss, aux

        (actor_loss, actor_aux), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state["actor_params"])
        actor_grads, actor_grad_norm = sanitize_grads(actor_grads)
        actor_params, actor_opt = adam_step(state["actor_params"], actor_grads, state["actor_opt"], sac_cfg.actor_lr)

        # ---------------------- Alpha update -----------------------
        sampled_logp = actor_aux["logp"]

        def alpha_loss_fn(log_alpha):
            alpha_now = jnp.exp(log_alpha)
            return -jnp.mean(alpha_now * jax.lax.stop_gradient(sampled_logp + sac_cfg.target_entropy))

        alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(state["log_alpha"])
        alpha_grads = jnp.nan_to_num(alpha_grads, nan=0.0, posinf=0.0, neginf=0.0)
        log_alpha, alpha_opt = adam_step(state["log_alpha"], alpha_grads, state["alpha_opt"], sac_cfg.alpha_lr)
        log_alpha = jnp.clip(log_alpha, log_alpha_min, 5.0)

        target_q1_params = soft_update(state["target_q1_params"], q1_params, sac_cfg.tau)
        target_q2_params = soft_update(state["target_q2_params"], q2_params, sac_cfg.tau)

        new_state = {
            "actor_params": actor_params,
            "q1_params": q1_params,
            "q2_params": q2_params,
            "target_q1_params": target_q1_params,
            "target_q2_params": target_q2_params,
            "log_alpha": log_alpha,
            "actor_opt": actor_opt,
            "q1_opt": q1_opt,
            "q2_opt": q2_opt,
            "alpha_opt": alpha_opt,
        }
        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.exp(log_alpha),
            "q_backup_mean": critic_aux["q_backup_mean"],
            "logp_mean": actor_aux["logp_mean"],
            "q_pi_mean": actor_aux["q_pi_mean"],
            "q1_grad_norm": q1_grad_norm,
            "q2_grad_norm": q2_grad_norm,
            "actor_grad_norm": actor_grad_norm,
            "target_action_bad_rate": critic_aux["target_action_bad_rate"],
            "actor_action_bad_rate": actor_aux["actor_action_bad_rate"],
            "target_use_solver_rate": critic_aux["target_use_solver_rate"],
            "actor_use_solver_rate": actor_aux["actor_use_solver_rate"],
            "target_fallback_rate": critic_aux["target_fallback_rate"],
            "actor_fallback_rate": actor_aux["actor_fallback_rate"],
            "target_q_mat_finite_rate": critic_aux["target_q_mat_finite_rate"],
            "target_q_vec_finite_rate": critic_aux["target_q_vec_finite_rate"],
            "target_g_finite_rate": critic_aux["target_g_finite_rate"],
            "target_h_finite_rate": critic_aux["target_h_finite_rate"],
            "target_z_finite_rate": critic_aux["target_z_finite_rate"],
            "actor_q_mat_finite_rate": actor_aux["actor_q_mat_finite_rate"],
            "actor_q_vec_finite_rate": actor_aux["actor_q_vec_finite_rate"],
            "actor_g_finite_rate": actor_aux["actor_g_finite_rate"],
            "actor_h_finite_rate": actor_aux["actor_h_finite_rate"],
            "actor_z_finite_rate": actor_aux["actor_z_finite_rate"],
        }
        if extended_qp_diagnostics:
            metrics["target_inputs_finite_rate"] = critic_aux["target_inputs_finite_rate"]
            metrics["target_q_saturated_rate"] = critic_aux["target_q_saturated_rate"]
            metrics["target_max_abs_q_mean"] = critic_aux["target_max_abs_q_mean"]
            metrics["target_max_abs_b_mean"] = critic_aux["target_max_abs_b_mean"]
            metrics["target_delta_min_u_ref_mean"] = critic_aux["target_delta_min_u_ref_mean"]
            metrics["actor_inputs_finite_rate"] = actor_aux["actor_inputs_finite_rate"]
            metrics["actor_q_saturated_rate"] = actor_aux["actor_q_saturated_rate"]
            metrics["actor_max_abs_q_mean"] = actor_aux["actor_max_abs_q_mean"]
            metrics["actor_max_abs_b_mean"] = actor_aux["actor_max_abs_b_mean"]
            metrics["actor_delta_min_u_ref_mean"] = actor_aux["actor_delta_min_u_ref_mean"]
        return new_state, metrics, key_next

    return update


def build_ps2_action_fns(
    sac_cfg: Any,
    actor_cfg: ActorConfig,
    action_scale: jax.Array,
    action_low: jax.Array,
    action_high: jax.Array,
    proj: BCBFProjectionOps,
    *,
    phys_dim: int,
    disable_backup_fallback: bool = False,
    return_solver_info: bool = False,
):
    """Build jitted training/eval policy calls."""

    def physical_obs(obs_b: jax.Array) -> jax.Array:
        return obs_b[..., :phys_dim]

    def maybe_project(obs_b: jax.Array, act_b: jax.Array):
        if sac_cfg.use_projection and sac_cfg.project_actor_actions:
            if return_solver_info:
                return proj.project_with_info(physical_obs(obs_b), act_b)
            return proj.project(physical_obs(obs_b), act_b)
        if return_solver_info:
            return (
                act_b,
                jnp.ones((act_b.shape[0],), dtype=jnp.bool_),
                default_qp_finite_info(act_b.shape[0]),
            )
        return act_b

    def backup_action(obs_b: jax.Array):
        return proj.backup_policy(physical_obs(obs_b))

    @jax.jit
    def sample_action(params, obs: jax.Array, key: jax.Array):
        raw, logp, _ = sample_actor_action(
            params,
            obs[None, :],
            key,
            action_scale,
            actor_cfg,
            action_low=action_low,
            action_high=action_high,
        )
        if return_solver_info:
            safe, use_solver, finite_info = maybe_project(obs[None, :], raw)
        else:
            safe = maybe_project(obs[None, :], raw)
        finite_mask = jnp.all(jnp.isfinite(safe), axis=-1, keepdims=True)
        if not disable_backup_fallback:
            safe = jnp.where(finite_mask, safe, backup_action(obs[None, :]))
        safe = jnp.nan_to_num(safe, nan=0.0, posinf=0.0, neginf=0.0)
        safe = jnp.clip(safe, action_low, action_high)
        logp = jnp.nan_to_num(logp, nan=0.0, posinf=10.0, neginf=-10.0)
        if return_solver_info:
            finite_info_single = {k: v[0] for k, v in finite_info.items()}
            return safe[0], raw[0], logp[0], use_solver[0], finite_info_single
        return safe[0], raw[0], logp[0]

    @jax.jit
    def eval_action(params, obs: jax.Array):
        raw = actor_mean_action(
            params,
            obs[None, :],
            action_scale,
            actor_cfg,
            action_low=action_low,
            action_high=action_high,
        )
        if return_solver_info:
            safe, use_solver, finite_info = maybe_project(obs[None, :], raw)
        else:
            safe = maybe_project(obs[None, :], raw)
        finite_mask = jnp.all(jnp.isfinite(safe), axis=-1, keepdims=True)
        if not disable_backup_fallback:
            safe = jnp.where(finite_mask, safe, backup_action(obs[None, :]))
        safe = jnp.nan_to_num(safe, nan=0.0, posinf=0.0, neginf=0.0)
        safe = jnp.clip(safe, action_low, action_high)
        if return_solver_info:
            finite_info_single = {k: v[0] for k, v in finite_info.items()}
            return safe[0], raw[0], use_solver[0], finite_info_single
        return safe[0], raw[0]

    return sample_action, eval_action


def build_ps2_batched_action_fn(
    sac_cfg: Any,
    actor_cfg: ActorConfig,
    action_scale: jax.Array,
    action_low: jax.Array,
    action_high: jax.Array,
    proj: BCBFProjectionOps,
    *,
    phys_dim: int,
    disable_backup_fallback: bool = False,
):
    @jax.jit
    def sample_action_batch(actor_params: Dict[str, Any], obs_b: jax.Array, key: jax.Array):
        raw, logp, _ = sample_actor_action(
            actor_params,
            obs_b,
            key,
            action_scale,
            actor_cfg,
            action_low=action_low,
            action_high=action_high,
        )
        raw = jnp.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw = jnp.clip(raw, action_low, action_high)
        logp = jnp.nan_to_num(logp, nan=0.0, posinf=10.0, neginf=-10.0)
        logp = jnp.clip(logp, -20.0, 20.0)

        if sac_cfg.use_projection and sac_cfg.project_actor_actions:
            safe, use_solver, _ = proj.project_with_info(obs_b[..., :phys_dim], raw)
        else:
            safe = raw
            use_solver = jnp.ones((raw.shape[0],), dtype=jnp.bool_)

        finite_mask = jnp.all(jnp.isfinite(safe), axis=-1, keepdims=True)
        if not disable_backup_fallback:
            fallback = proj.backup_policy(obs_b[..., :phys_dim])
            safe = jnp.where(finite_mask, safe, fallback)
        safe = jnp.nan_to_num(safe, nan=0.0, posinf=0.0, neginf=0.0)
        safe = jnp.clip(safe, action_low, action_high)
        return safe, raw, logp, use_solver

    return sample_action_batch


def zero_ps2_chunk_metrics(
    update_sum_names: Sequence[str],
    episode_field_names: Sequence[str],
) -> Dict[str, jax.Array]:
    metrics: Dict[str, jax.Array] = {
        "global_step": jnp.int32(0),
        "updates": jnp.int32(0),
        "update_count": jnp.asarray(0.0, dtype=jnp.float32),
        "episode_count": jnp.asarray(0.0, dtype=jnp.float32),
    }
    for name in update_sum_names:
        metrics[name] = jnp.asarray(0.0, dtype=jnp.float32)
    for name in episode_field_names:
        metrics[name] = jnp.asarray(0.0, dtype=jnp.float32)
    return metrics


def build_ps2_update_fn_for(
    binding: PS2SystemBinding,
    sac_cfg: Any,
    actor_cfg: ActorConfig,
    cbf_cfg: Any,
    action_scale: jax.Array,
    backup_runtime: Any = None,
):
    """Build the JITed SAC update step for a system via its binding."""
    action_low, action_high = binding.action_bounds_fn(action_scale, cbf_cfg)
    return build_ps2_update_fn(
        sac_cfg,
        actor_cfg,
        action_scale,
        action_low,
        action_high,
        binding.projection_ops(cbf_cfg, backup_runtime),
        phys_dim=binding.phys_dim,
        disable_backup_fallback=binding.disable_backup_fallback_fn(sac_cfg),
        extended_qp_diagnostics=binding.extended_qp_diagnostics,
    )


def build_ps2_action_fns_for(
    binding: PS2SystemBinding,
    sac_cfg: Any,
    actor_cfg: ActorConfig,
    cbf_cfg: Any,
    action_scale: jax.Array,
    backup_runtime: Any = None,
    return_solver_info: bool = False,
):
    """Build the jitted training/eval policy calls for a system via its binding."""
    action_low, action_high = binding.action_bounds_fn(action_scale, cbf_cfg)
    return build_ps2_action_fns(
        sac_cfg,
        actor_cfg,
        action_scale,
        action_low,
        action_high,
        binding.projection_ops(cbf_cfg, backup_runtime),
        phys_dim=binding.phys_dim,
        disable_backup_fallback=binding.disable_backup_fallback_fn(sac_cfg),
        return_solver_info=return_solver_info,
    )


def build_ps2_batched_action_fn_for(
    binding: PS2SystemBinding,
    sac_cfg: Any,
    actor_cfg: ActorConfig,
    cbf_cfg: Any,
    action_scale: jax.Array,
    backup_runtime: Any = None,
):
    """Build the batched collection policy for a system via its binding."""
    action_low, action_high = binding.action_bounds_fn(action_scale, cbf_cfg)
    sample_action_batch = build_ps2_batched_action_fn(
        sac_cfg,
        actor_cfg,
        action_scale,
        action_low,
        action_high,
        binding.projection_ops(cbf_cfg, backup_runtime),
        phys_dim=binding.phys_dim,
        disable_backup_fallback=binding.disable_backup_fallback_fn(sac_cfg),
    )
    return sample_action_batch, action_low, action_high


def build_ps2_one_vec_step(
    *,
    env_fns: Any,
    sample_action_batch_fn,
    update_fn,
    sac_cfg: Any,
    action_low: jax.Array,
    action_high: jax.Array,
    proj: BCBFProjectionOps,
    phys_dim: int,
    sanitize_random_actions: bool,
    update_metric_pairs: Sequence[Tuple[str, str]],
    episode_fields_fn: Callable[[Any], Dict[str, jax.Array]],
    episode_field_names: Sequence[str],
):
    num_envs = int(sac_cfg.num_envs)
    batch_size_i32 = jnp.int32(sac_cfg.batch_size)
    update_after_i32 = jnp.int32(sac_cfg.update_after)
    update_every_i32 = jnp.int32(max(1, sac_cfg.update_every))
    num_envs_i32 = jnp.int32(num_envs)
    max_due_per_vec_step = max(1, int(np.ceil(float(num_envs) / float(max(1, sac_cfg.update_every)))))
    update_sum_names = tuple(sum_name for sum_name, _ in update_metric_pairs)

    def one_vec_step(carry: PS2LoopState, _):
        state_i = carry.state
        replay_i = carry.replay
        env_state_i = carry.env_state
        obs_i = carry.obs
        key_i = carry.key
        env_keys_i = carry.env_keys
        global_step_i = carry.global_step
        updates_i = carry.updates

        key_i, key_action, key_random = jax.random.split(key_i, 3)
        rand01 = jax.random.uniform(key_random, shape=(num_envs, env_fns.action_dim), dtype=jnp.float32)
        random_raw = action_low + rand01 * (action_high - action_low)
        if sac_cfg.use_projection:
            random_safe, _, _ = proj.project_with_info(obs_i[..., :phys_dim], random_raw)
        else:
            random_safe = random_raw
        if sanitize_random_actions:
            random_finite = jnp.all(jnp.isfinite(random_safe), axis=-1, keepdims=True)
            random_fallback = proj.backup_policy(obs_i[..., :phys_dim])
            random_safe = jnp.where(random_finite, random_safe, random_fallback)
            random_safe = jnp.nan_to_num(random_safe, nan=0.0, posinf=0.0, neginf=0.0)
            random_safe = jnp.clip(random_safe, action_low, action_high)

        policy_safe, _, _, _ = sample_action_batch_fn(state_i["actor_params"], obs_i, key_action)
        use_random = global_step_i < jnp.int32(sac_cfg.start_steps)
        act_i = jnp.where(use_random, random_safe, policy_safe)

        env_keys_i, step_keys = split_env_keys(env_keys_i)
        env_state_i, next_obs_true_i, next_obs_out_i, rew_i, done_i, info_i = env_fns.step_batched(
            env_state_i,
            act_i,
            step_keys,
        )
        replay_next_obs_i = next_obs_true_i
        replay_i = jax_replay_add_batch(
            replay_i,
            {
                "obs": obs_i.astype(jnp.float32),
                "act": act_i.astype(jnp.float32),
                "rew": rew_i.astype(jnp.float32),
                "next_obs": replay_next_obs_i.astype(jnp.float32),
                "done": done_i.astype(jnp.float32),
            },
        )
        obs_i = next_obs_out_i

        prev_step = global_step_i
        global_step_i = global_step_i + num_envs_i32

        replay_ready = replay_i.size >= batch_size_i32
        lo = jnp.maximum(prev_step + jnp.int32(1), update_after_i32)
        hi = global_step_i
        due_raw = jnp.where(
            hi >= lo,
            hi // update_every_i32 - (lo - jnp.int32(1)) // update_every_i32,
            jnp.int32(0),
        )
        due = jnp.where(replay_ready, due_raw, jnp.int32(0))
        due = jnp.minimum(due, jnp.int32(max_due_per_vec_step))

        step_metrics = zero_ps2_chunk_metrics(update_sum_names, episode_field_names)
        step_metrics["episode_count"] = jnp.sum(info_i.episode_done.astype(jnp.float32))
        episode_fields = episode_fields_fn(info_i)
        for field_name in episode_field_names:
            step_metrics[field_name] = episode_fields[field_name]

        def due_body(i, upd_carry):
            state_u, replay_u, key_u, updates_u, step_metrics_u = upd_carry

            def do_updates(inner):
                state_d, replay_d, key_d, updates_d, step_metrics_d = inner

                def grad_body(_, grad_inner):
                    state_g, replay_g, key_g, updates_g, step_metrics_g = grad_inner
                    key_g, key_sample, key_upd = jax.random.split(key_g, 3)
                    batch = jax_replay_sample(replay_g, sac_cfg.batch_size, key_sample, PS2_REPLAY_FIELDS)
                    state_g, upd_metrics, _ = update_fn(state_g, batch, key_upd)
                    updates_g = updates_g + jnp.int32(1)
                    step_metrics_g = dict(step_metrics_g)
                    step_metrics_g["update_count"] = step_metrics_g["update_count"] + 1.0
                    for sum_name, metric_key in update_metric_pairs:
                        step_metrics_g[sum_name] = step_metrics_g[sum_name] + upd_metrics[metric_key]
                    return state_g, replay_g, key_g, updates_g, step_metrics_g

                return jax.lax.fori_loop(
                    0,
                    sac_cfg.gradient_steps,
                    grad_body,
                    (state_d, replay_d, key_d, updates_d, step_metrics_d),
                )

            return jax.lax.cond(i < due, do_updates, lambda x: x, (state_u, replay_u, key_u, updates_u, step_metrics_u))

        state_i, replay_i, key_i, updates_i, step_metrics = jax.lax.fori_loop(
            0,
            max_due_per_vec_step,
            due_body,
            (state_i, replay_i, key_i, updates_i, step_metrics),
        )

        new_carry = PS2LoopState(
            state=state_i,
            replay=replay_i,
            env_state=env_state_i,
            obs=obs_i,
            key=key_i,
            env_keys=env_keys_i,
            global_step=global_step_i,
            updates=updates_i,
        )
        return new_carry, step_metrics

    return one_vec_step


def make_ps2_chunk_fn_getter(one_vec_step):
    chunk_fn_cache: Dict[int, Any] = {}

    def get_chunk_fn(length: int):
        if length not in chunk_fn_cache:

            @jax.jit
            def _chunk_fn(loop_s: PS2LoopState):
                loop_s, seq_metrics = jax.lax.scan(one_vec_step, loop_s, xs=None, length=length)
                chunk = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), seq_metrics)
                chunk["global_step"] = loop_s.global_step
                chunk["updates"] = loop_s.updates
                return loop_s, chunk

            chunk_fn_cache[length] = _chunk_fn
        return chunk_fn_cache[length]

    return get_chunk_fn


__all__ = [
    "PS2LoopState",
    "PS2_REPLAY_FIELDS",
    "build_ps2_action_fns",
    "build_ps2_batched_action_fn",
    "build_ps2_one_vec_step",
    "build_ps2_update_fn",
    "default_qp_finite_info",
    "init_sac_state",
    "make_ps2_chunk_fn_getter",
    "ps2_replay_init",
    "snapshot_sac_state",
    "split_env_keys",
    "to_jnp_batch",
    "validate_action_bounds",
    "validate_action_scale",
    "zero_ps2_chunk_metrics",
]
