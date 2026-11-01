"""PS2 policy trainer with CIL projection.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import pickle
import time
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.backup_policy.unicycle_learned_backup import load_learned_unicycle_backup_policy
from ps2rl.cil.unicycle_backup_cbf import (
    UnicycleBCBFConfig,
    UnicycleBackupCBFProjector,
    BCBFSystem,
    backup_policy_batch,
    solve_backup_cbf_qp_batch,
    solve_backup_cbf_qp_batch_with_info,
)
from ps2rl.envs.unicycle_env import UnicycleEnvConfig, build_unicycle_env
from ps2rl.utils.policy import ActorConfig
from ps2rl.utils.networks import CriticConfig
from ps2rl.cil.cil_policy import BCBFProjectionOps
from ps2rl.phase2_ps2.ps2_trainer_core import (
    PS2LoopState,
    PS2SystemBinding,
    SACConfig,
    build_ps2_action_fns_for,
    build_ps2_batched_action_fn,
    build_ps2_one_vec_step,
    build_ps2_update_fn_for,
    init_sac_state,
    make_ps2_chunk_fn_getter,
    ps2_replay_init,
    snapshot_sac_state as _snapshot_state,
    validate_action_scale as _validate_action_scale,
)
from ps2rl.utils.seed import make_prng_key

_PHYS_DIM = 3


def _action_bounds(action_scale: jax.Array, cbf_cfg: UnicycleBCBFConfig) -> Tuple[jax.Array, jax.Array]:
    """Unicycle actuator box: the symmetric [-action_scale, action_scale]."""
    return -action_scale, action_scale


_BINDING = PS2SystemBinding(
    phys_dim=_PHYS_DIM,
    extended_qp_diagnostics=False,
    project_with_info_fn=solve_backup_cbf_qp_batch_with_info,
    project_fn=solve_backup_cbf_qp_batch,
    backup_policy_fn=backup_policy_batch,
    action_bounds_fn=_action_bounds,
    disable_backup_fallback_fn=lambda sac_cfg: False,
)


def _make_projection_ops(cbf_cfg: UnicycleBCBFConfig, backup_runtime: BCBFSystem | None) -> BCBFProjectionOps:
    return _BINDING.projection_ops(cbf_cfg, backup_runtime)


def _build_update_fn(
    sac_cfg: SACConfig,
    actor_cfg: ActorConfig,
    cbf_cfg: UnicycleBCBFConfig,
    action_scale: jax.Array,
    backup_runtime: BCBFSystem | None = None,
):
    """Create one JITed SAC update step (unicycle: symmetric action box)."""
    return build_ps2_update_fn_for(_BINDING, sac_cfg, actor_cfg, cbf_cfg, action_scale, backup_runtime)


def _build_action_fns(
    sac_cfg: SACConfig,
    actor_cfg: ActorConfig,
    cbf_cfg: UnicycleBCBFConfig,
    action_scale: jax.Array,
    backup_runtime: BCBFSystem | None = None,
    return_solver_info: bool = False,
):
    """Build jitted training/eval policy calls."""
    return build_ps2_action_fns_for(
        _BINDING, sac_cfg, actor_cfg, cbf_cfg, action_scale, backup_runtime, return_solver_info=return_solver_info
    )


def _evaluate_policy(
    env_cfg: UnicycleEnvConfig,
    eval_action_fn,
    actor_params,
    seed: int,
    episodes: int,
) -> Dict[str, Any]:
    """Roll out eval episodes on the JAX env and aggregate tracking metrics.

    Drives best-checkpoint selection during training (via return_mean).
    """
    env_fns = build_unicycle_env(env_cfg)
    base_key = make_prng_key(seed + 777)
    returns = []
    speeds = []
    y_errors = []
    psi_errors = []
    safes = []
    trajectory = {
        "obs": [],
        "next_obs": [],
        "act": [],
        "rew": [],
        "safe": [],
        "speed_error_abs": [],
        "y_error_abs": [],
        "psi_error_abs": [],
        "y_ref": [],
        "v_ref": [],
        "psi_ref": [],
        "ref_time_sec": [],
    }
    for ep in range(episodes):
        key_ep = jax.random.fold_in(base_key, 10_000 + ep)
        state, obs = env_fns.reset(key_ep)
        done = False
        ep_ret = 0.0
        step_idx = 0
        while not done:
            obs_j = jnp.asarray(obs, dtype=jnp.float32)
            safe_action, raw_action = eval_action_fn(actor_params, obs_j)
            _ = raw_action
            # The auto-reset key is unused: the loop breaks on done, and
            # next_obs_true (pre-reset) is what the metrics consume.
            key_step = jax.random.fold_in(key_ep, step_idx)
            state, next_obs_true, _, rew, done_j, info = env_fns.step(state, safe_action, key_step)
            done = bool(done_j)
            step_idx += 1
            act_np = np.asarray(safe_action, dtype=np.float64)
            next_obs = np.asarray(next_obs_true, dtype=np.float64)
            rew = float(rew)
            ep_ret += rew
            speed_error_abs = float(info.speed_error_abs)
            y_error_abs = float(info.y_error_abs)
            psi_error_abs = float(info.psi_error_abs)
            is_safe = float(info.is_safe)
            speeds.append(speed_error_abs)
            y_errors.append(y_error_abs)
            psi_errors.append(psi_error_abs)
            safes.append(is_safe)

            if ep == 0:
                trajectory["obs"].append(np.asarray(obs, dtype=np.float64))
                trajectory["next_obs"].append(next_obs)
                trajectory["act"].append(act_np)
                trajectory["rew"].append(rew)
                trajectory["safe"].append(is_safe)
                trajectory["speed_error_abs"].append(speed_error_abs)
                trajectory["y_error_abs"].append(y_error_abs)
                trajectory["psi_error_abs"].append(psi_error_abs)
                trajectory["y_ref"].append(float(info.y_ref))
                trajectory["v_ref"].append(float(info.v_ref))
                trajectory["psi_ref"].append(float(info.psi_ref))
                trajectory["ref_time_sec"].append(float(info.ref_time_sec))
            obs = next_obs_true
        returns.append(ep_ret)
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "speed_error_abs_mean": float(np.mean(speeds)) if speeds else 0.0,
        "y_error_abs_mean": float(np.mean(y_errors)) if y_errors else 0.0,
        "psi_error_abs_mean": float(np.mean(psi_errors)) if psi_errors else 0.0,
        "safe_rate": float(np.mean(safes)) if safes else 1.0,
        "trajectory": {k: np.asarray(v) for k, v in trajectory.items()},
    }


_UPDATE_HISTORY_KEYS: Tuple[str, ...] = (
    "critic_loss",
    "actor_loss",
    "alpha",
    "q1_grad_norm",
    "q2_grad_norm",
    "actor_grad_norm",
    "target_action_bad_rate",
    "actor_action_bad_rate",
    "target_use_solver_rate",
    "actor_use_solver_rate",
    "target_fallback_rate",
    "actor_fallback_rate",
    "target_q_mat_finite_rate",
    "target_q_vec_finite_rate",
    "target_g_finite_rate",
    "target_h_finite_rate",
    "target_z_finite_rate",
    "actor_q_mat_finite_rate",
    "actor_q_vec_finite_rate",
    "actor_g_finite_rate",
    "actor_h_finite_rate",
    "actor_z_finite_rate",
)

_UPDATE_METRIC_PAIRS: Tuple[Tuple[str, str], ...] = tuple((f"{key}_sum", key) for key in _UPDATE_HISTORY_KEYS)

_EPISODE_HISTORY_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("ep_return", "completed_return"),
    ("ep_len", "completed_len"),
    ("ep_safe_rate", "completed_safe_rate"),
    ("ep_speed_error_abs", "completed_speed_error_abs"),
    ("ep_y_error_abs", "completed_y_error_abs"),
    ("ep_psi_error_abs", "completed_psi_error_abs"),
)

_EPISODE_FIELD_NAMES: Tuple[str, ...] = tuple(f"{key}_sum" for key, _ in _EPISODE_HISTORY_FIELDS)


def _episode_fields(info: Any) -> Dict[str, jax.Array]:
    return {
        f"{hist_key}_sum": jnp.sum(getattr(info, info_field))
        for hist_key, info_field in _EPISODE_HISTORY_FIELDS
    }


def _run_training_jax(
    sac_cfg: SACConfig,
    env_cfg: UnicycleEnvConfig,
    cbf_cfg: UnicycleBCBFConfig,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    if int(sac_cfg.num_envs) <= 0:
        raise ValueError(f"num_envs must be positive, got {sac_cfg.num_envs}")
    if int(sac_cfg.steps_per_jit) <= 0:
        raise ValueError(f"steps_per_jit must be positive, got {sac_cfg.steps_per_jit}")

    num_envs = int(sac_cfg.num_envs)
    steps_per_jit = int(sac_cfg.steps_per_jit)
    total_steps = int(sac_cfg.total_steps)
    total_vec_steps = (total_steps + num_envs - 1) // num_envs

    jax_key = make_prng_key(sac_cfg.seed)
    env_fns = build_unicycle_env(env_cfg)

    action_scale_np = np.asarray(jax.device_get(env_fns.action_scale), dtype=np.float64)
    action_low_np = np.asarray(jax.device_get(env_fns.action_low), dtype=np.float64)
    action_high_np = np.asarray(jax.device_get(env_fns.action_high), dtype=np.float64)
    action_scale = jnp.asarray(action_scale_np, dtype=jnp.float32)
    expected_action_scale = np.array([cbf_cfg.a_max, cbf_cfg.r_max], dtype=np.float64)

    _validate_action_scale("jax_env.action_scale", action_scale_np, expected_action_scale)
    if not (np.array_equal(action_low_np, -action_scale_np) and np.array_equal(action_high_np, action_scale_np)):
        raise ValueError(
            "unicycle env action box must be exactly [-action_scale, action_scale]; "
            f"got low={action_low_np.tolist()}, high={action_high_np.tolist()}, scale={action_scale_np.tolist()}"
        )
    if cbf_cfg.backup_policy_mode.strip().lower() == "learned":
        learned = load_learned_unicycle_backup_policy(cbf_cfg.learned_backup_policy_path)
        _validate_action_scale("learned_backup_policy.action_scale", learned.action_scale, expected_action_scale)

    actor_cfg = ActorConfig(
        obs_dim=env_fns.obs_dim,
        action_dim=env_fns.action_dim,
        hidden_sizes=(sac_cfg.hidden_size, sac_cfg.hidden_size),
    )
    critic_cfg = CriticConfig(
        obs_dim=env_fns.obs_dim,
        act_dim=env_fns.action_dim,
        hidden_sizes=(sac_cfg.hidden_size, sac_cfg.hidden_size),
    )

    jax_key, key_state, key_env = jax.random.split(jax_key, 3)
    state = init_sac_state(key_state, actor_cfg, critic_cfg, sac_cfg)
    replay = ps2_replay_init(sac_cfg.replay_size, env_fns.obs_dim, env_fns.action_dim)
    projector = UnicycleBackupCBFProjector(cbf_cfg)
    backup_runtime = projector.runtime
    proj_ops = _make_projection_ops(cbf_cfg, backup_runtime)

    update_fn = _build_update_fn(sac_cfg, actor_cfg, cbf_cfg, action_scale, backup_runtime=backup_runtime)
    action_low = jnp.asarray(action_low_np, dtype=jnp.float32)
    action_high = jnp.asarray(action_high_np, dtype=jnp.float32)
    sample_action_batch_fn = build_ps2_batched_action_fn(
        sac_cfg,
        actor_cfg,
        action_scale,
        action_low,
        action_high,
        proj_ops,
        phys_dim=_PHYS_DIM,
    )
    _, eval_action_fn = _build_action_fns(
        sac_cfg,
        actor_cfg,
        cbf_cfg,
        action_scale,
        backup_runtime=backup_runtime,
    )

    env_keys = jax.random.split(key_env, num_envs)
    env_state, obs = env_fns.reset_batched(env_keys)
    loop_state = PS2LoopState(
        state=state,
        replay=replay,
        env_state=env_state,
        obs=obs,
        key=jax_key,
        env_keys=env_keys,
        global_step=jnp.int32(0),
        updates=jnp.int32(0),
    )

    one_vec_step = build_ps2_one_vec_step(
        env_fns=env_fns,
        sample_action_batch_fn=sample_action_batch_fn,
        update_fn=update_fn,
        sac_cfg=sac_cfg,
        action_low=action_low,
        action_high=action_high,
        proj=proj_ops,
        phys_dim=_PHYS_DIM,
        sanitize_random_actions=True,
        update_metric_pairs=_UPDATE_METRIC_PAIRS,
        episode_fields_fn=_episode_fields,
        episode_field_names=_EPISODE_FIELD_NAMES,
    )
    get_chunk_fn = make_ps2_chunk_fn_getter(one_vec_step)

    history: Dict[str, List[float]] = {
        "step": [],
        "ep_return": [],
        "ep_len": [],
        "ep_safe_rate": [],
        "ep_speed_error_abs": [],
        "ep_y_error_abs": [],
        "ep_psi_error_abs": [],
        "critic_loss": [],
        "actor_loss": [],
        "alpha": [],
        "q1_grad_norm": [],
        "q2_grad_norm": [],
        "actor_grad_norm": [],
        "target_action_bad_rate": [],
        "actor_action_bad_rate": [],
        "target_use_solver_rate": [],
        "actor_use_solver_rate": [],
        "target_fallback_rate": [],
        "actor_fallback_rate": [],
        "target_q_mat_finite_rate": [],
        "target_q_vec_finite_rate": [],
        "target_g_finite_rate": [],
        "target_h_finite_rate": [],
        "target_z_finite_rate": [],
        "actor_q_mat_finite_rate": [],
        "actor_q_vec_finite_rate": [],
        "actor_g_finite_rate": [],
        "actor_h_finite_rate": [],
        "actor_z_finite_rate": [],
        "eval_return_mean": [],
        "eval_safe_rate": [],
        "eval_speed_error_abs_mean": [],
        "eval_y_error_abs_mean": [],
        "eval_psi_error_abs_mean": [],
    }

    latest_upd_metrics: Dict[str, float] | None = None
    latest_episode_metrics: Dict[str, float] | None = None
    best_eval_stats: Dict[str, float] | None = None
    best_eval_step = 0
    best_eval_score = float("-inf")
    best_state: Dict[str, Any] | None = None
    t0 = time.time()
    last_log_t = t0
    next_eval_step = int(sac_cfg.eval_every)
    next_log_step = int(sac_cfg.log_every)

    vec_done = 0
    while vec_done < total_vec_steps:
        this_vec = min(steps_per_jit, total_vec_steps - vec_done)
        loop_state, chunk = get_chunk_fn(this_vec)(loop_state)
        vec_done += this_vec

        chunk_host = jax.device_get(chunk)
        step_now = int(min(total_steps, int(chunk_host["global_step"])))
        upd_count = float(chunk_host["update_count"])
        ep_count = float(chunk_host["episode_count"])

        if upd_count > 0.0:
            inv_upd = 1.0 / max(1.0, upd_count)
            latest_upd_metrics = {key: float(chunk_host[f"{key}_sum"] * inv_upd) for key in _UPDATE_HISTORY_KEYS}
            if sac_cfg.record_update_metrics:
                for key in _UPDATE_HISTORY_KEYS:
                    history[key].append(latest_upd_metrics[key])

        if ep_count > 0.0:
            inv_ep = 1.0 / max(1.0, ep_count)
            history["step"].append(float(step_now))
            latest_episode_metrics = {"last_step": float(step_now)}
            for key, _ in _EPISODE_HISTORY_FIELDS:
                value = float(chunk_host[f"{key}_sum"] * inv_ep)
                history[key].append(value)
                latest_episode_metrics[key] = value

        while step_now >= next_eval_step and next_eval_step <= total_steps:
            eval_stats = _evaluate_policy(
                env_cfg,
                eval_action_fn,
                loop_state.state["actor_params"],
                sac_cfg.seed + next_eval_step,
                sac_cfg.eval_episodes,
            )
            history["eval_return_mean"].append(eval_stats["return_mean"])
            history["eval_safe_rate"].append(eval_stats["safe_rate"])
            history["eval_speed_error_abs_mean"].append(eval_stats["speed_error_abs_mean"])
            history["eval_y_error_abs_mean"].append(eval_stats["y_error_abs_mean"])
            history["eval_psi_error_abs_mean"].append(eval_stats["psi_error_abs_mean"])
            eval_score = float(eval_stats["return_mean"])
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_eval_step = next_eval_step
                best_eval_stats = {
                    "return_mean": float(eval_stats["return_mean"]),
                    "return_std": float(eval_stats["return_std"]),
                    "safe_rate": float(eval_stats["safe_rate"]),
                    "speed_error_abs_mean": float(eval_stats["speed_error_abs_mean"]),
                    "y_error_abs_mean": float(eval_stats["y_error_abs_mean"]),
                    "psi_error_abs_mean": float(eval_stats["psi_error_abs_mean"]),
                }
                best_state = _snapshot_state(loop_state.state)
            next_eval_step += int(sac_cfg.eval_every)

        while step_now >= next_log_step and next_log_step <= total_steps:
            now = time.time()
            step_rate = float(max(1, sac_cfg.log_every) / max(1e-6, now - last_log_t))
            last_log_t = now
            print(
                f"step={next_log_step} replay={int(jax.device_get(loop_state.replay.size))} "
                f"updates={int(jax.device_get(loop_state.updates))} "
                f"eps/sec={step_rate:.1f} "
                f"ineq={projector.num_qp_inequalities} "
                f"env_dt={env_cfg.dt:.4f} "
                f"cbf_dt={cbf_cfg.dt:.4f} "
                f"horizon={cbf_cfg.horizon:.3f} "
                f"N={cbf_cfg.num_steps} "
                f"num_envs={num_envs}"
            )
            if latest_upd_metrics is not None:
                print(
                    "  "
                    f"critic={latest_upd_metrics['critic_loss']:.4f} "
                    f"actor={latest_upd_metrics['actor_loss']:.4f} "
                    f"alpha={latest_upd_metrics['alpha']:.4f} "
                    f"q1_gn={latest_upd_metrics['q1_grad_norm']:.3f} "
                    f"actor_gn={latest_upd_metrics['actor_grad_norm']:.3f} "
                    f"bad_target={latest_upd_metrics['target_action_bad_rate']:.3f} "
                    f"bad_actor={latest_upd_metrics['actor_action_bad_rate']:.3f} "
                    f"solver_target={latest_upd_metrics['target_use_solver_rate']:.3f} "
                    f"solver_actor={latest_upd_metrics['actor_use_solver_rate']:.3f} "
                    f"fb_target={latest_upd_metrics['target_fallback_rate']:.3f} "
                    f"fb_actor={latest_upd_metrics['actor_fallback_rate']:.3f}"
                )
                print(
                    "  "
                    f"finite_target(q_mat={latest_upd_metrics['target_q_mat_finite_rate']:.3f}, "
                    f"q_vec={latest_upd_metrics['target_q_vec_finite_rate']:.3f}, "
                    f"g={latest_upd_metrics['target_g_finite_rate']:.3f}, "
                    f"h={latest_upd_metrics['target_h_finite_rate']:.3f}, "
                    f"z={latest_upd_metrics['target_z_finite_rate']:.3f}) "
                    f"finite_actor(q_mat={latest_upd_metrics['actor_q_mat_finite_rate']:.3f}, "
                    f"q_vec={latest_upd_metrics['actor_q_vec_finite_rate']:.3f}, "
                    f"g={latest_upd_metrics['actor_g_finite_rate']:.3f}, "
                    f"h={latest_upd_metrics['actor_h_finite_rate']:.3f}, "
                    f"z={latest_upd_metrics['actor_z_finite_rate']:.3f})"
                )
            next_log_step += int(sac_cfg.log_every)

    total_time = time.time() - t0
    eval_stats = _evaluate_policy(env_cfg, eval_action_fn, loop_state.state["actor_params"], sac_cfg.seed + 99_999, sac_cfg.eval_episodes)
    final_eval_score = float(eval_stats["return_mean"])
    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_eval_step = total_steps
        best_eval_stats = {
            "return_mean": float(eval_stats["return_mean"]),
            "return_std": float(eval_stats["return_std"]),
            "safe_rate": float(eval_stats["safe_rate"]),
            "speed_error_abs_mean": float(eval_stats["speed_error_abs_mean"]),
            "y_error_abs_mean": float(eval_stats["y_error_abs_mean"]),
            "psi_error_abs_mean": float(eval_stats["psi_error_abs_mean"]),
        }
        best_state = _snapshot_state(loop_state.state)

    updates_final = int(jax.device_get(loop_state.updates))
    summary = {
        "total_steps": total_steps,
        "updates": updates_final,
        "wall_time_sec": float(total_time),
        "steps_per_sec": float(total_steps / max(total_time, 1e-6)),
        "num_qp_inequalities": projector.num_qp_inequalities,
        "num_backup_inequalities": projector.num_backup_inequalities,
        "env_dt": float(env_cfg.dt),
        "cbf_dt": float(cbf_cfg.dt),
        "horizon_T": cbf_cfg.horizon,
        "N_steps": cbf_cfg.num_steps,
        "backup_policy_mode": cbf_cfg.backup_policy_mode,
        "terminal_set_mode": "lqr",
        "capture_set_mode": "lqr",
        "base_set_c": float(cbf_cfg.base_set_c),
        "num_envs": num_envs,
        "steps_per_jit": steps_per_jit,
        "final_eval_return_mean": eval_stats["return_mean"],
        "final_eval_safe_rate": eval_stats["safe_rate"],
        "final_eval_speed_error_abs_mean": eval_stats["speed_error_abs_mean"],
        "final_eval_y_error_abs_mean": eval_stats["y_error_abs_mean"],
        "final_eval_psi_error_abs_mean": eval_stats["psi_error_abs_mean"],
        "best_eval_step": int(best_eval_step),
        "best_eval_return_mean": float(best_eval_stats["return_mean"]) if best_eval_stats is not None else float(eval_stats["return_mean"]),
        "best_eval_safe_rate": float(best_eval_stats["safe_rate"]) if best_eval_stats is not None else float(eval_stats["safe_rate"]),
        "best_eval_speed_error_abs_mean": float(best_eval_stats["speed_error_abs_mean"]) if best_eval_stats is not None else float(eval_stats["speed_error_abs_mean"]),
        "best_eval_y_error_abs_mean": float(best_eval_stats["y_error_abs_mean"]) if best_eval_stats is not None else float(eval_stats["y_error_abs_mean"]),
        "best_eval_psi_error_abs_mean": float(best_eval_stats["psi_error_abs_mean"]) if best_eval_stats is not None else float(eval_stats["psi_error_abs_mean"]),
    }

    result = {
        "summary": summary,
        "history": history,
        "eval": eval_stats,
        "best_eval": best_eval_stats
        if best_eval_stats is not None
        else {
            "return_mean": float(eval_stats["return_mean"]),
            "return_std": float(eval_stats["return_std"]),
            "safe_rate": float(eval_stats["safe_rate"]),
            "speed_error_abs_mean": float(eval_stats["speed_error_abs_mean"]),
            "y_error_abs_mean": float(eval_stats["y_error_abs_mean"]),
            "psi_error_abs_mean": float(eval_stats["psi_error_abs_mean"]),
        },
        "configs": {
            "sac": asdict(sac_cfg),
            "env": asdict(env_cfg),
            "cbf": asdict(cbf_cfg),
        },
    }

    result["final_state"] = _snapshot_state(loop_state.state)
    result["best_state"] = best_state if best_state is not None else result["final_state"]

    if output_dir is not None:
        with open(f"{output_dir}/summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(f"{output_dir}/configs.json", "w", encoding="utf-8") as f:
            json.dump(result["configs"], f, indent=2)
        with open(f"{output_dir}/final_weights.pkl", "wb") as f:
            pickle.dump(result["final_state"], f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{output_dir}/best_weights.pkl", "wb") as f:
            pickle.dump(result["best_state"], f, protocol=pickle.HIGHEST_PROTOCOL)

    return result


def run_training(
    sac_cfg: SACConfig,
    env_cfg: UnicycleEnvConfig,
    cbf_cfg: UnicycleBCBFConfig,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    return _run_training_jax(sac_cfg, env_cfg, cbf_cfg, output_dir=output_dir)
