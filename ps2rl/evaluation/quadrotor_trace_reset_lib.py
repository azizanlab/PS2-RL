"""Trajectory-conditioned reset-library builder for quadrotor backup-policy training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
import pickle
import re
from typing import Any

import numpy as np

from ps2rl.base_controller.quadrotor_dlqr import QuadrotorDLQR
from ps2rl.sets.base_sets import EllipsoidBaseSet
from ps2rl.sets.quadrotor_sets import QuadrotorSafeSet
from ps2rl.cil.quadrotor_backup_cbf import QuadrotorBCBFConfig
from ps2rl.utils.paths import resolve_existing_path
from ps2rl.utils.quaternion import (
    normalize_quaternion_np,
    quaternion_from_euler_zyx_np,
    quaternion_multiply_np,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

_POOL_NAMES = ("general_trace", "near_ceiling", "bridge", "base_shell")


def _resolve_existing_path(raw: str | Path) -> Path:
    return resolve_existing_path(raw, bases=(Path.cwd(), PROJECT_ROOT))


def _sanitize_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_jsonable(payload), f, indent=2)


def _string_dtype(max_len: int) -> np.dtype[np.str_]:
    return np.dtype(f"<U{max(1, int(max_len))}")


def _string_array(values: list[str] | tuple[str, ...]) -> np.ndarray:
    if not values:
        return np.zeros((0,), dtype=_string_dtype(1))
    max_len = max(len(str(v)) for v in values)
    return np.asarray([str(v) for v in values], dtype=_string_dtype(max_len))


def _string_full(length: int, value: str) -> np.ndarray:
    return np.full((int(length),), str(value), dtype=_string_dtype(len(str(value))))


def _gather_trace_paths(
    outputs_dir: Path,
    *,
    reference_run_names: tuple[str, ...],
    reference_run_glob: str,
    reference_glob: str,
    max_traces: int,
) -> list[Path]:
    run_dirs: list[Path] = []
    if reference_run_names:
        for raw in reference_run_names:
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = outputs_dir / candidate
            if not candidate.exists() or not candidate.is_dir():
                raise FileNotFoundError(f"Run directory not found: {raw}")
            run_dirs.append(candidate.resolve())
    else:
        run_glob = reference_run_glob.strip() or "*"
        run_dirs = [p.resolve() for p in sorted(outputs_dir.glob(run_glob)) if p.is_dir()]

    if not run_dirs:
        raise FileNotFoundError(f"No run directories matched under {outputs_dir}")

    trace_paths: list[Path] = []
    for run_dir in run_dirs:
        matches = sorted(run_dir.glob(reference_glob))
        if not matches:
            raise FileNotFoundError(
                f"No trace files matched reference_glob='{reference_glob}' under run_dir={run_dir}"
            )
        trace_paths.extend(p.resolve() for p in matches if p.is_file())

    if max_traces > 0:
        trace_paths = trace_paths[: int(max_traces)]
    if not trace_paths:
        raise FileNotFoundError("No best_episode_trace.npz files were selected.")
    return trace_paths


_STAGED_TRACE_RE = re.compile(r"trace_seed(\d+)_runIdx(\d+)\.npz$")


def _gather_trace_paths_explicit(
    trace_dir: Path,
    *,
    glob: str,
    select_seed: int,
    run_idx_min: int,
    run_idx_max: int,
) -> list[Path]:
    """Explicit reset-library trace selection from the shipped flat trace dir."""
    selected: list[tuple[int, Path]] = []
    for p in sorted(trace_dir.glob(glob)):
        m = _STAGED_TRACE_RE.search(p.name)
        if not m:
            continue
        seed, run_idx = int(m.group(1)), int(m.group(2))
        if seed == int(select_seed) and int(run_idx_min) <= run_idx <= int(run_idx_max):
            selected.append((run_idx, p.resolve()))
    selected.sort(key=lambda t: t[0])
    run_indices = [ri for ri, _ in selected]
    expected = list(range(int(run_idx_min), int(run_idx_max) + 1))
    if run_indices != expected:
        raise ValueError(
            f"Explicit trace selection under {trace_dir} (seed={select_seed}) got "
            f"runIdx {run_indices}, expected {expected}."
        )
    return [p for _, p in selected]


def _load_trace_file(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}

    required = ["obs", "act", "ref_state", "ref_time_sec", "ref_progress"]
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"Trace file {path} is missing required keys: {missing}")

    obs = np.asarray(arrays["obs"], dtype=np.float64)
    act = np.asarray(arrays["act"], dtype=np.float64)
    if obs.ndim != 2 or obs.shape[1] != 10:
        raise ValueError(f"'obs' must have shape (T, 10), got {obs.shape} from {path}")
    if act.ndim != 2 or act.shape[0] != obs.shape[0] or act.shape[1] != 4:
        raise ValueError(f"'act' must have shape (T, 4), got {act.shape} from {path}")

    order = np.arange(obs.shape[0], dtype=np.int64)
    if "step_in_episode" in arrays:
        step_in_episode = np.asarray(arrays["step_in_episode"], dtype=np.int32).reshape(-1)
        if step_in_episode.shape[0] != obs.shape[0]:
            raise ValueError(f"'step_in_episode' length mismatch in {path}")
        order = np.argsort(step_in_episode, kind="stable")

    trace: dict[str, np.ndarray] = {}
    for key, arr in arrays.items():
        arr_np = np.asarray(arr)
        if arr_np.ndim > 0 and arr_np.shape[0] == obs.shape[0]:
            trace[key] = arr_np[order]
        else:
            trace[key] = arr_np

    trace["obs"] = np.asarray(trace["obs"], dtype=np.float64)
    trace["act"] = np.asarray(trace["act"], dtype=np.float64)
    trace["ref_state"] = np.asarray(trace["ref_state"], dtype=np.float64)
    trace["obs"][:, 6:10] = normalize_quaternion_np(trace["obs"][:, 6:10])
    trace["ref_state"][:, 6:10] = normalize_quaternion_np(trace["ref_state"][:, 6:10])
    return trace


def _load_reference_bundle_full(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "states" not in data or "omega_cmd" not in data:
            raise KeyError(f"Reference bundle {path} must contain 'states' and 'omega_cmd'.")
        if "a_cmd" not in data or "t" not in data:
            raise KeyError(f"Reference bundle {path} must contain 'a_cmd' and 't' for reset-library construction.")
        states = np.asarray(data["states"], dtype=np.float64)
        omega_cmd = np.asarray(data["omega_cmd"], dtype=np.float64)
        a_cmd = np.asarray(data["a_cmd"], dtype=np.float64).reshape(-1)
        t = np.asarray(data["t"], dtype=np.float64).reshape(-1)
    if states.ndim != 2 or states.shape[1] != 10:
        raise ValueError(f"Expected reference states with shape (N, 10), got {states.shape} from {path}")
    if omega_cmd.shape != (states.shape[0], 3):
        raise ValueError(f"Expected reference omega_cmd shape {(states.shape[0], 3)}, got {omega_cmd.shape} from {path}")
    if a_cmd.shape[0] != states.shape[0] or t.shape[0] != states.shape[0]:
        raise ValueError(f"Reference bundle lengths do not match states length {states.shape[0]} from {path}")
    states[:, 6:10] = normalize_quaternion_np(states[:, 6:10])
    return {
        "states": states,
        "omega_cmd": omega_cmd,
        "a_cmd": a_cmd,
        "t": t,
    }


@dataclass(frozen=True)
class QuadrotorTraceSourceConfig:
    reference_path: str = "ps2rl/envs/assets/quadrotor_powerloop_reference.npz"
    max_traces: int = 20
    trace_set_label: str = "omega_runIdx_0to19"
    # Explicit staged-trace selection: when ``staged_trace_dir`` is non-empty the
    # library is built from the shipped flat ``trace_seed{S}_runIdx{NN}.npz`` files
    # (seed==select_seed, select_run_idx_min..max, numeric-sorted) instead of the
    # legacy ``sorted(glob(run_dirs))[:max_traces]`` ordering artifact.
    staged_trace_dir: str = ""
    staged_trace_glob: str = "trace_seed*_runIdx*.npz"
    select_seed: int = 0
    select_run_idx_min: int = 0
    select_run_idx_max: int = 19


@dataclass(frozen=True)
class QuadrotorResetLibraryConfig:
    near_ceiling_margin: float = 0.35
    bridge_num_interp: int = 41
    base_shell_distance: float = 0.10
    base_shell_terminal_margin: float = 0.10

    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: int = 0

    position_perturb_min: float = 0.05
    position_perturb_max: float = 0.40
    velocity_perturb_min: float = 0.10
    velocity_perturb_max: float = 2.50
    tilt_perturb_deg_min: float = 2.0
    tilt_perturb_deg_max: float = 35.0
    yaw_perturb_deg_min: float = 1.0
    yaw_perturb_deg_max: float = 12.0

    general_region_multiplier: float = 1.0
    near_ceiling_region_multiplier: float = 1.5
    bridge_region_multiplier: float = 1.8
    base_shell_region_multiplier: float = 0.8

    mix_general_low: float = 0.45
    mix_general_high: float = 0.20
    mix_near_ceiling_low: float = 0.15
    mix_near_ceiling_high: float = 0.35
    mix_bridge_low: float = 0.05
    mix_bridge_high: float = 0.30
    mix_base_shell_low: float = 0.35
    mix_base_shell_high: float = 0.15

    max_resample_tries: int = 80
    heldout_val_per_region: int = 64
    heldout_test_per_region: int = 64
    heldout_seed: int = 1234


def _base_shell_pool_name(name: str) -> str:
    return "base_shell" if str(name) == "capture_shell" else str(name)


def _base_shell_region_array(key: str, arr: np.ndarray) -> np.ndarray:
    if str(key) in ("region", "primary_region") and arr.dtype.kind in ("U", "S", "O"):
        return np.asarray([_base_shell_pool_name(str(x)) for x in arr.reshape(-1)]).reshape(arr.shape)
    return arr


_RESET_CBF_KEYS = (
    "gravity", "dt", "z_des", "z_max", "a_cmd_min", "a_cmd_max", "omega_max",
    "base_set_c", "base_set_smooth_gain",
    "lqr_q_z", "lqr_q_vx", "lqr_q_vy", "lqr_q_vz",
    "lqr_q_thetax", "lqr_q_thetay", "lqr_q_thetaz",
    "lqr_r_a_cmd", "lqr_r_omega_x", "lqr_r_omega_y", "lqr_r_omega_z",
)
_RESET_TRACE_SOURCE_KEYS = (
    "max_traces", "trace_set_label", "staged_trace_dir", "staged_trace_glob",
    "select_seed", "select_run_idx_min", "select_run_idx_max",
)


def _reset_relevant_cfg(cfg: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    """``asdict(cfg)`` restricted to the reset-relevant ``keys``."""
    return {k: v for k, v in asdict(cfg).items() if k in keys}


@dataclass
class QuadrotorResetLibrary:
    trace_source_cfg: QuadrotorTraceSourceConfig
    library_cfg: QuadrotorResetLibraryConfig
    cbf_cfg: QuadrotorBCBFConfig
    all_pools: dict[str, np.ndarray] = field(default_factory=dict)
    split_pools: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    nominal_dataset: dict[str, np.ndarray] = field(default_factory=dict)
    heldout_reset_sets: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.safe_set = QuadrotorSafeSet.from_cbf_config(self.cbf_cfg)
        self.base_set = EllipsoidBaseSet(
            QuadrotorDLQR.from_config(self.cbf_cfg),
            float(self.cbf_cfg.base_set_c),
            smooth_gain=float(self.cbf_cfg.base_set_smooth_gain),
        )

    def curriculum_pool_weights(self, curriculum_scale: float) -> dict[str, float]:
        s = float(np.clip(curriculum_scale, 0.0, 1.0))
        cfg = self.library_cfg
        weights = {
            "general_trace": (1.0 - s) * cfg.mix_general_low + s * cfg.mix_general_high,
            "near_ceiling": (1.0 - s) * cfg.mix_near_ceiling_low + s * cfg.mix_near_ceiling_high,
            "bridge": (1.0 - s) * cfg.mix_bridge_low + s * cfg.mix_bridge_high,
            "base_shell": (1.0 - s) * cfg.mix_base_shell_low + s * cfg.mix_base_shell_high,
        }
        return weights

    def _region_multiplier(self, pool_name: str) -> float:
        cfg = self.library_cfg
        if pool_name == "near_ceiling":
            return float(cfg.near_ceiling_region_multiplier)
        if pool_name == "bridge":
            return float(cfg.bridge_region_multiplier)
        if pool_name == "base_shell":
            return float(cfg.base_shell_region_multiplier)
        return float(cfg.general_region_multiplier)

    def _sample_pool_name(self, rng: np.random.Generator, split: str, curriculum_scale: float) -> str:
        available = []
        weights = []
        for pool_name, weight in self.curriculum_pool_weights(curriculum_scale).items():
            states = self.split_pools.get(split, {}).get(pool_name)
            if states is None or len(states) == 0:
                continue
            available.append(pool_name)
            weights.append(max(float(weight), 0.0))
        if not available:
            raise ValueError(f"No reset pools are available for split='{split}'")
        prob = np.asarray(weights, dtype=np.float64)
        prob /= np.sum(prob)
        return str(rng.choice(np.asarray(available, dtype=object), p=prob))

    def _perturb_ranges(self, curriculum_scale: float, pool_name: str) -> dict[str, float]:
        s = float(np.clip(curriculum_scale, 0.0, 1.0))
        cfg = self.library_cfg
        mult = self._region_multiplier(pool_name)
        return {
            "pos": mult * ((1.0 - s) * cfg.position_perturb_min + s * cfg.position_perturb_max),
            "vel": mult * ((1.0 - s) * cfg.velocity_perturb_min + s * cfg.velocity_perturb_max),
            "tilt_rad": np.deg2rad(
                mult * ((1.0 - s) * cfg.tilt_perturb_deg_min + s * cfg.tilt_perturb_deg_max)
            ),
            "yaw_rad": np.deg2rad(
                mult * ((1.0 - s) * cfg.yaw_perturb_deg_min + s * cfg.yaw_perturb_deg_max)
            ),
        }

    def _is_safe_np(self, x: np.ndarray) -> bool:
        return bool(np.asarray(self.safe_set.contains(x), dtype=bool))

    def _base_shell_np(self, x: np.ndarray) -> bool:
        capture_dist = float(np.asarray(self.base_set.smooth_distance(x), dtype=np.float64))
        terminal_margin = float(np.asarray(self.base_set.margin(x), dtype=np.float64))
        cfg = self.library_cfg
        return (capture_dist <= float(cfg.base_shell_distance)) or (
            terminal_margin >= -float(cfg.base_shell_terminal_margin)
        )

    def perturb_state(
        self,
        base_state: np.ndarray,
        *,
        rng: np.random.Generator,
        curriculum_scale: float,
        pool_name: str,
    ) -> np.ndarray:
        x0 = np.asarray(base_state, dtype=np.float64).reshape(10)
        ranges = self._perturb_ranges(curriculum_scale, pool_name)
        for _ in range(max(1, int(self.library_cfg.max_resample_tries))):
            x = x0.copy()
            x[0:3] += rng.uniform(-ranges["pos"], ranges["pos"], size=3)
            x[3:6] += rng.uniform(-ranges["vel"], ranges["vel"], size=3)
            q_delta = quaternion_from_euler_zyx_np(
                float(rng.uniform(-ranges["tilt_rad"], ranges["tilt_rad"])),
                float(rng.uniform(-ranges["tilt_rad"], ranges["tilt_rad"])),
                float(rng.uniform(-ranges["yaw_rad"], ranges["yaw_rad"])),
            )
            x[6:10] = normalize_quaternion_np(quaternion_multiply_np(q_delta, x0[6:10]))
            if self._is_safe_np(x):
                return x
        x_safe = x0.copy()
        x_safe[6:10] = normalize_quaternion_np(x_safe[6:10])
        if not self._is_safe_np(x_safe):
            raise RuntimeError(f"Could not sample a safe perturbed reset from pool='{pool_name}'")
        return x_safe

    def sample_reset(
        self,
        rng: np.random.Generator,
        *,
        split: str = "train",
        curriculum_scale: float = 0.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        pool_name = self._sample_pool_name(rng, split, curriculum_scale)
        states = self.split_pools[split][pool_name]
        idx = int(rng.integers(0, len(states)))
        x = self.perturb_state(states[idx], rng=rng, curriculum_scale=curriculum_scale, pool_name=pool_name)
        return x, {
            "split": split,
            "pool_name": pool_name,
            "pool_index": idx,
            "curriculum_scale": float(np.clip(curriculum_scale, 0.0, 1.0)),
        }

    def sample_perturbed_region_states(
        self,
        pool_name: str,
        *,
        count: int,
        seed: int,
        curriculum_scale: float = 1.0,
        split: str | None = None,
    ) -> np.ndarray:
        states = self.all_pools.get(pool_name) if split is None else self.split_pools.get(split, {}).get(pool_name)
        if states is None or len(states) == 0 or count <= 0:
            return np.zeros((0, 10), dtype=np.float64)
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(int(count)):
            idx = int(rng.integers(0, len(states)))
            out.append(self.perturb_state(states[idx], rng=rng, curriculum_scale=curriculum_scale, pool_name=pool_name))
        return np.asarray(out, dtype=np.float64)

    def _build_fixed_reset_set(
        self,
        *,
        split: str,
        per_region: int,
        seed: int,
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        states_out: list[np.ndarray] = []
        regions_out: list[str] = []
        for region_idx, pool_name in enumerate(_POOL_NAMES):
            states = self.split_pools.get(split, {}).get(pool_name)
            if states is None or len(states) == 0:
                continue
            for _ in range(int(per_region)):
                idx = int(rng.integers(0, len(states)))
                local_seed = int(seed + 10_000 * (region_idx + 1) + idx + len(states_out))
                local_rng = np.random.default_rng(local_seed)
                states_out.append(
                    self.perturb_state(states[idx], rng=local_rng, curriculum_scale=1.0, pool_name=pool_name)
                )
                regions_out.append(pool_name)
        if not states_out:
            return {
                "states": np.zeros((0, 10), dtype=np.float64),
                "region": np.zeros((0,), dtype=_string_dtype(1)),
                "split": _string_full(0, split),
            }
        return {
            "states": np.asarray(states_out, dtype=np.float64),
            "region": _string_array(regions_out),
            "split": _string_full(len(states_out), split),
        }

    def rebuild_fixed_eval_sets(self) -> None:
        cfg = self.library_cfg
        self.heldout_reset_sets = {
            "val": self._build_fixed_reset_set(
                split="val",
                per_region=int(cfg.heldout_val_per_region),
                seed=int(cfg.heldout_seed),
            ),
            "test": self._build_fixed_reset_set(
                split="test",
                per_region=int(cfg.heldout_test_per_region),
                seed=int(cfg.heldout_seed + 97_531),
            ),
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "trace_source_cfg": _reset_relevant_cfg(self.trace_source_cfg, _RESET_TRACE_SOURCE_KEYS),
            "library_cfg": asdict(self.library_cfg),
            "cbf_cfg": _reset_relevant_cfg(self.cbf_cfg, _RESET_CBF_KEYS),
            "all_pools": self.all_pools,
            "split_pools": self.split_pools,
            "nominal_dataset": self.nominal_dataset,
            "heldout_reset_sets": self.heldout_reset_sets,
            "metadata": {},
        }
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "QuadrotorResetLibrary":
        with open(Path(path), "rb") as f:
            payload = pickle.load(f)
        cbf_valid = {f.name for f in fields(QuadrotorBCBFConfig) if f.init}
        cbf_cfg = QuadrotorBCBFConfig(
            **{k: v for k, v in payload["cbf_cfg"].items() if k in cbf_valid}
        )
        lib_valid = {f.name for f in fields(QuadrotorResetLibraryConfig) if f.init}
        return cls(
            trace_source_cfg=QuadrotorTraceSourceConfig(**payload["trace_source_cfg"]),
            library_cfg=QuadrotorResetLibraryConfig(
                **{k: v for k, v in payload["library_cfg"].items() if k in lib_valid}
            ),
            cbf_cfg=cbf_cfg,
            all_pools={
                _base_shell_pool_name(k): np.asarray(v, dtype=np.float64) for k, v in payload["all_pools"].items()
            },
            split_pools={
                split: {_base_shell_pool_name(k): np.asarray(v, dtype=np.float64) for k, v in pools.items()}
                for split, pools in payload["split_pools"].items()
            },
            nominal_dataset={
                k: _base_shell_region_array(k, np.asarray(v))
                for k, v in payload.get("nominal_dataset", {}).items()
            },
            heldout_reset_sets={
                split: {k: _base_shell_region_array(k, np.asarray(v)) for k, v in data.items()}
                for split, data in payload.get("heldout_reset_sets", {}).items()
            },
            metadata=dict(payload.get("metadata", {})),
        )


def _split_pool(states: np.ndarray, *, rng: np.random.Generator, cfg: QuadrotorResetLibraryConfig) -> dict[str, np.ndarray]:
    n = int(states.shape[0])
    if n == 0:
        return {split: np.zeros((0, 10), dtype=np.float64) for split in ("train", "val", "test")}
    order = rng.permutation(n)
    n_train = int(np.floor(cfg.train_fraction * n))
    n_val = int(np.floor(cfg.val_fraction * n))
    n_test = max(0, n - n_train - n_val)
    if n >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = max(1, n_test)
        overflow = n_train + n_val + n_test - n
        while overflow > 0:
            if n_train >= n_val and n_train >= n_test and n_train > 1:
                n_train -= 1
            elif n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            overflow -= 1
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val : n_train + n_val + n_test]
    return {
        "train": np.asarray(states[train_idx], dtype=np.float64),
        "val": np.asarray(states[val_idx], dtype=np.float64),
        "test": np.asarray(states[test_idx], dtype=np.float64),
    }


def _append_pool(pool_store: dict[str, list[np.ndarray]], pool_name: str, states: np.ndarray) -> None:
    if states.size == 0:
        return
    pool_store.setdefault(pool_name, []).append(np.asarray(states, dtype=np.float64))


def _safe_mask_np(safe_set: QuadrotorSafeSet, states: np.ndarray) -> np.ndarray:
    return np.asarray([bool(np.asarray(safe_set.contains(x), dtype=bool)) for x in states], dtype=bool)


def _base_shell_mask_np(
    base_set: EllipsoidBaseSet,
    states: np.ndarray,
    cfg: QuadrotorResetLibraryConfig,
) -> np.ndarray:
    out = np.zeros((states.shape[0],), dtype=bool)
    for i, x in enumerate(states):
        capture_dist = float(np.asarray(base_set.smooth_distance(x), dtype=np.float64))
        terminal_margin = float(np.asarray(base_set.margin(x), dtype=np.float64))
        out[i] = (capture_dist <= float(cfg.base_shell_distance)) or (
            terminal_margin >= -float(cfg.base_shell_terminal_margin)
        )
    return out


def _build_bridge_segments(
    states: np.ndarray,
    safe_mask: np.ndarray,
    *,
    num_interp: int,
) -> list[np.ndarray]:
    if states.shape[0] == 0:
        return []
    bridges: list[np.ndarray] = []
    unsafe = ~safe_mask
    if not np.any(unsafe):
        return bridges
    idx = np.arange(states.shape[0], dtype=np.int32)
    starts = idx[(unsafe) & np.concatenate([[True], ~unsafe[:-1]])]
    ends = idx[(unsafe) & np.concatenate([~unsafe[1:], [True]])]
    for start_unsafe, end_unsafe in zip(starts.tolist(), ends.tolist()):
        left = start_unsafe - 1
        right = end_unsafe + 1
        if left < 0 or right >= states.shape[0]:
            continue
        if (not safe_mask[left]) or (not safe_mask[right]):
            continue
        alpha = np.linspace(0.0, 1.0, max(2, int(num_interp)), dtype=np.float64)
        seg = (1.0 - alpha[:, None]) * states[left][None, :] + alpha[:, None] * states[right][None, :]
        seg[:, 6:10] = normalize_quaternion_np(seg[:, 6:10])
        bridges.append(np.asarray(seg, dtype=np.float64))
    return bridges


def build_quadrotor_reset_library(
    *,
    trace_source_cfg: QuadrotorTraceSourceConfig,
    library_cfg: QuadrotorResetLibraryConfig,
    cbf_cfg: QuadrotorBCBFConfig,
) -> QuadrotorResetLibrary:
    reference_bundle = _load_reference_bundle_full(_resolve_existing_path(trace_source_cfg.reference_path))
    if str(trace_source_cfg.staged_trace_dir).strip():
        outputs_dir = _resolve_existing_path(trace_source_cfg.staged_trace_dir)  # trace_name base
        trace_paths = _gather_trace_paths_explicit(
            outputs_dir,
            glob=str(trace_source_cfg.staged_trace_glob),
            select_seed=int(trace_source_cfg.select_seed),
            run_idx_min=int(trace_source_cfg.select_run_idx_min),
            run_idx_max=int(trace_source_cfg.select_run_idx_max),
        )
    else:
        outputs_dir = _resolve_existing_path(trace_source_cfg.vanilla_quadtrack_outputs_dir)
        trace_paths = _gather_trace_paths(
            outputs_dir,
            reference_run_names=tuple(trace_source_cfg.reference_run_name),
            reference_run_glob=trace_source_cfg.reference_run_glob,
            reference_glob=trace_source_cfg.reference_glob,
            max_traces=int(trace_source_cfg.max_traces),
        )

    safe_set = QuadrotorSafeSet.from_cbf_config(cbf_cfg)
    base_set = EllipsoidBaseSet(
        QuadrotorDLQR.from_config(cbf_cfg),
        float(cbf_cfg.base_set_c),
        smooth_gain=float(cbf_cfg.base_set_smooth_gain),
    )

    pool_lists: dict[str, list[np.ndarray]] = {name: [] for name in _POOL_NAMES}
    nominal_data_lists: dict[str, list[np.ndarray]] = {
        "states": [],
        "nom_act": [],
        "ref_progress": [],
        "ref_time_sec": [],
        "source_kind": [],
        "trace_name": [],
        "region": [],
    }
    trace_records: list[dict[str, Any]] = []
    num_bridge_segments = 0

    def add_real_trajectory(
        *,
        states: np.ndarray,
        actions: np.ndarray,
        ref_progress: np.ndarray,
        ref_time_sec: np.ndarray,
        trace_name: str,
        source_kind: str,
    ) -> None:
        nonlocal num_bridge_segments
        x = np.asarray(states, dtype=np.float64)
        x[:, 6:10] = normalize_quaternion_np(x[:, 6:10])
        u = np.asarray(actions, dtype=np.float64)
        safe_mask = _safe_mask_np(safe_set, x)
        near_ceiling_mask = safe_mask & ((cbf_cfg.z_max - x[:, 2]) <= float(library_cfg.near_ceiling_margin))
        base_shell_mask = safe_mask & _base_shell_mask_np(base_set, x, library_cfg)
        general_mask = safe_mask & (~near_ceiling_mask) & (~base_shell_mask)

        _append_pool(pool_lists, "general_trace", x[general_mask])
        _append_pool(pool_lists, "near_ceiling", x[near_ceiling_mask])
        _append_pool(pool_lists, "base_shell", x[base_shell_mask])

        primary_region = _string_full(x.shape[0], "general_trace")
        primary_region[near_ceiling_mask] = "near_ceiling"
        primary_region[base_shell_mask] = "base_shell"

        safe_idx = np.flatnonzero(safe_mask)
        if safe_idx.size > 0:
            nominal_data_lists["states"].append(x[safe_idx])
            nominal_data_lists["nom_act"].append(u[safe_idx])
            nominal_data_lists["ref_progress"].append(np.asarray(ref_progress[safe_idx], dtype=np.float64))
            nominal_data_lists["ref_time_sec"].append(np.asarray(ref_time_sec[safe_idx], dtype=np.float64))
            nominal_data_lists["source_kind"].append(_string_full(safe_idx.size, source_kind))
            nominal_data_lists["trace_name"].append(_string_full(safe_idx.size, trace_name))
            nominal_data_lists["region"].append(np.asarray(primary_region[safe_idx]))

        bridges = _build_bridge_segments(x, safe_mask, num_interp=int(library_cfg.bridge_num_interp))
        if bridges:
            num_bridge_segments += len(bridges)
            _append_pool(pool_lists, "bridge", np.concatenate(bridges, axis=0))

        trace_records.append(
            {
                "trace_name": trace_name,
                "source_kind": source_kind,
                "num_states_total": int(x.shape[0]),
                "num_safe_states": int(np.sum(safe_mask)),
                "num_near_ceiling_states": int(np.sum(near_ceiling_mask)),
                "num_base_shell_states": int(np.sum(base_shell_mask)),
                "num_bridge_segments": int(len(bridges)),
                "num_bridge_states": int(sum(seg.shape[0] for seg in bridges)),
            }
        )

    for trace_path in trace_paths:
        trace = _load_trace_file(trace_path)
        add_real_trajectory(
            states=np.asarray(trace["obs"], dtype=np.float64),
            actions=np.asarray(trace["act"], dtype=np.float64),
            ref_progress=np.asarray(trace["ref_progress"], dtype=np.float64).reshape(-1),
            ref_time_sec=np.asarray(trace["ref_time_sec"], dtype=np.float64).reshape(-1),
            trace_name=str(trace_path.relative_to(outputs_dir)),
            source_kind="vanilla_trace",
        )

    ref_states = np.asarray(reference_bundle["states"], dtype=np.float64)
    ref_states[:, 6:10] = normalize_quaternion_np(ref_states[:, 6:10])
    ref_actions = np.column_stack(
        [
            np.asarray(reference_bundle["a_cmd"], dtype=np.float64).reshape(-1),
            np.asarray(reference_bundle["omega_cmd"], dtype=np.float64),
        ]
    )
    t_ref = np.asarray(reference_bundle["t"], dtype=np.float64).reshape(-1)
    if t_ref.size > 1:
        ref_progress = np.linspace(0.0, 1.0, t_ref.size, dtype=np.float64)
    else:
        ref_progress = np.zeros((t_ref.size,), dtype=np.float64)
    add_real_trajectory(
        states=ref_states,
        actions=ref_actions,
        ref_progress=ref_progress,
        ref_time_sec=t_ref,
        trace_name=str(_resolve_existing_path(trace_source_cfg.reference_path).relative_to(PROJECT_ROOT)),
        source_kind="reference_bundle",
    )

    all_pools = {
        pool_name: (
            np.concatenate(pool_lists[pool_name], axis=0)
            if pool_lists[pool_name]
            else np.zeros((0, 10), dtype=np.float64)
        )
        for pool_name in _POOL_NAMES
    }

    split_rng = np.random.default_rng(int(library_cfg.split_seed))
    split_pools = {
        split: {}
        for split in ("train", "val", "test")
    }
    pool_split_counts: dict[str, dict[str, int]] = {}
    for pool_name, states in all_pools.items():
        parts = _split_pool(states, rng=split_rng, cfg=library_cfg)
        pool_split_counts[pool_name] = {split: int(parts[split].shape[0]) for split in parts}
        for split, x_split in parts.items():
            split_pools[split][pool_name] = x_split

    nominal_dataset = {
        key: (
            np.concatenate(values, axis=0)
            if values and key in {"states", "nom_act", "ref_progress", "ref_time_sec"}
            else np.concatenate(values, axis=0)
            if values
            else (
                np.zeros((0, 10), dtype=np.float64)
                if key == "states"
                else np.zeros((0, 4), dtype=np.float64)
                if key == "nom_act"
                else np.zeros((0,), dtype=np.float64)
                if key in {"ref_progress", "ref_time_sec"}
                else np.zeros((0,), dtype=_string_dtype(1))
            )
        )
        for key, values in nominal_data_lists.items()
    }

    metadata = {
        "trace_source_cfg": asdict(trace_source_cfg),
        "library_cfg": asdict(library_cfg),
        "base_set_c": float(cbf_cfg.base_set_c),
        "trace_records": trace_records,
        "num_trace_files_loaded": int(len(trace_paths)),
        "num_bridge_segments_total": int(num_bridge_segments),
        "pool_counts_total": {pool_name: int(states.shape[0]) for pool_name, states in all_pools.items()},
        "pool_split_counts": pool_split_counts,
        "nominal_dataset_size": int(nominal_dataset["states"].shape[0]),
    }

    library = QuadrotorResetLibrary(
        trace_source_cfg=trace_source_cfg,
        library_cfg=library_cfg,
        cbf_cfg=cbf_cfg,
        all_pools=all_pools,
        split_pools=split_pools,
        nominal_dataset=nominal_dataset,
        metadata=metadata,
    )
    library.rebuild_fixed_eval_sets()
    library.metadata["heldout_set_counts"] = {
        split: int(data["states"].shape[0]) for split, data in library.heldout_reset_sets.items()
    }
    return library


def export_quadrotor_reset_library_metadata(
    library: QuadrotorResetLibrary,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        **library.metadata,
        "trace_source_cfg": _reset_relevant_cfg(library.trace_source_cfg, _RESET_TRACE_SOURCE_KEYS),
        "library_cfg": asdict(library.library_cfg),
        "cbf_cfg": _reset_relevant_cfg(library.cbf_cfg, _RESET_CBF_KEYS),
    }
    _write_json(out_dir / "reset_library_metadata.json", metadata)
    np.savez(
        out_dir / "heldout_reset_sets.npz",
        val_states=np.asarray(library.heldout_reset_sets["val"]["states"], dtype=np.float64),
        val_region=np.asarray(library.heldout_reset_sets["val"]["region"]),
        test_states=np.asarray(library.heldout_reset_sets["test"]["states"], dtype=np.float64),
        test_region=np.asarray(library.heldout_reset_sets["test"]["region"]),
    )
    return metadata


__all__ = [
    "QuadrotorResetLibrary",
    "QuadrotorResetLibraryConfig",
    "QuadrotorTraceSourceConfig",
    "build_quadrotor_reset_library",
    "export_quadrotor_reset_library_metadata",
]
