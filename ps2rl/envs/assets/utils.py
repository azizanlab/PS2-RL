"""Reference-trajectory bundle I/O for the quadrotor environment.

One-time, host-side NumPy loaders for the powerloop reference bundle that ships
alongside this module (``quadrotor_powerloop_reference.npz``). Kept out of the
JAX env hot loop; ``ps2rl.envs.quadrotor_env`` re-exports these names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from ps2rl.utils.quaternion import normalize_quaternion_np


def _default_reference_path() -> str:
    bundle_path = (Path(__file__).resolve().parent / "quadrotor_powerloop_reference.npz").resolve()
    if bundle_path.exists():
        return str(bundle_path)
    raise FileNotFoundError(f"Default reference trajectory file not found: {bundle_path}")


def _load_reference_bundle(path: str) -> Dict[str, np.ndarray]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Reference trajectory file not found: {p}")
    if p.suffix.lower() == ".npz":
        with np.load(p) as payload:
            if "states" not in payload:
                raise KeyError(f"Reference bundle {p} is missing required key 'states'.")
            states = np.asarray(payload["states"], dtype=np.float64)
            omega_cmd = (
                np.asarray(payload["omega_cmd"], dtype=np.float64)
                if "omega_cmd" in payload
                else np.zeros((states.shape[0], 3), dtype=np.float64)
            )
    else:
        states = np.asarray(np.load(p), dtype=np.float64)
        omega_cmd = np.zeros((states.shape[0], 3), dtype=np.float64)

    if states.ndim != 2 or states.shape[1] != 10:
        raise ValueError(f"Expected reference trajectory shape (N, 10), got {states.shape} from {p}")
    if states.shape[0] <= 0:
        raise ValueError(f"Reference trajectory is empty: {p}")
    if omega_cmd.ndim != 2 or omega_cmd.shape != (states.shape[0], 3):
        raise ValueError(
            f"Expected reference omega_cmd shape {(states.shape[0], 3)}, got {omega_cmd.shape} from {p}"
        )

    states_out = states.copy()
    states_out[:, 6:10] = np.asarray([normalize_quaternion_np(q) for q in states_out[:, 6:10]], dtype=np.float64)
    omega_out = omega_cmd.copy()
    return {
        "states": states_out,
        "omega_cmd": omega_out,
    }


def _load_reference_states(path: str) -> np.ndarray:
    return _load_reference_bundle(path)["states"]
