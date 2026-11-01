"""Path resolution + payload IO helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_existing_path(
    path: str | Path | None,
    *,
    bases: Iterable[Path],
    allow_none: bool = False,
) -> Path | None:
    """Return the first existing candidate of ``path`` resolved against ``bases``."""
    if allow_none and not path:
        return None
    raw = Path(path).expanduser()
    candidates: list[Path] = []
    seen: set[str] = set()
    for base in bases:
        candidate = (base / raw).resolve()
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n".join(f"  - {candidate}" for candidate in candidates) or f"  - {raw}"
    raise FileNotFoundError(f"Path not found: {path}\nTried:\n{tried}")


def resolve_output_root(raw: str | None, default_root: Path) -> Path:
    """Resolve an output-root argument, falling back to ``default_root`` when empty."""
    if not raw:
        return default_root.resolve()
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def load_pickle_payload(path: str | Path) -> dict[str, Any]:
    """Load a pickle file and assert the payload is a dict."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected pickle payload type in {path}: {type(payload)} (expected dict)")
    return payload


def load_json_payload(path: str | Path) -> dict[str, Any]:
    """Load a JSON file and assert the payload is a dict."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected JSON payload type in {path}: {type(payload)} (expected dict)")
    return payload


__all__ = [
    "load_json_payload",
    "load_pickle_payload",
    "resolve_existing_path",
    "resolve_output_root",
]
