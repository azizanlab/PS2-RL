#!/usr/bin/env python
"""Phase-2 PS2 training entrypoint (PS2 through the Control-Invariant Layer).

Thin ``--system`` dispatcher over the per-system training orchestration (the
numerical core is already unified in the ``ps2rl`` package). All other flags are
forwarded verbatim to the selected system's trainer; run
``train_phase2.py --system quadrotor --help`` for the full flag set.

  train_phase2.py --system quadrotor [quad PS2 flags ...]
  train_phase2.py --system unicycle  [uni PS2 flags ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--system", required=True, choices=("unicycle", "quadrotor"))
    known, rest = pre.parse_known_args(argv)  # --help falls through to the system trainer

    if known.system == "quadrotor":
        from ps2rl.phase2_ps2.quadrotor_ps2_entry import main as quad_main
        quad_main(rest)
    elif known.system == "unicycle":
        from ps2rl.phase2_ps2.unicycle_ps2_entry import main as uni_main
        uni_main(rest)
    else:  # pragma: no cover - argparse constrains choices
        raise SystemExit(f"Unknown --system {known.system!r}")


if __name__ == "__main__":
    main()
