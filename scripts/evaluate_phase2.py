#!/usr/bin/env python
"""Phase-2 (PS2) policy evaluation entrypoint.

Thin ``--system`` dispatcher over the system-specific Phase-2 evaluations
(unicycle lane-keeping / quadrotor powerloop PS2 policies). For the quadrotor,
``--vanilla_tracker`` selects the vanilla-tracking-policy evaluation (the objE
tracker whose best episodes seed the reset library) instead of the PS2 eval. All
other flags are forwarded verbatim to the selected evaluator; run e.g.
``evaluate_phase2.py --system quadrotor --help``.

  evaluate_phase2.py --system unicycle                     [uni PS2 eval flags ...]
  evaluate_phase2.py --system quadrotor                    [quad PS2 eval flags ...]
  evaluate_phase2.py --system quadrotor --vanilla_tracker  [vanilla-tracker eval flags ...]
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
    pre.add_argument("--vanilla_tracker", action="store_true", default=False)
    known, rest = pre.parse_known_args(argv)  # --help falls through to the system evaluator

    if known.system == "quadrotor":
        if known.vanilla_tracker:
            from ps2rl.evaluation.quadrotor_vanilla_eval import main as quad_vanilla_main
            quad_vanilla_main(rest)
        else:
            from ps2rl.evaluation.quadrotor_p2_eval import main as quad_main
            quad_main(rest)
    elif known.system == "unicycle":
        if known.vanilla_tracker:
            pre.error("--vanilla_tracker is only valid with --system quadrotor")
        from ps2rl.evaluation.unicycle_p2_eval import main as uni_main
        uni_main(rest)
    else:  # pragma: no cover - argparse constrains choices
        raise SystemExit(f"Unknown --system {known.system!r}")


if __name__ == "__main__":
    main()
