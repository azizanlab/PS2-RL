#!/usr/bin/env python
"""Phase-1 backup-policy evaluation entrypoint.

Unified ``--system`` + ``--mode`` dispatcher over the Phase-1 evaluations. Two
modes, available for both systems where implemented:

  * ``--mode compare`` — the learned-vs-analytic backup-policy comparison.
      unicycle:  rerun the lane invariant-set / safe-arrival-measure comparison
                 on a saved checkpoint.
      quadrotor: rerun the near-powerloop recoverability comparison; dispatches to
                 ``scripts/compare_backup_policies.py``.
  * ``--mode score`` — held-out scoring of a single saved learned backup policy
      (quadrotor only).

If ``--mode`` is omitted it defaults to the historical behavior of this
entrypoint: ``compare`` for unicycle, ``score`` for quadrotor. All other flags are
forwarded verbatim to the selected evaluator; run e.g.
``evaluate_phase1.py --system quadrotor --mode compare --help``.

  evaluate_phase1.py --system unicycle  --mode compare [invariant-compare flags ...]
  evaluate_phase1.py --system quadrotor --mode compare [recoverability-compare flags ...]
  evaluate_phase1.py --system quadrotor --mode score   [held-out-eval flags ...]
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
    pre.add_argument(
        "--mode",
        choices=("score", "compare"),
        default=None,
        help="'compare' = learned-vs-analytic backup comparison; 'score' = held-out "
        "scoring of a saved policy (quadrotor only). Default: unicycle->compare, "
        "quadrotor->score.",
    )
    known, rest = pre.parse_known_args(argv)  # --help falls through to the selected evaluator

    # Per-system default preserves the historical behavior of this entrypoint.
    mode = known.mode or ("compare" if known.system == "unicycle" else "score")

    if known.system == "unicycle":
        if mode != "compare":
            raise SystemExit(
                "unicycle Phase-1 evaluation only supports --mode compare "
                "(the learned-vs-analytic invariant-set comparison)."
            )
        from ps2rl.evaluation.unicycle_p1_eval import main as uni_main
        raise SystemExit(uni_main(rest))  # uni main returns an int exit code

    # quadrotor
    if mode == "compare":
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from compare_backup_policies import main as quad_compare_main
        quad_compare_main(rest)
    else:  # score
        from ps2rl.evaluation.quadrotor_p1_eval import main as quad_main
        quad_main(rest)


if __name__ == "__main__":
    main()
