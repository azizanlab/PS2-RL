#!/usr/bin/env python
"""Vanilla quadrotor tracker training (no safety projection).

The projection-OFF preset of the quadrotor Phase-2 SAC trainer: it produces the
reference-tracking policies used as the warm-start for Phase-2 and as the source
of the reset-library traces. Equivalent to
``train_phase2.py --system quadrotor --disable_projection --no_project_actor_actions``.
All other flags are forwarded verbatim (``--help`` for the full flag set).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    from ps2rl.phase2_ps2.quadrotor_ps2_entry import main as quad_main
    # Force projection OFF (vanilla tracker); other flags pass through unchanged.
    quad_main(["--disable_projection", "--no_project_actor_actions", *argv])


if __name__ == "__main__":
    main()
