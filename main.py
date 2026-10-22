#!/usr/bin/env python3
"""CLI entry point: main.py <phase> [--resume] [--max-hours H]

Phases follow agent.md Section 11. Every phase is resumable and time-capped
(Section 6.5): re-running a finished phase is a no-op that takes seconds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import ROOT, load_config, set_seeds  # noqa: E402  (sets MPS fallback first)

PHASES = ("m0", "m1", "pilot", "m3", "m4", "m5", "m6", "m7", "m7b", "m8", "m10")


def phase_m0(args, cfg) -> int:
    from scripts.m0_env import run

    return run(cfg, args)


def phase_m1(args, cfg) -> int:
    from scripts.m1_data import run

    return run(cfg, args)


def phase_pilot(args, cfg) -> int:
    from scripts.pilot import run

    return run(cfg, args)


def phase_m3(args, cfg) -> int:
    from scripts.m3_embed import run

    return run(cfg, args)


def phase_m4(args, cfg) -> int:
    from scripts.m4_extract import run

    return run(cfg, args)


def phase_m5(args, cfg) -> int:
    from scripts.m5_graphs import run

    return run(cfg, args)


def phase_m6(args, cfg) -> int:
    from scripts.m6_train import run

    return run(cfg, args)


def _not_built(name: str):
    def _run(args, cfg) -> int:
        print(f"phase {name} is not implemented yet (agent.md Section 11)")
        return 2

    return _run


DISPATCH = {
    "m0": phase_m0,
    "m1": phase_m1,
    "pilot": phase_pilot,
    "m3": phase_m3,
    "m4": phase_m4,
    "m5": phase_m5,
    "m6": phase_m6,
    **{p: _not_built(p) for p in ("m7", "m7b", "m8", "m10")},
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=PHASES)
    ap.add_argument("--resume", action="store_true", help="continue from cache; finished phases no-op")
    ap.add_argument("--max-hours", type=float, default=None, help="exit cleanly with cache flushed after H hours")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dataset", default=None, help="restrict to one dataset")
    ap.add_argument("-n", type=int, default=None, help="override sample size (debug only)")
    ap.add_argument("--allow-validation", metavar="REASON", default=None,
                    help="unlock reads of the validation split; the reason is appended "
                         "to results/validation_access.log (agent.md §12, D12)")
    args = ap.parse_args(argv)

    if args.allow_validation:
        from src.validation_guard import allow

        allow(args.allow_validation)
        print(f"!! validation access UNLOCKED: {args.allow_validation}")

    cfg = load_config(args.config)
    seed = set_seeds(args.seed if args.seed is not None else cfg.get("seed"))
    print(f"=== phase {args.phase} | seed {seed} | device {cfg.get('device') or 'unset'} ===")
    return DISPATCH[args.phase](args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
