"""`python -m chess_variant_sandbox.fuzz_lifted`: the lifted-pair soak (R-L6a, run by R-L6b).

    python -m chess_variant_sandbox.fuzz_lifted --cases 200
    python -m chess_variant_sandbox.fuzz_lifted --hours 4 --checkpoint runs/fuzz-lifted/soak.json

Prints the coverage report; exits 1 if any check failed. Failures, with the seed and the spec,
are in the checkpoint file (and in `<checkpoint>.failures.json` beside it).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from chess_variant_sandbox.fuzz_lifted.record import PLIES
from chess_variant_sandbox.fuzz_lifted.run import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fuzz the lifted pairs against their checks.")
    parser.add_argument("--cases", type=int, default=200, help="seeds to walk (with --hours: at most)")
    parser.add_argument("--start", type=int, default=0, help="the first seed")
    parser.add_argument("--plies", type=int, default=PLIES)
    parser.add_argument("--hours", type=float, default=None, help="stop after this long")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="state file: written as it goes, resumed from if present")
    args = parser.parse_args(argv)

    stop = args.start + (args.cases if args.hours is None else max(args.cases, 10**9))
    deadline = time.monotonic() + args.hours * 3600 if args.hours is not None else None
    coverage = run(range(args.start, stop), args.plies, args.checkpoint, deadline)
    print(coverage.report())
    if args.checkpoint is not None and coverage.failures:
        out = args.checkpoint.with_suffix(".failures.json")
        out.write_text(json.dumps(coverage.failures, indent=1), encoding="utf-8")
        print(f"failures written to {out}")
    return 1 if coverage.failures else 0


if __name__ == "__main__":  # pragma: no cover - a soak runner
    sys.exit(main())
