"""The playability check a save runs, under a wall clock the site owns.

Phase 2's first row of `docs/plan-website.md`: *validate -> smoke -> screen on save,
synchronous, capped*. Validation is schema work and costs milliseconds, so it runs here.
The other two rungs play games in the Rust core, and how long they take is a property of
the draft -- a 16 by 16 board of sliders under a ten-thousand-ply limit is a valid
ruleset that takes seconds here and far longer on the box (§6, *compute exhaustion*).

## Why a child process

The core's work has to be *stopped*, not waited on. `random_playouts` takes a wall-clock
budget and honours it between games, but the screen's tournament takes none, and a
Python thread cannot cancel a call that is already inside the core. Killing a process
can, and needs nothing under `rust/src/`. So the two playing rungs run in a child that
this process kills at `CHECK_SECONDS`, and a draft that runs past it is refused with a
sentence rather than saved.

The child is this file run as a script, which imports the two gates and nothing of the
site -- about half a second of interpreter start, inside the same budget. It reads the
draft on stdin and writes one JSON verdict on stdout, so there is no shared state to
leave behind when it is killed.

## One save, one core, one clock

The rungs run at `WORKERS` threads rather than at the machine's. A save is one person
pressing a button, and what it costs the box should be a wall clock times one core
rather than a wall clock times whatever the box has. A game is a pure function of
`(ruleset, seed, agents)` and every seed is fixed before any thread starts, so the
verdict is the same at any thread count.

## Nothing is kept from a check that did not finish

`SiteApp._save_design` runs this before the local application writes anything, so a
refusal leaves no file under `runs/designed/` and no `games` row. A draft the site could
not screen is not a draft the site holds.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# The wall clock a save's check may hold a request thread and a core for.
#
# Measured on the development machine at one worker: the three templates take 0.03 to
# 0.14 seconds, chess 0.29, xiangqi 0.58, and shogi -- the most expensive committed
# ruleset -- 1.57. Half a second of interpreter start sits inside the same budget. The
# box is a small VPS and several times slower per core, so 15 leaves the heaviest real
# game roughly a factor of two of room and still refuses the 16 by 16 slider board that
# §6's threat row describes, which costs 7.5 seconds here.
CHECK_SECONDS = 15.0

# The smoke rung's own share, handed to the core, which honours it between games.
#
# It is not the cap and does not replace it: a draft that stalls in random play is
# refused by name -- *only k of 150 random games finished within the budget* -- rather
# than by the clock running out on the whole check, which says less about what to fix.
# The rung never grades a sample the clock cut short (Q451): it grades all its games or
# refuses, so a draft's verdict does not depend on how fast the box is.
SMOKE_SECONDS = 5.0

SMOKE_GAMES = 150

# Threads for both playing rungs. See *One save, one core, one clock*.
WORKERS = 1


class TooSlow(ValueError):
    """The check ran past its wall clock, so the draft was refused and nothing kept.

    A `ValueError` like `quota.QuotaError`, so the answer is the 400 carrying the
    sentence the page prints where the click was.
    """


class CheckFailed(ValueError):
    """The child died without a verdict. The draft is refused for the same reason."""


def refusal(seconds: float) -> str:
    """What a person is told when their draft outran the clock."""
    return (
        f"These rules took longer than {seconds:g} seconds to check, so nothing was "
        "saved. They are too slow to play on this site: try a smaller board, fewer "
        "pieces, shorter moves, or a lower turn limit."
    )


def check(spec: Any, seconds: float | None = None) -> dict[str, Any]:
    """Validate the draft here, then play the two gates in a child under the clock.

    The verdict is `screen_draft`'s shape -- `passed`, `reasons`, `stats`, `hash`,
    `seconds` -- with the smoke rung's own numbers under `smoke`. A failing smoke short
    circuits the screen, because the ladder is a cost ladder.
    """
    from chess_variant_sandbox.spec import spec_hash, validate_spec
    from chess_variant_sandbox.spec.plain import plain_problems
    from chess_variant_sandbox.web.designing import DesignError, settled

    if not isinstance(spec, dict):
        raise DesignError("That draft could not be read.")
    settled_spec = settled(spec)
    problems = validate_spec(settled_spec)
    if problems:
        raise DesignError("These rules aren't a complete game yet: "
                          + " ".join(plain_problems(problems[:4], settled_spec)))

    limit = CHECK_SECONDS if seconds is None else seconds
    started = time.perf_counter()
    verdict = _in_a_child(settled_spec, limit)
    return {
        **verdict,
        "hash": spec_hash(settled_spec),
        "seconds": round(time.perf_counter() - started, 3),
        "limit": limit,
    }


def _in_a_child(spec: dict[str, Any], seconds: float) -> dict[str, Any]:
    """Run the playing rungs in a child, and kill it when the clock runs out."""
    asked = json.dumps({
        "spec": spec,
        "games": SMOKE_GAMES,
        "smoke_seconds": min(SMOKE_SECONDS, seconds),
        "workers": WORKERS,
    })
    # `-I` keeps this directory off the child's `sys.path`, so running the file by path
    # cannot shadow an import with a module that happens to sit beside it.
    child = subprocess.Popen(
        [sys.executable, "-I", str(Path(__file__).resolve())],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, err = child.communicate(asked, timeout=seconds)
    except subprocess.TimeoutExpired:
        child.kill()
        child.communicate()
        raise TooSlow(refusal(seconds)) from None
    if child.returncode != 0 or not out.strip():
        # The child's last line is the maintainer's (the server's log has it); a person
        # is told what to do (D29).
        lines = (err or "").strip().splitlines()
        said = lines[-1][:200] if lines else f"exit {child.returncode}"
        print(f"screening: the check's child failed: {said}", file=sys.stderr)
        raise CheckFailed(
            "The playability check could not be run just now, so nothing was saved. "
            "Please try again."
        )
    verdict: dict[str, Any] = json.loads(out)
    return verdict


# ------------------------------------------------------------------ the child


def verdict(
    spec: dict[str, Any], games: int, smoke_seconds: float, workers: int
) -> dict[str, Any]:
    """Smoke, then screen if smoke passed. The child's whole job.

    `rung` is the last rung that ran, keyed as `candidacy.rungs` keys them, so the page
    can say where the draft got to in the ladder's own words. A failing smoke short
    circuits the screen, so the rung is `smoke`; otherwise the screen ran and it is
    `screen`. Validation never reaches here -- it refuses in the request thread.
    """
    from chess_variant_sandbox.loop.gates import smoke_test
    from chess_variant_sandbox.loop.screen import screen

    smoked = smoke_test(spec, games=games, time_budget=smoke_seconds, workers=workers)
    if not smoked.passed:
        return {
            "passed": False,
            "reasons": list(smoked.reasons),
            "stats": {},
            "smoke": smoked.as_dict(),
            "rung": "smoke",
        }
    screened = screen(spec, workers=workers)
    return {
        "passed": screened.passed,
        "reasons": list(screened.reasons),
        "stats": screened.stats,
        "smoke": smoked.as_dict(),
        "rung": "screen",
    }


def _child() -> None:
    asked = json.loads(sys.stdin.read())
    json.dump(
        verdict(
            asked["spec"],
            games=int(asked["games"]),
            smoke_seconds=float(asked["smoke_seconds"]),
            workers=int(asked["workers"]),
        ),
        sys.stdout,
    )
    sys.stdout.flush()


if __name__ == "__main__":
    _child()
