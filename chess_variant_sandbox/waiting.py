"""The waiting schedule: a run's test of whether a side can always wait (Q1184, T3b).

Pre-registered in `docs/plan-player-weaknesses.md` (*T3b's pre-registration*) before any
game of it was played. Beside each run of a library measurement, at the roster's top
searching rung:

- the **duel**: the rung with ``contempt=1`` (Q1171's diagnostic bit, 8192: a repetition
  it completes scores minus half a piece, so it declines every draw it can) against the
  plain rung, `PAIRS` seed-pairs, both seats a seed;
- the **control**: the plain rung against itself, `CONTROL` games;

both from the run's seed plus `OFFSET`, so no game is one of the run's own 256. Written as
ordinary records to ``waiting.jsonl`` in the run's directory, which `metrics/waiting.py`
reads for ``forced_repetition``. The players are the existing ones, driven through the
runner: nothing here chooses a move (CLAUDE.md, architecture rule 3), and no measured
player carries ``contempt``; only this schedule's duel player does.

A round or spy roster has no schedule: ``contempt`` reads zero there, so the duel would be
the plain player against itself.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from chess_variant_sandbox.tournament.record import GameRecord, read_records, write_records

PAIRS = 8
CONTROL = 16
OFFSET = 500
CONTEMPT = "contempt=1"
FILENAME = "waiting.jsonl"
STAMP = "waiting.json"
# What `Basis.waiting` records, with the code reading of this file beside it: a different
# schedule is different games, and a library stamp without it owes a replay.
SCHEDULE = f"contempt-duel:pairs={PAIRS},control={CONTROL},offset={OFFSET},rung=top"
_SOURCE = "chess_variant_sandbox/waiting.py"


def schedule_id() -> str:
    """The stamp's name for this schedule: `SCHEDULE` and the code that plays it."""
    from chess_variant_sandbox.measurement import code_fingerprint

    return f"{SCHEDULE}@{code_fingerprint(_SOURCE)}"


def top_rung(agents: tuple[str, ...]) -> str | None:
    """The roster's top searching rung: its last ``minimax:`` agent, or ``None``.

    ``minimax:nodes=25600`` on the canonical roster and ``minimax:net=<p>,nodes=25600`` on
    a rated net's; ``None`` on a round roster (``matrix:``), which plays no schedule.
    """
    searching = [agent for agent in agents if agent.startswith("minimax:")]
    return searching[-1] if searching else None


def configs(
    spec: dict[str, Any], agents: tuple[str, ...], seed: int, workers: int = 0
) -> list[Any]:
    """The schedule's two tournaments for the run at ``seed``, or none.

    The duel is `tournament.duel.duel_records`' paired schedule (both seats a seed, seeds
    ``seed + OFFSET`` onward); the control a self-play schedule from the same seed.
    """
    from chess_variant_sandbox.metrics.components import plays_in_rounds, plays_with_spies
    from chess_variant_sandbox.tournament.runner import TournamentConfig

    rung = top_rung(agents)
    if rung is None or plays_in_rounds(spec) or plays_with_spies(spec):
        return []
    start = seed + OFFSET
    return [
        TournamentConfig(
            agents=(f"{rung},{CONTEMPT}", rung), games_per_pairing=PAIRS, seed=start,
            workers=workers, include_self_play=False, collect_series=False, paired=True,
        ),
        TournamentConfig(
            agents=(rung,), games_per_pairing=CONTROL, seed=start, workers=workers,
            collect_series=False,
        ),
    ]


def play(
    spec: dict[str, Any],
    agents: tuple[str, ...],
    seed: int,
    workers: int = 0,
    run: Callable[..., list[Any]] | None = None,
) -> list[GameRecord]:
    """The schedule's games for the run at ``seed`` on this roster, or none.

    ``run`` plays a list of configs as `tournament.checkpoint.run_schedules` does (the
    default), so a caller can hand in its own; inside a campaign directory
    (`tournament.checkpoint.active`) both halves resume.
    """
    planned = configs(spec, agents, seed, workers)
    if not planned:
        return []
    if run is None:
        from chess_variant_sandbox.tournament.checkpoint import run_schedules

        run = run_schedules
    return [record for tournament in run(spec, planned) for record in tournament.records]


def save(records: list[GameRecord], run: Path) -> None:
    """Write the schedule beside the run, and which schedule it was (`STAMP`).

    The stamp is written where the schedule played no game too (a round roster), so a
    run is never asked for a schedule it already answered.
    """
    run.mkdir(parents=True, exist_ok=True)
    if records:
        write_records(records, run / FILENAME)
    else:
        (run / FILENAME).unlink(missing_ok=True)
    (run / STAMP).write_text(json.dumps({"schedule": schedule_id()}) + "\n", encoding="utf-8")


def played(run: Path) -> bool:
    """Whether this run holds today's schedule: its `STAMP` names `schedule_id`."""
    try:
        stamp = json.loads((run / STAMP).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(stamp, dict) and stamp.get("schedule") == schedule_id()


def load(run: Path) -> list[GameRecord] | None:
    """The schedule's games stored beside a run, or ``None`` where it has none."""
    path = run / FILENAME
    if not path.exists():
        return None
    return list(read_records(path)) or None
