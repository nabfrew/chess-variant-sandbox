"""Whether a side can always wait: ``forced_repetition`` and its gate (Q1184, T3b).

`docs/plan-player-weaknesses.md` *T3b's pre-registration* fixed all of this before any
game of it was played. A run of a library measurement carries a *waiting schedule* beside
its tournament (`chess_variant_sandbox/waiting.py`, persisted as ``waiting.jsonl``): at the
roster's top searching rung, a duel of that player with ``contempt=1`` (it scores a
repetition it completes at minus half a piece, so it declines every draw it can decline)
against the plain player, and the plain player's self-play as the control.

- Where the contempt player **wins** (``S - 2E > 0.5``), the plain players were drifting
  into repetitions a better player escapes, and the game owes nothing: the value is 0.
- Otherwise the repetition was the waiting side's to take, and the value is the control's
  repetition share: how often the strongest player, left to itself, draws by waiting.

Read from persisted records only. Absent where the run has no schedule (a visitor's one
run, a round or spy roster, a run played before the schedule existed): absence is not a
verdict, as for the seat.

The gate stands in code under every weights version, as the seat gate does and for its
reason (`index.gate_violations`): a gate added to a weights file would change verdicts
without staling a stored score, and this module is inside the `metrics` fingerprint.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from chess_variant_sandbox.metrics.scoring import Gate
from chess_variant_sandbox.tournament.record import GameRecord

FORCED_REPETITION = "forced_repetition"
WAITING_DETAIL = "waiting"
# The flag the schedule's duel player carries. A record is the duel's when exactly one seat
# names it, the control's when neither does.
CONTEMPT_FLAG = "contempt=1"
# The decisive gate's own line (`decisive_rate_worst_half` at least 0.2), read on the
# waiting player: fixed in the pre-registration before the library was read.
WAITING_MAXIMUM = 0.8
WAITING_GATE = Gate(
    component=FORCED_REPETITION,
    maximum=WAITING_MAXIMUM,
    reason=(
        "the strongest player draws four games in five by repetition against itself, and "
        "a player that refuses repetitions cannot win: a side can always wait, so the game "
        "is not a contest"
    ),
)


def _contempt_seat(record: GameRecord) -> int | None:
    seats = [seat for seat, agent in enumerate(record.agents) if CONTEMPT_FLAG in agent.split(",")]
    return seats[0] if len(seats) == 1 else None


def forced_repetition(records: list[GameRecord]) -> tuple[float | None, dict[str, Any]]:
    """The measurement and its detail, or ``(None, {})`` without a readable schedule.

    The duel is graded as `tournament.duel.grade` grades it on outcome alone: each
    seed's two games (the contempt player at each seat) give one estimate, ``S`` is their
    mean and ``E`` their standard error. Re-derived here rather than imported because a
    change to the duel module must stale this number, and only `metrics/` is fingerprinted
    for it.
    """
    duel: dict[int, list[float]] = {}
    duel_games = duel_repeated = 0
    control_games = control_repeated = 0
    rung = ""
    for record in records:
        seat = _contempt_seat(record)
        repeated = record.reason == "repetition"
        if seat is None:
            if any(CONTEMPT_FLAG in agent.split(",") for agent in record.agents):
                continue  # both seats contempt: no schedule plays this
            control_games += 1
            control_repeated += repeated
            rung = rung or record.agents[0]
        else:
            duel_games += 1
            duel_repeated += repeated
            duel.setdefault(record.seed, []).append(record.score_for(seat))
    pairs = [sum(scores) / 2.0 for _, scores in sorted(duel.items()) if len(scores) == 2]
    if not pairs or not control_games:
        return None, {}
    score = sum(pairs) / len(pairs)
    error = statistics.stdev(pairs) / math.sqrt(len(pairs)) if len(pairs) > 1 else 0.0
    r_plain = control_repeated / control_games
    contempt_wins = score - 2.0 * error > 0.5
    value = 0.0 if contempt_wins else r_plain
    detail = {
        "rung": rung,
        "pairs": len(pairs),
        "score": round(score, 4),
        "error": round(error, 4),
        "contempt_wins": contempt_wins,
        "r_duel": round(duel_repeated / duel_games, 4) if duel_games else None,
        "r_plain": round(r_plain, 4),
        "control_games": control_games,
    }
    return value, detail


def waiting_violations(raw: dict[str, float]) -> list[str]:
    """The waiting gate's refusal, or nothing; silent where the run has no schedule."""
    value = raw.get(FORCED_REPETITION)
    if value is None:
        return []
    failure = WAITING_GATE.check(value)
    return [failure] if failure else []
