"""Cheap rejection tests, run before anything expensive.

A generation's cost is dominated by the tournament, and most proposals do not deserve
one. A few hundred random playouts cost a fraction of a second in the core, and they
catch the failures that are obvious from any angle: a game nobody can move in, a game
that never ends, a game decided before either player has done anything.

Every gate here has to be cheap enough that running it on a doomed proposal is free
compared to the tournament it prevents. Anything requiring competent play belongs in
the index instead -- random playouts cannot tell a deep game from a shallow one, and a
gate that pretends otherwise will reject good designs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec.plain import plain_problems
from chess_variant_sandbox.spec.schema import SpecError

SMOKE_GAMES = 150
FORCED_RESULT_PLIES = 6

# How much random-play stalling is allowed before a ruleset is called non-terminating.
#
# This was 0.90 and it was rejecting real games. Measured across the twenty-two
# committed rulesets and the labelled mutant corpus, stall rate under *random* play does
# not separate a game from a non-game at all: makruk stalls 87% of the time, chess 79%,
# xiangqi 71%, courier 96% and shatranj 92%, while the deliberately broken uniform-army
# stalls 20% and cramped-board 11%. Random movers are simply bad at forcing mate, and
# they are worse at it the more material is on the board -- so the old threshold was
# closest to firing on exactly the big, slow, rich games most worth keeping.
#
# At 0.98 the gate accepts every real ruleset in the corpus, including the two it used
# to reject, and still catches the case it is actually for: a ruleset where essentially
# nothing ever ends. The broken games it now lets through are caught by the index, which
# is where anything needing competent play belongs.
MAX_RANDOM_STALL = 0.98

# The words each refusal is said in, a visitor's (`docs/plan-understanding.md` D42-D44): a
# *side*, *turns* and *the rules*, never a seat, plies or a compile. The stall sentence is
# matched by the designer, which appends which counter fired, so its words live here once.
STALLED = "ended on the turn limit or the no-progress rule"


@dataclass
class SmokeResult:
    """The verdict of the cheap gate, with the numbers behind it."""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "reasons": self.reasons, "stats": self.stats}


def smoke_test(
    spec: dict[str, Any],
    games: int = SMOKE_GAMES,
    seed: int = 0,
    time_budget: float = 20.0,
    workers: int = 0,
) -> SmokeResult:
    """Play fast random games and check the ruleset is not obviously broken.

    The playouts run in the Rust core, in parallel, with one independent stream per
    game rather than one shared generator drawn from in sequence. That is what makes
    the verdict independent of how the games were spread across threads -- the same
    seed produces the same set of games at any worker count -- but it does mean the
    games themselves differ from the ones the serial Python gate played, so the
    measured rates here are not bit-comparable with numbers recorded before the port.

    `time_budget` is refuse-or-whole: when the clock stops the core before all `games`
    finished, the ruleset is refused with a reason that says so, and the games that did
    finish are not graded. A verdict is always over the whole fixed sample.
    """
    reasons: list[str] = []

    try:
        engine = Engine(spec)
    except SpecError as error:
        # In the designer's words, as the editor lists them (`spec.plain`, D44).
        said = " ".join(plain_problems(error.errors, spec))
        return SmokeResult(passed=False, reasons=[f"the rules don't fit together: {said}"])
    except NotImplementedError as error:
        # `bridge.rust_supports` holding an intermediate state shut: the compiler knows a
        # rule the core does not yet. A candidate the core cannot play fails the first
        # rung by name rather than ending the run, so a mechanical operator that reaches
        # the rule costs a draw during the refusal window and nothing more.
        return SmokeResult(
            passed=False,
            reasons=[
                "the computer can't play one of these rules yet: "
                + str(error).removeprefix("the core cannot play this ruleset: ")
            ],
        )

    state = engine.initial_state()
    opening_moves = len(engine.legal_moves(state))
    if opening_moves == 0:
        return SmokeResult(
            passed=False,
            reasons=["the first player has no legal move from the starting position"],
            stats={"opening_moves": 0},
        )

    threads = workers if workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    playouts = engine.core.random_playouts(games, seed, time_budget, threads)

    plies: list[int] = []
    branching_total = 0
    branching_samples = 0
    outcomes = {"win_first": 0, "win_second": 0, "draw": 0}
    reasons_seen: dict[str, int] = {}

    for game_plies, winner, reason, sampled, samples in playouts:
        plies.append(game_plies)
        branching_total += sampled
        branching_samples += samples
        reasons_seen[reason] = reasons_seen.get(reason, 0) + 1
        if winner is None:
            outcomes["draw"] += 1
        else:
            outcomes["win_first" if winner == 0 else "win_second"] += 1
    played = len(playouts)

    # Refuse-or-whole (`docs/reference/wall-clock-budget.md`, Q451). The core abandons the
    # games not yet started when its clock runs out, so a cut sample is a function of the
    # machine's speed and, above one thread, of which games the scheduler reached first.
    # Grading the remainder let a slow box pass or fail a design on 90 games where a fast
    # one graded 150. The clock may decide *whether* there is a verdict, never what it is.
    if played < games:
        finished = "none" if played == 0 else f"only {played} of {games}"
        return SmokeResult(
            passed=False,
            reasons=[
                "random games of these rules take too long to finish, so they couldn't be "
                f"checked in the {time_budget:g} seconds allowed ({finished} finished)"
            ],
            stats={"opening_moves": float(opening_moves), "games": float(played)},
        )

    mean_plies = sum(plies) / len(plies)
    mean_branching = branching_total / max(branching_samples, 1)
    stall_rate = (
        reasons_seen.get("ply_limit", 0) + reasons_seen.get("no_progress", 0)
    ) / played
    forced_rate = sum(1 for p in plies if p <= FORCED_RESULT_PLIES) / played
    decided = outcomes["win_first"] + outcomes["win_second"]

    if mean_branching < 1.5:
        reasons.append(
            f"a player has {mean_branching:.2f} moves to choose from on average; there is "
            "nothing to decide"
        )
    if stall_rate > MAX_RANDOM_STALL:
        reasons.append(
            f"{stall_rate:.0%} of random games {STALLED}: the pieces can't force a result"
        )
    if decided == 0 and played >= 30:
        # Distinct from the stall check above, and it took a test to notice: a game can
        # reach a definite ending every time -- stalemate, repetition -- and still have
        # no reachable win. Random play is bad at winning, but across dozens of games
        # it will stumble into a victory if one exists at all.
        reasons.append(
            f"none of {played} random games had a winner; from this start, the way to win "
            "may be out of reach"
        )
    if forced_rate > 0.5:
        reasons.append(
            f"{forced_rate:.0%} of games end within {FORCED_RESULT_PLIES} turns even when "
            "both sides move at random: the game is decided before it begins"
        )
    if decided >= 20:
        # Under random play the two seats should be close to symmetric. A wide gap
        # means the first player's advantage is structural rather than strategic --
        # which no amount of skilful play will fix.
        #
        # The threshold is on *decisive* games, not games played. Gating on the latter
        # rejected the dull control off a single decisive result out of sixty, where
        # the skew is 100% by construction and means nothing at all.
        skew = abs(outcomes["win_first"] - outcomes["win_second"]) / decided
        if skew > 0.85:
            reasons.append(
                f"one side wins {skew:.0%} more often than the other even when both move "
                "at random: the starting position itself is lopsided"
            )

    return SmokeResult(
        passed=not reasons,
        reasons=reasons,
        stats={
            "games": float(played),
            "opening_moves": float(opening_moves),
            "mean_plies": round(mean_plies, 2),
            "mean_branching": round(mean_branching, 2),
            "stall_rate": round(stall_rate, 3),
            "forced_rate": round(forced_rate, 3),
            "draw_rate": round(outcomes["draw"] / played, 3),
        },
    )
