"""The rung between the smoke gate and the full tournament.

The evaluation ladder went validate -> smoke -> full roster, and the last step is two
orders of magnitude more expensive than the one before it. That was affordable while
every candidate cost a model call anyway. It stops being affordable the moment the loop
can generate hundreds of candidates for nothing (`loop/operators.py`), because the
tournament then dominates everything and most of what it plays does not deserve it.

So: a two-budget, few-games screen that answers one question -- *can a searching player
win this at all?* -- and then gets out of the way. It is not a small index. It has no
opinion about depth, elegance or variety, and it must not acquire one: everything that
needs competent play at more than one strength belongs in the index, and a screen that
guesses at those would reject good designs for being slow to reveal themselves. That is
exactly the mistake the random-playout gate made with shatranj and courier.

Two budgets rather than one because `_require_measurable_ladder` is right: a single
budget cannot say anything about search at all. Two is the smallest roster that can.

**Both thresholds are the loosest that keep every committed ruleset**, and that is the
whole design. See the numbers on the constants. A screen tuned any tighter would be the
shatranj mistake again, one rung further up the ladder.

**It no longer rejects `dull`, and the threshold was deliberately not moved.** Phase C
gave the search bounds, quiescence and ordering beyond captures, and the stronger player
now wins one game of eight in the shallow control where it used to win none -- so the
screen's own question, *can a searching player win this at all*, has truthfully changed
its answer. Raising `MIN_DECISIVE` to two would reject `dull` again and would put
`makruk` and `shatranj`, which measure exactly two, on the line with no margin at all.
That is the mistake this module exists not to make. `dull` is still gated out by the
index on `decisive_rate`, which is where a judgement needing competent play belongs, and
this rung still catches the thing it is for: `rps-hierarchy-chess`, where nothing ever
ends, is still refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from chess_variant_sandbox.agents.base import parse_agent
from chess_variant_sandbox.metrics.components import plays_in_rounds
from chess_variant_sandbox.spec.schema import is_spy_game
from chess_variant_sandbox.tournament import TournamentConfig, run_tournament

# The cheapest pair of budgets that still constitutes a ladder. Deliberately far below
# the rated roster: this asks whether search does anything, not how much.
SCREEN_AGENTS = ("minimax:nodes=100", "minimax:nodes=400")

# Per ordered pairing, so four pairings and eight games. Enough that "nothing ever ended"
# is a statement about the ruleset rather than about one unlucky game.
SCREEN_GAMES = 2

# A ruleset where no searching player ever reaches a result is either unwinnable or
# unplayably slow to win, and either way the full roster spends minutes discovering it.
#
# Measured over all twenty-eight committed rulesets at this roster, under the phase-C
# search: every real game produces at least two decisive games out of eight, the closest
# being makruk and shatranj at exactly two; `rps-hierarchy-chess`, which cannot be won by
# anybody, produces none. The margin between the tightest real game and the threshold is
# one game, and the check still catches the thing it is for.
#
# `dull` produces one and therefore passes, where before phase C it produced none. The
# threshold stays at one anyway: two would reject it and would leave makruk and shatranj
# sitting exactly on the line, which is a threshold fitted between adjacent points and is
# the shatranj mistake in miniature.
#
# **Round games share the floor of one** (Q944b, `docs/reference/round-roster-budgets.md`
# §7.3). Screened by `matrix:nodes=1600` and `6400` (`round_roster`), the three Gardner round
# drafts pass with p about 0.8 (`runs/Q944b/screen_odds.txt`), sometimes on one decisive game
# of eight, as makruk does on the alternating side; a round floor of two would refuse games
# like these about half the time. Whether they draw too much is the index's to say. A spy
# game (schema 27) is screened by the same floor.
MIN_DECISIVE = 1

# Mirrors the index's own quick-kill gate rather than inventing a second standard.
#
# Set loose on purpose. Measured here, atomic -- a real game that simply ends fast --
# reaches 0.62, and the deliberately broken `bastion` reaches 0.75. At eight games those
# are one game apart, so a threshold fitted between them would be reading sampling noise
# and would eventually throw out a real design for ending quickly. Bastion is left to the
# index, which sees it on a rated roster with the games to support the claim; this rung
# only catches a ruleset that is over before it starts in nearly every game.
MAX_QUICK_KILL = 0.75
QUICK_KILL_PLIES = 12


# Schema 26 (Q932): `minimax:` is refused on a game played in rounds, and `matrix:` on one
# that alternates (`docs/archive/plan-simultaneous.md` §6.2). A round game is played by the same
# roster with `matrix:nodes=F*N` wherever the alternating one has `minimax:nodes=N`, as
# `corpus.ROUND_AGENTS` mirrors the canonical ladder: `random` and `greedy` play rounds
# already (greedy reads the round as a matrix game). Only a bare node budget maps; an
# objective (`h=`, `fit=`, `values=`, `net=`) is refused on these games, and so is `mcts:`,
# so a roster holding one has no round counterpart.
#
# F is `ROUND_BUDGET_FACTOR`, the one rule for what a budget means in rounds (Q944a,
# `docs/reference/round-roster-budgets.md` §5-§6). `matrix:nodes=N` always finishes pass 1,
# greedy's whole one-round matrix (about 200 cells on Gardner, 700 on the watering hole),
# and plays greedy's move in every round whose matrix has N cells or more. At the same
# budget the bottom four rungs were greedy under other names; sixteen is the smallest power
# of two at which the bottom rung finishes a pass-2 cell in at least half the side-rounds of
# every library round game. Chosen by move counts, never by a score it produces.
ROUND_BUDGET_FACTOR = 16
_ROUND_SAFE = ("random", "greedy")
# The options that name a stored objective, which a spy game refuses as a round game does.
_OBJECTIVES = ("h", "fit", "values", "net")
_MINIMAX_BUDGET = re.compile(r"minimax:nodes=(\d+)")


def round_roster(agents: tuple[str, ...]) -> tuple[str, ...]:
    """The players a game in rounds is played by in place of ``agents``:
    `minimax:nodes=N` becomes `matrix:nodes=N*ROUND_BUDGET_FACTOR`, and a round player
    already named passes unchanged.

    Raises ``ValueError`` naming the first player with no round counterpart.
    """
    mapped: list[str] = []
    for agent in agents:
        budget = _MINIMAX_BUDGET.fullmatch(agent)
        if budget:
            mapped.append(f"matrix:nodes={int(budget.group(1)) * ROUND_BUDGET_FACTOR}")
        elif agent in _ROUND_SAFE or agent.startswith("matrix:"):
            mapped.append(agent)
        else:
            raise ValueError(f"{agent} cannot play a game in rounds")
    return tuple(mapped)


def spy_roster(agents: tuple[str, ...]) -> tuple[str, ...]:
    """The players a spy game (schema 27) is played by in place of ``agents``: the same
    names, since `random`, `greedy` and `minimax:nodes=N` play every phase of one
    (`docs/plan-spy.md` §3.3; Q932's mapping is not needed, `minimax:` plays the rounds
    itself). Only a bare node budget plays: a stored objective, `matrix:` and `mcts:` are
    refused on a spy game.

    Raises ``ValueError`` naming the first player that cannot play one."""
    for agent in agents:
        config = parse_agent(agent)
        objective = [key for key in _OBJECTIVES if key in config.options]
        if config.kind not in ("random", "greedy", "minimax") or objective:
            raise ValueError(f"{agent} cannot play a spy game")
    return agents


def roster_for(spec: dict[str, Any], agents: tuple[str, ...]) -> tuple[str, ...]:
    """``agents`` for a game that alternates, ``round_roster(agents)`` for one in rounds,
    ``spy_roster(agents)`` for a spy game (schema 27; the screen, the full roster, the
    analysis pass and `measure` all read one since Q935d, `docs/plan-spy.md` §3.8).

    Raises ``ValueError`` naming the first player that cannot play the game."""
    if is_spy_game(spec):
        return spy_roster(agents)
    return round_roster(agents) if plays_in_rounds(spec) else agents


@dataclass
class ScreenResult:
    """The verdict of the middle rung, with the numbers behind it."""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "reasons": self.reasons, "stats": self.stats}


def screen(
    spec: dict[str, Any],
    agents: tuple[str, ...] = SCREEN_AGENTS,
    games_per_pairing: int = SCREEN_GAMES,
    seed: int = 0,
    workers: int = 0,
) -> ScreenResult:
    """Play a handful of games between two small searches and report what happened.

    A game in rounds is played by `round_roster(agents)`: `matrix:nodes=1600` and
    `matrix:nodes=6400` by default, the two budgets at `ROUND_BUDGET_FACTOR` times, since
    below about a round's matrix a round player is greedy. A round counts two plies, so
    `QUICK_KILL_PLIES` is six moves a player either way. A spy game is played by the same
    names (`spy_roster`), which play every phase of it.
    """
    try:
        players = roster_for(spec, agents)
    except ValueError as error:
        return ScreenResult(passed=False, reasons=[f"no computer player here can test it: {error}"])
    result = run_tournament(
        spec,
        TournamentConfig(
            agents=players,
            games_per_pairing=games_per_pairing,
            seed=seed,
            workers=workers,
        ),
    )
    records = result.records
    if not records:
        return ScreenResult(passed=False, reasons=["no games were played"])

    decisive = sum(1 for record in records if record.winner is not None)
    quick = sum(
        1
        for record in records
        if record.winner is not None and record.plies < QUICK_KILL_PLIES
    )
    stats = {
        "games": float(len(records)),
        "decisive": float(decisive),
        "decisive_rate": round(decisive / len(records), 3),
        "quick_kill_rate": round(quick / len(records), 3),
        "mean_plies": round(sum(r.plies for r in records) / len(records), 1),
    }

    # Said to whoever drew the game, the designer's visitor included
    # (`docs/reference/plain-words.md` §2): a *computer player*, a *turn* for each player's
    # move, and no budget.
    reasons: list[str] = []
    if decisive < MIN_DECISIVE:
        reasons.append(
            f"no computer player won any of {len(records)} games: either nobody can win "
            "this game, or winning it takes a stronger player than these two small ones"
        )
    if stats["quick_kill_rate"] > MAX_QUICK_KILL:
        reasons.append(
            f"{stats['quick_kill_rate'] * 100:.0f}% of games were over within "
            f"{QUICK_KILL_PLIES} turns, which is a trap in the opening rather than a game"
        )
    return ScreenResult(passed=not reasons, reasons=reasons, stats=stats)
