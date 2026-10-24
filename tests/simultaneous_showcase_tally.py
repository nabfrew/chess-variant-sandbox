# compute-lease: light
"""What the rounds of a game in rounds did, counted off the core (Q918e).

`docs/archive/plan-simultaneous.md`, *Q918e, pre-registered*: the tally the showcase games' clauses
are graded on. It replays rounds through `Engine.apply_round_with_phase` and counts the
endings, the seat, the collisions by the two pieces' types and their outcome, the swaps, the
void moves and the initiative ties. It decides nothing about a round itself (CLAUDE.md, rule
3): which square two moves meet on, who wins and what is void are the core's answers.

``tests/test_showcases_q918e.py`` imports `RoundTally` for its random-play clauses. Run as a
script it reads measured runs:

    uv run python tests/simultaneous_showcase_tally.py archive/watering-hole.json runs/5c0b017473f5-0 ...

and prints the tally of the graded games (no `random` seat) and of all of them.
"""

from __future__ import annotations

import argparse
import collections
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import (
    MOVE_DROP,
    MOVE_PROMOTION,
    move_from,
    move_promotion,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState

COLLISION, VOID, INITIATIVE, BARRED = 0, 1, 2, 3
# Schema 28 (Q943a): an ordered round (the faster player), a slower move lost with its piece,
# a slower move unplayed because the faster ended the game (`docs/archive/plan-speed.md` §4.2).
ORDER, LOST, UNPLAYED = 5, 6, 7
BOTH_LOST = -1


def piece_at(engine: Engine, state: GameState, move: int) -> str:
    """The piece a move brings to its square, as it arrives: its id, ``P=Q`` for a pawn that
    promotes on the way (it meets as the queen, §4 step 3), ``@N`` for a dropped knight."""
    ids = engine.spec.piece_ids
    if move_type(move) == MOVE_DROP:
        return "@" + ids[move_promotion(move)]
    mover = ids[engine.spec.type_of(state.board[move_from(move)])]
    if move_type(move) == MOVE_PROMOTION:
        return f"{mover}={ids[move_promotion(move)]}"
    return mover


def is_swap(first: int, second: int) -> bool:
    """Two moves onto each other's origin, read off the words (§2.4: two chases)."""
    if MOVE_DROP in (move_type(first), move_type(second)):
        return False
    return move_to(first) == move_from(second) and move_to(second) == move_from(first)


@dataclass
class Collision:
    """One collision: the two pieces' ids in seat order and the winner (-1: both lost)."""

    first: str
    second: str
    winner: int


@dataclass
class RoundTally:
    """Counts over any number of games, added one round at a time."""

    games: int = 0
    plies: int = 0
    rounds: int = 0
    swaps: int = 0
    voids: int = 0
    initiative_ties: int = 0
    # Schema 28 (Q943c, `docs/archive/plan-speed.md` §6.2): Q943b's four tallies, as
    # `analysis.SPEED_COUNTS` counts them, and the games an ordered round ended.
    ordered: int = 0
    slower_void: int = 0
    slower_lost: int = 0
    ended_after_faster: int = 0
    # Decided games whose last round was ordered and ended with its faster move, won by the
    # faster player (§6.4: a royal, or the hole, taken by a move nothing could answer).
    won_by_faster: int = 0
    games_with_collision: int = 0
    collisions: list[Collision] = field(default_factory=list)
    endings: collections.Counter = field(default_factory=collections.Counter)
    seat: collections.Counter = field(default_factory=collections.Counter)
    # Per game, the winner of each tie the initiative decided, in order.
    tie_winners: list[list[int]] = field(default_factory=list)

    def round(self, engine: Engine, state: GameState, first: int, second: int) -> tuple[
        GameState, list[tuple[int, int, int]]
    ]:
        """Play one round through the core, count it, and return the position after it."""
        self.rounds += 1
        after, _phase, events = engine.apply_round_with_phase(state, first, second)
        ordered = any(event == ORDER for event, _square, _player in events)
        # Two words onto each other's origins in an ordered round are no swap: the faster
        # took the slower where it stood (`docs/archive/plan-speed.md` §2.5 B).
        self.swaps += not ordered and is_swap(first, second)
        for event, _square, player in events:
            if event == ORDER:
                self.ordered += 1
            elif event == LOST:
                self.slower_lost += 1
            elif event == UNPLAYED:
                self.ended_after_faster += 1
            elif event == VOID and ordered:
                self.slower_void += 1
            if event == COLLISION:
                self.collisions.append(
                    Collision(piece_at(engine, state, first), piece_at(engine, state, second), player)
                )
            elif event == VOID:
                self.voids += 1
            elif event == INITIATIVE:
                self.initiative_ties += 1
                self.tie_winners[-1].append(player)
        return after, list(events)

    def game(self, engine: Engine, moves: Sequence[int], reason: str, winner: int | None) -> None:
        """Replay one recorded game, round by round, and count its ending."""
        self.start_game()
        state = engine.initial_state()
        before = len(self.collisions)
        events: list[tuple[int, int, int]] = []
        for index in range(0, len(moves), 2):
            state, events = self.round(engine, state, moves[index], moves[index + 1])
        self.won_by_faster += winner is not None and any(
            event == UNPLAYED for event, _square, _player in events
        ) and any(event == ORDER and player == winner for event, _square, player in events)
        self.end_game(len(moves), reason, winner, len(self.collisions) > before)

    def start_game(self) -> None:
        self.tie_winners.append([])

    def end_game(self, plies: int, reason: str, winner: int | None, collided: bool) -> None:
        self.games += 1
        self.plies += plies
        self.games_with_collision += collided
        self.endings[(reason, winner)] += 1
        self.seat["draw" if winner is None else winner] += 1

    def pairs(self) -> collections.Counter:
        return collections.Counter(
            (c.first, c.second, {0: "p0 wins", 1: "p1 wins", BOTH_LOST: "both lost"}[c.winner])
            for c in self.collisions
        )

    def report(self, label: str) -> str:
        n = max(1, self.games)
        rounds = max(1, self.rounds)
        decided = self.seat[0] + self.seat[1]
        lines = [
            f"== {label}: {self.games} games, mean plies {self.plies / n:.1f}",
            f"  seat: p0 {self.seat[0]}, p1 {self.seat[1]}, draw {self.seat['draw']}; "
            f"p0 share of decided {self.seat[0] / max(1, decided):.3f}",
            "  endings: " + ", ".join(
                f"{reason}/{'draw' if who is None else 'p' + str(who)} {count} ({count / n:.3f})"
                for (reason, who), count in self.endings.most_common()
            ),
            f"  rounds {self.rounds}: collisions {len(self.collisions)} "
            f"({len(self.collisions) / rounds:.3f}), in {self.games_with_collision} games "
            f"({self.games_with_collision / n:.3f}); swaps {self.swaps} ({self.swaps / rounds:.3f}); "
            f"void moves {self.voids}; initiative ties {self.initiative_ties} "
            f"({self.initiative_ties / rounds:.3f})",
        ]
        if self.ordered:
            ordered = max(1, self.ordered)
            lines.append(
                f"  ordered rounds {self.ordered} ({self.ordered / rounds:.3f} of rounds): slower "
                f"void {self.slower_void} ({self.slower_void / ordered:.3f}), lost "
                f"{self.slower_lost} ({self.slower_lost / ordered:.3f}), ended after the faster "
                f"{self.ended_after_faster} ({self.ended_after_faster / ordered:.3f})"
            )
            lines.append(
                f"  decided by the faster move, the slower unplayed: {self.won_by_faster} "
                f"({self.won_by_faster / max(1, decided):.3f} of decided games)"
            )
        lines += [
            "  collisions by (p0 piece, p1 piece, outcome):",
        ]
        lines += [f"    {a} vs {b}: {who} x{count}" for (a, b, who), count in self.pairs().most_common()]
        return "\n".join(lines)


def tally_runs(
    engine: Engine, runs: Iterable[Path], graded: bool, among: frozenset[str] | None = None
) -> RoundTally:
    """The tally of every recorded game in ``runs``; ``graded`` leaves out any with `random`,
    and ``among`` (a band, Q943c) keeps only games whose two seats are both in it."""
    from chess_variant_sandbox.tournament.record import read_records

    tally = RoundTally()
    for run in runs:
        for record in read_records(Path(run) / "games.jsonl"):
            if graded and "random" in record.agents:
                continue
            if among is not None and not set(record.agents) <= among:
                continue
            tally.game(engine, record.moves, record.reason, record.winner)
    return tally


def main() -> None:
    from chess_variant_sandbox.spec import load_spec

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument(
        "--band", action="append", default=[], metavar="AGENTS",
        help="a comma-separated band: also tally the games whose two seats are both in it",
    )
    args = parser.parse_args()
    engine = Engine(load_spec(args.spec))
    print(tally_runs(engine, args.runs, graded=True).report("graded games (no random seat)"))
    print(tally_runs(engine, args.runs, graded=False).report("all games"))
    for band in args.band:
        among = frozenset(band.split(","))
        print(tally_runs(engine, args.runs, graded=True, among=among).report(f"band {band}"))


if __name__ == "__main__":
    main()
