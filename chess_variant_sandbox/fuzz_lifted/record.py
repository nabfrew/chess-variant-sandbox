"""A played case, as the engine wrote it: every position, the move or round made on it, the
moves the engine offered there, and what the wild-piece phase did.

The game is played by the core alone: `Engine.legal_moves` (or `legal_moves_for` in a round)
offers, a seeded draw picks, `Engine.apply_with_phase` (or `apply_round_with_phase`) answers.
Nothing here decides a move's legality or its effect, so a check reading a `Record` reads
the engine's own answers (CLAUDE.md rule 3). The draw is random, tilted toward the rare moves (`SPECIAL_SHARE`): a random player reaches the
odd corners of a rule faster than a strong one, which is what a fuzz wants.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import (
    MOVE_FOREIGN,
    MOVE_NORMAL,
    move_from,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState, Outcome, PhaseStep

# How often the draw is taken from the special moves alone (any kind but a plain step: a castle,
# a drop, a buy, a promotion, a sweep, a transit, a duck move, a command) and how often from the
# captures. A uniform draw castled in three cases of two thousand, so a castling check judged
# nothing; the tilt makes each rule's own events common, at no cost to determinism.
SPECIAL_SHARE = 0.35
CAPTURE_SHARE = 0.25

# How far a case is played. Short games keep a case cheap; forty plies reach a capture, a
# promotion on a six-rank board and most rules' second use.
PLIES = 40


def _draw(rng: random.Random, board: bytes, moves: list[int]) -> int:
    """A seeded pick: a third of the time among the special moves (any kind but a plain step,
    or a command), a quarter among the captures, else among all, falling through where a
    tier is empty."""
    roll = rng.random()
    if roll < SPECIAL_SHARE:
        special = [m for m in moves if move_type(m) != MOVE_NORMAL or m & MOVE_FOREIGN]
        if special:
            return rng.choice(special)
    elif roll < SPECIAL_SHARE + CAPTURE_SHARE:
        captures = [m for m in moves if board[move_to(m)] and move_to(m) != move_from(m)]
        if captures:
            return rng.choice(captures)
    return rng.choice(moves)


@dataclass(frozen=True)
class Step:
    """One move, or one round, and the positions either side of it."""

    index: int
    before: GameState
    moves: tuple[int, ...]  # one for a move; (player 0's, player 1's) for a round
    after: GameState
    phase: tuple[PhaseStep, ...]
    round: bool
    offered: tuple[int, ...] = ()  # the side to move's legal moves (empty for a round)

    @property
    def mover(self) -> int:
        """The side that moved (-1 for a round, where both did)."""
        return -1 if self.round else self.before.side_to_move

    @property
    def phase_targets(self) -> set[int]:
        """The squares the wild-piece phase stepped onto or came back on this step."""
        return {step.target for step in self.phase}


@dataclass
class Record:
    """A played case."""

    spec: dict[str, Any]
    engine: Engine
    steps: list[Step] = field(default_factory=list)
    outcome: Outcome | None = None

    @property
    def final(self) -> GameState:
        return self.steps[-1].after if self.steps else self.engine.initial_state()

    def positions(self) -> list[GameState]:
        """Every position the game passed through, the opening one first."""
        if not self.steps:
            return [self.engine.initial_state()]
        return [self.steps[0].before] + [step.after for step in self.steps]


def play(spec: dict[str, Any], seed: int, plies: int = PLIES, engine: Engine | None = None) -> Record:
    """Play ``spec`` for at most ``plies`` steps with a seeded uniform draw."""
    engine = engine or Engine(spec)
    rng = random.Random(seed ^ 0x11F7)
    state = engine.initial_state()
    record = Record(spec=spec, engine=engine)
    for index in range(plies):
        if engine.is_round(state):
            if engine.terminal(state) is not None:
                break
            first = engine.legal_moves_for(state, 0)
            second = engine.legal_moves_for(state, 1)
            if not first or not second:
                break
            pair = (_draw(rng, state.board, first), _draw(rng, state.board, second))
            after, phase, _ = engine.apply_round_with_phase(state, *pair)
            record.steps.append(Step(index, state, pair, after, phase, True))
        else:
            offered = engine.legal_moves(state)
            if not offered or engine.terminal(state, offered) is not None:
                break
            move = _draw(rng, state.board, offered)
            after, phase = engine.apply_with_phase(state, move)
            record.steps.append(
                Step(index, state, (move,), after, phase, False, tuple(offered))
            )
        state = after
    record.outcome = engine.terminal(state)
    return record
