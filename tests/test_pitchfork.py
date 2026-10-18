"""Q876: `pitchfork`, a game built around `rules.npcs[].immune`, in place of `wolves-immune`.

`docs/archive/plan-library-audit.md` (*`pitchfork`: wolves only a pawn may take*) pre-registered these
clauses before this file was written; the numbering is the document's. `wolves-immune` spelled
`wolves` with the key and played it move for move; here the key says that only a pawn may take
a wolf, and the wolves stand in pairs on the centre rings, where the pawns meet them at once.

Every test drives the core through `Engine` and asserts what the core did. The only choice of
a move in Python is the seeded random walk of clause 5, the same probe as Q787's and Q870's.
`hunt` is also how the plan's measured clauses M3 to M5 are read from the persisted games.
"""

from __future__ import annotations

import copy
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import WILD, compile_spec
from chess_variant_sandbox.engine.moves import MOVE_DROP, move_from, move_to, move_type
from chess_variant_sandbox.spec import load_spec, validate_spec
from chess_variant_sandbox.spec.schema import _schema_ten_uses

ROOT = Path(__file__).resolve().parents[1]
GAMES = 6  # seeds 0 to 5, as pre-registered
PLIES = 400
# The five arrays the mechanics hash seeds, which differ between two spellings of one game.
ZOBRIST = frozenset(
    {"zobrist_pieces", "zobrist_side", "zobrist_ep", "zobrist_colour", "zobrist_turn"}
)
# Taken when this file was written (clause 6): a regression pin, not a prediction.
LADDER = {1: 16, 2: 256, 3: 4620}
# How long a wolf's taker is watched for the walk to eat it (M5).
WATCH = 4


@dataclass
class Hunt:
    """What happened to the wolves in some games, read by replaying their moves."""

    games: int = 0
    plies: int = 0
    plies_with_a_wolf: int = 0
    taken_by: Counter[str] = field(default_factory=Counter)
    games_with_a_capture: int = 0
    left: int = 0
    takers_eaten: int = 0

    @property
    def taken(self) -> int:
        return sum(self.taken_by.values())


def _wolves(engine: Engine, state: Any) -> list[int]:
    compiled = engine.spec
    return [sq for sq, v in enumerate(state.board) if v and compiled.owner_of(v) == WILD]


def hunt(engine: Engine, games: Iterable[list[int]], into: Hunt | None = None) -> Hunt:
    """Replay each move list from the opening and count the wolves taken, by whom.

    A wolf is taken when a move's target holds one, and the taker is the mover's type. A
    taker is eaten when, within `WATCH` plies of its capture and without having moved, its
    square holds a wolf: the walk stepped onto it.
    """
    compiled = engine.spec
    tally = into if into is not None else Hunt()
    for moves in games:
        tally.games += 1
        state = engine.initial_state()
        took = False
        watching: list[tuple[int, int, int]] = []  # (square, ply of the capture, piece code)
        for ply, move in enumerate(moves, start=1):
            before = state
            if _wolves(engine, before):
                tally.plies_with_a_wolf += 1
            tally.plies += 1
            target = move_to(move)
            dropped = move_type(move) == MOVE_DROP
            victim = 0 if dropped else before.board[target]
            state = engine.apply(before, move)
            if victim and compiled.owner_of(victim) == WILD:
                mover = before.board[move_from(move)]
                tally.taken_by[compiled.piece_names[compiled.type_of(mover)]] += 1
                took = True
                # The walk has already stepped, so the check below sees an eat at once too.
                watching.append((target, ply, mover))
            kept = []
            for square, at, piece in watching:
                standing = state.board[square]
                if standing and compiled.owner_of(standing) == WILD:
                    tally.takers_eaten += 1
                elif standing == piece and ply - at < WATCH:
                    kept.append((square, at, piece))
            watching = kept
        tally.games_with_a_capture += took
        tally.left += len(_wolves(engine, state))
    return tally


def _random_game(engine: Engine, seed: int) -> list[int]:
    rng = random.Random(seed)
    state = engine.initial_state()
    moves: list[int] = []
    for _ in range(PLIES):
        if engine.terminal(state) is not None:
            break
        legal = engine.legal_moves(state)
        if not legal:
            break
        move = rng.choice(legal)
        moves.append(move)
        state = engine.apply(state, move)
    return moves


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return load_spec(ROOT / "archive" / "pitchfork.json")


@pytest.fixture(scope="module")
def engine(spec: dict[str, Any]) -> Engine:
    return Engine(spec)


def _reread(spec: dict[str, Any], immune: list[str] | None) -> dict[str, Any]:
    """`spec` with every wolf's `immune` set to ``immune``, or deleted for None."""
    other = copy.deepcopy(spec)
    for npc in other["rules"]["npcs"]:
        if immune is None:
            del npc["immune"]
        else:
            npc["immune"] = list(immune)
    assert validate_spec(other) == []
    return other


def _takers_of(engine: Engine, placement: dict[str, tuple[str, int]], target: str) -> list[str]:
    """The type of every piece with a legal move onto ``target``, first player to move."""
    state = engine.make_state(placement, side_to_move=0)
    square = engine._square_index(target)
    compiled = engine.spec
    return sorted(
        compiled.piece_ids[compiled.type_of(state.board[move_from(move)])]
        for move in engine.legal_moves(state)
        if move_to(move) == square
    )


class TestWhoMayTakeAWolf:
    """Clause 1: the word's three readings on one position."""

    POSITION = {
        "e4": ("W", WILD),
        "d3": ("P", 0),
        "g3": ("N", 0),
        "h1": ("B", 0),
        "e1": ("R", 0),
        "a4": ("Q", 0),
        "e5": ("K", 0),
        "h8": ("K", 1),
    }

    def test_only_the_pawn(self, engine: Engine) -> None:
        assert _takers_of(engine, self.POSITION, "e4") == ["P"]

    def test_with_the_word_deleted_every_piece(self, spec: dict[str, Any]) -> None:
        loose = Engine(_reread(spec, None))
        assert _takers_of(loose, self.POSITION, "e4") == sorted("PNBRQK")

    def test_with_wolves_word_only_the_knight(self, spec: dict[str, Any]) -> None:
        knights = Engine(_reread(spec, ["N"]))
        assert _takers_of(knights, self.POSITION, "e4") == ["N"]

    def test_it_is_written_with_the_word(self, spec: dict[str, Any]) -> None:
        assert _schema_ten_uses(spec) == {"rules.npcs[].immune"}


class TestTheSpelling:
    """Clause 2: one key where the long form needs five `captures` lists."""

    def test_the_tables_are_the_five_lists(self, spec: dict[str, Any]) -> None:
        long = _reread(spec, None)
        players = ["P", "N", "B", "R", "Q", "K"]
        for piece in long["pieces"]:
            if piece["id"] in ("N", "B", "R", "Q", "K"):
                piece["captures"] = list(players)
        assert validate_spec(long) == []
        mine = canonical_tables(compile_spec(spec))
        theirs = canonical_tables(compile_spec(long))
        assert mine.keys() == theirs.keys()
        assert {key for key in mine if mine[key] != theirs[key]} <= ZOBRIST


class TestTheOpening:
    """Clause 3: the wolves stand where the pawns' advance meets them."""

    def test_sixteen_moves(self, engine: Engine) -> None:
        assert len(engine.legal_moves(engine.initial_state())) == 16

    def test_a_centre_pawn_that_double_steps_is_eaten(self, engine: Engine) -> None:
        state = engine.initial_state()
        d2, d4 = engine._square_index("d2"), engine._square_index("d4")
        (move,) = [
            m for m in engine.legal_moves(state) if (move_from(m), move_to(m)) == (d2, d4)
        ]
        after = engine.apply(state, move)
        assert d4 in _wolves(engine, after)
        compiled = engine.spec
        pawn = compiled.piece_ids.index("P")
        pawns = [
            v for v in after.board
            if v and compiled.owner_of(v) == 0 and compiled.type_of(v) == pawn
        ]
        assert len(pawns) == 7


class TestThePrice:
    """Clause 4: the second wolf of a pair eats whoever took the first."""

    PAIR = {"b4": ("W", WILD), "c5": ("W", WILD), "h1": ("K", 0), "h8": ("K", 1)}

    def _take(self, engine: Engine, pawn_on: str, wolf_on: str) -> Any:
        state = engine.make_state({**self.PAIR, pawn_on: ("P", 0)}, side_to_move=0)
        origin, target = engine._square_index(pawn_on), engine._square_index(wolf_on)
        (move,) = [
            m for m in engine.legal_moves(state) if (move_from(m), move_to(m)) == (origin, target)
        ]
        return engine.apply(state, move)

    def test_taking_the_leader_costs_the_pawn(self, engine: Engine) -> None:
        after = self._take(engine, "d4", "c5")
        c5 = engine._square_index("c5")
        compiled = engine.spec
        # The follower stepped onto c5 and ate the pawn: one wolf left, and no pawn.
        assert _wolves(engine, after) == [c5]
        assert not any(
            v and compiled.owner_of(v) == 0 and compiled.piece_ids[compiled.type_of(v)] == "P"
            for v in after.board
        )

    def test_taking_the_follower_leaves_it_a_move(self, engine: Engine) -> None:
        after = self._take(engine, "a3", "b4")
        b4 = engine._square_index("b4")
        compiled = engine.spec
        assert compiled.owner_of(after.board[b4]) == 0
        assert compiled.piece_ids[compiled.type_of(after.board[b4])] == "P"
        assert _wolves(engine, after) == [engine._square_index("d4")]


class TestTheHunt:
    """Clause 5: in play, only pawns take wolves; without the word, others do too."""

    def test_random_games_hunt_with_pawns_only(self, engine: Engine) -> None:
        tally = hunt(engine, (_random_game(engine, seed) for seed in range(GAMES)))
        assert set(tally.taken_by) == {"pawn"}, tally.taken_by
        assert tally.games_with_a_capture >= 4, tally

    def test_without_the_word_other_pieces_hunt(self, spec: dict[str, Any]) -> None:
        loose = Engine(_reread(spec, None))
        tally = hunt(loose, (_random_game(loose, seed) for seed in range(GAMES)))
        assert set(tally.taken_by) - {"pawn"}, tally.taken_by


class TestTheLadder:
    """Clause 6: taken, not predicted."""

    @pytest.mark.parametrize("depth", sorted(LADDER))
    def test_the_ladder(self, engine: Engine, depth: int) -> None:
        assert engine.perft(engine.initial_state(), depth) == LADDER[depth]
