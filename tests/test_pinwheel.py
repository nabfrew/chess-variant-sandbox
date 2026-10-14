"""Q870: `pinwheel`, the hexagon set's game designed around what four variants were kept to show.

`docs/plan-library-audit.md` (*`pinwheel`: the hexagon set's new game, and what it predicts*)
pre-registered these clauses before this file was written; the numbering is the document's.
The game replaces `hex-glinski-turned` and `-turned-mirrored` (`rotate6`, `dihedral12`), which
played Glinski's game move for move, and the two `hex-glinski-flanks` games (a hexagon's seam).

Every test drives the core through `Engine` and asserts what the core did. The only choice of
a move in Python is the seeded random walk of clause 4, the same probe as Q787's.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.engine import _square_name
from chess_variant_sandbox.engine.moves import move_from, move_to
from chess_variant_sandbox.spec import load_spec, validate_spec
from chess_variant_sandbox.spec.complexity import expand_symmetry
from chess_variant_sandbox.spec.schema import _schema_eighteen_uses

ROOT = Path(__file__).resolve().parents[1]
GAMES = 6  # seeds 0 to 5, as pre-registered
PLIES = 300
# The five arrays the mechanics hash seeds, which differ between two spellings of one game.
ZOBRIST = frozenset(
    {"zobrist_pieces", "zobrist_side", "zobrist_ep", "zobrist_colour", "zobrist_turn"}
)
ONE_HAND = {"f7", "c8", "b6", "d3", "g2", "h4"}
OTHER_HAND = {"g6", "d8", "b7", "c4", "f2", "h3"}
# Taken when this file was written (clause 5): a regression pin, not a prediction.
LADDER = {1: 24, 2: 554, 3: 15278}


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return load_spec(ROOT / "archive" / "pinwheel.json")


@pytest.fixture(scope="module")
def engine(spec: dict[str, Any]) -> Engine:
    return Engine(spec)


@pytest.fixture(scope="module")
def unjoined(spec: dict[str, Any]) -> Engine:
    plain = copy.deepcopy(spec)
    del plain["rules"]["wrap"]
    assert validate_spec(plain) == []
    return Engine(plain)


def _written_out(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` with every descriptor's vectors listed under ``none``: its long form."""
    long = copy.deepcopy(spec)
    for piece in long["pieces"]:
        for move in piece["moves"]:
            listed: set[tuple[int, int]] = set()
            for vector in move["vectors"]:
                listed |= expand_symmetry((vector[0], vector[1]), move.get("symmetry", "none"))
            move["vectors"] = [list(v) for v in sorted(listed)]
            move["symmetry"] = "none"
    return long


def _knight_targets(engine: Engine, player: int) -> set[str]:
    state = engine.make_state(
        {"e5": ("N", player), "a5": ("K", 0), "i5": ("K", 1)}, side_to_move=player
    )
    start = engine._square_index("e5")
    return {
        _square_name(move_to(move), engine.spec.width)
        for move in engine.legal_moves(state)
        if move_from(move) == start
    }


class TestTheOneHandedKnight:
    """Clause 1: `rotate6` writes a piece `dihedral12` cannot."""

    def test_the_knight_reaches_one_hand_only(self, engine: Engine) -> None:
        assert _knight_targets(engine, 0) == ONE_HAND

    def test_written_dihedral12_it_reaches_both_hands(self, spec: dict[str, Any]) -> None:
        mirrored = copy.deepcopy(spec)
        knight = next(piece for piece in mirrored["pieces"] if piece["id"] == "N")
        knight["moves"][0]["symmetry"] = "dihedral12"
        assert validate_spec(mirrored) == []
        assert _knight_targets(Engine(mirrored), 0) == ONE_HAND | OTHER_HAND

    def test_the_second_players_knight_turns_the_same_way(self, engine: Engine) -> None:
        assert _knight_targets(engine, 1) == ONE_HAND

    def test_it_is_written_with_both_turns(self, spec: dict[str, Any]) -> None:
        assert _schema_eighteen_uses(spec) == {"symmetry=rotate6", "symmetry=dihedral12"}


class TestTheSpelling:
    """Clause 2: written out under `none`, it is the same game."""

    def test_the_long_forms_knight_lists_six_vectors(self, spec: dict[str, Any]) -> None:
        long = _written_out(spec)
        knight = next(piece for piece in long["pieces"] if piece["id"] == "N")
        assert len(knight["moves"][0]["vectors"]) == 6

    def test_the_tables_are_the_long_forms(self, spec: dict[str, Any]) -> None:
        mine = canonical_tables(compile_spec(spec))
        theirs = canonical_tables(compile_spec(_written_out(spec)))
        assert mine.keys() == theirs.keys()
        assert {key for key in mine if mine[key] != theirs[key]} <= ZOBRIST
        same = {key: value for key, value in mine.items() if key not in ZOBRIST}
        other = {key: value for key, value in theirs.items() if key not in ZOBRIST}
        assert repr(sorted(same.items())) == repr(sorted(other.items()))

    def test_the_perft_is_the_long_forms(self, engine: Engine, spec: dict[str, Any]) -> None:
        long = Engine(_written_out(spec))
        for depth in (1, 2, 3):
            assert engine.perft(engine.initial_state(), depth) == long.perft(
                long.initial_state(), depth
            ), depth


class TestTheSeam:
    """Clauses 3 and 4: the one-way join is out of reach at ply 1 and in play afterwards."""

    def test_nothing_crosses_at_ply_one(self, engine: Engine, unjoined: Engine) -> None:
        mine = engine.legal_moves(engine.initial_state())
        theirs = unjoined.legal_moves(unjoined.initial_state())
        assert len(mine) == 24
        assert sorted(mine) == sorted(theirs)

    def test_random_games_cross_it_both_ways(self, engine: Engine, unjoined: Engine) -> None:
        crossed_games = 0
        crossers: set[int] = set()
        for seed in range(GAMES):
            rng = random.Random(seed)
            state = engine.initial_state()
            crossed = False
            for _ in range(PLIES):
                if engine.terminal(state) is not None:
                    break
                moves = engine.legal_moves(state)
                if not moves:
                    break
                move = rng.choice(moves)
                # A crossing is a move the unjoined spec does not have in that position.
                if move not in unjoined.legal_moves(state):
                    crossed = True
                    crossers.add(state.side_to_move)
                state = engine.apply(state, move)
            crossed_games += crossed
        assert crossed_games >= 4, crossed_games
        assert crossers == {0, 1}


class TestTheLadder:
    """Clause 5: taken, not predicted."""

    @pytest.mark.parametrize("depth", sorted(LADDER))
    def test_the_ladder(self, engine: Engine, depth: int) -> None:
        assert engine.perft(engine.initial_state(), depth) == LADDER[depth]
