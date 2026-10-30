"""V1 of T2's showcase *crooked bishops* (`docs/archive/plan-pieces-moves.md` §5.7, Z1): the mechanism,
without players.

Chess with each bishop's slide given ``path: "zigzag"``: diagonal steps turning to a neighbouring
diagonal and back, so the path runs along a rank or a file on its colour. Its twin is chess
itself: the same file with the path taken off and the piece named bishop is `4c1922ffae26`.

Nothing here decides a move's legality (CLAUDE.md, rule 3): the lists are the core's, and
`path_oracle` (a declared oracle, T1's) is the second reading. Fairy-Stockfish has no reading of a
turning path (§5.4), so there is no second engine.
"""

from __future__ import annotations

import copy
import random
import re
from functools import cache
from pathlib import Path
from typing import Any

import path_oracle
import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import load_spec, spec_hash

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "archive" / "crooked-bishops.json"
HASH, TEST_SPEC_HASH, CHESS_HASH = "7e57cab1fd7e", "0ed28ce2fe25", "4c1922ffae26"  # Q1069a; HASH was 23f6613e9fa3
# Q1069a: the file and chess both draw on dead material now, so the twin is chess as it is
# (`CHESS_NOW`) while the parent named in the provenance stays chess as it was.
CHESS_NOW = "831c77e75650"
ZIGZAG = {"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4", "path": "zigzag"}
_MOVE = re.compile(r"^~?[A-Za-z]([a-z]\d+)[-x]([a-z]\d+)")


@cache
def _spec() -> dict[str, Any]:
    return load_spec(FILE)


@cache
def _engine() -> Engine:
    return Engine(_spec())


def _bishop(spec: dict[str, Any]) -> dict[str, Any]:
    return next(p for p in spec["pieces"] if p["id"] == "B")


def test_it_is_the_registered_game_and_chess_is_its_twin() -> None:
    spec = _spec()
    assert spec_hash(spec) == HASH
    assert spec["provenance"]["parents"] == [CHESS_HASH]
    assert _bishop(spec)["moves"] == [ZIGZAG] and _bishop(spec)["value"] == 3
    assert spec["appearance"]["icons"]["B"] == "bishop"
    test_spec = copy.deepcopy(spec)
    test_spec["rules"].pop("insufficient_material")  # T1's test copy predates Q1069a
    _bishop(test_spec)["name"] = "bolt"
    assert spec_hash(test_spec) == TEST_SPEC_HASH  # T1's tests/_path_bolt.json
    twin = copy.deepcopy(spec)
    _bishop(twin)["name"] = "bishop"
    del _bishop(twin)["moves"][0]["path"]
    assert spec_hash(twin) == CHESS_NOW


def _targets(engine: Engine, state: GameState, origin: str) -> set[str]:
    found = set()
    for text in engine.move_texts(state):
        matched = _MOVE.match(text)
        if matched and matched.group(1) == origin:
            found.add(matched.group(2))
    return found


def _position(*extra: tuple[str, str, int]) -> GameState:
    placement = {"d4": ("B", 0), "a1": ("K", 0), "h8": ("K", 1)}
    placement.update({square: (piece, player) for square, piece, player in extra})
    return _engine().make_state(placement, moved=list(placement))


def test_a_pawn_on_the_first_square_shuts_those_paths_and_keeps_the_others() -> None:
    """V1's hand position, in the file's own rules: an own pawn on e5 ends both paths that
    start north-east (e5 f4 g5 h4 and e5 d6 e7 d8); the six others still reach f4, h4, d6 and
    d8 their own way, and a square reached two ways is one move."""
    engine = _engine()
    before = _targets(engine, _position(), "d4")
    after = _targets(engine, _position(("e5", "P", 0)), "d4")
    lost = {"e5", "g5", "e7"}
    assert len(before) == 18
    assert lost <= before and after == before - lost
    assert {"f4", "h4", "d6", "d8"} <= after


def test_an_enemy_on_the_first_square_is_taken_and_ends_the_path() -> None:
    found = _targets(_engine(), _position(("e5", "P", 1)), "d4")
    assert "e5" in found and not {"g5", "e7"} & found


def test_the_opening_is_chess_s() -> None:
    """Twenty moves, as chess: every bishop's first square is an own pawn."""
    state = _engine().initial_state()
    assert len(_engine().legal_moves(state)) == 20


def _legal_by_oracle(spec: dict[str, Any], engine: Engine, state: GameState) -> set[int]:
    me = state.side_to_move
    keep = set()
    for move in engine.pseudo_legal_moves(state):
        after = engine.apply(state, move)
        royal = path_oracle.royal_square(spec, after.board, me)
        if royal is None or not path_oracle.attacked(spec, after.board, royal, 1 - me):
            keep.add(move)
    return keep


def _walk(seed: int, plies: int = 160) -> int:
    """A seeded random game of the file: after every move the core's crooked-bishop moves
    equal the oracle's walk of the paths, and its legal list equals the pseudo-legal one
    filtered by the oracle's attack walk."""
    spec, engine = _spec(), _engine()
    n_types = len(spec["pieces"])
    bishop = [p["id"] for p in spec["pieces"]].index("B")
    chooser = random.Random(seed)
    state = engine.initial_state()
    checked = 0
    for _ in range(plies):
        found = {
            (origin, target)
            for origin, target, _, _ in map(decode_move, engine.pseudo_legal_moves(state))
            if (state.board[origin] - 1) % n_types == bishop
        }
        assert found == path_oracle.path_targets(spec, state.board, state.side_to_move)
        legal = engine.legal_moves(state)
        assert set(legal) == _legal_by_oracle(spec, engine, state)
        checked += 1
        if not legal or engine.terminal(state, legal) is not None:
            break
        state = engine.apply(state, chooser.choice(legal))
    return checked


@pytest.mark.parametrize("seed", range(3))
def test_the_core_and_the_oracle_agree_over_a_seeded_game(seed: int) -> None:
    assert _walk(seed) >= 2


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(3, 20))
def test_the_core_and_the_oracle_agree_over_twenty_seeded_games(seed: int) -> None:
    assert _walk(seed) >= 2
