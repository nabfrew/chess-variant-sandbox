"""V1 of T3's showcase *the circling queen* (`docs/plan-pieces-moves.md` §5.7, C1): the mechanism,
without players.

Chess with the queen's slide given ``path: "circle"``: one-square steps in her eight directions,
each turning an eighth of a turn the same way, so her lines curl round. Its twin is chess itself:
the same file with the path taken off and the piece named queen is `4c1922ffae26`.

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
FILE = ROOT / "archive" / "circling-queen.json"
HASH, CHESS_HASH = "379efba235c5", "4c1922ffae26"
RING = {"kind": "slide", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "path": "circle"}
_MOVE = re.compile(r"^~?[A-Za-z]([a-z]\d+)[-x]([a-z]\d+)")


@cache
def _spec() -> dict[str, Any]:
    return load_spec(FILE)


@cache
def _engine() -> Engine:
    return Engine(_spec())


def _queen(spec: dict[str, Any]) -> dict[str, Any]:
    return next(p for p in spec["pieces"] if p["id"] == "Q")


def test_it_is_the_registered_game_and_chess_is_its_twin() -> None:
    spec = _spec()
    assert spec_hash(spec) == HASH
    assert spec["provenance"]["parents"] == [CHESS_HASH]
    assert _queen(spec)["moves"] == [RING] and _queen(spec)["value"] == 9
    assert spec["appearance"]["icons"]["Q"] == "queen"
    test_spec = copy.deepcopy(spec)
    _queen(test_spec)["name"] = "ring queen"
    assert spec_hash(test_spec) == spec_hash(load_spec(ROOT / "tests" / "_path_ring.json"))
    twin = copy.deepcopy(spec)
    _queen(twin)["name"] = "queen"
    del _queen(twin)["moves"][0]["path"]
    assert spec_hash(twin) == CHESS_HASH


def _targets(engine: Engine, state: GameState, origin: str) -> set[str]:
    found = set()
    for text in engine.move_texts(state):
        matched = _MOVE.match(text)
        if matched and matched.group(1) == origin:
            found.add(matched.group(2))
    return found


def test_she_reaches_the_ring_s_squares_and_not_a_queen_s_long_lines() -> None:
    """T1's hand count on the file: 32 squares from d4 on an empty board (kings a1 and h8), a
    knight's square among them, and none two squares down one of her own lines."""
    found = _targets(_engine(), _engine().make_state(
        {"d4": ("Q", 0), "a1": ("K", 0), "h8": ("K", 1)}, moved=["d4", "a1", "h8"]), "d4")
    assert len(found) == 32
    assert {"e6", "c6", "f5", "b3"} <= found
    assert not {"d6", "d2", "b4", "f4", "d8", "h4"} & found


def _two_circles(*extra: tuple[str, str, int]) -> GameState:
    """The circling queen on d4 checks the black king on e7 along its only two circles: east,
    north-east, north (e4 f5 f6) and west, north-west, north (c5 c6 d7). A black rook on h5
    can step into either, on f5 or on c5."""
    placement = {"d4": ("Q", 0), "a1": ("K", 0), "e7": ("K", 1), "h5": ("R", 1)}
    placement.update({square: (piece, player) for square, piece, player in extra})
    return _engine().make_state(placement, side_to_move=1, moved=list(placement))


def test_a_check_along_two_circles_is_not_answered_by_one_block() -> None:
    """V1's hand position: the rook may move to f5 or to c5 (both are pseudo-legal), but either
    block leaves the other circle open, so neither is legal; the king must step."""
    spec, engine = _spec(), _engine()
    state = _two_circles()
    royal = path_oracle.royal_square(spec, state.board, 1)
    assert path_oracle.attacked(spec, state.board, royal, 0)
    rook = [p["id"] for p in spec["pieces"]].index("R")
    pseudo = {
        target
        for origin, target, _, _ in map(decode_move, engine.pseudo_legal_moves(state))
        if state.board[origin] and (state.board[origin] - 1) % len(spec["pieces"]) == rook
    }
    for block in ("f5", "c5"):
        after = _two_circles((block, "R", 1))
        index = next(i for i, (a, b) in enumerate(zip(state.board, after.board, strict=True)) if a != b)
        assert index in pseudo  # the block is a move the rook has
        assert path_oracle.attacked(spec, after.board, royal, 0)  # and the other circle checks
    assert not {"f5", "c5"} & _targets(engine, state, "h5")
    legal_texts = engine.move_texts(state)
    assert legal_texts and all(text.startswith("K") for text in legal_texts)


def test_with_one_circle_shut_the_other_block_answers_the_check() -> None:
    """The control: with a black pawn already on c6 the check runs by e4 f5 f6 alone, and the
    rook's block on f5 is legal."""
    assert _targets(_engine(), _two_circles(("c6", "P", 1)), "h5") == {"f5"}


def test_the_opening_is_chess_s() -> None:
    """Twenty moves, as chess: the queen is shut in by her own pieces."""
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
    """A seeded random game of the file: after every move the core's queen moves equal the
    oracle's walk of the circles, and its legal list equals the pseudo-legal one filtered by
    the oracle's attack walk."""
    spec, engine = _spec(), _engine()
    n_types = len(spec["pieces"])
    queen = [p["id"] for p in spec["pieces"]].index("Q")
    chooser = random.Random(seed)
    state = engine.initial_state()
    checked = 0
    for _ in range(plies):
        found = {
            (origin, target)
            for origin, target, _, _ in map(decode_move, engine.pseudo_legal_moves(state))
            if (state.board[origin] - 1) % n_types == queen
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
