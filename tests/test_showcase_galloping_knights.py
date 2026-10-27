"""G7 of Q1023's showcase *chess with galloping knights* (`docs/archive/plan-schema-31.md` §6.4, L1): the
mechanism, without players.

Chess with the knight keeping its leap and riding lamely over ``[1, 2]`` from its second jump on
(``range: 0``, ``min_range: 2``): the single jump ignores its leg, every jump of a gallop is
blocked on its leg (one step along the longer axis), and a gallop's first landing blocks but is
never stopped on, since the leap already lands there. Its slide twin is *chess with nightriders*
(`archive/chess-with-nightriders.json`), the same lines with nothing between landings able to
block.

Nothing here decides a move's legality (CLAUDE.md, rule 3): the lists are the core's, and
`test_lame_rider._rider_targets` (a declared oracle, Q1007a's) plus the knight's leap is the second
reading. Fairy-Stockfish has no lame rider we rely on (§2.5), so there is no second engine.
"""

from __future__ import annotations

import copy
import random
import re
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from test_lame_rider import _rider_targets

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import load_spec, spec_hash

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "archive" / "chess-with-galloping-knights.json"
TWIN = ROOT / "archive" / "chess-with-nightriders.json"
HASH, CHESS_HASH = "69f1e486af87", "4c1922ffae26"
LEAP = {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8", "mode": "both"}
GALLOP = {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8", "range": 0, "min_range": 2,
          "mode": "both"}
_MOVE = re.compile(r"^~?[A-Za-z]([a-z]\d+)[-x]([a-z]\d+)")
KNIGHT_JUMPS = {(a * x, b * y) for a in (1, -1) for b in (1, -1) for x, y in ((1, 2), (2, 1))}


@cache
def _spec() -> dict[str, Any]:
    return load_spec(FILE)


@cache
def _engine() -> Engine:
    return Engine(_spec())


@cache
def _twin() -> Engine:
    return Engine(load_spec(TWIN))


def _knight(spec: dict[str, Any]) -> dict[str, Any]:
    return next(p for p in spec["pieces"] if p["id"] == "N")


def test_it_is_the_registered_game_and_chess_is_its_parent() -> None:
    spec = _spec()
    assert spec_hash(spec) == HASH
    assert spec["provenance"]["parents"] == [CHESS_HASH]
    assert _knight(spec)["moves"] == [LEAP, GALLOP]
    assert spec["appearance"]["icons"]["N"] == "knight"
    # One change on chess: the knight's name, value and gallop.
    chess = copy.deepcopy(spec)
    _knight(chess).update(name="knight", value=3, moves=[LEAP])
    assert spec_hash(chess) == CHESS_HASH


def _targets(engine: Engine, state: GameState, origin: str) -> set[str]:
    found = set()
    for text in engine.move_texts(state):
        matched = _MOVE.match(text)
        if matched and matched.group(1) == origin:
            found.add(matched.group(2))
    return found


def _board(*extra: tuple[str, str, int], side: int = 0) -> GameState:
    """A white knight on b1 whose b1-c3-d5-e7 line is the gallop under test; kings out of it."""
    placement = {"b1": ("N", 0), "h1": ("K", 0), "h8": ("K", 1)}
    placement.update({square: (piece, player) for square, piece, player in extra})
    return _engine().make_state(placement, side_to_move=side, moved=list(placement))


def test_on_an_open_board_it_gallops_to_the_edge() -> None:
    """Three leaps (a3, c3, d2), and gallops on to d5 and e7, and to f3 and h4."""
    assert _targets(_engine(), _board(), "b1") == {"a3", "c3", "d2", "d5", "e7", "f3", "h4"}


def test_a_piece_on_a_leg_stops_the_gallop_and_not_the_leap() -> None:
    """b2 is the first jump's leg: the leap to c3 stays, the gallop on to d5 and e7 goes. A piece
    on c4, the second jump's leg, takes d5 and e7 away; one on d6, the third's, only e7."""
    assert {"c3"} <= _targets(_engine(), _board(("b2", "P", 0)), "b1")
    assert not {"d5", "e7"} & _targets(_engine(), _board(("b2", "P", 0)), "b1")
    found = _targets(_engine(), _board(("c4", "P", 1)), "b1")
    assert "c3" in found and not {"d5", "e7"} & found
    found = _targets(_engine(), _board(("d6", "P", 1)), "b1")
    assert {"c3", "d5"} <= found and "e7" not in found


def test_the_first_landing_blocks_and_is_taken_only_by_the_leap() -> None:
    """An enemy on c3 is taken by the leap; the gallop through it is blocked."""
    found = _targets(_engine(), _board(("c3", "P", 1)), "b1")
    assert "c3" in found and not {"d5", "e7"} & found
    assert "e7" in _targets(_engine(), _board(("e7", "P", 1)), "b1")  # taken at a gallop's end


def _gallop_check(*extra: tuple[str, str, int]) -> GameState:
    """The knight on b1 checks the black king on e7 by b1-c3-d5-e7; a black rook on d8 may step to
    d6, the last jump's leg."""
    placement = {"b1": ("N", 0), "h1": ("K", 0), "e7": ("K", 1), "d8": ("R", 1)}
    placement.update({square: (piece, player) for square, piece, player in extra})
    return _engine().make_state(placement, side_to_move=1, moved=list(placement))


def test_a_check_at_a_gallop_is_answered_by_a_block_on_a_leg() -> None:
    state = _gallop_check()
    assert _engine().in_check(state, 1)
    assert "d6" in _targets(_engine(), state, "d8")
    # The nightrider is not blocked on a leg: there the same rook move does not answer the check.
    twin = _twin().make_state(
        {"b1": ("N", 0), "h1": ("K", 0), "e7": ("K", 1), "d8": ("R", 1)}, side_to_move=1,
        moved=["b1", "h1", "e7", "d8"])
    assert _twin().in_check(twin, 1)
    assert "d6" not in _targets(_twin(), twin, "d8")


def test_the_opening_is_chess_s_and_the_nightrider_s_raid_is_gone() -> None:
    """Twenty moves, as chess; the nightrider's b1 over c3 and d5 to e7 is not one of them."""
    state = _engine().initial_state()
    assert len(_engine().legal_moves(state)) == 20
    assert "e7" in _targets(_twin(), _twin().initial_state(), "b1")
    assert "e7" not in _targets(_engine(), state, "b1")


def _leap_targets(spec: dict[str, Any], board: bytes, side: int) -> set[tuple[int, int]]:
    """The knight's single jumps: any knight square on the board not held by its own side."""
    width, height, n_types = spec["board"]["width"], spec["board"]["height"], len(spec["pieces"])
    knight = [p["id"] for p in spec["pieces"]].index("N")
    found = set()
    for origin, value in enumerate(board):
        if value == 0 or (value - 1) // n_types != side or (value - 1) % n_types != knight:
            continue
        for dx, dy in KNIGHT_JUMPS:
            f, r = origin % width + dx, origin // width + dy
            if 0 <= f < width and 0 <= r < height:
                held = board[r * width + f]
                if held == 0 or (held - 1) // n_types != side:
                    found.add((origin, r * width + f))
    return found


def _walk(seed: int, plies: int = 160) -> int:
    """A seeded random game of the file: after every move the core's knight, bishop, rook and
    queen moves equal the oracle's walk of their lines plus the knight's leap."""
    spec, engine = _spec(), _engine()
    n_types = len(spec["pieces"])
    riders = {i for i, p in enumerate(spec["pieces"]) if p["id"] in "NBRQ"}
    chooser = random.Random(seed)
    state = engine.initial_state()
    checked = 0
    for _ in range(plies):
        found = {
            (origin, target)
            for origin, target, _, _ in map(decode_move, engine.pseudo_legal_moves(state))
            if (state.board[origin] - 1) % n_types in riders
        }
        side = state.side_to_move
        expected = _rider_targets(spec, state.board, side, n_types) | _leap_targets(
            spec, state.board, side)
        assert found == expected
        checked += 1
        legal = engine.legal_moves(state)
        if not legal or engine.terminal(state, legal) is not None:
            break
        state = engine.apply(state, chooser.choice(legal))
    return checked


@pytest.mark.parametrize("seed", range(3))
def test_the_core_and_the_oracle_agree_over_a_seeded_game(seed: int) -> None:
    assert _walk(seed) >= 2


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(3, 23))
def test_the_core_and_the_oracle_agree_over_twenty_seeded_games(seed: int) -> None:
    assert _walk(seed) >= 2
