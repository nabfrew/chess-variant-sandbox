"""N4 (`docs/reference/no-forced-mate.md` §10): the mop-up term.

Where one side is bare (royals alone, nothing in hand) and the other is not, a minimax player's
stand pat adds, for the armed side, how confined the bare side is: half its box (positions it
reaches moving alone, `Game::bare_reach`), half its legal moves. Built-in objective only; it
reads the core's legal lists, never a square. Held here to the oracle move for move and node
for node, on endings built from chess's pieces (chess only for its familiar endings) and on a
small board; inert where nobody is bare.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import _core
from chess_variant_sandbox.agents import minimax
from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.agents.minimax import (
    MOPUP,
    MinimaxAgent,
    is_armed,
    is_bare,
    mopup_weight,
    repairs_of,
)
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.tournament.play import play_game, play_game_python

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _chess_ending(piece: str, squares: tuple[str, str, str], max_plies: int = 0) -> Engine:
    """Chess's rules from a constructed start: a king and ``piece`` against a lone king."""
    spec = _load("rulesets/chess.json")
    spec["name"] = f"chess ending K{piece} v K"
    own, king, theirs = squares
    spec["setup"] = {
        "mirror": False,
        "placements": [
            {"piece": "K", "squares": [king], "player": 0},
            {"piece": piece, "squares": [own], "player": 0},
            {"piece": "K", "squares": [theirs], "player": 1},
        ],
    }
    for key in ("castling", "en_passant", "promotion"):
        spec["rules"].pop(key, None)
    spec["pieces"] = [p for p in spec["pieces"] if p["id"] in ("K", piece)]
    if max_plies:
        spec["rules"]["max_plies"] = max_plies
    return Engine(spec)


def test_the_constants_are_the_cores() -> None:
    assert minimax.MOPUP_PIECES == _core.MOPUP_PIECES
    assert minimax.MOPUP_SPAN == _core.MOPUP_SPAN
    assert minimax.MOPUP_AREA == _core.MOPUP_AREA


def test_the_option_is_bit_512() -> None:
    assert MOPUP == 512
    assert repairs_of({"mopup": 1}) == repairs_of({}) | MOPUP
    assert repairs_of({"mopup": 0}) == repairs_of({}) & ~MOPUP


def test_bare_and_armed() -> None:
    engine = _chess_ending("R", ("a1", "e1", "e6"))
    state = engine.initial_state()
    spec = engine.spec
    assert is_bare(spec, state, 1) and not is_bare(spec, state, 0)
    assert is_armed(spec, state, 0) and not is_armed(spec, state, 1)
    opening = Engine(_load("rulesets/chess.json")).initial_state()
    assert not is_bare(spec, opening, 0) and not is_bare(spec, opening, 1)


def test_the_weight_is_undefined_off_the_builtin_objective_and_without_royals() -> None:
    engine = Engine(_load("rulesets/chess.json"))
    assert mopup_weight(engine.spec, Evaluator.for_engine(engine)) > 0.0
    assert mopup_weight(engine.spec, object()) == 0.0
    antichess = Engine(_load("rulesets/antichess.json"))
    assert mopup_weight(antichess.spec, Evaluator.for_engine(antichess)) == 0.0


def test_the_box_of_a_cornered_king() -> None:
    """A lone king on a1 under a rook on h3 that its own king on h4 defends: it may walk the
    first two ranks but the rook's file, fourteen squares, and no further; the walk stops at the cap. The rook
    undefended (its king far away on a8) is taken, and the box opens to the whole board."""
    agent_for = lambda engine: MinimaxAgent(engine, nodes=100, repairs=repairs_of({"mopup": 1}))  # noqa: E731
    held = _chess_ending("R", ("h3", "h4", "a1"))
    assert agent_for(held)._bare_reach(held.initial_state(), 1, 64) == 14
    assert agent_for(held)._bare_reach(held.initial_state(), 1, 3) == 3
    loose = _chess_ending("R", ("h3", "a8", "a1"))
    assert agent_for(loose)._bare_reach(loose.initial_state(), 1, 64) == 64


@pytest.mark.parametrize(
    ("piece", "squares"), [("R", ("a1", "e1", "e6")), ("Q", ("h7", "b2", "d5"))]
)
def test_native_and_oracle_alike(piece: str, squares: tuple[str, str, str]) -> None:
    engine = _chess_ending(piece, squares, max_plies=12)
    agents = ("minimax:nodes=120,mopup=1", "minimax:nodes=120,mopup=1")
    native = play_game(engine, agents, seed=3)
    in_python = play_game_python(engine, agents, seed=3)
    assert native.moves == in_python.moves
    assert native.nodes == in_python.nodes


@pytest.mark.slow
@pytest.mark.parametrize(
    ("piece", "squares", "nodes"),
    [("R", ("a1", "e1", "e6"), 200), ("Q", ("h7", "b2", "d5"), 300)],
)
def test_native_and_oracle_alike_whole(piece: str, squares: tuple[str, str, str], nodes: int) -> None:
    engine = _chess_ending(piece, squares)
    agents = (f"minimax:nodes={nodes},mopup=1", f"minimax:nodes={nodes},mopup=1")
    native = play_game(engine, agents, seed=3)
    in_python = play_game_python(engine, agents, seed=3)
    assert native.moves == in_python.moves
    assert native.nodes == in_python.nodes


def test_the_term_moves_a_bare_ending() -> None:
    engine = _chess_ending("Q", ("h7", "b2", "d5"))
    armed = play_game(engine, ("minimax:nodes=1600,mopup=1", "minimax:nodes=1600"), seed=0)
    control = play_game(engine, ("minimax:nodes=1600,mopup=0", "minimax:nodes=1600"), seed=0)
    assert armed.moves != control.moves
    assert armed.winner == 0


def test_inert_where_nobody_is_bare() -> None:
    """From chess's opening nobody is bare within a short game, so the bit changes nothing."""
    spec = _load("rulesets/chess.json")
    spec["rules"]["max_plies"] = 30
    engine = Engine(spec)
    on = play_game(engine, ("minimax:nodes=400,mopup=1",) * 2, seed=5)
    off = play_game(engine, ("minimax:nodes=400,mopup=0",) * 2, seed=5)
    assert on.moves == off.moves
    assert on.nodes == off.nodes
