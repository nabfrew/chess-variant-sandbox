"""WK4 (`docs/reference/wakwak.md` §3 D, §9): the transposition table keeps a best move.

Every node of the main search files its best move beside its value (an upper bound keeps the
move its entry already held), and a node whose entry cannot answer tries that move first.
Ordering only: the same moves are searched, so a finished search's value is unchanged; a
node budget then buys deeper iterations. Held here to the oracle move for move and node for
node, on a game without a duck and on one with.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.agents.minimax import (
    DEFAULT_REPAIRS,
    TT_MOVE,
    MinimaxAgent,
    repairs_of,
)
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.tournament.play import play_game, play_game_python

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _engine(game: str) -> Engine:
    spec = _load("rulesets/gardner.json")
    if game == "duck":
        spec["name"] = "gardner with a duck"
        spec["pieces"].append({"id": "D", "name": "duck", "value": 0, "moves": []})
        spec["rules"]["win"] = "royal_capture"
        spec["rules"]["duck"] = {"piece": "D"}
        spec["rules"]["max_plies"] = 60
        spec["rules"].pop("checkmate", None)
    return Engine(spec)


def test_the_option_is_bit_256() -> None:
    assert repairs_of({"tt_move": 1}) == repairs_of({}) | TT_MOVE
    assert repairs_of({"tt_move": 0}) == repairs_of({}) & ~TT_MOVE
    assert TT_MOVE == 256


@pytest.mark.parametrize("game", ["gardner", "duck"])
@pytest.mark.parametrize("bits", ["tt_move=1", "tt_move=0"])
def test_native_and_oracle_alike(game: str, bits: str) -> None:
    engine = _engine(game)
    nodes = 300 if game == "gardner" else 200
    agents = (f"minimax:nodes={nodes},{bits}", f"minimax:nodes={nodes},tt_move=1")
    native = play_game(engine, agents, seed=7)
    in_python = play_game_python(engine, agents, seed=7)
    assert native.moves == in_python.moves
    assert native.nodes == in_python.nodes


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 4])
def test_native_and_oracle_alike_deeper(seed: int) -> None:
    engine = _engine("gardner")
    agents = ("minimax:nodes=1600,tt_move=1,keep=1", "minimax:nodes=1600,tt_move=1")
    native = play_game(engine, agents, seed=seed)
    in_python = play_game_python(engine, agents, seed=seed)
    assert native.moves == in_python.moves
    assert native.nodes == in_python.nodes


def test_the_move_moves_a_game() -> None:
    engine = _engine("gardner")
    # Spelled as the player before N4 (`mopup=0`), whose seed-2 game this test was written on.
    control = "minimax:nodes=800,tt_move=0,mopup=0"
    armed = play_game(engine, ("minimax:nodes=800,tt_move=1,mopup=0", control), seed=2)
    assert armed.moves != play_game(engine, (control,) * 2, seed=2).moves


@pytest.mark.parametrize("game", ["gardner", "duck"])
def test_a_finished_search_keeps_its_value_and_costs_less(game: str) -> None:
    """Over a few positions of a game, a depth-4 iterative search with the table's move
    reaches the same value as one without; on gardner in fewer nodes in total. (With a duck
    the move it files at a duck half overrides WK2's orderings there, and the total is not
    always lower: `wakwak.md` §9.)"""
    engine = _engine(game)
    record = play_game(engine, ("random", "random"), 3)
    state = engine.initial_state()
    without = DEFAULT_REPAIRS & ~TT_MOVE
    costs = {TT_MOVE | without: 0, without: 0}
    for move in record.moves[:6]:
        values = []
        for bits in costs:
            agent = MinimaxAgent(engine, nodes=10**7, max_depth=4, repairs=bits)
            agent._begin_search()
            for depth in range(1, 5):
                value = agent._negamax(state, depth, 0, -float("inf"), float("inf"))
            values.append(value)
            costs[bits] += agent._nodes
        assert values[0] == values[1]
        state = engine.apply(state, move)
    if game == "gardner":
        assert costs[TT_MOVE | without] < costs[without]
