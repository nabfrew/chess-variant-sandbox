"""The evaluator's spy term (schema 27, Q935h, `docs/plan-spy.md` §6.1).

Holding the only spy is worth `SPY_PIECES` mean non-royal pieces, mine minus theirs
(`eval.rs::spy_edge`, `agents/evaluation.py::Evaluator.spy_edge`). The core's copy is held to
the declared one; the term reads exactly that and nothing else, and nothing in a game without
a spy; and on a hand-built position it does what §6.1 says it is for: an answerer holding the
only spy no longer lets it be taken for a pawn, which the same oracle without the term does.
The core and the oracles agreeing move for move and node for node over whole games of the
four spy test specs, with the term on, is `tests/test_spy_players.py`.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from chess_variant_sandbox import _core
from chess_variant_sandbox.agents import build_agent, native_spec
from chess_variant_sandbox.agents.evaluation import (
    SPY_PIECES,
    Evaluator,
    _mean_non_royal,
    inexhaustible_types,
)
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.tournament.play import eval_values, seat_rng

HERE = Path(__file__).parent
ROOT = HERE.parent


def _engine(name: str) -> Engine:
    return Engine(json.loads((HERE / f"{name}.json").read_text(encoding="utf-8")))


def _position(engine: Engine, white: dict[str, str], black: dict[str, str], side: int = 0) -> GameState:
    placement = {square: (piece, 0) for square, piece in white.items()}
    placement.update({square: (piece, 1) for square, piece in black.items()})
    return engine.make_state(placement, side_to_move=side)


def _move(engine: Engine, moves: list[int], origin: str, target: str) -> int:
    a, b = engine._square_index(origin), engine._square_index(target)
    found = [m for m in moves if decode_move(m)[:2] == (a, b)]
    assert len(found) == 1, (origin, target)
    return found[0]


def test_the_weight_is_the_cores() -> None:
    assert _core.SPY_PIECES == SPY_PIECES == 1.0


@pytest.mark.parametrize(
    ("white", "black", "edge"),
    [
        ({"a1": "K", "c2": "S"}, {"e5": "K", "c4": "S"}, 0),  # both stand: alternating
        ({"a1": "K", "c2": "S"}, {"e5": "K", "c4": "N"}, 1),  # White holds the only one
        ({"a1": "K", "c2": "N"}, {"e5": "K", "c4": "S"}, -1),  # Black does
        ({"a1": "K", "c2": "N"}, {"e5": "K", "c4": "N"}, 0),  # neither: blind
    ],
)
def test_the_term_is_one_mean_piece_for_the_only_holder(
    white: dict[str, str], black: dict[str, str], edge: int
) -> None:
    """What the term adds, read off the declared evaluator with and without it: exactly one
    mean non-royal piece times the edge, from either side, in a turn and in a round."""
    engine = _engine("_spy_small")
    state = _position(engine, white, black)
    with_term = Evaluator.for_spec(engine.spec)
    without = dataclasses.replace(with_term, spy_pieces=0.0)
    mean = _mean_non_royal(
        with_term.values, engine.spec.royal_types, inexhaustible_types(engine.spec)
    )
    for player in (0, 1):
        sign = 1 if player == 0 else -1
        assert with_term.spy_edge(state.board, player) == sign * edge
        lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
        counts = (len(lists[0]), len(lists[1]))
        added = with_term.evaluate_round(engine, state, player, counts, lists) - (
            without.evaluate_round(engine, state, player, counts, lists)
        )
        assert added == SPY_PIECES * mean * sign * edge
        if edge == 0:
            assert with_term.evaluate(engine, state, player, 5) == without.evaluate(
                engine, state, player, 5
            )


def test_no_game_without_a_spy_reads_it() -> None:
    """The weight is zero by gate where nothing is a spy: the declared evaluator never asks
    (`spy_edge` asserts a spy), and the core's games are `test_spy_players.py::_INERT`'s."""
    engine = Engine(json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8")))
    assert engine.spec.spy is None
    state = engine.initial_state()
    evaluator = Evaluator.for_spec(engine.spec)
    assert evaluator.evaluate(engine, state, 0, 20) == dataclasses.replace(
        evaluator, spy_pieces=0.0
    ).evaluate(engine, state, 0, 20)


# The hand-built position of §6.1: an announced round on `_spy_small`, White holding the only
# spy and answering, Black announcing Rc5xc3, a capture of the spy. White may dodge with the
# spy or take the e4 pawn with its knight and let the spy go. By material alone the pawn is
# worth more than the spy (the derived table prices a pawn above a piece that cannot capture);
# with the term the spy is worth its value and one mean piece, so the holder keeps it.
_HOLDER = {"a1": "K", "c3": "S", "d2": "N"}
_ANNOUNCER = {"e5": "K", "c5": "R", "e4": "P"}
_PLAYERS = ("greedy", "minimax:nodes=100", "minimax:nodes=400", "minimax:nodes=1600", "minimax:nodes=6400")
# Below 1,600 nodes the same player without the term takes the pawn (the control); from
# there it sees the blind rounds the trade leads into and dodges anyway.
_SHALLOW = ("greedy", "minimax:nodes=100", "minimax:nodes=400")


def _holder_position() -> tuple[Engine, GameState, tuple[list[int], list[int]], int]:
    engine = _engine("_spy_small")
    state = _position(engine, _HOLDER, _ANNOUNCER)
    assert (engine.phase(state).name, engine.phase(state).announcer) == ("announced", 1)
    lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
    return engine, state, lists, _move(engine, lists[1], "c5", "c3")


def _oracle_answer(agent: str, spy_pieces: float) -> tuple[int, int]:
    engine, state, lists, announced = _holder_position()
    player = build_agent(agent, engine)
    player.evaluator = dataclasses.replace(player.evaluator, spy_pieces=spy_pieces)
    move = player.select_answer(state, 0, lists, announced, seat_rng(7, 0))
    return move, player.stats.nodes


@pytest.mark.parametrize("agent", _PLAYERS)
def test_the_holder_does_not_trade_its_spy_for_a_pawn(agent: str) -> None:
    engine, state, lists, announced = _holder_position()
    native = native_spec(agent, engine.spec.hash)
    assert native is not None
    move, _, nodes, _ = engine.select_round(state, 0, native, eval_values(engine.spec), 7, announced)
    assert move is not None
    assert decode_move(move)[0] == engine._square_index("c3"), engine.describe_move(state, move)
    if agent in _SHALLOW:
        assert (move, nodes) == _oracle_answer(agent, SPY_PIECES)


@pytest.mark.parametrize("agent", _SHALLOW)
def test_without_the_term_the_same_player_takes_the_pawn(agent: str) -> None:
    """The control: the declared oracle with the term at nought lets the spy go for the
    pawn, so the test above is the term's doing and not the position's."""
    engine, state, lists, _ = _holder_position()
    move, _ = _oracle_answer(agent, 0.0)
    assert move == _move(engine, lists[0], "d2", "e4"), engine.describe_move(state, move)


def test_the_derived_table_prices_the_pawn_above_the_spy() -> None:
    """Why the control trades: the trade wins material by the table, and loses it by the
    table plus the term."""
    engine, _, _, _ = _holder_position()
    evaluator = Evaluator.for_spec(engine.spec)
    pawn, spy = (evaluator.values[engine.spec.piece_ids.index(p)] for p in ("P", "S"))
    mean = _mean_non_royal(evaluator.values, engine.spec.royal_types, inexhaustible_types(engine.spec))
    assert spy < pawn < spy + SPY_PIECES * mean
