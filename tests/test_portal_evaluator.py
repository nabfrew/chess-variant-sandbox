"""The Python evaluator reads the two armies, as the core does, on a portal game (Q846).

A momentum portal end (schemas 19-21) stands on a board byte in the third owner's block
(`compile.WILD`). `rust/src/eval.rs::Evaluator::immunity` walks `all_occupied`, which holds
the two armies only; the oracle, `agents/evaluation.py::Evaluator.immunity`, walked every
byte and raised IndexError on a portal end, so no momentum portal game could be played
through the Python loop. Every momentum game in the archive reaches the term: a portal end
is out of the capture matrix, so each has one.

These hold the oracle to the core where it failed: the full-window probe
(`analysis._score_roots` in the core, `analysis._root_scores` in Python, through the same
evaluator) scores every root move alike on positions along a line of a momentum library
game and of a fold spec, and a momentum game plays the same through both loops.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from test_backends import _assert_same_played
from test_schema_twenty_four import fold_pair

from chess_variant_sandbox.agents.evaluation import Evaluator, immunity_threats
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.compile import WILD
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.analysis import _root_scores, _score_roots

ARCHIVE = Path(__file__).resolve().parents[1] / "archive"

# The momentum portal games in the archive, schema 19 (`wormhole`, a pair; `carousel`, two)
# and 20 (`ferry-no-straight-back`, a pair an end of which moves).
MOMENTUM = ("wormhole", "carousel", "ferry-no-straight-back")


def _momentum(name: str) -> Engine:
    return Engine(load_spec(ARCHIVE / f"{name}.json"))


def _line(engine: Engine, plies: int) -> list[GameState]:
    """The positions along a fixed line: at ply ``n`` the ``(7 n + 3)``-th legal move, mod
    the count, so it is the same line every run and reaches past the opening."""
    state = engine.initial_state()
    positions = [state]
    for ply in range(plies):
        legal = engine.legal_moves(state)
        if not legal or engine.terminal(state, legal) is not None:
            break
        state = engine.apply(state, legal[(7 * ply + 3) % len(legal)])
        positions.append(state)
    return positions


def _has_portal_byte(engine: Engine, state: GameState) -> bool:
    stride = engine.spec.n_facings * engine.spec.n_types
    return any(value and (value - 1) // stride == WILD for value in state.board)


@pytest.mark.parametrize("name", MOMENTUM)
def test_a_momentum_game_reaches_the_immunity_term(name: str) -> None:
    """The precondition: the term is defined, the evaluator reads it, and the board carries
    a portal end's byte, which is what the oracle used to fail on."""
    engine = _momentum(name)
    evaluator = Evaluator.for_engine(engine)
    state = engine.initial_state()
    assert immunity_threats(engine.spec) is not None
    assert evaluator.immune_pieces > 0
    assert _has_portal_byte(engine, state)
    evaluator.evaluate(engine, state, 0)


@pytest.mark.parametrize(
    "build",
    [lambda: _momentum("wormhole"), lambda: _momentum("carousel"), lambda: Engine(fold_pair())],
    ids=["wormhole", "carousel", "fold_pair"],
)
def test_the_core_and_the_oracle_score_portal_positions_alike(
    build: Callable[[], Engine],
) -> None:
    """The probe, to the bit, on every fourth position of a twelve-ply line. Asked without
    mobility, which the probe has no term for (as `test_seal.py` asks it)."""
    engine = build()
    evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
    assert evaluator.immune_pieces > 0 and immunity_threats(engine.spec) is not None
    positions = _line(engine, 12)[::4]
    assert len(positions) == 4
    for state in positions:
        legal = engine.legal_moves(state)
        assert legal
        assert _score_roots(engine, evaluator, state, legal) == _root_scores(
            engine, evaluator, state, legal
        )


@pytest.mark.parametrize("name", MOMENTUM)
def test_a_momentum_game_is_played_the_same_on_both_sides(name: str) -> None:
    _assert_same_played(_momentum(name), ("minimax:nodes=150", "greedy"))
