"""Schema 26 (Q918a): the core's rounds against the declared oracle, round by round.

`docs/plan-simultaneous.md` §6.1 and §7's Q918a box: random walks through the test specs,
every round resolved by `make_round` and by `tests/simultaneous_oracle.py`, the whole
position compared, and the incremental key held to the slow definition. The fast tier walks
a few games per spec; the slow tier walks many more. Schema 27's specs (Q942a) carry the
chase switches, and each round's two move lists are held to the oracle's rest filter.
"""

from __future__ import annotations

import json
import random
from functools import cache
from pathlib import Path

import pytest
from simultaneous_oracle import legal_for, resolve_round

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.state import GameState

HERE = Path(__file__).parent
SPECS = sorted(path.stem for path in HERE.glob("_simultaneous_*.json"))


@cache
def engine_for(name: str) -> Engine:
    return Engine(json.loads((HERE / f"{name}.json").read_text(encoding="utf-8")))


def walk(engine: Engine, seed: int, rounds: int) -> int:
    """Play `rounds` random rounds from the start, checking every one; returns rounds played."""
    rng = random.Random(seed)
    state: GameState = engine.initial_state()
    for played in range(rounds):
        if engine.terminal(state) is not None:
            return played
        lists = [engine.legal_moves_for(state, player) for player in (0, 1)]
        # Schema 27 (Q942a): the rest's filter and its lift, held to the oracle's.
        assert lists == [legal_for(engine, state, player) for player in (0, 1)]
        first = rng.choice(lists[0])
        second = rng.choice(lists[1])
        expected = resolve_round(engine, state, first, second)
        actual = engine.apply_round(state, first, second)
        assert actual == expected, (
            f"round {played + 1} of seed {seed}: {engine.describe_move(state, first)} | "
            f"{engine.describe_move(state, second)}"
        )
        assert actual.key == engine._compute_hash(actual)
        state = actual
    return rounds


@pytest.mark.parametrize("name", SPECS)
def test_the_core_and_the_oracle_agree_on_random_rounds(name: str) -> None:
    engine = engine_for(name)
    played = sum(walk(engine, seed, 40) for seed in range(3))
    assert played > 0


@pytest.mark.slow
@pytest.mark.parametrize("name", SPECS)
def test_the_core_and_the_oracle_agree_on_many_random_games(name: str) -> None:
    engine = engine_for(name)
    for seed in range(100, 140):
        walk(engine, seed, 400)
