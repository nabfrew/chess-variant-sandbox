"""Q1063: a sweep with two lengths, how far it moves (``range``, m) and how far it clears
(``sweep_range``, n, default m).

Over steps 1..n it clears every piece it passes and may land on any of those squares, as a
sweep always has; over steps n+1..m it moves as a slide, ended by the first piece. One
rule, not a case: n = m is the sweep every stored spec writes, and the normal form drops
it, so no name, table or perft moves. `tests/sweep_oracle.py` states the rule again by its
own means and holds the core to it, move by move and by perft (Fairy-Stockfish has no
sweep at all, so it refuses the key with the kind).
"""

from __future__ import annotations

import copy
import json
import random
import re
from collections import Counter
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import sweep_oracle

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.engine.state import NO_SQUARE, GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.normal import normal_form
from chess_variant_sandbox.spec.spelling import current_spelling

ROOT = Path(__file__).resolve().parent.parent
TWO_LENGTHS = Path(__file__).parent / "_sweep_three_slide_on.json"
# The two stored sweep games, by the names they had before Q1063 -- until Q1073 (2026-10-31)
# switched on the draw on dead material, a mechanics edit (was 42bd7d06828d, c7c6907901b3).
STORED = {
    "archive/chess-sweeping-bishops.json": "01cdfe2e679b",
    "archive/short-sweeping-bishop-chess.json": "ceec60c5fbbc",
}

KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}


def sweeper(sweep_range: int | None, reach: int = 0) -> dict[str, Any]:
    move: dict[str, Any] = {"kind": "sweep", "vectors": [[1, 1]], "symmetry": "rotate4"}
    if reach:
        move["range"] = reach
    if sweep_range is not None:
        move["sweep_range"] = sweep_range
    return move


def probe(move: dict[str, Any], at: str, extra: tuple[dict[str, Any], ...] = ()) -> dict[str, Any]:
    """The sweeper X on ``at``, kings on h1 and a8, blockers P."""
    return {
        "name": "two lengths probe",
        "board": {"width": 8, "height": 8},
        "pieces": [
            {"id": "X", "name": "sweeper", "value": 4, "moves": [copy.deepcopy(move)]},
            {"id": "K", "name": "king", "royal": True, "value": 0, "moves": [KING]},
            {"id": "P", "name": "blocker", "value": 1,
             "moves": [{"kind": "leap", "vectors": [[0, 1]], "mode": "move"}]},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "X", "player": 0, "squares": [at]},
            {"piece": "K", "player": 0, "squares": ["h1"]},
            {"piece": "K", "player": 1, "squares": ["a8"]},
            *extra,
        ]},
        "rules": {"win": "checkmate", "max_plies": 200},
    }


def blocker(square: str, player: int) -> dict[str, Any]:
    return {"piece": "P", "player": player, "squares": [square]}


@cache
def _engine(text: str) -> Engine:
    return Engine(json.loads(text))


def engine_of(spec: dict[str, Any]) -> Engine:
    return _engine(json.dumps(spec, sort_keys=True))


_MOVE = re.compile(r"^~?[A-Za-z]([a-z]\d+)[-x]([a-z]\d+)")


def targets(spec: dict[str, Any], origin: str) -> set[str]:
    engine = engine_of(spec)
    found = set()
    for text in engine.move_texts(engine.initial_state()):
        matched = _MOVE.match(text)
        if matched and matched.group(1) == origin:
            found.add(matched.group(2))
    return found


def two_lengths() -> dict[str, Any]:
    return json.loads(TWO_LENGTHS.read_text(encoding="utf-8"))


# ------------------------------------------------------------------- the spelling


@pytest.mark.parametrize(
    ("move", "phrase"),
    [
        (sweeper(4, reach=3), "may not exceed range (3)"),
        (sweeper(0, reach=3), "sweep_range (unlimited) may not exceed range (3)"),
        (sweeper(-1), "non-negative integer"),
        (sweeper(17), "may not exceed 16"),
        ({**sweeper(None), "sweep_range": True}, "non-negative integer"),
        ({"kind": "slide", "vectors": [[1, 1]], "sweep_range": 2}, "only applies to a sweep"),
    ],
)
def test_the_validator_refuses_what_it_cannot_honour(move: dict[str, Any], phrase: str) -> None:
    problems = validate_spec(probe(move, "a2"))
    assert any(phrase in problem and "sweep_range" in problem for problem in problems), problems


@pytest.mark.parametrize(("sweep_range", "reach"), [(1, 3), (2, 3), (3, 0), (16, 0)])
def test_the_validator_accepts_n_up_to_m(sweep_range: int, reach: int) -> None:
    assert validate_spec(probe(sweeper(sweep_range, reach), "a2")) == []


@pytest.mark.parametrize(("sweep_range", "reach"), [(3, 3), (0, 0)])
def test_n_equal_to_m_is_the_sweep_without_the_key(sweep_range: int, reach: int) -> None:
    written = probe(sweeper(sweep_range, reach), "a2")
    bare = probe(sweeper(None, reach), "a2")
    assert spec_hash(written) == spec_hash(bare)
    assert "sweep_range" not in json.dumps(normal_form(written))
    assert "sweep_range" not in json.dumps(current_spelling(written))
    assert canonical_tables(compile_spec(written)) == canonical_tables(compile_spec(bare))


def test_n_shorter_than_m_is_a_new_game() -> None:
    assert spec_hash(probe(sweeper(2), "a2")) != spec_hash(probe(sweeper(None), "a2"))


def test_a_length_the_line_never_reaches_compiles_to_the_whole_sweep() -> None:
    """From a2 the longest diagonal is six steps, so a sweep of seven clears all of it."""
    long_one = probe(sweeper(7), "a2")
    assert spec_hash(long_one) != spec_hash(probe(sweeper(None), "a2"))  # spelled apart
    assert _unkeyed(long_one) == _unkeyed(probe(sweeper(None), "a2"))


def _unkeyed(spec: dict[str, Any]) -> dict[str, Any]:
    """The tables but the Zobrist keys, which are seeded by the spec's name."""
    return {
        key: value for key, value in canonical_tables(compile_spec(spec)).items()
        if not key.startswith("zobrist")
    }


@pytest.mark.parametrize(("path", "name"), sorted(STORED.items()))
def test_the_stored_sweep_games_keep_their_names(path: str, name: str) -> None:
    assert spec_hash(load_spec(ROOT / path)) == name


# ------------------------------------------------------------------- the rule, by hand


def test_past_its_length_the_sweep_slides() -> None:
    """X on a2, its own piece on b3 (step 1), an enemy on d5 (step 3). Clearing two, it
    sweeps b3 to land on c4 or take d5, and d5 ends the ride; clearing three, d5 is swept
    too and the line runs to g8."""
    extra = (blocker("b3", 0), blocker("d5", 1))
    assert targets(probe(sweeper(2), "a2", extra), "a2") == {"b1", "c4", "d5"}
    assert targets(probe(sweeper(3), "a2", extra), "a2") == {"b1", "c4", "d5", "e6", "f7", "g8"}
    assert targets(probe(sweeper(None), "a2", extra), "a2") == {
        "b1", "c4", "d5", "e6", "f7", "g8"
    }


def test_an_own_piece_on_the_ride_blocks_it() -> None:
    extra = (blocker("e6", 0),)
    assert targets(probe(sweeper(2), "a2", extra), "a2") == {"b1", "b3", "c4", "d5"}


def test_it_may_land_anywhere_inside_its_sweep() -> None:
    """The reading Q1063 settles: as before, a sweep may stop at any square it clears."""
    extra = (blocker("b3", 1), blocker("c4", 1))
    assert targets(probe(sweeper(3), "a2", extra), "a2") >= {"b3", "c4", "d5"}


def test_check_through_the_sweep_and_along_the_ride() -> None:
    """The enemy king on a8 and X on g2: the line g2-a8 is six steps. Clearing three, X
    gives check over pieces only on its first three steps; a piece on the ride blocks."""
    def in_check(sweep_range: int, at: str) -> bool:
        spec = probe(sweeper(sweep_range), "g2", (blocker(at, 0),))
        engine = engine_of(spec)
        return engine.in_check(engine.initial_state(), 1)

    assert in_check(3, "e4")  # step 2, swept
    assert not in_check(3, "c6")  # step 4, on the ride
    assert in_check(4, "c6")
    assert in_check(0, "b7")  # the whole line sweeps


def test_fairy_stockfish_refuses_it_with_the_sweep() -> None:
    with pytest.raises(Untranslatable, match="sweep"):
        translate(two_lengths())


# ------------------------------------------------------------------- the oracle


def _oracle_position(engine: Engine, state: GameState, n_types: int) -> sweep_oracle.Position:
    ep = state.ep_target if state.ep_target != NO_SQUARE else None
    victim = state.ep_victim if state.ep_victim != NO_SQUARE else None
    return sweep_oracle.position_of(
        state.board, state.side_to_move, state.unmoved, n_types, ep, victim
    )


def _pairs_of_engine(engine: Engine, state: GameState) -> Counter[tuple[int, int]]:
    return Counter(decode_move(move)[:2] for move in engine.legal_moves(state))


def _pairs_of_oracle(oracle: sweep_oracle.Oracle, pos: sweep_oracle.Position) -> Counter[tuple[int, int]]:
    return Counter((origin, target) for origin, target, _, _ in oracle.legal(pos))


@pytest.mark.parametrize("seed", range(6))
def test_the_core_agrees_with_the_oracle_over_random_games(seed: int) -> None:
    spec = two_lengths()
    engine = engine_of(spec)
    oracle = sweep_oracle.Oracle(spec)
    n_types = len(spec["pieces"])
    rng = random.Random(seed)
    state = engine.initial_state()
    for _ in range(60):
        moves = engine.legal_moves(state)
        pos = _oracle_position(engine, state, n_types)
        assert _pairs_of_engine(engine, state) == _pairs_of_oracle(oracle, pos), (
            seed, state.ply
        )
        if not moves or engine.terminal(state, moves) is not None:
            break
        state = engine.apply(state, rng.choice(moves))


# The core's count from the start, checked against the oracle's own walk (depths 1-3 here,
# 4 in the slow tier).
PERFT = {1: 30, 2: 865, 3: 27_017, 4: 834_368}


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_perft_against_the_oracle(depth: int) -> None:
    spec = two_lengths()
    engine = engine_of(spec)
    oracle = sweep_oracle.Oracle(spec)
    state = engine.initial_state()
    pos = _oracle_position(engine, state, len(spec["pieces"]))
    assert engine.perft(state, depth) == PERFT[depth]
    if depth <= 2:
        assert oracle.perft(pos, depth) == PERFT[depth]


@pytest.mark.slow
@pytest.mark.parametrize("depth", [3, 4])
def test_perft_against_the_oracle_deep(depth: int) -> None:
    spec = two_lengths()
    engine = engine_of(spec)
    state = engine.initial_state()
    assert engine.perft(state, depth) == PERFT[depth]
    if depth == 3:
        oracle = sweep_oracle.Oracle(spec)
        assert oracle.perft(_oracle_position(engine, state, len(spec["pieces"])), depth) == PERFT[3]


# The stored sweep games' perft(4), taken by `bench` on the core before Q1063.
STORED_PERFT_4 = {
    "archive/chess-sweeping-bishops.json": 843_104,
    "archive/short-sweeping-bishop-chess.json": 525_345,
}


@pytest.mark.slow
@pytest.mark.parametrize(("path", "leaves"), sorted(STORED_PERFT_4.items()))
def test_the_stored_sweep_games_play_as_before(path: str, leaves: int) -> None:
    engine = engine_of(load_spec(ROOT / path))
    assert engine.perft(engine.initial_state(), 4) == leaves
