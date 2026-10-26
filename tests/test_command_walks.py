"""Q1007c: the `command` effect's core held to the declared sight oracle over random games.

`docs/plan-schema-31.md` §3.7 and §7 (Q1007c's acceptance): the core and `tests/sight_oracle.py`
agree move list by move list over seeded random games of K1 (`tests/_command_k1.json`, chess with
commanding knights), K2 (`_command_k2.json`, chess with a spellbinding king, `captures_own`) and a
test spec with long turns, drops and a freeze beside two commands (`_command_turns.json`), and
chess with turning pawns and commanding knights (`_command_facing.json`, a commanded turn). The
commanded moves follow the commander's own; their order among themselves is the core's walk, so
both lists are compared through `sight_oracle.normalise`. At every position the check answer is
held to the oracle's too, and the incremental key to the slow definition. The walks are biased
toward commanded moves (one in three plies plays one where one is offered, and one in two a
turncoat capture where one is offered), so the positions they reach are the ones the effect
makes. K1's perft is frozen at depths 1 to 3 after the oracle's own
count agreed with it.
"""

from __future__ import annotations

import json
import random
from functools import cache
from pathlib import Path

import pytest
from sight_oracle import SightOracle, normalise

from chess_variant_sandbox.engine.compile import SIGHT_COMMAND_OWN
from chess_variant_sandbox.engine.moves import MOVE_COMMANDED, decode_move
from chess_variant_sandbox.engine.state import GameState

HERE = Path(__file__).parent

WALKED = ("_command_k1", "_command_k2", "_command_turns", "_command_facing")
EIGHT_BY_EIGHT = frozenset({"_command_k1", "_command_k2", "_command_facing"})


@cache
def oracle(name: str) -> SightOracle:
    return SightOracle(json.loads((HERE / f"{name}.json").read_text(encoding="utf-8")))


def walk(name: str, games: int, plies: int = 200, judge: SightOracle | None = None) -> tuple[int, int]:
    """Play ``games`` seeded random games of ``name`` through the core, holding every list to the
    oracle's; the number of positions compared and of commanded moves played, so a walk that
    compared nothing, or never commanded, fails. ``judge`` walks a spec held outside `tests/`
    (an `archive/` showcase), ``name`` then only labelling a failure."""
    judge = judge or oracle(name)
    engine = judge.engine
    compared = commanded = 0
    for seed in range(games):
        rng = random.Random(seed)
        state = engine.initial_state()
        for _ in range(plies):
            if engine.terminal(state) is not None:
                break
            moves = engine.legal_moves(state)
            assert normalise(moves) == judge.legal_moves(state), (name, seed, state.ply)
            if judge.checkmate:
                side = state.side_to_move
                assert engine.in_check(state) == judge.royal_attacked(state, side)
            compared += 1
            if not moves:
                break
            ordered = [m for m in moves if m & MOVE_COMMANDED]
            turncoat = [
                m for m in ordered
                if decode_move(m)[0] != decode_move(m)[1] and state.board[decode_move(m)[1]]
            ]
            if turncoat and rng.random() < 1 / 2:
                move = rng.choice(turncoat)
            else:
                move = rng.choice(ordered if ordered and rng.random() < 1 / 3 else moves)
            if move & MOVE_COMMANDED:
                commanded += 1
                _commanded_move_is_sound(judge, state, move)
            state = engine.apply(state, move)
            assert state.key == engine._compute_hash(state)
    return compared, commanded


def _commanded_move_is_sound(judge: SightOracle, state: GameState, move: int) -> None:
    """§3.3, §3.4: a commanded piece is the opponent's, never royal, and takes none of the
    commander's pieces."""
    spec = judge.compiled
    origin, target, _, _ = decode_move(move)
    piece = state.board[origin]
    assert spec.owner_of(piece) == 1 - state.side_to_move
    assert spec.type_of(piece) not in spec.royal_types
    victim = state.board[target]
    if victim and target != origin:
        sight = spec.sight
        assert sight is not None
        assert any(entry.effect & SIGHT_COMMAND_OWN for row in sight.entries for entry in row), (
            "a turncoat capture in a game without captures_own"
        )
        assert spec.owner_of(victim) == 1 - state.side_to_move
        assert spec.type_of(victim) not in spec.royal_types


@pytest.mark.parametrize("name", WALKED)
def test_the_core_agrees_with_the_oracle_over_a_few_games(name: str) -> None:
    if name in EIGHT_BY_EIGHT:
        compared, commanded = walk(name, games=3, plies=100)
        assert compared > 30
    else:
        compared, commanded = walk(name, games=20)
        assert compared > 100
    assert commanded > 0


@pytest.mark.slow
@pytest.mark.parametrize("name", WALKED)
def test_the_core_agrees_with_the_oracle_over_twenty_games(name: str) -> None:
    compared, commanded = walk(name, games=20)
    assert compared > 200
    assert commanded > 20


def oracle_perft(judge: SightOracle, state: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    moves = judge.legal_moves(state)
    if depth == 1:
        return len(moves)
    return sum(oracle_perft(judge, judge.twin.apply(state, move), depth - 1) for move in moves)


# K1, frozen after the oracle agreed (depth 2 in the fast suite, 3 under `slow`): chess is 20,
# 400, 8,902. At depth 3 the knight that came out on move one commands what it eyes in the
# other army: Nc3 (or Na3, Nf3, Nh3) commands the enemy's d5/b5/e5/g5 pawns only once one has
# arrived there, so the six extra leaves are commanded pawn steps after 1.Nc3 d5, 1.Nf3 e5 and
# the like.
K1_PERFT = {1: 20, 2: 400, 3: 8908}


@pytest.mark.parametrize("depth", sorted(K1_PERFT))
def test_k1_perft_is_frozen(depth: int) -> None:
    engine = oracle("_command_k1").engine
    assert engine.perft(engine.initial_state(), depth) == K1_PERFT[depth]


def test_k1_perft_agrees_with_the_oracle_at_depth_two() -> None:
    judge = oracle("_command_k1")
    assert oracle_perft(judge, judge.engine.initial_state(), 2) == K1_PERFT[2]


@pytest.mark.slow
def test_k1_perft_agrees_with_the_oracle_at_depth_three() -> None:
    judge = oracle("_command_k1")
    assert oracle_perft(judge, judge.engine.initial_state(), 3) == K1_PERFT[3]
