"""Schema 30 (Q999a): the core held to the declared sight oracle over random games.

`docs/archive/plan-passive.md` §3.6: the core and `tests/sight_oracle.py` agree move list by move list
over seeded random games of S1 to S3 (`tests/_sight_madrasi.json`, `_sight_sheltering_king`,
`_sight_daunting_knights`, the showcases of §6.1 as test specs; Q999d archives them), an
alternating spec with checkmate, castling, en passant and a shield on the royal
(`_sight_wardens`), one with royal capture and drops (`_sight_drops`) and one with royal capture and contacts
(`_sight_contacts`; the two are refused together), a round spec
(`_sight_rounds`), a speed spec (`_sight_speed`), a spy spec with rests (`_sight_spy`) and an
alternating rest spec (`_sight_rest`), and since generation 37 (N2, `docs/plan-npc.md` §4.2) wild pieces that see:
wolves' chess with freezing wolves and shielding dryads (`_sight_wild_wolves`), and a 6x6
checkmate game with a disarming wolf beside the players' wardens (`_sight_wild_small`). At every position the check answer is held to the
oracle's too, and the incremental key to the slow definition. Twenty games each (three of
each 8x8 spec in the fast suite, its twenty under `slow`); and S1's perft is frozen at depths 1 to 3, after the oracle's own
count agreed with it.
"""

from __future__ import annotations

import json
import random
from functools import cache
from pathlib import Path

import pytest
from sight_oracle import SightOracle

from chess_variant_sandbox.engine.state import GameState

HERE = Path(__file__).parent

WALKED = (
    "_sight_madrasi", "_sight_sheltering_king", "_sight_daunting_knights", "_sight_wardens",
    "_sight_wild_wolves", "_sight_drops", "_sight_contacts", "_sight_rounds", "_sight_speed",
    "_sight_spy", "_sight_rest", "_sight_wild_small",
)


@cache
def oracle(name: str) -> SightOracle:
    return SightOracle(json.loads((HERE / f"{name}.json").read_text(encoding="utf-8")))


def walk(name: str, games: int, plies: int = 200) -> int:
    """Play ``games`` seeded random games of ``name`` through the core, holding every list to
    the oracle's; the number of positions compared, so a walk that compared nothing fails."""
    judge = oracle(name)
    engine = judge.engine
    compared = 0
    for seed in range(games):
        rng = random.Random(seed)
        state = engine.initial_state()
        for _ in range(plies):
            if engine.terminal(state) is not None:
                break
            if engine.is_round(state):
                lists = [engine.legal_moves_for(state, player) for player in (0, 1)]
                for player in (0, 1):
                    assert lists[player] == judge.legal_moves_for(state, player), (
                        name, seed, state.ply, player,
                    )
                compared += 1
                if not lists[0] or not lists[1]:
                    break
                state = engine.apply_round(state, rng.choice(lists[0]), rng.choice(lists[1]))
            else:
                moves = engine.legal_moves(state)
                assert moves == judge.legal_moves(state), (name, seed, state.ply)
                if judge.checkmate:
                    side = state.side_to_move
                    assert engine.in_check(state) == judge.royal_attacked(state, side)
                compared += 1
                if not moves:
                    break
                state = engine.apply(state, rng.choice(moves))
            assert state.key == engine._compute_hash(state)
    return compared


# The 8x8 specs cost about a second for three games; the 5x5 ones play all twenty in the fast
# suite, since their games are short.
EIGHT_BY_EIGHT = frozenset(WALKED[:5])
# A small spec whose games run long (the wild one's, which no capture shortens): eight games in
# the fast suite, its twenty under `slow`.
LONG_SMALL = frozenset({"_sight_wild_small"})


@pytest.mark.parametrize("name", WALKED)
def test_the_core_agrees_with_the_oracle_over_a_few_games(name: str) -> None:
    if name in EIGHT_BY_EIGHT:
        assert walk(name, games=3, plies=100) > 30
    elif name in LONG_SMALL:
        assert walk(name, games=8) > 100
    else:
        assert walk(name, games=20) > 100


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(EIGHT_BY_EIGHT | LONG_SMALL))
def test_the_core_agrees_with_the_oracle_over_twenty_games(name: str) -> None:
    assert walk(name, games=20) > 200


def oracle_perft(judge: SightOracle, state: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    moves = judge.legal_moves(state)
    if depth == 1:
        return len(moves)
    return sum(oracle_perft(judge, judge.twin.apply(state, move), depth - 1) for move in moves)


# S1 (Madrasi without its en passant observation rule), frozen after the oracle agreed: chess
# is 20, 400, 8,902; the 28 fewer leaves at depth 3 are the 1.e4 d5, 1.d4 e5-type pairs, a
# pawn frozen by the enemy pawn diagonally beside it.
S1_PERFT = {1: 20, 2: 400, 3: 8874}


@pytest.mark.parametrize("depth", sorted(S1_PERFT))
def test_s1_perft_is_frozen(depth: int) -> None:
    engine = oracle("_sight_madrasi").engine
    assert engine.perft(engine.initial_state(), depth) == S1_PERFT[depth]


def test_s1_perft_agrees_with_the_oracle_at_depth_two() -> None:
    judge = oracle("_sight_madrasi")
    assert oracle_perft(judge, judge.engine.initial_state(), 2) == S1_PERFT[2]


@pytest.mark.slow
def test_s1_perft_agrees_with_the_oracle_at_depth_three() -> None:
    judge = oracle("_sight_madrasi")
    assert oracle_perft(judge, judge.engine.initial_state(), 3) == S1_PERFT[3]
