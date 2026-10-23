"""Schema 29 (Q983a): the three rounds-only resting games, rewritten, are the old games move for
move (`docs/plan-rest.md` §4.5).

`tests/rest_equivalence.json` was captured on main's core before the bump (its generator is
`tests/rest_equivalence.py`): each old game's joint perft at depths 1 to 3 and twenty seeded
random walks. Here the rewritten files replay them: every round's two moves (the same seeded
choice over the same legal lists), each player's rested square, the result, and the joint
perft must be the fixture's exactly.

**The keys.** A game's Zobrist table is seeded from its mechanics hash
(`compile._build_zobrist`), and the rewrite is a new hash, so no key can be the fixture's.
What a key is *for* is telling positions apart, so the walk holds the next best thing: the
old keys and the new keys are the same partition of the positions met, a bijection fixed
across all twenty walks of a game (two positions share a new key exactly when they shared
an old one), which is what the repetition rule and a transposition table read. The
incremental key is also held to the slow definition after every round.
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import pytest
from rest_equivalence import FIXTURE, GAMES, MAX_PLIES, PERFT_DEPTHS, ROOT, rested_squares, rewrite

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.spec.spelling import current_spelling

FAST_WALKS = 4


@cache
def fixture() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return loaded


@cache
def engine_for(old_hash: str) -> Engine:
    return Engine(json.loads((ROOT / GAMES[old_hash]).read_text(encoding="utf-8")))


class KeyPartition:
    """Old keys to new keys, held to a bijection."""

    def __init__(self) -> None:
        self.forward: dict[int, int] = {}
        self.backward: dict[int, int] = {}

    def hold(self, old: int, new: int) -> None:
        assert self.forward.setdefault(old, new) == new, "one old position, two new keys"
        assert self.backward.setdefault(new, old) == old, "two old positions, one new key"


def replay(engine: Engine, walk: dict[str, Any], keys: KeyPartition) -> None:
    import random

    rng = random.Random(walk["seed"])
    state = engine.initial_state()
    keys.hold(walk["start"], state.key)
    for index, (first, second, old_key, rested) in enumerate(walk["rounds"]):
        assert state.ply < MAX_PLIES
        assert engine.terminal(state) is None, f"round {index}: the game ended early"
        chosen = (
            rng.choice(engine.legal_moves_for(state, 0)),
            rng.choice(engine.legal_moves_for(state, 1)),
        )
        assert chosen == (first, second), f"seed {walk['seed']} round {index}: other moves"
        state = engine.apply_round(state, first, second)
        assert rested_squares(state) == rested, f"seed {walk['seed']} round {index}: rest"
        assert state.key == engine._compute_hash(state)
        keys.hold(old_key, state.key)
    outcome = engine.terminal(state)
    result = None if outcome is None else [outcome.winner, outcome.reason]
    assert result == walk["result"]
    if result is None:
        assert state.ply >= MAX_PLIES


@pytest.mark.parametrize("old_hash", sorted(GAMES))
def test_the_rewritten_file_is_a_new_hash_and_valid(old_hash: str) -> None:
    spec = json.loads((ROOT / GAMES[old_hash]).read_text(encoding="utf-8"))
    assert spec_hash(spec) != old_hash
    assert spec["schema_version"] >= 29  # 29 for the rest; Q1008b re-spelt it at the current
    assert validate_spec(spec) == []
    assert all(piece.get("rest") == 1 for piece in spec["pieces"])


@pytest.mark.parametrize("old_hash", sorted(GAMES))
def test_the_joint_perft_is_the_old_games_to_depth_two(old_hash: str) -> None:
    engine = engine_for(old_hash)
    state = engine.initial_state()
    assert [engine.perft(state, depth) for depth in (1, 2)] == fixture()[old_hash]["perft"][:2]


@pytest.mark.parametrize("old_hash", sorted(GAMES))
def test_a_few_walks_replay_move_for_move(old_hash: str) -> None:
    keys = KeyPartition()
    for walk in fixture()[old_hash]["walks"][:FAST_WALKS]:
        replay(engine_for(old_hash), walk, keys)


@pytest.mark.slow
@pytest.mark.parametrize("old_hash", sorted(GAMES))
def test_every_walk_and_the_deep_perft_replay(old_hash: str) -> None:
    engine = engine_for(old_hash)
    state = engine.initial_state()
    assert [engine.perft(state, depth) for depth in PERFT_DEPTHS] == fixture()[old_hash]["perft"]
    keys = KeyPartition()
    for walk in fixture()[old_hash]["walks"]:
        replay(engine, walk, keys)


def test_the_retired_preview_is_kept_as_the_refusal_fixture_and_rewrites_to_the_test_spec() -> None:
    """`_retired_gardner_in_rounds_resting.json` stays byte for byte (§4.5): refused by
    admission and by the loader, and its rewrite -- this file's and the converter's -- is
    `_simultaneous_resting.json`'s game."""
    from chess_variant_sandbox.spec.schema import SpecError

    retired = json.loads(
        (ROOT / "tests" / "_retired_gardner_in_rounds_resting.json").read_text(encoding="utf-8")
    )
    assert spec_hash(retired) == "4da669bb3d50"
    assert validate_spec(retired)
    with pytest.raises(SpecError):
        Engine(retired)
    resting = json.loads((ROOT / GAMES["4da669bb3d50"]).read_text(encoding="utf-8"))
    assert spec_hash(rewrite(retired)) == spec_hash(resting)
    assert spec_hash(current_spelling(retired)) == spec_hash(resting)
