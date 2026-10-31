"""V1 of *the crowd* (TC8, `docs/plan-turncoat.md` §8.1): the mechanism on the showcase's own file.

`archive/the-crowd.json` is chess with four condottieri nobody owns on c4, f4, c5 and f5, moved by
whichever player has strictly more pieces on the king's step round one. Over seeded random games
the core's legal list equals the declared oracle's (`tests/neutral_oracle.py`: the counts from the
board, a held neutral's moves from the frame-0 twin) after every ply, and so do the holders; a
neutral is never moved on a tie; a held neutral never takes its holder's piece, nor a royal (since
Q1077 it attacks one, so no legal position leaves a king it could take); the
mirrored position (ranks flipped, sides swapped) gives the mirrored holder. Three games fast,
twenty of up to 400 plies slow.
"""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import move_foreign, move_from, move_to
from chess_variant_sandbox.spec.canonical import spec_hash
from tests.neutral_oracle import NeutralOracle

ROOT = Path(__file__).resolve().parent.parent
CROWD = ROOT / "archive" / "the-crowd.json"
NOBODY = 2


def _spec() -> dict[str, Any]:
    return json.loads(CROWD.read_text(encoding="utf-8"))


def _name(square: int, width: int) -> str:
    return f"{chr(ord('a') + square % width)}{square // width + 1}"


def _mirrored(engine: Engine, spec: dict[str, Any], state: Any) -> Any:
    """The position with ranks flipped and the two sides' pieces swapped, nobody's kept."""
    width, height = engine.spec.width, engine.spec.height
    ids = [piece["id"] for piece in spec["pieces"]]
    placement: dict[str, tuple[str, int]] = {}
    for square, value in enumerate(state.board):
        if not value:
            continue
        owner = engine.spec.owner_of(value)
        image = (height - 1 - square // width) * width + square % width
        placement[_name(image, width)] = (
            ids[engine.spec.type_of(value)], owner if owner == NOBODY else 1 - owner,
        )
    mirrored = engine.make_state(
        placement, side_to_move=1 - state.side_to_move, moved=list(placement),
    )
    return dataclasses.replace(mirrored, hands=engine.initial_state().hands)


def _walk(seed: int, plies: int) -> int:
    spec = _spec()
    engine = Engine(compile_spec(spec))
    oracle = NeutralOracle(spec, engine)
    width, height = engine.spec.width, engine.spec.height
    rng = random.Random(seed)
    state = engine.initial_state()
    made = 0
    for _ in range(plies):
        if engine.terminal(state) is not None:
            break
        core = sorted(engine.legal_moves(state))
        assert core == oracle.legal_moves(state), (seed, state.ply)
        mirror = _mirrored(engine, spec, state)
        for square, value in enumerate(state.board):
            if not value or engine.spec.owner_of(value) != NOBODY:
                continue
            holder = engine.controller_of(state, square)
            assert holder == oracle.holder(state, square), (seed, state.ply, square)
            image = (height - 1 - square // width) * width + square % width
            there = engine.controller_of(mirror, image)
            assert there == (-1 if holder == -1 else 1 - holder), (seed, state.ply, square)
            from_here = [m for m in core if move_foreign(m) and move_from(m) == square]
            if holder != state.side_to_move:
                assert from_here == [], (seed, state.ply, square)  # a tie, or the other side's
            for move in from_here:
                target = state.board[move_to(move)]
                if target:
                    assert engine.spec.owner_of(target) == 1 - state.side_to_move
                    assert engine.spec.type_of(target) not in engine.spec.royal_types
        foreign = [m for m in core if move_foreign(m)]
        # Lean toward the neutrals so the walk spends time in the scrum.
        move = rng.choice(foreign) if foreign and rng.random() < 0.5 else rng.choice(core)
        made += bool(move_foreign(move))
        state = engine.apply(state, move)
    return made


def test_the_file_is_the_registered_game():
    # Q1077 (2026-10-31): `takes_royal` on the condottiere; was 961b230bc5eb.
    assert spec_hash(_spec()) == "ea7aa465cbfb"


def test_three_walks_agree():
    assert sum(_walk(seed, 60) for seed in range(3)) > 0


@pytest.mark.slow
def test_twenty_walks_agree():
    assert sum(_walk(seed, 400) for seed in range(20)) > 20
