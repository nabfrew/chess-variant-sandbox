"""V1 of *the free companies* (TC10b, `docs/plan-turncoat.md` §8.5): the mechanism on the showcase's own file.

`archive/the-free-companies.json` is chess with four mercenaries nobody owns on c4, f4, c5 and
f5, which stand until hired: a player with one of their pieces next to a mercenary may spend the
whole move paying its price (2) for it, and may bribe an enemy mercenary next to one of theirs at
twice the price; captures pay bounties into the purse, and a mercenary carries none. TC10's walks
(`tests/test_showcase_condottieri.py`) on this file: over seeded random games, leaning toward
hires, bribes and captures and half of them starting rich, the core's legal list equals the
declared oracle's (`tests/neutral_oracle.py`) after every ply; every hire or bribe offered is of
the priced mercenary, never a royal or an unpriced piece; a hire spends the price, a bribe twice
it, and the piece becomes the buyer's where it stands; the incremental key is the full scan's.
Here also: taking a mercenary pays the taker nothing. A bribed facing piece keeps its heading on
the scratch spec of `tests/test_purse_hire.py` (§6). Three games fast, twenty of up to 400 plies
slow.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import MOVE_HIRE, move_to, move_type
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.schema import PURSE_MAX
from tests import test_purse_hire
from tests.neutral_oracle import NeutralOracle

ROOT = Path(__file__).resolve().parent.parent
FREE_COMPANIES = ROOT / "archive" / "the-free-companies.json"
NOBODY = 2


def _spec() -> dict[str, Any]:
    return json.loads(FREE_COMPANIES.read_text(encoding="utf-8"))


def _walk(seed: int, plies: int) -> tuple[int, int, int]:
    spec = _spec()
    engine = Engine(compile_spec(spec))
    oracle = NeutralOracle(spec, engine)
    ids = [piece["id"] for piece in spec["pieces"]]
    prices = {i: piece.get("price", 0) for i, piece in enumerate(spec["pieces"])}
    rng = random.Random(seed)
    state = engine.initial_state()
    if seed % 2:  # half the walks start rich, so bribes are affordable early
        state = state.with_(purse=(rng.randrange(4, PURSE_MAX + 1), rng.randrange(4, PURSE_MAX + 1)))
        state = state.with_(key=engine._compute_hash(state))
        state = state.with_(repetition=(state.key,))
    hired = bribed = taken = 0
    for _ in range(plies):
        moves = engine.legal_moves(state)
        if engine.terminal(state, moves) is not None:
            break
        assert sorted(moves) == oracle.legal_moves(state), (seed, state.ply)
        bought = [m for m in moves if move_type(m) == MOVE_HIRE]
        for move in bought:
            value = state.board[move_to(move)]
            piece_type = engine.spec.type_of(value)
            assert ids[piece_type] == "M", (seed, state.ply)
            assert piece_type not in engine.spec.royal_types and prices[piece_type]
        loud = bought + [m for m in moves if state.board[move_to(m)]]
        move = rng.choice(loud) if loud and rng.random() < 0.5 else rng.choice(moves)
        mover = state.side_to_move
        after = engine.apply(state, move)
        target = state.board[move_to(move)]
        if move_type(move) == MOVE_HIRE:
            square = move_to(move)
            price = prices[engine.spec.type_of(target)]
            if engine.spec.owner_of(target) == NOBODY:
                hired += 1
                assert after.purse[mover] == state.purse[mover] - price, (seed, state.ply)
            else:
                assert engine.spec.owner_of(target) == 1 - mover
                bribed += 1
                assert after.purse[mover] == state.purse[mover] - 2 * price, (seed, state.ply)
            assert engine.spec.owner_of(after.board[square]) == mover
            assert engine.spec.type_of(after.board[square]) == engine.spec.type_of(target)
            assert after.purse[1 - mover] == state.purse[1 - mover]
        elif target and ids[engine.spec.type_of(target)] == "M":
            taken += 1  # a mercenary taken, whoever's: no bounty on it
            assert after.purse == state.purse, (seed, state.ply)
        assert after.key == engine._compute_hash(after), (seed, state.ply)
        state = after
    return hired, bribed, taken


def test_the_file_is_the_registered_game():
    assert spec_hash(_spec()) == "f858b7a63dea"


def test_three_walks_agree():
    hired, _, _ = (sum(x) for x in zip(*(_walk(seed, 60) for seed in range(3)), strict=True))
    assert hired > 0


def test_a_bribed_facing_piece_keeps_its_heading():
    test_purse_hire.TestAChangeOfSide().test_a_bribed_facing_piece_keeps_its_heading()


@pytest.mark.slow
def test_twenty_walks_agree():
    hired, bribed, taken = (sum(x) for x in zip(*(_walk(seed, 400) for seed in range(20)), strict=True))
    assert hired > 20 and bribed > 5 and taken > 5
