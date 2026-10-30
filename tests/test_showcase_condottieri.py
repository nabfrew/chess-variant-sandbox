"""V1 of *the condottieri* (TC10, `docs/plan-turncoat.md` §8.3): the mechanism on the showcase's own file.

`tests/_retired_the_condottieri.json` (`retired_specs.RETIRED_PREVIEWS`; it missed V2's hire and bribe bars
and retired, §8.3) is chess with four mercenaries nobody owns on b4, g4, b5 and g5,
which stand until hired: a player with one of their pieces next to a mercenary may spend the
whole move paying its price (3) for it, and may bribe an enemy mercenary next to one of theirs at
twice the price; captures pay bounties into the purse. Over seeded random games, leaning toward
hires, bribes and captures and half of them starting rich: the core's legal list equals the
declared oracle's (`tests/neutral_oracle.py`: the hires and bribes worked out from the board, a
hirer's piece next to the target) after every ply; every hire or bribe offered is of the priced
mercenary, never a royal or an unpriced piece; a hire spends the price, a bribe twice it, and the
piece becomes the buyer's where it stands; the incremental key is the full scan's. A bribed facing
piece keeps its heading on the scratch spec of `tests/test_purse_hire.py` (§6). Three games fast,
twenty of up to 400 plies slow.
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
CONDOTTIERI = ROOT / "tests" / "_retired_the_condottieri.json"
NOBODY = 2


def _spec() -> dict[str, Any]:
    return json.loads(CONDOTTIERI.read_text(encoding="utf-8"))


def _walk(seed: int, plies: int) -> tuple[int, int]:
    spec = _spec()
    engine = Engine(compile_spec(spec))
    oracle = NeutralOracle(spec, engine)
    ids = [piece["id"] for piece in spec["pieces"]]
    prices = {i: piece.get("price", 0) for i, piece in enumerate(spec["pieces"])}
    rng = random.Random(seed)
    state = engine.initial_state()
    if seed % 2:  # half the walks start rich, so bribes are affordable early
        state = state.with_(purse=(rng.randrange(6, PURSE_MAX + 1), rng.randrange(6, PURSE_MAX + 1)))
        state = state.with_(key=engine._compute_hash(state))
        state = state.with_(repetition=(state.key,))
    hired = bribed = 0
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
        if move_type(move) == MOVE_HIRE:
            square = move_to(move)
            before = state.board[square]
            price = prices[engine.spec.type_of(before)]
            if engine.spec.owner_of(before) == NOBODY:
                hired += 1
                assert after.purse[mover] == state.purse[mover] - price, (seed, state.ply)
            else:
                assert engine.spec.owner_of(before) == 1 - mover
                bribed += 1
                assert after.purse[mover] == state.purse[mover] - 2 * price, (seed, state.ply)
            assert engine.spec.owner_of(after.board[square]) == mover
            assert engine.spec.type_of(after.board[square]) == engine.spec.type_of(before)
            assert after.purse[1 - mover] == state.purse[1 - mover]
        assert after.key == engine._compute_hash(after), (seed, state.ply)
        state = after
    return hired, bribed


def test_the_file_is_the_registered_game():
    assert spec_hash(_spec()) == "80134a4bced2"


def test_three_walks_agree():
    hired, bribed = (sum(x) for x in zip(*(_walk(seed, 60) for seed in range(3)), strict=True))
    assert hired > 0


def test_a_bribed_facing_piece_keeps_its_heading():
    test_purse_hire.TestAChangeOfSide().test_a_bribed_facing_piece_keeps_its_heading()


@pytest.mark.slow
def test_twenty_walks_agree():
    hired, bribed = (sum(x) for x in zip(*(_walk(seed, 400) for seed in range(20)), strict=True))
    assert hired > 20 and bribed > 5
