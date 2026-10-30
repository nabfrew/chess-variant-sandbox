"""V1 of *the indemnity* (TC9c, `docs/plan-turncoat.md` §8.6): the mechanism on the showcase's own file.

`tests/_retired_the_indemnity.json` (`retired_specs.RETIRED_PREVIEWS`; it missed V2's market clause and retired, §8.6) is chess in which a knight, bishop, rook or queen taken pays its
bounty (1, 1, 2, 4) into the purse of the player who lost it (`rules.purse.bounty_to:
"loser"`), and a whole move may buy a pawn, knight or rook onto an empty square of the buyer's
first rank, each player starting with two coins. Over seeded random games, leaning toward buys
and captures, half of them starting near the cap: the buys offered are the oracle's (every
affordable market type on every empty square of the buyer's own first rank; out of check all
of them, in check a subset), so a buy never lands on an occupied square or outside the region;
a buy spends its price and leaves the opponent's purse alone; a capture pays its bounty to the
side that lost the piece and nothing to the taker, never past `PURSE_MAX`; the incremental key
is the full scan's after every ply, the purse in it; and the core's perft, which makes and
unmakes every move, equals a count by copying `apply`, so unmake puts the loser's coins back.
Three games fast, twenty of up to 400 plies slow. The walks are TC9b's
(`tests/test_showcase_levy.py`), with the payee turned round.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import MOVE_BUY, decode_move, move_to, move_type
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.schema import PURSE_MAX

ROOT = Path(__file__).resolve().parent.parent
INDEMNITY = ROOT / "tests" / "_retired_the_indemnity.json"


def _spec() -> dict[str, Any]:
    return json.loads(INDEMNITY.read_text(encoding="utf-8"))


def _buys(engine: Engine, state: GameState) -> set[int]:
    """The oracle: every affordable market type on every empty square of the mover's first rank."""
    purse = engine.spec.purse
    assert purse is not None
    side = state.side_to_move
    width, height = engine.spec.width, engine.spec.height
    rank = 0 if side == 0 else height - 1
    out = set()
    for piece_type, price in zip(purse.market, purse.price, strict=True):
        if price > state.purse[side]:
            continue
        for square in range(rank * width, (rank + 1) * width):
            if state.board[square] == 0:
                out.add(square | (square << 8) | (MOVE_BUY << 16) | (piece_type << 20))
    return out


def _count(engine: Engine, state: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    if engine.terminal(state) is not None:
        return 0
    return sum(_count(engine, engine.apply(state, m), depth - 1) for m in engine.legal_moves(state))


def _bounty(engine: Engine, spec: dict[str, Any], value: int) -> int:
    return int(spec["pieces"][engine.spec.type_of(value)].get("bounty", 0))


def _walk(seed: int, plies: int) -> tuple[int, int]:
    spec = _spec()
    engine = Engine(compile_spec(spec))
    purse = engine.spec.purse
    assert purse is not None and purse.to_loser
    rng = random.Random(seed)
    state = engine.initial_state()
    if seed % 2:  # half the walks start rich, so the cap is met
        state = state.with_(purse=(rng.randrange(PURSE_MAX + 1), PURSE_MAX - rng.randrange(6)))
        state = state.with_(key=engine._compute_hash(state))
        state = state.with_(repetition=(state.key,))
    bought = paid = 0
    for _ in range(plies):
        moves = engine.legal_moves(state)
        if engine.terminal(state, moves) is not None:
            break
        offered = {m for m in moves if move_type(m) == MOVE_BUY}
        expected = _buys(engine, state)
        if engine.in_check(state):
            assert offered <= expected, (seed, state.ply)
        else:
            assert offered == expected, (seed, state.ply)
        if state.ply % 25 == 0:
            assert engine.perft(state, 2, stop_at_terminal=True) == _count(engine, state, 2), (seed, state.ply)
        loud = [m for m in moves if move_type(m) == MOVE_BUY or state.board[move_to(m)]]
        move = rng.choice(loud) if loud and rng.random() < 0.4 else rng.choice(moves)
        mover = state.side_to_move
        after = engine.apply(state, move)
        taken = state.board[move_to(move)]
        if move_type(move) == MOVE_BUY:
            bought += 1
            price = purse.price[purse.market.index(decode_move(move)[3])]
            assert after.purse[mover] == state.purse[mover] - price, (seed, state.ply)
            assert after.purse[1 - mover] == state.purse[1 - mover], (seed, state.ply)
        else:
            assert after.purse[mover] == state.purse[mover], (seed, state.ply)
            due = state.purse[1 - mover]
            if taken and engine.spec.owner_of(taken) == 1 - mover:
                bounty = _bounty(engine, spec, taken)
                paid += int(bounty > 0)
                due = min(PURSE_MAX, due + bounty)
            assert after.purse[1 - mover] == due, (seed, state.ply)
        assert all(0 <= coins <= PURSE_MAX for coins in after.purse)
        assert after.key == engine._compute_hash(after), (seed, state.ply)
        state = after
    return bought, paid


def test_the_file_is_the_registered_game():
    assert spec_hash(_spec()) == "d9787f06e362"


def test_three_walks_agree():
    bought, paid = (sum(x) for x in zip(*(_walk(seed, 60) for seed in range(3)), strict=True))
    assert bought > 0 and paid > 0


@pytest.mark.slow
def test_twenty_walks_agree():
    bought, paid = (sum(x) for x in zip(*(_walk(seed, 400) for seed in range(20)), strict=True))
    assert bought > 20 and paid > 20
