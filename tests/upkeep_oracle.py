"""The declared Python oracle for a move's cost (H1, Q1125, `docs/plan-hunger.md` §2).

`pieces[].upkeep` and `rules.purse.upkeep`: a move costs its piece's upkeep times the rows from
its owner's first row to the square the piece stands on after it, as the type it is after it,
paid from the purse before the move; a move the purse cannot pay for is not legal. CLAUDE.md
rule 3 allows this module: it is a declared test oracle, and it reads the **spec**, never the
compiled cost table, so a wrong table is a disagreement here and not a shared mistake.

It asks the engine one question answered elsewhere: the moves of the same position in the
**twin** (the spec with every upkeep and the rule removed, the purse kept), which is the game
the filter narrows. The cost of each of those moves is worked out below, by file and rank, and
the oracle's list is the twin's less what the side to move cannot pay for (a buy its price as
well), in the twin's order.

Which square a move pays for (§2.3, D4 as re-cut by Q1130): a step, en passant, a sweep and a
displacement their target; a castle the castling piece's target (the partner travels free); a
rotation its own square; a capture a strike placed ``"stay"`` reaches (rifle, ``effect:
"capture"``) the mover's own square; a buy the bought piece's square (and its price); a hire
and a keep nothing. A promotion pays both arrivals under ``"both"`` and the new type's alone
under ``"new"``. The arms this module serves use the readings it states and no other.
"""

from __future__ import annotations

import copy
from typing import Any

from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.engine.moves import (
    MOVE_BUY,
    MOVE_CASTLE,
    MOVE_DISPLACE,
    MOVE_DROP,
    MOVE_EN_PASSANT,
    MOVE_HIRE,
    MOVE_KEEP,
    MOVE_PROMOTION,
    MOVE_ROTATE,
    decode_move,
)
from chess_variant_sandbox.engine.state import GameState


def twin(spec: dict[str, Any]) -> dict[str, Any]:
    """The game without the move's cost: every ``upkeep`` and ``rules.purse.upkeep`` gone, and
    the purse with them (and the bounties it held) where it had nothing else to spend on. The
    twin's moves of a position do not depend on its purse except through buys, which keep it."""
    out = copy.deepcopy(spec)
    for piece in out["pieces"]:
        piece.pop("upkeep", None)
    purse = out["rules"]["purse"]
    purse.pop("upkeep", None)
    if "market" not in purse and not purse.get("bribe"):
        del out["rules"]["purse"]
        for piece in out["pieces"]:
            piece.pop("bounty", None)
    return out


def _rates(spec: dict[str, Any]) -> list[int]:
    """Each type's upkeep, in declaration order (the compiler's type index)."""
    return [int(piece.get("upkeep", 0)) for piece in spec["pieces"]]


def _promotion_reading(spec: dict[str, Any]) -> str:
    upkeep = spec["rules"]["purse"]["upkeep"]
    return upkeep.get("promotion", "both") if isinstance(upkeep, dict) else "both"


def rows(spec: dict[str, Any], square: int, player: int) -> int:
    """Rows of ``square`` from ``player``'s own first row: the rank for the first player, the
    rank counted from the far side for the second (a mirror and a rotation both flip it)."""
    width, height = spec["board"]["width"], spec["board"]["height"]
    rank = square // width
    return rank if player == 0 else height - 1 - rank


def _stays(spec: dict[str, Any], engine: Engine, state: GameState, kind: int, target: int) -> bool:
    """A rifle's capture: ``rules.strike`` with ``position: "stay"`` and ``effect: "capture"``
    reaching the victim (not immune, the mover in ``by`` when it is stated)."""
    strike = spec["rules"].get("strike")
    if not isinstance(strike, dict) or strike.get("position") != "stay":
        return False
    assert strike.get("effect", "capture") == "capture", "the oracle reads a rifle only"
    victim = state.board[state.ep_victim] if kind == MOVE_EN_PASSANT else state.board[target]
    if not victim:
        return False
    immune = set(strike.get("immune") or [])
    ids = [piece["id"] for piece in spec["pieces"]]
    return ids[engine.spec.type_of(victim)] not in immune


def cost(spec: dict[str, Any], engine: Engine, state: GameState, move: int) -> int:
    """What ``move`` costs the side to move, its price aside (a buy's is `price`)."""
    origin, target, kind, promotion = decode_move(move)
    player = state.side_to_move
    rates = _rates(spec)

    def at(piece_type: int, square: int) -> int:
        return rates[piece_type] * rows(spec, square, player)

    if kind in (MOVE_HIRE, MOVE_KEEP):
        return 0
    if kind in (MOVE_BUY, MOVE_DROP):
        return at(promotion % engine.spec.n_types, target)
    mover = engine.spec.type_of(state.board[origin])
    if kind == MOVE_ROTATE:
        return at(mover, origin)
    if kind == MOVE_CASTLE:
        return at(mover, target)
    square = origin if _stays(spec, engine, state, kind, target) else target
    new = (
        promotion if kind == MOVE_PROMOTION
        else promotion - 1 if kind == MOVE_DISPLACE and promotion else None
    )
    if new is None:
        return at(mover, square)
    if _promotion_reading(spec) == "new":
        return at(new, square)
    return at(mover, square) + at(new, square)


def price(spec: dict[str, Any], engine: Engine, move: int) -> int:
    """A buy's price from the spec (0 for every other move)."""
    _origin, _target, kind, promotion = decode_move(move)
    if kind != MOVE_BUY:
        return 0
    return int(spec["pieces"][promotion % engine.spec.n_types].get("price", 0))


def purse_after(spec: dict[str, Any], engine: Engine, state: GameState, move: int) -> int:
    """The mover's purse after ``move``: less its cost and price, then the bounty of what it
    took (the opponent's piece on the target, or en passant's), capped at 99. The arms take
    nothing but on the target or in passing."""
    _origin, target, kind, _promotion = decode_move(move)
    mover = state.side_to_move
    left = state.purse[mover] - cost(spec, engine, state, move) - price(spec, engine, move)
    victim = (
        state.board[state.ep_victim] if kind == MOVE_EN_PASSANT
        else 0 if kind in (MOVE_BUY, MOVE_DROP, MOVE_HIRE, MOVE_CASTLE, MOVE_ROTATE)
        else state.board[target]
    )
    if victim and engine.spec.owner_of(victim) == 1 - mover:
        left += int(spec["pieces"][engine.spec.type_of(victim)].get("bounty", 0))
    return min(left, 99)


def moves(
    spec: dict[str, Any], engine: Engine, twin_engine: Engine, state: GameState
) -> list[int]:
    """The oracle's legal moves: the twin's, less those the side to move cannot pay for."""
    coins = state.purse[state.side_to_move]
    return [
        move for move in twin_engine.legal_moves(state)
        if cost(spec, engine, state, move) + price(spec, engine, move) <= coins
    ]
