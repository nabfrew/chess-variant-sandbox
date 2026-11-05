"""The declared Python oracle for the rest (schema 29, Q983a, `docs/plan-rest.md` §3, §5.3).

`pieces[].rest`: a piece that makes its player's move rests for that player's next *n* turns
(*n* its type's), where a turn is an alternating turn of that player or a round. This module
works the rule out from the compiled tables, a move list with nothing resting and the board
the move left, and the tests hold the core to it move list by move list and rest list by rest
list. CLAUDE.md rule 3 allows it: it is a declared test oracle.

It asks the engine only questions answered elsewhere: the moves a player has on a board with
**nothing resting** (the core's generator with an empty list, which every game without a rest
plays), and, for a game with `must_capture`, the same from its twin without the rule, so that
`must_capture` can be applied here after the rest, as §3.7 orders it. The other list filters
(`forbid_check`, `forbid_immobility`, the drop-mate rule) judge each move alone, so they commute
with the rest's filter and stay in the engine's list; only the lift's two tests, which this
module makes itself, see the difference, and they come out the same either way. It never asks the core what
the rest does: the filter, the lift, the count, the end of a rest and which piece enters are
all worked out below.

A position's rest is ``((square, turns left), ...)`` per player, by square
(`GameState.rest`). The oracle's own lists carry each entry's code too, ``(square, left,
code)``, read off the board the entry began on; every update drops an entry whose square no
longer holds its code, so between moves the code is always the board's.

Beside a command (SR5e, `docs/plan-schema-next.md` §2.7 row V7, as ruled 2026-10-27): a
commanded move enters its piece in its **owner's** list, counted on the owner's turns, and a
resting piece of either list is offered to neither player, so the lift reads the player's every
offer, commands included.
"""

from __future__ import annotations

import copy
import json
from functools import cache
from typing import Any

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import (
    MOVE_CASTLE,
    MOVE_DROP,
    MOVE_EN_PASSANT,
    MOVE_FOLD,
    MOVE_NORMAL,
    MOVE_PORTAL,
    MOVE_PROMOTION,
    MOVE_ROTATE,
    MOVE_SWEEP,
    decode_move,
    move_foreign,
)
from chess_variant_sandbox.engine.state import GameState

NO_REST: tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] = ((), ())
Entry = tuple[int, int, int]
Lists = tuple[list[Entry], list[Entry]]


def lists_of(state: GameState) -> Lists:
    """The position's resting pieces with their codes, read off its board."""
    return (
        [(square, left, state.board[square]) for square, left in state.rest[0]],
        [(square, left, state.board[square]) for square, left in state.rest[1]],
    )


def as_rest(lists: Lists) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """The boundary's shape: ``(square, left)`` by square, per player."""
    return (
        tuple(sorted((square, left) for square, left, _ in lists[0])),
        tuple(sorted((square, left) for square, left, _ in lists[1])),
    )


def after_turn(
    engine: Engine, lists: Lists, board: bytes, player: int, end: int | None,
    owner: int | None = None,
) -> Lists:
    """§3.2-§3.4: the end of ``player``'s turn on ``board`` (the board the turn left). Count
    ``player``'s entries down (an entry at 0 leaves), drop every entry of either player whose
    square no longer holds its code, then enter the piece standing on ``end`` in ``owner``'s
    list (``player``'s unless a command moved the piece: SR5e, its owner's) if it is
    ``owner``'s and its type (the type it is now) rests, replacing any entry on that square."""
    spec = engine.spec
    owner = player if owner is None else owner
    counted = [list(lists[0]), list(lists[1])]
    counted[player] = [(square, left - 1, code) for square, left, code in counted[player] if left > 1]
    kept = [[entry for entry in counted[owner] if board[entry[0]] == entry[2]] for owner in (0, 1)]
    if end is not None:
        value = board[end]
        if value and spec.owner_of(value) == owner:
            turns = spec.rest[spec.type_of(value)] if spec.rest else 0
            if turns:
                kept[owner] = [entry for entry in kept[owner] if entry[0] != end]
                kept[owner].append((end, turns, value))
    return kept[0], kept[1]


def end_of_move(engine: Engine, before: GameState, after: GameState, move: int) -> int | None:
    """Where the piece that made an alternating move stands afterwards, by the move and the
    board alone: a drop's and a castling's target, a turn in place's origin, and otherwise the
    origin if a piece of the mover's still stands there (a strike or contact that stays) or
    else the target. A fold's own move is made by no piece. Whatever the board then holds
    there is what `after_turn` reads, so a mover blown up or eaten stands nowhere."""
    origin, target, kind, _ = decode_move(move)
    if kind == MOVE_FOLD:
        return None
    if kind in (MOVE_DROP, MOVE_CASTLE):
        return target
    if kind == MOVE_ROTATE:
        return origin
    player = before.side_to_move
    held = after.board[origin]
    if held and engine.spec.owner_of(held) == player:
        return origin
    return target


def _takes(move: int, board: bytes) -> bool:
    """What `must_capture` counts as a capture (the core's `restrict`, `takes`)."""
    _, target, kind, _ = decode_move(move)
    return kind in (MOVE_EN_PASSANT, MOVE_SWEEP) or (
        kind in (MOVE_NORMAL, MOVE_PROMOTION, MOVE_PORTAL) and board[target] != 0
    )


def commands(engine: Engine) -> bool:
    """Whether a sight commands (SR5e): then a resting piece is moved by neither player."""
    sight = engine.spec.sight
    return sight is not None and sight.has_command


def resting_squares(engine: Engine, state: GameState, mover: int) -> set[int]:
    """The squares whose pieces ``mover`` may not move: its own resting pieces, and beside a
    command the opponent's too (SR5e)."""
    squares = {square for square, _ in state.rest[mover]}
    if commands(engine):
        squares |= {square for square, _ in state.rest[1 - mover]}
    return squares


def _source(engine: Engine) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(engine.spec.source)
    return loaded


@cache
def unrestricted(engine: Engine) -> Engine:
    """``engine`` itself, or for a game with `must_capture` its twin without the rule (and
    without the rest, which an empty list leaves out anyway), whose list is the one the rest
    filters before `must_capture` (see the module's note on the other filters)."""
    spec = engine.spec
    if not spec.must_capture:
        return engine
    twin = copy.deepcopy(_source(engine))
    twin["rules"].pop("must_capture")
    for piece in twin["pieces"]:
        piece.pop("rest", None)
    twin["name"] = f"{twin['name']} (without must_capture)"
    return Engine(twin)


def _restrict(engine: Engine, state: GameState, moves: list[int]) -> list[int]:
    if engine.spec.must_capture and any(_takes(move, state.board) for move in moves):
        return [move for move in moves if _takes(move, state.board)]
    return moves


def legal(engine: Engine, state: GameState, player: int | None = None) -> list[int]:
    """``player``'s moves under the rest (§3.7): the list with nothing resting, less every
    resting piece's moves (never a drop, never a fold's own move), unless that leaves none
    (the lift); then `restrict`; and if that empties a list the rest shortened, the list
    again without the rest. ``player`` given is a round's seat (`legal_moves_for`); absent,
    the side to move's turn. In the core's order."""
    base_engine = unrestricted(engine)
    stripped = state.with_(rest=NO_REST)
    if player is None:
        mover = state.side_to_move
        base = base_engine.legal_moves(stripped)
    else:
        mover = player
        base = base_engine.legal_moves_for(stripped, player)
    resting = resting_squares(engine, state, mover)

    def rests(move: int) -> bool:
        origin, _, kind, _ = decode_move(move)
        return kind not in (MOVE_DROP, MOVE_FOLD) and origin in resting

    kept = [move for move in base if not rests(move)]
    shortened = bool(kept) and len(kept) < len(base)
    if not kept:
        kept = base
    final = _restrict(engine, state, kept)
    if shortened and not final:
        final = _restrict(engine, state, base)
    return final


def lifted(engine: Engine, state: GameState, player: int | None = None) -> bool:
    """Whether the rest was lifted here: a piece rests and the list offers one of its moves."""
    mover = state.side_to_move if player is None else player
    resting = resting_squares(engine, state, mover)
    return any(
        decode_move(move)[0] in resting and decode_move(move)[2] not in (MOVE_DROP, MOVE_FOLD)
        for move in legal(engine, state, player)
    )


def apply(engine: Engine, state: GameState, move: int) -> tuple[GameState, Lists]:
    """An alternating move: the core's position, and the rest lists the oracle says it must
    carry (the caller compares ``as_rest`` of them with the position's)."""
    after = engine.apply(state, move)
    mover = state.side_to_move
    owner = mover
    if move_foreign(move):
        # SR5e: a commanded piece rests in its owner's list; a held piece, nobody's, reads as
        # the mover's (and so enters no list, being the mover's in none).
        found = engine.spec.owner_of(state.board[decode_move(move)[0]])
        owner = found if found in (0, 1) else mover
    lists = after_turn(
        engine, lists_of(state), after.board, mover,
        end_of_move(engine, state, after, move), owner,
    )
    return after, lists
