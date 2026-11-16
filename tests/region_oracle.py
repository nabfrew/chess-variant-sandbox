"""The declared oracle for `pieces[].region` (R-L6c, `docs/reference/engine-and-spec-record.md`
*R-L6c*): **a move after which a player's piece stands, newly, on a square outside its region
is no move**, and a move's own landing is judged as the piece that sets out (its tables).

It reads the rule from the spec alone and asks the core nothing about regions: the twin is
the same spec with every `region` taken out, so its core offers each move the region would
bar, and the oracle keeps a twin move only if the position the twin's core plays it into has
no player's piece standing outside its region on a square whose byte the move changed. The
spec's own moves must be exactly those, at every node of a walk (a perft whose every node is a
set comparison). Python plays no move and judges no legality here: the twin's core lists and
plays, the oracle reads a board (CLAUDE.md rule 3: a declared test oracle).

Sound only where the twin and the spec differ by the region alone at every node the walk
reaches, so the specs it is given use `royal_capture` (no check filter, whose attacks a
region changes) and no rule that reads a region by another reading (a rebirth home, a blast's
placement: those remove a piece that may not stand, and have tests of their own).
"""

from __future__ import annotations

import copy
from typing import Any

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import _region_squares
from chess_variant_sandbox.engine.moves import (
    MOVE_DISPLACE,
    MOVE_EN_PASSANT,
    MOVE_NORMAL,
    MOVE_PORTAL,
    MOVE_PROMOTION,
    MOVE_PUSH,
    MOVE_SWEEP,
    move_from,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState


def without_regions(spec: dict[str, Any]) -> dict[str, Any]:
    twin = copy.deepcopy(spec)
    for piece in twin["pieces"]:
        piece.pop("region", None)
    return twin


def regions(engine: Engine, spec: dict[str, Any]) -> dict[tuple[int, int], frozenset[int]]:
    """``(player, type) -> squares`` for every type with a region, read from the spec."""
    width, height = spec["board"]["width"], spec["board"]["height"]
    orientation = str(spec.get("setup", {}).get("orientation", "mirror"))
    found: dict[tuple[int, int], frozenset[int]] = {}
    for piece in spec["pieces"]:
        if piece.get("region"):
            for player in (0, 1):
                squares = _region_squares(piece["region"], player, width, height, orientation)
                assert squares is not None
                found[(player, engine.spec.piece_ids.index(piece["id"]))] = squares
    return found


def leaves_one_outside(
    engine: Engine, bound: dict[tuple[int, int], frozenset[int]], before: GameState, after: GameState,
) -> bool:
    for square, value in enumerate(after.board):
        if not value or value == before.board[square]:
            continue
        owner = engine.spec.owner_of(value)
        region = bound.get((owner, engine.spec.type_of(value)))
        if region is not None and square not in region:
            return True
    return False


# The kinds whose target is a landing of the mover's own tables (its move, a capture, a
# promotion's arrival, a sweep, a push, a displacing hop, a portal's exit).
_TABLE_KINDS = (MOVE_NORMAL, MOVE_EN_PASSANT, MOVE_PROMOTION, MOVE_SWEEP, MOVE_PORTAL, MOVE_PUSH,
                MOVE_DISPLACE)


def lands_outside(
    engine: Engine, bound: dict[tuple[int, int], frozenset[int]], state: GameState, move: int,
) -> bool:
    """The mover's own landing, judged as the piece that sets out: a move whose target is
    outside the mover's region is no move whatever the move makes of it (a side switch
    landing in the new owner's region included)."""
    mover = state.board[move_from(move)]
    if not mover or move_type(move) not in _TABLE_KINDS:
        return False
    region = bound.get((engine.spec.owner_of(mover), engine.spec.type_of(mover)))
    return region is not None and move_to(move) not in region


def oracle_moves(twin: Engine, bound: dict[tuple[int, int], frozenset[int]], state: GameState) -> list[int]:
    return [
        m for m in twin.legal_moves(state)
        if not lands_outside(twin, bound, state, m)
        and not leaves_one_outside(twin, bound, state, twin.apply(state, m))
    ]


def walk(spec: dict[str, Any], state: GameState | None, depth: int) -> tuple[int, int]:
    """Compare the spec's moves with the oracle's at every node to ``depth``; returns the
    leaf count and how many twin moves the region barred along the way (so a test can say
    the walk met the rule)."""
    engine, twin = Engine(spec), Engine(without_regions(spec))
    bound = regions(engine, spec)
    start = engine.initial_state() if state is None else state
    barred = 0

    def go(node: GameState, left: int) -> int:
        nonlocal barred
        if left == 0 or engine.terminal(node) is not None:
            return 1
        mine = engine.legal_moves(node)
        theirs = oracle_moves(twin, bound, node)
        assert sorted(mine) == sorted(theirs), (left, sorted(set(mine) ^ set(theirs)))
        barred += len(twin.legal_moves(node)) - len(theirs)
        return sum(go(engine.apply(node, m), left - 1) for m in mine)

    return go(start, depth), barred
