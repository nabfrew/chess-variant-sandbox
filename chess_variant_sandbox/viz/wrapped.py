"""A board whose opposite edges are the same edge, drawn once for both renderers.

The play board in ``web/static/page/board.js`` and the offline replay in ``viz/html.py``
are two programs that draw the same game, so the wrapped board's geometry is one
JavaScript file, ``wrapped.js`` beside this module, that both of them load. What is shared is every answer that has a right answer:
where a square is drawn, which copies of it surround the outline, how pale each one is,
what the edge labels say, the step a crossing move takes, and what a square the rules
single out looks like. What is not shared is the motion, because the two surfaces animate
differently and neither is wrong.

A marker is here rather than in either renderer for the same reason the geometry is. It
is drawn inside the cell, so it lands on a ghost copy at that ring's paleness and follows
the view shift without either surface arranging it.

**There is usually no true boundary.** The stored board runs file ``a`` to file ``h``,
but on a wrapping axis that is normally one cut through a surface with no seam of its
own. So the view carries an offset, the outline follows it, and the labels follow the
outline. Nothing here treats offset zero as the real board -- it is only the frame the
squares happen to be named in, and ``resetView`` is how a reader gets back to it.

**Except where the rules put one there.** A seam every player may cross both ways is
nowhere in particular. A seam somebody may cross one way and not the other is somewhere:
that line is the only place on its axis where a piece meets a rule, so its position is a
fact of the game rather than a choice of frame. ``barrierHtml`` draws it where it really
is, it stays put while the view shifts, and the outline stands down on that axis rather
than drawing a second, movable boundary beside a fixed one.

**The DSL is not read here.** ``rules.wrap`` spells its permissions as edge names, and
``engine/compile.py`` is the one place that means anything by them. ``wrap_payload``
sends the compiled rule -- per player, per axis, may leave by the low edge and by the
high edge -- so the page never parses ``"left"``. ``goal_payload`` and ``zone_payload``
do the same for the squares a game ends on and the squares a piece promotes on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chess_variant_sandbox.engine.compile import (
    WILD,
    CompiledSpec,
    HexCopy,
    PounceRule,
    PromotionRule,
    ReachRule,
    WrapRule,
)
from chess_variant_sandbox.engine.engine import _square_name as square_name
from chess_variant_sandbox.engine.moves import (
    MOVE_PORTAL,
    move_fold,
    move_portal,
    move_to,
    move_type,
)
from chess_variant_sandbox.spec.schema import POUNCE_KINDS, WRAP_PAIRS


def goal_payload(reach: ReachRule | None) -> list[list[int]] | None:
    """The squares that end the game when reached, per player.

    ``None`` for a game whose rules single out no square, which is what both renderers
    test to skip the whole of this. Otherwise ``[player] -> sorted squares``, straight
    off the compiled rule: ``rules.reach`` spells its squares as names and mirrors them
    by a keyword, and ``engine/compile.py`` is the one place that means anything by
    either, so what crosses is the resolved pair of sets and the page never parses
    ``"e4"`` or ``"rotate"``.

    Sorted because the two lists are compared element-wise on the far side, where both
    sides aiming at the same squares is what decides whether a marker names a side.
    """
    if reach is None:
        return None
    return [sorted(reach.squares[0]), sorted(reach.squares[1])]


def zone_payload(promotions: tuple[PromotionRule, ...]) -> list[list[int]] | None:
    """The squares a piece promotes on, per player.

    ``None`` for a game with no promotion at all, which is what both renderers test to
    draw nothing and to offer no toggle. Otherwise ``[player] -> sorted squares``,
    straight off the compiled rules: ``rules.promotion`` states its zone once from the
    first player's side, as ``"last"``, a rank, a list of ranks or a list of squares, and
    ``spec.schema.promotion_squares`` reflects it for the second, so what crosses is the
    resolved pair of sets and the page never parses ``"last"``.

    The union across the rules, because the mark belongs to a square and not to a piece:
    shogi promotes six pieces in one zone, and a band drawn per rule would be the same
    band six times. A player whose targets are empty promotes into nothing and gets no
    band, which is the reading ``promote_on_entry`` already takes.

    Sorted because the two lists are compared element-wise on the far side, where two
    sides sharing a band is what decides whether a mark names a side.
    """
    if not promotions:
        return None
    bands: tuple[set[int], set[int]] = (set(), set())
    for rule in promotions:
        for player in (0, 1):
            if not rule.targets[player]:
                continue
            bands[player].update(rule.squares[player])
    if not bands[0] and not bands[1]:
        return None
    return [sorted(bands[0]), sorted(bands[1])]


def wrap_payload(
    wrap: WrapRule | None, copies: tuple[HexCopy, ...] = ()
) -> dict[str, Any] | None:
    """The compiled wrap rule, as the board needs it.

    ``None`` for a board with edges, which is what both renderers test to skip the whole
    of this. Otherwise ``[player] -> (may leave by the low edge, may leave by the high
    edge)`` per axis, exactly as ``WrapRule`` holds it -- low is file ``a`` and rank 1.

    **A hexagon (schema 16) sends its three pairs of sides instead**, each shaped like an
    axis (low and high are `schema.WRAP_PAIRS`' own), and ``copies``: the six copies of the
    board, each with the side a piece leaves by to reach it, the pair that side belongs
    to, its ``vector`` (a cell's place in the copy minus its place on the board, in files
    and ranks) and which players may cross to it. That is ``CompiledSpec.copies`` as the
    compiler built it, so the page never parses ``"ne"`` and never works a copy out.
    """
    if wrap is None:
        return None
    if copies:
        side_pair = {side: key for key, pair in WRAP_PAIRS.items()
                     for side in (pair.low, pair.high)}
        return {
            **{key: [list(row) for row in getattr(wrap, key)] for key in WRAP_PAIRS},
            "copies": [
                {"side": copy.side, "pair": side_pair[copy.side],
                 "high": copy.side == WRAP_PAIRS[side_pair[copy.side]].high,
                 "vector": list(copy.vector), "permission": list(copy.permission)}
                for copy in copies
            ],
        }
    return {
        "files": [list(pair) for pair in wrap.files],
        "ranks": [list(pair) for pair in wrap.ranks],
    }


def walls_payload(compiled: CompiledSpec) -> dict[str, list[int]]:
    """``obstacles`` and ``absent``, as every board payload sends them.

    ``CompiledSpec.obstacles`` folds a hexagon's absent corners in with its walls, which
    is right for the tables and wrong for a drawing: a wall is a cell nothing may stand
    on, drawn hatched, and an absent corner is not a cell at all, drawn as nothing. So
    ``obstacles`` carries the walls alone and ``absent`` the cells the board does not
    have. Both empty on every board that is not a hexagon, which is every board before
    schema 16.
    """
    return {
        "obstacles": sorted(compiled.obstacles - compiled.absent),
        "absent": sorted(compiled.absent),
    }


def pounce_kinds_payload(pounce: PounceRule | None) -> list[str | None] | None:
    """Each type's pounce kind, by name, as the board payloads send it (generation 40, N3c):
    ``None`` for a type that does not pounce, and ``None`` for the whole where nothing does.
    The compiled ``PounceRule.kind`` read back through ``POUNCE_KINDS``, so the page words a
    pounce by the rule the core played, never by the spec's spelling."""
    if pounce is None:
        return None
    return [POUNCE_KINDS[kind - 1] if kind else None for kind in pounce.kind]


def portal_payload(compiled: CompiledSpec) -> dict[str, Any] | None:
    """Schema 19: what the page draws a portal end with, per internal type, or ``None``.

    The static half of the rule, sent once per ruleset (`docs/archive/plan-portals.md` §4): for
    each end's type, its group's number (``pair``, the colour's and the glyph's number), its
    id (``id``, ``1o``, ``2a``) and its letter in it (``end``: ``orange`` or ``blue`` for a
    pair's, ``a``, ``b``, ... for a map's), the type of the end it leads to (``dest``,
    ``None`` for a way out only), who may move it, whose pieces may go in by it (``open``,
    per player: some piece may) and which pieces (``admits``, per player, as piece ids, and
    ``filtered``, whether the end names them; schema 21, `docs/archive/plan-portals-2.md` §5.2).
    Where the ends stand is on the board, as every piece's square is. Compiled answers
    only, so the page reads no DSL and works out nothing.
    """
    portals = compiled.portals
    if portals is None:
        return None
    ends = []
    for index, piece_type in enumerate(portals.end_types):
        group = portals.group_of[index]
        piece_id = compiled.piece_ids[piece_type]
        letter = piece_id[len(str(group + 1)):]
        to = portals.dest[piece_type]
        # A group of two ends is always a pair (the validator's), lettered o and b.
        paired = portals.group_of.count(group) == 2
        ends.append({
            "type": piece_type,
            "pair": group + 1,
            "id": piece_id,
            "end": {"o": "orange", "b": "blue"}[letter] if paired else letter,
            "dest": to if to >= 0 else None,
            "movedBy": sorted(portals.movers[piece_type]),
            "open": [bool(mask) for mask in portals.admits[piece_type]],
            "filtered": portals.carried[piece_type],
            # Schema 22: the end's rule set, "momentum" or "fold".
            "kind": portals.kinds[index] if portals.kinds else "momentum",
            "admits": [
                [pid for bit, pid in enumerate(compiled.piece_ids[:portals.declared_types])
                 if mask >> bit & 1]
                for mask in portals.admits[piece_type]
            ],
        })
    return {"ends": ends}


def fold_payload(engine: Any, state: Any) -> list[dict[str, Any]]:
    """Schema 22: where each fold stands in ``state``, for the page to draw under the pieces.

    One ``{square, type, pair}`` per fold end, in end order: its square (the state's
    ``folds``, read through `Engine.fold_square`), its end's internal type (the key into
    `portal_payload`'s ends) and its group's number (the colour). Empty for every game
    without folds. A fold stands on no square of the board byte, so this is the only place
    the page learns where one is; the page draws it as a pool under the pieces (Q843d,
    `docs/archive/plan-portals-2.md` §5.3).
    """
    portals = engine.spec.portals
    if portals is None or portals.folds is None:
        return []
    found = []
    for end, piece_type in enumerate(portals.end_types):
        square = engine.fold_square(state, end)
        if square is not None:
            found.append({
                "square": square, "type": piece_type, "pair": portals.group_of[end] + 1,
            })
    return found


def fold_transit(engine: Any, state: Any, move: int) -> dict[str, Any] | None:
    """Schema 22: what the page draws a move through a fold with, read off the position it
    is played in, or ``None`` for a move that went through none.

    The engine's own route (`Engine.fold_route`): the fold it went in by (``foldIn``) and
    the end that one leads to, where it came out and ended (``foldOut``, the move's target
    since schema 24), with their names and the group's number (``foldPair``, the colour).
    The trip (the ripple), the trail at rest and the destination dots read these and work
    out nothing (Q843d, `docs/archive/plan-portals-2.md` §5.3; their revised look is §5.3R's).
    """
    route = engine.fold_route(state, move)
    if route is None:
        return None
    way_in, way_out = route
    # The tag is the fold end gone through, plus one (`MOVE_FOLD`'s word, §3.3).
    group = engine.spec.portals.group_of[move_fold(move) - 1] + 1
    return {
        "foldIn": way_in,
        "foldInName": square_name(way_in, engine.spec.width),
        "foldOut": way_out,
        "foldOutName": square_name(way_out, engine.spec.width),
        "foldPair": group,
    }


def portal_transit(engine: Any, state: Any, move: int) -> dict[str, Any] | None:
    """Schema 19: what the page draws a transit with, read off the position it is played in.

    ``None`` for any move that is not a ``MOVE_PORTAL``. Otherwise the end it goes in by
    (``portal``, the engine's own `portal_entry`), the end that one leads to, beside which
    it comes out (``portalOut``: the entry end's ``dest`` type, found on the board as the
    entry is), and the direction it comes out in (``exitDir``, an index into
    ``PORTAL_DIRECTIONS``), which is the compiled exit table read backwards: the direction
    whose step from ``portalOut`` under the mover's wrap permission is the move's own
    target. Squares come with their names. The two-leg trail, the two-beat animation and
    the chooser's words (`docs/archive/plan-portals.md` §4, D6) read these and work out nothing;
    the move's legality and its exit are the engine's.
    """
    spec = engine.spec
    portals = spec.portals
    if portals is None or move_type(move) != MOVE_PORTAL:
        return None
    entry = engine.portal_entry(state, move)
    if entry is None:
        return None
    to = portals.dest[move_portal(move)]
    if to < 0:
        return None
    out = state.board.find(bytes((spec.encode_piece(WILD, to),)))
    if out < 0:
        return None
    target = move_to(move)
    row = portals.exits[state.side_to_move][out]
    return {
        "portal": entry,
        "portalName": square_name(entry, spec.width),
        "portalOut": out,
        "portalOutName": square_name(out, spec.width),
        "exitDir": row.index(target) if target in row else None,
    }


# The JavaScript half, a plain file beside this one: the application serves it as one of
# the page's scripts (`web/page/__init__.py`) and the offline replay inlines it.
WRAPPED_FILE = Path(__file__).with_name("wrapped.js")
WRAPPED = WRAPPED_FILE.read_text(encoding="utf-8")
