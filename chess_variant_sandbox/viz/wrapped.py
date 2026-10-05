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
    CompiledSpec,
    HexCopy,
    PromotionRule,
    ReachRule,
    WrapRule,
)
from chess_variant_sandbox.spec.schema import WRAP_PAIRS


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


# The JavaScript half, a plain file beside this one: the application serves it as one of
# the page's scripts (`web/page/__init__.py`) and the offline replay inlines it.
WRAPPED_FILE = Path(__file__).with_name("wrapped.js")
WRAPPED = WRAPPED_FILE.read_text(encoding="utf-8")
