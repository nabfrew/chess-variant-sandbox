"""A member's mark: an invader piece of their own, generated from their account's id.

R2 of `docs/plan-return.md` §8 (§6.3), drawn in the family of the site's own mark, the
invader rook (`shell.ROOK`): a 16 × 16 pixel figure, mirror-symmetric like the rook, made of
four parts stacked as the rook is, each chosen from a short list by a hash of the id:

- the **crest** (rows 0-2): a rook's battlements, three merlons, a crown's points, a pawn's
  ball, a bishop's mitre, a king's cross, or an invader's antennae;
- the **head** (rows 3-7): wide or narrow, a jaw or none, and one of four pairs of eyes;
- the **body** (rows 8-10): three widths, with arms or without;
- the **base** (rows 11-15): the rook's splayed legs, a plinth, tentacles, stilts or feet.

That is 7 × 2 × 2 × 4 × 3 × 2 × 5 = 6,720 figures in `HUES` colours, the colour stepped
through the id (`hue`) so neighbours in joining order never share one. The task's rule
(2026-10-04, Q597): **a function of the member's id only, no stored data, no upload**. So
there is no salt and no *change my mark*: the plan's salt (§6.3) is a stored column, which
the task rules out, and a repeat among the first thirty people is about 2% likely (est.,
`6,720 × 8` marks) and harmless, since the handle is always shown beside it. The handle is
never read, so a renamed handle keeps its mark.

**Nothing is stored or served as an image.** A payload that names a person carries
`mark(user)`, the figure's one SVG path and its hue's number, and the page draws it inline
in the hue's token (`--mk-<n>`), so it follows the theme. `svg(...)` is the same drawing
for a page the server renders. The id itself never leaves the server through the mark: a
SHA-256 of it, cut to a few bytes, chooses the parts.
"""

from __future__ import annotations

import hashlib
from html import escape
from typing import Any

from chess_variant_sandbox.web.page.shell import mark_path

# Each part is its rows' left halves, eight columns; the right half is the mirror. Column
# 7 is the one beside the middle.
CRESTS: tuple[tuple[str, str, str], ...] = (
    ("........", ".##..##.", ".##..##."),  # the rook's four merlons
    ("........", ".##...##", ".##...##"),  # three merlons, one in the middle
    (".#...#..", ".##.###.", ".#######"),  # a crown's points
    ("......##", ".....###", "......##"),  # a pawn's ball
    (".......#", "......##", ".....###"),  # a bishop's mitre
    (".......#", ".....###", ".......#"),  # a king's cross
    ("..#.....", "...#....", "....#..."),  # an invader's antennae
)

# The head: where its rows start (the brow row, then the rest), then the eyes.
HEADS: tuple[tuple[int, int], ...] = ((1, 2), (2, 2))
JAWS: tuple[bool, ...] = (False, True)
# The eyes, as the columns left empty in the head's rows 5 and 6.
EYES: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (
    ((4, 5), (4, 5)),  # the rook's two square eyes
    ((5,), (5,)),      # two slits
    ((3, 4, 5), ()),   # a wide visor
    ((6,), (6,)),      # set close together
)
# The body: where rows 8 to 10 start, and whether arms stand out from it.
BODIES: tuple[int, ...] = (3, 2, 4)
ARMS: tuple[bool, ...] = (False, True)
BASES: tuple[tuple[str, str, str, str, str], ...] = (
    ("..######", ".#######", ".###.##.", ".##.....", "##......"),  # the rook's legs
    ("..######", ".#######", "..######", ".#######", "########"),  # a plinth
    ("..######", ".#######", ".#.#.#.#", "#.#.#.#.", ".#.#.#.#"),  # tentacles
    ("..######", ".#######", "..#..#..", "..#..#..", ".##..##."),  # stilts
    ("...#####", "..######", ".##..##.", ".#....#.", "##....##"),  # feet
)

HUES = 8
"""How many hues a mark is drawn in: the tokens `--mk-1` to `--mk-8` (`tokens.py`)."""


def _row(start: int, holes: tuple[int, ...] = ()) -> str:
    return "".join("." if col < start or col in holes else "#" for col in range(8))


def _choices(user: int) -> list[int]:
    digest = hashlib.sha256(f"member-mark:{int(user)}".encode()).digest()
    return list(digest[:8])


def rows(user: int) -> tuple[str, ...]:
    """The figure for this account, sixteen rows of sixteen, `#` for ink."""
    pick = _choices(user)
    crest = CRESTS[pick[0] % len(CRESTS)]
    brow, rest = HEADS[pick[1] % len(HEADS)]
    jaw = JAWS[pick[2] % len(JAWS)]
    eyes = EYES[pick[3] % len(EYES)]
    body = BODIES[pick[4] % len(BODIES)]
    arms = ARMS[pick[5] % len(ARMS)]
    base = BASES[pick[6] % len(BASES)]
    half = [
        *crest,
        _row(brow),
        _row(rest),
        _row(rest, eyes[0]),
        _row(rest, eyes[1]),
        _row(rest + 1 if jaw else rest),
        _row(body) if not arms else ".#." + _row(body)[3:],
        _row(body) if not arms else "##." + _row(body)[3:],
        _row(body),
        *base,
    ]
    return tuple(left + left[::-1] for left in half)


def hue(user: int) -> int:
    """Which of the `HUES` tokens this account's mark is drawn in, 1 to `HUES`.

    Stepped through the id rather than hashed: accounts are numbered in the order people
    joined, and three is prime to eight, so any eight who joined one after another wear
    eight different colours. A group of ten is told apart by colour first.
    """
    return int(user) * 3 % HUES + 1


def mark(user: int) -> dict[str, Any]:
    """What a payload carries for a person: the figure's path and its hue's number."""
    return {"d": mark_path(rows(user)), "hue": hue(user)}


def svg(user: int, size: int = 20) -> str:
    """The mark inline, for a page the server renders (the header's corner, your page)."""
    drawn = mark(user)
    return (
        f'<svg class="memberMark" viewBox="0 0 16 16" width="{size}" height="{size}" '
        'shape-rendering="crispEdges" aria-hidden="true" focusable="false">'
        f'<path fill="var(--mk-{drawn["hue"]})" d="{escape(drawn["d"])}"/></svg>'
    )
