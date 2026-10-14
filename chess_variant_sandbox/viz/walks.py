"""Every wild piece's walk, in order, as a board draws it (Q852).

The wolves walk (`docs/reference/designer-case-study-wolves.md` §1 steps 1, 6 and 9, §3
fix 3) found that nothing on the site drew where a wild piece goes: the game page's still
board showed four wolves and no patrol, and the designer tinted a walk's squares with no
order, so ``a4 b5 c4 b3 a3`` and any shuffle of it looked the same, and a walk that jumps
from h4 to a5 looked exactly like the circuit it was meant to be. This module is what the
still boards draw from: one entry per walk, its squares in the order the pieces on it step,
the squares a piece starts on, and the steps between two squares that do not touch.

**The DSL is not read here.** A walk's squares are ``schema.npc_routes``' and its starts
``schema.npc_starts``': the one resolver the validator and ``engine/compile.py`` share, so
a walk means here what the compiled ``next`` table means (`tests/test_walks.py` holds the
two together). Both skip what is malformed rather than raising, which is what lets a draft
that is *not a game yet* draw its walks too, so a refusal about them can be seen on the
board. Whether two squares touch is the compiled board's: its lattice, and the seams it
joins (``WrapRule.files_join``/``ranks_join``, a hexagon's ``copies``), which a wild piece
crosses whoever may (it owns no seat, as the play board's coming arrows say).

Nothing here works out a square a piece could reach: it lays out squares the rules name.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from chess_variant_sandbox.spec.describe import RULE_WORDS
from chess_variant_sandbox.spec.schema import (
    format_square,
    hex_distance,
    npc_routes,
    npc_starts,
)

_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
          "ten", "eleven", "twelve")

# `npc_routes`' label for a walk: the entry, and whether it is that entry's mirror.
_LABEL = re.compile(r"^rules\.npcs\[(?P<index>\d+)\](?P<mirror> \(mirrored\))?$")

# How many of a walk's jumps its sentence names before it counts the rest.
_SAID = 3


def _count(n: int, unit: str) -> str:
    word = _WORDS[n] if 0 <= n < len(_WORDS) else str(n)
    return f"{word} {unit}{'' if n == 1 else 's'}"


def step_apart(
    a: int,
    b: int,
    width: int,
    height: int,
    *,
    lattice: str = "square",
    files_join: bool = False,
    ranks_join: bool = False,
    copies: Iterable[Sequence[int]] = (),
) -> tuple[int, int, int]:
    """How far apart squares ``a`` and ``b`` are, the nearest way round the board's seams:
    ``(files, ranks, steps)``, the first two unsigned and ``steps`` the moves of one square
    it takes (the larger of the two on a square lattice, the hex distance on a hex one).

    A seam counts whoever may cross it: ``files_join``/``ranks_join`` on a rectangle, and
    on a hexagon each copy's ``vector`` (a cell of the copy plus the vector is the same
    cell of the board), both read off the compiled board, never off ``rules.wrap``.
    """
    fa, ra = a % width, a // width
    fb, rb = b % width, b // width
    images: list[tuple[int, int]] = [(fb - fa, rb - ra)]
    if lattice == "hex":
        for vector in copies:
            dx, dy = int(vector[0]), int(vector[1])
            images += [(fb - dx - fa, rb - dy - ra), (fb + dx - fa, rb + dy - ra)]
    else:
        file_shifts = (0, width, -width) if files_join else (0,)
        rank_shifts = (0, height, -height) if ranks_join else (0,)
        images = [(fb + sf - fa, rb + sr - ra) for sf in file_shifts for sr in rank_shifts]

    def steps(image: tuple[int, int]) -> int:
        df, dr = image
        return hex_distance(df, dr) if lattice == "hex" else max(abs(df), abs(dr))

    nearest = min(images, key=lambda image: (steps(image), abs(image[0]) + abs(image[1])))
    return abs(nearest[0]), abs(nearest[1]), steps(nearest)


def leap_words(a: int, b: int, width: int, apart: tuple[int, int, int], lattice: str) -> str:
    """One jump in words: *steps from h4 to a5, seven files apart*. On a rectangle the
    axes named are the ones too far to touch (h4 to a5 is seven files apart, and one rank,
    which would touch, so it is not named); on a hexagon, how many cells apart."""
    files, ranks, steps = apart
    names = format_square(a % width, a // width), format_square(b % width, b // width)
    if lattice == "hex":
        far = _count(steps, "cell")
    else:
        parts = [_count(n, unit) for n, unit in ((files, "file"), (ranks, "rank")) if n > 1]
        far = " and ".join(parts)
    return f"from {names[0]} to {names[1]}, {far} apart"


def walks(
    rules: Any,
    width: int,
    height: int,
    *,
    orientation: str = "mirror",
    piece_ids: Sequence[str] = (),
    lattice: str = "square",
    files_join: bool = False,
    ranks_join: bool = False,
    copies: Iterable[Sequence[int]] = (),
) -> list[dict[str, Any]]:
    """Every wild piece's walk, one entry per walk the board stands a piece on.

    Each entry: ``name``, the designer card's (*wild pieces 1*, *wild pieces 1's
    mirror*); ``entry`` and ``mirrored``, which ``rules.npcs`` entry it is; ``piece`` (the
    letter) and ``type`` (its index in ``piece_ids``, or None while the draft names no
    such piece); ``squares``, in the order a piece on it steps, the last stepping back to
    the first; ``starts``, where a piece stands on it at ply zero (``also_at``'s included,
    several pieces on one walk being one walk); ``leaps``, each step between two squares
    that do not touch, ``{from, to, files, ranks, says}``; and ``says``, the sentence
    naming them, or None where it has none. Empty without ``rules.npcs``.
    """
    copies = [tuple(vector) for vector in copies]
    npcs = rules.get("npcs") if isinstance(rules, dict) else None
    title = RULE_WORDS["npcs"][0]
    starts = npc_starts(rules, width, height, orientation)
    out: list[dict[str, Any]] = []
    for label, squares in npc_routes(rules, width, height, orientation):
        found = _LABEL.match(label)
        if found is None:
            continue
        index, mirrored = int(found["index"]), bool(found["mirror"])
        entry = npcs[index] if isinstance(npcs, list) else {}
        piece = entry.get("piece") if isinstance(entry, dict) else None
        own = [square for at, where, square in starts
               if at == index and where.endswith("(mirrored)") == mirrored]
        leaps: list[dict[str, Any]] = []
        if len(squares) > 1:
            for position, a in enumerate(squares):
                b = squares[(position + 1) % len(squares)]
                apart = step_apart(a, b, width, height, lattice=lattice,
                                   files_join=files_join, ranks_join=ranks_join, copies=copies)
                if apart[2] > 1:
                    leaps.append({"from": a, "to": b, "files": apart[0], "ranks": apart[1],
                                  "says": leap_words(a, b, width, apart, lattice)})
        name = f"{title} {index + 1}'s mirror" if mirrored else f"{title} {index + 1}"
        out.append({
            "name": name,
            "entry": index,
            "mirrored": mirrored,
            "piece": piece if isinstance(piece, str) else None,
            "type": list(piece_ids).index(piece) if isinstance(piece, str) and piece in piece_ids
            else None,
            "squares": list(squares),
            "starts": own,
            "leaps": leaps,
            "says": _walk_says(name, leaps),
        })
    return out


def _walk_says(name: str, leaps: list[dict[str, Any]]) -> str | None:
    """The sentence under the board for a walk that jumps, or None for one that doesn't."""
    if not leaps:
        return None
    said = [leap["says"] for leap in leaps[:_SAID]]
    if len(leaps) > _SAID:
        listed = ", ".join(said) + f", and {_count(len(leaps) - _SAID, 'more jump')}"
    elif len(said) == 1:
        listed = said[0]
    else:
        listed = ", ".join(said[:-1]) + f", and {said[-1]}"
    return f"{name[0].upper()}{name[1:]}'s walk steps {listed}."


def walk_payload(spec: Any, compiled: Any) -> list[dict[str, Any]]:
    """`walks` for a draft or a game, on the board ``compiled`` (a ``CompiledSpec``) has:
    its size, lattice, seams and pieces. ``spec`` is the rules to read the walks from,
    which for a draft that is not a game yet is the draft itself, drawn on its stand-in's
    board (`designing.board_preview`)."""
    if not isinstance(spec, dict):
        return []
    setup = spec.get("setup") if isinstance(spec.get("setup"), dict) else {}
    wrap = getattr(compiled, "wrap", None)
    return walks(
        spec.get("rules"),
        compiled.width,
        compiled.height,
        orientation=(setup or {}).get("orientation", "mirror"),
        piece_ids=list(compiled.piece_ids),
        lattice=getattr(compiled, "lattice", "square") or "square",
        files_join=bool(wrap is not None and wrap.files_join),
        ranks_join=bool(wrap is not None and wrap.ranks_join),
        copies=[copy.vector for copy in getattr(compiled, "copies", ()) or ()],
    )
