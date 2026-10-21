"""Each game's own picture for a link preview (G6 of `docs/plan-growth.md`, §2.2).

G5 gave every page the site's one picture (`picture.py`). A game's own address now names a
picture of that game: 1200 × 630, the size Open Graph and Twitter's large card ask for, in
the site picture's look (paper, ink, the invader rook), with the game's name set in Fraunces
on the left and its starting position on a walnut-framed board on the right.

**What it is drawn from.** The position is the card's still board, `library.opening(game)`:
the core's own initial state through `Engine` (`card_rules.still`), with each piece's
drawing from `assets/pieces/` or the game's uploaded one. Nothing here works out a position,
a move or which squares exist; it reads the payload the card and the game page draw, and
draws what a thumbnail draws (`stillboard.js` with `plain`): the cells, walls, absent cells,
portal ends and pieces, with no marks, seams, hands or walks.

**Where a square is drawn.** The page has one `squareXY` (`viz/wrapped.js`); this is a
picture drawn on the server, which cannot run it, so `cell_place` restates its arithmetic
for the one view a picture uses: flat, unflipped, the first player at the bottom. On a hex
lattice a rank is half a cell further right per rank up and closer by `HEX_ROW`, which is
read from `wrapped.js` itself rather than written twice. `tests/test_game_picture.py` holds
the two to the same answer for both lattices.

**How it is made.** The drawing is an SVG (`svg`), rasterised by resvg (`png`), a Rust SVG
renderer installed with the site (`resvg-py`, the `site` extra): no browser, no system
library, no font of the machine's. The two faces are the site's own (`fonts/`, made by
`fonts/make.py`), passed in by file with the system's skipped, so a picture is the same
bytes on every machine with the same resvg. The name of a picture's file is the digest of
what it draws, the faces and the renderer (`key`): a picture is made once per drawing and
served for a year, and anything that would change the drawing (the game's name, its rules,
the core's still board under a new card key, an icon, a token, a face) is a new name.
`site/game_pictures.py` keeps them on disk and serves them.

Measured with `python -m chess_variant_sandbox.web.page.game_picture --measure`.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import struct
import sys
import time
from collections.abc import Mapping, Sequence
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

from chess_variant_sandbox.viz.wrapped import WRAPPED
from chess_variant_sandbox.web.page import picture, shell
from chess_variant_sandbox.web.page.tokens import LIGHT

PREFIX = "/picture/game/"
"""Where a game's picture is served: `PREFIX + file_name(digest, key)`."""

ROUTE = re.compile(r"^/picture/game/([0-9a-f]{8,64})-([0-9a-f]{16})\.png$")
"""A picture's address, read back into the game's hash and the picture's key."""

PROVEN_PREFIX = "/picture/proven/"
"""Where a proven position's picture is served (G10): `position_path(digest, id, key)`."""

PROVEN_ROUTE = re.compile(
    r"^/picture/proven/([0-9a-f]{8,64})/([0-9a-f]{8,64})-([0-9a-f]{16})\.png$")
"""A position's picture's address, read back into the game's hash, the position's name
(`puzzles.name`, a hex digest) and the picture's key."""

POSITION_NAME = re.compile(r"^[0-9a-f]{8,64}$")
"""A position name a picture can be made for: the `puzzles.name` form `PROVEN_ROUTE` reads."""

WIDTH, HEIGHT, TYPE = picture.WIDTH, picture.HEIGHT, picture.TYPE

FONT_DIR = Path(__file__).with_name("fonts")
HEADING_FACE = ("Fraunces", FONT_DIR / "fraunces-v38-latin-600-opsz72.ttf")
TEXT_FACE = ("Source Sans 3", FONT_DIR / "source-sans-3-v20-latin-600.ttf")
# As an attribute says them: quoted, since CSS reads an unquoted `3` as a number and resvg
# then draws the text in no face at all.
HEADING_FAMILY = f"'{HEADING_FACE[0]}'"
TEXT_FAMILY = f"'{TEXT_FACE[0]}'"

# The page's margins, and the room the board may take on the right.
MARGIN = 72
GAP = 56
BOARD_MOST = (620, 534)
FRAME = 16
CELL_MOST = 128

# The name's sizes, largest first: the largest that sets it in `NAME_LINES` lines.
NAME_SIZES = (80, 72, 64, 56, 48, 42)
NAME_LINES = 3
NAME_LEADING = 1.06
LINE_SIZE = 28
SITE_SIZE = 26
MARK = 48


def _hex_row() -> float:
    """`wrapped.js`'s `HEX_ROW`, read from it: the one number a hex row's spacing is."""
    found = re.search(r"const HEX_ROW = ([^;]+);", WRAPPED)
    if found is None or found.group(1).strip() != "Math.sqrt(3) / 2":
        raise RuntimeError("wrapped.js's HEX_ROW moved; restate it in game_picture.py")
    return math.sqrt(3) / 2


HEX_ROW = _hex_row()


def available() -> bool:
    """Whether this machine can make a picture: resvg is the `site` extra's, and a checkout
    without it names the site's own picture for every game."""
    try:
        import resvg_py  # noqa: F401
    except ImportError:
        return False
    return True


def renderer() -> str:
    """The renderer's name and version, which is part of every picture's key."""
    try:
        from importlib.metadata import version

        return "resvg-py " + version("resvg-py")
    except Exception:  # noqa: BLE001 - absent is a name too
        return "resvg-py absent"


def alt(name: str) -> str:
    """What a game's picture shows, for a reader who cannot see it (`og:image:alt`)."""
    return f"{name or 'An unnamed game'}: its starting position, on a wooden board"


def position_alt(name: str, side: str) -> str:
    """What a proven position's picture shows (G10), never its answer: the game, whose move
    it is, and the board."""
    return (f"A proven position from {name or 'an unnamed game'}: {side} to move and win, "
            "on a wooden board")


# ----------------------------------------------------------------------------- the words


@lru_cache(maxsize=4)
def _advances(path: Path) -> tuple[int, dict[int, int], int]:
    """A TrueType face's units per em, each character's advance, and the advance of a
    character it has no glyph for: read from `head`, `hhea`, `hmtx` and `cmap` (format 4)."""
    data = path.read_bytes()
    count = struct.unpack(">H", data[4:6])[0]
    tables: dict[str, int] = {}
    for index in range(count):
        tag, _, offset, _ = struct.unpack(">4sIII", data[12 + 16 * index: 28 + 16 * index])
        tables[tag.decode("latin-1")] = offset
    em = struct.unpack(">H", data[tables["head"] + 18: tables["head"] + 20])[0]
    metrics = struct.unpack(">H", data[tables["hhea"] + 34: tables["hhea"] + 36])[0]
    hmtx = tables["hmtx"]
    widths = [struct.unpack(">H", data[hmtx + 4 * i: hmtx + 4 * i + 2])[0]
              for i in range(metrics)]

    def advance(glyph: int) -> int:
        return widths[min(glyph, metrics - 1)]

    cmap = tables["cmap"]
    subtables = struct.unpack(">H", data[cmap + 2: cmap + 4])[0]
    chars: dict[int, int] = {}
    for index in range(subtables):
        platform, encoding, offset = struct.unpack(
            ">HHI", data[cmap + 4 + 8 * index: cmap + 12 + 8 * index])
        at = cmap + offset
        if (platform, encoding) != (3, 1) or struct.unpack(">H", data[at: at + 2])[0] != 4:
            continue
        segments = struct.unpack(">H", data[at + 6: at + 8])[0] // 2

        def column(n: int, base: int = at, segs: int = segments) -> int:
            return base + 14 + 2 * segs * n + (2 if n else 0)

        for seg in range(segments):
            end = struct.unpack(">H", data[column(0) + 2 * seg: column(0) + 2 * seg + 2])[0]
            start = struct.unpack(">H", data[column(1) + 2 * seg: column(1) + 2 * seg + 2])[0]
            delta = struct.unpack(">h", data[column(2) + 2 * seg: column(2) + 2 * seg + 2])[0]
            where = column(3) + 2 * seg
            ranged = struct.unpack(">H", data[where: where + 2])[0]
            for code in range(start, min(end, 0xFFFE) + 1):
                if ranged == 0:
                    glyph = (code + delta) & 0xFFFF
                else:
                    spot = where + ranged + 2 * (code - start)
                    glyph = struct.unpack(">H", data[spot: spot + 2])[0]
                    glyph = (glyph + delta) & 0xFFFF if glyph else 0
                if glyph:
                    chars[code] = advance(glyph)
        break
    return em, chars, advance(0)


def text_width(text: str, face: tuple[str, Path], size: float) -> float:
    """How wide `text` is set in `face` at `size` px, from the face's own advances (no
    kerning, which only ever closes a line up)."""
    em, chars, missing = _advances(face[1])
    return sum(chars.get(ord(char), missing) for char in text) * size / em


def _fit(words: Sequence[str], width: float, size: float) -> list[str] | None:
    """`words` set in lines no wider than `width`, or None where one word alone is wider."""
    lines: list[str] = []
    for word in words:
        if text_width(word, HEADING_FACE, size) > width:
            return None
        if lines and text_width(lines[-1] + " " + word, HEADING_FACE, size) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines


def name_lines(name: str, width: float) -> tuple[int, list[str]]:
    """The size a game's name is set at and its lines: the largest size that sets it in
    `NAME_LINES` lines across `width`; at the smallest, the last line cut with an ellipsis."""
    words = (name or "An unnamed game").split()
    for size in NAME_SIZES:
        lines = _fit(words, width, size)
        if lines is not None and len(lines) <= NAME_LINES:
            return size, lines
    size = NAME_SIZES[-1]
    lines = []
    for word in words:
        while text_width(word, HEADING_FACE, size) > width and len(word) > 1:
            word = word[:-2] + "…"
        if lines and text_width(lines[-1] + " " + word, HEADING_FACE, size) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    if len(lines) > NAME_LINES:
        last = lines[NAME_LINES - 1]
        while text_width(last + "…", HEADING_FACE, size) > width and " " in last:
            last = last.rsplit(" ", 1)[0]
        lines = [*lines[: NAME_LINES - 1], last + "…"]
    return size, lines


# ----------------------------------------------------------------------------- the board


def cell_place(square: int, width: int, height: int, hexed: bool) -> tuple[float, float]:
    """The corner of the box a square is drawn in, in cells, on the flat unflipped view:
    `wrapped.js`'s `cellXY(squareCell(square))` with nothing wrapping and nobody flipped."""
    file, rank = square % width, square // width
    if not hexed:
        return float(file), float(height - 1 - rank)
    return file + rank / 2, (height - 1 - rank) * HEX_ROW


def hex_corners(x: float, y: float, cell: float) -> list[tuple[float, float]]:
    """The six corners of the pointy-top hexagon in a cell's box (`wrapped.js` `hexCorners`)."""
    cx, cy = x + cell / 2, y + cell / 2
    r, h = cell / math.sqrt(3), cell / 2
    return [(cx, cy - r), (cx + h, cy - r / 2), (cx + h, cy + r / 2),
            (cx, cy + r), (cx - h, cy + r / 2), (cx - h, cy - r / 2)]


def shade(file: int, rank: int, hexed: bool) -> str:
    """A cell's wood (`wrapped.js` `cellShade`): two on squares, three on hexagons."""
    if not hexed:
        return LIGHT["sq-dk"] if (file + rank) % 2 == 0 else LIGHT["sq-lt"]
    return (LIGHT["sq-lt"], LIGHT["sq-dk"], _mix(LIGHT["sq-lt"], LIGHT["sq-dk"]))[
        (file - rank) % 3]


def _mix(one: str, two: str) -> str:
    """Two `#rrggbb` colours half and half, as `color-mix(in srgb, ... 50%, ...)` mixes."""
    a = [int(one[i: i + 2], 16) for i in (1, 3, 5)]
    b = [int(two[i: i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round((x + y) / 2):02x}" for x, y in zip(a, b, strict=True))


def _num(value: float) -> str:
    """A coordinate as short as it can be said exactly enough: the drawing's text is the
    picture's key, so it must not wander in the last digit between machines."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _box(shape: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """The board's drawn extent in cells: (left, top, right, bottom) over every cell that is
    there, a hexagon's corners included."""
    width, height = int(shape["width"]), int(shape["height"])
    hexed = shape.get("lattice") == "hex"
    absent = set(shape.get("absent") or [])
    xs: list[float] = []
    ys: list[float] = []
    for square in range(width * height):
        if square in absent:
            continue
        x, y = cell_place(square, width, height, hexed)
        if hexed:
            for cx, cy in hex_corners(x, y, 1.0):
                xs.append(cx)
                ys.append(cy)
        else:
            xs += [x, x + 1]
            ys += [y, y + 1]
    return min(xs), min(ys), max(xs), max(ys)


def _cell(x: float, y: float, cell: float, hexed: bool, attrs: str) -> str:
    if hexed:
        points = " ".join(f"{_num(px)},{_num(py)}" for px, py in hex_corners(x, y, cell))
        return f'<polygon points="{points}" {attrs}/>'
    return (f'<rect x="{_num(x)}" y="{_num(y)}" width="{_num(cell)}" height="{_num(cell)}" '
            f"{attrs}/>")


def _piece(shape: Mapping[str, Any], value: int, x: float, y: float, cell: float,
           drawings: dict[str, str]) -> str:
    """One piece from its board byte, as `stillboard.js`'s `stillPieceHtml` draws it: its
    drawing, turned only where the piece has a facing (a single-colour set's half turn is
    whose it is), else a disc with its letter; a portal end as a stone ringed in its pair's
    colour."""
    types = int(shape["nTypes"])
    stride = int(shape.get("sideStride") or types)
    kind, player = (value - 1) % types, (value - 1) // stride
    facing = ((value - 1) % stride) // types
    cx, cy = x + cell / 2, y + cell / 2
    ends = {int(end["type"]): end for end in (shape.get("portals") or {}).get("ends") or []}
    if kind in ends:
        colour = LIGHT.get(f"pair-{ends[kind].get('pair', 1)}", LIGHT["pair-1"])
        return (f'<circle cx="{_num(cx)}" cy="{_num(cy)}" r="{_num(cell * 0.36)}" '
                f'fill="{LIGHT["portal-stone"]}" stroke="{LIGHT["portal-edge"]}" '
                f'stroke-width="{_num(cell * 0.03)}"/>'
                f'<circle cx="{_num(cx)}" cy="{_num(cy)}" r="{_num(cell * 0.24)}" fill="none" '
                f'stroke="{colour}" stroke-width="{_num(cell * 0.1)}"/>')
    icons = shape.get("icons") or []
    icon = icons[kind][player] if player <= 1 and kind < len(icons) else None
    if icon:
        headings = shape.get("facingAngles") or []
        directional = (shape.get("directional") or [])
        heading = (headings[player] if player < len(headings) else None) \
            if kind < len(directional) and directional[kind] else None
        angle = (heading[facing] if facing < len(heading) else 0) if heading \
            else (180 if icon.get("rotate") else 0)
        stem = str(icon.get("stem") or icon["href"])
        ref = drawings.setdefault(stem, f"d{len(drawings)}")
        side = _icon_side(cell)
        at_x, at_y = round(x + (cell - side) / 2), round(y + (cell - side) / 2)
        turn = f" rotate({_num(angle)} {_num(side / 2)} {_num(side / 2)})" if angle else ""
        return f'<use href="#{ref}" transform="translate({at_x} {at_y}){turn}"/>'
    fill, ink = (("p0", "p0-ink"), ("p1", "p1-ink"))[player] if player <= 1 else ("p2", "p2-ink")
    ids = shape.get("pieceIds") or []
    letter = str(ids[kind]) if kind < len(ids) else "?"
    size = cell * (0.4 if len(letter) < 2 else 0.3)
    return (f'<circle cx="{_num(cx)}" cy="{_num(cy)}" r="{_num(cell * 0.34)}" '
            f'fill="{LIGHT[fill]}" stroke="{LIGHT["ink"]}" stroke-width="{_num(cell * 0.025)}"/>'
            f'<text x="{_num(cx)}" y="{_num(cy + size * 0.35)}" font-size="{_num(size)}" '
            f'font-family="{TEXT_FAMILY}" font-weight="600" text-anchor="middle" '
            f'fill="{LIGHT[ink]}">{escape(letter)}</text>')


def board(shape: Mapping[str, Any], left: float, top: float, cell: float,
          flipped: bool = False) -> tuple[str, str]:
    """The board's cells and pieces with the box's top left at (`left`, `top`), and the
    `<defs>` its drawings need: each drawing once, used by every piece that wears it.

    `flipped` turns the board a half turn, the second player at the bottom, as the page
    draws a proven position for that player to move (`proven.js` `provenBoard`): every cell
    is a box the half turn takes to another box of the board's extent, on squares and on
    hexagons alike, and each piece stands upright, as the page's still board leaves it."""
    width, height = int(shape["width"]), int(shape["height"])
    hexed = shape.get("lattice") == "hex"
    absent = set(shape.get("absent") or [])
    walls = set(shape.get("obstacles") or [])
    x0, y0, x1, y1 = _box(shape)
    cells: list[str] = []
    pieces: list[str] = []
    drawings: dict[str, str] = {}
    hrefs: dict[str, str] = {}
    for pair in shape.get("icons") or []:
        for icon in pair or []:
            if icon:
                hrefs[str(icon.get("stem") or icon["href"])] = str(icon["href"])
    board_bytes = list(shape.get("board") or [])
    for rank in range(height - 1, -1, -1):
        for file in range(width):
            square = rank * width + file
            if square in absent:
                continue
            u, v = cell_place(square, width, height, hexed)
            if flipped:
                u, v = x0 + x1 - 1 - u, y0 + y1 - 1 - v
            x, y = left + (u - x0) * cell, top + (v - y0) * cell
            cells.append(_cell(x, y, cell, hexed, f'fill="{shade(file, rank, hexed)}"'))
            if square in walls:
                cells.append(_cell(x, y, cell, hexed, f'fill="{LIGHT["wall"]}"'))
            value = int(board_bytes[square]) if square < len(board_bytes) else 0
            if value:
                pieces.append(_piece(shape, value, x, y, cell, drawings))
    side = _icon_side(cell)
    defs = "".join(
        f'<image id="{ref}" href="{escape(hrefs[stem], quote=True)}" '
        f'width="{side}" height="{side}" preserveAspectRatio="xMidYMid meet"/>'
        for stem, ref in drawings.items()
    )
    return "".join(cells) + "".join(pieces), defs


def _icon_side(cell: float) -> int:
    """A drawing's side in whole pixels: 88% of its cell, as the page draws it."""
    return max(1, round(cell * 0.88))


# --------------------------------------------------------------------------- the picture


def svg(name: str, shape: Mapping[str, Any], words: str | Sequence[str] = "",
        flipped: bool = False) -> str:
    """The whole picture as SVG: the mark and the site's name, the game's name and `words`
    (its board and pieces, `game.board_words`; or several lines, as a proven position's
    picture says whose move it is and what to find, G10) on the left, the board on the
    right, turned a half turn where `flipped`."""
    under_lines = [words] if isinstance(words, str) else list(words)
    under_lines = [line for line in under_lines if line]
    x0, y0, x1, y1 = _box(shape)
    span_x, span_y = x1 - x0, y1 - y0
    most_x, most_y = BOARD_MOST[0] - 2 * FRAME, BOARD_MOST[1] - 2 * FRAME
    cell = min(most_x / span_x, most_y / span_y, CELL_MOST)
    inner_w, inner_h = span_x * cell, span_y * cell
    outer_w, outer_h = inner_w + 2 * FRAME, inner_h + 2 * FRAME
    frame_x = WIDTH - 64 - outer_w
    frame_y = (HEIGHT - outer_h) / 2
    drawn, defs = board(shape, frame_x + FRAME, frame_y + FRAME, cell, flipped)

    room = frame_x - GAP - MARGIN
    size, lines = name_lines(name, room)
    leading = size * NAME_LEADING
    block = MARK + 40 + leading * len(lines) + (
        24 + LINE_SIZE * 1.3 * len(under_lines) if under_lines else 0)
    top = (HEIGHT - block) / 2
    art = MARK / 16
    mark = (f'<g transform="translate({MARGIN} {_num(top)}) scale({_num(art)})">'
            f'<path fill="{LIGHT["ink"]}" d="{shell.mark_path()}"/></g>')
    site = (f'<text x="{_num(MARGIN + MARK + 18)}" y="{_num(top + MARK / 2 + SITE_SIZE * 0.35)}" '
            f'font-family="{TEXT_FAMILY}" font-weight="600" font-size="{SITE_SIZE}" '
            f'fill="{LIGHT["muted"]}">{escape(shell.NAME)}</text>')
    first = top + MARK + 40 + size * 0.82
    heading = "".join(
        f'<text x="{MARGIN}" y="{_num(first + leading * index)}" font-family="{HEADING_FAMILY}" '
        f'font-weight="600" font-size="{size}" fill="{LIGHT["ink"]}">{escape(line)}</text>'
        for index, line in enumerate(lines)
    )
    under = ""
    for index, line in enumerate(under_lines):
        y = first + leading * (len(lines) - 1) + size * 0.25 + 24 + LINE_SIZE
        y += LINE_SIZE * 1.3 * index
        under += (f'<text x="{MARGIN}" y="{_num(y)}" font-family="{TEXT_FAMILY}" '
                  f'font-weight="600" font-size="{LINE_SIZE}" fill="{LIGHT["muted"]}">'
                  f"{escape(line)}</text>")
    frame = (f'<rect x="{_num(frame_x)}" y="{_num(frame_y)}" width="{_num(outer_w)}" '
             f'height="{_num(outer_h)}" rx="8" fill="{LIGHT["frame"]}"/>')
    shadow = (f'<rect x="{_num(frame_x)}" y="{_num(frame_y + 6)}" width="{_num(outer_w)}" '
              f'height="{_num(outer_h)}" rx="10" fill="#28190a" opacity="0.16"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}"><defs>{defs}</defs>'
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{LIGHT["bg"]}"/>'
        f"{mark}{site}{heading}{under}{shadow}{frame}{drawn}</svg>"
    )


def faces() -> tuple[Path, ...]:
    return (HEADING_FACE[1], TEXT_FACE[1])


@lru_cache(maxsize=1)
def _faces_digest() -> str:
    digest = hashlib.sha256()
    for path in faces():
        digest.update(path.read_bytes())
    return digest.hexdigest()


def key(drawing: str) -> str:
    """A picture's key: the digest of its drawing, its faces and its renderer, so the same
    drawing is one file and anything that would move a pixel is a new one."""
    found = hashlib.sha256(f"game picture 1\0{renderer()}\0{_faces_digest()}\0".encode())
    found.update(drawing.encode("utf-8"))
    return found.hexdigest()[:16]


def file_name(digest: str, picture_key: str) -> str:
    return f"{digest}-{picture_key}.png"


def path(digest: str, picture_key: str) -> str:
    """A game's picture's address on this server."""
    return PREFIX + file_name(digest, picture_key)


def position_file_name(digest: str, position: str, picture_key: str) -> str:
    """A proven position's picture under the pictures' directory: a directory a game, the
    file named by the position's name and the picture's key (G10)."""
    return f"proven/{digest}/{position}-{picture_key}.png"


def position_path(digest: str, position: str, picture_key: str) -> str:
    """A proven position's picture's address on this server (G10)."""
    return f"{PROVEN_PREFIX}{digest}/{position}-{picture_key}.png"


_DRAWING = re.compile(
    r'<image id="(d\d+)" href="(data:image/svg\+xml[^"]*)" width="(\d+)" height="(\d+)"')


def png(drawing: str) -> bytes:
    """The drawing as PNG, through resvg with the picture's own faces and nothing of the
    machine's. ImportError where resvg is not installed (`available`).

    Each piece's drawing is set once at its own size first and placed as a PNG: resvg lays
    an SVG inside an SVG out on a layer the size of the whole picture, about 30 ms a piece
    at 1200 × 630, where the drawing alone at 55 px is under 2 ms and a PNG placed is next
    to nothing. The drawing's text, and so the key, is the SVG as `svg` wrote it."""
    import resvg_py

    def own_size(found: re.Match[str]) -> str:
        ref, href, width, height = found.groups()
        alone = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
                 f'<image href="{href}" width="{width}" height="{height}" '
                 'preserveAspectRatio="xMidYMid meet"/></svg>')
        set_alone = bytes(resvg_py.svg_to_bytes(svg_string=alone, skip_system_fonts=True))
        return (f'<image id="{ref}" href="data:image/png;base64,'
                f'{base64.b64encode(set_alone).decode("ascii")}" width="{width}" '
                f'height="{height}"')

    return bytes(resvg_py.svg_to_bytes(
        svg_string=_DRAWING.sub(own_size, drawing), skip_system_fonts=True,
        font_files=[str(face) for face in faces()],
        width=WIDTH, height=HEIGHT,
    ))


# --------------------------------------------------------------------------- measuring


def _peak_rss() -> int:
    """This process's peak resident memory so far, in bytes (Linux and Windows)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        kernel.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        if not kernel.K32GetProcessMemoryInfo(
                kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def measure(files: Sequence[Path], out: Path | None, repeat: int) -> dict[str, Any]:
    """What a picture costs, in two phases, as the site pays it: first every ruleset's still
    board (the core's compile, which the site's card answers usually hold already), then
    every picture `1 + repeat` times. Each row says the picture's CPU seconds (the first
    draw, and the median of the repeats), its SVG's, its bytes; the summary says the
    process's peak memory before and after the picture phase, whose rise is the most that
    drawing one picture at a time adds to a process that already holds the still boards."""
    from chess_variant_sandbox.web import library
    from chess_variant_sandbox.web.page import game as drawing

    games = []
    began = time.process_time()
    for file in files:
        spec = json.loads(file.read_text(encoding="utf-8"))
        name = str(spec.get("name") or file.stem)
        game = library.Game(hash=file.stem, name=name, description="", path=file,
                            source="rulesets", spec=spec)
        games.append((file, game, library.opening(game)))
    stills = time.process_time() - began
    before = _peak_rss()
    rows: list[dict[str, Any]] = []
    for file, game, shape in games:
        if shape is None:
            rows.append({"file": file.name, "drawn": False})
            continue
        began = time.process_time()
        drawn = svg(game.name, shape, drawing.board_words(game.spec))
        drew = time.process_time()
        made = png(drawn)
        first = time.process_time() - drew
        again = []
        for _ in range(repeat):
            began_again = time.process_time()
            if png(svg(game.name, shape, drawing.board_words(game.spec))) != made:
                raise RuntimeError(f"{file.name}: two drawings of one picture differ")
            again.append(time.process_time() - began_again)
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{file.stem}.png").write_bytes(made)
        rows.append({
            "file": file.name, "drawn": True, "svg_s": round(drew - began, 4),
            "first_s": round(first, 4),
            "median_s": round(sorted(again)[len(again) // 2], 4) if again else None,
            "bytes": len(made),
        })
    drawn_rows = [row for row in rows if row["drawn"]]

    def middle(key: str) -> Any:
        values = sorted(row[key] for row in drawn_rows if row[key] is not None)
        return values[len(values) // 2] if values else None

    return {
        "rows": rows,
        "summary": {
            "pictures": len(drawn_rows), "undrawable": len(rows) - len(drawn_rows),
            "renderer": renderer(), "still_boards_s": round(stills, 3),
            "median_s": middle("median_s"), "slowest_s": max(
                (row["median_s"] or row["first_s"] for row in drawn_rows), default=None),
            "median_first_s": middle("first_s"), "median_svg_s": middle("svg_s"),
            "median_bytes": middle("bytes"),
            "largest_bytes": max((row["bytes"] for row in drawn_rows), default=None),
            "peak_mb_before_pictures": round(before / 2**20, 1),
            "peak_mb_after_pictures": round(_peak_rss() / 2**20, 1),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Draw games' link-preview pictures and say what each cost.")
    parser.add_argument("--measure", nargs="+", type=Path, required=True,
                        help="ruleset files to draw")
    parser.add_argument("--out", type=Path, help="where to write the PNGs (default: nowhere)")
    parser.add_argument("--repeat", type=int, default=5,
                        help="draws per picture after the first, for the median (default 5)")
    args = parser.parse_args(argv)
    found = measure(args.measure, args.out, args.repeat)
    for row in found["rows"]:
        print(json.dumps(row))
    print(json.dumps(found["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
