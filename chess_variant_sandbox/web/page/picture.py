"""The site's picture for a link preview (G5 of `docs/plan-growth.md`, §2.2).

A link pasted into a chat program or a post is unfurled from the page's Open Graph and
Twitter card tags (`front.head_lines`), and their picture is this one: 1200 × 630, the size
both ask for, in the club room's look (`docs/plan-visual.md` §4): paper, the invader rook
and the name in Fraunces on the left, the site's headline under it, and on the right a
walnut-framed maple board with invader rooks standing on it as pieces. Every colour is a
light-theme token (`tokens.LIGHT`), the faces are the site's own (`shell.fonts`), the rooks
are `shell.ROOK`; nothing in it is anybody else's drawing, so it carries no licence beyond
the repository's.

**How it is made.** Unfurlers take PNG or JPEG, not SVG, and the picture has words in the
site's faces, which the standard library cannot set. So `source()` writes the picture as
one HTML page, and `render` photographs it once with a headless Chrome into
`web/static/picture/`, beside a `manifest.json` row: the file, its size, its digest, and the
digest of the page it was photographed from. The file's name carries the start of its own
digest, so a new picture is a new address and the route can answer with a year's cache, as
the fonts do. Kept under `BYTES_MOST`, since some unfurlers drop a larger picture. A test
compares the row's source digest with `source()`'s: change a token,
the mark or the headline the picture shows, and the suite asks for a new photograph,
naming the command:

    uv run python -m chess_variant_sandbox.web.page.picture

**G6** gives each game a picture of its own starting position. Until then a game's page
points at this one through `web/page/game.py`'s `picture`, the one place G6 changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

from chess_variant_sandbox.web import landing
from chess_variant_sandbox.web.page import shell
from chess_variant_sandbox.web.page.tokens import GRAIN_DK, GRAIN_LT, LIGHT, WALNUT, noise

PREFIX = "/static/picture/"
"""Where the pictures are served, by a lookup in `MANIFEST` and nothing else."""

DIR = shell.STATIC_DIR / "picture"
MANIFEST = DIR / "manifest.json"

WIDTH = 1200
HEIGHT = 630
TYPE = "image/png"

BYTES_MOST = 300_000
"""The largest picture this makes: some unfurlers (WhatsApp is the one usually named) are
reported to show no picture above about 300 KB."""

ALT = f"{shell.NAME}: the invader rook beside a wooden board"
"""What the picture shows, for a reader who cannot see it (`og:image:alt`)."""

# The board on the right: eight squares a side, each `SQUARE` px, in a walnut frame.
SQUARE = 58
FILES = 8
FRAME = 16
BOARD = FILES * SQUARE + 2 * FRAME

# Where the rooks stand, (file, rank from the top, side): a game a few moves in, both sides
# spread out, no square named and nothing a reader could take for a real position.
STANDING: tuple[tuple[int, int, str], ...] = (
    (1, 0, "dark"), (5, 0, "dark"), (3, 1, "dark"), (6, 2, "dark"),
    (2, 3, "dark"), (4, 4, "light"), (1, 5, "light"), (6, 5, "light"),
    (3, 6, "light"), (0, 7, "light"), (5, 7, "light"),
)


def outline(rows: tuple[str, ...] = shell.ROOK) -> tuple[str, ...]:
    """The drawing one art pixel larger all round, the added ring marked `#`: the ink edge a
    light piece is read by on a light square (`plan-visual.md` §3.2)."""
    size = len(rows) + 2

    def on(x: int, y: int) -> bool:
        return 0 <= y - 1 < len(rows) and 0 <= x - 1 < len(rows[0]) and rows[y - 1][x - 1] == "#"

    return tuple(
        "".join(
            "#" if not on(x, y) and any(on(x + dx, y + dy) for dx in (-1, 0, 1)
                                        for dy in (-1, 0, 1)) else "."
            for x in range(size)
        )
        for y in range(size)
    )


def _path(rows: tuple[str, ...], shift: int = 0) -> str:
    """`shell.mark_path` for a drawing of any size, moved `shift` art pixels in."""
    parts = []
    for y, row in enumerate(rows):
        x = 0
        while x < len(row):
            if row[x] != "#":
                x += 1
                continue
            start = x
            while x < len(row) and row[x] == "#":
                x += 1
            parts.append(f"M{start + shift} {y + shift}h{x - start}v1h{start - x}z")
    return "".join(parts)


def piece(side: str) -> str:
    """One invader rook as a piece: ink for the dark side; paper inside an ink edge for the
    light one."""
    if side == "dark":
        body = f'<path fill="{LIGHT["ink"]}" d="{_path(shell.ROOK, 1)}"/>'
    else:
        body = (f'<path fill="{LIGHT["panel"]}" d="{_path(shell.ROOK, 1)}"/>'
                f'<path fill="{LIGHT["ink"]}" d="{_path(outline())}"/>')
    return ('<svg class="piece" viewBox="0 0 18 18" shape-rendering="crispEdges">'
            f"{body}</svg>")


def _square(file: int, rank: int) -> str:
    light = (file + rank) % 2 == 0
    # Each square its own window of its wood, as `wood.js` lays them on a play board.
    x, y = (file * 97 + rank * 41) % 256, (rank * 113 + file * 29) % 256
    kind = "lt" if light else "dk"
    standing = [side for f, r, side in STANDING if (f, r) == (file, rank)]
    return (f'<div class="sq {kind}"><i style="background-position:-{x}px -{y}px"></i>'
            + (piece(standing[0]) if standing else "") + "</div>")


def source() -> str:
    """The picture as one HTML page, which `render` photographs. Its fonts are named by file
    and found beside it, in `fonts/`, when it is photographed."""
    faces = "".join(
        f'@font-face{{font-family:"{entry["family"]}";src:url("fonts/{name}") '
        f'format("woff2");font-weight:{entry["weight"]};font-display:block}}'
        for name, entry in sorted(shell.fonts().items())
        if entry["role"] in ("headings", "text")
    )
    heading = next(e["family"] for e in shell.fonts().values() if e["role"] == "headings")
    text = next(e["family"] for e in shell.fonts().values() if e["role"] == "text")
    squares = "".join(_square(f, r) for r in range(FILES) for f in range(FILES))
    left = 72
    words = 1200 - BOARD - left - 64 - left
    return (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\"><style>"
        f"{faces}"
        "*{box-sizing:border-box;margin:0}"
        f"html,body{{width:{WIDTH}px;height:{HEIGHT}px;overflow:hidden}}"
        # The ground is the paper's colour without its fibre: the fibre is invisible at the
        # size a chat program shows a picture, and it doubles the file (380 KB against 170),
        # past the 300 KB some unfurlers are reported to stop showing a picture at.
        f"body{{background:{LIGHT['bg']};color:{LIGHT['ink']};"
        f"font-family:\"{text}\",sans-serif;position:relative}}"
        f".words{{position:absolute;left:{left}px;top:0;bottom:0;width:{words}px;"
        "display:flex;flex-direction:column;justify-content:center}"
        # The mark at seven pixels an art pixel, whole pixels as in the header.
        ".mark{width:112px;height:112px;display:block;margin-bottom:30px}"
        f"h1{{font:600 66px/1.04 \"{heading}\",serif;letter-spacing:-.5px}}"
        f".line{{font-size:29px;line-height:1.38;color:{LIGHT['muted']};margin-top:26px}}"
        f".board{{position:absolute;right:{left}px;top:{(HEIGHT - BOARD) // 2}px;"
        f"width:{BOARD}px;height:{BOARD}px;padding:{FRAME}px;border-radius:8px;"
        f"background:{noise(*WALNUT)} 0 0/240px,{LIGHT['frame']};"
        "box-shadow:0 10px 34px rgba(40,25,10,.28),0 2px 6px rgba(40,25,10,.18);"
        f"display:grid;grid-template-columns:repeat({FILES},{SQUARE}px)}}"
        f".sq{{width:{SQUARE}px;height:{SQUARE}px;position:relative}}"
        f".sq.lt{{background:{LIGHT['sq-lt']}}}.sq.dk{{background:{LIGHT['sq-dk']}}}"
        ".sq i{position:absolute;inset:0}"
        f".sq.lt i{{background-image:{noise(*GRAIN_LT)};opacity:{LIGHT['grain-lt-a']}}}"
        f".sq.dk i{{background-image:{noise(*GRAIN_DK)};opacity:{LIGHT['grain-dk-a']}}}"
        f".piece{{position:absolute;left:{(SQUARE - 54) // 2}px;top:{(SQUARE - 54) // 2}px;"
        "width:54px;height:54px}"
        "</style></head><body>"
        '<div class="words">'
        '<svg class="mark" viewBox="0 0 16 16" shape-rendering="crispEdges">'
        f'<path fill="{LIGHT["ink"]}" d="{shell.mark_path()}"/></svg>'
        f"<h1>{escape(shell.NAME)}</h1>"
        f'<p class="line">{escape(landing.HEADLINE)}</p>'
        "</div>"
        f'<div class="board">{squares}</div>'
        "</body></html>\n"
    )


def source_digest() -> str:
    return hashlib.sha256(source().encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------- the served file


@lru_cache(maxsize=1)
def pictures() -> dict[str, dict[str, Any]]:
    """File name -> its manifest row: the whole of what `PREFIX` serves."""
    held = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {entry["file"]: entry for entry in held["files"]}


def site_entry() -> dict[str, Any]:
    (entry,) = [entry for entry in pictures().values() if entry["role"] == "site"]
    return entry


def site_path() -> str:
    """The site's picture's address on this server, a path: the head writes it on the
    site's own address."""
    return PREFIX + site_entry()["file"]


def size(png: bytes) -> tuple[int, int]:
    """A PNG's width and height, from its header."""
    if png[:8] != b"\x89PNG\r\n\x1a\n" or png[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    width, height = struct.unpack(">II", png[16:24])
    return width, height


# ------------------------------------------------------------------------- making it

WINDOWS_CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe"


def render(chrome: Path) -> dict[str, Any]:
    """Photograph `source()` at 1200 × 630 with a headless Chrome, put the PNG in `DIR`
    under a name carrying its digest, and write its manifest row (the old picture goes)."""
    page = source()
    with tempfile.TemporaryDirectory() as scratch:
        where = Path(scratch)
        (where / "fonts").mkdir()
        for name in shell.fonts():
            shutil.copy(shell.FONT_DIR / name, where / "fonts" / name)
        (where / "picture.html").write_text(page, encoding="utf-8")
        shot = where / "picture.png"
        subprocess.run(
            [str(chrome), "--headless=new", "--disable-gpu", "--hide-scrollbars",
             "--force-device-scale-factor=1",
             f"--window-size={WIDTH},{HEIGHT}", f"--screenshot={shot}",
             "--virtual-time-budget=4000", (where / "picture.html").as_uri()],
            check=True, capture_output=True, timeout=120,
        )
        png = shot.read_bytes()
    if size(png) != (WIDTH, HEIGHT):
        raise RuntimeError(f"Chrome photographed {size(png)}, not {(WIDTH, HEIGHT)}")
    if len(png) > BYTES_MOST:
        raise RuntimeError(f"the picture is {len(png)} bytes, over {BYTES_MOST}")
    digest = hashlib.sha256(png).hexdigest()
    name = f"site-{digest[:12]}.png"
    DIR.mkdir(exist_ok=True)
    for old in DIR.glob("site-*.png"):
        old.unlink()
    (DIR / name).write_bytes(png)
    entry = {
        "file": name, "role": "site", "type": TYPE, "width": WIDTH, "height": HEIGHT,
        "bytes": len(png), "sha256": digest,
        "source": hashlib.sha256(page.encode("utf-8")).hexdigest(),
        "alt": ALT,
    }
    MANIFEST.write_text(json.dumps({
        "note": "The site's link-preview picture (G5, `web/page/picture.py`), served at "
                "GET /static/picture/<file> and nothing else, cached for a year: a new "
                "picture is a new file name. `source` is the digest of the page it was "
                "photographed from; remake it with `uv run python -m "
                "chess_variant_sandbox.web.page.picture`.",
        "files": [entry],
    }, indent=2) + "\n", encoding="utf-8")
    pictures.cache_clear()
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Photograph the site's link-preview picture.")
    parser.add_argument("--chrome", default=os.environ.get("CHROME", WINDOWS_CHROME),
                        help="the Chrome to photograph it with (default: $CHROME, else "
                             "where Windows puts it)")
    args = parser.parse_args(argv)
    entry = render(Path(args.chrome))
    print(f"{entry['file']}: {entry['width']} x {entry['height']}, {entry['bytes']} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
