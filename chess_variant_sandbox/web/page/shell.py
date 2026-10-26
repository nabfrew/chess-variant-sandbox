"""What every page wears, on both servers: the typefaces, the mark, the favicon, and the
header's and footer's parts.

The application's page (`web/page/__init__.py`) and the site's own pages
(`site/pages.py`) are two shells drawn from one stylesheet; what they share beyond the
stylesheet lives here, so a change reaches both (plan-visual.md §5.4).

**The typefaces** (§3.1, §5.2): Fraunces for headings, Source Sans 3 for text, Source Code
Pro for numbers and moves. Self-hosted, never fetched from another origin (the privacy
notice's promise, and the content-security policy's `default-src 'self'`): the Latin
subset as woff2 in `web/static/fonts/`, one `manifest.json` row a file with its digest,
source and licence (OFL 1.1, the text beside it), served at `FONT_PREFIX` by a lookup in
that manifest and cached for a year, since a new version is a new file name. Each family
is declared with its system fallback after it, so a page whose fonts are blocked is the
page as it was.

**The mark** (§6, D2): the invader rook, Q626's original 16 × 16 pixel drawing
(`docs/reference/visual-clubroom-twist.py`), a rook's battlements as an arcade invader's
antennae. `ROOK` is the drawing and everything else is made from it here, at import, by
the standard library: the header's inline SVG (32 px, two screen pixels to an art pixel),
`/favicon.svg` (ink, cream in a dark tab), and the PNGs a browser that takes no SVG icon
asks for (`/favicon.ico` at 16 and 32, `/apple-touch-icon.png` at 180 on paper). So there
is no image file to fall out of step with the drawing.

**The shell's script** (`web/static/shell.js`, §3.6, §5.1): a classic script in the head,
so the viewer's own theme is on `<html>` before the first paint and the page never
flashes the other one. It also wires the footer's theme switch and the phone's menu.
"""

from __future__ import annotations

import json
import struct
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any

from chess_variant_sandbox.web.page import assets
from chess_variant_sandbox.web.page.tokens import DARK, LIGHT

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

FONT_PREFIX = "/static/fonts/"
FONT_DIR = STATIC_DIR / "fonts"

SHELL_SCRIPT_ROUTE = "/static/shell.js"
SHELL_SCRIPT_FILE = STATIC_DIR / "shell.js"
# Named by its digest in every head, for a year (G12, `assets.py`).
assets.serve_files({SHELL_SCRIPT_ROUTE: SHELL_SCRIPT_FILE})

STORY_SCRIPT_ROUTE = "/static/story.js"
STORY_SCRIPT_FILE = STATIC_DIR / "story.js"
"""The landing's draw (LP6, `docs/plan-landing.md` §4.1): in the single page's head after the
shell's, it puts the story drawn from `<html data-stories>` on `<html>` before the first
paint. Served as the shell's is, by its digest for a year and by this route `no-store`."""
assets.serve_files({STORY_SCRIPT_ROUTE: STORY_SCRIPT_FILE})

HEAD_SCRIPTS = {SHELL_SCRIPT_ROUTE: SHELL_SCRIPT_FILE, STORY_SCRIPT_ROUTE: STORY_SCRIPT_FILE}
"""The scripts a head carries, outside both `SCRIPTS` tables: route -> file."""


@lru_cache(maxsize=1)
def fonts() -> dict[str, dict[str, Any]]:
    """The font manifest, by file name. Read once: it changes only with a release."""
    manifest = json.loads((FONT_DIR / "manifest.json").read_text(encoding="utf-8"))
    return {entry["file"]: entry for entry in manifest["files"]}


def _family(role: str) -> dict[str, Any]:
    (entry,) = [entry for entry in fonts().values() if entry["role"] == role]
    return entry


def stack(role: str) -> str:
    """A role's family and its system fallback, as a `font-family` value."""
    entry = _family(role)
    return f"'{entry['family']}', {entry['fallback']}"


def faces() -> str:
    """The `@font-face` rules, one a file. `swap`, so text is never invisible waiting."""
    return "".join(
        f"""
  @font-face {{ font-family: '{entry['family']}'; font-style: normal;
    font-weight: {entry['weight']}; font-display: swap;
    src: url({FONT_PREFIX}{entry['file']}) format('woff2'); }}"""
        for entry in fonts().values()
    )


def preloads() -> str:
    """`<link rel=preload>` for the faces every page sets text in, so the first paint
    waits for them less. The mono face is left to load when a number first needs it."""
    return "".join(
        f'<link rel="preload" href="{FONT_PREFIX}{_family(role)["file"]}" as="font" '
        'type="font/woff2" crossorigin>\n'
        for role in ("headings", "text")
    )


# The three families as the stylesheet names them.
HEAD = stack("headings")
BODY = stack("text")
MONO = stack("numbers and moves")


# ------------------------------------------------------------------------------ the mark

ROOK: tuple[str, ...] = (
    "................",
    ".##..##..##..##.",
    ".##..##..##..##.",
    ".##############.",
    "..############..",
    "..##..####..##..",
    "..##..####..##..",
    "..############..",
    "...##########...",
    "...##########...",
    "...##########...",
    "..############..",
    ".##############.",
    ".###.##..##.###.",
    ".##..........##.",
    "##............##",
)


def mark_path(rows: tuple[str, ...] = ROOK) -> str:
    """One SVG path of unit squares, merged along each row so it stays small."""
    assert len(rows) == 16 and all(len(row) == 16 for row in rows)
    parts = []
    for y, row in enumerate(rows):
        x = 0
        while x < 16:
            if row[x] != "#":
                x += 1
                continue
            start = x
            while x < 16 and row[x] == "#":
                x += 1
            parts.append(f"M{start} {y}h{x - start}v1h{start - x}z")
    return "".join(parts)


def mark(size: int = 32) -> str:
    """The mark inline, drawn in the text's colour (`currentColor`), for the header."""
    return (
        f'<svg class="mark" viewBox="0 0 16 16" width="{size}" height="{size}" '
        'shape-rendering="crispEdges" aria-hidden="true" focusable="false">'
        f'<path fill="currentColor" d="{mark_path()}"/></svg>'
    )


def favicon_svg() -> bytes:
    """`/favicon.svg`: the mark in ink, and in cream on a dark browser's tab."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" '
        'shape-rendering="crispEdges"><style>path{fill:'
        f"{LIGHT['ink']}}}@media (prefers-color-scheme:dark){{path{{fill:{DARK['ink']}}}}}"
        f'</style><path d="{mark_path()}"/></svg>'
    ).encode()


def _hex(value: str) -> tuple[int, int, int]:
    return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)


def png(size: int, ink: str, ground: str | None = None, pad: int = 0) -> bytes:
    """The mark as a PNG `size` pixels square: whole screen pixels per art pixel, centred,
    `pad` art-free pixels round it at least, on `ground` (transparent when None)."""
    scale = max(1, (size - 2 * pad) // 16)
    offset = (size - 16 * scale) // 2
    fill = (*_hex(ink), 255)
    back = (0, 0, 0, 0) if ground is None else (*_hex(ground), 255)
    rows = []
    for y in range(size):
        line = bytearray(b"\x00")  # filter type 0 for every scanline
        art_y = (y - offset) // scale if y >= offset else -1
        for x in range(size):
            art_x = (x - offset) // scale if x >= offset else -1
            on = 0 <= art_y < 16 and 0 <= art_x < 16 and ROOK[art_y][art_x] == "#"
            line.extend(fill if on else back)
        rows.append(bytes(line))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
        + chunk(b"IEND", b"")
    )


def ico(images: tuple[bytes, ...], sizes: tuple[int, ...]) -> bytes:
    """An `.ico` holding PNG images, which every browser since 2007 reads."""
    out = struct.pack("<HHH", 0, 1, len(images))
    at = 6 + 16 * len(images)
    for image, size in zip(images, sizes, strict=True):
        out += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(image), at)
        at += len(image)
    return out + b"".join(images)


# The icons are the action green: a mid tone that reads on a light tab bar and a dark one,
# which a PNG cannot choose between as the SVG does.
_ICON_INK = LIGHT["act"]


@lru_cache(maxsize=1)
def icons() -> dict[str, tuple[bytes, str]]:
    """Route -> (bytes, content type), the whole of what the icon routes answer."""
    return {
        "/favicon.svg": (favicon_svg(), "image/svg+xml"),
        "/favicon.ico": (
            ico((png(16, _ICON_INK), png(32, _ICON_INK)), (16, 32)),
            "image/x-icon",
        ),
        # A phone's home screen: the mark on paper, with the margin iOS rounds off.
        "/apple-touch-icon.png": (png(180, LIGHT["ink"], LIGHT["bg"], pad=18), "image/png"),
    }


# A day: the names do not change when the drawing does, so not a year.
ICON_CACHE = "public, max-age=86400"


def head_links() -> str:
    """What every page's `<head>` carries beyond its title: the icons, the shell's script
    (before the stylesheet, so the viewer's theme is set before anything is painted) and
    the font preloads."""
    return (
        '<link rel="icon" href="/favicon.svg" type="image/svg+xml">\n'
        '<link rel="icon" href="/favicon.ico" sizes="16x16 32x32">\n'
        '<link rel="apple-touch-icon" href="/apple-touch-icon.png">\n'
        f'<script src="{assets.name_of(SHELL_SCRIPT_ROUTE)}"></script>\n' + preloads()
    )


# ------------------------------------------------------------------ the header and footer

NAME = "Chess variant sandbox"

TAGLINE = "This is my vibe-slop project. There are many like it, but this one is mine."
"""The header's second line as served, on the application's page and the site's own pages
(the user's words, 2026-10-13). A view may write its one-line summary over it; `core.js`'s
`render` puts this back before a view that writes none."""

# The site's sections (§3.4): where a person goes from any page. The games are an address of
# their own, `/game` (Q723), the same from every page; the rest are hash routes inside the
# application, and `root` is "" on the application's page and "/" on the site's own
# pages, which are other addresses.
SECTIONS: tuple[tuple[str, str], ...] = (
    ("/game", "Games"),
    ("#/design", "Build"),
    ("#/whats-new", "What's new"),
)

PEOPLE = ("/players", "People")
"""Q760: the people page, a section for signed-in members on the site only. It goes after
*Games*. The site's own pages draw it when somebody is signed in (`sections(people=True)`);
the application's page adds it once `/api/me` says a member is reading (`core.js`'s
`paintIdentity`), since the local application has no people and a visitor would be sent to
the door."""


def brand() -> str:
    """The mark and the name, the header's left end."""
    return f'{mark(32)}<span class="word">{NAME}</span>'


def sections(root: str = "", people: bool = False) -> str:
    """The sections, and the button a phone opens them with (`shell.js`). Without the
    script the button stays hidden and the sections wrap under the name instead. `people`
    adds `PEOPLE` after the games."""
    shown = (SECTIONS[0], PEOPLE, *SECTIONS[1:]) if people else SECTIONS
    links = "".join(
        f'<a href="{where if where.startswith("/") else root + where}">{words}</a>'
        for where, words in shown
    )
    return (
        f'<nav class="sections" id="sections" aria-label="sections">{links}</nav>'
        '<button type="button" class="menu" id="menu" aria-controls="sections" '
        'aria-expanded="false" aria-label="menu" hidden>'
        '<svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true" focusable="false">'
        '<path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor" stroke-width="1.6" '
        'stroke-linecap="round"/></svg></button>'
    )


# The footer's theme switch (§3.6): the viewer's own choice, kept in their browser.
THEMES = ("light", "dark", "system")


def theme_switch() -> str:
    buttons = "".join(
        f'<button type="button" data-theme-pick="{pick}" aria-pressed="{str(pick == "system").lower()}"'
        f">{pick}</button>"
        for pick in THEMES
    )
    return f'<span class="themePick" role="group" aria-label="colour theme">{buttons}</span>'


def footer(links: str = "") -> str:
    """The footer on both servers (V7): the mark and the name signing the page, the links
    (`#foot`, which the application's `paintIdentity` writes and the site's pages are
    handed), and the theme switch, which no view rewrites."""
    return (
        '<footer class="muted small">'
        f'<span class="sign">{mark(16)}<span>{NAME}</span></span>'
        f'<span id="foot">{links}</span>{theme_switch()}</footer>'
    )
