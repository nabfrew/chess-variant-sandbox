"""The shell every page wears (plan-visual.md §6, §3.4, §3.6; task V2): the mark and the
icons made from it, the header and footer on both servers, and the shell's script, which
sets the viewer's theme before the first paint, keeps their choice, and opens the phone's
menu. The script runs here in Node against a DOM of its own; nothing here plays a game."""

from __future__ import annotations

import json
import re
import struct
import threading
import zlib
from http.client import HTTPConnection
from pathlib import Path

import node_session
import page_names
import pytest

from chess_variant_sandbox.site import routes
from chess_variant_sandbox.web import library, page
from chess_variant_sandbox.web.page import shell
from chess_variant_sandbox.web.page.tokens import DARK, LIGHT
from chess_variant_sandbox.web.server import App, Server, handler_for

RULESETS = Path(page.__file__).resolve().parents[3] / "rulesets"


def _pages() -> dict[str, str]:
    return {"/": page.PAGE, **page_names.site_pages()}


def _png_size_and_pixels(data: bytes) -> tuple[int, list[bytes]]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    at, idat, size = 8, b"", 0
    while at < len(data):
        (length,) = struct.unpack(">I", data[at:at + 4])
        kind, body = data[at + 4:at + 8], data[at + 8:at + 8 + length]
        (crc,) = struct.unpack(">I", data[at + 8 + length:at + 12 + length])
        assert zlib.crc32(kind + body) == crc, kind
        if kind == b"IHDR":
            size = struct.unpack(">I", body[:4])[0]
        if kind == b"IDAT":
            idat += body
        at += 12 + length
    raw = zlib.decompress(idat)
    stride = 1 + 4 * size
    return size, [raw[row * stride + 1:(row + 1) * stride] for row in range(size)]


class TestTheMark:
    def test_the_drawing_is_the_invader_rook_q626_drew(self) -> None:
        """Sixteen by sixteen, symmetric left to right like the piece, with battlements,
        two eyes, and legs where the base would be."""
        assert len(shell.ROOK) == 16 and all(len(row) == 16 for row in shell.ROOK)
        assert all(row == row[::-1] for row in shell.ROOK)
        assert shell.ROOK[1] == ".##..##..##..##."
        assert shell.ROOK[5] == "..##..####..##.."
        cells = sum(row.count("#") for row in shell.ROOK)
        filled = sum(int(part.split("h")[1].split("v")[0])
                     for part in shell.mark_path().split("M")[1:])
        assert filled == cells

    def test_the_header_draws_it_at_32_pixels_with_the_name(self) -> None:
        """Two screen pixels to an art pixel; the name stays *Chess variant sandbox*."""
        for path, html in _pages().items():
            header = html[html.index('<header class="top">'):html.index("</header>")]
            assert 'class="mark" viewBox="0 0 16 16" width="32" height="32"' in header, path
            assert 'shape-rendering="crispEdges"' in header, path
            assert '<span class="word">Chess variant sandbox</span>' in header, path

    def test_the_favicon_svg_is_the_mark_in_ink_and_cream_in_a_dark_tab(self) -> None:
        svg = shell.favicon_svg().decode("utf-8")
        assert shell.mark_path() in svg
        assert LIGHT["ink"] in svg and DARK["ink"] in svg
        assert "prefers-color-scheme:dark" in svg

    def test_the_pngs_are_whole_pixels_of_the_drawing(self) -> None:
        """At 16 one screen pixel an art pixel, at 32 two, at 180 nine on paper."""
        icons = shell.icons()
        touch, kind = icons["/apple-touch-icon.png"]
        assert kind == "image/png"
        size, rows = _png_size_and_pixels(touch)
        assert size == 180
        ground = bytes((*bytes.fromhex(LIGHT["bg"][1:]), 255))
        ink = bytes((*bytes.fromhex(LIGHT["ink"][1:]), 255))
        assert rows[0][:4] == ground
        # The top-left battlement's first art pixel: art (1, 1), offset 18, scale 9.
        assert rows[27][4 * 27:4 * 28] == ink and rows[26][4 * 26:4 * 27] == ground
        for small in (16, 32):
            image = shell.png(small, LIGHT["act"])
            got, pixels = _png_size_and_pixels(image)
            assert got == small
            scale = small // 16
            lit = {
                (x // scale, y // scale)
                for y, row in enumerate(pixels) for x in range(small) if row[4 * x + 3] == 255
            }
            assert lit == {
                (x, y) for y, row in enumerate(shell.ROOK) for x, c in enumerate(row) if c == "#"
            }

    def test_the_ico_holds_16_and_32(self) -> None:
        data, kind = shell.icons()["/favicon.ico"]
        assert kind == "image/x-icon"
        reserved, kind_code, count = struct.unpack("<HHH", data[:6])
        assert (reserved, kind_code, count) == (0, 1, 2)
        sizes = []
        for entry in range(count):
            width, _, _, _, _, _, length, at = struct.unpack(
                "<BBBBHHII", data[6 + 16 * entry:22 + 16 * entry]
            )
            got, _ = _png_size_and_pixels(data[at:at + length])
            assert got == width
            sizes.append(width)
        assert sizes == [16, 32]


class TestTheShell:
    def test_every_page_links_the_icons_and_loads_the_shell_first(self) -> None:
        for path, html in _pages().items():
            head = html[:html.index("</head>")]
            assert '<link rel="icon" href="/favicon.svg" type="image/svg+xml">' in head, path
            assert '<link rel="icon" href="/favicon.ico"' in head, path
            assert '<link rel="apple-touch-icon" href="/apple-touch-icon.png">' in head, path
            # Before the stylesheet, so the theme is on <html> before anything is painted.
            assert head.index(shell.SHELL_SCRIPT_ROUTE) < head.index("<style>"), path

    def test_every_page_has_the_sections_the_menu_and_the_theme_switch(self) -> None:
        for path, html in _pages().items():
            assert 'id="sections"' in html and 'id="menu"' in html, path
            assert re.search(r'<button type="button" class="menu" id="menu"[^>]* hidden>', html)
            for pick in shell.THEMES:
                assert f'data-theme-pick="{pick}"' in html, path
        # The site's pages are other addresses, so their sections point into the app.
        assert '<a href="/#/design">Build</a>' in page_names.site_pages()["/credits"]
        assert '<a href="#/design">Build</a>' in page.PAGE

    def test_the_app_keeps_the_ids_its_views_write_into(self) -> None:
        for name in ("home", "tagline", "runs", "who", "foot", "view"):
            assert f'id="{name}"' in page.PAGE, name

    def test_the_icon_and_shell_routes_are_constants_served_to_anybody(self) -> None:
        for route in (*shell.icons(), shell.SHELL_SCRIPT_ROUTE, shell.FONT_PREFIX):
            assert route in routes.STATIC, route


def _get(*paths: str) -> list[tuple[int, dict[str, str], bytes]]:
    app = App(RULESETS, library.RUNS_DIR)
    server = Server(("127.0.0.1", 0), handler_for(app))
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02},
                     daemon=True).start()
    answers = []
    try:
        for path in paths:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            try:
                connection.request("GET", path)
                response = connection.getresponse()
                answers.append((response.status, dict(response.getheaders()), response.read()))
            finally:
                connection.close()
    finally:
        server.shutdown()
        server.server_close()
    return answers


class TestTheServerAnswersThem:
    def test_icons_fonts_and_the_shell_script(self) -> None:
        """N7: `/favicon.svg` answered 404 on both servers; the site inherits these
        routes from this handler."""
        font = next(iter(shell.fonts()))
        asked = (*shell.icons(), shell.SHELL_SCRIPT_ROUTE, shell.FONT_PREFIX + font)
        answers = dict(zip(asked, _get(*asked), strict=True))
        for route, (body, kind) in shell.icons().items():
            status, headers, got = answers[route]
            assert (status, headers["Content-Type"], got) == (200, kind, body), route
            assert headers["Cache-Control"] == shell.ICON_CACHE
        status, headers, got = answers[shell.SHELL_SCRIPT_ROUTE]
        assert status == 200 and headers["Content-Type"] == "text/javascript"
        assert got == shell.SHELL_SCRIPT_FILE.read_bytes()
        status, headers, got = answers[shell.FONT_PREFIX + font]
        assert status == 200 and headers["Content-Type"] == "font/woff2"
        assert "immutable" in headers["Cache-Control"]
        assert got == (shell.FONT_DIR / font).read_bytes()

    def test_nothing_else_under_the_font_prefix(self) -> None:
        asked = (shell.FONT_PREFIX + "manifest.json", shell.FONT_PREFIX + "../shell.js",
                 shell.FONT_PREFIX + "nothing.woff2", "/favicon.png")
        for route, (status, _, _) in zip(asked, _get(*asked), strict=True):
            assert status == 404, route


# A DOM with only what the shell's script touches: <html>'s attributes and classes, the
# theme buttons, the menu button and the sections, events and localStorage.
_DOM = """
function node(attrs) {
  const listeners = {};
  const el = {
    attrs: Object.assign({}, attrs || {}), dataset: {}, hidden: true, children: [],
    classes: new Set(),
    classList: { add: c => el.classes.add(c), contains: c => el.classes.has(c),
                 toggle: (c, on) => { if (on === undefined ? !el.classes.has(c) : on)
                   el.classes.add(c); else el.classes.delete(c); } },
    setAttribute(k, v) { el.attrs[k] = String(v); },
    getAttribute(k) { return k in el.attrs ? el.attrs[k] : null; },
    removeAttribute(k) { delete el.attrs[k]; },
    addEventListener(kind, fn) { (listeners[kind] = listeners[kind] || []).push(fn); },
    fire(kind, event) { (listeners[kind] || []).forEach(fn => fn(event || { stopPropagation() {} })); },
    contains(other) { return other === el || el.children.includes(other); },
    querySelectorAll() { return el.children; },
  };
  return el;
}
const root = node();
const picks = ['light', 'dark', 'system'].map(pick => {
  const b = node(); b.dataset.themePick = pick; return b; });
const menu = node(), sections = node();
sections.children = ['/game', '#/design', '#/whats-new'].map(h => node({ href: h }));
const docListeners = {}, fired = [];
globalThis.document = {
  documentElement: root,
  getElementById: id => ({ menu, sections })[id] || null,
  querySelectorAll: () => picks,
  addEventListener(kind, fn) { (docListeners[kind] = docListeners[kind] || []).push(fn); },
  dispatchEvent(event) { fired.push(event.type); },
};
globalThis.Event = function (type) { this.type = type; };
const store = {};
globalThis.window = {
  location: { pathname: '/', hash: '#/design/abc' },
  addEventListener() {},
  localStorage: STORAGE,
};
"""

_STORE = """{ getItem: k => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = v; },
               removeItem: k => { delete store[k]; } }"""
_REFUSED = """{ getItem() { throw new Error('denied'); }, setItem() { throw new Error('denied'); },
                 removeItem() { throw new Error('denied'); } }"""


def _run(storage: str, before: str, after: str) -> dict[str, object]:
    script = shell.SHELL_SCRIPT_FILE.read_text(encoding="utf-8")
    source = (
        _DOM.replace("STORAGE", storage) + before + "\n" + script
        + "\nconst atPaint = root.getAttribute('data-theme');\n"
        + "(docListeners.DOMContentLoaded || []).forEach(fn => fn());\n" + after
    )
    done = node_session.run(source)
    assert done.returncode == 0, done.stderr
    said: dict[str, object] = json.loads(done.stdout.strip().splitlines()[-1])
    return said


class TestTheShellScript:
    def test_a_kept_choice_is_on_html_before_the_first_paint(self) -> None:
        said = _run(_STORE, "store.theme = 'dark';",
                    "console.log(JSON.stringify({ atPaint, js: root.classes.has('js'),"
                    " pressed: picks.map(b => b.attrs['aria-pressed']) }));")
        assert said == {"atPaint": "dark", "js": True, "pressed": ["false", "true", "false"]}

    def test_no_choice_follows_the_system(self) -> None:
        said = _run(_STORE, "", "console.log(JSON.stringify({ atPaint }));")
        assert said == {"atPaint": None}

    def test_a_pick_is_applied_kept_and_announced(self) -> None:
        said = _run(_STORE, "", """
          picks[0].fire('click');
          const light = [root.getAttribute('data-theme'), store.theme];
          picks[2].fire('click');
          console.log(JSON.stringify({ light, system: [root.getAttribute('data-theme'),
            store.theme === undefined], fired }));""")
        assert said == {"light": ["light", "light"], "system": [None, True],
                        "fired": ["themechange", "themechange"]}

    def test_refused_storage_leaves_the_page_on_the_system_and_working(self) -> None:
        said = _run(_REFUSED, "", """
          picks[1].fire('click');
          console.log(JSON.stringify({ atPaint, now: root.getAttribute('data-theme') }));""")
        assert said == {"atPaint": None, "now": "dark"}

    def test_the_menu_opens_and_closes_and_marks_the_section(self) -> None:
        said = _run(_STORE, "", """
          const shown = !menu.hidden;
          menu.fire('click');
          const opened = [root.classes.has('menu-open'), menu.attrs['aria-expanded']];
          sections.fire('click');
          console.log(JSON.stringify({ shown, opened,
            closed: [root.classes.has('menu-open'), menu.attrs['aria-expanded']],
            current: sections.children.map(a => a.attrs['aria-current'] || '') }));""")
        assert said == {"shown": True, "opened": [True, "true"], "closed": [False, "false"],
                        "current": ["", "page", ""]}


# Q723: the landing page is `/` for everybody and in no section; the games are `/game`, and
# every hash route still works at `/`, a saved `/#/` being the games.
@pytest.mark.parametrize("path, hash_, current", [
    ("/", "", ["", "", ""]), ("/game", "", ["page", "", ""]), ("/", "#/", ["page", "", ""]),
    ("/", "#/game/abc", ["page", "", ""]), ("/game", "#/game/abc", ["page", "", ""]),
    ("/", "#/whats-new", ["", "", "page"]), ("/", "#/round", ["", "", ""]),
    ("/game", "#/about", ["", "", ""]), ("/players", "", ["", "", ""]),
])
def test_the_section_follows_the_route(path: str, hash_: str, current: list[str]) -> None:
    said = _run(_STORE, f"window.location.pathname = {json.dumps(path)};"
                f"window.location.hash = {json.dumps(hash_)};",
                "console.log(JSON.stringify({ current: sections.children.map("
                "a => a.attrs['aria-current'] || '') }));")
    assert said == {"current": current}
