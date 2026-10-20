"""The site's link-preview picture (G5 of `docs/plan-growth.md`, §2.2; `web/page/picture.py`).

The picture is a file photographed once from a page the code writes, so what is held here
is that the file in the tree is that photograph and is served as one:

- the manifest names one site picture, whose bytes, digest and name agree, a PNG of
  1200 × 630 under the size some unfurlers stop at;
- it was photographed from the page `picture.source()` writes today: a token, the mark or
  the headline the picture shows changing makes this fail, naming the command that remakes it;
- the page it is photographed from draws only in the light theme's tokens;
- the server answers it at its address as `image/png`, cached for a year, and nothing else
  under that prefix.

The site's pages naming it in their heads are `test_site_doors.py`'s `TestThePreviews`.
Nothing here runs Chrome, a store or a game.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import threading
import zlib
from http.client import HTTPConnection
from pathlib import Path

from chess_variant_sandbox.site import routes, seo
from chess_variant_sandbox.web import library, page
from chess_variant_sandbox.web.page import picture, shell
from chess_variant_sandbox.web.page.tokens import LIGHT
from chess_variant_sandbox.web.server import IMMUTABLE, App, Server, handler_for

RULESETS = Path(page.__file__).resolve().parents[3] / "rulesets"

REMAKE = "uv run python -m chess_variant_sandbox.web.page.picture"


def _png_header(data: bytes) -> tuple[int, int]:
    """A PNG's size, after checking its signature and every chunk's checksum."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    at, size, kinds = 8, (0, 0), []
    while at < len(data):
        (length,) = struct.unpack(">I", data[at:at + 4])
        kind, body = data[at + 4:at + 8], data[at + 8:at + 8 + length]
        (crc,) = struct.unpack(">I", data[at + 8 + length:at + 12 + length])
        assert zlib.crc32(kind + body) == crc, kind
        if kind == b"IHDR":
            size = struct.unpack(">II", body[:8])
        kinds.append(kind)
        at += 12 + length
    assert kinds[0] == b"IHDR" and kinds[-1] == b"IEND"
    return size


class TestTheFile:
    def test_the_manifest_names_one_site_picture_and_it_is_that_file(self) -> None:
        entry = picture.site_entry()
        data = (picture.DIR / entry["file"]).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        assert entry["sha256"] == digest and entry["bytes"] == len(data)
        # The name carries the bytes' digest, so a new picture is a new address and a
        # year's cache is safe.
        assert entry["file"] == f"site-{digest[:12]}.png"
        assert (entry["type"], entry["width"], entry["height"]) == (
            picture.TYPE, picture.WIDTH, picture.HEIGHT)
        assert entry["alt"] == picture.ALT

    def test_it_is_a_png_of_1200_by_630_small_enough_to_unfurl(self) -> None:
        data = (picture.DIR / picture.site_entry()["file"]).read_bytes()
        assert _png_header(data) == (1200, 630)
        assert picture.size(data) == (1200, 630)
        assert len(data) <= picture.BYTES_MOST

    def test_nothing_but_the_manifest_and_what_it_names_is_in_the_directory(self) -> None:
        """A remade picture replaces the old file rather than leaving it beside."""
        held = {path.name for path in picture.DIR.iterdir()}
        assert held == {picture.MANIFEST.name, *picture.pictures()}

    def test_it_was_photographed_from_the_page_the_code_writes_today(self) -> None:
        assert picture.site_entry()["source"] == picture.source_digest(), (
            "the picture's page changed (a token, the mark, the name or the headline): "
            f"remake the picture with `{REMAKE}`, look at it, and commit the new file")
        assert REMAKE.split("python -m ")[1] in json.loads(
            picture.MANIFEST.read_text(encoding="utf-8"))["note"].replace("`", "")


class TestThePage:
    """What the picture is photographed from."""

    def test_it_draws_only_in_the_light_themes_tokens(self) -> None:
        colours = set(re.findall(r"#[0-9a-fA-F]{6}\b", picture.source()))
        assert colours, "the page names no colour at all"
        assert colours <= set(LIGHT.values()), colours - set(LIGHT.values())

    def test_it_carries_the_mark_the_name_and_the_headline_and_the_sites_faces(self) -> None:
        drawn = picture.source()
        assert shell.mark_path() in drawn
        assert f"<h1>{shell.NAME}</h1>" in drawn
        heading = next(e for e in shell.fonts().values() if e["role"] == "headings")
        assert f'src:url("fonts/{heading["file"]}")' in drawn
        assert f"repeat({picture.FILES},{picture.SQUARE}px)" in drawn

    def test_a_light_piece_is_edged_by_a_ring_round_the_drawing(self) -> None:
        ring = picture.outline()
        assert len(ring) == 18 and all(len(row) == 18 for row in ring)
        for y, row in enumerate(shell.ROOK):
            for x, cell in enumerate(row):
                # The ring never covers the drawing, and a cell beside the drawing is in it.
                if cell == "#":
                    assert ring[y + 1][x + 1] == "."
                    if x == 0 or row[x - 1] != "#":
                        assert ring[y + 1][x] == "#"
        assert sum(row.count("#") for row in ring) > 0
        assert all(row == row[::-1] for row in ring)


def _get(*asked: tuple[str, str]) -> list[tuple[int, dict[str, str], bytes]]:
    app = App(RULESETS, library.RUNS_DIR)
    server = Server(("127.0.0.1", 0), handler_for(app))
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02},
                     daemon=True).start()
    answers = []
    try:
        for method, path in asked:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            try:
                connection.request(method, path)
                response = connection.getresponse()
                answers.append((response.status, dict(response.getheaders()), response.read()))
            finally:
                connection.close()
    finally:
        server.shutdown()
        server.server_close()
    return answers


class TestTheServerAnswersIt:
    def test_it_is_served_as_a_png_cached_for_a_year(self) -> None:
        address = picture.site_path()
        (status, headers, body), (head, head_headers, nothing) = _get(
            ("GET", address), ("HEAD", address))
        assert status == 200
        assert headers["Content-Type"] == "image/png"
        assert headers["Cache-Control"] == IMMUTABLE
        assert body == (picture.DIR / picture.site_entry()["file"]).read_bytes()
        assert _png_header(body) == (1200, 630)
        assert head == 200 and nothing == b""
        assert head_headers["Content-Length"] == str(len(body))

    def test_nothing_else_under_its_prefix(self) -> None:
        asked = [("GET", picture.PREFIX + name) for name in (
            "manifest.json", "../shell.js", "site.png", "nothing.png", "")]
        for (_, path), (status, _, _) in zip(asked, _get(*asked), strict=True):
            assert status == 404, path

    def test_it_is_a_constant_served_to_anybody_and_crawlers_may_fetch_it(self) -> None:
        assert picture.PREFIX in routes.STATIC
        assert picture.site_path().startswith(picture.PREFIX)
        assert not seo.disallowed(picture.site_path())
