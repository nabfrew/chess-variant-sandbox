"""The pages' scripts and stylesheets are cacheable, and no markup carries inline style (G12).

`docs/plan-growth.md` G12: the stylesheet out of the page into a versioned file, the
scripts under versioned names with a long cache instead of `no-store`, a second visit to
`/` under 150 KB, and the content policy without `'unsafe-inline'` for style. Held here:

- **A name changes whenever the bytes do** (`web/page/assets.py`): a file's name is the
  digest of what it answers, an edited file gets a new name at once, and its old name is a
  404 rather than the new bytes under it.
- **What a second visit costs**: every file a page names answers under a name cached for a
  year, so a browser that has been here asks again only for the page itself (`no-store`)
  and the data its script reads. The page and whatever else it names that is not cached
  for long come to a few kilobytes, measured compressed as Caddy sends them.
- **No style is inline**: no served page carries `<style>`, no markup a server or a
  script writes carries `style=`, and every `u-` class markup names is a rule
  (`web/page/utilities.py`), so the policy's `style-src 'self'` costs nothing.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import re
import threading
import time
from http.client import HTTPConnection
from pathlib import Path

import page_names
import page_text
import pytest

from chess_variant_sandbox.site import pages as site_pages
from chess_variant_sandbox.viz.wrapped import WRAPPED_FILE
from chess_variant_sandbox.web import library, page
from chess_variant_sandbox.web.page import assets, front, shell, utilities
from chess_variant_sandbox.web.server import IMMUTABLE, App, Server, handler_for

PACKAGE = Path(page.__file__).resolve().parents[2]
RULESETS = PACKAGE.parent / "rulesets"

# The budget for what a second visit to `/` fetches again beside the data the page's
# script reads: the page and anything it names that is not cached for long, compressed.
# The page is about 8 KB of it (G12 measured 8.4 KB gzipped, 27.5 KB raw).
SECOND_VISIT_BUDGET = 30_000

# The plan's figure for the whole of a second visit, the library's data included.
PLAN_BUDGET = 150_000

_NAMED = re.compile(r'<(?:script|link)\b[^>]*?\b(?:src|href)="(/[^"]+)"')

# Every script the pages run, as files: the application's, the replay's shared one and
# the site's.
SCRIPT_FILES = sorted({*page.SCRIPTS.values(), *site_pages.SCRIPTS.values(),
                       *shell.HEAD_SCRIPTS.values(), WRAPPED_FILE})


def _get(app: App, *paths: str) -> list[tuple[int, dict[str, str], bytes]]:
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


@pytest.fixture()
def app(tmp_path: Path) -> App:
    return App(RULESETS, library.RUNS_DIR, browser_dir=tmp_path / "absent")


def _served_pages() -> dict[str, str]:
    """Every HTML page a server draws: the application's (whose drawing a game's and a
    proven position's own addresses share), the doors, the local drawing of the rule page
    and each of the site's."""
    from chess_variant_sandbox.web.page import rules_page

    listed = [front.Listed("0123456789ab", "A <game>")]
    return {
        "/": page.PAGE,
        "/ on the site": front.home("https://site.example"),
        "/game on the site": front.games("", listed),
        "/rules locally": rules_page.local_bytes().decode("utf-8"),
        **page_names.site_pages(),
    }


class TestTheNameIsTheBytes:
    def test_a_name_is_the_digest_of_what_it_answers(self) -> None:
        for route in (*page.SCRIPTS, *site_pages.SCRIPTS, *shell.HEAD_SCRIPTS):
            named = assets.name_of(route)
            body, kind = assets.answer(named)  # type: ignore[misc]
            assert named == assets.versioned(route, hashlib.sha256(body).hexdigest()[:12])
            assert kind == "text/javascript"
        for named in re.findall(r'href="(/static/style/[^"]+)"', "".join(_served_pages().values())):
            body, kind = assets.answer(named)  # type: ignore[misc]
            assert named.split(".")[-2] == hashlib.sha256(body).hexdigest()[:12], named
            assert kind == "text/css; charset=utf-8"

    def test_an_edit_is_a_new_name_and_the_old_one_is_gone(self, tmp_path: Path) -> None:
        """A running server whose file changes names the new bytes on the next page, and
        never answers them under the name the old ones had: a page drawn before the change
        runs the script it was drawn with (from the cache) or none, never a newer one."""
        file = tmp_path / "probe.js"
        file.write_text("const one = 1;\n", encoding="utf-8")
        route = "/static/test-g12/probe.js"
        assets.serve_files({route: file})
        before = assets.name_of(route)
        assert assets.answer(before) == (file.read_bytes(), "text/javascript")
        file.write_text("const one = 2; // edited\n", encoding="utf-8")
        stamp = time.time() + 5
        os.utime(file, (stamp, stamp))
        after = assets.name_of(route)
        assert after != before
        assert assets.answer(after) == (file.read_bytes(), "text/javascript")
        assert assets.answer(before) is None
        assert assets.plain(after) == assets.plain(before) == route

    def test_a_sheet_is_named_by_its_text(self) -> None:
        one = assets.sheet("test-g12-a", ".a { color: var(--ink); }\n")
        two = assets.sheet("test-g12-a", ".a { color: var(--muted); }\n")
        assert one != two and assets.answer(one) is None
        assert assets.answer(two) == (b".a { color: var(--muted); }\n", "text/css; charset=utf-8")

    def test_names_that_are_not_held_are_refused(self) -> None:
        core = assets.name_of(page.SCRIPT_PREFIX + "core.js")
        for asked in (
            core.replace(core.split(".")[-2], "0" * 12),
            page.SCRIPT_PREFIX + "nothing.0123456789ab.js",
            "/static/style/nothing.0123456789ab.css",
            core + "/..",
            core.replace(".js", ".css"),
        ):
            assert assets.answer(asked) is None, asked


class TestASecondVisit:
    def test_every_file_the_page_names_is_cached_for_a_year(self, app: App) -> None:
        [(status, headers, body)] = _get(app, "/")
        assert status == 200 and headers["Cache-Control"] == "no-store"
        named = [path for path in _NAMED.findall(body.decode("utf-8"))
                 if path.startswith(("/static/page/", "/static/style/", "/static/shell"))]
        assert len(named) == len(page.SCRIPTS) + 2  # the scripts, the shell's, the sheet
        for path, (status, headers, _) in zip(named, _get(app, *named), strict=True):
            assert status == 200, path
            assert headers["Cache-Control"] == IMMUTABLE, path

    def test_it_transfers_only_the_page_and_its_data(self, app: App) -> None:
        """What a browser that has been here fetches again: the page, and anything it
        names whose cache does not outlast a day. Each body counted gzipped, as Caddy's
        `encode zstd gzip` sends it (zstd is smaller). The data the page's script reads on
        `/` (`/api/me`, `/api/library` for the landing's boards) is no-store by design and
        is the rest of the plan's 150 KB: measured in a browser against the main
        checkout's library at 120.6 KB gzipped (`reference/site-as-built.md`, *Cacheable
        assets*)."""
        [(_, _, body)] = _get(app, "/")
        named = sorted(set(_NAMED.findall(body.decode("utf-8"))))
        fetched_again = [body]
        for path, (status, headers, got) in zip(named, _get(app, *named), strict=True):
            assert status == 200, path
            cache = headers["Cache-Control"]
            age = re.search(r"max-age=(\d+)", cache)
            if "no-store" in cache or age is None or int(age[1]) < 86400:
                fetched_again.append(got)
        again = sum(len(gzip.compress(part, 6)) for part in fetched_again)
        assert again < SECOND_VISIT_BUDGET, again
        assert again < PLAN_BUDGET

    def test_the_page_no_longer_carries_the_stylesheet(self) -> None:
        sheet = assets.answer(page.STYLESHEET)
        assert sheet is not None and len(sheet[0]) > 100_000
        assert len(page.PAGE.encode("utf-8")) < 40_000


class TestNoStyleIsInline:
    def test_no_served_page_carries_a_style_element_or_attribute(self) -> None:
        for path, html in _served_pages().items():
            assert "<style" not in html, path
            assert re.findall(r"<[^>]*\sstyle\s*=", html) == [], path
            assert html.count('<link rel="stylesheet"') >= 1, path

    def test_no_script_writes_a_style_attribute(self) -> None:
        """A value a script works out goes through `styleClass` (`viz/wrapped.js`), never
        into a `style=` the policy would make inert."""
        for file in SCRIPT_FILES:
            text = file.read_text(encoding="utf-8")
            # An attribute in markup: `style=` straight before its quote or a template value.
            assert re.findall(r"""\sstyle=(?:["']|\\["']|\$\{)""", text) == [], file.name
            assert "setAttribute('style'" not in text and 'setAttribute("style"' not in text
            assert "<style" not in text, file.name

    def test_every_class_named_is_a_rule_and_every_rule_is_named(self) -> None:
        sources = "".join(file.read_text(encoding="utf-8") for file in SCRIPT_FILES)
        sources += "".join(_served_pages().values())
        used = set(re.findall(r"(?<![\w-])u-[a-z0-9_-]*[a-z0-9_]", sources))
        assert used == set(utilities.UTILITIES)
        for name, declaration in utilities.UTILITIES.items():
            assert f".{name} {{ {declaration} !important; }}" in utilities.CSS
            assert "var(--" in declaration or "#" not in declaration, name

    def test_the_sheet_holds_cytoscapes_rule_under_the_id_it_looks_for(self) -> None:
        """Cytoscape writes a `<style>` of its own unless an element with this id exists;
        the policy would refuse it, so the page's sheet carries the id and the rule."""
        assert f'id="{page.CYTOSCAPE_SHEET_ID}"' in page.PAGE
        # First, where Cytoscape puts its own: the map's canvas is the container and is
        # `position: absolute` by a later rule of the page's, which must still win.
        sheet = assets.sheet_text(page.STYLESHEET)
        assert sheet.startswith("  .__________cytoscape_container { position: relative; }")
        assert ".exploreCanvas { position: absolute;" in sheet
        vendor = (PACKAGE / "web" / "static" / "vendor" / "cytoscape-3.34.3.min.js").read_text(
            encoding="utf-8")
        assert f'"{page.CYTOSCAPE_SHEET_ID}"' in vendor

    def test_the_tests_read_the_page_with_its_sheet(self) -> None:
        """`page_text.PAGE` puts the sheet back as a `<style>`, so a test that looks for a
        rule in the page still finds it."""
        assert ".u-mt-6 { margin-top: 6px !important; }" in page_text.PAGE
