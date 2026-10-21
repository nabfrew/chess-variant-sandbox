"""Shareable proven positions (G10 of `docs/plan-growth.md`, §2.5).

Each proven position has an address a crawler can read, `/game/<hash>/proven/<position>`,
named by what the position is (`puzzles.name`), with its own title, description, canonical
and picture; the list, `/game/<hash>/proven`, links every position; the page has a control
that copies the position's address. Proved here over HTTP from a running site, against the
proven test game of `test_site_proven` (a walk, never chess):

- **a position's served HTML** carries its own tags, its picture's address once the picture
  is made, words that say whose move it is and what to find and never the answer, and a
  link back to the list;
- **its picture** is the position itself, the core's replay of the record's prefix, drawn
  the same bytes twice, made once for many reads, and served for a year only while the
  position is offered;
- **an unknown or no-longer-offered position** answers 404 with the game's positions page,
  `noindex`, its canonical the list's; a game with no record, an unknown or a killed game
  is the not-found page;
- **the sitemap** lists the list and every position of a listed game, and nothing of a game
  taken off;
- **the copy-link control** copies the position's canonical address.

The site tests need a real Postgres and skip with a reason without one
(`tests/site_store.py`).
"""

from __future__ import annotations

import dataclasses
import html
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from html import unescape
from pathlib import Path
from typing import Any

import pytest
from test_site_doors import _get, _tag, _view, needs_resvg
from test_site_game_page import SHORT, SHORT_HASH, SITE_URL, Fetched, _fetch
from test_site_links import Site, _load, _running_site
from test_site_proven import (
    AT,
    BOTH,
    BOTH_ID,
    DIGEST,
    ENGINE,
    LOSES,
    LOSES_ID,
    ONE,
    ONE_ID,
    SPEC,
    TAKE,
    _proven_view,
    _record,
    _write,
)

from chess_variant_sandbox import puzzles
from chess_variant_sandbox.site import admin, mail, pages, proven, screening
from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.page import game as drawing
from chess_variant_sandbox.web.page import game_picture, picture, proven_page
from chess_variant_sandbox.web.page.shell import NAME

IDS = (ONE_ID, BOTH_ID, LOSES_ID)
"""The record's positions in the order the site lists them: shortest proved win first."""

_SITEMAP = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """The proven test game in the library with its record, and a game with none."""
    for name in ("rulesets", "runs", "archive"):
        (tmp_path / name).mkdir()
    (tmp_path / "rulesets" / "proven.json").write_text(json.dumps(SPEC), encoding="utf-8")
    (tmp_path / "rulesets" / "short.json").write_text(json.dumps(SHORT), encoding="utf-8")
    _write(tmp_path / "runs", _record([ONE, LOSES, BOTH]))
    return tmp_path


@pytest.fixture
def site(tree: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Site]:
    monkeypatch.setenv(mail.URL_VAR, SITE_URL)
    monkeypatch.setattr(screening, "check", lambda spec: {"passed": True})
    with _running_site(tree) as running:
        _load(running, tree)
        yield running


def _position_path(position: str) -> str:
    return drawing.proven_path(DIGEST, position)


def _sides() -> tuple[str, str]:
    game = library.Game(hash=DIGEST, name="proven test", description="", path=Path(),
                        source="committed", spec=SPEC)
    return library.sides(game)


def _offered(site: Site) -> proven.Record:
    found = proven.load(site.app.runs_dir, DIGEST, SPEC)
    assert found is not None
    return found


def _is_list(page: Fetched) -> bool:
    """Whether a page is the positions' list: every position linked at its own address."""
    view = _view(page)
    return all(f'href="{_position_path(one)}"' in view for one in IDS)


# ----------------------------------------------------------------- a position's page


class TestAPositionsPage:
    @pytest.fixture(autouse=True)
    def _no_pictures(self, site: Site) -> None:
        site.app.pictures.can_draw = False

    def test_a_position_answers_with_its_own_head(self, site: Site) -> None:
        page = _fetch(site.port, _position_path(ONE_ID))
        assert page.status == 200
        found = _offered(site).find(ONE_ID)
        assert found is not None and found.single
        title = "proven test, position 1: find the winning move"
        assert page.title == f"{title} · {NAME}"
        said = proven.position_description(_sides(), found)
        assert unescape(page.meta("description") or "") == said
        assert said.startswith(f"{proven.side_said(_sides(), 0)} to move")
        assert len(said) <= drawing.DESCRIPTION_LIMIT
        address = SITE_URL + _position_path(ONE_ID)
        assert page.canonical == address and page.meta("robots") is None
        assert _tag(page, "og:title") == _tag(page, "twitter:title") == title
        assert _tag(page, "og:description") == said
        assert _tag(page, "og:url") == address
        # With no picture of its own made, the site's stands in: never a broken link.
        assert _tag(page, "og:image") == SITE_URL + picture.site_path()

    def test_the_view_says_whose_move_and_what_to_find_and_never_the_answer(
        self, site: Site
    ) -> None:
        page = _fetch(site.port, _position_path(ONE_ID))
        view = unescape(_view(page))
        side = proven.side_said(_sides(), 0)
        assert f"{side} to move. The computer proved a win from here in 1 turn" in view
        assert "Only one move here is proved to win: find it." in view
        assert f'href="{drawing.proven_path(DIGEST)}"' in view
        assert f'href="{drawing.path(DIGEST)}"' in view
        # The answer is the proved move: nowhere on the page, nor in what it says.
        answer = ENGINE.describe_move(AT, TAKE)
        assert answer not in page.html
        for entry in ONE["scores"]:
            assert entry["text"] not in page.html
        # Nothing counts or names anybody who solved it.
        assert "solved by" not in page.readable.lower()

    def test_each_position_has_its_own_title_description_and_address(
        self, site: Site
    ) -> None:
        pages_ = {one: _fetch(site.port, _position_path(one)) for one in IDS}
        assert all(page.status == 200 for page in pages_.values())
        assert len({page.title for page in pages_.values()}) == len(IDS)
        assert len({page.canonical for page in pages_.values()}) == len(IDS)
        # Two winning moves is no *find the move*: it is to be played out.
        both = pages_[BOTH_ID]
        assert both.title.startswith("proven test, position 2: play the win out")
        assert "find" not in (both.meta("description") or "").lower()

    def test_the_api_names_the_title_the_address_carries(self, site: Site) -> None:
        answer = site.browser().get(f"/api/proven/{DIGEST}/{ONE_ID}")
        assert answer.status == 200
        page = _fetch(site.port, _position_path(ONE_ID))
        assert f"{answer.payload['title']} · {NAME}" == page.title


class TestTheList:
    @pytest.fixture(autouse=True)
    def _no_pictures(self, site: Site) -> None:
        site.app.pictures.can_draw = False

    def test_the_list_has_its_own_head_and_links_every_position(self, site: Site) -> None:
        page = _fetch(site.port, drawing.proven_path(DIGEST))
        assert page.status == 200
        assert page.title == f"Proven positions on proven test · {NAME}"
        assert page.canonical == SITE_URL + drawing.proven_path(DIGEST)
        assert page.meta("robots") is None
        assert len(unescape(page.meta("description") or "")) <= drawing.DESCRIPTION_LIMIT
        assert _is_list(page)
        view = _view(page)
        assert view.index(_position_path(ONE_ID)) < view.index(_position_path(BOTH_ID)) \
            < view.index(_position_path(LOSES_ID))

    def test_the_games_own_page_links_its_positions(self, site: Site) -> None:
        view = _view(_fetch(site.port, drawing.path(DIGEST)))
        assert f'href="{drawing.proven_path(DIGEST)}"' in view
        assert drawing.proven_said(3) in unescape(view)
        assert drawing.proven_path(SHORT_HASH) not in _view(_fetch(site.port,
                                                                   drawing.path(SHORT_HASH)))


class TestAPositionNotOffered:
    @pytest.fixture(autouse=True)
    def _no_pictures(self, site: Site) -> None:
        site.app.pictures.can_draw = False

    def test_an_unknown_position_is_a_404_with_the_list(self, site: Site) -> None:
        page = _fetch(site.port, _position_path("0123456789abcdef"))
        assert page.status == 404
        assert page.meta("robots") == "noindex"
        assert page.canonical == SITE_URL + drawing.proven_path(DIGEST)
        assert _is_list(page)

    def test_a_position_no_longer_offered_is_a_404_with_the_list(
        self, site: Site, tree: Path
    ) -> None:
        assert _fetch(site.port, _position_path(ONE_ID)).status == 200
        # A re-proof that leaves the position out: a record of the other two.
        where = _write(tree / "runs", _record([LOSES, BOTH]))
        where.write_text(where.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        page = _fetch(site.port, _position_path(ONE_ID))
        assert page.status == 404 and page.meta("robots") == "noindex"
        view = _view(page)
        assert f'href="{_position_path(ONE_ID)}"' not in view
        assert f'href="{_position_path(BOTH_ID)}"' in view

    def test_a_game_with_no_record_or_no_game_is_the_not_found_page(
        self, site: Site
    ) -> None:
        lost = pages.not_found()
        for path in (drawing.proven_path(SHORT_HASH),
                     drawing.proven_path(SHORT_HASH, ONE_ID),
                     drawing.proven_path("0123456789abcdef"),
                     drawing.proven_path("0123456789abcdef", ONE_ID)):
            page = _fetch(site.port, path)
            assert page.status == 404 and page.html == lost, path

    def test_a_killed_game_has_no_positions(self, site: Site) -> None:
        admin.kill_game(site.store, DIGEST)
        site.store.commit()
        for path in (drawing.proven_path(DIGEST), _position_path(ONE_ID)):
            page = _fetch(site.port, path)
            assert page.status == 404 and page.html == pages.not_found(), path

    def test_a_game_not_listed_keeps_its_positions_out_of_the_index(
        self, site: Site
    ) -> None:
        with site.store.cursor() as cursor:
            cursor.execute("update games set rested = now() where hash = %s", (DIGEST,))
        site.store.commit()
        for path in (drawing.proven_path(DIGEST), _position_path(ONE_ID)):
            page = _fetch(site.port, path)
            assert page.status == 200 and page.meta("robots") == "noindex", path


# ---------------------------------------------------------------------- the sitemap


def _sitemap(port: int) -> dict[str, str | None]:
    found = _fetch(port, "/sitemap.xml")
    assert found.status == 200
    root = ET.fromstring(found.html.encode("utf-8"))
    rows: dict[str, str | None] = {}
    for url in root.findall(f"{_SITEMAP}url"):
        loc, day = url.find(f"{_SITEMAP}loc"), url.find(f"{_SITEMAP}lastmod")
        assert loc is not None and loc.text
        rows[loc.text] = None if day is None else day.text
    return rows


class TestTheSitemap:
    def test_it_lists_the_list_and_every_position_of_a_listed_game(self, site: Site) -> None:
        rows = _sitemap(site.port)
        wanted = [drawing.proven_path(DIGEST), *(_position_path(one) for one in IDS)]
        for path in wanted:
            assert rows.get(SITE_URL + path, "absent") == "2026-10-18", path
        assert not any("/proven" in loc for loc in rows if SHORT_HASH in loc)
        # Each address the sitemap names is a page that answers 200 and is indexable.
        site.app.pictures.can_draw = False
        for path in wanted:
            page = _fetch(site.port, path)
            assert page.status == 200 and page.meta("robots") is None, path
            assert page.canonical == SITE_URL + path

    def test_a_game_taken_off_takes_its_positions_with_it(self, site: Site) -> None:
        admin.kill_game(site.store, DIGEST)
        site.store.commit()
        site.app.listing.clear()
        assert not any("/proven" in loc for loc in _sitemap(site.port))


# ---------------------------------------------------------------------- the picture


def _shape(width: int, height: int, at: int) -> dict[str, Any]:
    """A still board's payload with one lettered piece on square `at`."""
    board = [0] * (width * height)
    board[at] = 1
    return {"width": width, "height": height, "board": board, "nTypes": 1,
            "sideStride": 1, "pieceIds": ["G"], "icons": []}


class TestTheDrawing:
    def test_a_flipped_board_is_the_half_turn(self) -> None:
        """The second player to move is drawn from their side, as the page draws it: the
        piece on the first square stands where the last square's would."""
        for width, height in ((3, 2), (4, 4)):
            last = width * height - 1
            turned, _ = game_picture.board(_shape(width, height, 0), 0, 0, 10, flipped=True)
            plain, _ = game_picture.board(_shape(width, height, last), 0, 0, 10)
            circle = re.compile(r"<circle [^>]*>")
            assert len(circle.findall(turned)) == 1
            assert circle.findall(turned) == circle.findall(plain), (width, height)
            assert circle.findall(turned) != circle.findall(
                game_picture.board(_shape(width, height, 0), 0, 0, 10)[0])
        # Unflipped, the drawing is the one G6 has always made.
        shape = _shape(3, 2, 0)
        assert game_picture.svg("x", shape, "words") == game_picture.svg(
            "x", shape, ("words",))

    def test_a_positions_drawing_is_its_own_position_from_the_cores_replay(self) -> None:
        position = proven.Position(
            id=puzzles.name(AT), number=1, moves=tuple(ONE["moves"]), side=0,
            ends_in=1, pieces=2, single=True, others_lost=False, scores=(),
        )
        sides = _sides()
        drawn = proven.drawing("proven test", SPEC, position, sides)
        assert drawn == proven.drawing("proven test", SPEC, position, sides)
        game = library.Game(hash=DIGEST, name="proven test", description="", path=Path(),
                            source="committed", spec=SPEC)
        start = library.opening(game)
        assert start is not None
        assert drawn != game_picture.svg("proven test", start, proven.picture_words(
            sides, position))
        for line in proven.picture_words(sides, position):
            assert html.escape(line) in drawn
        assert ENGINE.describe_move(AT, TAKE) not in drawn
        turned = dataclasses.replace(position, side=1)
        assert proven.drawing("proven test", SPEC, turned, sides) != drawn


@needs_resvg
class TestThePositionsPicture:
    @pytest.fixture(autouse=True)
    def _pictures(self, site: Site) -> None:
        site.app.pictures.can_draw = True

    def _made(self, site: Site, position: str) -> str:
        _fetch(site.port, _position_path(position))
        site.app.pictures.settle()
        image = _tag(_fetch(site.port, _position_path(position)), "og:image")
        assert image is not None
        return image

    def test_a_position_names_its_own_picture_once_it_is_made(self, site: Site) -> None:
        image = self._made(site, ONE_ID)
        found = game_picture.PROVEN_ROUTE.match(image[len(SITE_URL):])
        assert found is not None and found.groups()[:2] == (DIGEST, ONE_ID)
        page = _fetch(site.port, _position_path(ONE_ID))
        alt = game_picture.position_alt("proven test", proven.side_said(_sides(), 0))
        assert _tag(page, "og:image:alt") == _tag(page, "twitter:image:alt") == alt
        assert _tag(page, "twitter:image") == image
        status, headers, body = _get(site.port, image[len(SITE_URL):])
        assert status == 200 and headers["content-type"] == "image/png"
        assert "max-age=31536000" in headers["cache-control"]
        assert picture.size(body) == (1200, 630) and len(body) < picture.BYTES_MOST

    def test_the_picture_is_drawn_the_same_bytes_and_once_for_many_reads(
        self, site: Site
    ) -> None:
        made: list[str] = []
        drawn_by = site.app.pictures.make

        def counted(drawing_text: str) -> bytes:
            made.append(drawing_text)
            return drawn_by(drawing_text)

        site.app.pictures.make = counted
        for _ in range(3):
            _fetch(site.port, _position_path(ONE_ID))
            site.browser().get(f"/api/proven/{DIGEST}/{ONE_ID}")
        site.app.pictures.settle()
        image = self._made(site, ONE_ID)
        position_drawings = [one for one in made if "to move and win" in one]
        assert len(position_drawings) == 1
        assert game_picture.png(position_drawings[0]) == game_picture.png(position_drawings[0])
        # Each position is a picture of its own, and none is the game's.
        other = self._made(site, BOTH_ID)
        game_image = _tag(_fetch(site.port, drawing.path(DIGEST)), "og:image")
        assert len({image, other, game_image}) == 3

    def test_the_picture_is_served_only_while_the_position_is_offered(
        self, site: Site, tree: Path
    ) -> None:
        image = self._made(site, ONE_ID)[len(SITE_URL):]
        found = game_picture.PROVEN_ROUTE.match(image)
        assert found is not None
        key = found.group(3)
        assert _get(site.port, image)[0] == 200
        for path in (game_picture.position_path(DIGEST, ONE_ID, "0123456789abcdef"),
                     game_picture.position_path(DIGEST, "0123456789abcdef", key),
                     game_picture.position_path(SHORT_HASH, ONE_ID, key),
                     image.replace(".png", ".svg")):
            assert _get(site.port, path)[0] == 404, path
        where = _write(tree / "runs", _record([LOSES, BOTH]))
        where.write_text(where.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        assert _get(site.port, image)[0] == 404


# --------------------------------------------------------------------- the copy link


_POSITION = (
    "const SPOT = { id: 'abcdef0123456789', number: 1, side: 0, endsIn: 1, single: false,"
    " said: 'The computer proved a win.', preview: { width: 5, height: 5 }, moves: [] };"
    " boardPreview = () => '<svg></svg>';"
)

_CLIPBOARD = """
let CLICK = null; const COPIED = [];
document.addEventListener = (kind, handler) => { if (kind === 'click') CLICK = handler; };
location.origin = '%s';
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: {
  clipboard: { writeText: text => { COPIED.push(text); return Promise.resolve(); } } } });
wireCopyLinks.done = false;
wireCopyLinks();
const press = dataset => CLICK({ target: { closest: selector =>
  selector === '[data-copy-link]' ? { dataset, textContent: 'copy link' } : null } });
press({ copyLink: '%s', copyPosition: '%s' });
press({ copyLink: '%s' });
"""


class TestTheCopyLink:
    def test_the_frame_carries_one_copy_control_naming_the_position(self) -> None:
        from test_web import _in_page

        drawn = html.unescape(_in_page(_proven_view() + _POSITION, "provenFrameHtml(SPOT)"))
        assert drawn.count("data-copy-link=") == 1
        assert f'data-copy-link="{DIGEST}"' in drawn
        assert 'data-copy-position="abcdef0123456789"' in drawn
        assert f">{proven.PAGE_WORDS['copy']}</button>" in drawn
        # In the control bar, beside the view's one primary, never a share prompt.
        bar = drawn[drawn.index('data-frame="controls"'):]
        assert bar.index("data-copy-link") < bar.index("</div>")
        assert "share" not in drawn.lower()

    def test_it_copies_the_positions_canonical_address(self) -> None:
        from test_web import _in_page

        copied, game = _in_page(_CLIPBOARD % (SITE_URL, DIGEST, ONE_ID, DIGEST), "COPIED")
        drawn = proven_page.Drawn(
            hash=DIGEST, title="", description="", path=_position_path(ONE_ID),
            indexable=True, heading="", back="", back_said="", base=SITE_URL)
        assert copied == proven_page.canonical(drawn) == SITE_URL + _position_path(ONE_ID)
        # A game's own control still copies the game's address.
        assert game == SITE_URL + drawing.path(DIGEST)

    def test_an_address_no_longer_offered_says_why_the_list_opens_at_its_first(
        self
    ) -> None:
        from test_web import _in_page

        said = html.unescape(_in_page(_proven_view(), "provenViewHtml(PROVEN.data, true)"))
        assert proven.PAGE_WORDS["gone"] in said
        plain = html.unescape(_in_page(_proven_view(), "provenViewHtml(PROVEN.data, false)"))
        assert proven.PAGE_WORDS["gone"] not in plain
