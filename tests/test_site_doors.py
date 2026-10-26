"""The doors for crawlers, and the rest of the head (G4 of `docs/plan-growth.md`).

What a search engine gets over HTTP from a running site, against a library of inline specs
(`test_site_game_page`'s walks, never chess):

- **`robots.txt`** lets in every public page and the files a page needs to draw, and keeps
  out the API, the doors to an account, members' pages, the suggestion box, the admin page
  and the token links; it names the sitemap on the site's own address;
- **`sitemap.xml`** lists the fixed public pages and exactly the games whose own address
  says nothing against indexing: never a saved draft, a resting, killed or deleted game,
  each at the address its page names as canonical;
- **each kind of page** has a description that fits a search result and a canonical on the
  site's address, and no two pages share a title or a description; a page robots keeps out
  says `noindex` itself;
- **the landing's headline and first paragraph** are outside the `<template>`, the first
  words in the view, in the hero's own markup so the router's clone lands them where they
  stood; `/game` lists every listed game for a reader without script, inside `<noscript>`;
- **the link previews** (G5): every page carries Open Graph and Twitter card tags, its title,
  description, type, whole address and the site's picture, which is served at 1200 × 630;
- **the rule language as a page** (G8): `/rules` in the site's shell with every heading of
  `RULES.md` an anchor, linked from the landing page, beside an unchanged `/rules.md`, and
  `llms.txt` naming both and the games list;
- **a picture per game** (G6): a game's page names its own picture once the site has made
  it, the site's until then; `/game` asks for every listed game's; the picture is served
  for a year to anybody who can open the game, and is the 404 for anybody else.

The site tests need a real Postgres and skip with a reason without one
(`tests/site_store.py`).
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from html import unescape
from http.client import HTTPConnection
from pathlib import Path
from urllib.robotparser import RobotFileParser

import page_names
import pytest
from test_site_game_page import (
    DISTANT,
    DISTANT_HASH,
    DRAFT,
    DRAFT_HASH,
    LONG,
    LONG_HASH,
    MEDIUM,
    MEDIUM_HASH,
    SCORE,
    SHORT,
    SHORT_HASH,
    SITE_URL,
    Fetched,
    _fetch,
    _measure,
)
from test_site_links import Site, _load, _running_site, _saved

from chess_variant_sandbox.site import admin, mail, pages, routes, screening, seo
from chess_variant_sandbox.spec import reference
from chess_variant_sandbox.web import landing
from chess_variant_sandbox.web.page import front, game_picture, picture, rules_page
from chess_variant_sandbox.web.page import game as drawing
from chess_variant_sandbox.web.page.shell import NAME

LIBRARY = {SHORT_HASH, MEDIUM_HASH, LONG_HASH, DISTANT_HASH}

# The site's public pages, each answered 200 to anybody.
PUBLIC = ("/", "/game", "/rules", "/badges", "/terms", "/privacy", "/credits",
          "/request-invitation", f"/game/{SHORT_HASH}")

# What a crawler must be kept out of, one address of each kind.
KEPT_OUT = ("/api/library", f"/api/game/{SHORT_HASH}", "/signin", "/signin?then=/c/abc",
            "/join?code=abc", "/admin", "/profile", "/players", "/players?q=ada",
            "/players/ada", "/suggestions", "/mail/off/abcdefghijklmnop", "/c/abc123",
            "/browser/manifest.json")

# What a page needs to draw, which a crawler that renders must be able to fetch.
LET_IN = (*PUBLIC, "/game?from=mail", "/rules.md", "/llms.txt", "/static/page/core.js",
          "/static/shell.js", "/static/site/post.js", "/static/fonts/x.woff2",
          "/static/vendor/x.js", "/favicon.svg", "/static/picture/site-0.png",
          f"/picture/game/{SHORT_HASH}-0123456789abcdef.png")

_SITEMAP = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """`test_site_game_page`'s library: four walks, the short one measured."""
    for name in ("rulesets", "runs", "archive"):
        (tmp_path / name).mkdir()
    for filename, spec in (("short.json", SHORT), ("medium.json", MEDIUM),
                           ("long.json", LONG), ("distant.json", DISTANT)):
        (tmp_path / "rulesets" / filename).write_text(json.dumps(spec), encoding="utf-8")
    _measure(tmp_path, SHORT, SCORE)
    return tmp_path


@pytest.fixture
def site(tree: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Site]:
    monkeypatch.setenv(mail.URL_VAR, SITE_URL)
    # The playability check plays games and decides nothing asked here.
    monkeypatch.setattr(screening, "check", lambda spec: {"passed": True})
    with _running_site(tree) as running:
        _load(running, tree)
        yield running


def _robots(port: int) -> RobotFileParser:
    found = _fetch(port, "/robots.txt")
    assert found.status == 200
    parser = RobotFileParser()
    parser.parse(found.html.splitlines())
    return parser


def _sitemap(port: int) -> list[tuple[str, str | None]]:
    """The sitemap's (address, last changed) rows, in order."""
    found = _fetch(port, "/sitemap.xml")
    assert found.status == 200
    root = ET.fromstring(found.html.encode("utf-8"))
    assert root.tag == f"{_SITEMAP}urlset"
    rows = []
    for url in root.findall(f"{_SITEMAP}url"):
        loc = url.find(f"{_SITEMAP}loc")
        assert loc is not None and loc.text
        day = url.find(f"{_SITEMAP}lastmod")
        rows.append((loc.text, None if day is None else day.text))
    return rows


def _games_in(rows: list[tuple[str, str | None]]) -> set[str]:
    prefix = f"{SITE_URL}/game/"
    return {loc[len(prefix):] for loc, _ in rows if loc.startswith(prefix)}


def _view(page: Fetched) -> str:
    """The markup the server drew into `#view`."""
    found = re.search(r'<div id="view">(.*?)</div>\n  <footer', page.html, re.S)
    assert found is not None
    return found.group(1)


def _restocked(site: Site) -> None:
    """The store moved: the listing is read again rather than in five minutes."""
    site.app.listing.clear()


# ------------------------------------------------------------------------- robots.txt


class TestRobots:
    def test_it_answers_as_plain_text_and_names_the_sitemap(self, site: Site) -> None:
        found = _fetch(site.port, "/robots.txt")
        assert found.status == 200
        assert found.headers["content-type"] == "text/plain; charset=utf-8"
        assert f"Sitemap: {SITE_URL}/sitemap.xml" in found.html.splitlines()
        head = _fetch(site.port, "/robots.txt", "HEAD")
        assert head.status == 200 and head.html == ""
        # And the one table of what the site serves names both files.
        assert {seo.ROBOTS_PATH, seo.SITEMAP_PATH} <= set(routes.STATIC)

    def test_it_lets_in_the_public_pages_and_what_they_draw_with(self, site: Site) -> None:
        robots = _robots(site.port)
        for path in LET_IN:
            assert robots.can_fetch("Googlebot", SITE_URL + path), path

    def test_it_keeps_out_the_api_accounts_members_and_tokens(self, site: Site) -> None:
        robots = _robots(site.port)
        for path in KEPT_OUT:
            assert not robots.can_fetch("Googlebot", SITE_URL + path), path

    def test_a_page_robots_keeps_out_says_noindex_itself(self, site: Site) -> None:
        """A page fetched by a link from elsewhere is not indexed for being linked."""
        for path in ("/signin", "/join?code=abc"):
            page = _fetch(site.port, path)
            assert page.status == 200, path
            assert page.meta("robots") == "noindex", path


# ------------------------------------------------------------------------ sitemap.xml


class TestSitemap:
    def test_it_lists_the_pages_then_every_listed_game(self, site: Site) -> None:
        found = _fetch(site.port, "/sitemap.xml")
        assert found.headers["content-type"] == "application/xml; charset=utf-8"
        rows = _sitemap(site.port)
        locs = [loc for loc, _ in rows]
        assert locs[: len(seo.PAGES)] == [SITE_URL + path for path in seo.PAGES]
        assert len(locs) == len(set(locs))
        assert _games_in(rows) == LIBRARY
        # Each game with the day its page last changed.
        assert all(day and re.fullmatch(r"\d{4}-\d{2}-\d{2}", day)
                   for loc, day in rows if "/game/" in loc)

    def test_no_draft_resting_killed_or_deleted_game_is_listed(self, site: Site) -> None:
        _saved(site.signed_in("ada"), DRAFT)
        with site.store.cursor() as cursor:
            cursor.execute("update games set rested = now() where hash = %s", (LONG_HASH,))
        admin.kill_game(site.store, MEDIUM_HASH)
        site.store.commit()
        _restocked(site)

        listed = _games_in(_sitemap(site.port))

        assert SHORT_HASH in listed
        assert not listed & {DRAFT_HASH, LONG_HASH, MEDIUM_HASH}

    def test_it_lists_exactly_the_games_whose_page_may_be_indexed(self, site: Site) -> None:
        """The sitemap and each game's own `noindex` are one rule (`game_page.indexable`):
        every game the site holds is listed if and only if its page answers and says
        nothing against indexing."""
        _saved(site.signed_in("ada"), DRAFT)
        with site.store.cursor() as cursor:
            cursor.execute("update games set rested = now() where hash = %s", (LONG_HASH,))
        site.store.commit()
        _restocked(site)

        listed = _games_in(_sitemap(site.port))
        held = set(site.app.known()) | {DRAFT_HASH}
        for digest in held:
            page = _fetch(site.port, f"/game/{digest}")
            indexable = page.status == 200 and page.meta("robots") is None
            assert (digest in listed) == indexable, digest
        assert listed <= held

    def test_every_address_is_the_one_its_page_names_as_canonical(self, site: Site) -> None:
        for loc, _ in _sitemap(site.port):
            page = _fetch(site.port, loc[len(SITE_URL):])
            assert page.status == 200, loc
            assert page.canonical == loc, loc
            assert page.meta("robots") is None, loc

    def test_without_a_site_address_it_names_the_one_asked(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A development machine has no site address: the files, which must name whole
        addresses, take the one the request was made to, and a canonical is the path."""
        monkeypatch.delenv(mail.URL_VAR)
        here = f"http://127.0.0.1:{site.port}"
        assert _sitemap(site.port)[0] == (f"{here}/", None)
        assert f"Sitemap: {here}/sitemap.xml" in _fetch(site.port, "/robots.txt").html
        assert _fetch(site.port, "/badges").canonical == "/badges"


# ------------------------------------------------------------------------- the heads


class TestTheHeads:
    def test_each_public_page_has_a_description_and_its_canonical(self, site: Site) -> None:
        for path in PUBLIC:
            page = _fetch(site.port, path)
            assert page.status == 200, path
            said = page.meta("description")
            assert said and len(unescape(said)) <= drawing.DESCRIPTION_LIMIT, (path, said)
            assert page.canonical == SITE_URL + path, path
            assert page.meta("robots") is None, path
            assert len(re.findall(r'<meta name="description"', page.html)) == 1, path
            assert page.html.index('<meta name="description"') < page.html.index("</head>")

    def test_no_two_pages_share_a_title_or_a_description(self, site: Site) -> None:
        seen = [_fetch(site.port, path) for path in (*PUBLIC, "/signin", "/join",
                                                       f"/game/{MEDIUM_HASH}", "/nowhere")]
        titles = [page.title for page in seen]
        descriptions = [page.meta("description") for page in seen]
        assert len(set(titles)) == len(titles), titles
        assert len(set(descriptions)) == len(descriptions), descriptions

    def test_every_page_the_site_draws_says_what_it_is(self) -> None:
        """Every page in `site.pages`, the members' own among them, without a server: a
        description that fits, never one another page has."""
        drawn = page_names.site_pages()
        said = {}
        for path, html in drawn.items():
            found = re.search(r'<meta name="description" content="([^"]*)">', html)
            assert found, path
            said[path] = unescape(found.group(1))
            assert len(said[path]) <= drawing.DESCRIPTION_LIMIT, path
        assert len(set(said.values())) == len(said), said

    def test_the_not_found_page_is_never_indexed_and_names_no_address(
        self, site: Site
    ) -> None:
        page = _fetch(site.port, "/nowhere")
        assert page.status == 404
        assert page.meta("robots") == "noindex"
        assert page.canonical is None
        assert page.meta("description") == pages.NOT_FOUND_SAYS


# ------------------------------------------------------------------ the front doors


class TestTheFrontDoors:
    def test_the_landings_first_words_are_outside_the_template(self, site: Site) -> None:
        page = _fetch(site.port, "/")
        view = _view(page)
        words = " ".join(unescape(re.sub(r"<[^>]+>", " ", view)).split())
        # The first readable words in the view are what the site is.
        # LP4: the question with the leading story's ending (none holding: the first
        # idea's), then the lead.
        said = " ".join(unescape(re.sub(r"<[^>]+>", " ", landing.drawn_intro())).split())
        assert words.startswith(said), words
        assert landing.QUESTION in page.readable and landing.LEAD in page.readable
        assert page.title == front.HOME_TITLE
        assert unescape(page.meta("description")) == front.HOME_DESCRIPTION

    def test_the_drawn_words_are_the_heros_own_markup(self) -> None:
        """The router clones the template over the drawn words, so the headline lands where
        it already stood: what is drawn is the start of the hero's markup, word for word."""
        drawn = landing.drawn_intro()
        html = landing.current_html()
        # Up to the question's first ending, and then the lead and the two ways in.
        start = drawn[drawn.index('<section class="landingHero heroB">'):drawn.index("</span>") + 7]
        assert start in html
        assert html.index(start) == html.index('<section class="landingHero heroB">')
        rest = drawn[drawn.index('<p class="landingLead">'):drawn.index("</a></div>") + 10]
        assert rest in html

    def test_no_link_points_at_a_missing_anchor(self, site: Site) -> None:
        """LP3: the score's sections left the landing for `#/score`. Every fragment a page
        of the site links (`#x`, `/#x`, `/rules#x`) names an id its target page holds, the
        single page's templates included, and every route a link names (`#/score`) is one
        the router draws."""
        pages_ = {path: _fetch(site.port, path).html for path in ("/", "/game", "/rules")}
        ids = {path: set(re.findall(r'\bid="([^"]+)"', html)) for path, html in pages_.items()}
        ids["/game"] |= ids["/"]
        core = (ROOT / "chess_variant_sandbox/web/static/page/core.js").read_text(
            encoding="utf-8")
        for path, html in pages_.items():
            for href in re.findall(r'\bhref="([^"]*#[^"]*)"', html):
                where, fragment = href.split("#", 1)
                if not fragment:
                    continue
                if fragment.startswith("/"):
                    route = fragment.strip("/").split("/")[0]
                    assert f"parts[0] === '{route}'" in core, (path, href)
                    continue
                target = where or path
                assert fragment in ids[target], (path, href)
        for anchor in landing.SCORE_ANCHORS:
            assert pages_["/"].count(f'id="{anchor}"') == 1, anchor
        assert f'<a href="{landing.SCORE_ROUTE}">' in pages_["/"]
        assert '<a href="/#/score">the score</a>' in pages_["/rules"]

    def test_the_games_are_listed_for_a_reader_without_script(self, site: Site) -> None:
        page = _fetch(site.port, "/game")
        assert page.title == front.GAMES_TITLE
        view = _view(page)
        # Inside `<noscript>`: never drawn where the router runs.
        assert view.startswith("<noscript>") and view.endswith("</noscript>")
        links = dict(page.links())
        for digest in LIBRARY:
            assert f"/game/{digest}" in links, digest
        assert links[f"/game/{SHORT_HASH}"] == "Short walk"

    def test_a_listed_game_leaves_the_games_list_with_its_listing(self, site: Site) -> None:
        admin.kill_game(site.store, MEDIUM_HASH)
        site.store.commit()
        _restocked(site)
        assert f"/game/{MEDIUM_HASH}" not in dict(_fetch(site.port, "/game").links())

    def test_a_head_request_answers_as_the_get(self, site: Site) -> None:
        for path in ("/", "/game", "/sitemap.xml"):
            got = _fetch(site.port, path)
            head = _fetch(site.port, path, "HEAD")
            assert head.status == 200 and head.html == "", path
            assert head.headers["content-length"] == got.headers["content-length"], path



# ------------------------------------------------------------- the link previews (G5)


def _tag(page: Fetched, key: str) -> str | None:
    """One Open Graph (`property`) or Twitter card (`name`) tag's value, unescaped; the tag
    must be in the head, and there once."""
    kind = "property" if key.startswith("og:") else "name"
    found = re.findall(rf'<meta {kind}="{re.escape(key)}" content="([^"]*)">', page.html)
    assert len(found) <= 1, (key, found)
    if not found:
        return None
    assert page.html.index(f'<meta {kind}="{key}"') < page.html.index("</head>"), key
    return unescape(found[0])


def _picture_address() -> str:
    return SITE_URL + picture.site_path()


class TestThePreviews:
    """G5 of `docs/plan-growth.md`: a link pasted into a chat program or a post unfurls
    into the page's title, description and the site's picture, from the tags in the head.

    With no game's own picture made (G6's are `TestTheGamePictures`'), so that every page
    here names the site's."""

    @pytest.fixture(autouse=True)
    def _no_game_pictures(self, site: Site) -> None:
        site.app.pictures.can_draw = False

    def test_every_public_page_carries_the_open_graph_and_twitter_tags(
        self, site: Site
    ) -> None:
        for path in PUBLIC:
            page = _fetch(site.port, path)
            assert page.status == 200, path
            said = page.meta("description")
            assert said, path
            # Open Graph: the title, the description, the type, the site and a whole
            # address for both the page (its canonical) and the picture.
            assert _tag(page, "og:site_name") == NAME, path
            assert _tag(page, "og:type") == front.PREVIEW_TYPE == "website", path
            assert _tag(page, "og:title") == page.title.removesuffix(f" · {NAME}"), path
            assert _tag(page, "og:description") == unescape(said), path
            assert _tag(page, "og:url") == page.canonical == SITE_URL + path, path
            assert _tag(page, "og:image") == _picture_address(), path
            assert _tag(page, "og:image:type") == "image/png", path
            assert (_tag(page, "og:image:width"), _tag(page, "og:image:height")) == (
                "1200", "630"), path
            assert _tag(page, "og:image:alt") == picture.ALT, path
            # Twitter's card says the same, with the picture across the top.
            assert _tag(page, "twitter:card") == "summary_large_image", path
            assert _tag(page, "twitter:title") == _tag(page, "og:title"), path
            assert _tag(page, "twitter:description") == unescape(said), path
            assert _tag(page, "twitter:image") == _picture_address(), path

    def test_each_kind_of_page_previews_under_its_own_title(self, site: Site) -> None:
        titles = {path: _tag(_fetch(site.port, path), "og:title") for path in PUBLIC}
        assert titles["/"] == NAME
        assert titles["/game"] == "Games"
        # A game's own page names the game, never the site alone.
        assert titles[f"/game/{SHORT_HASH}"] == "Short walk: play it online"
        assert len(set(titles.values())) == len(titles), titles

    def test_a_games_page_points_at_the_sites_picture_until_its_own_is_made(
        self, site: Site
    ) -> None:
        for digest in (SHORT_HASH, MEDIUM_HASH):
            page = _fetch(site.port, f"/game/{digest}")
            assert _tag(page, "og:image") == _picture_address(), digest
            assert _tag(page, "og:description") == unescape(page.meta("description") or "")
        # One place says which picture a game's page shows: the site's without its own.
        drawn = drawing.Drawn(hash=SHORT_HASH, name="x", description="", spec={},
                              score=None, stale=False, near=(), indexable=True,
                              base=SITE_URL)
        assert drawing.picture(drawn) == _picture_address()
        assert drawing.picture(drawn) in drawing.head(drawn)

    def test_the_picture_each_page_names_is_served_at_1200_by_630(self, site: Site) -> None:
        image = _tag(_fetch(site.port, f"/game/{SHORT_HASH}"), "og:image")
        assert image and image.startswith(SITE_URL)
        connection = HTTPConnection("127.0.0.1", site.port, timeout=60)
        try:
            connection.request("GET", image[len(SITE_URL):])
            found = connection.getresponse()
            body = found.read()
            headers = {k.lower(): v for k, v in found.getheaders()}
        finally:
            connection.close()
        assert found.status == 200
        assert headers["content-type"] == "image/png"
        assert "max-age=31536000" in headers["cache-control"]
        assert picture.size(body) == (1200, 630)
        assert len(body) == picture.site_entry()["bytes"]

    def test_a_page_kept_from_the_index_still_unfurls(self, site: Site) -> None:
        """A challenge link is what a member pastes into a chat: no canonical, so no
        `og:url` (the unfurler keeps the address it was given), but the rest is there."""
        _saved(site.signed_in("ada"), DRAFT)
        for path in ("/c/abc123", f"/game/{DRAFT_HASH}", "/signin"):
            page = _fetch(site.port, path)
            assert page.status == 200, path
            assert page.meta("robots") == "noindex", path
            assert _tag(page, "og:title") and _tag(page, "og:description"), path
            assert _tag(page, "og:image") == _picture_address(), path
            assert _tag(page, "og:url") == page.canonical, path

    def test_without_a_site_address_no_tag_names_a_part_address(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A development machine has only paths, and a preview's address and picture must
        be whole: they are left out rather than written as paths."""
        monkeypatch.delenv(mail.URL_VAR)
        for path in ("/", "/badges"):
            page = _fetch(site.port, path)
            assert _tag(page, "og:title") and _tag(page, "twitter:card"), path
            for key in ("og:url", "og:image", "twitter:image"):
                assert _tag(page, key) is None, (path, key)


def _get(port: int, path: str) -> tuple[int, dict[str, str], bytes]:
    connection = HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        connection.request("GET", path)
        found = connection.getresponse()
        return found.status, {k.lower(): v for k, v in found.getheaders()}, found.read()
    finally:
        connection.close()


needs_resvg = pytest.mark.skipif(
    not game_picture.available(), reason="resvg-py is the site extra's: uv sync --extra site")


@needs_resvg
class TestTheGamePictures:
    """G6: a game's page names a picture of its own starting position, drawn on the site
    from the still board, once that picture is made; the site's picture until then."""

    @pytest.fixture(autouse=True)
    def _game_pictures(self, site: Site) -> None:
        """Pictures on, which the suite leaves off elsewhere (`conftest.py`)."""
        site.app.pictures.can_draw = True

    def test_a_games_page_names_its_own_picture_once_it_is_made(self, site: Site) -> None:
        first = _fetch(site.port, f"/game/{SHORT_HASH}")
        assert _tag(first, "og:image") == _picture_address()
        site.app.pictures.settle()

        page = _fetch(site.port, f"/game/{SHORT_HASH}")
        image = _tag(page, "og:image")
        assert image and image.startswith(SITE_URL + game_picture.PREFIX)
        found = game_picture.ROUTE.match(image[len(SITE_URL):])
        assert found is not None and found.group(1) == SHORT_HASH
        assert _tag(page, "twitter:image") == image
        assert _tag(page, "og:image:alt") == _tag(page, "twitter:image:alt") \
            == game_picture.alt("Short walk")
        assert (_tag(page, "og:image:width"), _tag(page, "og:image:height")) == ("1200", "630")

        status, headers, body = _get(site.port, image[len(SITE_URL):])
        assert status == 200
        assert headers["content-type"] == "image/png"
        assert "max-age=31536000" in headers["cache-control"]
        assert picture.size(body) == (1200, 630)
        assert len(body) < picture.BYTES_MOST

    def test_each_game_has_a_picture_of_its_own(self, site: Site) -> None:
        for digest in (SHORT_HASH, MEDIUM_HASH):
            _fetch(site.port, f"/game/{digest}")
        site.app.pictures.settle()
        images = {digest: _tag(_fetch(site.port, f"/game/{digest}"), "og:image")
                  for digest in (SHORT_HASH, MEDIUM_HASH)}
        assert len(set(images.values())) == 2
        bodies = {_get(site.port, str(image)[len(SITE_URL):])[2] for image in images.values()}
        assert len(bodies) == 2

    def test_the_games_list_asks_for_every_listed_games_picture(self, site: Site) -> None:
        assert _fetch(site.port, "/game").status == 200
        site.app.pictures.settle()
        assert site.app.pictures.held() == len(LIBRARY)
        for digest in LIBRARY:
            image = _tag(_fetch(site.port, f"/game/{digest}"), "og:image")
            assert image and image.startswith(SITE_URL + game_picture.PREFIX), digest

    def test_only_a_made_picture_of_a_game_a_visitor_can_open_is_served(
        self, site: Site
    ) -> None:
        _fetch(site.port, "/game")
        site.app.pictures.settle()
        images = {digest: str(_tag(_fetch(site.port, f"/game/{digest}"), "og:image"))
                  [len(SITE_URL):] for digest in (SHORT_HASH, MEDIUM_HASH, LONG_HASH)}
        assert all(_get(site.port, path)[0] == 200 for path in images.values())

        admin.kill_game(site.store, MEDIUM_HASH)
        site.store.commit()
        with site.store.cursor() as cursor:
            cursor.execute("update games set deleted = now(), spec = null where hash = %s",
                           (LONG_HASH,))
        site.store.commit()
        named = game_picture.ROUTE.match(images[SHORT_HASH])
        assert named is not None
        key = named.group(2)
        for path in (images[MEDIUM_HASH], images[LONG_HASH],
                     game_picture.path(SHORT_HASH, "0123456789abcdef"),
                     game_picture.path("0123456789abcdef", key),
                     images[SHORT_HASH] + "x", images[SHORT_HASH].replace(".png", ".svg")):
            assert _get(site.port, path)[0] == 404, path
        assert _get(site.port, images[SHORT_HASH])[0] == 200

    def test_a_picture_is_drawn_once_for_many_reads(self, site: Site) -> None:
        made: list[str] = []
        drawn_by = site.app.pictures.make

        def counted(drawing_text: str) -> bytes:
            made.append(drawing_text)
            return drawn_by(drawing_text)

        site.app.pictures.make = counted
        for _ in range(4):
            _fetch(site.port, f"/game/{SHORT_HASH}")
            _fetch(site.port, "/game")
        site.app.pictures.settle()
        for _ in range(3):
            _fetch(site.port, f"/game/{SHORT_HASH}")
        site.app.pictures.settle()
        assert len(made) == len(LIBRARY)


# ------------------------------------------------------------- the rule language (G8)


ROOT = Path(__file__).resolve().parent.parent


class TestTheRuleLanguage:
    def test_rules_is_the_page_in_the_sites_shell_with_every_heading_an_anchor(
        self, site: Site
    ) -> None:
        from test_rules_page import github_slug, headings_of

        page = _fetch(site.port, rules_page.PATH)
        assert page.status == 200
        assert page.headers["content-type"] == "text/html; charset=utf-8"
        assert page.title == rules_page.TITLE
        assert unescape(page.meta("description") or "") == rules_page.DESCRIPTION
        assert page.canonical == SITE_URL + rules_page.PATH
        # The site's own footer, not the local application's.
        assert '<a href="/credits">credits</a>' in page.html
        taken: dict[str, int] = {}
        for level, words in headings_of((ROOT / "RULES.md").read_text(encoding="utf-8")):
            ident = github_slug(words, taken)
            assert f'<h{level} id="{ident}">' in page.html, ident
        # The same drawing of the reference as the local application's.
        assert rules_page.body() in page.html

    def test_rules_md_is_unchanged(self, site: Site) -> None:
        found = _fetch(site.port, reference.PUBLISHED)
        assert found.status == 200
        assert found.html.encode("utf-8") == (ROOT / "RULES.md").read_bytes()
        assert found.headers["content-type"] == "text/markdown; charset=utf-8"
        assert found.headers["cache-control"] == "public, max-age=300"

    def test_the_landing_and_the_sitemap_name_it(self, site: Site) -> None:
        assert f'href="{rules_page.PATH}"' in _fetch(site.port, "/").html
        assert (SITE_URL + rules_page.PATH, None) in _sitemap(site.port)
        assert rules_page.PATH in seo.PAGES

    def test_llms_txt_names_both_and_the_games_list(self, site: Site) -> None:
        found = _fetch(site.port, seo.LLMS_PATH)
        assert found.status == 200
        assert found.headers["content-type"] == "text/plain; charset=utf-8"
        text = found.html
        lines = text.splitlines()
        assert lines[0] == f"# {NAME}"
        assert lines[2] == f"> {seo.LLMS_SUMMARY}"
        links = dict((url, words) for words, url in
                     re.findall(r"^- \[(.+?)\]\((\S+?)\)", text, re.M))
        assert links[SITE_URL + reference.PUBLISHED] == "The rule language, as Markdown"
        assert links[SITE_URL + reference.PAGE] == "The rule language, as a page"
        assert links[SITE_URL + front.GAMES_PATH] == "The games"
        # Every game the sitemap lists, by its name, and nothing else.
        games = {url[len(SITE_URL + "/game/"):]: words for url, words in links.items()
                 if url.startswith(SITE_URL + "/game/")}
        assert set(games) == _games_in(_sitemap(site.port)) == LIBRARY
        assert games[SHORT_HASH] == "Short walk"
        assert f"({SITE_URL}{seo.DESIGNER})" in text
        assert text.endswith("\n") and "\r" not in text

    def test_a_game_name_cannot_break_out_of_its_link(self) -> None:
        said = seo.llms("https://x", [front.Listed("ab" * 6, "a ]bad[ name\nhere")])
        assert said.splitlines()[-1] == r"- [a \]bad\[ name here](https://x/game/abababababab)"
