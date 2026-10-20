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
  stood; `/game` lists every listed game for a reader without script, inside `<noscript>`.

The site tests need a real Postgres and skip with a reason without one
(`tests/site_store.py`).
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from html import unescape
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
from chess_variant_sandbox.web import landing
from chess_variant_sandbox.web.page import front
from chess_variant_sandbox.web.page import game as drawing

LIBRARY = {SHORT_HASH, MEDIUM_HASH, LONG_HASH, DISTANT_HASH}

# The site's public pages, each answered 200 to anybody.
PUBLIC = ("/", "/game", "/badges", "/terms", "/privacy", "/credits", "/request-invitation",
          f"/game/{SHORT_HASH}")

# What a crawler must be kept out of, one address of each kind.
KEPT_OUT = ("/api/library", f"/api/game/{SHORT_HASH}", "/signin", "/signin?then=/c/abc",
            "/join?code=abc", "/admin", "/profile", "/players", "/players?q=ada",
            "/players/ada", "/suggestions", "/mail/off/abcdefghijklmnop", "/c/abc123",
            "/browser/manifest.json")

# What a page needs to draw, which a crawler that renders must be able to fetch.
LET_IN = (*PUBLIC, "/game?from=mail", "/rules.md", "/static/page/core.js",
          "/static/shell.js", "/static/site/post.js", "/static/fonts/x.woff2",
          "/static/vendor/x.js", "/favicon.svg")

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
        assert words.startswith(f"{landing.HEADLINE} {landing.LEAD}"), words
        assert landing.HEADLINE in page.readable and landing.LEAD in page.readable
        assert page.title == front.HOME_TITLE
        assert page.meta("description") == front.HOME_DESCRIPTION

    def test_the_drawn_words_are_the_heros_own_markup(self) -> None:
        """The router clones the template over the drawn words, so the headline lands where
        it already stood: what is drawn is the start of the hero's markup, word for word."""
        drawn = landing.drawn_intro()
        # Up to the two ways in every reader is offered; the template adds the reader's own.
        start = drawn[drawn.index('<section class="landingHero">'):drawn.index("</a></div>") + 4]
        assert start in landing.HTML
        assert landing.HTML.index(start) == landing.HTML.index('<section class="landingHero">')

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

