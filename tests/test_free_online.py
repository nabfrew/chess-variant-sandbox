"""Every page a search engine reads says the games are free, online, in the browser (Q1068).

The user, 2026-10-30: *"For SEO make sure indexing sees free online in browser on all the
pages."* So every page the site lets a crawler index carries the three words in what an
indexer reads first (the served `<title>` or meta description, which the Open Graph and
Twitter card tags repeat) and in the text a person sees on the page, all in the HTML the
server sends, never only after a script has run:

- **every address the sitemap lists**, fetched from a running site: the fixed pages, every
  listed game's own page and every proven position's, so a page added to the sitemap is
  walked here without this file being touched;
- **every game in the committed library**, drawn as its own address is drawn, so a name or a
  description that pushes the words out of a game's title or search line fails here;
- **the site's constant pages**, drawn without a store, so the words are held even where no
  database is set up and the site tests skip.

Each title stays near a search result's width and each description within
`game.DESCRIPTION_LIMIT`. A page robots keeps out (a sign-in, a token's) is no page an
indexer reads, and is not asked.
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from html import unescape
from pathlib import Path

import pytest
from test_site_game_page import SHORT, SITE_URL, Fetched, _fetch
from test_site_links import Site, _load, _running_site

from chess_variant_sandbox.site import mail, pages, screening, seo
from chess_variant_sandbox.web import landing
from chess_variant_sandbox.web.page import front, news_page, rules_page
from chess_variant_sandbox.web.page import game as drawing

ROOT = Path(__file__).resolve().parent.parent
WORDS = ("free", "online", "browser")
_SITEMAP = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

# A title a little over a search result's width is allowed where the page's own words
# need it (a long game name); none runs on past this.
TITLE_ROOM = 80


def _says(text: str) -> bool:
    return all(re.search(rf"\b{word}\b", text, re.I) for word in WORDS)


def _meta(html: str, kind: str, key: str) -> str:
    found = re.findall(rf'<meta {kind}="{re.escape(key)}" content="([^"]*)">', html)
    assert len(found) == 1, (key, found)
    return unescape(found[0])


def _title(html: str) -> str:
    found = re.search(r"<title>(.*?)</title>", html, re.S)
    assert found is not None
    return unescape(found.group(1))


def _visible(html: str) -> str:
    """The words in the body a person reads: no head, script, style or template."""
    body = html.split("</head>", 1)[1]
    body = re.sub(r"<(script|style|template)\b.*?</\1>", " ", body, flags=re.S)
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", body)).split())


def assert_free_online(html: str, where: str) -> None:
    """The three words in the served title or description, in the preview's tags, and in
    the page's visible text; the title and description of a search result's size."""
    title = _title(html)
    description = _meta(html, "name", "description")
    assert _says(title + " " + description), (where, title, description)
    assert _says(description) or _says(title), (where, title, description)
    for kind, key in (("property", "og:description"), ("name", "twitter:description")):
        assert _meta(html, kind, key) == description, (where, key)
    for kind, key in (("property", "og:title"), ("name", "twitter:title")):
        assert _meta(html, kind, key) in title, (where, key)
    assert len(title) <= TITLE_ROOM, (where, title)
    assert len(description) <= drawing.DESCRIPTION_LIMIT, (where, description)
    assert _says(_visible(html)), (where, "visible text")


# ------------------------------------------------------------------- without a store


class TestThePagesDrawnWithoutAStore:
    def test_the_front_door_and_the_games(self) -> None:
        home = front.home(SITE_URL)
        assert_free_online(home, "/")
        # The words are in the landing's first lines, outside the template, where the
        # hero's own lines stand once the router clones it.
        intro = home.split('<div id="view">', 1)[1].split("</div>\n  <footer", 1)[0]
        assert landing.NO_ACCOUNT in unescape(intro) and _says(landing.NO_ACCOUNT)
        assert front.HOME_TITLE == "Chess variant sandbox: play free online in your browser"
        assert len(front.HOME_TITLE) <= front.TITLE_LIMIT
        assert_free_online(front.games(SITE_URL, ()), "/game")
        assert len(front.GAMES_TITLE) <= front.TITLE_LIMIT

    def test_the_sites_constant_pages(self) -> None:
        drawn = {
            "/credits": pages.credits(), "/privacy": pages.privacy(),
            "/terms": pages.terms_page(), "/badges": pages.badges(),
            "/request-invitation": pages.request_invitation(), "/rules": pages.rules(),
            "/whats-new": pages.whats_new({"entries": []}),
        }
        for where, html in drawn.items():
            assert_free_online(html, where)
        for title in (news_page.TITLE, rules_page.TITLE):
            assert len(title) <= front.TITLE_LIMIT + 2, title

    @pytest.mark.parametrize("path", sorted((ROOT / "rulesets").glob("*.json")),
                             ids=lambda path: path.stem)
    def test_every_library_game_says_it_of_itself(self, path: Path) -> None:
        spec = json.loads(path.read_text(encoding="utf-8"))
        name = str(spec.get("name") or "")
        drawn = drawing.Drawn(
            hash=hashlib.sha256(path.name.encode()).hexdigest()[:16], name=name,
            description=str(spec.get("description") or ""), spec=spec, score=None,
            stale=False, near=(), indexable=True, base=SITE_URL)
        html = drawing.draw(drawn)
        assert_free_online(html, path.stem)
        # About this game: its title and its search line lead with *Play ‹name› free
        # online in your browser*, and so does the page's own line under the board's words.
        said = f"Play {name} free online in your browser"
        assert _title(html).startswith(said)
        assert _meta(html, "name", "description").startswith(said + ".")
        assert said in _visible(html)


# ------------------------------------------------------------------- the running site


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    for name in ("rulesets", "runs", "archive"):
        (tmp_path / name).mkdir()
    (tmp_path / "rulesets" / "short.json").write_text(json.dumps(SHORT), encoding="utf-8")
    # The proven test game, so the sitemap lists a positions' page and a position's.
    from test_site_proven import ONE, SPEC, _record, _write

    (tmp_path / "rulesets" / "proven.json").write_text(json.dumps(SPEC), encoding="utf-8")
    _write(tmp_path / "runs", _record([ONE]))
    return tmp_path


@pytest.fixture
def site(tree: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Site]:
    monkeypatch.setenv(mail.URL_VAR, SITE_URL)
    monkeypatch.setattr(screening, "check", lambda spec: {"passed": True})
    with _running_site(tree) as running:
        _load(running, tree)
        yield running


class TestTheRunningSite:
    def test_every_address_the_sitemap_lists_says_it(self, site: Site) -> None:
        found = _fetch(site.port, seo.SITEMAP_PATH)
        assert found.status == 200
        root = ET.fromstring(found.html.encode("utf-8"))
        paths = [str(loc.text)[len(SITE_URL):] for loc in root.iter(f"{_SITEMAP}loc")]
        assert set(seo.PAGES) <= set(paths)
        games = [path for path in paths if drawing.digest_of(path)]
        proven = [path for path in paths if drawing.proven_of(path)]
        assert len(games) >= 2 and len(proven) >= 2, paths
        for path in paths:
            page: Fetched = _fetch(site.port, path)
            assert page.status == 200, path
            assert_free_online(page.html, path)
