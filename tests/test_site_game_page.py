"""A real, crawlable page per game (G3 of `docs/plan-growth.md`).

Every game used to live behind `/#/game/<hash>`, and the part after `#` never reaches a
server, so a crawler or a link unfurler saw one untitled page for all of them. Each game now
has its own address, `/game/<hash>`. What is held here is what a reader *without script*
gets over HTTP from a running site, against a library of inline specs (the dull control with
its ply limit moved, and a designer template far from all of them), never chess:

- **the head**: the game's name in `<title>`, a description of at most 160 characters (its
  description's first sentence), a canonical address on the site's own name; `noindex` for a
  game that opens by its link and is not listed (a saved draft, a resting game);
- **the body**: the name, the description's paragraphs, the score with what it rests on
  (current and provisional), and each near game as a link to its own address, all outside
  `<script>`, `<style>` and `<template>`;
- **the doors**: a killed or deleted game and an invented hash are the not-found page's 404,
  as `/api/game/<hash>` is; a near list never names somebody's unlisted draft;
- **the old links**: `/#/game/<hash>` still reaches the game. The server is asked for `/` and
  answers the page; the page's own router, fetched from the same server and run under Node,
  opens the game and hands the address bar over to `/game/<hash>`;
- **the local application** answers the address with its page too, so a reload there works.

The site tests need a real Postgres and skip with a reason without one
(`tests/site_store.py`).
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Iterator
from html import unescape
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

import node_session
import pytest
from test_site_links import (
    DULL,
    FAR,
    SERVE_FAST,
    Site,
    _far,
    _load,
    _near,
    _running_site,
    _saved,
)
from test_web import _runs_at

from chess_variant_sandbox import corpus
from chess_variant_sandbox.site import admin, game_page, mail, screening
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.web.page import game as drawing
from chess_variant_sandbox.web.server import App as LocalApp
from chess_variant_sandbox.web.server import Server, handler_for

SITE_URL = "https://variants.example"

ABOUT = (
    "A small test game where every piece walks forward. It ends when one side has no "
    "pieces left, which comes quickly.\n\nThe second paragraph says how it differs from its "
    "neighbours: only the turn limit."
)


def _described(spec: dict[str, Any], name: str, about: str) -> dict[str, Any]:
    """The spec under its own name and description, neither of which moves its hash."""
    return {**spec, "name": name, "description": about}


SHORT = _described(_near(2, "x"), "Short walk", ABOUT)
MEDIUM = _described(_near(4, "x"), "Medium walk", "Like the short walk, a little longer.")
LONG = _described(_near(6, "x"), "Long walk", "Like the short walk, longer still.")
DISTANT = _described(_far("x"), "Distant game", "Nothing like the walks.")
DRAFT = _described(_near(8, "x"), "Somebody's secret draft", "Not listed anywhere.")

SHORT_HASH, MEDIUM_HASH, LONG_HASH, DISTANT_HASH, DRAFT_HASH = (
    spec_hash(one) for one in (SHORT, MEDIUM, LONG, DISTANT, DRAFT)
)

SCORE = 0.612


def _measure(root: Path, spec: dict[str, Any], index: float, **basis: Any) -> None:
    """One measurement on disk for `spec`, on two seeds, its basis current unless `basis`
    says otherwise (an earlier scoring rules version makes it provisional)."""
    digest = spec_hash(spec)
    out = corpus.measurement_dir(digest, root / "runs")
    out.mkdir(parents=True, exist_ok=True)
    stamped = {**corpus.current_basis(digest, spec=spec).as_dict(), **basis}
    weights = str(stamped["weights_version"])
    (out / "canonical.json").write_text(json.dumps({
        "hash": digest, "name": spec["name"], "index": index, "sd": 0.01, "viable": True,
        "violations": [], "weights_version": weights, "agents": list(corpus.CANONICAL_AGENTS),
        "games": 256, "families": {}, "basis": stamped, "seeds": [0, 1], "totals_used": 2,
        "runs": _runs_at((0, 1), index, 0.01),
    }), encoding="utf-8")
    (out / f"index-{weights}.json").write_text(json.dumps({
        "total": index, "scores": {}, "families": {},
    }), encoding="utf-8")


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """Four library games, the short walk measured; nothing else."""
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


class Fetched:
    """One answer, read the way a crawler reads it: headers, and the text outside script."""

    def __init__(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.status = status
        self.headers = headers
        self.html = body.decode("utf-8")

    @property
    def title(self) -> str:
        found = re.search(r"<title>(.*?)</title>", self.html, re.S)
        return "" if found is None else unescape(found.group(1))

    def meta(self, name: str) -> str | None:
        found = re.search(rf'<meta name="{name}" content="([^"]*)">', self.html)
        return None if found is None else found.group(1)

    @property
    def canonical(self) -> str | None:
        found = re.search(r'<link rel="canonical" href="([^"]*)">', self.html)
        return None if found is None else found.group(1)

    @property
    def readable(self) -> str:
        """The words a reader without script sees: no script, style or template, no tags."""
        text = re.sub(r"<(script|style|template)\b.*?</\1>", " ", self.html, flags=re.S)
        return " ".join(unescape(re.sub(r"<[^>]+>", " ", text)).split())

    def links(self) -> list[tuple[str, str]]:
        """Every link outside script and template, as (address, words)."""
        body = re.sub(r"<(script|template)\b.*?</\1>", " ", self.html, flags=re.S)
        return [(unescape(href), " ".join(unescape(re.sub(r"<[^>]+>", "", words)).split()))
                for href, words in re.findall(r'<a [^>]*href="([^"]+)"[^>]*>(.*?)</a>', body,
                                              re.S)]


def _fetch(port: int, path: str, method: str = "GET") -> Fetched:
    connection = HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        connection.request(method, path)
        found = connection.getresponse()
        return Fetched(found.status, {k.lower(): v for k, v in found.getheaders()},
                       found.read())
    finally:
        connection.close()


# -------------------------------------------------------------------------- the words


class TestTheWords:
    """What the page says, from what it is handed; no store, no server."""

    def test_the_description_is_the_first_sentence_and_fits_a_search_result(self) -> None:
        assert drawing.first_sentence("One. Two.") == "One."
        assert drawing.first_sentence("Worth 0.5 a game, roughly! Then more.") == (
            "Worth 0.5 a game, roughly!")
        long = "word " * 60 + "end."
        cut = drawing.first_sentence(long)
        assert len(cut) <= drawing.DESCRIPTION_LIMIT and cut.endswith("…")
        assert drawing.paragraphs(ABOUT)[1].startswith("The second paragraph")

    def test_a_current_score_says_what_it_rests_on(self) -> None:
        said = drawing.score_line({"index": 0.5, "status": "current", "seeds": [0, 1, 2, 3],
                                   "revision": "v12", "measured_at": "2026-10-12"}, False)
        assert said == ("Score 0.500 out of 1, the average of four runs, under scoring rules "
                        "version 12, measured 12 October 2026.")

    def test_a_provisional_score_is_never_said_as_current(self) -> None:
        said = drawing.score_line({"index": 0.5, "status": "provisional", "seeds": [0],
                                   "revision": "v11", "measured_at": "2026-10-03"}, True)
        assert said.startswith("Score 0.500 out of 1, from one run, provisional:")
        assert "(scoring rules version 11, 3 October 2026)" in said

    def test_no_number_is_said_as_none(self) -> None:
        assert drawing.score_line(None, True).startswith("Out of date")
        assert drawing.score_line(None, False) == "Not measured yet, so it has no score."
        assert drawing.score_line({"index": None, "seeds": [0, 1]}, False).startswith(
            "No score")


# ------------------------------------------------------------- what a crawler reads


class TestWhatACrawlerReads:
    def test_a_library_game_has_its_name_description_and_canonical_in_the_head(
        self, site: Site
    ) -> None:
        page = _fetch(site.port, f"/game/{SHORT_HASH}")

        assert page.status == 200
        assert page.headers["content-type"] == "text/html; charset=utf-8"
        assert page.title == "Short walk: play it online · Chess variant sandbox"
        assert page.meta("description") == (
            "A small test game where every piece walks forward.")
        assert page.canonical == f"{SITE_URL}/game/{SHORT_HASH}"
        assert page.meta("robots") is None
        # The page is the application's, under its policy: no script runs inline.
        assert "script-src 'self'" in page.headers["content-security-policy"]
        assert not re.search(r"<script(?![^>]*\bsrc=)(?![^>]*application/json)[^>]*>",
                             page.html)

    def test_the_body_says_what_the_game_is_and_how_it_scored_without_script(
        self, site: Site
    ) -> None:
        words = _fetch(site.port, f"/game/{SHORT_HASH}").readable

        assert "Short walk" in words
        for block in drawing.paragraphs(ABOUT):
            assert block in words
        assert "5 × 5 board · 1 kind of piece" in words
        # The score with its basis: the run count and the scoring rules' version.
        assert re.search(r"Score 0\.612 out of 1, the average of two runs, under scoring "
                         r"rules version \d+, measured \d+ \w+ \d{4}\.", words), words

    def test_a_provisional_score_is_marked_on_the_page(self, site: Site, tree: Path) -> None:
        _measure(tree, MEDIUM, 0.4, weights_version="v1")

        words = _fetch(site.port, f"/game/{MEDIUM_HASH}").readable

        assert "Score 0.400 out of 1, the average of two runs, provisional:" in words
        assert "scoring rules version 1," in words

    def test_the_near_games_are_links_to_their_own_addresses(self, site: Site) -> None:
        links = dict(_fetch(site.port, f"/game/{SHORT_HASH}").links())

        assert links[f"/game/{MEDIUM_HASH}"] == "Medium walk"
        assert links[f"/game/{LONG_HASH}"] == "Long walk"
        assert links["/game"] == "← all games"
        # A game nothing is near still names its nearest.
        assert any(href.startswith("/game/") and len(href) > len("/game/")
                   for href, _ in _fetch(site.port, f"/game/{DISTANT_HASH}").links())

    def test_a_head_request_answers_as_the_get(self, site: Site) -> None:
        # The first read asks for the game's picture (G6), and the next names it: both
        # reads below come after it is made.
        _fetch(site.port, f"/game/{SHORT_HASH}")
        site.app.pictures.settle()
        got = _fetch(site.port, f"/game/{SHORT_HASH}")
        head = _fetch(site.port, f"/game/{SHORT_HASH}", "HEAD")
        assert head.status == 200 and head.html == ""
        assert head.headers["content-length"] == got.headers["content-length"]


class TestWhoIsListed:
    def test_a_saved_draft_opens_by_its_link_and_says_noindex(self, site: Site) -> None:
        _saved(site.signed_in("ada"), DRAFT)

        page = _fetch(site.port, f"/game/{DRAFT_HASH}")

        assert page.status == 200
        assert page.meta("robots") == "noindex"
        assert "Somebody's secret draft" in page.title
        # And no listed game's page names it.
        for digest in (SHORT_HASH, MEDIUM_HASH, LONG_HASH, DISTANT_HASH):
            near = _fetch(site.port, f"/game/{digest}")
            assert DRAFT_HASH not in near.html and "secret draft" not in near.html

    def test_a_resting_game_keeps_its_page_and_says_noindex(self, site: Site) -> None:
        with site.store.cursor() as cursor:
            cursor.execute("update games set rested = now() where hash = %s", (LONG_HASH,))
        site.store.commit()
        assert _fetch(site.port, f"/game/{LONG_HASH}").meta("robots") == "noindex"
        assert _fetch(site.port, f"/game/{SHORT_HASH}").meta("robots") is None

    def test_a_killed_or_deleted_game_and_an_invented_one_are_not_found(
        self, site: Site
    ) -> None:
        admin.kill_game(site.store, MEDIUM_HASH)
        site.store.commit()
        with site.store.cursor() as cursor:
            cursor.execute("update games set deleted = now(), spec = null where hash = %s",
                           (LONG_HASH,))
        site.store.commit()

        for path in (f"/game/{MEDIUM_HASH}", f"/game/{LONG_HASH}", "/game/0123456789ab",
                     "/game/not-a-hash", f"/game/{SHORT_HASH}/more"):
            page = _fetch(site.port, path)
            assert page.status == 404, path
            assert page.title == "No page here", path
        # As the game's API is.
        assert _fetch(site.port, f"/api/game/{MEDIUM_HASH}").status == 404


# ---------------------------------------------------------------------- the old links


_ROUTER = """
const __el = () => ({ textContent: '', style: {}, dataset: {}, hidden: false,
  addEventListener() {}, append() {}, appendChild() {}, remove() {}, setAttribute() {},
  removeAttribute() {}, querySelector: () => null, querySelectorAll: () => [],
  classList: { add() {}, remove() {}, toggle() {} } });
global.document = { getElementById: () => __el(), querySelector: () => null,
  querySelectorAll: () => [], addEventListener() {}, body: __el(), createElement: __el,
  title: 'as served' };
global.window = { addEventListener() {}, scrollTo() {}, innerWidth: 1200,
  matchMedia: () => ({ matches: false, addEventListener() {} }) };
global.location = { pathname: PATH, hash: HASH, origin: 'https://site.example',
  get href() { return 'https://site.example' + this.pathname + this.hash; } };
const __moved = [];
const __to = (kind, url) => {
  __moved.push(kind + ' ' + url);
  const at = new URL(url, location.href);
  location.pathname = at.pathname; location.hash = at.hash;
};
global.history = { replaceState: (s, t, url) => __to('replace', url),
                   pushState: (s, t, url) => __to('push', url) };
"""

_STUBS = """
const __shown = [];
const __show = (what) => { __shown.push(what); return Promise.resolve(); };
showLanding = () => __show('landing');
showLibrary = () => __show('library');
showGame = (hash) => __show('game ' + hash);
showBoard = (session) => __show('play ' + session.id);
showDesigner = (from) => __show('design ' + from);
api = (path) => Promise.resolve({ id: path.split('/').pop() });
pending = () => {};
VIEWER = null; ADMIN = false; ACCOUNTS = true;
"""


def _route(core: str, path: str, hash_: str, then: str = "") -> dict[str, Any]:
    """Draw `path` + `hash_` with this `core.js`, then run `then`: what was drawn, how the
    address moved, where it ended and the tab's title."""
    head = _ROUTER.replace("PATH", json.dumps(path)).replace("HASH", json.dumps(hash_))
    names = ("showLanding", "showLibrary", "showGame", "showExplorer", "showWhatsNew",
             "showRound", "showDesigner", "showBotBuilder", "showBoard", "showRun",
             "stopPolling")
    declared = "".join(f"var {name} = () => Promise.resolve();\n" for name in names)
    source = (head + declared + core + _STUBS + "render();\n" + then
              + "\nsetTimeout(() => console.log(JSON.stringify({ shown: __shown, moved: __moved,"
              " at: location.pathname + location.hash, title: document.title })), 20);\n")
    done = node_session.run(source)
    assert done.returncode == 0, done.stderr
    said: dict[str, Any] = json.loads(done.stdout.strip().splitlines()[-1])
    return said


class TestTheOldLinks:
    def test_an_old_link_reaches_the_game_and_the_address_bar_names_it(
        self, site: Site
    ) -> None:
        """`/#/game/<hash>`: the server is asked for `/`, and its page's router, fetched
        from the same server, opens the game and replaces the address with the game's own."""
        landing = _fetch(site.port, "/")
        assert landing.status == 200
        core_route = re.search(r'<script src="(/static/page/core\.js)"></script>', landing.html)
        assert core_route is not None
        core = _fetch(site.port, core_route.group(1))
        assert core.status == 200

        said = _route(core.html, "/", f"#/game/{SHORT_HASH}")

        assert said["shown"] == [f"game {SHORT_HASH}"]
        assert said["moved"] == [f"replace /game/{SHORT_HASH}"]
        assert said["at"] == f"/game/{SHORT_HASH}"
        # And that address is one the server answers with the game's own page.
        assert _fetch(site.port, said["at"]).canonical == f"{SITE_URL}/game/{SHORT_HASH}"


class TestTheRouterReadsTheAddress:
    """The page's router on `core.js` as the package serves it: no server needed."""

    CORE = (Path(__file__).resolve().parent.parent / "chess_variant_sandbox" / "web" /
            "static" / "page" / "core.js").read_text(encoding="utf-8")
    GAME = "abcdef012345"

    def test_a_games_own_address_draws_the_game_and_moves_nothing(self) -> None:
        said = _route(self.CORE, f"/game/{self.GAME}", "")
        assert said == {"shown": [f"game {self.GAME}"], "moved": [],
                        "at": f"/game/{self.GAME}", "title": "as served"}

    def test_a_route_in_the_fragment_still_wins_at_a_games_address(self) -> None:
        said = _route(self.CORE, f"/game/{self.GAME}", "#/play/s42")
        assert said["shown"] == ["play s42"]
        # `#/` there is the games, said as `/game`.
        said = _route(self.CORE, f"/game/{self.GAME}", "#/")
        assert said["shown"] == ["library"] and said["at"] == "/game"

    def test_a_way_to_a_game_goes_to_its_address_and_back_redraws(self) -> None:
        said = _route(self.CORE, "/game", "",
                      f"go('#/game/{self.GAME}');"
                      "location.pathname = '/game'; onHistory();")
        assert said["shown"] == ["library", f"game {self.GAME}", "library"]
        assert said["moved"] == [f"push /game/{self.GAME}"]
        assert said["title"] == "Games · Chess variant sandbox"

    def test_a_copied_link_is_the_games_own_address(self) -> None:
        done = node_session.run(
            _ROUTER.replace("PATH", '"/game"').replace("HASH", '""') + self.CORE
            + f"\nconsole.log(gameLink('{self.GAME}'));")
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == (
            f"https://site.example/game/{self.GAME}")


# ------------------------------------------------------------- the local application


class TestTheLocalApplication:
    def test_it_answers_a_games_address_with_its_page(self, tree: Path) -> None:
        app = LocalApp(tree / "rulesets", tree / "runs", archive_dir=tree / "archive")
        server = Server(("127.0.0.1", 0), handler_for(app))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        try:
            page = _fetch(server.server_port, f"/game/{SHORT_HASH}")
            assert page.status == 200
            assert '<div id="view"></div>' in page.html
        finally:
            server.shutdown()
            server.server_close()


# ---------------------------------------------------------------------------- the cost


class TestTheCost:
    def test_a_warm_read_costs_a_small_part_of_a_core_second(self, site: Site) -> None:
        """§2.1: a game's page read costs about 1 CPU-s on the box through the API. Its own
        address must cost under 0.1 there; the box runs this code about three times slower
        than a development machine, so a warm read is held to 0.03 s of this thread here."""
        site.app.game_page(SHORT_HASH)
        spent = []
        for _ in range(5):
            began = time.thread_time()
            assert site.app.game_page(SHORT_HASH) is not None
            spent.append(time.thread_time() - began)
        assert sorted(spent)[2] < 0.03, spent

    def test_each_read_is_counted_by_the_hour_as_a_visitors(self, site: Site) -> None:
        _fetch(site.port, f"/game/{SHORT_HASH}")
        _fetch(site.port, "/game/0123456789ab")
        drained = site.app.usage.drain(final=True)
        statuses = sorted(key[3] for key in drained.hours if key[1] == game_page.ROUTE)
        assert statuses == [200, 404]


def test_the_inline_specs_are_five_games() -> None:
    """Four library games and a draft, none of them `dull` itself."""
    assert len({SHORT_HASH, MEDIUM_HASH, LONG_HASH, DISTANT_HASH, DRAFT_HASH}) == 5
    assert spec_hash(DULL) not in {SHORT_HASH, MEDIUM_HASH, LONG_HASH}
    assert spec_hash(FAR) == DISTANT_HASH
