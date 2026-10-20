"""What the site names as near a game, and who may build from it. G7, then M5.

The G7 row of `docs/archive/plan-game-graph.md` put every pair of the store's games in a
`links` table, ranked by `spec/relate.py`'s count of rules and narrowed per person, because
a draft could be a neighbour and an unlisted draft's existence is its owner's business.
M5 of `docs/plan-map.md` gave *near* one meaning everywhere: the map's likeness, over the
map's own games (`web/graphing.near`), which are the same for everybody. So a game's page
names the map's games most like it, a draft's page ranks the map's games against it, no
draft is named on anybody's page, and `0038_links_retired.sql` dropped the table.

What stays G7's is the line between *openable* and *findable* (`site/finding.py`): an
unlisted draft opens by its link, and nobody but its owner (or an admin) may build the
game between it and a library game. It is asked over HTTP against a real store, of a
signed-out visitor, another member, the owner and an admin, because *a person cannot see
this* is a property of the answer a browser gets and not of a function's return value.

**The library here is four specs in a `tmp_path` of its own**, three of them one rule
apart and one far from all of them, without a committed ruleset being involved. The last
class asks it of real games: that the site, the local application and the map name the
same games in the same order, over eighteen committed and archived rulesets, named.

Everything needs a real Postgres and skips with a reason without one; `tests/test_site.py`
has the recipe for a throwaway cluster.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import threading
from collections.abc import Iterator
from http.client import HTTPConnection
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any

import pytest
import site_store
from retired_specs import REBUILT

from chess_variant_sandbox.site import (
    admin,
    drafts,
    finding,
    identity,
    migrate,
    screening,
)
from chess_variant_sandbox.site import load as loader
from chess_variant_sandbox.site import pages as site_pages
from chess_variant_sandbox.site import server as site_server
from chess_variant_sandbox.site.db import connect
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.spec import relate, spec_hash
from chess_variant_sandbox.web import designing, families, library
from chess_variant_sandbox.web.server import REQUEST_HEADER, REQUEST_VALUE, Server
from chess_variant_sandbox.web.server import App as LocalApp

# `serve_forever` looks for its stop flag every half second by default, so each test
# that stopped a server waited out that half second in `shutdown()`. 20 ms stops it at once.
SERVE_FAST = {"poll_interval": 0.02}

TEST_URL_ENV = "CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL"
PASSWORD = "a-long-enough-one"

DULL = json.loads((library.RULESET_DIR / "dull.json").read_text(encoding="utf-8"))


def _near(plies: int, name: str) -> dict[str, Any]:
    """`dull` with its ply limit moved: a different game, one rule from its siblings."""
    spec = json.loads(json.dumps(DULL))
    spec["rules"]["max_plies"] = plies
    spec["name"] = name
    return spec


def _far(name: str = "far") -> dict[str, Any]:
    """A designer template, which is several rules from anything `dull` is."""
    spec = json.loads(json.dumps(designing.templates()[0]["spec"]))
    spec["name"] = name
    return spec


LEFT = _near(2, "left")
MIDDLE = _near(4, "middle")
RIGHT = _near(6, "right")
FAR = _far()
DRAFT = _near(8, "a draft somebody kept")

LEFT_HASH = spec_hash(LEFT)
MIDDLE_HASH = spec_hash(MIDDLE)
RIGHT_HASH = spec_hash(RIGHT)
FAR_HASH = spec_hash(FAR)
DRAFT_HASH = spec_hash(DRAFT)

LIBRARY = {LEFT_HASH: LEFT, MIDDLE_HASH: MIDDLE, RIGHT_HASH: RIGHT, FAR_HASH: FAR}


# ------------------------------------------------------------------------ the harness


class Answer:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.body = body

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", "replace")

    @property
    def payload(self) -> Any:
        return json.loads(self.text)


class Browser:
    """One person's browser: a cookie jar and two verbs, over a real socket."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.cookie: str | None = None

    def get(self, path: str) -> Answer:
        return self._ask("GET", path, None)

    def post(self, path: str, payload: dict[str, Any] | None = None) -> Answer:
        return self._ask("POST", path, payload or {})

    def _ask(self, method: str, path: str, payload: dict[str, Any] | None) -> Answer:
        headers: dict[str, str] = {}
        if self.cookie is not None:
            headers["Cookie"] = f"{site_pages.COOKIE}={self.cookie}"
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers[REQUEST_HEADER] = REQUEST_VALUE
        connection = HTTPConnection("127.0.0.1", self.port, timeout=60)
        try:
            connection.request(method, path, body=body, headers=headers)
            found = connection.getresponse()
            answer = Answer(found.status, found.read())
            self._keep(found.headers.get("Set-Cookie", ""))
        finally:
            connection.close()
        return answer

    def _keep(self, header: str) -> None:
        if not header:
            return
        jar = SimpleCookie()
        jar.load(header)
        morsel = jar.get(site_pages.COOKIE)
        if morsel is None:
            return
        self.cookie = None if str(morsel["max-age"]) == "0" else morsel.value


class Site:
    """A running site over a library of four specs, and the store under it."""

    def __init__(self, app: SiteApp, store: Any, port: int) -> None:
        self.app = app
        self.store = store
        self.port = port

    def browser(self) -> Browser:
        return Browser(self.port)

    def person(self, handle: str, role: str = admin.MEMBER) -> int:
        with self.store.cursor() as cursor:
            cursor.execute(
                "insert into users (handle, email, password, role)"
                " values (%s, %s, %s, %s) returning id",
                (handle, f"{handle}@example.invalid",
                 identity.hash_password(PASSWORD), role),
            )
            found = cursor.fetchone()
        self.store.commit()
        return int(found[0])

    def signed_in(self, handle: str, role: str = admin.MEMBER) -> Browser:
        self.person(handle, role)
        browser = self.browser()
        answer = browser.post("/api/signin", {"handle": handle, "password": PASSWORD})
        assert answer.status == 200, answer.text
        return browser


@contextlib.contextmanager
def _running_site(root: Path) -> Iterator[Site]:
    """The site on a port of its own, over `root`'s library and the session's store."""
    url, name = site_store.emptied("no store to run the site on")

    def open_one() -> Any:
        connection = connect(url)
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {name}")
        return connection

    store = open_one()
    # Committed, so the `search_path` survives a test's rollback. `migrate.run` used to
    # commit it in passing, when every test migrated a schema of its own.
    store.commit()
    app = SiteApp(
        root / "rulesets",
        root / "runs",
        archive_dir=root / "archive",
        connect=open_one,
    )
    server = Server(("127.0.0.1", 0), site_server.site_handler_for(app))
    threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
    try:
        yield Site(app, store, server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        store.close()


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A checkout holding the four specs and nothing measured."""
    for name in ("rulesets", "runs", "archive"):
        (tmp_path / name).mkdir()
    for filename, spec in (
        ("left.json", LEFT), ("middle.json", MIDDLE),
        ("right.json", RIGHT), ("far.json", FAR),
    ):
        (tmp_path / "rulesets" / filename).write_text(json.dumps(spec), encoding="utf-8")
    return tmp_path


@pytest.fixture
def site(tree: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Site]:
    with _running_site(tree) as running:
        # The playability check plays games and decides nothing this file asks about.
        # Off by default; a test that wants an unlisted draft turns it round.
        monkeypatch.setattr(screening, "check", lambda spec: {"passed": True})
        yield running


def _load(site: Site, root: Path) -> loader.Report:
    report = loader.load(site.store, root)
    site.store.commit()
    return report


def _saved(browser: Browser, spec: dict[str, Any]) -> Answer:
    answer = browser.post("/api/design/save", {"spec": spec, "parent": ""})
    assert answer.status == 200, answer.text
    return answer


def _link_hashes(browser: Browser, digest: str) -> set[str]:
    answer = browser.get(f"/api/game/{digest}")
    assert answer.status == 200, answer.text
    return {str(link["hash"]) for link in answer.payload["links"]}


@contextlib.contextmanager
def _scratch_store() -> Iterator[Any]:
    """A connection to an empty schema of its own, no migration run, dropped afterwards."""
    url = os.environ.get(TEST_URL_ENV, "").strip()
    if not url:
        pytest.skip(f"{TEST_URL_ENV} is unset: no store to migrate")
    pytest.importorskip("psycopg", reason="the site extra is not installed here")
    name = f"links_test_{secrets.token_hex(6)}"
    store = connect(url)
    with store.cursor() as cursor:
        cursor.execute(f"create schema {name}")
        cursor.execute(f"set search_path to {name}")
    store.commit()
    try:
        yield store
    finally:
        store.rollback()
        with store.cursor() as cursor:
            cursor.execute(f"drop schema if exists {name} cascade")
        store.commit()
        store.close()


# ----------------------------------------------------------- the table that retired


class TestTheLinksTableRetired:
    """M5: no list ranks by the store's pairs any more, so nothing writes them and
    `0038_links_retired.sql` takes the table, rows and all, off a store that has one."""

    def test_the_migration_drops_a_table_that_holds_rows(self, tree: Path) -> None:
        before = [one for one in migrate.available() if one.number < 38]
        with _scratch_store() as store:
            older = tree / "migrations-before-0038"
            older.mkdir()
            for migration in before:
                (older / migration.name).write_text(migration.sql, encoding="utf-8")
            migrate.run(store, older)
            with store.cursor() as cursor:
                cursor.execute(
                    "insert into games (hash, spec, name, origin) values"
                    " ('aaaa', '{}'::jsonb, 'a', 'corpus'), ('bbbb', '{}'::jsonb, 'b', 'corpus')"
                )
                cursor.execute("insert into links (a, b, units, relate)"
                               " values ('aaaa', 'bbbb', 1, 'r')")
            store.commit()

            report = migrate.run(store)
            store.commit()

            assert list(report.applied) == ["0038_links_retired.sql"], report.summary()
            with store.cursor() as cursor:
                cursor.execute("select to_regclass('links')")
                assert cursor.fetchone()[0] is None
                cursor.execute("select count(*) from games")
                assert cursor.fetchone()[0] == 2, "the games the pairs named stay"

    def test_the_load_and_a_save_write_no_pair(self, site: Site, tree: Path) -> None:
        report = _load(site, tree)
        assert "links" not in report.summary()
        _saved(site.signed_in("ada"), DRAFT)
        with site.store.cursor() as cursor:
            cursor.execute("select to_regclass('links')")
            assert cursor.fetchone()[0] is None


# ----------------------------------------------------------- who is named as near


class TestWhoIsNamedAsNear:
    """One meaning of near, and one answer for everybody: the map's games.

    A game's page names the map's games most like it, and the map draws the library less
    what an admin took off it -- the same for a visitor, a member, a draft's owner and an
    admin. So an unlisted draft is named on nobody's page, which is G7's acceptance kept
    without a per-person table: its existence stays its owner's business.
    """

    def _unlisted_draft(self, site: Site, monkeypatch: pytest.MonkeyPatch) -> Browser:
        monkeypatch.setattr(
            screening, "check", lambda spec: {"passed": False, "reasons": ["too dull"]}
        )
        ada = site.signed_in("ada")
        kept = _saved(ada, DRAFT)
        assert kept.payload["visibility"] == drafts.UNLISTED
        return ada

    def test_a_game_s_page_names_the_map_s_games_most_alike_first(
        self, site: Site, tree: Path
    ) -> None:
        _load(site, tree)

        answer = site.browser().get(f"/api/game/{LEFT_HASH}")

        assert answer.status == 200, answer.text
        page = answer.payload
        assert {str(link["hash"]) for link in page["links"]} == {
            MIDDLE_HASH, RIGHT_HASH, FAR_HASH
        }
        alike = [link["likeness"] for link in page["links"]]
        assert alike == sorted(alike, reverse=True)
        assert page["onMap"] is True and page["open"] == families.MUTUAL
        # The far game is the least like `left`, and is still named: there is no far case.
        assert page["links"][-1]["hash"] == FAR_HASH

    def test_the_page_and_the_map_agree(self, site: Site, tree: Path) -> None:
        """The map's panel lists each game's `like`; the page opens on the same games."""
        _load(site, tree)
        drawn = site.browser().get("/api/graph").payload
        for node in drawn["nodes"]:
            page = site.browser().get(f"/api/game/{node['hash']}").payload
            assert [link["hash"] for link in page["links"][: page["open"]]] == [
                row["hash"] for row in node["like"]
            ], node["name"]
            assert (page["community"] or {}).get("id") == node["community"]

    def test_nobody_is_told_of_an_unlisted_draft(
        self, site: Site, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not a visitor, not another member, not even its owner on a library game's page:
        the map draws no drafts, so no list of the map's games names one."""
        _load(site, tree)
        ada = self._unlisted_draft(site, monkeypatch)
        boss = site.signed_in("boss", admin.ADMIN)

        for person in (site.browser(), site.signed_in("bob"), ada, boss):
            answer = person.get(f"/api/game/{LEFT_HASH}")
            assert answer.status == 200
            # Not its hash, not its name, not a sentence about what it differs in.
            assert DRAFT_HASH not in answer.text
            assert str(DRAFT["name"]) not in answer.text

    def test_a_draft_s_own_page_names_the_library_games_most_like_it(
        self, site: Site, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ranked against the map's games by the map's relation, and the same list for
        whoever opens it by its link."""
        _load(site, tree)
        ada = self._unlisted_draft(site, monkeypatch)

        lists = []
        for person in (ada, site.browser()):
            page = person.get(f"/api/game/{DRAFT_HASH}").payload
            assert (page["onMap"], page["community"]) == (False, None)
            lists.append([link["hash"] for link in page["links"]])
        assert lists[0] == lists[1]
        assert set(lists[0]) == set(LIBRARY)
        assert lists[0][-1] == FAR_HASH

    def test_a_game_on_trial_is_not_on_the_map_and_so_named_by_nobody(
        self, site: Site, tree: Path
    ) -> None:
        """The map draws the library -- committed and archived games -- and near is the
        map's, so a draft its designer put on trial is near nothing until it joins it."""
        _load(site, tree)
        _saved(site.signed_in("ada"), DRAFT)
        site_store.on_trial(site.store, DRAFT_HASH)

        assert DRAFT_HASH not in _link_hashes(site.browser(), LEFT_HASH)

    def test_a_killed_game_is_named_to_nobody(self, site: Site, tree: Path) -> None:
        """An admin's *no more of this* reaches the near lists as well as the map."""
        _load(site, tree)
        admin.kill_game(site.store, MIDDLE_HASH)
        site.store.commit()

        boss = site.signed_in("boss", admin.ADMIN)
        assert MIDDLE_HASH not in _link_hashes(site.browser(), LEFT_HASH)
        assert MIDDLE_HASH not in _link_hashes(boss, LEFT_HASH)

    def test_a_retired_game_is_named_to_nobody(self, site: Site, tree: Path) -> None:
        _load(site, tree)
        with site.store.cursor() as cursor:
            cursor.execute("update games set retired = now() where hash = %s", (MIDDLE_HASH,))
        site.store.commit()

        assert MIDDLE_HASH not in _link_hashes(site.browser(), LEFT_HASH)


# ---------------------------------------------- the gap button on a person's own game


def _two_rules_out() -> dict[str, Any]:
    """Two declared rules from `left`, so there is a game in between to be built."""
    spec = _near(8, "a draft two rules out")
    spec["rules"]["stalemate"] = "loss"
    return spec


GAPPED = _two_rules_out()
GAPPED_HASH = spec_hash(GAPPED)

REFUSED = (
    "{} is not one of the games the map draws, so there is no gap between these two to"
    " build a game in"
)
"""What `?between=` said of every draft before Q354, and still says to a stranger."""


def _between(browser: Browser, first: str, second: str) -> Answer:
    return browser.get(f"/api/design/{first}?between={second}")


class TestTheGapButtonOnAPersonsOwnGame:
    """*Make the game in between* with a person's own draft at one end (Q354).

    `?between=` used to take only the games the map draws, which kept every draft out --
    its owner's too -- and let in a game an admin had killed. It asks `finding.findable`
    now, the store's predicate, of both ends: the path's as well as the query's, because
    anybody may *open* an unlisted draft by its link and that is not the same as being
    allowed to find it.
    """

    def _unlisted_draft(self, site: Site, monkeypatch: pytest.MonkeyPatch) -> Browser:
        monkeypatch.setattr(
            screening, "check", lambda spec: {"passed": False, "reasons": ["too dull"]}
        )
        ada = site.signed_in("ada")
        kept = _saved(ada, GAPPED)
        assert kept.payload["visibility"] == drafts.UNLISTED
        return ada

    def test_the_owner_s_unlisted_draft_is_either_end_of_a_gap(
        self, site: Site, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _load(site, tree)
        ada = self._unlisted_draft(site, monkeypatch)
        assert relate.units(GAPPED, LEFT) == 2, "the fixture is no longer a gap"

        for first, second in ((GAPPED_HASH, LEFT_HASH), (LEFT_HASH, GAPPED_HASH)):
            answer = _between(ada, first, second)

            assert answer.status == 200, answer.text
            made = answer.payload["between"]
            assert "spec" in made, made.get("why")
            assert relate.units(made["spec"], GAPPED) == 1
            assert relate.units(made["spec"], LEFT) == 1

    def test_a_stranger_is_refused_as_a_stranger_was_before(
        self, site: Site, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A visitor and another member, at either end: the status and the sentence the
        map's rule gave, word for word, and no rules of the draft's in the answer."""
        _load(site, tree)
        self._unlisted_draft(site, monkeypatch)
        refused = {"error": REFUSED.format(GAPPED["name"])}

        for stranger in (site.browser(), site.signed_in("bob")):
            for first, second in ((GAPPED_HASH, LEFT_HASH), (LEFT_HASH, GAPPED_HASH)):
                answer = _between(stranger, first, second)

                assert (answer.status, answer.payload) == (400, refused)

    def test_an_admin_may_build_from_anybody_s_live_draft(
        self, site: Site, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An admin is shown every live row, which is what `/admin` already lists."""
        _load(site, tree)
        self._unlisted_draft(site, monkeypatch)
        boss = site.signed_in("boss", admin.ADMIN)

        answer = _between(boss, LEFT_HASH, GAPPED_HASH)

        assert answer.status == 200, answer.text
        assert "spec" in answer.payload["between"]

    def test_a_game_on_trial_is_everybody_s_end(
        self, site: Site, tree: Path
    ) -> None:
        """The other half of the rule. A game on trial is public, so everybody may build
        between it and a library game. `site` stubs the check to pass."""
        _load(site, tree)
        _saved(site.signed_in("ada"), GAPPED)
        site_store.on_trial(site.store, GAPPED_HASH)

        answer = _between(site.browser(), LEFT_HASH, GAPPED_HASH)

        assert answer.status == 200, answer.text
        assert "spec" in answer.payload["between"]

    def test_a_game_an_admin_killed_is_no_end_for_anybody(
        self, site: Site, tree: Path
    ) -> None:
        """The map dropped a killed game at G7 and the gap button went on taking it,
        because the tree it asked knows nothing of the kill switch. `left` and `middle`
        are one rule apart, so the answer is a sentence rather than a game -- but it is
        an answer, and after the kill it is a refusal."""
        _load(site, tree)
        before = _between(site.browser(), LEFT_HASH, MIDDLE_HASH)
        assert before.status == 200, before.text
        assert "why" in before.payload["between"]

        admin.kill_game(site.store, MIDDLE_HASH)
        site.store.commit()

        refused = {"error": REFUSED.format(MIDDLE["name"])}
        for person in (site.browser(), site.signed_in("boss", admin.ADMIN)):
            answer = _between(person, LEFT_HASH, MIDDLE_HASH)
            assert (answer.status, answer.payload) == (400, refused)

    def test_findable_narrows_to_the_games_it_is_asked_about(
        self, site: Site, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`among` is the two ends, and a hash the store holds no row for is not one."""
        _load(site, tree)
        self._unlisted_draft(site, monkeypatch)
        with site.store.cursor() as cursor:
            cursor.execute("select id from users where handle = 'ada'")
            ada = int(cursor.fetchone()[0])
        asked = [LEFT_HASH, GAPPED_HASH, "0" * 16]

        assert finding.findable(site.store, among=asked) == {LEFT_HASH}
        assert finding.findable(site.store, ada, among=asked) == {LEFT_HASH, GAPPED_HASH}
        assert finding.findable(site.store, admin=True, among=asked) == {
            LEFT_HASH, GAPPED_HASH
        }
        assert finding.findable(site.store, among=[]) == set()
        assert finding.findable(site.store) == set(LIBRARY)


class TestTheGapButtonOnADraftsOwnPage:
    """*Make the game in between* where a person can reach it (Q429).

    The route above has let an owner build between their draft and a library game since
    Q354, and no page offered it: the map draws no drafts. A draft's own page now carries
    the library game most like it -- the first of its own *games like this one*, the map's
    likeness (M5) -- and the game in between, answered only where the press would be
    (`App._nearest` asks `findable` of both ends, as `_between` does). So the
    owner of an unlisted draft is offered it, and a stranger who opened the draft by its
    link -- which anybody may -- is sent nothing to press.
    """

    @pytest.fixture(autouse=True)
    def _no_fairy_stockfish(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A game page asks Fairy-Stockfish about its opponents, a second engine's
        handshake per game that none of these tests reads."""
        from chess_variant_sandbox import fairy

        monkeypatch.setattr(fairy, "available", lambda: False)
        monkeypatch.setattr(fairy, "verdict", lambda spec: None)

    def _unlisted_draft(self, site: Site, monkeypatch: pytest.MonkeyPatch) -> Browser:
        monkeypatch.setattr(
            screening, "check", lambda spec: {"passed": False, "reasons": ["too dull"]}
        )
        ada = site.signed_in("ada")
        kept = _saved(ada, GAPPED)
        assert kept.payload["visibility"] == drafts.UNLISTED
        return ada

    def _page(self, browser: Browser, digest: str) -> dict[str, Any]:
        answer = browser.get(f"/api/game/{digest}")
        assert answer.status == 200, answer.text
        return dict(answer.payload)

    def test_the_owner_is_offered_it_and_the_press_builds_it(
        self, site: Site, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The library game most like it is the first the page's own list names -- three
        are two rules out, alike as one another, and the map's relation breaks the tie --
        and the press is the map's request, which builds a game one rule from each end."""
        _load(site, tree)
        ada = self._unlisted_draft(site, monkeypatch)

        page = self._page(ada, GAPPED_HASH)
        near = page["between"]
        first = page["links"][0]

        assert near["hash"] == first["hash"] in {LEFT_HASH, MIDDLE_HASH, RIGHT_HASH}
        assert (near["units"], near["likeness"]) == (2, first["likeness"])
        assert {near["from"], near["taken"]} == {GAPPED_HASH, near["hash"]}

        pressed = _between(ada, near["from"], near["taken"])
        assert pressed.status == 200, pressed.text
        made = pressed.payload["between"]
        assert made["words"] == near["words"]
        assert relate.units(made["spec"], GAPPED) == 1
        assert relate.units(made["spec"], LIBRARY[near["hash"]]) == 1

    def test_a_stranger_who_opens_it_by_its_link_is_offered_nothing(
        self, site: Site, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The page opens for them, as an unlisted draft always has, and carries no gap:
        the press would be the 400 a stranger is given, so there is no button to press."""
        _load(site, tree)
        self._unlisted_draft(site, monkeypatch)

        for stranger in (site.browser(), site.signed_in("bob")):
            page = self._page(stranger, GAPPED_HASH)

            assert page["between"] is None
            assert page["name"] == GAPPED["name"], "the draft itself still opens"

    def test_a_game_on_trial_is_offered_to_everybody(
        self, site: Site, tree: Path
    ) -> None:
        """The same rule the other way round: a game on trial is everybody's end of a gap
        (the route's test above), so its page offers the button to a visitor too.
        `site` stubs the check to pass."""
        _load(site, tree)
        _saved(site.signed_in("ada"), GAPPED)
        site_store.on_trial(site.store, GAPPED_HASH)

        page = self._page(site.browser(), GAPPED_HASH)
        near = page["between"]

        assert near["hash"] == page["links"][0]["hash"] and "words" in near

    def test_the_nearest_is_among_the_games_this_person_may_be_shown(
        self, site: Site, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A game an admin killed is off the map and no end of a gap, so it is not the
        most alike either: the next one is, and its press is answered."""
        _load(site, tree)
        ada = self._unlisted_draft(site, monkeypatch)
        killed = self._page(ada, GAPPED_HASH)["between"]["hash"]
        admin.kill_game(site.store, killed)
        site.store.commit()

        page = self._page(ada, GAPPED_HASH)
        near = page["between"]

        assert killed not in {link["hash"] for link in page["links"]}
        assert near["hash"] == page["links"][0]["hash"] != killed
        pressed = _between(ada, near["from"], near["taken"])
        assert pressed.status == 200, pressed.text
        assert "spec" in pressed.payload["between"]


# --------------------------------------------------------------------- and the map


class TestTheMapAsksTheStore:
    """The first thing earlier stages left for G7: the map reads the tree, and the tree
    knows nothing of the kill switch."""

    def _nodes(self, browser: Browser) -> set[str]:
        answer = browser.get("/api/graph")
        assert answer.status == 200, answer.text
        return {
            digest
            for node in answer.payload["nodes"]
            for digest in node["hashes"]
        }

    def test_a_game_an_admin_killed_leaves_the_map(self, site: Site, tree: Path) -> None:
        _load(site, tree)
        assert self._nodes(site.browser()) == set(LIBRARY)

        admin.kill_game(site.store, MIDDLE_HASH)
        site.store.commit()

        drawn = self._nodes(site.browser())
        assert MIDDLE_HASH not in drawn
        assert drawn == set(LIBRARY) - {MIDDLE_HASH}

    def test_nobody_s_draft_is_on_it(self, site: Site, tree: Path) -> None:
        """Still true, and deliberately: a per-viewer map is a stage of its own."""
        _load(site, tree)
        ada = site.signed_in("ada")
        _saved(ada, DRAFT)

        assert DRAFT_HASH not in self._nodes(ada)


# ---------------------------------- the site's games are the local application's and the map's

# Eighteen of the committed and archived rulesets, named by file because the claims are
# about these games: chess and the games one or two rules from it; shogi, which nothing
# here is within three rules of; and the corpus's twin pair. Eighteen rather than the
# whole library, so the class's map is drawn in a second or two.
COMMITTED = (
    "chess", "almost", "berolina", "atomic", "crazyhouse", "kingofthehill", "knightmate",
    "shogi", "courier", "shatranj", "makruk", "gardner",
)
ARCHIVED = (
    "perpetual", "chess-sweeping-bishops", "marchfield", "reckoning", "wolves",
    "wolves-immune",
)


def _corpus_spec(stem: str) -> dict[str, Any]:
    if stem in REBUILT:  # `wolves-immune`, retired by Q876 for `pitchfork`; rebuilt from `wolves`
        return REBUILT[stem]()
    folder = library.RULESET_DIR if stem in COMMITTED else library.ARCHIVE_DIR
    return dict(json.loads((folder / f"{stem}.json").read_text(encoding="utf-8")))


@pytest.fixture(scope="class")
def corpus_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A checkout holding the eighteen, committed and archived where they really are."""
    root = tmp_path_factory.mktemp("corpus")
    for name in ("rulesets", "runs", "archive"):
        (root / name).mkdir()
    for stem in (*COMMITTED, *ARCHIVED):
        folder = "rulesets" if stem in COMMITTED else "archive"
        (root / folder / f"{stem}.json").write_text(
            json.dumps(_corpus_spec(stem)), encoding="utf-8"
        )
    return root


@pytest.fixture(scope="class")
def corpus(corpus_tree: Path) -> Iterator[Site]:
    """One site and one load for the class. Every test in it only reads."""
    with _running_site(corpus_tree) as running:
        _load(running, corpus_tree)
        yield running


def _hash_of(stem: str) -> str:
    return spec_hash(_corpus_spec(stem))


def _local_links(root: Path, digest: str) -> list[dict[str, Any]]:
    """What the local application's game page lists, over the same tree, as JSON."""
    app = LocalApp(root / "rulesets", root / "runs", archive_dir=root / "archive")
    return list(json.loads(json.dumps(app.get(f"/api/game/{digest}")["links"])))


class TestTheSiteTheLocalApplicationAndTheMapNameTheSameGames:
    """M5's acceptance on real games: one ranking of *near* on the map, the game page and
    the panel, on both servers. Chess, which many games are like; shogi, which few are; and
    the corpus's twin pair, which the map draws as one dot."""

    def test_chess_lists_the_same_games_on_both_servers_and_the_map(
        self, corpus: Site, corpus_tree: Path
    ) -> None:
        chess = _hash_of("chess")

        answer = corpus.browser().get(f"/api/game/{chess}")

        assert answer.status == 200, answer.text
        site = answer.payload
        assert site["links"] == _local_links(corpus_tree, chess)
        node = next(
            node for node in corpus.browser().get("/api/graph").payload["nodes"]
            if node["hash"] == chess
        )
        assert [link["hash"] for link in site["links"][: site["open"]]] == [
            row["hash"] for row in node["like"]
        ]
        assert [link["likeness"] for link in site["links"][: site["open"]]] == [
            row["likeness"] for row in node["like"]
        ]

    def test_shogi_far_from_most_games_lists_the_same_games_on_both(
        self, corpus: Site, corpus_tree: Path
    ) -> None:
        """Nothing here is within three rules of shogi, and its page is ordered by the
        same likeness as chess's: no second ranking takes over far out."""
        shogi = _hash_of("shogi")

        answer = corpus.browser().get(f"/api/game/{shogi}")

        assert answer.status == 200, answer.text
        site = answer.payload["links"]
        assert site == _local_links(corpus_tree, shogi)
        assert min(link["units"] for link in site) > 3, "the fixture is no longer far"
        alike = [link["likeness"] for link in site]
        assert alike == sorted(alike, reverse=True)
        assert all("distance" not in link for link in site)

    def test_the_page_for_a_game_s_own_address_names_them_in_the_same_order(
        self, corpus: Site
    ) -> None:
        chess = _hash_of("chess")
        page = corpus.browser().get(f"/api/game/{chess}").payload
        html = corpus.browser().get(f"/game/{chess}").text
        named = [link["name"] for link in page["links"]]
        at = [html.index(f">{name}</a>") for name in named]
        assert at == sorted(at)

    def test_a_second_spelling_reads_as_the_dot_the_map_draws(self, corpus: Site) -> None:
        """`wolves-immune` compiles to `wolves`' tables, so the map draws one dot for the
        two and both pages name the games like that dot -- neither names the other."""
        wolves, immune = _hash_of("wolves"), _hash_of("wolves-immune")

        one = corpus.browser().get(f"/api/game/{wolves}").payload["links"]
        other = corpus.browser().get(f"/api/game/{immune}").payload["links"]

        assert [link["hash"] for link in one] == [link["hash"] for link in other]
        assert not {wolves, immune} & {link["hash"] for link in one}
