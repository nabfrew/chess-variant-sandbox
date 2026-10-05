"""How near two of the store's games are, and who may be told. G7.

The G7 row of `docs/archive/plan-game-graph.md`, which is *the plan* below: the migration,
the load step, a draft's pairs at save, and rows removed on delete. Its acceptance is three
sentences and each one is a class here -- a second load compares nothing, deleting a game
leaves no row that names it, and a private draft's links reach only somebody who may see
the draft.

The last is the stage's point and is asked of four people: a signed-out visitor, another
signed-in member, the draft's owner and an admin. It is asked over HTTP against a real
store, because *a person cannot see this* is a property of the answer a browser gets and
not of a function's return value.

**The library here is four specs in a `tmp_path` of its own**, three of them one rule
apart and one far from all of them, so the near case and the far case are both real
without a committed ruleset being involved. Nothing here works out a legal move, an
outcome or a score; `spec/relate.py` says what differs and this stores it.

T4 of the plan's §9.5 adds the similarity index to every row, and the last class asks
what that is for: that the site names the neighbours the local application names, in the
same order. That claim is about real games -- chess, which many games are near, and
shogi, which nothing is -- so it is asked over eighteen committed and archived rulesets,
and its tests say which games in their names.

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

from chess_variant_sandbox.site import (
    admin,
    badges,
    drafts,
    identity,
    links,
    migrate,
    resting,
    rounds,
    screening,
)
from chess_variant_sandbox.site import load as loader
from chess_variant_sandbox.site import pages as site_pages
from chess_variant_sandbox.site import server as site_server
from chess_variant_sandbox.site.db import connect
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.spec import relate, spec_hash
from chess_variant_sandbox.web import designing, graphing, library
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

    def pairs(self) -> dict[tuple[str, str], int]:
        with self.store.cursor() as cursor:
            cursor.execute("select a, b, units from links")
            return {(str(a), str(b)): int(units) for a, b, units in cursor.fetchall()}

    def row(self, first: str, second: str) -> tuple[Any, ...] | None:
        low, high = sorted((first, second))
        with self.store.cursor() as cursor:
            cursor.execute(
                "select units, kind, kinds, unit, sentence, differences, relate"
                "  from links where a = %s and b = %s",
                (low, high),
            )
            return cursor.fetchone()

    def likeness(self) -> dict[tuple[str, str], tuple[Any, Any]]:
        """Every pair's similarity index and its counts, as the table holds them."""
        return _likeness(self.store)


def _likeness(store: Any) -> dict[tuple[str, str], tuple[Any, Any]]:
    with store.cursor() as cursor:
        cursor.execute("select a, b, similarity, counts from links")
        return {(str(a), str(b)): (index, counts) for a, b, index, counts in cursor.fetchall()}


@contextlib.contextmanager
def _running_site(root: Path) -> Iterator[Site]:
    """The site on a port of its own, over `root`'s library and the session's store."""
    url, name = site_store.emptied("no store to keep links in")

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


class _Counting:
    """A connection that writes down which verb each statement went through."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[str] = []

    @contextlib.contextmanager
    def cursor(self) -> Iterator[Any]:
        with self.inner.cursor() as cursor:
            yield _CountingCursor(cursor, self.calls)


class _CountingCursor:
    def __init__(self, inner: Any, calls: list[str]) -> None:
        self.inner = inner
        self.calls = calls

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("execute")
        return self.inner.execute(*args, **kwargs)

    def executemany(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("executemany")
        return self.inner.executemany(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _link_hashes(browser: Browser, digest: str) -> set[str]:
    answer = browser.get(f"/api/game/{digest}")
    assert answer.status == 200, answer.text
    return {str(link["hash"]) for link in answer.payload["links"]}


# --------------------------------------------------------------- what the load writes


class TestTheTableTheLoadFills:
    """Every pair of the store's games, in sort order, under this reading of them."""

    def test_every_pair_gets_a_row_with_relate_s_own_count(self, site: Site, tree: Path) -> None:
        report = _load(site, tree)

        wanted = {
            tuple(sorted((first, second))): relate.units(LIBRARY[first], LIBRARY[second])
            for first in LIBRARY
            for second in LIBRARY
            if first < second
        }
        assert len(wanted) == 6, "four games make six pairs"
        assert site.pairs() == wanted
        assert (report.links.added, report.links.computed) == (6, 6)

    def test_a_near_pair_carries_the_reading_and_a_far_one_carries_the_count(
        self, site: Site, tree: Path
    ) -> None:
        """Past three rules a list of differences is a ranking rather than a reading, so
        the columns are left empty. `NEAR` is `web/library.py`'s, shared rather than
        repeated."""
        _load(site, tree)

        near = site.row(LEFT_HASH, MIDDLE_HASH)
        assert near is not None
        relation = relate.relate(LEFT, MIDDLE)
        assert relation.units == 1, "the fixture's siblings are one rule apart"
        units, kind, kinds, unit, sentence, differences, version = near
        assert units == 1
        assert (kind, unit, sentence) == (
            relation.differences[0].kind,
            relation.differences[0].unit,
            relation.sentence,
        )
        assert list(kinds) == [relation.differences[0].kind]
        assert differences == [one.as_dict() for one in relation.differences]
        assert version == links.RELATE

        far = site.row(LEFT_HASH, FAR_HASH)
        assert far is not None
        assert far[0] > links.NEAR, "the fixture's far spec is well past the threshold"
        assert far[1:6] == ("", [], "", "", None)

    def test_every_pair_carries_relate_s_index_and_its_counts(
        self, site: Site, tree: Path
    ) -> None:
        """T4 of the plan's §9.5. The far pairs too: a far game's relatives are ranked,
        and the index is what the site orders by beside the count."""
        _load(site, tree)

        wanted = {}
        for first in LIBRARY:
            for second in LIBRARY:
                if first < second:
                    alike = relate.similarity(LIBRARY[first], LIBRARY[second])
                    wanted[(first, second)] = (alike.index, alike.counts)
        assert site.likeness() == wanted
        assert all(index > 0 for index, _ in wanted.values()), "six different games"

    def test_the_stamp_names_the_reading_and_the_weighing(self) -> None:
        """The stored index moves with the ratio and the setting's weight, which can
        change without `RELATE_VERSION` moving, so both are in the row's stamp."""
        assert relate.RELATE_VERSION in links.RELATE
        assert f"{relate.SIMILARITY_RATIO}:1" in links.RELATE
        assert f"setting {relate.SETTING_WEIGHT:g}" in links.RELATE

    def test_a_second_load_compares_nothing_and_writes_nothing(
        self, site: Site, tree: Path
    ) -> None:
        """G7's first acceptance sentence, and the reason the deploy can run this every
        time. The comparison is the expensive half."""
        _load(site, tree)
        before = site.pairs()

        second = _load(site, tree)

        assert (second.links.added, second.links.updated) == (0, 0)
        assert second.links.computed == 0
        assert second.links.unchanged == 6
        assert site.pairs() == before

    def test_the_pairs_go_to_the_store_in_one_batch(self, site: Site, tree: Path) -> None:
        """Not one statement per pair: at 500 games the round trips were about 60 s of a
        115 s version bump, and a save pays the same per pair against the library. What
        is sent is one read of the rows held and one batch, however many pairs there are."""
        _load(site, tree)
        wanted = site.pairs()
        with site.store.cursor() as cursor:
            cursor.execute("delete from links")
        counted = _Counting(site.store)

        filled = links.fill(counted, LIBRARY)
        links.forget(site.store, LEFT_HASH)
        saved = links.write(counted, LEFT_HASH, LEFT)

        assert (filled.added, saved.added) == (6, 3)
        assert counted.calls == [
            "execute",  # fill: the pairs held
            "executemany",  # fill: all six
            "execute",  # write: the other games' specs
            "execute",  # write: this game's pairs held
            "executemany",  # write: all three
        ]
        assert site.pairs() == wanted

    def test_a_reading_that_moves_is_taken_again(
        self, site: Site, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`relate` is on the row, so a version bump recomputes rather than a migration."""
        _load(site, tree)
        monkeypatch.setattr(links, "RELATE", "r-next")

        again = _load(site, tree)

        assert (again.links.added, again.links.updated) == (0, 6)
        assert again.links.computed == 6
        with site.store.cursor() as cursor:
            cursor.execute("select distinct relate from links")
            assert [row[0] for row in cursor.fetchall()] == ["r-next"]


# ------------------------------------------------- the first load after `0013`


@contextlib.contextmanager
def _scratch_store() -> Iterator[Any]:
    """A connection to an empty schema of its own, no migration run, dropped afterwards."""
    url = os.environ.get(TEST_URL_ENV, "").strip()
    if not url:
        pytest.skip(f"{TEST_URL_ENV} is unset: no store to keep links in")
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


class TestTheFirstLoadAfterTheIndexArrived:
    """T4's first acceptance sentence: the first load after `0013_similarity.sql`
    compares every pair and the second compares none.

    Asked of a store that really was at `0012`: the migrations before `0013` run, the
    games are loaded, and the links are the rows the code of that time wrote -- `relate`'s
    count and reading stamped with `RELATE_VERSION` alone, and no index. Then `0013` runs
    over them, which is what the deploy does.
    """

    def test_the_old_rows_are_compared_once_and_then_never(
        self, tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        older = tmp_path / "migrations-before-0013"
        older.mkdir()
        for migration in migrate.available():
            if migration.number < 13:
                (older / migration.name).write_text(migration.sql, encoding="utf-8")

        with _scratch_store() as store:
            migrate.run(store, older)
            with monkeypatch.context() as patched:
                patched.setattr(links, "fill", lambda connection, specs: links.Written())
                # Today's badges read `games.screen_passed`, which `0016_drafts.sql` adds;
                # the load of that time granted from a table without it, and no badge is
                # what this test is about.
                patched.setattr(badges, "grant_for_games", lambda connection, games: [])
                # The same for the resting sweep, which reads `0020_resting.sql`'s column.
                patched.setattr(resting, "rest", lambda connection: [])
                # And the round's three reads, of `0021_rounds.sql`'s tables (Q503h).
                patched.setattr(rounds, "confirmed", lambda connection: {})
                patched.setattr(
                    rounds, "sweep", lambda connection: rounds.Swept(None, [], [], [])
                )
                patched.setattr(loader, "shortlisted_specs", lambda connection, known: {})
                loader.load(store, tree)
            with store.cursor() as cursor:
                for first in LIBRARY:
                    for second in LIBRARY:
                        if first < second:
                            relation = relate.relate(LIBRARY[first], LIBRARY[second])
                            cursor.execute(
                                "insert into links (a, b, units, kind, kinds, unit,"
                                " sentence, differences, relate) values"
                                " (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)",
                                (first, second, relation.units,
                                 *links._reading(relation), relate.RELATE_VERSION),
                            )
            store.commit()

            report = migrate.run(store)
            store.commit()
            # First of what the deploy applies; every later migration comes with it.
            assert report.applied[0] == "0013_similarity.sql", report.summary()
            assert set(_likeness(store).values()) == {(None, None)}

            once = loader.load(store, tree)
            store.commit()
            assert (once.links.added, once.links.updated) == (0, 6)
            assert once.links.computed == 6
            assert None not in {index for index, _ in _likeness(store).values()}

            twice = loader.load(store, tree)
            store.commit()
            assert (twice.links.added, twice.links.updated) == (0, 0)
            assert (twice.links.computed, twice.links.unchanged) == (0, 6)

    def test_a_row_without_the_index_is_not_held_whatever_its_stamp(
        self, site: Site, tree: Path
    ) -> None:
        """`_held` reads the columns as well as the stamp, so a row that somehow carries
        this stamp and no index is compared again rather than trusted."""
        _load(site, tree)
        with site.store.cursor() as cursor:
            cursor.execute(
                "update links set similarity = null where a = %s and b = %s",
                tuple(sorted((LEFT_HASH, FAR_HASH))),
            )
        site.store.commit()

        again = _load(site, tree)

        assert (again.links.added, again.links.updated) == (0, 1)
        assert again.links.computed == 1


# ------------------------------------------------------------- what a save writes

class TestADraftHasItsLinksAtOnce:
    def test_the_save_writes_the_draft_s_pairs_in_its_own_transaction(
        self, site: Site, tree: Path
    ) -> None:
        _load(site, tree)
        ada = site.signed_in("ada")

        _saved(ada, DRAFT)

        assert set(site.pairs()) >= {
            tuple(sorted((DRAFT_HASH, digest))) for digest in LIBRARY
        }
        assert site.pairs()[tuple(sorted((DRAFT_HASH, LEFT_HASH)))] == relate.units(
            DRAFT, LEFT
        )

    def test_saving_the_same_rules_again_compares_nothing(
        self, site: Site, tree: Path
    ) -> None:
        _load(site, tree)
        ada = site.signed_in("ada")
        _saved(ada, DRAFT)
        before = site.pairs()

        _saved(ada, DRAFT)

        assert site.pairs() == before

    def test_a_save_compares_this_game_against_the_library_and_nothing_else(
        self, site: Site, tree: Path
    ) -> None:
        """What the plan's §4 costs as *a new game*: one comparison per other game, and
        no second pass over the pairs that already exist. The wall clock is in the plan;
        the count is what a test can hold."""
        _load(site, tree)
        _saved(site.signed_in("ada"), DRAFT)
        links.forget(site.store, DRAFT_HASH)
        site.store.commit()

        written = links.write(site.store, DRAFT_HASH, DRAFT)
        site.store.commit()

        assert written.computed == len(LIBRARY)
        assert (written.added, written.updated) == (len(LIBRARY), 0)


# ------------------------------------------------------------- who may be shown one


class TestWhoMayBeShownALink:
    """G7's third acceptance sentence, asked of four people.

    An unlisted draft opens by its link, as it always has. What it must not do is be
    *found* -- named on somebody else's page, counted in their neighbour list, or spelled
    out in a rule difference. That is the line between openable and findable, and it is
    one predicate in `site/links.py`.
    """

    def _unlisted_draft(self, site: Site, monkeypatch: pytest.MonkeyPatch) -> Browser:
        monkeypatch.setattr(
            screening, "check", lambda spec: {"passed": False, "reasons": ["too dull"]}
        )
        ada = site.signed_in("ada")
        kept = _saved(ada, DRAFT)
        assert kept.payload["visibility"] == drafts.UNLISTED
        return ada

    def test_a_visitor_is_told_nothing_about_it_at_all(
        self, site: Site, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _load(site, tree)
        self._unlisted_draft(site, monkeypatch)

        answer = site.browser().get(f"/api/game/{LEFT_HASH}")

        assert answer.status == 200
        assert {str(link["hash"]) for link in answer.payload["links"]} == {
            MIDDLE_HASH, RIGHT_HASH
        }
        # Not its hash, not its name, not a sentence about what it differs in.
        assert DRAFT_HASH not in answer.text
        assert str(DRAFT["name"]) not in answer.text

    def test_another_signed_in_person_is_told_nothing_either(
        self, site: Site, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _load(site, tree)
        self._unlisted_draft(site, monkeypatch)
        bob = site.signed_in("bob")

        answer = bob.get(f"/api/game/{LEFT_HASH}")

        assert DRAFT_HASH not in answer.text
        assert str(DRAFT["name"]) not in answer.text

    def test_the_owner_sees_their_own_draft_among_the_neighbours(
        self, site: Site, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _load(site, tree)
        ada = self._unlisted_draft(site, monkeypatch)

        assert _link_hashes(ada, LEFT_HASH) == {MIDDLE_HASH, RIGHT_HASH, DRAFT_HASH}

    def test_an_admin_sees_it_too(
        self, site: Site, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _load(site, tree)
        self._unlisted_draft(site, monkeypatch)
        boss = site.signed_in("boss", admin.ADMIN)

        assert DRAFT_HASH in _link_hashes(boss, LEFT_HASH)

    def test_a_draft_that_passed_the_check_is_still_found_by_nobody_else(
        self, site: Site, tree: Path
    ) -> None:
        """Q503a: a save lists nothing, so a pass alone makes nobody else's page name it."""
        _load(site, tree)
        _saved(site.signed_in("ada"), DRAFT)

        assert DRAFT_HASH not in _link_hashes(site.browser(), LEFT_HASH)

    def test_a_game_on_trial_is_public_and_everybody_sees_it(
        self, site: Site, tree: Path
    ) -> None:
        """The other half of the same rule: visibility decides, and a game its designer
        published is listed. `site` stubs the check to pass."""
        _load(site, tree)
        _saved(site.signed_in("ada"), DRAFT)
        site_store.on_trial(site.store, DRAFT_HASH)

        assert DRAFT_HASH in _link_hashes(site.browser(), LEFT_HASH)

    def test_a_killed_game_is_named_to_nobody(self, site: Site, tree: Path) -> None:
        """An admin's *no more of this* reaches the links as well as the listing."""
        _load(site, tree)
        admin.kill_game(site.store, MIDDLE_HASH)
        site.store.commit()

        boss = site.signed_in("boss", admin.ADMIN)
        assert MIDDLE_HASH not in _link_hashes(site.browser(), LEFT_HASH)
        assert MIDDLE_HASH not in _link_hashes(boss, LEFT_HASH)

    def test_the_far_game_is_reached_by_the_nearest_four_rule(
        self, site: Site, tree: Path
    ) -> None:
        """A game nothing is within three rules of is shown its nearest few instead of an
        empty panel, which is `web/library.py`'s rule and the store's half of it."""
        _load(site, tree)

        assert _link_hashes(site.browser(), FAR_HASH) == {
            LEFT_HASH, MIDDLE_HASH, RIGHT_HASH
        }


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
    its owner's too -- and let in a game an admin had killed. It asks `links.findable`
    now, the predicate every neighbour list here is narrowed by, of both ends: the path's
    as well as the query's, because anybody may *open* an unlisted draft by its link and
    that is not the same as being allowed to find it.
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
        """The other half of the rule. A game on trial is named on everybody's neighbour
        list, so everybody may build between it and its neighbour. `site` stubs the check
        to pass."""
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

        assert links.findable(site.store, among=asked) == {LEFT_HASH}
        assert links.findable(site.store, ada, among=asked) == {LEFT_HASH, GAPPED_HASH}
        assert links.findable(site.store, admin=True, among=asked) == {
            LEFT_HASH, GAPPED_HASH
        }
        assert links.findable(site.store, among=[]) == set()
        assert links.findable(site.store) == set(LIBRARY)


class TestTheGapButtonOnADraftsOwnPage:
    """*Make the game in between* where a person can reach it (Q429).

    The route above has let an owner build between their draft and a library game since
    Q354, and no page offered it: the map draws no drafts. A draft's own page now carries
    the library game nearest it and the game in between, answered only where the press
    would be (`App._nearest` asks `findable` of both ends, as `_between` does). So the
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
        """The nearest library game is the first the page's own neighbour list names --
        three are two rules out, and the name breaks the tie as it does there -- and the
        press is the map's request, which builds a game one rule from each end."""
        _load(site, tree)
        ada = self._unlisted_draft(site, monkeypatch)

        page = self._page(ada, GAPPED_HASH)
        near = page["between"]

        assert near["hash"] == page["links"][0]["hash"] == LEFT_HASH
        assert (near["name"], near["units"], near["distance"]) == ("left", 2, None)
        assert {near["from"], near["taken"]} == {GAPPED_HASH, LEFT_HASH}

        pressed = _between(ada, near["from"], near["taken"])
        assert pressed.status == 200, pressed.text
        made = pressed.payload["between"]
        assert made["words"] == near["words"]
        assert relate.units(made["spec"], GAPPED) == 1
        assert relate.units(made["spec"], LEFT) == 1

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

        near = self._page(site.browser(), GAPPED_HASH)["between"]

        assert near["hash"] == LEFT_HASH and "words" in near

    def test_the_nearest_is_among_the_games_this_person_may_be_shown(
        self, site: Site, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A game an admin killed is off the map and no end of a gap, so it is not the
        nearest either: the next one is, and its press is answered."""
        _load(site, tree)
        ada = self._unlisted_draft(site, monkeypatch)
        admin.kill_game(site.store, LEFT_HASH)
        site.store.commit()

        near = self._page(ada, GAPPED_HASH)["between"]

        assert near["hash"] == MIDDLE_HASH
        pressed = _between(ada, near["from"], near["taken"])
        assert pressed.status == 200, pressed.text
        assert "spec" in pressed.payload["between"]


# ------------------------------------------------------------ what a delete takes


class TestWhatADeleteTakesWithIt:
    """G7's second acceptance sentence. A link is the rules said from one step away."""

    def _retire(self, site: Site, digest: str) -> None:
        with site.store.cursor() as cursor:
            cursor.execute(
                "update games set retired = now() where hash = %s", (digest,)
            )
        site.store.commit()

    def test_deleting_a_game_leaves_no_row_that_names_it(
        self, site: Site, tree: Path
    ) -> None:
        _load(site, tree)
        assert any(MIDDLE_HASH in pair for pair in site.pairs())
        self._retire(site, MIDDLE_HASH)
        boss = site.signed_in("boss", admin.ADMIN)

        answer = boss.post(f"/api/game/{MIDDLE_HASH}/delete")

        assert answer.status == 200, answer.text
        assert not any(MIDDLE_HASH in pair for pair in site.pairs())
        # The other games keep theirs: a delete is about one game's rules.
        assert site.pairs()[tuple(sorted((LEFT_HASH, RIGHT_HASH)))] == relate.units(
            LEFT, RIGHT
        )

    def test_saving_the_deleted_rules_again_does_not_write_them_back(
        self, site: Site, tree: Path
    ) -> None:
        """A delete with a way back is not one, and a link is a sentence about the rules."""
        _load(site, tree)
        ada = site.signed_in("ada")
        _saved(ada, DRAFT)
        self._retire(site, DRAFT_HASH)
        with site.store.cursor() as cursor:
            cursor.execute(
                "update games set owner = (select id from users where handle = 'ada')"
                " where hash = %s",
                (DRAFT_HASH,),
            )
        site.store.commit()
        assert ada.post(f"/api/game/{DRAFT_HASH}/delete").status == 200

        _saved(ada, DRAFT)

        assert not any(DRAFT_HASH in pair for pair in site.pairs())

    def test_a_retired_game_keeps_its_rows_and_is_simply_not_offered(
        self, site: Site, tree: Path
    ) -> None:
        """§4 of the plan: a retired game's rows stay and the listing query hides it."""
        _load(site, tree)
        self._retire(site, MIDDLE_HASH)

        assert any(MIDDLE_HASH in pair for pair in site.pairs())
        assert MIDDLE_HASH not in _link_hashes(site.browser(), LEFT_HASH)


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


# ------------------------------------------- the site's order is the local application's

# Eighteen of the committed and archived rulesets, named by file because the plan's claims
# are about these games. Chess and the games one or two rules from it, so its links are
# many and the index orders them differently from the alphabet; shogi, which nothing is
# within three rules of, with the four `distance.py` names for it on the whole corpus
# (crazyhouse, courier, shatranj, makruk) and the four the count of rules names instead
# (marchfield, reckoning, shatranj, gardner), so the two rules pick different games; and
# the corpus's twin pair. Eighteen rather than seventy-one, because the whole corpus's
# first load is about sixteen seconds here and these eighteen take the class about two.
# The whole corpus was run once by hand for T4's record: all 71 pages agree.
COMMITTED = (
    "chess", "almost", "berolina", "atomic", "crazyhouse", "kingofthehill", "knightmate",
    "shogi", "courier", "shatranj", "makruk", "gardner",
)
ARCHIVED = (
    "perpetual", "chess-sweeping-bishops", "marchfield", "reckoning", "wolves",
    "wolves-immune",
)


def _corpus_spec(stem: str) -> dict[str, Any]:
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


class TestTheSiteOrdersAsTheLocalApplicationDoes:
    """T4 of `docs/archive/plan-game-graph.md` §9.5, and what T5 left: the site asks this table
    which neighbours a game has, and it has to name the ones the local application names,
    in the same order. Within three rules that is the count, then the index, then the
    name. Past three it is `distance.py`'s four, where the table used to hand over the
    count's four and the page only re-sorted them."""

    def test_chess_near_many_lists_the_same_links_on_both(
        self, corpus: Site, corpus_tree: Path
    ) -> None:
        chess = _hash_of("chess")

        answer = corpus.browser().get(f"/api/game/{chess}")

        assert answer.status == 200, answer.text
        site = answer.payload["links"]
        assert site == _local_links(corpus_tree, chess)
        assert all(link["units"] <= links.NEAR for link in site)
        assert [link["hash"] for link in site[:3]] == [
            _hash_of("perpetual"), _hash_of("almost"), _hash_of("chess-sweeping-bishops")
        ], "a clock, a piece, then a mechanism: the index's order, not the alphabet's"

    def test_shogi_far_from_everything_lists_the_same_four_on_both(
        self, corpus: Site, corpus_tree: Path
    ) -> None:
        shogi = _hash_of("shogi")

        answer = corpus.browser().get(f"/api/game/{shogi}")

        assert answer.status == 200, answer.text
        site = answer.payload["links"]
        assert site == _local_links(corpus_tree, shogi)
        wanted = ["crazyhouse", "courier", "shatranj", "makruk"]
        assert [link["hash"] for link in site] == [_hash_of(stem) for stem in wanted]
        assert all(link["distance"] is not None for link in site)

    def test_shogi_s_four_are_the_features_and_not_the_count_s(
        self, corpus: Site
    ) -> None:
        """The case the old table got wrong. By the count of rules, then the name, shogi's
        four would be marchfield, reckoning, shatranj and gardner."""
        shogi = _hash_of("shogi")
        by_count = sorted(
            (relate.units(_corpus_spec("shogi"), _corpus_spec(stem)), stem)
            for stem in (*COMMITTED, *ARCHIVED)
            if stem != "shogi"
        )
        assert [stem for _, stem in by_count[:4]] == [
            "marchfield", "reckoning", "shatranj", "gardner"
        ], "the fixture no longer separates the two rules"

        found = links.neighbours(corpus.store, shogi)

        assert [link.hash for link in found] == [
            _hash_of(stem) for stem in ("crazyhouse", "courier", "shatranj", "makruk")
        ]

    def test_the_table_orders_chess_s_links_by_count_then_index_then_name(
        self, corpus: Site
    ) -> None:
        """The order the cap cuts at, so the store cuts where `library.neighbours` does."""
        chess = _hash_of("chess")

        found = links.neighbours(corpus.store, chess)

        keys = [(link.units, link.similarity, link.name.lower(), link.hash) for link in found]
        assert keys == sorted(keys)
        assert all(link.similarity is not None for link in found)
        cut = links.neighbours(corpus.store, chess, most=3)
        assert [link.hash for link in cut] == [link.hash for link in found[:3]]
        assert [link.name for link in cut] != sorted(link.name for link in found)[:3]

    def test_the_wolves_twins_keep_their_own_row_and_their_own_index(
        self, corpus: Site, tmp_path: Path
    ) -> None:
        """T4's last sentence. The map merges two specs that compile to one set of tables
        and reads them at 0; the table compiles nothing, so the pair keeps what `relate`
        says, as it keeps its own count, which is what the game page reads too."""
        wolves, immune = _corpus_spec("wolves"), _corpus_spec("wolves-immune")
        games = {game.hash: game for game in library.discover(
            corpus.app.ruleset_dir, corpus.app.runs_dir, corpus.app.archive_dir
        )}
        cache = graphing._Cache(tmp_path / graphing.CACHE_FILE)
        first, second = games[spec_hash(wolves)], games[spec_hash(immune)]
        assert cache.twin(first) == cache.twin(second), "the corpus's twins have parted"

        low, high = sorted((first.hash, second.hash))
        pair = (low, high)
        index, counts = corpus.likeness()[pair]
        alike = relate.similarity(wolves, immune)
        assert (index, counts) == (alike.index, alike.counts)
        assert index > 0
        assert corpus.pairs()[pair] == relate.units(wolves, immune) > 0
