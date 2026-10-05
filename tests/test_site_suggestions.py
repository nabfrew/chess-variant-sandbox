"""The suggestion box: who may post, who may read, what limits a post, and leaving.

Q279, from the user's inbox of 2026-09-17: *"Add a place to add ideas of game-rules or
other website ideas that could be added."*

The acceptance is four sentences and each one is a class below.

**A signed-in person posts and reads their own.** Over HTTP, because the refusal a
visitor gets is a route-table answer rather than a function's.

**Nobody but the poster and an admin reads a suggestion.** A member asking for the
maintainer's page gets the 404 an invented path gets, and `GET /api/suggestions` answers
with that person's own rows and nothing else -- so the test posts as two people and
checks each reads one.

**The limits hold with nobody watching.** A length cap, a rolling-day allowance through
`site/quota.py`, and the cap again as a check constraint, which is asked of the table
directly rather than of the module that respects it.

**The text is text.** A suggestion that is nothing but a script tag comes back byte for
byte and no page on the site writes markup, which is checked on the page source rather
than inferred from the payload.

Everything here needs a real Postgres and skips with a reason without one;
`tests/test_site.py` has the recipe for a throwaway cluster.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Iterator
from http.client import HTTPConnection
from http.cookies import SimpleCookie
from typing import Any

import page_text
import pytest
import site_store

from chess_variant_sandbox.site import admin, db, identity, quota, routes, suggestions
from chess_variant_sandbox.site import pages as site_pages
from chess_variant_sandbox.site import server as site_server
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.server import REQUEST_HEADER, REQUEST_VALUE, Server

# `serve_forever` looks for its stop flag every half second by default, so each test
# that stopped a server waited out that half second in `shutdown()`. 20 ms stops it at once.
SERVE_FAST = {"poll_interval": 0.02}

PASSWORD = "a-long-enough-one"


# ------------------------------------------------------------------------ the harness


class Answer:
    def __init__(self, status: int, body: bytes, headers: Any = None) -> None:
        self.status = status
        self.body = body
        self.headers = headers

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
        connection = HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            connection.request(method, path, body=body, headers=headers)
            found = connection.getresponse()
            answer = Answer(found.status, found.read(), found.headers)
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
    """A running site, its store, and the two things a test sets up by hand."""

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
                (
                    handle,
                    f"{handle}@example.invalid",
                    identity.hash_password(PASSWORD),
                    role,
                ),
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

    def rows(self) -> list[tuple[Any, ...]]:
        with self.store.cursor() as cursor:
            cursor.execute("select id, author, kind, body, done from suggestions")
            return list(cursor.fetchall())


@contextlib.contextmanager
def _running_site() -> Iterator[Site]:
    """The site on a port of its own, over the session's store, emptied first."""
    url, name = site_store.emptied("no database to post a suggestion into")

    def open_one() -> Any:
        connection = db.connect(url)
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {name}")
        return connection

    store = open_one()
    # Committed, so the `search_path` survives a test's rollback. `migrate.run` used to
    # commit it in passing, when every test migrated a schema of its own.
    store.commit()
    app = SiteApp(library.RULESET_DIR, library.RUNS_DIR, connect=open_one)
    server = Server(("127.0.0.1", 0), site_server.site_handler_for(app))
    threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
    try:
        yield Site(app, store, server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        store.close()


@pytest.fixture
def site() -> Iterator[Site]:
    with _running_site() as running:
        yield running


# ------------------------------------------------------------------ posting and reading


class TestPostingOne:
    def test_a_signed_in_person_posts_and_reads_it_back(self, site: Site) -> None:
        """The whole row, end to end: the form's two fields in, one row out."""
        browser = site.signed_in("ada")

        made = browser.post(
            "/api/suggest",
            {"kind": "rule", "body": "a piece that swaps places with the one it takes"},
        )
        assert made.status == 200, made.text
        posted = made.payload["suggestion"]
        assert posted["kind"] == "rule"
        assert posted["label"] == suggestions.LABELS["rule"]
        assert posted["state"] == "open"
        assert posted["body"].startswith("a piece that swaps")

        mine = browser.get("/api/suggestions").payload
        assert [one["body"] for one in mine["suggestions"]] == [posted["body"]]
        assert mine["limits"]["left"] == quota.SUGGESTIONS_A_DAY - 1

    def test_both_kinds_are_offered_and_a_third_is_refused(self, site: Site) -> None:
        """Two kinds, and the column refuses anything the form could not have sent."""
        browser = site.signed_in("ada")
        assert [kind["kind"] for kind in browser.get("/api/suggestions").payload["kinds"]] == [
            "rule",
            "site",
        ]
        for kind in suggestions.KINDS:
            assert browser.post("/api/suggest", {"kind": kind, "body": "x"}).status == 200

        refused = browser.post("/api/suggest", {"kind": "bug", "body": "x"})
        assert refused.status == 400
        assert "'bug'" in refused.payload["error"]

    def test_an_empty_suggestion_is_refused(self, site: Site) -> None:
        browser = site.signed_in("ada")
        refused = browser.post("/api/suggest", {"kind": "site", "body": "   "})
        assert refused.status == 400
        assert "says nothing" in refused.payload["error"]

    def test_posting_sends_no_mail_and_writes_no_file(self, site: Site) -> None:
        """The box is for people. Nothing reads it into a queue and nobody is notified.

        The app is built with no sender at all here, so a path that tried to mail would
        raise rather than pass quietly; what is asserted is that the post is a row and
        the row is all of it.
        """
        browser = site.signed_in("ada")
        assert browser.post("/api/suggest", {"kind": "site", "body": "dark mode"}).status == 200
        assert len(site.rows()) == 1


class TestWhoReadsIt:
    def test_a_visitor_is_refused_and_told_to_sign_in(self, site: Site) -> None:
        """Both ways round: posting and reading name a person, so both need one."""
        browser = site.browser()

        refused = browser.post("/api/suggest", {"kind": "site", "body": "hello"})
        assert refused.status == 400
        assert "Sign in" in refused.payload["error"]

        looking = browser.get("/api/suggestions")
        assert looking.status == 400
        assert "Sign in" in looking.payload["error"]

        assert site.rows() == []

    def test_a_visitor_asking_for_the_page_is_sent_to_the_door(self, site: Site) -> None:
        """A profile's answer, for the same reason: they asked to write something down,
        and being sent to the door is the answer to that rather than a refusal."""
        answer = site.browser().get("/suggestions")
        assert answer.status == 303
        assert answer.headers["Location"] == "/signin"

    def test_one_person_never_reads_another_persons(self, site: Site) -> None:
        ada = site.signed_in("ada")
        grace = site.signed_in("grace")
        ada.post("/api/suggest", {"kind": "rule", "body": "ada's idea"})
        grace.post("/api/suggest", {"kind": "site", "body": "grace's idea"})

        assert [one["body"] for one in ada.get("/api/suggestions").payload["suggestions"]] == [
            "ada's idea"
        ]
        assert [one["body"] for one in grace.get("/api/suggestions").payload["suggestions"]] == [
            "grace's idea"
        ]

    def test_a_member_cannot_reach_the_maintainers_page_at_all(self, site: Site) -> None:
        """The box is behind the admin gate that already exists, not beside it.

        A member gets the answer an invented path gets, from the same message, so the
        refusal does not tell them the whole box is somewhere.
        """
        member = site.signed_in("ada")
        member.post("/api/suggest", {"kind": "rule", "body": "ada's idea"})

        refused = member.get("/api/admin")
        invented = member.get("/api/nothing-here")
        assert refused.status == invented.status == 404
        assert "ada's idea" not in refused.text

    def test_an_admin_reads_every_suggestion_with_who_wrote_it(self, site: Site) -> None:
        site.signed_in("ada").post("/api/suggest", {"kind": "rule", "body": "ada's idea"})
        site.signed_in("grace").post("/api/suggest", {"kind": "site", "body": "grace's idea"})
        boss = site.signed_in("boss", role=admin.ADMIN)

        box = boss.get("/api/admin").payload["suggestions"]
        assert {(one["handle"], one["body"]) for one in box} == {
            ("ada", "ada's idea"),
            ("grace", "grace's idea"),
        }
        assert {one["state"] for one in box} == {"open"}


class TestTheMaintainerClearsOne:
    def test_marking_one_done_is_reversible_and_keeps_the_text(self, site: Site) -> None:
        ada = site.signed_in("ada")
        ada.post("/api/suggest", {"kind": "rule", "body": "ada's idea"})
        boss = site.signed_in("boss", role=admin.ADMIN)
        one = boss.get("/api/admin").payload["suggestions"][0]

        done = boss.post("/api/admin", {"act": "done-suggestion", "suggestion": one["id"]})
        assert done.status == 200
        assert "done" in done.payload["said"]
        after = boss.get("/api/admin").payload["suggestions"][0]
        assert after["state"] == "done" and after["body"] == "ada's idea"

        # The poster sees the mark too, which is the only word they get back.
        assert ada.get("/api/suggestions").payload["suggestions"][0]["state"] == "done"

        again = boss.post("/api/admin", {"act": "open-suggestion", "suggestion": one["id"]})
        assert again.status == 200
        assert boss.get("/api/admin").payload["suggestions"][0]["state"] == "open"

    def test_deleting_one_takes_the_text_away_for_good(self, site: Site) -> None:
        """A real delete, because nothing in the store references a suggestion.

        That is the difference from a game, whose row has to stay behind a null spec
        (`0010_deleted.sql`): no match, rating, vote or badge hangs off this table, so no
        count moves when a row goes.
        """
        ada = site.signed_in("ada")
        ada.post("/api/suggest", {"kind": "rule", "body": "ada's idea"})
        boss = site.signed_in("boss", role=admin.ADMIN)
        one = boss.get("/api/admin").payload["suggestions"][0]

        gone = boss.post("/api/admin", {"act": "drop-suggestion", "suggestion": one["id"]})
        assert gone.status == 200
        assert "ada" in gone.payload["said"]

        assert site.rows() == []
        assert boss.get("/api/admin").payload["suggestions"] == []
        assert ada.get("/api/suggestions").payload["suggestions"] == []

    def test_a_member_cannot_clear_anything(self, site: Site) -> None:
        ada = site.signed_in("ada")
        ada.post("/api/suggest", {"kind": "rule", "body": "ada's idea"})
        with site.store.cursor() as cursor:
            cursor.execute("select id from suggestions")
            which = int(cursor.fetchone()[0])

        for act in ("done-suggestion", "drop-suggestion"):
            refused = ada.post("/api/admin", {"act": act, "suggestion": which})
            assert refused.status == 404, act
        assert len(site.rows()) == 1


# ------------------------------------------------------------------------- the limits


class TestTheLimits:
    def test_a_suggestion_longer_than_the_cap_is_refused_in_a_sentence(
        self, site: Site
    ) -> None:
        browser = site.signed_in("ada")
        refused = browser.post(
            "/api/suggest", {"kind": "site", "body": "x" * (quota.SUGGESTION_LENGTH + 1)}
        )
        assert refused.status == 400
        assert str(quota.SUGGESTION_LENGTH) in refused.payload["error"]
        assert site.rows() == []

        # One character under is fine, so the refusal is the cap and not a fear of length.
        assert browser.post(
            "/api/suggest", {"kind": "site", "body": "x" * quota.SUGGESTION_LENGTH}
        ).status == 200

    def test_the_cap_is_the_columns_as_well_as_the_modules(self, site: Site) -> None:
        """Asked of the table directly, because a cap one writer respects is not a cap.

        `site/suggestions.py` is the only writer today. The check constraint is what
        makes the limit true of anything that ever writes here.
        """
        author = site.person("ada")
        with (
            pytest.raises(Exception) as error,  # noqa: PT011 - the driver's own class
            site.store.cursor() as cursor,
        ):
            cursor.execute(
                "insert into suggestions (author, kind, body) values (%s, %s, %s)",
                (author, "site", "x" * (quota.SUGGESTION_LENGTH + 1)),
            )
        site.store.rollback()
        assert "suggestions_body_check" in str(error.value)

    def test_the_daily_allowance_refuses_the_next_one(self, site: Site) -> None:
        """The rolling-day limit, through the site's own quota module.

        The count is over the table rather than in memory, so it survives a restart --
        which is the difference between this and `quota.Presses`, whose allowance is a
        press and not a row.
        """
        browser = site.signed_in("ada")
        for number in range(quota.SUGGESTIONS_A_DAY):
            answer = browser.post("/api/suggest", {"kind": "site", "body": f"idea {number}"})
            assert answer.status == 200, answer.text

        refused = browser.post("/api/suggest", {"kind": "site", "body": "one too many"})
        assert refused.status == 400
        assert str(quota.SUGGESTIONS_A_DAY) in refused.payload["error"]
        assert "rolls" in refused.payload["error"]
        assert len(site.rows()) == quota.SUGGESTIONS_A_DAY
        assert browser.get("/api/suggestions").payload["limits"]["left"] == 0

    def test_the_allowance_is_per_person(self, site: Site) -> None:
        ada = site.signed_in("ada")
        for number in range(quota.SUGGESTIONS_A_DAY):
            ada.post("/api/suggest", {"kind": "site", "body": f"idea {number}"})
        grace = site.signed_in("grace")
        assert grace.post("/api/suggest", {"kind": "rule", "body": "mine"}).status == 200

    def test_the_numbers_are_on_the_maintainers_page(self, site: Site) -> None:
        """`site/quota.py` is the one place a limit is written down, so it is the one
        place the page reads."""
        boss = site.signed_in("boss", role=admin.ADMIN)
        labels = {row["label"]: row["value"] for row in boss.get("/api/admin").payload["limits"]}
        assert labels["Suggestions one person may post a day"] == str(quota.SUGGESTIONS_A_DAY)
        assert labels["Characters one suggestion may be"] == str(quota.SUGGESTION_LENGTH)


# --------------------------------------------------------------------- text, not markup


class TestTheTextIsText:
    RAW = "<script>alert('q279')</script> & <b>bold</b>"

    def test_a_script_tag_survives_the_round_trip_untouched(self, site: Site) -> None:
        """Stored as typed and handed back as typed. There is no escaping step because
        there is no markup step: every page sets `textContent`."""
        ada = site.signed_in("ada")
        made = ada.post("/api/suggest", {"kind": "site", "body": self.RAW})
        assert made.status == 200
        assert made.payload["suggestion"]["body"] == self.RAW
        assert ada.get("/api/suggestions").payload["suggestions"][0]["body"] == self.RAW

        boss = site.signed_in("boss", role=admin.ADMIN)
        assert boss.get("/api/admin").payload["suggestions"][0]["body"] == self.RAW

    def test_neither_page_that_draws_one_writes_markup(self) -> None:
        """The structural half, which no payload can prove: `pages.py` has no
        `innerHTML` in it, so a suggestion cannot become markup anywhere along the path
        (§6, and the rule `web/changelog.py` already follows)."""
        for page in (page_text.inlined(site_pages.suggestions("ada")), page_text.inlined(site_pages.admin("boss"))):
            assert "innerHTML" not in page
            assert "textContent" in page

    def test_the_page_never_holds_a_suggestion_of_its_own(self) -> None:
        """The markup is a constant and the rows arrive by fetch, so nothing anybody
        wrote is ever rendered by the server."""
        assert "<textarea" in page_text.inlined(site_pages.suggestions("ada"))
        assert "/api/suggest" in page_text.inlined(site_pages.suggestions("ada"))


# ------------------------------------------------------------------------- leaving


class TestSomebodyWhoLeaves:
    def test_a_departed_persons_suggestion_stays_and_reads_as_a_former_player(
        self, site: Site
    ) -> None:
        """The rule every other row of theirs follows (`0005_leaving.sql`).

        Leaving empties the person out of their rows rather than removing them, so the
        idea stays and the name goes -- the same `coalesce` a match, a badge and the
        admin's people table already read through. A suggestion is about the site rather
        than about its author, and the maintainer may already have acted on it.
        """
        ada = site.signed_in("ada")
        ada.post("/api/suggest", {"kind": "rule", "body": "ada's idea"})
        boss = site.signed_in("boss", role=admin.ADMIN)

        with site.store.cursor():
            admin.forget(site.store, "ada")
        site.store.commit()

        box = boss.get("/api/admin").payload["suggestions"]
        assert len(box) == 1
        assert box[0]["body"] == "ada's idea"
        assert box[0]["handle"] == identity.FORMER

    def test_a_killed_account_keeps_its_suggestions_too(self, site: Site) -> None:
        """Killing is a door shut, never a record removed -- `kill_user`'s own rule."""
        who = site.person("ada")
        with site.store.cursor() as cursor:
            cursor.execute(
                "insert into suggestions (author, kind, body) values (%s, %s, %s)",
                (who, "site", "ada's idea"),
            )
        site.store.commit()
        admin.kill_user(site.store, who)
        site.store.commit()

        boss = site.signed_in("boss", role=admin.ADMIN)
        box = boss.get("/api/admin").payload["suggestions"]
        assert [one["body"] for one in box] == ["ada's idea"]
        assert box[0]["handle"] == "ada"


# ------------------------------------------------------------------- the way to the box


class TestTheWayToIt:
    def test_the_page_and_the_two_routes_are_in_the_one_table(self) -> None:
        """`site/routes.py` is what the site serves, and a route not in it is not one."""
        assert "/suggestions" in routes.PAGES
        assert "/suggestions" in routes.PERSONAL
        assert routes.find("POST", "/api/suggest") is not None
        assert routes.find("GET", "/api/suggestions") is not None
        # Neither is an admin route: the box itself is behind `/api/admin`, which is.
        assert "/api/suggest" not in routes.ADMIN_ONLY
        assert "/api/suggestions" not in routes.ADMIN_ONLY

    def test_every_footer_offers_it(self) -> None:
        """One link on every page of the site's own, beside players and credits."""
        for page in (
            page_text.inlined(site_pages.profile("ada")),
            page_text.inlined(site_pages.players("ada")),
            site_pages.privacy(),
        ):
            assert '<a href="/suggestions">suggestions</a>' in page
        # And not on the page you are already on.
        assert '<a href="/suggestions">suggestions</a>' not in page_text.inlined(site_pages.suggestions("ada"))

    def test_the_application_offers_it_only_where_there_are_accounts(self) -> None:
        """The local application has no store and serves no such page, so its footer
        must not offer a link that 404s. `ACCOUNTS` is the gate, as it is for players."""
        from page_text import PAGE

        assert "link('/suggestions', 'suggestions')" in PAGE
        gated = PAGE[PAGE.index("if (ACCOUNTS) {", PAGE.index("function paintIdentity")) :]
        assert "link('/suggestions', 'suggestions')" in gated[: gated.index("}")]

    def test_the_privacy_notice_says_what_a_suggestion_keeps(self) -> None:
        page = site_pages.privacy()
        assert "six things and nothing else" in page
        assert "Anything you post in the suggestion box" in page
        assert "read by you and by the person who runs the site and by nobody else" in page
