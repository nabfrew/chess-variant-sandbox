"""The admin page, the two kill switches, leaving, and the numbers that ration the box.

Task 1.8 of `docs/plan-website.md`. The row's own acceptance is two sentences -- *a
killed user cannot sign in and a killed game is not listed* -- and both are asked over
HTTP here, because both are things somebody does to a running site rather than functions
that return the right thing.

Three claims beyond that acceptance are worth holding.

**A member cannot tell the admin page exists.** The refusal is the 404 an invented path
gets, from the same message, so the test compares the two answers rather than checking a
status code on its own.

**Leaving is exactly what `/privacy` says and nothing else.** The notice is prose in
`pages.py`; the test reads it and then checks the store against it -- the email gone, the
hash gone, the sessions gone, the name reading *a former player*, and every match still
there with its result unchanged.

**The rationed numbers are in one module.** `site/quota.py` or a test fails, because the
whole value of gathering them was that the next person changing one has one place to look.

Everything needs a real Postgres and skips with a reason without one; `tests/test_site.py`
has the recipe for a throwaway cluster.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import threading
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection
from http.cookies import SimpleCookie
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import page_text
import pytest
import site_store

from chess_variant_sandbox import cli
from chess_variant_sandbox.site import (
    admin,
    awards,
    browse,
    db,
    drafts,
    identity,
    inviting,
    mail,
    matches,
    migrate,
    quota,
    routes,
    screening,
    unbeaten,
    votes,
)
from chess_variant_sandbox.site import load as loader
from chess_variant_sandbox.site import pages as site_pages
from chess_variant_sandbox.site import server as site_server
from chess_variant_sandbox.site import terms as site_terms
from chess_variant_sandbox.site.db import connect
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.spec import relate, spec_hash
from chess_variant_sandbox.web import designing, library
from chess_variant_sandbox.web.server import REQUEST_HEADER, REQUEST_VALUE, Server

# `serve_forever` looks for its stop flag every half second by default, so each test
# that stopped a server waited out that half second in `shutdown()`. 20 ms stops it at once.
SERVE_FAST = {"poll_interval": 0.02}


REPOSITORY = Path(__file__).resolve().parents[1]
DULL = json.loads((library.RULESET_DIR / "dull.json").read_text(encoding="utf-8"))

AGENT = "minimax:nodes=400"
STRONGEST = f"minimax:nodes={quota.STRONG_NODES}"
PASSWORD = "a-long-enough-one"
ADMIN_PASSWORD = "a-made-up-one-for-the-tests"
"""What `ADMIN_PASSWORD` is set to here, and it is this file's own invention. Nothing in
these tests reads the box's environment: the fixture clears both keys."""

SITE_URL = "https://chess-variant-sandbox.example"


DULL_HASH = spec_hash(DULL)
"""`dull` as the library lists it, which is the point: the per-game kill switch is about
a game the site is offering, so the test's game has to be one of those rather than a spec
invented here that no listing would have carried anyway."""


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
        connection = HTTPConnection("127.0.0.1", self.port, timeout=30)
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


class Sent:
    """A sender that keeps each message instead of posting it. No mail leaves this machine."""

    def __init__(self) -> None:
        self.messages: list[mail.Message] = []

    def __call__(self, message: mail.Message) -> None:
        self.messages.append(message)


class Site:
    """A running site, its store, and the few things a test has to set up by hand."""

    def __init__(
        self, app: SiteApp, store: Any, port: int, sent: Sent, connect: Any
    ) -> None:
        self.app = app
        self.store = store
        self.port = port
        self.sent = sent
        self.connect = connect

    def browser(self) -> Browser:
        return Browser(self.port)

    def person(self, handle: str, role: str = admin.MEMBER) -> int:
        """One account with a password, made the way an accepted invitation makes one."""
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

    def game(self, digest: str = DULL_HASH, spec: dict[str, Any] | None = None) -> str:
        body = spec or DULL
        with self.store.cursor() as cursor:
            cursor.execute(
                "insert into games (hash, spec, name, origin)"
                " values (%s, %s::jsonb, %s, %s)",
                (digest, json.dumps(body), body["name"], "corpus"),
            )
        self.store.commit()
        return digest

    def listed(self) -> set[str]:
        """Which hashes `/api/library` is offering right now."""
        payload = self.browser().get("/api/library").payload
        return {str(game["hash"]) for game in payload["games"]}

    def count(self, table: str, where: str = "true") -> int:
        with self.store.cursor() as cursor:
            cursor.execute(f"select count(*) from {table} where {where}")  # noqa: S608
            return int(cursor.fetchone()[0])


@contextlib.contextmanager
def _running_site(root: Path | None = None) -> Iterator[Site]:
    """The site on a port of its own, over the session's migrated store (`tests/site_store.py`), emptied first.

    With `root`, the library is `root`'s own `rulesets/`, `runs/` and `archive/`, so a
    save lands there rather than in the checkout.
    """
    url, name = site_store.emptied("no database to administer")

    def open_one() -> Any:
        connection = connect(url)
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {name}")
        return connection

    store = open_one()
    # Committed, so the `search_path` survives a test's rollback. `migrate.run` used to
    # commit it in passing, when every test migrated a schema of its own.
    store.commit()
    sent = Sent()
    if root is None:
        app = SiteApp(library.RULESET_DIR, library.RUNS_DIR, connect=open_one, send=sent)
    else:
        app = SiteApp(
            root / "rulesets",
            root / "runs",
            archive_dir=root / "archive",
            connect=open_one,
            send=sent,
        )
    server = Server(("127.0.0.1", 0), site_server.site_handler_for(app))
    threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
    try:
        yield Site(app, store, server.server_port, sent, open_one)
    finally:
        server.shutdown()
        server.server_close()
        store.close()


class _NoSocket:
    """A server that never bound. `serve` hands it a handler class and stops it."""

    server_port = 0
    RequestHandlerClass: Any = None

    def serve_forever(self) -> None:
        raise KeyboardInterrupt

    def server_close(self) -> None:
        pass


class _NoSampler:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _serve_without_a_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    """Everything `serve` does besides read its environment and print, taken out.

    What is left is the start-up path: the database url, the maintenance account's check
    and the four lines. A test of those needs no port and no store.
    """
    app = SimpleNamespace(usage=None, _connection=None, checks=_NoSampler())
    monkeypatch.setattr(site_server, "SiteApp", lambda *a, **k: app)
    monkeypatch.setattr(site_server, "listen", lambda *a, **k: _NoSocket())
    monkeypatch.setattr(site_server, "site_handler_for", lambda _app: object())
    monkeypatch.setattr(site_server, "warm", lambda _app: None)
    monkeypatch.setattr(site_server.usage, "Sampler", _NoSampler)


@pytest.fixture
def site(monkeypatch: pytest.MonkeyPatch) -> Iterator[Site]:
    # No key, so even a path that reached the live sender would stop before the network.
    monkeypatch.setenv(mail.URL_VAR, SITE_URL)
    monkeypatch.delenv(mail.KEY_VAR, raising=False)
    # No maintenance account either, unless the test asks for one: whatever this machine
    # has in its own environment is the box's business and never a test's.
    monkeypatch.delenv(admin.NAME_VAR, raising=False)
    monkeypatch.delenv(admin.PASSWORD_VAR, raising=False)
    with _running_site() as running:
        yield running


@pytest.fixture
def corpus_listed(site: Site, request: pytest.FixtureRequest) -> None:
    """For a test that lists the whole committed corpus through `Browser`, which has a 30 s timeout.

    The process's first listing pays the smoke gate and the Fairy-Stockfish translation
    for every ruleset: 13 s on an xdist worker's one core with no load (Q415). That is too
    close to the socket's limit on a shared machine. `conftest.warm_library` pays it
    here instead. It is asked for by name, after `site` has found a database, so a run
    without one skips before paying anything.
    """
    request.getfixturevalue("warm_library")


# --------------------------------------------------------------------- who is an admin


class TestWhoIsAnAdmin:
    """One column, one command, and no way in from the page itself."""

    def test_the_command_promotes_by_handle_and_puts_them_back(self, site: Site) -> None:
        user = site.person("ada")
        assert not admin.is_admin(site.store, user)

        assert admin.set_role(site.store, "ADA", admin.ADMIN) == "ada"
        site.store.commit()
        assert admin.is_admin(site.store, user)

        admin.set_role(site.store, "ada", admin.MEMBER)
        site.store.commit()
        assert not admin.is_admin(site.store, user)

    def test_a_handle_nobody_has_is_refused_by_name(self, site: Site) -> None:
        with pytest.raises(admin.AdminError, match="nobody"):
            admin.set_role(site.store, "nobody")

    def test_a_killed_admin_is_not_one(self, site: Site) -> None:
        """`is_admin` reads `revoked` too, so shutting somebody out takes the page with
        it rather than leaving a maintainer who cannot sign in but could still act."""
        user = site.person("boss", admin.ADMIN)
        admin.kill_user(site.store, user)
        site.store.commit()
        assert not admin.is_admin(site.store, user)

    def test_a_member_cannot_tell_the_page_is_there(self, site: Site) -> None:
        """The refusal is the 404 an invented path gets, with the same sentence in it: for
        the page, the not-found page (V7), byte for byte what an invented address gets."""
        browser = site.signed_in("cass")
        invented = browser.get("/api/nowhere")
        lost = browser.get("/nowhere")
        page = browser.get("/admin")
        api = browser.get("/api/admin")
        acting = browser.post("/api/admin", {"act": "revoke-sessions"})

        assert invented.status == 404
        assert page.status == 404
        assert api.status == 404
        assert acting.status == 404
        assert lost.status == 404
        assert page.text == lost.text and "No page here" in page.text
        assert api.payload["error"] == "no such endpoint: /api/admin"

    def test_a_visitor_is_sent_to_the_door_rather_than_refused(self, site: Site) -> None:
        """Somebody with no cookie may well be an admin who is signed out, so `/admin`
        answers them the way `/profile` does."""
        answer = site.browser().get("/admin")
        assert answer.status == 303

    def test_an_admin_gets_the_page_and_the_listing(self, site: Site) -> None:
        browser = site.signed_in("boss", admin.ADMIN)
        site.game()

        page = browser.get("/admin")
        assert page.status == 200
        assert "Maintenance" in page.text
        assert "innerHTML" not in page_text.inlined(page.text)

        payload = browser.get("/api/admin").payload
        assert [person["handle"] for person in payload["users"]] == ["boss"]
        assert [game["hash"] for game in payload["games"]] == [DULL_HASH]
        assert payload["limits"], "the page has no numbers to show"

    # Q744: it lists the committed corpus and the home page through `Browser`'s 30 s
    # socket, so the process's cold first listing is paid in setup (`corpus_listed`); it
    # timed out in the call under eight workers and coverage (plan-tests.md §3).
    @pytest.mark.usefixtures("corpus_listed")
    def test_an_admin_is_sent_the_refreshers_status_and_a_visitor_never(
        self, site: Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q705: `/api/admin` carries `web/refreshing.py`'s view of the status file; no
        file reads *not running*, never an error; nobody else is sent a word of it."""
        from chess_variant_sandbox import refresher

        runs = tmp_path / "runs"
        runs.mkdir()
        monkeypatch.setattr(site.app, "runs_dir", runs)
        boss = site.signed_in("boss", admin.ADMIN)
        answer = boss.get("/api/admin")
        assert answer.status == 200
        assert answer.payload["refresher"]["state"] == "none"
        assert "not running" in answer.payload["refresher"]["said"][0]

        refresher.Ledger(runs).update(
            state="working", owed_counts={"replay": 2},
            doing={"hash": DULL_HASH, "name": "dull", "needs": "replay"})
        status = boss.get("/api/admin").payload["refresher"]
        assert status["running"] is True and status["doing"]["name"] == "dull"

        member = site.signed_in("cass")
        visitor = site.browser()
        for who in (member, visitor):
            assert who.get("/api/admin").status == 404
            assert who.get("/api/scoring").status in (401, 403, 404)
            assert "refresher" not in who.get("/api/library").text
            assert "refresher-said" not in who.get("/").text

    # Q744: it lists the committed corpus and the home page through `Browser`'s 30 s
    # socket, so the process's cold first listing is paid in setup (`corpus_listed`); it
    # timed out in the call under eight workers and coverage (plan-tests.md §3).
    @pytest.mark.usefixtures("corpus_listed")
    def test_an_admin_is_sent_how_much_scores_moved_and_a_visitor_never(
        self, site: Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """EV6 (Q690): `/api/admin` carries `history.volatility` for each game with a
        stored measurement, in words beside its interval; nobody else is sent any of it,
        and a game page's payload carries none of it."""
        from chess_variant_sandbox import history

        runs = tmp_path / "runs"
        (runs / DULL_HASH).mkdir(parents=True)
        (runs / DULL_HASH / "canonical.json").write_text(json.dumps(
            {"index": 0.42, "ci_low": 0.40, "ci_high": 0.44, "weights_version": "v12"}),
            encoding="utf-8")
        history.append(DULL_HASH, runs, history.entry_for(
            1, {"index": 0.40, "runs": [{}], "weights_version": "v12", "basis": None},
            None, "2026-09-01", "replay", ("why",)))
        monkeypatch.setattr(site.app, "runs_dir", runs)

        boss = site.signed_in("boss", admin.ADMIN)
        [row] = boss.get("/api/admin").payload["volatility"]
        assert row["hash"] == DULL_HASH and row["revisions"] == 2
        assert row["spread"] == pytest.approx(0.02)
        assert row["said"].startswith("Moved 0.020 across its last 2 scores at weights v12")
        assert "±0.020" in row["said"]
        assert 'id="score-volatility"' in boss.get("/admin").text

        member = site.signed_in("cass")
        visitor = site.browser()
        for who in (member, visitor):
            assert who.get("/api/admin").status == 404
            assert "volatility" not in who.get("/api/library").text
            assert "volatility" not in who.get(f"/api/game/{DULL_HASH}").text
            assert "score-volatility" not in who.get("/").text

    def test_the_page_counts_what_hangs_off_each_person(self, site: Site) -> None:
        browser = site.signed_in("boss", admin.ADMIN)
        user = site.person("dana")
        identity.invite(site.store, "eve@example.invalid", issuer=user)
        site.store.commit()

        found = {
            person["handle"]: person
            for person in browser.get("/api/admin").payload["users"]
        }
        assert found["boss"]["sessions"] == 1
        assert found["dana"]["sessions"] == 0
        assert found["dana"]["invites"] == 1
        assert found["dana"]["state"] == "active"

    def test_an_invitations_code_is_never_printed(self, site: Site) -> None:
        """A listing carrying the codes would be a way to take over an open invitation,
        and the page has no use for one."""
        browser = site.signed_in("boss", admin.ADMIN)
        made = identity.invite(site.store, "eve@example.invalid")
        site.store.commit()

        answer = browser.get("/api/admin")
        assert made.code not in answer.text
        assert answer.payload["invites"][0]["email"] == "eve@example.invalid"
        assert answer.payload["invites"][0]["state"] == "open"


# ------------------------------------------------- the maintainer's account, from the
# environment


class TestTheEnvironmentsAccount:
    """Q365. One account the box's environment gives the site, and no new door.

    `ADMIN_ACCOUNT_NAME` and `ADMIN_PASSWORD` make an ordinary `users` row with the admin
    role, so every claim below is asked over HTTP through `/api/signin` and `/admin` --
    the same route and the same gate everybody else meets. What is worth holding is the
    refusing: a wrong password, an unset key, a visitor, a member, and a name somebody
    already signs in under.
    """

    def test_the_name_is_the_key_and_the_users_word_is_its_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(admin.NAME_VAR, raising=False)
        assert admin.account_name() == "secret-admin"

        monkeypatch.setenv(admin.NAME_VAR, " keeper-of-keys ")
        assert admin.account_name() == "keeper-of-keys"

    def test_signing_in_reaches_the_panel_and_the_hash_is_what_is_stored(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(admin.PASSWORD_VAR, ADMIN_PASSWORD)
        browser = site.browser()

        answer = browser.post(
            "/api/signin", {"handle": "secret-admin", "password": ADMIN_PASSWORD}
        )
        assert answer.status == 200, answer.text
        assert answer.payload["handle"] == "secret-admin"
        assert browser.get("/api/me").payload["admin"] is True
        assert browser.get("/admin").status == 200
        assert browser.get("/api/admin").status == 200

        handle, stored, role = _account(site)
        assert (handle, role) == ("secret-admin", admin.ADMIN)
        assert stored.startswith("scrypt$")
        assert ADMIN_PASSWORD not in stored

    def test_a_wrong_password_opens_nothing(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(admin.PASSWORD_VAR, ADMIN_PASSWORD)
        browser = site.browser()

        answer = browser.post(
            "/api/signin", {"handle": "secret-admin", "password": "not-that-one"}
        )
        assert answer.status == 401
        assert browser.cookie is None
        assert browser.get("/api/admin").status == 404

    def test_with_no_password_set_the_account_cannot_sign_in(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """And the row it left behind stops working the moment the key goes, rather than
        keeping a hash that still opens the door."""
        monkeypatch.setenv(admin.PASSWORD_VAR, ADMIN_PASSWORD)
        first = site.browser()
        assert (
            first.post(
                "/api/signin", {"handle": "secret-admin", "password": ADMIN_PASSWORD}
            ).status
            == 200
        )

        monkeypatch.delenv(admin.PASSWORD_VAR)
        second = site.browser()
        answer = second.post(
            "/api/signin", {"handle": "secret-admin", "password": ADMIN_PASSWORD}
        )

        assert answer.status == 401
        assert second.cookie is None
        assert _account(site)[1] is None

    def test_no_password_and_no_row_writes_no_account(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(admin.PASSWORD_VAR, raising=False)

        answer = site.browser().post(
            "/api/signin", {"handle": "secret-admin", "password": ADMIN_PASSWORD}
        )

        assert answer.status == 401
        site.store.rollback()
        assert site.count("users") == 0

    def test_nobody_else_reaches_the_panel(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A visitor is sent to the door and a signed-in member is told there is no such
        page, exactly as they were before this account existed."""
        monkeypatch.setenv(admin.PASSWORD_VAR, ADMIN_PASSWORD)
        visitor = site.browser()
        member = site.signed_in("cass")

        assert visitor.get("/admin").status == 303
        assert visitor.get("/api/admin").status == 404
        assert member.get("/admin").status == 404
        assert member.get("/api/admin").status == 404
        assert member.get("/api/me").payload["admin"] is False

    def test_renaming_the_account_moves_the_one_row(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The row is found by its reserved address, so a changed key leaves no second
        admin behind under the old name."""
        monkeypatch.setenv(admin.PASSWORD_VAR, ADMIN_PASSWORD)
        site.browser().post(
            "/api/signin", {"handle": "secret-admin", "password": ADMIN_PASSWORD}
        )

        monkeypatch.setenv(admin.NAME_VAR, "keeper-of-keys")
        renamed = site.browser()
        answer = renamed.post(
            "/api/signin", {"handle": "keeper-of-keys", "password": ADMIN_PASSWORD}
        )

        assert answer.status == 200, answer.text
        site.store.rollback()
        assert site.count("users") == 1
        assert _account(site)[0] == "keeper-of-keys"
        assert (
            site.browser()
            .post("/api/signin", {"handle": "secret-admin", "password": ADMIN_PASSWORD})
            .status
            == 401
        )

    def test_a_handle_somebody_has_is_refused_rather_than_taken_over(
        self, site: Site, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Q369 moved where the refusal goes. It used to be a 400 naming `NAME_VAR`, sent
        to whoever signed in as that handle -- which is a stranger, and which told them
        both that there is a maintenance account and what its key is called. The clash
        now turns the account off the way a short password does, in the journal, and the
        person whose handle it is signs in as themselves.
        """
        site.person("cass")
        monkeypatch.setenv(admin.NAME_VAR, "cass")
        monkeypatch.setenv(admin.PASSWORD_VAR, ADMIN_PASSWORD)
        monkeypatch.setattr(admin, "_said_the_clash", False)

        answer = site.browser().post(
            "/api/signin", {"handle": "cass", "password": PASSWORD}
        )

        assert answer.status == 200, answer.text
        assert answer.payload["handle"] == "cass"
        assert admin.NAME_VAR in capsys.readouterr().out
        site.store.rollback()
        with site.store.cursor() as cursor:
            cursor.execute("select role from users where handle = 'cass'")
            assert cursor.fetchone()[0] == admin.MEMBER
            cursor.execute("select count(*) from users where email = %s", (admin.RESERVED_EMAIL,))
            assert cursor.fetchone()[0] == 0, "no maintenance account was made"

    def test_off_includes_the_row_an_earlier_name_wrote(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q379. The reconcile runs at every sign-in now, whichever handle it names, and a
        clash takes the password off the account's row as a missing key does.

        Before, it ran only for the configured handle and a clash left the row alone, so
        after the key moved to a held name the row written under the old one kept its
        handle and its password, and the old name went on signing in as an admin.
        """
        monkeypatch.setenv(admin.PASSWORD_VAR, ADMIN_PASSWORD)
        first = site.browser().post(
            "/api/signin", {"handle": "secret-admin", "password": ADMIN_PASSWORD}
        )
        assert first.status == 200, first.text
        site.person("cass")
        monkeypatch.setenv(admin.NAME_VAR, "cass")
        monkeypatch.setattr(admin, "_said_the_clash", False)

        old = site.browser().post(
            "/api/signin", {"handle": "secret-admin", "password": ADMIN_PASSWORD}
        )

        assert old.status == 401
        assert _account(site)[1] is None

    def test_a_password_under_the_sites_own_minimum_turns_the_account_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q365b. A password the site would refuse from a person is refused here too, and
        the answer is no maintenance account rather than no site.

        The sentence names the key and the minimum. The value is not in it, and neither
        is anything that would say how long the value is.
        """
        monkeypatch.delenv(admin.NAME_VAR, raising=False)
        monkeypatch.setenv(admin.PASSWORD_VAR, "tiny")

        assert admin.account_check() is None
        refusal = admin.account_refusal()
        assert refusal is not None
        assert admin.PASSWORD_VAR in refusal
        assert str(identity.MIN_PASSWORD) in refusal
        assert "tiny" not in refusal
        assert "\n" not in refusal

    def test_a_name_that_is_not_a_handle_turns_the_account_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(admin.NAME_VAR, "!!")
        monkeypatch.setenv(admin.PASSWORD_VAR, ADMIN_PASSWORD)

        assert admin.account_check() is None
        refusal = admin.account_refusal()
        assert refusal is not None
        assert admin.NAME_VAR in refusal
        assert ADMIN_PASSWORD not in refusal

    def test_no_password_at_all_is_not_a_refusal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The environment asking for no account is not the environment getting it wrong,
        and the start-up line is a different sentence."""
        monkeypatch.delenv(admin.NAME_VAR, raising=False)
        monkeypatch.delenv(admin.PASSWORD_VAR, raising=False)

        assert admin.account_check() is None
        assert admin.account_refusal() is None

    def test_a_refused_password_leaves_the_site_serving_and_says_why(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Q365b's first repair, at the one place that used to exit 2.

        On 2026-09-19 a deploy put a short `ADMIN_PASSWORD` on the box, `serve` refused
        before it bound, systemd gave up after five starts and the site was down. `serve`
        is driven here with its bind, its warm-up and its sampler replaced, so the test
        reaches the start-up lines without a socket or a database. What it asserts is that
        the function returns at all.
        """
        monkeypatch.setenv(db.ENV_VAR, db.SOCKET_URL)
        monkeypatch.delenv(admin.NAME_VAR, raising=False)
        monkeypatch.setenv(admin.PASSWORD_VAR, "tiny")
        _serve_without_a_socket(monkeypatch)

        site_server.serve()

        printed = capsys.readouterr().out
        assert admin.PASSWORD_VAR in printed
        assert str(identity.MIN_PASSWORD) in printed
        assert "tiny" not in printed

    def test_a_usable_password_is_said_without_the_handle(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv(db.ENV_VAR, db.SOCKET_URL)
        monkeypatch.setenv(admin.NAME_VAR, "keeper-of-keys")
        monkeypatch.setenv(admin.PASSWORD_VAR, ADMIN_PASSWORD)
        _serve_without_a_socket(monkeypatch)

        site_server.serve()

        printed = capsys.readouterr().out
        assert "can sign in" in printed
        assert "keeper-of-keys" not in printed
        assert ADMIN_PASSWORD not in printed

    def test_the_launch_page_carries_the_way_in(self, site: Site) -> None:
        """The main page's header offers `/admin` to a maintainer and to nobody else, so
        signing in as this account is the whole of getting there."""
        page = site.browser().get("/")

        assert page.status == 200
        assert "if (ADMIN) who.append(link('/admin', 'Maintenance')" in page_text.inlined(
            page.text
        )


def _account(site: Site) -> tuple[str | None, str | None, str]:
    """The environment's account as the store holds it: handle, password, role."""
    site.store.rollback()
    with site.store.cursor() as cursor:
        cursor.execute(
            "select handle, password, role from users where email = %s",
            (admin.RESERVED_EMAIL,),
        )
        found = cursor.fetchone()
    assert found is not None, "the environment's account was never written"
    return found[0], found[1], str(found[2])


# -------------------------------------------------------------- inviting from the page


class TestMembersSendInvitations:
    """Q719 (`docs/plan-launch.md` §5.3a): `/admin` gives a member a number of
    invitations, zero by default; the member sends them from their own page, each one
    taking one off, and is refused in a sentence at zero. The invitation is the ordinary
    one, with the member as its issuer."""

    @staticmethod
    def _give(site: Site, boss: Browser, handle: str, count: Any) -> Answer:
        return boss.post(
            "/api/admin", {"act": "set-invites", "user": _id(site, handle), "count": count}
        )

    def test_nobody_has_any_until_the_admin_gives_some(self, site: Site) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        cass = site.signed_in("cass")
        rows = {row["handle"]: row for row in boss.get("/api/admin").payload["users"]}
        assert rows["cass"]["toGive"] == 0
        assert "invites" in cass.get("/api/profile").payload
        assert cass.get("/api/profile").payload["invites"] is None

        answer = self._give(site, boss, "cass", 2)
        assert answer.status == 200, answer.text
        assert answer.payload["said"] == "cass can send 2 invitation(s) now"
        rows = {row["handle"]: row for row in boss.get("/api/admin").payload["users"]}
        assert rows["cass"]["toGive"] == 2
        box = cass.get("/api/profile").payload["invites"]
        assert (box["left"], box["said"]) == (2, "You can invite 2 more people.")

    def test_the_admin_sets_a_whole_number_in_range_and_only_for_somebody_who_can_sign_in(
        self, site: Site
    ) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        site.person("cass")
        for wrong in (-1, inviting.MOST + 1, "3", 1.5, True, None):
            answer = self._give(site, boss, "cass", wrong)
            assert answer.status == 400, wrong
        assert inviting.to_give(site.store, _id(site, "cass")) == 0
        boss.post("/api/admin", {"act": "kill-user", "user": _id(site, "cass")})
        answer = self._give(site, boss, "cass", 3)
        assert answer.status == 400
        assert "cannot sign in" in answer.payload["error"]

    def test_a_member_with_invitations_sends_one_and_the_count_goes_down(
        self, site: Site
    ) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        cass = site.signed_in("cass")
        self._give(site, boss, "cass", 2)

        answer = cass.post("/api/invite", {"email": " Eve@Example.com "})
        assert answer.status == 200, answer.text
        assert answer.payload["said"] == "Invited eve@example.com. The link is on its way."
        assert answer.payload["box"]["left"] == 1
        assert answer.payload["box"]["said"] == "You can invite one more person."
        # The ordinary invitation: the same row, link and mail, with cass as its issuer.
        [(code, issuer)] = _invitations(site, "eve@example.com")
        assert issuer == _id(site, "cass")
        assert site.sent.messages == [
            mail.invitation("eve@example.com", identity.link(SITE_URL, code), "cass")
        ]
        assert site.sent.messages[0].subject == "cass invited you to try Chess variant sandbox"
        listed = {row["email"]: row for row in boss.get("/api/admin").payload["invites"]}
        assert listed["eve@example.com"]["issuer"] == "cass"
        # And it makes an account like any other.
        joined = site.browser().post(
            "/api/join", {"code": code, "handle": "eve", "password": PASSWORD}
        )
        assert joined.status == 200, joined.text

        cass.post("/api/invite", {"email": "fay@example.com"})
        box = cass.get("/api/profile").payload["invites"]
        assert (box["left"], box["sent"], box["said"]) == (0, 2, inviting.NONE_LEFT)

    def test_at_zero_a_member_is_refused_in_a_sentence_and_nothing_is_sent(
        self, site: Site
    ) -> None:
        cass = site.signed_in("cass")
        answer = cass.post("/api/invite", {"email": "eve@example.com"})
        assert answer.status == 400
        assert answer.payload["error"] == inviting.NONE_LEFT
        assert inviting.NONE_LEFT == (
            "You have no invitations left to send. Ask the person who runs the site if "
            "you would like to invite somebody else."
        )
        assert site.sent.messages == []
        assert site.count("invites") == 0
        assert inviting.to_give(site.store, _id(site, "cass")) == 0

    def test_a_refused_address_or_a_failed_send_gives_the_invitation_back(
        self, site: Site
    ) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        cass = site.signed_in("cass")
        site.person("dora")
        self._give(site, boss, "cass", 1)

        taken = cass.post("/api/invite", {"email": "dora@example.invalid"})
        assert taken.status == 400
        # Without whose account it is: that is `/admin`'s to read, not a member's.
        assert taken.payload["error"] == inviting.ALREADY_JOINED
        assert "dora" not in taken.payload["error"].replace("dora@", "")
        malformed = cass.post("/api/invite", {"email": "not an address"})
        assert malformed.status == 400
        assert "does not look like an email address" in malformed.payload["error"]

        def refuse(message: mail.Message) -> None:
            raise mail.MailError("the sending service refused the message (422): no")

        site.app.send = refuse
        failed = cass.post("/api/invite", {"email": "eve@example.com"})
        assert failed.status == 400
        assert site.count("invites") == 0
        assert inviting.to_give(site.store, _id(site, "cass")) == 1

    def test_two_presses_at_once_spend_the_last_one_once(self, site: Site) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        cass = site.signed_in("cass")
        self._give(site, boss, "cass", 1)
        answers: list[Answer] = []
        threads = [
            threading.Thread(
                target=lambda address=address: answers.append(
                    cass.post("/api/invite", {"email": address})
                )
            )
            for address in ("eve@example.com", "fay@example.com")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sorted(answer.status for answer in answers) == [200, 400]
        assert site.count("invites") == 1
        assert inviting.to_give(site.store, _id(site, "cass")) == 0

    def test_a_visitor_is_asked_to_sign_in_and_only_an_admin_sets_the_number(
        self, site: Site
    ) -> None:
        cass = site.signed_in("cass")
        answer = site.browser().post("/api/invite", {"email": "eve@example.com"})
        assert answer.status == 400
        assert answer.payload["error"] == "Sign in to invite somebody."
        for who in (cass, site.browser()):
            given = self._give(site, who, "cass", 5)
            assert given.status == 404
        assert inviting.to_give(site.store, _id(site, "cass")) == 0
        assert site.sent.messages == []

    def test_leaving_clears_the_number(self, site: Site) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        cass = site.person("cass")
        self._give(site, boss, "cass", 3)
        admin.forget(site.store, "cass")
        site.store.commit()
        assert inviting.to_give(site.store, cass) == 0

    def test_the_pages_carry_the_box_and_the_form(self, site: Site) -> None:
        page = page_text.inlined(site.signed_in("boss", admin.ADMIN).get("/admin").text)
        assert "act: 'set-invites'" in page and "<th>can invite</th>" in page
        mine = page_text.inlined(site.signed_in("cass").get("/profile").text)
        assert 'id="invite"' in mine
        assert "post('/api/invite', { email: field.value })" in mine
        assert "innerHTML" not in mine[mine.index("function drawInvite"):][:2500]


class TestInvitingFromThePage:
    """Q261. The form on `/admin` and `chess-variant-sandbox site invite` call one function,
    `identity.send_invitation`, so there is one way to invite and one set of refusals."""

    def test_the_form_writes_the_invitation_and_mails_its_link(self, site: Site) -> None:
        boss = site.signed_in("boss", admin.ADMIN)

        answer = boss.post("/api/admin", {"act": "invite", "email": " Eve@Example.com "})
        assert answer.status == 200, answer.text
        assert answer.payload["said"] == "invited eve@example.com: the link is on its way"

        [(code, issuer)] = _invitations(site, "eve@example.com")
        assert issuer == _id(site, "boss")
        assert site.sent.messages == [
            mail.invitation("eve@example.com", identity.link(SITE_URL, code), "boss")
        ]
        # The admin who pressed the button is who the invitation names (L8, Q567).
        assert site.sent.messages[0].subject == "boss invited you to try Chess variant sandbox"

    def test_the_list_says_to_whom_when_and_whether_it_was_accepted(
        self, site: Site
    ) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        boss.post("/api/admin", {"act": "invite", "email": "eve@example.com"})
        identity.invite(site.store, "gus@example.com", lifetime=timedelta(days=-1))
        site.store.commit()

        listed = {row["email"]: row for row in boss.get("/api/admin").payload["invites"]}
        eve = listed["eve@example.com"]
        assert eve["issuer"] == "boss"
        sent_at = datetime.fromisoformat(eve["created"])
        assert abs(datetime.now(timezone.utc) - sent_at) < timedelta(minutes=5)
        assert (eve["accepted"], eve["state"]) == (False, "open")
        assert (listed["gus@example.com"]["accepted"], listed["gus@example.com"]["state"]) == (
            False,
            "expired",
        )

        [(code, _)] = _invitations(site, "eve@example.com")
        joined = site.browser().post(
            "/api/join", {"code": code, "handle": "eve", "password": PASSWORD}
        )
        assert joined.status == 200, joined.text
        listed = {row["email"]: row for row in boss.get("/api/admin").payload["invites"]}
        assert (listed["eve@example.com"]["accepted"], listed["eve@example.com"]["state"]) == (
            True,
            "accepted",
        )

    def test_an_address_that_has_an_account_is_refused_in_a_sentence(
        self, site: Site
    ) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        site.person("ada")

        answer = boss.post("/api/admin", {"act": "invite", "email": "ADA@example.invalid"})
        assert answer.status == 400
        assert answer.payload["error"] == (
            "ada@example.invalid already has an account (ada), so there is nobody to "
            "invite; they can sign in"
        )
        assert site.sent.messages == []
        assert site.count("invites") == 0

    def test_a_shut_out_account_is_pointed_at_letting_them_back_in(self, site: Site) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        admin.kill_user(site.store, site.person("mallory"))
        site.store.commit()

        answer = boss.post(
            "/api/admin", {"act": "invite", "email": "mallory@example.invalid"}
        )
        assert answer.status == 400
        assert "let them back in" in answer.payload["error"]
        assert site.sent.messages == []

    def test_somebody_who_left_can_be_invited_again(self, site: Site) -> None:
        """Leaving empties the address, so it no longer belongs to an account."""
        boss = site.signed_in("boss", admin.ADMIN)
        site.person("ada")
        admin.forget(site.store, "ada")
        site.store.commit()

        answer = boss.post("/api/admin", {"act": "invite", "email": "ada@example.invalid"})
        assert answer.status == 200, answer.text
        assert [message.to for message in site.sent.messages] == ["ada@example.invalid"]

    def test_inviting_an_address_again_mails_a_fresh_link_and_keeps_the_first(
        self, site: Site
    ) -> None:
        """A maintainer who types an address twice means *they did not get it*. Both
        links work until one of them makes the account."""
        boss = site.signed_in("boss", admin.ADMIN)
        for _ in range(2):
            again = boss.post("/api/admin", {"act": "invite", "email": "eve@example.com"})
            assert again.status == 200, again.text

        first, second = site.sent.messages
        assert first.to == second.to == "eve@example.com"
        assert first.body != second.body
        (earlier, _), (later, _) = _invitations(site, "eve@example.com")

        joined = site.browser().post(
            "/api/join", {"code": earlier, "handle": "eve", "password": PASSWORD}
        )
        assert joined.status == 200, joined.text
        late = site.browser().post(
            "/api/join", {"code": later, "handle": "eve2", "password": PASSWORD}
        )
        assert late.status == 401
        assert "already has an account" in late.payload["error"]

        states = [row["state"] for row in boss.get("/api/admin").payload["invites"]]
        assert states == ["accepted", "accepted"]

    def test_a_send_that_fails_leaves_no_invitation(self, site: Site) -> None:
        boss = site.signed_in("boss", admin.ADMIN)

        def refuse(message: mail.Message) -> None:
            raise mail.MailError("the sending service refused the message (422): no")

        site.app.send = refuse
        answer = boss.post("/api/admin", {"act": "invite", "email": "eve@example.com"})
        assert answer.status == 400
        assert "refused the message" in answer.payload["error"]
        assert site.count("invites") == 0

    def test_a_malformed_address_is_refused_by_the_same_check(self, site: Site) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        answer = boss.post("/api/admin", {"act": "invite", "email": "not an address"})
        assert answer.status == 400
        assert "does not look like an email address" in answer.payload["error"]
        assert site.sent.messages == []

    def test_only_an_admin_can_invite(self, site: Site) -> None:
        """A member and a visitor both get the 404 an invented path gets."""
        member = site.signed_in("cass")
        for who in (member, site.browser()):
            answer = who.post("/api/admin", {"act": "invite", "email": "eve@example.com"})
            assert answer.status == 404
            assert answer.payload["error"] == "no such endpoint: /api/admin"
        assert site.sent.messages == []
        assert site.count("invites") == 0

    def test_a_post_without_the_pages_header_sends_nothing(self, site: Site) -> None:
        """§6's cross-site answer holds for this act as for the others, even for an admin."""
        boss = site.signed_in("boss", admin.ADMIN)
        status = _post_without_header(
            site.port, boss.cookie, "/api/admin", {"act": "invite", "email": "eve@example.com"}
        )
        assert status == 403
        assert site.sent.messages == []
        assert site.count("invites") == 0

    def test_the_page_carries_the_form(self, site: Site) -> None:
        page = page_text.inlined(site.signed_in("boss", admin.ADMIN).get("/admin").text)
        assert 'id="invite-email"' in page
        assert "act: 'invite'" in page
        assert "innerHTML" not in page

    def test_the_page_and_the_command_call_one_function(
        self,
        site: Site,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The command runs against the same store, with its live sender swapped for the
        recorder, and both paths are seen passing through `send_invitation`."""
        called: list[str] = []
        real = identity.send_invitation

        def spy(*args: Any, **kwargs: Any) -> Any:
            called.append(str(args[1]))
            return real(*args, **kwargs)

        monkeypatch.setattr(identity, "send_invitation", spy)
        monkeypatch.setattr(db, "database_url", lambda: "postgresql://scratch")
        monkeypatch.setattr(db, "connect", lambda *_: site.connect())
        monkeypatch.setattr(mail, "send", site.sent)

        boss = site.signed_in("boss", admin.ADMIN)
        assert boss.post("/api/admin", {"act": "invite", "email": "eve@example.com"}).status == 200
        assert cli.main(["site", "invite", "fay@example.com", "--url", SITE_URL]) == 0
        assert called == ["eve@example.com", "fay@example.com"]

        [(code, _)] = _invitations(site, "fay@example.com")
        assert site.sent.messages[-1] == mail.invitation(
            "fay@example.com", identity.link(SITE_URL, code)
        )

        capsys.readouterr()
        site.person("gil")
        assert cli.main(["site", "invite", "gil@example.invalid", "--url", SITE_URL]) == 2
        assert "already has an account (gil)" in capsys.readouterr().err


# ------------------------------------------------------------------- the kill switches


class TestEarlierScores:
    """Q649: a score's history is the maintainer's until the user answers
    `docs/plan-confidence.md` §5 Q1. The admin reads every line, on `/admin` and on the
    game's page; a visitor and a member are sent none."""

    @staticmethod
    def _replaced(site: Site) -> None:
        from chess_variant_sandbox import corpus, history

        runs = site.app.runs_dir
        out = corpus.measurement_dir(DULL_HASH, runs)
        out.mkdir(parents=True, exist_ok=True)
        (out / "canonical.json").write_text(json.dumps({
            "hash": DULL_HASH, "name": DULL["name"], "index": 0.4, "runs": [],
            "basis": None,
        }), encoding="utf-8")
        history.append(DULL_HASH, runs, history.entry_for(
            1, {"index": 0.3, "runs": [{}] * 4, "weights_version": "v11",
                "basis": {"agents": ["random"], "net": ""}},
            {"agents": ["random"], "net": "p=net:abababababab"},
            "2026-09-28", "replay", ("its nets were none",)))

    def test_the_admin_reads_every_line_and_a_visitor_none(self, drafting: Site) -> None:
        drafting.game()
        self._replaced(drafting)
        boss = drafting.signed_in("boss", admin.ADMIN)

        [row] = boss.get("/api/admin").payload["history"]
        assert row["hash"] == DULL_HASH and row["name"] == DULL["name"]
        assert row["entries"][0]["index"] == 0.3
        [line] = boss.get(f"/api/game/{DULL_HASH}").payload["history"]
        assert line["index"] == 0.3 and line["moved"] == ["net"]

        for viewer in (drafting.browser(), drafting.signed_in("dana")):
            answer = viewer.get(f"/api/game/{DULL_HASH}")
            assert answer.status == 200
            assert answer.payload["history"] == []
            assert "0.3" not in json.dumps(answer.payload.get("history"))


class TestThePerUserKillSwitch:
    """The row's first acceptance, and the session that has to stop with it."""

    def test_a_killed_user_cannot_sign_in_and_their_session_stops(
        self, site: Site
    ) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        theirs = site.signed_in("mallory")
        assert theirs.get("/api/profile").status == 200

        killed = boss.post("/api/admin", {"act": "kill-user", "user": _id(site, "mallory")})
        assert killed.status == 200
        assert "mallory" in killed.payload["said"]

        # The session they were in the middle of, and the password they came with.
        assert theirs.get("/api/profile").status == 400
        again = site.browser().post(
            "/api/signin", {"handle": "mallory", "password": PASSWORD}
        )
        assert again.status == 401
        assert site.count("sessions", f"user_id = {_id(site, 'mallory')}") == 0

    def test_letting_them_back_in_takes_the_password_they_had(self, site: Site) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        user = site.person("mallory")
        boss.post("/api/admin", {"act": "kill-user", "user": user})

        back = boss.post("/api/admin", {"act": "revive-user", "user": user})
        assert back.status == 200
        answer = site.browser().post(
            "/api/signin", {"handle": "mallory", "password": PASSWORD}
        )
        assert answer.status == 200

    def test_killing_somebody_leaves_their_games_alone(self, site: Site) -> None:
        """A leaderboard with a hole in it is a leaderboard that lies about the people
        still on it, so the matches stay exactly as they were."""
        site.game()
        user = site.person("mallory")
        match = matches.start(site.store, DULL_HASH, matches.SANDBOX, ["human", AGENT], user)
        site.store.commit()

        admin.kill_user(site.store, user)
        site.store.commit()
        assert matches.row(site.store, match.id).game == DULL_HASH


class TestThePerGameKillSwitch:
    """The row's second acceptance: not listed, and not startable."""

    @pytest.mark.usefixtures("corpus_listed")
    def test_a_killed_game_is_not_listed_and_cannot_be_started(self, site: Site) -> None:
        site.game()
        assert DULL_HASH in site.listed()

        admin.kill_game(site.store, DULL_HASH)
        site.store.commit()

        assert DULL_HASH not in site.listed()
        with pytest.raises(matches.MatchError, match="taken off the site"):
            matches.start(site.store, DULL_HASH, matches.SANDBOX, ["human", AGENT])

    def test_a_killed_games_own_page_is_gone_too(self, site: Site) -> None:
        """The third door. A hash somebody bookmarked opens nothing, and the answer is
        the one an unknown hash gets rather than one that says *this was taken down*."""
        browser = site.browser()
        site.game()
        assert browser.get(f"/api/game/{DULL_HASH}").status == 200

        admin.kill_game(site.store, DULL_HASH)
        site.store.commit()

        gone = browser.get(f"/api/game/{DULL_HASH}")
        assert gone.status == 404
        assert DULL_HASH in gone.payload["error"]

    def test_starting_it_over_http_is_refused_in_a_sentence(self, site: Site) -> None:
        browser = site.signed_in("ada")
        site.game()
        admin.kill_game(site.store, DULL_HASH)
        site.store.commit()

        answer = browser.post(
            "/api/session", {"hash": DULL_HASH, "seats": ["human", AGENT]}
        )
        assert answer.status == 400
        assert "cannot be played" in answer.payload["error"]

    def test_the_matches_already_played_on_it_stay(self, site: Site) -> None:
        site.game()
        user = site.person("ada")
        match = matches.start(site.store, DULL_HASH, matches.SANDBOX, ["human", AGENT], user)
        site.store.commit()

        admin.kill_game(site.store, DULL_HASH)
        site.store.commit()
        assert matches.row(site.store, match.id).id == match.id

    @pytest.mark.usefixtures("corpus_listed")
    def test_the_admin_page_is_where_a_killed_game_is_still_visible(
        self, site: Site
    ) -> None:
        """The one listing that shows it, because otherwise there is no way back."""
        browser = site.signed_in("boss", admin.ADMIN)
        site.game()
        browser.post("/api/admin", {"act": "kill-game", "hash": DULL_HASH})

        listed = browser.get("/api/admin").payload["games"]
        assert [game["state"] for game in listed] == ["killed"]

        back = browser.post("/api/admin", {"act": "revive-game", "hash": DULL_HASH})
        assert back.status == 200
        assert DULL_HASH in site.listed()

    def test_a_hash_the_store_has_never_heard_of_is_refused_by_name(
        self, site: Site
    ) -> None:
        with pytest.raises(admin.AdminError, match="no game with the hash"):
            admin.kill_game(site.store, "not-a-hash")


@pytest.fixture
def drafting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Site]:
    """A running site over a library of its own: `dull` committed, nothing saved yet."""
    monkeypatch.setenv(mail.URL_VAR, SITE_URL)
    monkeypatch.delenv(mail.KEY_VAR, raising=False)
    for name in ("rulesets", "runs", "archive"):
        (tmp_path / name).mkdir()
    (tmp_path / "rulesets" / "dull.json").write_text(json.dumps(DULL), encoding="utf-8")
    with _running_site(tmp_path) as running:
        yield running


def _screened(
    monkeypatch: pytest.MonkeyPatch, passed: bool, reasons: tuple[str, ...] = ()
) -> None:
    """Fix the playability check's verdict. What it decides is `test_candidacy.py`'s, and
    the wall clock on it is `test_site_screening.py`'s."""
    verdict = {"passed": passed, "reasons": list(reasons), "stats": {}}
    monkeypatch.setattr(screening, "check", lambda spec: verdict)


def _template(index: int = 0, name: str = "a draft somebody kept") -> dict[str, Any]:
    spec = json.loads(json.dumps(designing.templates()[index]["spec"]))
    spec["name"] = name
    return spec


def _saved(browser: Browser, spec: dict[str, Any], parent: str = "") -> Answer:
    answer = browser.post("/api/design/save", {"spec": spec, "parent": parent})
    assert answer.status == 200, answer.text
    return answer


class TestADraftSavedOnTheSite:
    """Q273. A save writes the draft's `games` row, and that row decides what the site does
    with the draft: whether it is listed, whether a match can start, and the kill switch.
    """

    def test_a_draft_that_passes_the_check_is_unlisted_with_the_pass_recorded(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q503a: a save keeps a draft whatever the check says, and records the verdict."""
        _screened(monkeypatch, passed=True)
        ada = drafting.signed_in("ada")

        kept = _saved(ada, _template())

        digest = kept.payload["hash"]
        assert kept.payload["listed"] is False
        assert kept.payload["visibility"] == "unlisted"
        assert kept.payload["screenPassed"] is True
        assert "measured its score on your computer" in kept.payload["said"]
        with drafting.store.cursor() as cursor:
            cursor.execute(
                "select g.origin, g.visibility, g.parent, g.revoked, g.screen_passed,"
                "       g.published, u.handle"
                "  from games g join users u on u.id = g.owner where g.hash = %s",
                (digest,),
            )
            assert cursor.fetchone() == (
                "designed", "unlisted", None, None, True, None, "ada"
            )
        assert digest not in drafting.listed()
        assert drafting.browser().get(f"/api/game/{digest}").status == 200

    def test_a_draft_that_fails_the_check_is_unlisted_and_opens_by_its_link(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reason = "no computer player won any of 8 games"
        _screened(monkeypatch, passed=False, reasons=(reason,))
        ada = drafting.signed_in("ada")

        kept = _saved(ada, _template())

        digest = kept.payload["hash"]
        assert kept.payload["listed"] is False
        assert kept.payload["visibility"] == "unlisted"
        assert "did not pass the playability check" in kept.payload["said"]
        assert reason in kept.payload["said"]
        assert digest not in drafting.listed()
        assert drafting.browser().get(f"/api/game/{digest}").status == 200

    def test_a_draft_with_files_and_no_row_is_not_listed(self, drafting: Site) -> None:
        """Written straight to `runs/designed/`, the way the local application saves. The
        site cannot seat anybody at it or take it down, so it does not offer it."""
        kept = designing.save(_template(), runs_dir=drafting.app.runs_dir)

        shown = drafting.listed()
        assert kept["hash"] not in shown
        assert DULL_HASH in shown

    def test_an_admin_kills_a_draft_from_the_page_and_it_is_gone(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _screened(monkeypatch, passed=True)
        ada = drafting.signed_in("ada")
        digest = _saved(ada, _template()).payload["hash"]
        boss = drafting.signed_in("boss", admin.ADMIN)

        rows = {game["hash"]: game for game in boss.get("/api/admin").payload["games"]}
        row = rows[digest]
        assert (row["origin"], row["owner"], row["visibility"], row["state"]) == (
            "designed", "ada", "unlisted", "draft"
        )
        site_store.on_trial(drafting.store, digest)
        rows = {game["hash"]: game for game in boss.get("/api/admin").payload["games"]}
        assert (rows[digest]["visibility"], rows[digest]["state"]) == ("public", "listed")
        killed = boss.post("/api/admin", {"act": "kill-game", "hash": digest})
        assert killed.status == 200, killed.text

        assert digest not in drafting.listed()
        for path in (f"/api/game/{digest}", f"/api/design/{digest}",
                     f"/api/design/standing/{digest}"):
            assert ada.get(path).status == 404, path
        refused = ada.post("/api/session", {"hash": digest, "seats": ["human", AGENT]})
        assert refused.status == 400
        assert "cannot be played" in refused.payload["error"]

        # Saving the same rules again neither revives them nor makes a second row.
        again = _saved(ada, _template(name="the same rules, renamed"))
        assert again.payload["hash"] == digest
        assert again.payload["listed"] is False
        assert "taken off the site" in again.payload["said"]
        assert drafting.count("games", f"hash = '{digest}' and revoked is not null") == 1

    def test_a_game_on_trial_plays_in_sandbox_and_leaderboard_says_why_not(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """§3: playable against the roster once listed. Its result counts once a picker
        ladder is rated, so the card says unrated and leaderboard mode refuses in a
        sentence."""
        _screened(monkeypatch, passed=True)
        ada = drafting.signed_in("ada")
        digest = _saved(ada, _template()).payload["hash"]
        site_store.on_trial(drafting.store, digest)

        cards = {card["hash"]: card for card in ada.get("/api/library").payload["games"]}
        assert cards[digest]["rated"] is False

        started = ada.post(
            "/api/session",
            {"hash": digest, "seats": [AGENT, "human"], "mode": matches.SANDBOX},
        )
        assert started.status == 200, started.text
        match = started.payload["id"]
        replied = ada.post(f"/api/session/{match}/agent")
        assert replied.status == 200, replied.text
        assert matches.row(drafting.store, match).plies == 1

        counted = ada.post(
            "/api/session",
            {"hash": digest, "seats": ["human", AGENT], "mode": matches.LEADERBOARD},
        )
        assert counted.status == 400
        assert "have no ratings yet" in counted.payload["error"]

    def test_the_same_rules_saved_twice_are_one_row_and_the_first_saver_keeps_it(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _screened(monkeypatch, passed=True)
        first = _saved(drafting.signed_in("ada"), _template()).payload["hash"]
        second = _saved(
            drafting.signed_in("bob"), _template(name="bob's name for it")
        ).payload["hash"]

        assert first == second
        assert drafting.count("games", f"hash = '{first}'") == 1
        with drafting.store.cursor() as cursor:
            cursor.execute(
                "select u.handle from games g join users u on u.id = g.owner"
                " where g.hash = %s",
                (first,),
            )
            assert cursor.fetchone() == ("ada",)

    def test_a_fork_names_its_parent_when_the_store_holds_it(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _screened(monkeypatch, passed=True)
        ada = drafting.signed_in("ada")
        parent = _saved(ada, _template(0)).payload["hash"]
        child = _saved(ada, _template(1, "a fork"), parent=parent).payload["hash"]
        orphan = _saved(ada, _template(2, "an orphan"), parent="0" * 12).payload["hash"]

        with drafting.store.cursor() as cursor:
            cursor.execute(
                "select hash, parent from games where hash = any(%s)", ([child, orphan],)
            )
            assert dict(cursor.fetchall()) == {child: parent, orphan: None}


def _designed_rows(site: Site, owner: int, count: int, **columns: Any) -> list[str]:
    """`count` designed rows for `owner`, written straight to the store.

    For the caps, which count rows and never read their rules; saving two hundred drafts
    through the route would test the check two hundred times and the cap once.
    """
    made = []
    given = {"visibility": drafts.UNLISTED, **columns}
    names = "".join(f", {name}" for name in given)
    marks = "".join(", %s" for _ in given)
    with site.store.cursor() as cursor:
        for index in range(count):
            digest = f"{owner:04d}{index:012d}"
            cursor.execute(
                f"insert into games (hash, spec, name, owner, origin{names})"
                f" values (%s, '{{}}'::jsonb, %s, %s, 'designed'{marks})",
                (digest, f"draft {index}", owner, *given.values()),
            )
            made.append(digest)

    site.store.commit()
    return made


def _user_id(site: Site, handle: str) -> int:
    with site.store.cursor() as cursor:
        cursor.execute("select id from users where handle = %s", (handle,))
        return int(cursor.fetchone()[0])


class TestPublishing:
    """Q503a. A save keeps a draft; *Publish* puts it on trial, and a press without a
    spot-checked reading is refused with the sentence that says how to get one (Q503e;
    the publish that succeeds through the route is `test_site_publish.py`'s). Everything
    before the reading -- who may press, the check, the two caps -- is held here, over
    HTTP where the route is the claim."""

    def _passed_draft(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch, handle: str = "ada"
    ) -> tuple[Browser, str]:
        _screened(monkeypatch, passed=True)
        person = drafting.signed_in(handle)
        return person, _saved(person, _template()).payload["hash"]

    def test_a_visitor_is_asked_to_sign_in(self, drafting: Site) -> None:
        answer = drafting.browser().post("/api/design/publish", {"hash": DULL_HASH})
        assert answer.status == 400
        assert "Sign in to publish a game" in answer.payload["error"]

    def test_the_designer_is_told_a_reading_is_needed_and_nothing_moves(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ada, digest = self._passed_draft(drafting, monkeypatch)

        answer = ada.post("/api/design/publish", {"hash": digest})

        assert answer.status == 400
        assert answer.payload["error"] == drafts.NEEDS_A_READING
        assert "Measure it here" in answer.payload["error"]
        assert drafting.count(
            "games", f"hash = '{digest}' and visibility = 'unlisted' and published is null"
        ) == 1
        assert digest not in drafting.listed()

    def test_the_designers_panel_says_so_and_its_button_is_off(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`GET /api/design/standing/<hash>` carries the designer's *Publish* panel: the
        check ticked, the reading not, the button off and the reason beside it."""
        ada, digest = self._passed_draft(drafting, monkeypatch)

        panel = ada.get(f"/api/design/standing/{digest}").payload["publish"]

        assert (panel["level"], panel["mine"], panel["open"]) == ("Draft", True, False)
        assert [tick["met"] for tick in panel["ticks"]] == [True, False]
        assert panel["waiting"] == drafts.NEEDS_A_READING
        # Said once (Q537): the line above the measuring control says how to get a reading, so the
        # tick only names what a publish needs.
        assert panel["ticks"][1]["says"] == drafts.READING_TICK
        lines = [panel["says"], panel["waiting"]] + [tick["says"] for tick in panel["ticks"]]
        assert " | ".join(lines).count("Measure it here") == 1

        bob = drafting.signed_in("bob")
        theirs = bob.get(f"/api/design/standing/{digest}").payload["publish"]
        assert (theirs["mine"], theirs["ticks"], theirs["open"]) == (False, [], False)
        assert "only its designer can publish" in theirs["says"]

    def test_only_the_designer_may_press_it(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, digest = self._passed_draft(drafting, monkeypatch)
        bob = drafting.signed_in("bob")

        answer = bob.post("/api/design/publish", {"hash": digest})

        assert answer.status == 400
        assert "Only the person who saved this game" in answer.payload["error"]

    def test_a_draft_that_failed_the_check_is_refused_for_that(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _screened(monkeypatch, passed=False, reasons=("too dull",))
        ada = drafting.signed_in("ada")
        digest = _saved(ada, _template()).payload["hash"]

        answer = ada.post("/api/design/publish", {"hash": digest})

        assert answer.status == 400
        assert "not passed the playability check" in answer.payload["error"]

    def test_a_killed_draft_is_not_there_to_publish(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ada, digest = self._passed_draft(drafting, monkeypatch)
        admin.kill_game(drafting.store, digest)
        drafting.store.commit()

        assert ada.post("/api/design/publish", {"hash": digest}).status == 404

    def test_a_library_game_has_nothing_to_publish(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drafting.game()
        ada = drafting.signed_in("ada")

        answer = ada.post("/api/design/publish", {"hash": DULL_HASH})

        assert answer.status == 400
        assert "came with the site" in answer.payload["error"]

    def test_five_games_on_trial_stop_a_sixth(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ada, digest = self._passed_draft(drafting, monkeypatch)
        # Published long ago, so the day's cap is not what answers.
        _designed_rows(
            drafting, _user_id(drafting, "ada"), quota.PUBLISHED_AT_ONCE,
            visibility="public", screen_passed=True,
            published=datetime.now(timezone.utc) - timedelta(days=30),
        )

        answer = ada.post("/api/design/publish", {"hash": digest})

        assert answer.status == 400
        assert f"limit of {quota.PUBLISHED_AT_ONCE}" in answer.payload["error"]
        assert "on trial" in answer.payload["error"]

    def test_a_game_that_left_the_listing_frees_its_place(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Killed rows are not on trial, so the count is under the cap and the press gets
        as far as the reading."""
        ada, digest = self._passed_draft(drafting, monkeypatch)
        rows = _designed_rows(
            drafting, _user_id(drafting, "ada"), quota.PUBLISHED_AT_ONCE,
            visibility="public", screen_passed=True,
            published=datetime.now(timezone.utc) - timedelta(days=30),
        )
        admin.kill_game(drafting.store, rows[0])
        drafting.store.commit()

        answer = ada.post("/api/design/publish", {"hash": digest})

        assert answer.payload["error"] == drafts.NEEDS_A_READING

    def test_one_publish_a_rolling_day(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ada, digest = self._passed_draft(drafting, monkeypatch)
        _designed_rows(
            drafting, _user_id(drafting, "ada"), quota.PUBLISHES_A_DAY,
            visibility="public", screen_passed=True,
            published=datetime.now(timezone.utc) - timedelta(hours=23),
        )

        answer = ada.post("/api/design/publish", {"hash": digest})

        assert answer.status == 400
        assert "in the last day" in answer.payload["error"]
        assert f"limit of {quota.PUBLISHES_A_DAY}" in answer.payload["error"]

    def test_with_a_reading_publish_lists_it_once_and_the_day_is_then_spent(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What the route will do once Q503e attaches readings, asked of `drafts.publish`
        with a stand-in reading: the game is listed and stamped, a second press says it
        already is, and the day's cap then stops the next draft."""
        _screened(monkeypatch, passed=True)
        ada = drafting.signed_in("ada")
        first = _saved(ada, _template(0)).payload["hash"]
        second = _saved(ada, _template(1, "another")).payload["hash"]
        owner = _user_id(drafting, "ada")

        done = drafts.publish(drafting.store, first, owner, site_store.STAND_IN_READING, site_terms.VERSION)
        drafting.store.commit()
        assert not done.already and "on trial" in done.said
        assert first in drafting.listed()
        assert drafting.count("games", f"hash = '{first}' and published is not null") == 1
        again = drafts.publish(drafting.store, first, owner, site_store.STAND_IN_READING, site_terms.VERSION)
        assert again.already
        with pytest.raises(quota.QuotaError, match="in the last day"):
            drafts.publish(drafting.store, second, owner, site_store.STAND_IN_READING, site_terms.VERSION)
        drafting.store.rollback()

        panel = ada.get(f"/api/design/standing/{first}").payload["publish"]
        assert (panel["level"], panel["ticks"], panel["waiting"]) == ("On trial", [], "")


def _respelled(
    spec: dict[str, Any], name: str = "my own game", moves: bool = True, squares: bool = True
) -> dict[str, Any]:
    """The same rules under another spelling: lists whose order is not a rule turned round,
    and a new name. Another hash, the same `relate.identity_key`. `moves` and `squares`
    choose which, so a test can hold three spellings of one game at once."""
    made = json.loads(json.dumps(spec))
    made["name"] = name
    if moves:
        made["pieces"] = list(reversed(made["pieces"]))
        for piece in made["pieces"]:
            piece["moves"] = list(reversed(piece["moves"]))
    if squares:
        made["setup"]["placements"] = [
            {**row, "squares": list(reversed(row["squares"]))}
            for row in reversed(made["setup"]["placements"])
        ]
    return made


class TestTheSameGameRefusal:
    """Q503c, option E2. *Publish* refuses, by that game's name, a draft whose rules are a
    Library, Baseline or trial game's under another spelling (`drafts.twin`). A copy is told
    it is a copy before anything it could wait out -- the caps, and the reading that cannot
    be attached until Q503e -- and the key never lends the draft a measurement."""

    ORIGINAL = "the original"

    def _original(self, drafting: Site, origin: str, **columns: Any) -> str:
        """The template, as a game that came with the site under `origin`."""
        spec = _template(0, self.ORIGINAL)
        digest = drafting.game(spec_hash(spec), spec)
        sets = ", ".join(f"{name} = %s" for name in ("origin", *columns))
        with drafting.store.cursor() as cursor:
            cursor.execute(
                f"update games set {sets} where hash = %s",  # noqa: S608
                (origin, *columns.values(), digest),
            )
        drafting.store.commit()
        return digest

    def _copy(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch, handle: str = "ada"
    ) -> tuple[Browser, str]:
        _screened(monkeypatch, passed=True)
        person = drafting.signed_in(handle)
        spec = _respelled(_template(0))
        assert spec_hash(spec) != spec_hash(_template(0, self.ORIGINAL))
        assert relate.identity_key(spec) == relate.identity_key(_template(0))
        return person, _saved(person, spec).payload["hash"]

    def _unmoved(self, drafting: Site, digest: str) -> None:
        assert drafting.count(
            "games", f"hash = '{digest}' and visibility = 'unlisted' and published is null"
        ) == 1
        assert digest not in drafting.listed()

    @pytest.mark.parametrize(
        ("origin", "level"), [("archive", "in the Library"), ("corpus", "a Baseline game")]
    )
    def test_a_copy_of_a_game_that_came_with_the_site_is_refused_by_its_name(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch, origin: str, level: str
    ) -> None:
        self._original(drafting, origin)
        ada, digest = self._copy(drafting, monkeypatch)

        answer = ada.post("/api/design/publish", {"hash": digest})

        assert answer.status == 400
        said = answer.payload["error"]
        assert said.startswith(f"These rules play the same as {self.ORIGINAL} ({level})")
        assert "they are only written differently" in said and "spelling" not in said
        assert f"open {self.ORIGINAL}" in said
        assert said != drafts.NEEDS_A_READING
        self._unmoved(drafting, digest)

    def test_a_copy_of_a_game_on_trial_is_refused_by_its_name(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _screened(monkeypatch, passed=True)
        bob = drafting.signed_in("bob")
        theirs = _saved(bob, _template(0, "bob's game")).payload["hash"]
        site_store.on_trial(drafting.store, theirs)
        ada, digest = self._copy(drafting, monkeypatch)

        answer = ada.post("/api/design/publish", {"hash": digest})

        assert answer.status == 400
        assert answer.payload["error"].startswith(
            "These rules play the same as bob's game (on trial)"
        )
        self._unmoved(drafting, digest)

    def test_a_game_that_is_not_on_the_site_is_not_a_reason(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Somebody's unlisted draft, a game on trial that was killed, and a Baseline game
        the corpus retired are none of the three levels, so the press gets as far as the
        reading."""
        _screened(monkeypatch, passed=True)
        bob = drafting.signed_in("bob")
        _saved(bob, _respelled(_template(0), "bob's draft", moves=False))
        carol = drafting.signed_in("carol")
        killed = _saved(
            carol, _respelled(_template(0), "carol's game", squares=False)
        ).payload["hash"]
        site_store.on_trial(drafting.store, killed)
        admin.kill_game(drafting.store, killed)
        drafting.store.commit()
        self._original(drafting, "corpus", retired=datetime.now(timezone.utc))
        ada, digest = self._copy(drafting, monkeypatch)

        answer = ada.post("/api/design/publish", {"hash": digest})

        assert answer.payload["error"] == drafts.NEEDS_A_READING

    def test_a_copy_is_told_so_before_the_days_cap(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._original(drafting, "corpus")
        ada, digest = self._copy(drafting, monkeypatch)
        _designed_rows(
            drafting, _user_id(drafting, "ada"), quota.PUBLISHES_A_DAY,
            visibility="public", screen_passed=True,
            published=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        answer = ada.post("/api/design/publish", {"hash": digest})

        assert "play the same as" in answer.payload["error"]

    def test_a_reading_does_not_get_a_copy_published(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Asked of `drafts.publish` with a stand-in reading, as Q503a's tests ask what the
        route will do once Q503e attaches one: the copy is refused all the same."""
        self._original(drafting, "archive")
        _, digest = self._copy(drafting, monkeypatch)

        with pytest.raises(drafts.PublishError, match="play the same as the original"):
            drafts.publish(
                drafting.store, digest, _user_id(drafting, "ada"), site_store.STAND_IN_READING,
                site_terms.VERSION,
            )
        drafting.store.rollback()
        self._unmoved(drafting, digest)

    def test_the_key_names_a_game_and_lends_the_copy_nothing(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Move order follows spelling, so two spellings under one seed play different
        games: the original's measurement stays with its hash, and all a match hands back
        is which game it was."""
        original = self._original(drafting, "archive")
        with drafting.store.cursor() as cursor:
            cursor.execute(
                "insert into measurements (hash, directory, state, headline)"
                " values (%s, %s, %s, %s)",
                (original, f"runs/{original}", "measured", 0.5),
            )
        drafting.store.commit()
        ada, digest = self._copy(drafting, monkeypatch)
        spec = _respelled(_template(0))

        found = drafts.twin(drafting.store, digest, spec)
        ada.post("/api/design/publish", {"hash": digest})

        assert found == drafts.Twin(original, self.ORIGINAL, "in the Library")
        assert set(vars(found)) == {"hash", "name", "level"}
        assert drafting.count("measurements", f"hash = '{digest}'") == 0
        assert drafting.count("provisional", f"game = '{digest}'") == 0
        assert drafting.count("measurements", f"hash = '{original}'") == 1

    def test_the_designers_panel_says_so_and_names_the_game(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = self._original(drafting, "corpus")
        ada, digest = self._copy(drafting, monkeypatch)

        panel = ada.get(f"/api/design/standing/{digest}").payload["publish"]

        assert panel["open"] is False
        assert panel["waiting"].startswith("These rules play the same as the original")
        assert panel["twin"] == {"hash": original, "name": self.ORIGINAL}
        assert [tick["met"] for tick in panel["ticks"]] == [True, False]
        # Under the twin's refusal the tick still names what a publish needs.
        assert panel["ticks"][1]["says"] == drafts.READING_TICK

    def test_a_draft_that_copies_nothing_has_no_twin_on_its_panel(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _screened(monkeypatch, passed=True)
        ada = drafting.signed_in("ada")
        digest = _saved(ada, _template()).payload["hash"]

        panel = ada.get(f"/api/design/standing/{digest}").payload["publish"]

        assert (panel["twin"], panel["waiting"]) == (None, drafts.NEEDS_A_READING)


class TestTheSaveCaps:
    """Q503a's three guards on a save: the body's size, the drafts a person keeps, and the
    saves a person makes in a day."""

    def test_a_design_body_over_the_cap_is_refused_before_it_is_read(
        self, drafting: Site
    ) -> None:
        ada = drafting.signed_in("ada")
        spec = _template()
        spec["description"] = "x" * quota.DESIGN_BODY_BYTES

        answer = ada.post("/api/design/save", {"spec": spec, "parent": ""})

        assert answer.status == 413
        assert f"at most {quota.DESIGN_BODY_BYTES // 1024} KB" in answer.payload["error"]
        assert drafting.count("games") == 0

    def test_the_cap_is_the_designers_alone(self, drafting: Site) -> None:
        """Another route's large body is that route's business: a suggestion over the
        cap is refused for its length, in its own words, not with the designer's 413."""
        ada = drafting.signed_in("ada")
        answer = ada.post(
            "/api/suggest", {"kind": "rule", "body": "x" * quota.DESIGN_BODY_BYTES}
        )
        assert answer.status == 400

    def test_a_body_under_the_cap_is_saved(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _screened(monkeypatch, passed=True)
        ada = drafting.signed_in("ada")
        spec = _template()
        spec["description"] = "x" * (quota.DESIGN_BODY_BYTES // 2)

        assert ada.post("/api/design/save", {"spec": spec, "parent": ""}).status == 200

    def test_two_hundred_drafts_stop_a_new_one_but_not_the_same_rules_again(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _screened(monkeypatch, passed=True)
        ada = drafting.signed_in("ada")
        kept = _saved(ada, _template(0)).payload["hash"]
        _designed_rows(drafting, _user_id(drafting, "ada"), quota.DRAFTS_A_PERSON - 1)

        refused = ada.post("/api/design/save", {"spec": _template(1, "one more")})
        assert refused.status == 400
        assert f"limit of {quota.DRAFTS_A_PERSON}" in refused.payload["error"]

        again = _saved(ada, _template(0, "renamed"))
        assert again.payload["hash"] == kept and again.payload["created"] is False

    def test_the_thirty_first_save_of_a_day_is_refused(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _screened(monkeypatch, passed=True)
        ada = drafting.signed_in("ada")
        owner = _user_id(drafting, "ada")
        for _ in range(quota.SAVES_A_DAY - 1):
            drafting.app.saves.save(owner)
        _saved(ada, _template())

        refused = ada.post("/api/design/save", {"spec": _template(), "parent": ""})

        assert refused.status == 429
        assert f"saved {quota.SAVES_A_DAY} times in the last day" in refused.payload["error"]


class TestTheSavesCountRolls:
    """No database. The day is a rolling one, as `Presses`' hour is."""

    def test_the_oldest_save_falling_out_of_the_day_frees_one(self) -> None:
        now = [0.0]
        saves = quota.Saves(clock=lambda: now[0])
        for _ in range(quota.SAVES_A_DAY):
            saves.save(1)
        with pytest.raises(quota.Spent, match="available in 24 hours"):
            saves.save(1)
        saves.save(2)  # somebody else's day is their own

        now[0] = quota.SAVES_WINDOW.total_seconds() + 1
        saves.save(1)


class TestTheMigrationPutsListedDraftsBack:
    """`0016_drafts.sql` over a store really at `0015`, with the rows the code of that time
    wrote: a draft that passed (public), one that did not (unlisted), a killed one that had
    passed, and a library game. Then the profile of the designer whose games went back."""

    def test_public_drafts_go_back_to_draft_and_nothing_else_is_stamped(
        self, tmp_path: Path
    ) -> None:
        older = tmp_path / "migrations-before-0016"
        older.mkdir()
        for migration in migrate.available():
            if migration.number < 16:
                (older / migration.name).write_text(migration.sql, encoding="utf-8")
        url = site_store.database("no database to carry drafts across 0016 in")
        schema = f"before_drafts_{secrets.token_hex(6)}"
        store = connect(url)
        try:
            with store.cursor() as cursor:
                cursor.execute(f"create schema {schema}")
                cursor.execute(f"set search_path to {schema}")
            store.commit()
            migrate.run(store, older)
            store.commit()
            with store.cursor() as cursor:
                cursor.execute(
                    "insert into users (handle, email, password) values"
                    " ('ada', 'ada@example.invalid', 'x') returning id"
                )
                ada = int(cursor.fetchone()[0])
                rows = (
                    ("a" * 12, "designed", "public", None),
                    ("b" * 12, "designed", "unlisted", None),
                    ("c" * 12, "designed", "public", "2026-01-02"),
                    ("d" * 12, "corpus", "public", None),
                )
                for digest, origin, visibility, revoked in rows:
                    cursor.execute(
                        "insert into games (hash, spec, name, owner, visibility, origin,"
                        " created, revoked)"
                        " values (%s, '{}'::jsonb, %s, %s, %s, %s, '2026-01-01', %s)",
                        (digest, f"game {digest[0]}", ada, visibility, origin, revoked),
                    )
                cursor.execute("select * from games order by hash")
                before = cursor.fetchall()
            store.commit()

            report = migrate.run(store)
            store.commit()
            assert report.applied[0] == drafts.MIGRATION, report.summary()

            with store.cursor() as cursor:
                cursor.execute(
                    "select hash, visibility, screen_passed, published from games"
                    " order by hash"
                )
                assert cursor.fetchall() == [
                    ("a" * 12, "unlisted", True, None),
                    ("b" * 12, "unlisted", False, None),
                    ("c" * 12, "unlisted", True, None),
                    ("d" * 12, "public", False, None),
                ]
                cursor.execute(
                    "select hash, spec, name, owner, parent, origin, created, nominated,"
                    " revoked, retired, deleted from games order by hash"
                )
                after = cursor.fetchall()
            # Every column that was there is as it was, but visibility.
            columns = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11]
            assert after == [tuple(row[i] for i in columns) for row in before]

            # The profile's reading: the live draft that went back, not the one that never
            # passed, not the killed one, and not the library's.
            back = drafts.returned(store, ada)
            assert [game.hash for game in back] == ["a" * 12]
        finally:
            store.rollback()
            with store.cursor() as cursor:
                cursor.execute(f"drop schema if exists {schema} cascade")
            store.commit()
            store.close()

    def test_the_designers_profile_says_why(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Over the running site: a draft saved before the migration ran, which is what
        the migration put back, is named on its designer's profile; one saved today is
        not, and nobody else's profile names either."""
        _screened(monkeypatch, passed=True)
        ada = drafting.signed_in("ada")
        today = _saved(ada, _template()).payload["hash"]
        [old] = _designed_rows(
            drafting, _user_id(drafting, "ada"), 1, screen_passed=True,
            created=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        payload = ada.get("/api/profile").payload

        assert [game["hash"] for game in payload["backToDraft"]] == [old]
        assert today not in {game["hash"] for game in payload["backToDraft"]}
        assert drafting.signed_in("bob").get("/api/profile").payload["backToDraft"] == []
        page = page_text.inlined(ada.get("/profile").text)
        assert "Your games back to Draft" in page
        assert "Their links still work" in page
        # Q730. Publishing opened with Q503e, so the page says how, not that it is coming.
        assert "opens soon" not in page and "press Publish" in page


class TestTheThreeStatesAGameRowCanBeIn:

    """No database. Which word a row reads as, and what the page's buttons act on."""

    def _row(self, revoked: Any = None, retired: Any = None) -> admin.Listed:
        return admin.Listed(
            hash=DULL_HASH, name="dull", origin="corpus", revoked=revoked,
            matches=0, retired=retired,
        )

    def test_a_row_with_neither_stamp_is_listed(self) -> None:
        assert self._row().state == "listed"

    def test_a_stamp_from_the_loader_reads_retired(self) -> None:
        assert self._row(retired=datetime.now(timezone.utc)).state == "retired"

    def test_the_kill_switch_wins_because_it_is_the_act_to_undo(self) -> None:
        both = self._row(
            revoked=datetime.now(timezone.utc), retired=datetime.now(timezone.utc)
        )
        assert both.state == "killed"
        assert both.as_dict()["killed"] is True

    def test_the_pages_buttons_read_the_kill_switch_and_not_the_state_word(self) -> None:
        """Otherwise a retired game offers *Put back*, which clears a column nothing set
        and leaves the row exactly as it was."""
        page = page_text.inlined(site_pages.admin("boss"))
        assert "if (!game.killed)" in page
        assert "retired" in page, "the page says what the third state means"


def _tree(drafting: Site) -> Path:
    return Path(drafting.app.runs_dir).parent


def _load_tree(drafting: Site) -> Any:
    """A load of this tree, with a second spec in it so the corpus is never empty.

    The `drafting` fixture commits one ruleset, and taking that one out would leave a
    tree the loader refuses to retire from at all -- an empty `rulesets/` reads as a
    half-shipped checkout rather than as an emptied corpus.
    """
    companion = _tree(drafting) / "rulesets" / "other.json"
    if not companion.exists():
        spec = json.loads(json.dumps(DULL))
        spec["rules"]["max_plies"] = int(spec["rules"]["max_plies"]) + 2
        spec["name"] = "the other game"
        companion.write_text(json.dumps(spec), encoding="utf-8")
    # A picker ladder for `dull`, so the loader seeds its rating table: a profile lists a
    # person's record on rated games, and an unrated game has no row there to keep or to
    # mark.
    ladder = _tree(drafting) / "runs" / "opponents" / f"{DULL_HASH}.json"
    if not ladder.exists():
        ladder.parent.mkdir(parents=True, exist_ok=True)
        ladder.write_text(
            json.dumps(
                {
                    "ruleset_hash": DULL_HASH,
                    "standings": [
                        {"name": AGENT, "rating": 300.0, "games": 24},
                        {"name": "random", "rating": -300.0, "games": 24},
                    ],
                }
            ),
            encoding="utf-8",
        )
    report = loader.load(drafting.store, _tree(drafting))
    drafting.store.commit()
    return report


def _retire_dull(drafting: Site) -> Any:
    """Load, take `dull` out of the corpus, load again -- a deploy after a retirement."""
    _load_tree(drafting)
    (_tree(drafting) / "rulesets" / "dull.json").unlink()
    return _load_tree(drafting)


class TestAGameRetiredFromTheCorpus:
    """Q287, over a running site: what a deploy's load does to a game that has left.

    The tree here is the `drafting` fixture's, so `dull` is a committed spec the loader
    writes as a corpus row and the draft beside it is somebody's own. Taking the file out
    and loading again is exactly what a deploy after a retirement does.
    """

    def _tree(self, drafting: Site) -> Path:
        return _tree(drafting)

    def _load(self, drafting: Site) -> Any:
        return _load_tree(drafting)

    def _retire(self, drafting: Site) -> Any:
        return _retire_dull(drafting)

    def test_it_leaves_the_listing_and_the_saved_draft_stays(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both halves of the listing's door. `drafts.listed` is what a card from `runs/`
        is offered against and `admin.unlisted_games` is what a corpus card is dropped
        by, and a draft is neither retired nor dropped."""
        _screened(monkeypatch, passed=True)
        self._load(drafting)
        draft = _saved(drafting.signed_in("ada"), _template()).payload["hash"]
        site_store.on_trial(drafting.store, draft)
        assert {DULL_HASH, draft} <= drafting.listed()


        self._retire(drafting)

        assert DULL_HASH in admin.unlisted_games(drafting.store)
        assert DULL_HASH not in drafts.listed(drafting.store)
        assert draft in drafts.listed(drafting.store)
        listed = drafting.listed()
        assert DULL_HASH not in listed and draft in listed

    def test_its_row_is_still_read_by_hash(self, drafting: Site) -> None:
        """Not the kill switch's 404. The row is what a profile reads a name off and what
        a past match is rebuilt from, so retirement stops short of taking it away."""
        self._retire(drafting)
        assert browse.names(drafting.store, [DULL_HASH]) == {DULL_HASH: DULL["name"]}
        assert DULL_HASH not in admin.revoked_games(drafting.store)

    def test_a_game_already_played_on_it_is_still_readable(self, drafting: Site) -> None:
        """A match is a fact about somebody's afternoon, and the spec it was played under
        is in the row rather than in the tree -- so the position still rebuilds."""
        self._load(drafting)
        ada = drafting.person("ada")
        match = matches.start(
            drafting.store, DULL_HASH, matches.SANDBOX, ["human", AGENT], ada
        )
        drafting.store.commit()

        self._retire(drafting)

        assert matches.row(drafting.store, match.id).game == DULL_HASH
        assert matches.session(drafting.store, match.id).id == match.id

    def test_no_new_game_can_be_started_on_it(self, drafting: Site) -> None:
        self._retire(drafting)
        ada = drafting.person("ada")
        with pytest.raises(matches.MatchError, match="retired from the library"):
            matches.start(
                drafting.store, DULL_HASH, matches.SANDBOX, ["human", AGENT], ada
            )

    def test_the_admin_page_says_retired_and_still_offers_the_kill_switch(
        self, drafting: Site
    ) -> None:
        """The buttons act on `revoked`, so a retired game can still be taken down; the
        state word is what tells a maintainer which of the two states it is in."""
        self._retire(drafting)
        boss = drafting.signed_in("boss", admin.ADMIN)

        shown = {game["hash"]: game for game in boss.get("/api/admin").payload["games"]}
        assert shown[DULL_HASH]["state"] == "retired"
        assert shown[DULL_HASH]["killed"] is False

        boss.post("/api/admin", {"act": "kill-game", "hash": DULL_HASH})
        after = {game["hash"]: game for game in boss.get("/api/admin").payload["games"]}
        assert after[DULL_HASH]["state"] == "killed"

    def test_a_persons_own_games_stay_on_their_profile_marked(
        self, drafting: Site
    ) -> None:
        """Their games are theirs. Dropping the row would make the profile's table
        disagree with the counts above it, so the row stays and says what happened."""
        self._load(drafting)
        ada = drafting.signed_in("ada")
        user = _id(drafting, "ada")
        match = matches.start(
            drafting.store, DULL_HASH, matches.LEADERBOARD, ["human", AGENT], user
        )
        matches.resign(drafting.store, match.id, by=user)
        drafting.store.commit()

        self._retire(drafting)

        mine = {
            game["hash"]: game
            for game in ada.get("/api/profile").payload["games"]
        }
        assert mine[DULL_HASH]["retired"] is True
        assert mine[DULL_HASH]["played"] >= 1
        assert "game.retired" in page_text.inlined(site_pages.profile("ada")), "the page draws the mark"

    def test_putting_the_spec_back_puts_the_game_back(self, drafting: Site) -> None:
        self._retire(drafting)
        (self._tree(drafting) / "rulesets" / "dull.json").write_text(
            json.dumps(DULL), encoding="utf-8"
        )

        self._load(drafting)

        assert DULL_HASH not in admin.unlisted_games(drafting.store)
        assert DULL_HASH in drafting.listed()


def _as_sent(spec: dict[str, Any]) -> dict[str, Any]:
    """A spec as the site's game page sends it: its provenance without the maintainer's
    records (W7, `server.visitors_provenance`); every rule as written."""
    from chess_variant_sandbox.site.server import visitors_provenance

    return {**spec, "provenance": visitors_provenance(spec["provenance"])}


class TestARetiredGamesPage:
    """Q303, the first half. A retired game keeps a working page.

    The local application draws a game's page from the tree, so a retired game used to
    answer the 404 an absent spec has always given. The rules are in the row, and that is
    what the page is drawn from now (`site/kept.py`). Nothing here computes a move: the
    page shows rules and records, and the play panel is what goes.
    """

    def test_the_page_is_drawn_from_the_stores_copy_of_the_rules(
        self, drafting: Site
    ) -> None:
        _retire_dull(drafting)
        assert DULL_HASH not in drafting.app.known(), "the tree no longer has the spec"

        answer = drafting.browser().get(f"/api/game/{DULL_HASH}")

        assert answer.status == 200, answer.text
        page = answer.payload
        assert page["name"] == DULL["name"]
        assert page["spec"] == _as_sent(DULL), "the rules on the page are the row's own copy"
        assert page["ruleLines"], "and they are described, not merely carried"
        assert page["standing"]["retired"] is True

    def test_a_game_the_tree_still_holds_is_drawn_and_marked_the_way_it_was(
        self, drafting: Site
    ) -> None:
        """The fallback is a fallback. A present spec never reaches it, and its page says
        nothing about retirement."""
        _load_tree(drafting)
        page = drafting.browser().get(f"/api/game/{DULL_HASH}").payload
        assert page["standing"]["retired"] is False
        assert page["spec"] == _as_sent(DULL)

    def test_no_new_game_can_be_started_from_that_page(self, drafting: Site) -> None:
        """The label is the page's half; this is the server's. A retired game's page is
        readable and its board is not offered."""
        _retire_dull(drafting)
        ada = drafting.signed_in("ada")

        answer = ada.post(
            "/api/session", {"hash": DULL_HASH, "seats": ["human", AGENT]}
        )

        assert answer.status == 400
        assert "retired from the library" in answer.text

    def test_a_killed_game_still_has_no_page_at_all(self, drafting: Site) -> None:
        """Retirement made a page where there was none; the kill switch is untouched by
        that. A killed game answers the 404 an unknown hash gets, whatever the row holds.
        """
        _load_tree(drafting)
        boss = drafting.signed_in("boss", admin.ADMIN)
        boss.post("/api/admin", {"act": "kill-game", "hash": DULL_HASH})

        assert drafting.browser().get(f"/api/game/{DULL_HASH}").status == 404


class TestWhoMayDeleteAGame:
    """No database. The one place who-may-delete is decided (`admin.GameState`)."""

    def _state(self, owner: int | None = 7, retired: bool = True,
               deleted: bool = False) -> admin.GameState:
        when = datetime.now(timezone.utc)
        return admin.GameState(
            hash=DULL_HASH, name="dull", owner=owner,
            revoked=None,
            retired=when if retired else None,
            deleted=when if deleted else None,
        )

    def test_the_creator_may(self) -> None:
        assert self._state().may_delete(7) is True

    def test_an_admin_may_whoever_made_it(self) -> None:
        assert self._state().may_delete(99, admin=True) is True

    def test_nobody_else_may(self) -> None:
        assert self._state().may_delete(99) is False

    def test_a_visitor_may_not_even_when_the_game_has_no_owner(self) -> None:
        """`owner` is null on a corpus row loaded before any account existed, and a
        visitor's id is `None`. Two nulls that must not compare equal."""
        assert self._state(owner=None).may_delete(None) is False

    def test_a_game_still_in_the_tree_may_not_be_deleted_by_anyone(self) -> None:
        """Its rules are in the repository, so the next deploy would write them back."""
        assert self._state(retired=False).may_delete(7) is False
        assert self._state(retired=False).may_delete(7, admin=True) is False

    def test_a_deleted_game_cannot_be_deleted_again(self) -> None:
        assert self._state(deleted=True).may_delete(7, admin=True) is False


class TestDeletingAGameOutright:
    """Q303, the second half. The user, 2026-09-18: *"a button available to the game's
    creator and admin to delete completely. The game rules are then actually deleted."*

    The row stays as a placeholder without a spec, because every reference to a game in
    this store cascades -- so deleting the row would delete the matches people played on
    it, their ratings and their votes. That decision is `docs/plan-website.md`'s and
    `0010_deleted.sql`'s; these tests hold both halves of it, the rules gone and the
    statistics unmoved.
    """

    def _owned_by(self, drafting: Site, user: int) -> None:
        """Hand the retired row to a member.

        Every row the loader retires is the maintainer's, because the loader stamps only
        the games it owns. The creator's half of the button is for a game somebody's own
        account put there, so the test puts the row in that state directly rather than
        waiting for the phase that uploads one.
        """
        with drafting.store.cursor() as cursor:
            cursor.execute(
                "update games set owner = %s where hash = %s", (user, DULL_HASH)
            )
        drafting.store.commit()

    def _delete(self, browser: Browser) -> Answer:
        return browser.post(f"/api/game/{DULL_HASH}/delete")

    def test_the_creator_deletes_it_and_the_rules_are_gone(
        self, drafting: Site
    ) -> None:
        _retire_dull(drafting)
        ada = drafting.signed_in("ada")
        self._owned_by(drafting, _id(drafting, "ada"))

        answer = self._delete(ada)

        assert answer.status == 200, answer.text
        assert answer.payload["was"] == DULL["name"], "the name, one last time"
        with drafting.store.cursor() as cursor:
            cursor.execute(
                "select spec, name, deleted from games where hash = %s", (DULL_HASH,)
            )
            spec, name, when = cursor.fetchone()
        assert spec is None, "the rules themselves are deleted"
        assert name == admin.DELETED_NAME
        assert when is not None

    def test_an_admin_deletes_a_game_they_did_not_make(self, drafting: Site) -> None:
        _retire_dull(drafting)
        self._owned_by(drafting, drafting.person("ada"))
        boss = drafting.signed_in("boss", admin.ADMIN)

        assert self._delete(boss).status == 200
        assert DULL_HASH in admin.deleted_games(drafting.store)

    def test_anybody_else_is_refused_and_the_rules_stay(self, drafting: Site) -> None:
        _retire_dull(drafting)
        self._owned_by(drafting, drafting.person("ada"))
        bob = drafting.signed_in("bob")

        answer = self._delete(bob)

        assert answer.status == 400
        assert "only the person who made a game" in answer.text
        assert DULL_HASH not in admin.deleted_games(drafting.store)

    def test_a_visitor_is_asked_to_sign_in_before_anything_runs(
        self, drafting: Site
    ) -> None:
        _retire_dull(drafting)
        answer = self._delete(drafting.browser())
        assert answer.status == 400
        assert "Sign in to delete a game" in answer.text
        assert DULL_HASH not in admin.deleted_games(drafting.store)

    def test_a_game_the_tree_still_holds_is_refused_by_name(
        self, drafting: Site
    ) -> None:
        """The refusal a deploy would otherwise make silently, three hours later."""
        _load_tree(drafting)
        boss = drafting.signed_in("boss", admin.ADMIN)

        answer = self._delete(boss)

        assert answer.status == 400
        assert "only a retired game can be deleted" in answer.text

    def test_deleting_it_twice_is_refused_rather_than_repeated(
        self, drafting: Site
    ) -> None:
        _retire_dull(drafting)
        boss = drafting.signed_in("boss", admin.ADMIN)
        assert self._delete(boss).status == 200

        again = self._delete(boss)

        assert again.status == 400
        assert "already been deleted" in again.text

    def test_nobodys_statistics_change(self, drafting: Site) -> None:
        """The whole reason the row stays. A match, a rating, a vote and a badge on this
        game are all still there afterwards, and each still names the same hash."""
        _load_tree(drafting)
        ada = drafting.signed_in("ada")
        user = _id(drafting, "ada")
        match = matches.start(
            drafting.store, DULL_HASH, matches.LEADERBOARD, ["human", AGENT], user
        )
        matches.resign(drafting.store, match.id, by=user)
        votes.cast(drafting.store, user, DULL_HASH, 4)
        awards.grant(
            drafting.store,
            [awards.Award(kind="first-blood", holder=user, game=DULL_HASH)],
        )
        drafting.store.commit()
        before = {
            table: drafting.count(table, f"game = '{DULL_HASH}'")
            for table in ("matches", "ratings", "votes", "awards")
        }
        assert all(count > 0 for count in before.values()), before
        _retire_dull(drafting)
        self._owned_by(drafting, user)

        assert self._delete(ada).status == 200

        after = {
            table: drafting.count(table, f"game = '{DULL_HASH}'")
            for table in ("matches", "ratings", "votes", "awards")
        }
        assert after == before
        assert matches.row(drafting.store, match.id).game == DULL_HASH

    def test_what_is_left_names_it_a_deleted_game(self, drafting: Site) -> None:
        """Every reader that resolves a hash to a name -- the profile's table, a milestone
        line, a badge -- asks `browse.names`, and the row itself is what answers."""
        _load_tree(drafting)
        ada = drafting.signed_in("ada")
        user = _id(drafting, "ada")
        match = matches.start(
            drafting.store, DULL_HASH, matches.LEADERBOARD, ["human", AGENT], user
        )
        matches.resign(drafting.store, match.id, by=user)
        drafting.store.commit()
        _retire_dull(drafting)
        self._owned_by(drafting, user)
        assert self._delete(ada).status == 200

        assert browse.names(drafting.store, [DULL_HASH]) == {
            DULL_HASH: admin.DELETED_NAME
        }
        mine = {
            game["hash"]: game for game in ada.get("/api/profile").payload["games"]
        }
        assert mine[DULL_HASH]["name"] == admin.DELETED_NAME
        assert mine[DULL_HASH]["played"] >= 1, "their games are still their games"
        assert DULL_HASH in ada.get("/api/profile").payload["deleted"]

    def test_its_page_its_card_and_a_new_match_are_all_gone(
        self, drafting: Site
    ) -> None:
        _retire_dull(drafting)
        boss = drafting.signed_in("boss", admin.ADMIN)
        assert self._delete(boss).status == 200

        assert drafting.browser().get(f"/api/game/{DULL_HASH}").status == 404
        assert DULL_HASH not in drafting.listed()
        assert DULL_HASH not in drafts.listed(drafting.store)
        assert DULL_HASH in admin.unlisted_games(drafting.store)
        with pytest.raises(matches.MatchError, match="have been deleted"):
            matches.start(
                drafting.store,
                DULL_HASH,
                matches.SANDBOX,
                ["human", AGENT],
                drafting.person("ada"),
            )

    def test_a_deploy_never_writes_the_rules_back(self, drafting: Site) -> None:
        """The delete has no way back, so a spec that reappears in the tree is left on
        disk and named in the load's own output."""
        _retire_dull(drafting)
        boss = drafting.signed_in("boss", admin.ADMIN)
        assert self._delete(boss).status == 200
        (_tree(drafting) / "rulesets" / "dull.json").write_text(
            json.dumps(DULL), encoding="utf-8"
        )

        report = _load_tree(drafting)

        assert any(DULL_HASH in note for note in report.notes), report.summary()
        assert DULL_HASH in admin.deleted_games(drafting.store)
        assert DULL_HASH not in drafting.listed()
        assert browse.names(drafting.store, [DULL_HASH]) == {
            DULL_HASH: admin.DELETED_NAME
        }

    def test_the_store_itself_refuses_a_row_that_says_both_things(
        self, drafting: Site
    ) -> None:
        """`0010_deleted.sql`'s check. A deleted row cannot hold rules and a live row
        cannot lack them, so *the rules are gone* is the schema's answer rather than a
        rule each writer has to remember."""
        _retire_dull(drafting)
        for statement in (
            "update games set deleted = now() where hash = %s",
            "update games set spec = null where hash = %s",
        ):
            with (
                pytest.raises(Exception, match="games_deleted_has_no_spec"),
                drafting.store.cursor() as cursor,
            ):
                cursor.execute(statement, (DULL_HASH,))
            drafting.store.rollback()

    def test_the_admin_page_shows_the_state_and_offers_no_button(
        self, drafting: Site
    ) -> None:
        _retire_dull(drafting)
        boss = drafting.signed_in("boss", admin.ADMIN)
        assert self._delete(boss).status == 200

        shown = {game["hash"]: game for game in boss.get("/api/admin").payload["games"]}
        assert shown[DULL_HASH]["state"] == "deleted"
        assert shown[DULL_HASH]["gone"] is True
        assert "if (game.gone)" in page_text.inlined(site_pages.admin("boss"))


class TestRevokingEveryone:
    def test_the_button_signs_everybody_out_and_keeps_every_account(
        self, site: Site
    ) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        theirs = site.signed_in("ada")
        assert site.count("sessions") == 2

        answer = boss.post("/api/admin", {"act": "revoke-sessions"})
        assert answer.status == 200
        assert site.count("sessions") == 0
        assert site.count("users") == 2
        assert theirs.get("/api/profile").status == 400

        back = site.browser().post(
            "/api/signin", {"handle": "ada", "password": PASSWORD}
        )
        assert back.status == 200


# ------------------------------------------------------------------------------ leaving


class TestLeaving:
    """`chess-variant-sandbox site forget`, against the words at `/privacy` that promise it."""

    def test_the_notice_still_says_what_the_command_does(self) -> None:
        """The prose and the code are one commit's worth of each other, so the test
        reads the notice rather than restating it."""
        notice = site_pages.privacy()
        assert identity.FORMER in notice
        assert "email address and your password hash are deleted" in notice
        assert "The matches themselves stay" in notice

    def test_the_person_goes_and_the_games_stay(self, site: Site) -> None:
        site.game()
        browser = site.signed_in("ada")
        user = _id(site, "ada")
        match = matches.start(site.store, DULL_HASH, matches.SANDBOX, ["human", AGENT], user)
        site.store.commit()
        assert site.count("sessions") == 1

        assert admin.forget(site.store, "ada") == "ada"
        site.store.commit()

        # What is gone: the address, the hash, every session, the name.
        with site.store.cursor() as cursor:
            cursor.execute(
                "select email, password, handle, forgotten from users where id = %s",
                (user,),
            )
            email, password, handle, forgotten = cursor.fetchone()
        assert (email, password, handle) == (None, None, None)
        assert forgotten is not None
        assert site.count("sessions") == 0
        assert browser.get("/api/profile").status == 400

        # What stays: the row, the match and the person's place in it.
        assert site.count("users", f"id = {user}") == 1
        kept = matches.row(site.store, match.id)
        assert kept.game == DULL_HASH
        assert user in (
            None if kept.seat0 is None else kept.seat0.user,
            None if kept.seat1 is None else kept.seat1.user,
        )

    def test_every_reading_of_their_name_says_a_former_player(self, site: Site) -> None:
        user = site.person("ada")
        admin.forget(site.store, "ada")
        site.store.commit()

        assert identity.handle_of(site.store, user) == identity.FORMER
        assert unbeaten.handles(site.store)[user] == identity.FORMER
        listed = {person.id: person for person in admin.people(site.store)}
        assert listed[user].handle == identity.FORMER
        assert listed[user].state == "left"

    def test_they_cannot_sign_in_afterwards(self, site: Site) -> None:
        site.person("ada")
        admin.forget(site.store, "ada")
        site.store.commit()

        answer = site.browser().post(
            "/api/signin", {"handle": "ada", "password": PASSWORD}
        )
        assert answer.status == 401

    def test_two_people_can_leave(self, site: Site) -> None:
        """`users.email` is unique, so leaving had to empty it rather than write a
        sentinel address -- which the second departure is what proves."""
        site.person("ada")
        site.person("bob")
        admin.forget(site.store, "ada")
        admin.forget(site.store, "bob")
        site.store.commit()
        assert site.count("users", "forgotten is not null") == 2

    def test_somebody_who_left_cannot_be_let_back_in(self, site: Site) -> None:
        user = site.person("ada")
        admin.forget(site.store, "ada")
        site.store.commit()
        with pytest.raises(admin.AdminError, match="forgotten"):
            admin.revive_user(site.store, user)

    def test_the_page_offers_it_and_asks_first(self, site: Site) -> None:
        browser = site.signed_in("boss", admin.ADMIN)
        user = site.person("ada")

        assert "cannot be undone" in page_text.inlined(browser.get("/admin").text)
        answer = browser.post("/api/admin", {"act": "forget-user", "user": user})
        assert answer.status == 200
        assert identity.handle_of(site.store, user) == identity.FORMER

        again = browser.post("/api/admin", {"act": "forget-user", "user": user})
        assert again.status == 400
        assert "already left" in again.payload["error"]

    def test_a_handle_nobody_has_is_refused_by_name(self, site: Site) -> None:
        with pytest.raises(admin.AdminError, match="no user with the handle"):
            admin.forget(site.store, "nobody")


# -------------------------------------------------------------------------- the quota


class TestTheNumbersAreInOnePlace:
    def test_the_sweeps_deadline_moved_to_the_quota_module(self) -> None:
        """§7 asked for one table. `matches.IDLE_LIMIT` is kept as a name and is the
        same object, so there is one value rather than two that agree today."""
        assert matches.IDLE_LIMIT is quota.IDLE_LIMIT

    def test_the_admin_page_reads_every_number_from_that_module(self) -> None:
        """The page prints the limits rather than naming them, so a changed constant
        reaches the page without a page change."""
        labels = [label for label, _ in quota.summary()]
        values = [value for _, value in quota.summary()]
        assert len(labels) == len(set(labels))
        assert str(quota.MATCHES_A_DAY) in values
        assert str(quota.STRONG_SEATS) in values


class TestTheBudgetInAnAgentString:
    """Budget is part of an agent's identity, so the cap reads the string."""

    @pytest.mark.parametrize(
        ("agent", "nodes"),
        [
            ("minimax:nodes=200000", 200_000),
            ("minimax:nodes=1600", 1_600),
            ("greedy", 0),
            ("random", 0),
        ],
    )
    def test_the_nodes_are_read_out_of_the_name(self, agent: str, nodes: int) -> None:
        assert quota.nodes_in(agent) == nodes

    def test_only_the_strongest_counts_against_the_seat_cap(self) -> None:
        assert quota.is_strong(STRONGEST)
        assert not quota.is_strong(AGENT)
        assert not quota.is_strong(None)


class TestTheDailyLimit:
    def test_a_person_is_stopped_at_the_limit_in_a_sentence(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(quota, "MATCHES_A_DAY", 2)
        site.game()
        user = site.person("ada")
        for _ in range(2):
            matches.start(site.store, DULL_HASH, matches.SANDBOX, ["human", AGENT], user)
        site.store.commit()

        with pytest.raises(quota.QuotaError, match="which is the limit of 2"):
            matches.start(site.store, DULL_HASH, matches.SANDBOX, ["human", AGENT], user)

    def test_the_refused_match_leaves_no_row_behind(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(quota, "MATCHES_A_DAY", 1)
        site.game()
        user = site.person("ada")
        matches.start(site.store, DULL_HASH, matches.SANDBOX, ["human", AGENT], user)
        site.store.commit()
        with pytest.raises(quota.QuotaError):
            matches.start(site.store, DULL_HASH, matches.SANDBOX, ["human", AGENT], user)
        assert site.count("matches") == 1

    def test_it_rolls_rather_than_resetting_at_midnight(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A calendar limit is a limit somebody can wait out on a clock."""
        monkeypatch.setattr(quota, "MATCHES_A_DAY", 1)
        site.game()
        user = site.person("ada")
        first = matches.start(
            site.store, DULL_HASH, matches.SANDBOX, ["human", AGENT], user
        )
        _backdate(site, first.id, timedelta(hours=25))

        assert quota.matches_today(site.store, user) == 0
        matches.start(site.store, DULL_HASH, matches.SANDBOX, ["human", AGENT], user)

    def test_a_visitor_has_no_day_to_count(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """There is no id to count by, so the daily limit does not apply at all. The
        ceiling on a visitor is the front door's connection limit, and a second worse
        copy of it does not belong here (§7)."""
        monkeypatch.setattr(quota, "MATCHES_A_DAY", 0)
        site.game()
        quota.check_new_match(site.store, None, AGENT)
        started = matches.start(site.store, DULL_HASH, matches.SANDBOX, ["human", AGENT])
        assert started.seat0 is None

    def test_the_player_reads_the_sentence_over_http(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(quota, "MATCHES_A_DAY", 0)
        browser = site.signed_in("ada")
        site.game()

        answer = browser.post(
            "/api/session", {"hash": DULL_HASH, "seats": ["human", AGENT]}
        )
        assert answer.status == 400
        assert "which is the limit of 0" in answer.payload["error"]


class TestTheStrongestSeat:
    def test_the_cap_is_over_the_whole_site_not_over_one_person(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Eight players holding a seat each cost the box what one player holding eight
        would, so the cap counts seats and not people."""
        monkeypatch.setattr(quota, "STRONG_SEATS", 1)
        site.game()
        ada = site.person("ada")
        bob = site.person("bob")
        matches.start(site.store, DULL_HASH, matches.SANDBOX, ["human", STRONGEST], ada)
        site.store.commit()

        with pytest.raises(quota.QuotaError, match="hardest is busy"):
            matches.start(
                site.store, DULL_HASH, matches.SANDBOX, ["human", STRONGEST], bob
            )

    def test_a_cheaper_opponent_is_never_refused_by_it(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(quota, "STRONG_SEATS", 0)
        site.game()
        user = site.person("ada")
        matches.start(site.store, DULL_HASH, matches.SANDBOX, ["human", AGENT], user)

    def test_a_seat_nobody_has_moved_in_stops_holding_the_cap(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise the cap would be a cap on matches ever started and left open: a
        sandbox row never finishes and the sweep only closes leaderboard ones."""
        monkeypatch.setattr(quota, "STRONG_SEATS", 1)
        site.game()
        user = site.person("ada")
        idle = matches.start(
            site.store, DULL_HASH, matches.SANDBOX, ["human", STRONGEST], user
        )
        assert quota.strong_seats(site.store) == 1

        _backdate(site, idle.id, quota.SEAT_ACTIVE + timedelta(minutes=1))
        assert quota.strong_seats(site.store) == 0
        matches.start(site.store, DULL_HASH, matches.SANDBOX, ["human", STRONGEST], user)

    def test_a_finished_match_holds_no_seat(self, site: Site) -> None:
        site.game()
        user = site.person("ada")
        match = matches.start(
            site.store, DULL_HASH, matches.SANDBOX, ["human", STRONGEST], user
        )
        with site.store.cursor() as cursor:
            cursor.execute(
                "update matches set finished = now() where id = %s", (match.id,)
            )
        site.store.commit()
        assert quota.strong_seats(site.store) == 0


class TestBusyIsSaidBeforeTheRefusal:
    """L11 of `docs/plan-launch.md` (§5.2, Q566): the opponent list marks the strongest
    player *busy* from the free-seat count, so a refused start is never the first a person
    hears of it. The game's `standing` carries the words; the page reads them."""

    def test_the_strongest_is_busy_exactly_while_a_start_would_be_refused(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(quota, "STRONG_SEATS", 1)
        site.game()
        visitor = site.browser()
        assert visitor.get(f"/api/game/{DULL_HASH}").payload["standing"]["busy"] == {}
        assert quota.busy(site.store) is False

        held = matches.start(
            site.store, DULL_HASH, matches.SANDBOX, ["human", STRONGEST], site.person("bob")
        )
        site.store.commit()
        standing = visitor.get(f"/api/game/{DULL_HASH}").payload["standing"]
        assert standing["busy"] == {STRONGEST: quota.BUSY_SAID}
        assert quota.BUSY_SAID == "busy, back in a few minutes"
        # What the list says is what a start meets.
        with pytest.raises(quota.QuotaError, match="come back in a few minutes"):
            matches.start(
                site.store, DULL_HASH, matches.SANDBOX, ["human", STRONGEST],
                site.person("ada"),
            )
        site.store.rollback()

        # The seat lapses after its idle minutes, and the mark goes with it.
        _backdate(site, held.id, quota.SEAT_ACTIVE + timedelta(minutes=1))
        assert visitor.get(f"/api/game/{DULL_HASH}").payload["standing"]["busy"] == {}

    def test_only_the_strongest_is_ever_marked(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(quota, "STRONG_SEATS", 0)
        site.game()
        busy = site.browser().get(f"/api/game/{DULL_HASH}").payload["standing"]["busy"]
        assert list(busy) == [STRONGEST]


class TestVisitorsPlayUpToHard:
    """L15 of `docs/plan-launch.md` (§3.6, decided A on 2026-10-01): with the shared
    password off the front door, a visitor plays computer players up to *hard*, and
    anything stronger is for signed-in people. Refused at the server, said in the list."""

    HARD = f"minimax:nodes={quota.VISITOR_NODES}"
    OUTSIDE = "fairystockfish:nodes=200000"

    def test_hard_is_the_line(self) -> None:
        assert self.HARD == "minimax:nodes=6400"
        assert not quota.members_only(self.HARD)
        assert not quota.members_only(AGENT)
        assert not quota.members_only("mcts:nodes=1600")
        assert not quota.members_only("random")
        assert quota.members_only(STRONGEST)
        assert quota.members_only(self.OUTSIDE)
        # Every strong seat is a members' seat, so a visitor never holds one.
        assert quota.STRONG_NODES > quota.VISITOR_NODES

    def test_a_visitor_plays_hard_and_is_refused_hardest(self, site: Site) -> None:
        site.game()
        matches.start(site.store, DULL_HASH, matches.SANDBOX, ["human", self.HARD], None)
        rows = site.count("matches")
        for seats in (["human", STRONGEST], [STRONGEST, "human"], [STRONGEST, AGENT],
                      ["human", self.OUTSIDE]):
            with pytest.raises(matches.MatchError, match="sign in to play it"):
                matches.start(site.store, DULL_HASH, matches.SANDBOX, seats, None)
        assert site.count("matches") == rows

    def test_a_signed_in_person_plays_hardest(self, site: Site) -> None:
        site.game()
        matches.start(
            site.store, DULL_HASH, matches.SANDBOX, ["human", STRONGEST], site.person("ada")
        )
        assert quota.strong_seats(site.store) == 1

    def test_the_refusal_names_the_player_and_the_way_in(self) -> None:
        said = matches.members_only_said(STRONGEST)
        assert said == (
            "hardest is for people with an account: sign in to play it. Without one you "
            "can play every computer player up to hard"
        )
        assert "minimax" not in said and "nodes" not in said

    def test_the_route_refuses_a_visitor_in_a_sentence(self, site: Site) -> None:
        site.game()
        visitor = site.browser()
        refused = visitor.post(
            "/api/session",
            {"hash": DULL_HASH, "seats": ["human", STRONGEST], "mode": matches.SANDBOX},
        )
        assert refused.status == 400
        assert "sign in to play it" in refused.payload["error"]
        played = visitor.post(
            "/api/session",
            {"hash": DULL_HASH, "seats": ["human", self.HARD], "mode": matches.SANDBOX},
        )
        assert played.status == 200, played.payload

    def test_the_list_says_so_to_a_visitor_and_not_to_a_member(self, site: Site) -> None:
        site.game()
        said = site.browser().get(f"/api/game/{DULL_HASH}").payload["standing"]["signInFor"]
        assert said.get(STRONGEST) == matches.SIGN_IN_TO_PLAY == "sign in to play"
        assert self.HARD not in said and AGENT not in said
        assert all(quota.members_only(agent) for agent in said)
        member = site.signed_in("ada")
        assert member.get(f"/api/game/{DULL_HASH}").payload["standing"]["signInFor"] == {}


class TestAWatchAgainstTheStrongestSeat:
    """Q301: a watch's strong seats are counted together, never each against the same one.

    A watch is two computer players, so it can hold two strong seats. The cap used to be
    asked once per seat, and one seat short of it both seats found the last one free and
    the site ended one over. Each test fills the site to a point with somebody else's
    game, because the cap is site-wide, and then asks what the next person may start.
    """

    LIMIT = 3
    """An odd cap, so a watch of two strong players cannot land on it exactly."""

    def _live(self, site: Site, monkeypatch: pytest.MonkeyPatch, live: int) -> int:
        """The site with `live` strong seats taken by bob, and the next person's id."""
        monkeypatch.setattr(quota, "STRONG_SEATS", self.LIMIT)
        site.game()
        bob = site.person("bob")
        for _ in range(live):
            matches.start(site.store, DULL_HASH, matches.SANDBOX, ["human", STRONGEST], bob)
        site.store.commit()
        assert quota.strong_seats(site.store) == live
        return site.person("ada")

    def test_one_short_a_watch_with_one_strong_seat_starts(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ada = self._live(site, monkeypatch, self.LIMIT - 1)

        matches.start(site.store, DULL_HASH, matches.SANDBOX, [STRONGEST, AGENT], ada)
        assert quota.strong_seats(site.store) == self.LIMIT

    def test_one_short_a_watch_with_two_strong_seats_is_refused(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ada = self._live(site, monkeypatch, self.LIMIT - 1)
        rows = site.count("matches")

        with pytest.raises(quota.QuotaError, match="only one is free"):
            matches.start(
                site.store, DULL_HASH, matches.SANDBOX, [STRONGEST, STRONGEST], ada
            )
        assert site.count("matches") == rows, "the refused watch left a row behind"
        assert quota.strong_seats(site.store) == self.LIMIT - 1

    def test_a_visitor_one_short_is_refused_before_the_cap(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A visitor never reaches the cap: the strongest is for signed-in people (L15),
        and that refusal comes first and takes no seat."""
        self._live(site, monkeypatch, self.LIMIT - 1)

        with pytest.raises(matches.MatchError, match="sign in to play it"):
            matches.start(site.store, DULL_HASH, matches.SANDBOX, [STRONGEST, STRONGEST])
        assert quota.strong_seats(site.store) == self.LIMIT - 1

    @pytest.mark.parametrize(
        ("seats", "said"),
        [
            ([STRONGEST, AGENT], "hardest is busy"),
            ([AGENT, STRONGEST], "hardest is busy"),
            ([STRONGEST, STRONGEST], "none is free"),
        ],
    )
    def test_at_the_limit_no_watch_with_a_strong_seat_starts(
        self, site: Site, monkeypatch: pytest.MonkeyPatch, seats: list[str], said: str
    ) -> None:
        ada = self._live(site, monkeypatch, self.LIMIT)

        with pytest.raises(quota.QuotaError, match=said):
            matches.start(site.store, DULL_HASH, matches.SANDBOX, seats, ada)
        assert quota.strong_seats(site.store) == self.LIMIT

    def test_at_the_limit_a_watch_of_cheaper_players_still_starts(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ada = self._live(site, monkeypatch, self.LIMIT)
        matches.start(site.store, DULL_HASH, matches.SANDBOX, [AGENT, AGENT], ada)

    def test_the_check_itself_counts_every_seat_it_is_given(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One call for the whole match, so no loop over seats can pass it twice."""
        ada = self._live(site, monkeypatch, self.LIMIT - 1)

        quota.check_new_match(site.store, ada, STRONGEST)
        quota.check_new_match(site.store, ada, STRONGEST, AGENT)
        with pytest.raises(quota.QuotaError):
            quota.check_new_match(site.store, ada, STRONGEST, STRONGEST)

    @pytest.mark.parametrize(("live", "starts"), [(LIMIT - 1, True), (LIMIT, False)])
    def test_a_game_somebody_plays_is_checked_as_it_was(
        self, site: Site, monkeypatch: pytest.MonkeyPatch, live: int, starts: bool
    ) -> None:
        """One person, one strong opponent: starts one short, refused at the limit, and
        refused in the sentence it always was."""
        ada = self._live(site, monkeypatch, live)
        seats = ["human", STRONGEST]

        if starts:
            matches.start(site.store, DULL_HASH, matches.SANDBOX, seats, ada)
            assert quota.strong_seats(site.store) == self.LIMIT
            return
        with pytest.raises(quota.QuotaError) as refused:
            matches.start(site.store, DULL_HASH, matches.SANDBOX, seats, ada)
        assert str(refused.value) == (
            f"hardest is busy: it plays {self.LIMIT} games at once and all {self.LIMIT} "
            f"are running; pick a weaker opponent, or come back in a few minutes"
        )


@pytest.mark.usefixtures("corpus_listed")
class TestWhatThePageIsToldAboutTheViewer:
    """One boolean, so a header can offer the link to the people it opens for.

    The link itself is `chess_variant_sandbox/web/page/`'s and is Q197's; what the site owes it is the
    answer, beside the `signedIn` the mode control already reads.
    """

    def test_the_listing_and_a_game_both_say_whether_this_is_an_admin(
        self, site: Site
    ) -> None:
        site.game()
        boss = site.signed_in("boss", admin.ADMIN)
        member = site.signed_in("ada")

        assert boss.get("/api/library").payload["site"]["admin"] is True
        assert member.get("/api/library").payload["site"]["admin"] is False
        assert site.browser().get("/api/library").payload["site"]["admin"] is False

        standing = boss.get(f"/api/game/{DULL_HASH}").payload["standing"]
        assert standing["admin"] is True
        assert standing["signedIn"] is True

    def test_a_killed_admin_stops_being_one_on_the_page(self, site: Site) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        admin.kill_user(site.store, _id(site, "boss"))
        site.store.commit()
        assert boss.get("/api/library").payload["site"]["admin"] is False


# ----------------------------------------------------------------- the route table

class TestTheRouteTable:
    def test_the_two_admin_routes_are_the_sites_own(self) -> None:
        """They are not the local application's, so they belong in `OWN` beside the vote
        and the profile rather than in the allowlist `tests/test_site.py` compares."""
        paths = {route.path for route in routes.OWN}
        assert "/api/admin" in paths
        assert routes.find("GET", "/api/admin") is not None
        assert routes.find("POST", "/api/admin") is not None

    def test_the_delete_is_the_sites_own_and_asks_for_a_sign_in(self) -> None:
        """Q303. The local application keeps no accounts, so there is nobody whose game it
        could be; the route is the site's own and a visitor is refused before the handler
        runs. It is not admin-only either -- the creator of a game reaches it too."""
        route = routes.find("POST", f"/api/game/{DULL_HASH}/delete")
        assert route in routes.OWN
        assert route is not None and route.visitor.startswith("Sign in")
        assert "/api/game/<hash>/delete" not in routes.ADMIN_ONLY

    def test_every_admin_only_path_is_one_the_site_actually_serves(self) -> None:
        served = set(routes.PAGES) | {route.path for route in routes.OWN}
        assert set(routes.ADMIN_ONLY) <= served


# ------------------------------------------------------------------------- small helpers


def _id(site: Site, handle: str) -> int:
    with site.store.cursor() as cursor:
        cursor.execute("select id from users where handle = %s", (handle,))
        return int(cursor.fetchone()[0])


def _invitations(site: Site, email: str) -> list[tuple[str, int | None]]:
    """Each invitation to this address as its code and issuer, oldest first."""
    with site.store.cursor() as cursor:
        cursor.execute(
            "select code, issuer from invites where email = %s order by created, code",
            (email,),
        )
        return [(str(row[0]), row[1]) for row in cursor.fetchall()]


def _post_without_header(
    port: int, cookie: str | None, path: str, payload: dict[str, Any]
) -> int:
    """A POST as a form on another site could make it, with the cookie and no header."""
    headers = {"Content-Type": "application/json"}
    if cookie is not None:
        headers["Cookie"] = f"{site_pages.COOKIE}={cookie}"
    connection = HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        connection.request("POST", path, body=json.dumps(payload), headers=headers)
        found = connection.getresponse()
        found.read()
        return int(found.status)
    finally:
        connection.close()


def _backdate(site: Site, match_id: str, ago: timedelta) -> None:
    """Make a match look as if nothing has happened on it for `ago`."""
    with site.store.cursor() as cursor:
        cursor.execute(
            "update matches set started = started - %s,"
            " moved = coalesce(moved, started) - %s where id = %s",
            (ago, ago, match_id),
        )
    site.store.commit()
