"""Report a game: who may, how often, where it lands, and what it does not do.

Q503d, `docs/plan-game-pipeline.md` §5. The acceptance is three sentences and each is a
class below.

**A signed-in person reports a game from its page.** Over HTTP, because a visitor's
refusal is a route-table answer, and the game page's standing is what the page draws the
link from.

**One report a person a game, 10 a day.** The first is the table's primary key and is
asked of the table directly; the second is `quota.REPORTS_A_DAY` through the route.

**Listed on `/admin` with the kill switch beside it.** The admin reads every report with
the game's switch state, and the kill it presses is the games table's own act. A report
by itself hides nothing.

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

from chess_variant_sandbox.site import admin, db, identity, quota, reports
from chess_variant_sandbox.site import pages as site_pages
from chess_variant_sandbox.site import server as site_server
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.server import REQUEST_HEADER, REQUEST_VALUE, Server

SERVE_FAST = {"poll_interval": 0.02}
PASSWORD = "a-long-enough-one"
DULL = json.loads((library.RULESET_DIR / "dull.json").read_text(encoding="utf-8"))
DULL_HASH = spec_hash(DULL)


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

    def report(self, digest: str, reason: str = reports.SPAM, note: str = "") -> Answer:
        return self.post(f"/api/game/{digest}/report", {"reason": reason, "note": note})

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


class Site:
    """A running site, its store, and the things a test sets up by hand."""

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
                (handle, f"{handle}@example.invalid", identity.hash_password(PASSWORD), role),
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

    def game(self, digest: str = DULL_HASH, name: str | None = None) -> str:
        """A corpus row. `dull` itself by default, the game the library lists; any other
        digest is a row of dull's rules under a name of its own, which is all a report
        reads."""
        with self.store.cursor() as cursor:
            cursor.execute(
                "insert into games (hash, spec, name, origin) values (%s, %s::jsonb, %s, %s)",
                (digest, json.dumps(DULL), name or DULL["name"], "corpus"),
            )
        self.store.commit()
        return digest

    def rows(self) -> list[tuple[Any, ...]]:
        with self.store.cursor() as cursor:
            cursor.execute("select reporter, game, reason, note from reports order by created")
            return list(cursor.fetchall())

    def listed(self) -> set[str]:
        payload = self.browser().get("/api/library").payload
        return {str(game["hash"]) for game in payload["games"]}


@contextlib.contextmanager
def _running_site() -> Iterator[Site]:
    url, name = site_store.emptied("no database to file a report in")

    def open_one() -> Any:
        connection = db.connect(url)
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {name}")
        return connection

    store = open_one()
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


def _other(number: int) -> str:
    """A made-up digest for the day's cap, one per game."""
    return f"{number:064x}"


# --------------------------------------------------------- reporting from the game page


class TestReportingAGame:
    def test_a_signed_in_person_reports_a_game_and_the_row_is_filed(self, site: Site) -> None:
        digest = site.game()
        ada = site.signed_in("ada")

        page = ada.get(f"/api/game/{digest}").payload["standing"]
        assert page["yourReport"] is None
        assert [one["reason"] for one in page["reportReasons"]] == ["spam", "offensive", "copy"]

        filed = ada.report(digest, reports.COPY, "  the same as chess  ")
        assert filed.status == 200, filed.text
        assert filed.payload["label"] == reports.LABELS[reports.COPY]
        assert "Reported" in filed.payload["said"]

        who = site.rows()
        assert len(who) == 1
        assert who[0][1:] == (digest, "copy", "the same as chess")

        # The page now draws the line saying so, rather than the link.
        again = ada.get(f"/api/game/{digest}").payload["standing"]
        assert again["yourReport"] == {"reason": "copy", "label": reports.LABELS["copy"]}

    def test_the_note_is_optional_and_stored_as_null_when_empty(self, site: Site) -> None:
        digest = site.game()
        assert site.signed_in("ada").report(digest, reports.OFFENSIVE).status == 200
        assert site.rows()[0][3] is None

    def test_a_visitor_is_refused_and_offered_no_link(self, site: Site) -> None:
        digest = site.game()
        visitor = site.browser()
        refused = visitor.report(digest)
        assert refused.status == 400
        assert "Sign in to report a game" in refused.payload["error"]
        assert site.rows() == []

        standing = visitor.get(f"/api/game/{digest}").payload["standing"]
        assert standing["signedIn"] is False
        assert "reportReasons" not in standing and "yourReport" not in standing

    def test_a_reason_outside_the_three_is_refused(self, site: Site) -> None:
        digest = site.game()
        refused = site.signed_in("ada").report(digest, "boring")
        assert refused.status == 400
        assert "'boring'" in refused.payload["error"]
        assert site.rows() == []

    def test_a_note_over_the_cap_is_refused_and_one_at_it_is_not(self, site: Site) -> None:
        digest = site.game()
        ada = site.signed_in("ada")
        refused = ada.report(digest, reports.SPAM, "x" * (quota.REPORT_NOTE_LENGTH + 1))
        assert refused.status == 400
        assert str(quota.REPORT_NOTE_LENGTH) in refused.payload["error"]
        assert ada.report(digest, reports.SPAM, "x" * quota.REPORT_NOTE_LENGTH).status == 200

    def test_a_killed_game_and_an_unknown_one_are_the_same_404(self, site: Site) -> None:
        digest = site.game()
        with site.store.cursor() as cursor:
            admin.kill_game(cursor.connection, digest)
        site.store.commit()
        ada = site.signed_in("ada")
        killed = ada.report(digest)
        unknown = ada.report(_other(7))
        assert killed.status == unknown.status == 404
        assert site.rows() == []

    def test_a_report_hides_nothing_by_itself(self, site: Site) -> None:
        """Killing stays the maintainer's act: a report files a row and that is all."""
        digest = site.game()
        assert digest in site.listed()
        for handle in ("ada", "grace", "alan"):
            assert site.signed_in(handle).report(digest).status == 200
        assert digest in site.listed()
        assert site.browser().get(f"/api/game/{digest}").status == 200


# ------------------------------------------------------------------------- the limits


class TestTheLimits:
    def test_one_report_a_person_a_game(self, site: Site) -> None:
        digest = site.game()
        ada = site.signed_in("ada")
        assert ada.report(digest).status == 200
        second = ada.report(digest, reports.OFFENSIVE)
        assert second.status == 400
        assert "already reported" in second.payload["error"]
        assert [row[2] for row in site.rows()] == ["spam"]

        # Somebody else may report the same game.
        assert site.signed_in("grace").report(digest).status == 200
        assert len(site.rows()) == 2

    def test_once_is_the_tables_rule_as_well_as_the_modules(self, site: Site) -> None:
        digest = site.game()
        ada = site.person("ada")
        with site.store.cursor() as cursor:
            cursor.execute(
                "insert into reports (reporter, game, reason) values (%s, %s, 'spam')",
                (ada, digest),
            )
        site.store.commit()
        with (
            pytest.raises(Exception) as error,  # noqa: PT011 - the driver's own class
            site.store.cursor() as cursor,
        ):
            cursor.execute(
                "insert into reports (reporter, game, reason) values (%s, %s, 'copy')",
                (ada, digest),
            )
        site.store.rollback()
        assert "reports_pkey" in str(error.value)

    def test_the_reason_and_the_note_are_the_columns_rules_too(self, site: Site) -> None:
        digest = site.game()
        ada = site.person("ada")
        for reason, note, constraint in (
            ("boring", None, "reports_reason_check"),
            ("spam", "x" * (quota.REPORT_NOTE_LENGTH + 1), "reports_note_check"),
        ):
            with (
                pytest.raises(Exception) as error,  # noqa: PT011 - the driver's own class
                site.store.cursor() as cursor,
            ):
                cursor.execute(
                    "insert into reports (reporter, game, reason, note)"
                    " values (%s, %s, %s, %s)",
                    (ada, digest, reason, note),
                )
            site.store.rollback()
            assert constraint in str(error.value)

    def test_ten_a_day_and_the_next_is_refused(self, site: Site) -> None:
        games = [site.game(_other(n), f"game {n}") for n in range(quota.REPORTS_A_DAY + 1)]
        ada = site.signed_in("ada")
        for digest in games[:-1]:
            answer = ada.report(digest)
            assert answer.status == 200, answer.text
        refused = ada.report(games[-1])
        assert refused.status == 400
        assert str(quota.REPORTS_A_DAY) in refused.payload["error"]
        assert "rolls" in refused.payload["error"]
        assert len(site.rows()) == quota.REPORTS_A_DAY

        # Per person: somebody else still has a whole day.
        assert site.signed_in("grace").report(games[-1]).status == 200

    def test_the_day_is_rolling(self, site: Site) -> None:
        games = [site.game(_other(n), f"game {n}") for n in range(quota.REPORTS_A_DAY + 1)]
        ada = site.signed_in("ada")
        for digest in games[:-1]:
            assert ada.report(digest).status == 200
        with site.store.cursor() as cursor:
            cursor.execute(
                "update reports set created = now() - interval '25 hours'"
                " where game = %s",
                (games[0],),
            )
        site.store.commit()
        assert ada.report(games[-1]).status == 200

    def test_the_limit_is_on_the_admin_pages_list(self) -> None:
        labels = dict(quota.summary())
        assert labels["Games one person may report a day"] == str(quota.REPORTS_A_DAY)


# ------------------------------------------------------------ the maintainer's page


class TestTheMaintainersList:
    def test_an_admin_reads_every_report_with_the_game_and_who(self, site: Site) -> None:
        digest = site.game()
        site.signed_in("ada").report(digest, reports.COPY, "<b>chess</b>")
        site.signed_in("grace").report(digest, reports.SPAM)
        boss = site.signed_in("boss", role=admin.ADMIN)

        listed = boss.get("/api/admin").payload["reports"]
        assert {(one["handle"], one["reason"], one["note"]) for one in listed} == {
            ("ada", "copy", "<b>chess</b>"),
            ("grace", "spam", ""),
        }
        assert {one["hash"] for one in listed} == {digest}
        assert {one["name"] for one in listed} == {DULL["name"]}
        assert {one["killed"] for one in listed} == {False}
        # Newest first.
        assert listed[0]["handle"] == "grace"

    def test_the_kill_beside_it_takes_the_game_off_and_puts_it_back(
        self, site: Site
    ) -> None:
        digest = site.game()
        site.signed_in("ada").report(digest)
        boss = site.signed_in("boss", role=admin.ADMIN)

        killed = boss.post("/api/admin", {"act": "kill-game", "hash": digest})
        assert killed.status == 200, killed.text
        assert boss.get("/api/admin").payload["reports"][0]["killed"] is True
        assert digest not in site.listed()

        # The report stays, so the maintainer can see why it was taken off.
        assert len(site.rows()) == 1

        assert boss.post("/api/admin", {"act": "revive-game", "hash": digest}).status == 200
        assert boss.get("/api/admin").payload["reports"][0]["killed"] is False
        assert digest in site.listed()

    def test_dismissing_hides_the_report_and_keeps_the_row_so_none_can_repeat_it(
        self, site: Site
    ) -> None:
        digest = site.game()
        ada = site.signed_in("ada")
        ada.report(digest)
        site.signed_in("grace").report(digest)
        boss = site.signed_in("boss", role=admin.ADMIN)
        listed = boss.get("/api/admin").payload["reports"]
        one = next(row for row in listed if row["handle"] == "ada")

        dismissed = boss.post(
            "/api/admin", {"act": "dismiss-report", "reporter": one["reporter"], "hash": digest}
        )
        assert dismissed.status == 200, dismissed.text

        # It leaves the list, and only that one does.
        left = boss.get("/api/admin").payload["reports"]
        assert [row["handle"] for row in left] == ["grace"]
        # The row is still there, so ada is still refused.
        assert len(site.rows()) == 2
        again = ada.report(digest)
        assert again.status == 400
        assert "already reported" in again.payload["error"]
        assert len(site.rows()) == 2
        # Dismissing twice is harmless; an unknown report is a 404.
        body = {"act": "dismiss-report", "reporter": one["reporter"], "hash": digest}
        assert boss.post("/api/admin", body).status == 200
        missing = {"act": "dismiss-report", "reporter": 999999, "hash": digest}
        assert boss.post("/api/admin", missing).status == 404

    def test_a_dismissed_report_is_listed_apart_newest_dismissal_first(
        self, site: Site
    ) -> None:
        """Q518: `/admin` can look back on what was dismissed, with when."""
        digest = site.game()
        site.signed_in("ada").report(digest, reports.SPAM, "looks like chess")
        site.signed_in("grace").report(digest)
        boss = site.signed_in("boss", role=admin.ADMIN)
        before = boss.get("/api/admin").payload
        assert before["dismissedReports"] == []
        assert before["dismissedCount"] == 0

        by_handle = {row["handle"]: row for row in before["reports"]}
        for handle in ("ada", "grace"):
            body = {
                "act": "dismiss-report", "reporter": by_handle[handle]["reporter"],
                "hash": digest,
            }
            assert boss.post("/api/admin", body).status == 200
        # Grace's dismissal is later by the clock the store keeps, so it comes first.
        with site.store.cursor() as cursor:
            cursor.execute(
                "update reports set dismissed = dismissed - interval '1 hour'"
                " where reporter = %s",
                (by_handle["ada"]["reporter"],),
            )
        site.store.commit()

        after = boss.get("/api/admin").payload
        assert after["reports"] == []
        assert after["dismissedCount"] == 2
        shown = after["dismissedReports"]
        assert [row["handle"] for row in shown] == ["grace", "ada"]
        assert all(row["dismissed"] for row in shown)
        assert shown[1]["note"] == "looks like chess"
        assert shown[1]["label"] == reports.LABELS[reports.SPAM]
        assert shown[1]["name"] == DULL["name"]

    def test_the_dismissed_list_is_capped_and_counts_them_all(self, site: Site) -> None:
        digest = site.game()
        for handle in ("ada", "grace", "alan"):
            site.signed_in(handle).report(digest)
        with site.store.cursor() as cursor:
            cursor.execute("update reports set dismissed = now()")
            shown, count = reports.dismissed(cursor.connection, limit=2)
        site.store.commit()
        assert count == 3
        assert len(shown) == 2

    def test_a_member_never_sees_the_dismissed_ones(self, site: Site) -> None:
        digest = site.game()
        ada = site.signed_in("ada")
        ada.report(digest, reports.SPAM, "a dismissed secret")
        with site.store.cursor() as cursor:
            cursor.execute("update reports set dismissed = now()")
        site.store.commit()
        refused = ada.get("/api/admin")
        assert refused.status == 404
        assert "a dismissed secret" not in refused.text

    def test_a_member_cannot_dismiss(self, site: Site) -> None:
        digest = site.game()
        ada = site.signed_in("ada")
        ada.report(digest)
        with site.store.cursor() as cursor:
            cursor.execute("select reporter from reports")
            reporter = int(cursor.fetchone()[0])
        refused = ada.post(
            "/api/admin", {"act": "dismiss-report", "reporter": reporter, "hash": digest}
        )
        assert refused.status == 404
        with site.store.cursor() as cursor:
            cursor.execute("select count(*) from reports where dismissed is not null")
            assert cursor.fetchone()[0] == 0
        site.store.commit()
        assert site.browser().post(
            "/api/admin", {"act": "dismiss-report", "reporter": reporter, "hash": digest}
        ).status in (401, 404)

    def test_a_member_never_sees_the_list(self, site: Site) -> None:
        digest = site.game()
        ada = site.signed_in("ada")
        ada.report(digest, reports.SPAM, "a secret note")
        refused = ada.get("/api/admin")
        assert refused.status == 404
        assert "a secret note" not in refused.text

    def test_a_reporter_who_left_reads_as_a_former_player(self, site: Site) -> None:
        digest = site.game()
        site.signed_in("ada").report(digest)
        with site.store.cursor() as cursor:
            admin.forget(cursor.connection, "ada")
        site.store.commit()
        boss = site.signed_in("boss", role=admin.ADMIN)
        assert boss.get("/api/admin").payload["reports"][0]["handle"] == identity.FORMER


# ------------------------------------------------------------------------ the pages


class TestThePages:
    def test_the_admin_page_has_the_table_and_draws_the_kill_beside_each(self) -> None:
        page = page_text.inlined(site_pages.admin("boss"))
        assert "<h2>Reports</h2>" in page
        assert '<tbody id="reports"></tbody>' in page
        drawn = page[page.index("function reports(rows)"):]
        drawn = drawn[: drawn.index("\n}\n")]
        assert "act({act: 'kill-game', hash: one.hash}, 'reports-note')" in drawn
        assert "act({act: 'revive-game', hash: one.hash}, 'reports-note')" in drawn
        assert (
            "act({act: 'dismiss-report', reporter: one.reporter, hash: one.hash},\n"
            "        'reports-note')" in drawn
        )
        # Every cell a text node: no markup written from a report.
        assert "innerHTML" not in drawn

    def test_a_press_on_a_report_is_answered_beside_the_reports(self) -> None:
        """Q517: the answer went to the note at the top of `/admin`, out of sight of the
        button a long scroll below it. The table's own note sits under it."""
        page = page_text.inlined(site_pages.admin("boss"))
        table = page.index('<tbody id="reports"></tbody>')
        note = page.index('<p class="note" id="reports-note"></p>')
        assert table < note < page.index("<h2>Suggestions</h2>")
        drawn = page[page.index("function reports(rows)"):]
        drawn = drawn[: drawn.index("\n}\n")]
        # Every press on a report row speaks there, and none at the top of the page.
        assert drawn.count("'reports-note')") == 3
        assert drawn.count("act({") == 3

    def test_the_admin_page_folds_the_dismissed_reports_under_the_open_ones(self) -> None:
        page = page_text.inlined(site_pages.admin("boss"))
        note = page.index('<p class="note" id="reports-note"></p>')
        folded = page.index('<details id="dismissed-reports">')
        assert note < folded < page.index('<tbody id="dismissed"></tbody>')
        assert page.index('<tbody id="dismissed"></tbody>') < page.index("<h2>Suggestions</h2>")
        assert "dismissedReports(payload.dismissedReports || [], payload.dismissedCount || 0)" in page
        drawn = page[page.index("function dismissedReports(rows, count)"):]
        drawn = drawn[: drawn.index("\n}\n")]
        # Read only, and every cell a text node.
        assert "act({" not in drawn
        assert "innerHTML" not in drawn
        assert "day(one.dismissed)" in drawn

    def test_the_game_page_carries_the_quiet_link_under_the_rules(self) -> None:
        page = page_text.PAGE
        assert "Report this game (spam, offensive, a copy)" in page
        assert "'/api/game/' + game.hash + '/report'" in page
        # Inside the rules' section (V6 folds the game page into sections), after where the
        # pieces start and before the next section.
        rules = page.index("(game.setupLines || [])")
        assert rules < page.index("${reportHtml(game)}") < page.index("sectionHtml('kind'")
