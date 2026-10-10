"""A visitor asks for an invitation, and the maintainer grants it or not (Q831).

`docs/plan-website.md`, *Public or invite-only, indefinitely*, decided 2026-10-10: *"Make a
request invite button, with an optional message, suggesting they say how they found the
site."* `site/asking.py` has the rules; each is asked here over HTTP against a running site
on a real PostgreSQL, because each is a thing a stranger does to a running site.

- **The way there** is a button wherever a visitor meets the invitation wall: the sign-in
  page, the privacy notice, a challenge link opened without an account.
- **The request is kept and reaches `/admin`**, which grants it (the ordinary invitation,
  through `identity.send_invitation`) or declines it (nothing sent).
- **The visitor reads one sentence**, whether a row was written or not: an address that
  already has an account, or already has a request kept, writes nothing and says so to
  nobody.
- **The limits**: the address and the note checked and capped, one kept request an
  address, `quota.ASKS_A_DAY` site-wide, `quota.ASKS_A_SOURCE` a connection, and no mail
  but the invitation the maintainer chooses to send.

Everything with a store needs `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL` and skips without it;
`tests/test_site.py` has the recipe for a scratch cluster.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

import pytest
import site_words
from test_site_admin import (
    PASSWORD,
    SITE_URL,
    Answer,
    Browser,
    Site,
    _id,
    _invitations,
    _running_site,
)

from chess_variant_sandbox.site import admin, asking, identity, mail, quota, routes
from chess_variant_sandbox.site import pages as site_pages
from chess_variant_sandbox.web.server import REQUEST_HEADER, REQUEST_VALUE

ROUTE = "/api/ask-invitation"
STATIC = Path(site_pages.__file__).resolve().parent / "static"


@pytest.fixture
def site(monkeypatch: pytest.MonkeyPatch) -> Iterator[Site]:
    """A running site whose connection cap is out of the way, for the tests about the rest.

    Every request here comes from one address, the test's; the cap per connection has a
    test of its own below, with the cap as it is.
    """
    monkeypatch.setenv(mail.URL_VAR, SITE_URL)
    monkeypatch.delenv(mail.KEY_VAR, raising=False)
    monkeypatch.delenv(admin.NAME_VAR, raising=False)
    monkeypatch.delenv(admin.PASSWORD_VAR, raising=False)
    monkeypatch.setattr(quota, "ASKS_A_SOURCE", 1000)
    with _running_site() as running:
        yield running


def _ask(site: Site, email: Any, note: Any = None, *, source: str | None = None) -> Answer:
    """One request for an invitation, from a visitor, optionally through the proxy's header."""
    headers = {"Content-Type": "application/json", REQUEST_HEADER: REQUEST_VALUE}
    if source is not None:
        headers["X-Forwarded-For"] = source
    body: dict[str, Any] = {"email": email}
    if note is not None:
        body["note"] = note
    connection = HTTPConnection("127.0.0.1", site.port, timeout=30)
    try:
        connection.request("POST", ROUTE, body=json.dumps(body).encode("utf-8"), headers=headers)
        found = connection.getresponse()
        return Answer(found.status, found.read())
    finally:
        connection.close()


def _rows(site: Site) -> list[tuple[Any, ...]]:
    with site.store.cursor() as cursor:
        cursor.execute(
            "select email, note, answered is not null, granted from invite_asks order by id"
        )
        found = cursor.fetchall()
    site.store.commit()
    return [tuple(row) for row in found]


def _pending(boss: Browser) -> list[dict[str, Any]]:
    answer = boss.get("/api/admin")
    assert answer.status == 200, answer.text
    return list(answer.payload["asks"])


def _confirmed(answer: Answer) -> None:
    assert answer.status == 200, answer.text
    assert answer.payload == {"said": asking.CONFIRMED}


# ------------------------------------------------------------------- the way there


class TestTheWayThere:
    """Wherever a visitor meets the invitation wall, the button is there."""

    def test_the_sign_in_page_offers_the_button_beside_the_wall(self) -> None:
        page = site_pages.signin()
        wall = page.index("Accounts are by invitation for now")
        button = page.index('id="ask-invitation"')
        assert wall < button
        assert f'<a class="button" id="ask-invitation" href="{asking.PATH}">' in page
        assert f">{asking.BUTTON}</a>" in page
        assert asking.BUTTON == "Request an invitation"

    def test_the_privacy_notice_says_what_a_request_keeps_and_offers_the_button(self) -> None:
        page = site_pages.privacy()
        part = page[page.index("<h2>Asking for an invitation</h2>"):]
        part = part[: part.index("<h2>", 5)]
        assert "keeps the email address and the note" in part
        assert "no other mail goes to it" in part
        assert f"deleted {quota.ASK_KEPT.days} days after it is answered" in part
        assert f'href="{asking.PATH}"' in part
        # A visitor's request is not one of a member's six things.
        assert "It keeps six things and nothing else." in page

    def test_a_challenge_link_opened_without_an_account_offers_it_too(self) -> None:
        script = (STATIC / "challenge.js").read_text(encoding="utf-8")
        visitor = script[script.index("if (says.signin) {"):]
        visitor = visitor[: visitor.index("if (says.play)")]
        assert f"ask.href = '{asking.PATH}';" in visitor
        assert f"ask.textContent = '{asking.BUTTON}';" in visitor

    def test_the_form_is_an_address_an_optional_note_and_send(self) -> None:
        page = site_pages.request_invitation()
        assert f"<h1>{asking.BUTTON}</h1>" in page
        assert '<input id="ask-email" type="email"' in page
        assert f'maxlength="{quota.ASK_EMAIL_LENGTH}" required' in page
        assert f'<textarea id="ask-note" rows="3" maxlength="{quota.ASK_NOTE_LENGTH}">' in page
        assert "a note, if you like" in page
        # The user's own suggestion, as the note's hint.
        assert asking.NOTE_HINT in page
        assert "How did you find the site?" in asking.NOTE_HINT
        assert 'id="ask-send" type="submit">send the request</button>' in page
        assert '<script src="/static/site/ask.js"></script>' in page

    def test_the_route_is_open_to_a_visitor_and_the_page_is_a_constant(self) -> None:
        route = routes.find("POST", ROUTE)
        assert route is not None and route in routes.OWN
        assert route.visitor == ""
        assert asking.PATH in routes.PAGES
        assert asking.PATH not in routes.PERSONAL

    def test_the_script_puts_the_answer_where_the_form_was_as_text(self) -> None:
        script = (STATIC / "ask.js").read_text(encoding="utf-8")
        assert f"post('{ROUTE}'," in script
        assert "done.textContent = answer.said;" in script
        assert "form.replaceChildren(done);" in script
        assert "innerHTML" not in script
        assert "sending.disabled = true;" in script

    def test_a_visitor_reaches_the_page_and_the_sign_in_page_over_http(
        self, site: Site
    ) -> None:
        visitor = site.browser()
        page = visitor.get(asking.PATH)
        assert page.status == 200
        assert 'id="ask-form"' in page.text
        assert f'href="{asking.PATH}"' in visitor.get("/signin").text


# ---------------------------------------------------------- kept, and on the panel


class TestTheRequestReachesThePanel:
    def test_a_request_is_kept_and_listed_for_the_admin_with_nothing_mailed(
        self, site: Site
    ) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        _confirmed(_ask(site, " Eve@Example.com ", "A friend on a chess forum."))
        assert _rows(site) == [("eve@example.com", "A friend on a chess forum.", False, None)]
        [listed] = _pending(boss)
        assert listed["email"] == "eve@example.com"
        assert listed["note"] == "A friend on a chess forum."
        assert listed["created"]
        assert listed["invited"] is None
        assert site.sent.messages == []

    def test_the_note_is_optional(self, site: Site) -> None:
        _confirmed(_ask(site, "fay@example.com"))
        _confirmed(_ask(site, "gus@example.com", "   \n  "))
        assert [row[:2] for row in _rows(site)] == [
            ("fay@example.com", None), ("gus@example.com", None)
        ]

    def test_pending_requests_are_listed_oldest_first(self, site: Site) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        for name in ("one", "two", "three"):
            _confirmed(_ask(site, f"{name}@example.com"))
        assert [row["email"] for row in _pending(boss)] == [
            "one@example.com", "two@example.com", "three@example.com"
        ]

    def test_a_member_and_a_visitor_cannot_read_or_answer_them(self, site: Site) -> None:
        _confirmed(_ask(site, "eve@example.com"))
        ask = _rows_ids(site)[0]
        for browser in (site.signed_in("cass"), site.browser()):
            assert browser.get("/api/admin").status == 404
            for act in ("grant-ask", "decline-ask"):
                assert browser.post("/api/admin", {"act": act, "ask": ask}).status == 404
        assert _rows(site) == [("eve@example.com", None, False, None)]
        assert site.sent.messages == []


def _rows_ids(site: Site) -> list[int]:
    with site.store.cursor() as cursor:
        cursor.execute("select id from invite_asks order by id")
        found = [int(row[0]) for row in cursor.fetchall()]
    site.store.commit()
    return found


# ------------------------------------------------------------- granted or declined


class TestTheAdminAnswers:
    def test_granting_sends_the_ordinary_invitation_from_the_admin(self, site: Site) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        _confirmed(_ask(site, "eve@example.com", "Saw it on a forum."))
        [listed] = _pending(boss)
        answer = boss.post("/api/admin", {"act": "grant-ask", "ask": listed["id"]})
        assert answer.status == 200, answer.text
        assert answer.payload["said"] == "invited eve@example.com: the link is on its way"
        [(code, issuer)] = _invitations(site, "eve@example.com")
        assert issuer == _id(site, "boss")
        # The one message: the ordinary invitation, naming the admin.
        assert site.sent.messages == [
            mail.invitation("eve@example.com", identity.link(SITE_URL, code), "boss")
        ]
        assert _pending(boss) == []
        assert boss.get("/api/admin").payload["asksAnswered"] == {
            "granted": 1, "declined": 0, "days": quota.ASK_KEPT.days
        }
        assert _rows(site) == [("eve@example.com", "Saw it on a forum.", True, True)]
        # And it makes an account like any other.
        joined = site.browser().post(
            "/api/join", {"code": code, "handle": "eve", "password": PASSWORD}
        )
        assert joined.status == 200, joined.text

    def test_declining_sends_nothing_and_leaves_no_invitation(self, site: Site) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        _confirmed(_ask(site, "eve@example.com"))
        [listed] = _pending(boss)
        answer = boss.post("/api/admin", {"act": "decline-ask", "ask": listed["id"]})
        assert answer.status == 200, answer.text
        assert "nothing was sent" in answer.payload["said"]
        assert site.sent.messages == []
        assert site.count("invites") == 0
        assert _pending(boss) == []
        assert boss.get("/api/admin").payload["asksAnswered"]["declined"] == 1

    def test_a_request_is_answered_once(self, site: Site) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        _confirmed(_ask(site, "eve@example.com"))
        [listed] = _pending(boss)
        assert boss.post("/api/admin", {"act": "grant-ask", "ask": listed["id"]}).status == 200
        for act in ("grant-ask", "decline-ask"):
            again = boss.post("/api/admin", {"act": act, "ask": listed["id"]})
            assert again.status == 400
            assert again.payload["error"] == "that request has already been answered"
        assert len(site.sent.messages) == 1
        assert site.count("invites") == 1

    def test_an_address_that_joined_since_it_asked_is_refused_and_stays_pending(
        self, site: Site
    ) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        _confirmed(_ask(site, "cass@example.invalid"))
        site.person("cass")  # `Site.person` gives cass this address
        [listed] = _pending(boss)
        answer = boss.post("/api/admin", {"act": "grant-ask", "ask": listed["id"]})
        assert answer.status == 400
        assert "already has an account" in answer.payload["error"]
        assert [row["id"] for row in _pending(boss)] == [listed["id"]]
        assert site.sent.messages == []

    def test_a_failed_send_leaves_the_request_pending_and_no_invitation(
        self, site: Site
    ) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        _confirmed(_ask(site, "eve@example.com"))
        [listed] = _pending(boss)

        def refuse(_message: mail.Message) -> None:
            raise mail.MailError("the sender is down")

        site.app.send = refuse
        answer = boss.post("/api/admin", {"act": "grant-ask", "ask": listed["id"]})
        assert answer.status >= 400
        assert site.count("invites") == 0
        assert [row["id"] for row in _pending(boss)] == [listed["id"]]

    def test_an_ask_named_wrongly_is_refused(self, site: Site) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        for wrong in (None, "1", 1.5, True):
            answer = boss.post("/api/admin", {"act": "grant-ask", "ask": wrong})
            assert answer.status == 400, wrong
        assert boss.post("/api/admin", {"act": "decline-ask", "ask": 999}).status == 400


# --------------------------------------------------------------- one answer for all


class TestOneAnswerWhateverHappened:
    """No account's or request's existence is revealed: the same status and the same
    sentence for a new address, a member's, a shut-out member's and a repeat."""

    def test_a_members_address_gets_the_same_answer_and_writes_nothing(
        self, site: Site
    ) -> None:
        site.person("cass")
        fresh = _ask(site, "new@example.com")
        member = _ask(site, "CASS@example.invalid", "let me in")
        _confirmed(fresh)
        _confirmed(member)
        assert (member.status, member.body) == (fresh.status, fresh.body)
        assert [row[0] for row in _rows(site)] == ["new@example.com"]

    def test_a_shut_out_members_address_gets_it_too(self, site: Site) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        site.person("cass")
        boss.post("/api/admin", {"act": "kill-user", "user": _id(site, "cass")})
        _confirmed(_ask(site, "cass@example.invalid"))
        assert _rows(site) == []

    def test_a_repeat_while_one_is_pending_writes_nothing_and_keeps_the_first_note(
        self, site: Site
    ) -> None:
        _confirmed(_ask(site, "eve@example.com", "first"))
        _confirmed(_ask(site, "EVE@example.com", "second"))
        assert _rows(site) == [("eve@example.com", "first", False, None)]

    def test_an_answered_request_holds_the_address_for_the_kept_window_then_goes(
        self, site: Site
    ) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        _confirmed(_ask(site, "eve@example.com", "first"))
        [listed] = _pending(boss)
        boss.post("/api/admin", {"act": "decline-ask", "ask": listed["id"]})
        _confirmed(_ask(site, "eve@example.com", "again"))
        assert _rows(site) == [("eve@example.com", "first", True, False)]
        assert _pending(boss) == []

        with site.store.cursor() as cursor:
            cursor.execute(
                "update invite_asks set answered = now() - %s - interval '1 minute'",
                (quota.ASK_KEPT,),
            )
        site.store.commit()
        _confirmed(_ask(site, "eve@example.com", "a month on"))
        assert _rows(site) == [("eve@example.com", "a month on", False, None)]

    def test_the_admin_page_sweeps_the_old_answers_too(self, site: Site) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        _confirmed(_ask(site, "eve@example.com"))
        [listed] = _pending(boss)
        boss.post("/api/admin", {"act": "grant-ask", "ask": listed["id"]})
        with site.store.cursor() as cursor:
            cursor.execute(
                "update invite_asks set answered = now() - %s - interval '1 minute'",
                (quota.ASK_KEPT,),
            )
        site.store.commit()
        assert boss.get("/api/admin").payload["asksAnswered"]["granted"] == 0
        assert _rows(site) == []

    def test_an_address_with_an_earlier_invitation_may_ask_and_the_panel_says_so(
        self, site: Site
    ) -> None:
        """The earlier link may be in a junk folder; granting sends a fresh one."""
        boss = site.signed_in("boss", admin.ADMIN)
        boss.post("/api/admin", {"act": "invite", "email": "eve@example.com"})
        _confirmed(_ask(site, "eve@example.com", "I never got it"))
        [listed] = _pending(boss)
        assert listed["invited"] is not None
        assert boss.post("/api/admin", {"act": "grant-ask", "ask": listed["id"]}).status == 200
        assert len(_invitations(site, "eve@example.com")) == 2


# ---------------------------------------------------------------------- the text


class TestTheText:
    def test_a_missing_or_malformed_address_is_refused_in_one_sentence(
        self, site: Site
    ) -> None:
        for wrong in ("", "   ", "nobody", "a@b", "a b@example.com", "@example.com",
                      "eve@", None, 7, ["eve@example.com"], "eve\x00@example.com",
                      "eve@exa\u0007mple.com"):
            answer = _ask(site, wrong)
            assert answer.status == 400, wrong
            assert answer.payload["error"] == asking.NOT_AN_ADDRESS, wrong
        assert _rows(site) == []

    def test_an_address_over_the_cap_is_refused(self, site: Site) -> None:
        local = "e" * (quota.ASK_EMAIL_LENGTH - len("@example.com"))
        _confirmed(_ask(site, f"{local}@example.com"))
        answer = _ask(site, f"{local}x@example.com")
        assert answer.status == 400
        assert answer.payload["error"] == asking.NOT_AN_ADDRESS
        assert len(_rows(site)) == 1

    def test_a_note_is_capped_and_says_by_how_much(self, site: Site) -> None:
        _confirmed(_ask(site, "eve@example.com", "n" * quota.ASK_NOTE_LENGTH))
        answer = _ask(site, "fay@example.com", "n" * (quota.ASK_NOTE_LENGTH + 1))
        assert answer.status == 400
        assert answer.payload["error"] == (
            f"A note is up to {quota.ASK_NOTE_LENGTH} characters and this one is "
            f"{quota.ASK_NOTE_LENGTH + 1}. A line or two is plenty."
        )
        assert [row[0] for row in _rows(site)] == ["eve@example.com"]

    def test_a_note_that_is_not_text_is_refused(self, site: Site) -> None:
        for wrong in (7, ["a"], {"a": 1}, True):
            assert _ask(site, "eve@example.com", wrong).status == 400, wrong
        assert _rows(site) == []

    def test_a_note_keeps_its_lines_and_loses_control_characters(self, site: Site) -> None:
        _confirmed(_ask(site, "eve@example.com", "  a forum\r\nthen\ta search\x00\x1b  "))
        assert _rows(site)[0][1] == "a forum\nthen\ta search"

    def test_tags_are_kept_as_typed_and_never_marked_up(self, site: Site) -> None:
        boss = site.signed_in("boss", admin.ADMIN)
        note = '<script>alert("hi")</script> <b>bold</b>'
        _confirmed(_ask(site, "eve@example.com", note))
        assert _pending(boss)[0]["note"] == note
        script = (STATIC / "admin.js").read_text(encoding="utf-8")
        drawn = script[script.index("function asks("):]
        drawn = drawn[: drawn.index("\n}\n")]
        assert "cell(row, one.note || '\\u2014', 'ask-note');" in drawn
        assert "innerHTML" not in drawn

    def test_the_table_holds_its_own_caps(self, site: Site) -> None:
        """The check constraints and the one row an address, true of the table itself."""
        import psycopg

        for sql, params in (
            ("insert into invite_asks (email, note) values (%s, %s)",
             ("eve@example.com", "n" * (quota.ASK_NOTE_LENGTH + 1))),
            ("insert into invite_asks (email, note) values (%s, %s)", ("eve@example.com", "")),
            ("insert into invite_asks (email) values (%s)",
             ("e" * (quota.ASK_EMAIL_LENGTH + 1),)),
            ("insert into invite_asks (email, answered) values (%s, now())",
             ("eve@example.com",)),
        ):
            with pytest.raises(psycopg.errors.CheckViolation), site.store.cursor() as cursor:
                cursor.execute(sql, params)
            site.store.rollback()
        with site.store.cursor() as cursor:
            cursor.execute("insert into invite_asks (email) values ('eve@example.com')")
        site.store.commit()
        with pytest.raises(psycopg.errors.UniqueViolation), site.store.cursor() as cursor:
            cursor.execute("insert into invite_asks (email) values ('eve@example.com')")
        site.store.rollback()


# ---------------------------------------------------------------------- the caps


class TestTheCaps:
    def test_the_site_keeps_so_many_a_day_and_then_refuses_every_address_alike(
        self, site: Site
    ) -> None:
        site.person("cass")
        with site.store.cursor() as cursor:
            # Yesterday's do not count; today's do.
            cursor.execute(
                "insert into invite_asks (email, created)"
                " select 'old' || n || '@example.com', now() - interval '25 hours'"
                " from generate_series(1, %s) n",
                (quota.ASKS_A_DAY,),
            )
            cursor.execute(
                "insert into invite_asks (email)"
                " select 'today' || n || '@example.com' from generate_series(1, %s) n",
                (quota.ASKS_A_DAY - 1,),
            )
        site.store.commit()
        _confirmed(_ask(site, "last@example.com"))
        refusals = [_ask(site, email) for email in
                    ("next@example.com", "cass@example.invalid", "last@example.com")]
        for refused in refusals:
            assert refused.status == 429
            assert refused.payload == {"error": quota.ASKS_FULL}
        assert len({refused.body for refused in refusals}) == 1
        assert site.count("invite_asks") == 2 * quota.ASKS_A_DAY
        assert site.sent.messages == []

    def test_one_connection_may_send_so_many_a_day_written_or_not(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(quota, "ASKS_A_SOURCE", 3)
        site.person("cass")
        here, there = "203.0.113.7", "198.51.100.9"
        for email in ("a@example.com", "cass@example.invalid", "a@example.com"):
            _confirmed(_ask(site, email, source=here))
        refused = _ask(site, "b@example.com", source=here)
        assert refused.status == 429
        assert refused.payload == {"error": quota.ASKED_ENOUGH}
        # A malformed form is refused before it is counted, so a typo costs no try ...
        assert _ask(site, "nobody", source=there).status == 400
        # ... and another connection has its own day.
        _confirmed(_ask(site, "b@example.com", source=there))
        assert [row[0] for row in _rows(site)] == ["a@example.com", "b@example.com"]

    def test_the_connection_count_rolls_over_a_day(self) -> None:
        now = [0.0]
        asks = quota.Asks(clock=lambda: now[0])
        for _ in range(quota.ASKS_A_SOURCE):
            asks.take("203.0.113.7")
        with pytest.raises(quota.Spent):
            asks.take("203.0.113.7")
        asks.take("198.51.100.9")
        now[0] += 24 * 3600 + 1
        asks.take("203.0.113.7")

    def test_the_numbers_are_in_the_quota_module_and_on_the_admin_page(self) -> None:
        values = dict(quota.summary())
        assert values["Requests for an invitation the site keeps a day"] == str(quota.ASKS_A_DAY)
        assert values["Requests for an invitation one connection may send a day"] == str(
            quota.ASKS_A_SOURCE
        )
        assert values["Characters one request's note may be"] == str(quota.ASK_NOTE_LENGTH)
        assert (quota.ASKS_A_DAY, quota.ASK_NOTE_LENGTH, quota.ASK_EMAIL_LENGTH) == (20, 500, 254)


# ---------------------------------------------------------------------- the words


def test_every_sentence_a_visitor_reads_here_is_in_a_visitors_words() -> None:
    said = {
        "the answer": asking.CONFIRMED,
        "a bad address": asking.NOT_AN_ADDRESS,
        "the hint": asking.NOTE_HINT,
        "a full day": quota.ASKS_FULL,
        "a busy connection": quota.ASKED_ENOUGH,
        "the page": site_words.visible(site_pages.request_invitation()),
    }
    trouble = [f"{where}: {phrase!r}" for where, text in said.items()
               for phrase in site_words.hits(text)]
    assert not trouble, trouble
    # The one answer says nothing about whether the address is known.
    for word in ("account", "already", "member"):
        assert word not in asking.CONFIRMED
