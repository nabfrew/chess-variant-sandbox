"""*Your move*: the one opt-in email and *Games waiting on your move* (Q572).

`docs/archive/plan-return.md` §1.5 and the dated exception to test 3 of `docs/reference/no-dark-patterns.md`
(the user, 2026-10-01): a member may switch on, off by default, one email when it becomes
their move in a game between two people, and the site sends nothing else. What is proved,
on specs rather than chess:

- **test 3, in code**: every message the package builds is an invitation or a move email,
  and a move email is built only by `move_email.due`, only for a person whose box is
  ticked;
- the box is off by default, and each condition of `site/move_email.py` holds: one email a
  turn, none to the mover, none when the match is open, none past the daily cap, none for
  a draw offer, a resignation or a cancellation;
- a failed send never blocks a move, and no real mail leaves (a recording sender);
- one click off, by GET and by the mail program's POST, with no sign-in;
- the landing page's *Games waiting on your move*: the person's own move only, in the
  server's order, absent when empty; and the page code has no count, dot or title number.

The store tests skip without `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL`, as every site test
does; the words, the source scans and the page checks need nothing.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from site_words import hits
from test_site_matches import LADDER, SHORT, SHORT_HASH, Scratch, _dull, _scratch

from chess_variant_sandbox.site import admin, mail, move_email, pages, quota, routes
from chess_variant_sandbox.site import correspondence as c
from chess_variant_sandbox.site import matches as m
from chess_variant_sandbox.site.server import SignInFirst, SiteApp
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.web import library

PACKAGE = Path(__file__).resolve().parents[1] / "chess_variant_sandbox"
SITE = "https://site.example"
TOKEN = "Tok_en-made-for-the-test"

# Twelve plies, unrated (so practice): long enough that a turn passes several times.
LONG = _dull(12)
LONG_HASH = spec_hash(LONG)


class Sent:
    """A sender that keeps the message instead of posting it."""

    def __init__(self) -> None:
        self.messages: list[mail.Message] = []

    def __call__(self, message: mail.Message) -> None:
        self.messages.append(message)


class Refusing:
    """A sender that always fails, the way the service or the network can."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.tried = 0

    def __call__(self, message: mail.Message) -> None:
        self.tried += 1
        raise self.error


# --------------------------------------------------------------------------- the message


class TestTheMessage:
    def _built(self) -> mail.Message:
        return mail.your_move(
            to="ada@example.invalid", game="Short game", opponent="bo", played="a1-a2",
            link=f"{SITE}/#/play/Xc81kq2", off=f"{SITE}/mail/off/{TOKEN}",
        )

    def test_it_says_whose_move_in_which_game_and_links_to_it(self) -> None:
        said = self._built()
        assert said.to == "ada@example.invalid"
        assert said.subject == "Your move: Short game against bo"
        assert said.body.startswith("bo played a1-a2 in your game of Short game. It's your move:")
        assert f"{SITE}/#/play/Xc81kq2" in said.body

    def test_it_says_how_to_switch_it_off_in_one_click(self) -> None:
        said = self._built()
        off = f"{SITE}/mail/off/{TOKEN}"
        assert "Switch them off (one click, no sign-in)" in said.body
        assert said.body.rstrip().endswith(off)
        assert dict(said.headers) == {
            "List-Unsubscribe": f"<{off}>",
            "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
        }

    def test_it_promises_nothing_else_and_pushes_nothing(self) -> None:
        said = self._built()
        assert "and nothing else" in said.body
        for pushing in ("remind", "hurry", "days left", "streak", "don't miss", "!"):
            assert pushing not in said.body.lower() + said.subject.lower(), pushing

    def test_the_words_pass_the_plain_words_guard(self) -> None:
        said = self._built()
        for text in (said.subject, said.body, move_email.BOX_SAID, move_email.BOX_UNDER,
                     move_email.OFF_SAID, move_email.OFF_UNKNOWN):
            assert hits(text) == [], (text, hits(text))

    def test_the_box_says_exactly_what_it_does(self) -> None:
        assert move_email.BOX_UNDER.startswith("Off unless you tick it.")
        assert "One email each time it becomes your move" in move_email.BOX_UNDER
        assert "nothing else, ever" in move_email.BOX_UNDER

    def test_the_headers_reach_the_service_and_an_invitation_has_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The live sender, with its POST caught before it leaves: no real mail."""
        posted: list[dict[str, Any]] = []

        class Answer:
            status = 200

            def read(self) -> bytes:
                return b"{}"

            def __enter__(self) -> Answer:
                return self

            def __exit__(self, *_: object) -> None:
                return None

        def caught(request: Any, timeout: float) -> Answer:
            posted.append(json.loads(request.data))
            return Answer()

        monkeypatch.setenv(mail.KEY_VAR, "a-key-that-goes-nowhere")
        monkeypatch.setenv(mail.FROM_VAR, "site@example.invalid")
        monkeypatch.setattr(mail.urllib.request, "urlopen", caught)
        mail.send(self._built())
        mail.send(mail.invitation("lee@example.invalid", f"{SITE}/join?code=x"))
        assert posted[0]["headers"]["List-Unsubscribe"] == f"<{SITE}/mail/off/{TOKEN}>"
        assert "headers" not in posted[1]


# ------------------------------------------------------------------- test 3, in the code


def _package() -> Iterator[tuple[Path, ast.Module]]:
    for path in sorted(PACKAGE.rglob("*.py")):
        yield path, ast.parse(path.read_text(encoding="utf-8"))


def _calls(tree: ast.Module) -> Iterator[tuple[str, str]]:
    """Every call in a module, as (the enclosing function, the name called)."""
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                called = node.func
                name = (
                    called.attr if isinstance(called, ast.Attribute)
                    else called.id if isinstance(called, ast.Name) else ""
                )
                yield function.name, name


@pytest.fixture(scope="module")
def package_calls() -> list[tuple[Path, list[tuple[str, str]]]]:
    """Every call in the package, by module: parsed and walked once for the two tests that
    read it, where each walked the whole package itself, 2.1-2.4 s apiece (Q749)."""
    return [(path, list(_calls(tree))) for path, tree in _package()]


class TestTestThree:
    """Every message the package builds for a person is an invitation or a move email, and
    a move email is built in one place, which reads the box first. The one other message
    is the owner's word of a request for an invitation (Q979), which goes to the address
    the box's settings name and to no visitor or member."""

    def test_messages_are_built_only_by_the_two_builders(self, package_calls) -> None:
        built = {
            (path.name, function)
            for path, calls in package_calls
            for function, name in calls
            if name == "Message"
        }
        assert built == {
            ("mail.py", "invitation"),
            ("mail.py", "your_move"),
            ("mail.py", "asked_for_invitation"),
        }

    def test_each_builder_has_one_caller(self, package_calls) -> None:
        callers: dict[str, set[tuple[str, str]]] = {
            "invitation": set(), "your_move": set(), "asked_for_invitation": set()
        }
        for path, calls in package_calls:
            if path.name == "mail.py":
                continue
            for function, name in calls:
                if name in callers:
                    callers[name].add((path.name, function))
        assert callers == {
            "invitation": {("identity.py", "send_invitation")},
            "your_move": {("move_email.py", "due")},
            # The nested `tell` and the method around it, which the walk counts both.
            "asked_for_invitation": {("server.py", "_tell_owner"), ("server.py", "tell")},
        }

    def test_the_owners_message_goes_to_the_owner_alone(self) -> None:
        """Its recipient is `mail.owner_address()`, read once, and nothing else."""
        source = (PACKAGE / "site" / "server.py").read_text(encoding="utf-8")
        body = source[source.index("def _tell_owner(") :]
        body = body[: body.index("\n    def ", 1)]
        assert "owner = mail.owner_address()" in body
        assert "mail.asked_for_invitation(owner, " in body

    def test_nothing_else_reaches_the_sending_service(self) -> None:
        for path in sorted(PACKAGE.rglob("*.py")):  # the text, so nothing is parsed
            text = path.read_text(encoding="utf-8")
            if path.name == "mail.py":
                continue
            assert "api.resend.com" not in text, path
            assert "smtplib" not in text, path

    def test_a_move_email_is_built_only_past_the_box(self) -> None:
        """In `due`, the box is read and refused before `mail.your_move` is reached."""
        source = (PACKAGE / "site" / "move_email.py").read_text(encoding="utf-8")
        body = source[source.index("def due(") :]
        assert body.index("if not on") < body.index("mail.your_move(")


# ---------------------------------------------------------------------------- the store


@pytest.fixture
def store() -> Iterator[Scratch]:
    with _scratch() as scratch:
        scratch.game(SHORT_HASH, SHORT, LADDER)
        scratch.game(LONG_HASH, LONG)
        yield scratch


@pytest.fixture
def sent() -> Sent:
    return Sent()


@pytest.fixture
def app(store: Scratch, sent: Sent, monkeypatch: pytest.MonkeyPatch) -> Iterator[SiteApp]:
    monkeypatch.setenv(mail.URL_VAR, SITE)
    made = SiteApp(library.RULESET_DIR, library.RUNS_DIR, connect=store.open, send=sent)
    yield made
    made.viewer = None


def _as(app: SiteApp, viewer: int | None) -> SiteApp:
    app.viewer = viewer
    return app


def _mover(store: Scratch, match_id: str) -> int:
    return int(m.session(store.connection, match_id).snapshot()["sideToMove"])


def _send_move(app: SiteApp, store: Scratch, match_id: str, people: tuple[int, int]) -> Any:
    """The player to move sends the engine's first offered move, through the site."""
    snapshot = m.session(store.connection, match_id).snapshot()
    store.connection.commit()
    who = people[int(snapshot["sideToMove"])]
    return _as(app, who).post(
        f"/api/session/{match_id}/move", {"move": snapshot["moves"][0]["move"]}
    )


def _rows(store: Scratch) -> list[tuple[Any, ...]]:
    with store.connection.cursor() as cursor:
        cursor.execute("select match, recipient, ply, failed from move_emails order by id")
        found = cursor.fetchall()
    store.connection.commit()
    return found


def _game(store: Scratch, *ticked: str) -> tuple[m.Row, tuple[int, int]]:
    ada, bo = store.user("ada"), store.user("bo")
    for user, handle in ((ada, "ada"), (bo, "bo")):
        if handle in ticked:
            move_email.set_on(store.connection, user, True)
    found = c.start(store.connection, LONG_HASH, [ada, bo])
    store.connection.commit()
    return found, (ada, bo)


class TestTheBox:
    def test_it_is_off_by_default(self, store: Scratch, app: SiteApp) -> None:
        ada = store.user("ada")
        assert move_email.is_on(store.connection, ada) is False
        box = _as(app, ada).get("/api/profile")["moveEmail"]
        assert box == {"on": False, "said": move_email.BOX_SAID, "under": move_email.BOX_UNDER}

    def test_a_member_ticks_and_unticks_it_from_their_page(
        self, store: Scratch, app: SiteApp
    ) -> None:
        ada = store.user("ada")
        assert _as(app, ada).post("/api/move-email", {"on": True})["on"] is True
        assert app.get("/api/profile")["moveEmail"]["on"] is True
        assert app.post("/api/move-email", {"on": False})["on"] is False
        assert move_email.is_on(store.connection, ada) is False

    def test_a_visitor_cannot_reach_it(self, app: SiteApp) -> None:
        with pytest.raises(SignInFirst):
            _as(app, None).post("/api/move-email", {"on": True})
        assert routes.find("POST", "/api/move-email") is not None

    def test_the_off_token_is_made_once_and_kept(self, store: Scratch) -> None:
        ada = store.user("ada")
        move_email.set_on(store.connection, ada, True)
        first = _token(store, ada)
        move_email.set_on(store.connection, ada, False)
        move_email.set_on(store.connection, ada, True)
        assert first and _token(store, ada) == first

    def test_leaving_clears_the_box_and_its_token(self, store: Scratch) -> None:
        ada = store.user("ada")
        move_email.set_on(store.connection, ada, True)
        admin.forget(store.connection, "ada")
        store.connection.commit()
        assert move_email.is_on(store.connection, ada) is False
        assert _token(store, ada) is None


def _token(store: Scratch, user: int) -> str | None:
    with store.connection.cursor() as cursor:
        cursor.execute("select mail_off from users where id = %s", (user,))
        found = cursor.fetchone()[0]
    store.connection.commit()
    return found


class TestWhenOneIsSent:
    def test_nothing_is_sent_to_somebody_who_did_not_tick(
        self, store: Scratch, app: SiteApp, sent: Sent
    ) -> None:
        found, people = _game(store)
        for _ in range(3):
            _send_move(app, store, found.id, people)
        assert sent.messages == [] and _rows(store) == []

    def test_a_move_sends_one_to_the_person_whose_move_it_now_is(
        self, store: Scratch, app: SiteApp, sent: Sent
    ) -> None:
        found, people = _game(store, "ada", "bo")
        mover = _mover(store, found.id)
        answer = _send_move(app, store, found.id, people)
        assert answer["sideToMove"] == 1 - mover, "the engine passed the turn"
        assert len(sent.messages) == 1
        message = sent.messages[0]
        waiting = ("ada", "bo")[1 - mover]
        assert message.to == f"{waiting}@example.invalid"
        assert message.subject == f"Your move: {LONG['name']} against {('ada', 'bo')[mover]}"
        played = answer["history"][-1]["text"]
        assert f"{('ada', 'bo')[mover]} played {played} in your game of" in message.body
        assert f"{SITE}/#/play/{found.id}" in message.body
        token = _token(store, people[1 - mover])
        assert f"{SITE}/mail/off/{token}" in message.body
        assert _rows(store) == [(found.id, people[1 - mover], 1, None)]

    def test_one_a_turn_and_never_to_the_mover(
        self, store: Scratch, app: SiteApp, sent: Sent
    ) -> None:
        found, people = _game(store, "ada", "bo")
        expected = []
        for _ in range(5):
            row = m.row(store.connection, found.id)
            store.connection.commit()
            if not row.running:
                break
            mover = _mover(store, found.id)
            _send_move(app, store, found.id, people)
            if m.row(store.connection, found.id).running:
                expected.append(f"{('ada', 'bo')[1 - mover]}@example.invalid")
            store.connection.commit()
        assert [message.to for message in sent.messages] == expected
        assert len({row[2] for row in _rows(store)}) == len(expected)

    def test_a_turn_that_does_not_pass_sends_nothing(self, store: Scratch) -> None:
        """The engine's side to move after the move decides: still the mover, no email."""
        found, people = _game(store, "ada", "bo")
        mover = _mover(store, found.id)
        snapshot = {"sideToMove": mover, "history": [{"text": "a1-a2"}]}
        assert move_email.due(store.connection, found.id, mover, snapshot) is None

    def test_the_same_move_never_sends_two(
        self, store: Scratch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(mail.URL_VAR, SITE)
        found, people = _game(store, "ada", "bo")
        mover = _mover(store, found.id)
        snapshot = {"sideToMove": 1 - mover, "history": [{"text": "a1-a2"}]}
        assert move_email.due(store.connection, found.id, mover, snapshot) is not None
        assert move_email.due(store.connection, found.id, mover, snapshot) is None

    def test_none_when_the_match_is_open(
        self, store: Scratch, app: SiteApp, sent: Sent
    ) -> None:
        found, people = _game(store, "ada", "bo")
        waiting = people[1 - _mover(store, found.id)]
        _as(app, waiting).get(f"/api/session/{found.id}")  # the page's poll
        _send_move(app, store, found.id, people)
        assert sent.messages == []
        # Closed for more than a minute: the next turn's email goes.
        with store.connection.cursor() as cursor:
            cursor.execute(
                "update matches set seen0 = seen0 - interval '2 minutes',"
                " seen1 = seen1 - interval '2 minutes' where id = %s", (found.id,)
            )
        store.connection.commit()
        _send_move(app, store, found.id, people)
        _send_move(app, store, found.id, people)
        assert len(sent.messages) == 2

    def test_none_past_the_daily_cap_and_admin_shows_the_count(
        self, store: Scratch, app: SiteApp, sent: Sent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(quota, "MOVE_EMAILS_A_DAY", 1)
        found, people = _game(store, "ada", "bo")
        _send_move(app, store, found.id, people)
        _send_move(app, store, found.id, people)
        store.connection.commit()
        assert len(sent.messages) == 1
        assert move_email.sent_today(store.connection) == 1
        boss = store.user("boss")
        _as(app, boss)._request.role = admin.ADMIN
        view = app._admin_view()["correspondence"]
        assert (view["mailed"], view["mailCap"]) == (1, 1)

    def test_a_draw_offer_a_resignation_or_a_cancel_sends_nothing(
        self, store: Scratch, app: SiteApp, sent: Sent
    ) -> None:
        for action in ("draw", "cancel", "resign"):
            store.connection.rollback()
            with store.connection.cursor() as cursor:
                cursor.execute("delete from matches; delete from users")
            store.connection.commit()
            found, people = _game(store, "ada", "bo")
            _as(app, people[0]).post(f"/api/session/{found.id}/{action}", {})
        assert sent.messages == [] and _rows(store) == []

    def test_none_for_somebody_who_can_no_longer_sign_in(
        self, store: Scratch, app: SiteApp, sent: Sent
    ) -> None:
        found, people = _game(store, "ada", "bo")
        waiting = people[1 - _mover(store, found.id)]
        with store.connection.cursor() as cursor:
            cursor.execute("update users set revoked = now() where id = %s", (waiting,))
        store.connection.commit()
        _send_move(app, store, found.id, people)
        assert sent.messages == []

    def test_none_where_the_site_has_no_address_to_link_to(
        self, store: Scratch, app: SiteApp, sent: Sent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(mail.URL_VAR)
        found, people = _game(store, "ada", "bo")
        answer = _send_move(app, store, found.id, people)
        assert len(answer["history"]) == 1
        assert sent.messages == [] and _rows(store) == []


class TestAFailedSendNeverBlocksAMove:
    @pytest.mark.parametrize(
        "error", [mail.MailError("the service said no"), RuntimeError("the network fell over")]
    )
    def test_the_move_stands_and_the_failure_is_written(
        self, store: Scratch, app: SiteApp, error: Exception
    ) -> None:
        refusing = Refusing(error)
        app.send = refusing
        found, people = _game(store, "ada", "bo")
        answer = _send_move(app, store, found.id, people)
        assert refusing.tried == 1
        assert len(answer["history"]) == 1
        assert m.row(store.connection, found.id).plies == 1
        [(match, _recipient, ply, failed)] = _rows(store)
        assert (match, ply) == (found.id, 1) and str(error) in failed


# ----------------------------------------------------------------------- one click off


class TestOneClickOff:
    def test_get_and_the_mail_programs_post_both_switch_it_off(self) -> None:
        from test_site import _running_site

        with _running_site() as site:
            with site.store.cursor() as cursor:
                cursor.execute(
                    "insert into users (handle, email) values ('ana', 'ana@example.invalid')"
                    " returning id"
                )
                ana = int(cursor.fetchone()[0])
            move_email.set_on(site.store, ana, True)
            site.store.commit()
            token = _token_on(site.store, ana)
            visitor = site.browser()  # no cookie: no sign-in
            page = visitor.get(f"/mail/off/{token}")
            assert page.status == 200 and move_email.OFF_SAID in page.text
            assert move_email.is_on(site.store, ana) is False
            site.store.commit()
            move_email.set_on(site.store, ana, True)
            site.store.commit()
            # RFC 8058: a POST with no page header and a form body, as a mail program sends.
            from http.client import HTTPConnection

            connection = HTTPConnection("127.0.0.1", site.port, timeout=30)
            connection.request(
                "POST", f"/mail/off/{token}", body=b"List-Unsubscribe=One-Click",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            answer = connection.getresponse()
            answer.read()
            connection.close()
            assert answer.status == 200
            assert move_email.is_on(site.store, ana) is False
            site.store.commit()
            unknown = visitor.get("/mail/off/" + "x" * 32)
            assert unknown.status == 200 and move_email.OFF_UNKNOWN in unknown.text

    def test_the_address_takes_only_a_tokens_shape(self) -> None:
        assert pages.MAIL_OFF.match("/mail/off/abcDEF_-1234567890")
        for wrong in ("/mail/off/", "/mail/off/short", "/mail/off/../admin",
                      "/mail/off/a b c d e f g h i j", "/mail/off/" + "a" * 65):
            assert not pages.MAIL_OFF.match(wrong), wrong
        assert routes.MAIL_OFF == "/mail/off/"

    def test_the_page_passes_the_plain_words_guard(self) -> None:
        from site_words import visible

        for said in (move_email.OFF_SAID, move_email.OFF_UNKNOWN):
            assert hits(visible(pages.mail_off(said))) == []


def _token_on(connection: Any, user: int) -> str:
    with connection.cursor() as cursor:
        cursor.execute("select mail_off from users where id = %s", (user,))
        found = cursor.fetchone()[0]
    connection.commit()
    return str(found)


# ------------------------------------------------------------ games waiting on your move


class TestGamesWaitingOnYourMove:
    def test_the_landing_page_reads_only_your_own_move_in_order(
        self, store: Scratch, app: SiteApp
    ) -> None:
        ada, bo = store.user("ada"), store.user("bo")
        people = (ada, bo)
        made = []
        for age in (timedelta(days=1), timedelta(hours=3), timedelta(days=2)):
            found = c.start(store.connection, LONG_HASH, [ada, bo])
            store.connection.commit()
            store.backdate(found.id, age)
            made.append(found.id)
        _send_move(app, store, made[0], people)  # the turn passes in the first one
        store.connection.commit()
        waiting = _as(app, ada).get("/api/challenges")["waiting"]
        mine = [i for i in made if people[_mover(store, i)] == ada]
        assert [entry["id"] for entry in waiting] == [
            e["id"] for e in c.started_order([
                {"id": i, "yourMove": True, "left": next(
                    w["left"] for w in waiting if w["id"] == i)} for i in mine
            ])
        ]
        assert {entry["id"] for entry in waiting} == set(mine)
        for entry in waiting:
            assert entry["yourMove"] is True
            assert entry["name"] == LONG["name"]
            assert entry["opponent"] == "bo"
            assert entry["toMove"].startswith("You have ")
        theirs = _as(app, bo).get("/api/challenges")["waiting"]
        assert {entry["id"] for entry in theirs} == set(made) - set(mine)

    def test_nothing_waits_and_the_list_is_empty(self, store: Scratch, app: SiteApp) -> None:
        ada = store.user("ada")
        assert _as(app, ada).get("/api/challenges")["waiting"] == []


# --------------------------------------------------------------------------- the page


WAITING = {
    "waiting": [{
        "id": "g1", "game": "abc", "name": "Short game", "opponent": "bo", "rated": True,
        "plies": 3, "moved": "2026-10-01T10:00:00+00:00",
        "toMove": "You have 4 days 6 hours to move.", "yourMove": True,
    }],
    "forYou": [],
}


class TestThePage:
    def test_the_landing_page_draws_the_games_waiting_on_you(self) -> None:
        from test_web import _in_page

        drawn = _in_page("", f"forYouHtml({json.dumps(WAITING)})")
        assert "Games waiting on your move" in drawn
        assert "Short game" in drawn and "<b>bo</b>" in drawn
        assert "You have 4 days 6 hours to move." in drawn
        assert 'data-play="g1"' in drawn and ">Your move</button>" in drawn
        assert "A challenge for you" not in drawn

    def test_absent_when_nothing_waits(self) -> None:
        from test_web import _in_page

        empty = json.dumps({"waiting": [], "forYou": []})
        assert _in_page("", f"forYouHtml({empty})") == ""
        assert _in_page("", "forYouHtml(null)") == ""

    def test_the_profile_has_correspondence_in_place_of_the_interim_list(self) -> None:
        profile = (PACKAGE / "site" / "static" / "profile.js").read_text(encoding="utf-8")
        assert "Games against people" not in profile
        for heading in ("'Your move'", "'Their move'", "'Email'"):
            assert heading in profile, heading
        markup = pages.profile("ada")
        assert "<h2>Correspondence</h2>" in markup and 'id="move-email"' in markup

    def test_no_count_dot_title_or_escalation_in_the_new_page_code(self) -> None:
        browse = (PACKAGE / "web" / "static" / "page" / "browse.js").read_text(encoding="utf-8")
        start = browse.index("// ------------------------------------------- what waits on you")
        landing = _code(browse[start : browse.index("function wireForYou", start)])
        profile = (PACKAGE / "site" / "static" / "profile.js").read_text(encoding="utf-8")
        box = _code(profile[profile.index("function drawMoveEmail") :
                            profile.index("\n}\n", profile.index("function drawMoveEmail"))])
        for code in (landing, box):
            assert "document.title" not in code
            assert ".length}" not in code and "length +" not in code, "a count"
            for mark in ("badge", "dot", "animation", "blink", "pulse", "urgent", "--bad",
                         "countdown", "setInterval"):
                assert not re.search(rf"(?<![\w-]){mark}(?![\w-])", code, re.I), mark

    def test_the_privacy_notice_says_it(self) -> None:
        notice = pages.privacy()
        assert "becomes your move" in notice
        assert "off unless you tick it" in notice
        assert "No other mail goes there, ever" in notice


def _code(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))
