"""Finding an opponent (Q571, `docs/archive/plan-return.md` §1.4): the store, the routes and the pages.

Each path is proved on specs rather than chess (rule 5 of CLAUDE.md): the two `dull`
variants of `test_site_matches.py`, one with a loaded ladder (rated) and one without
(practice). *Open to a game of this*, a challenge by handle, a challenge link opened by a
member, by a visitor who then joins or signs in, and a second time, a rematch, the pause,
the two abuse limits, and leaving. Accepting is proved to start the match through
`correspondence.start` and nothing else, so Q570's refusals hold.

Beside them, the seven tests of `docs/reference/no-dark-patterns.md` on the new page code: no expiry,
no countdown, no badge or count, and the box off by default.

The store tests need `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL` and skip without it, as
every site test does. The page checks need node, or nothing.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from test_site_matches import (
    LADDER,
    SHORT,
    SHORT_HASH,
    UNRATED,
    UNRATED_HASH,
    Scratch,
    _scratch,
)

from chess_variant_sandbox.site import admin, inviting, pages, quota, routes
from chess_variant_sandbox.site import challenges as ch
from chess_variant_sandbox.site import correspondence as c
from chess_variant_sandbox.site import matches as m
from chess_variant_sandbox.site.server import SignInFirst, SiteApp
from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.page import marks

REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE = REPOSITORY / "chess_variant_sandbox"
STATIC = PACKAGE / "site" / "static"


@pytest.fixture
def store() -> Iterator[Scratch]:
    """A store with one rated game (a loaded ladder), one without, and nobody yet."""
    with _scratch() as scratch:
        scratch.game(SHORT_HASH, SHORT, LADDER)
        scratch.game(UNRATED_HASH, UNRATED)
        yield scratch


def _three(store: Scratch) -> tuple[int, int, int]:
    return store.user("ada"), store.user("bo"), store.user("cy")


def _play_out(store: Scratch, match_id: str) -> dict[str, Any]:
    """Both players send the engine's first offered move; the short game then ends."""
    answer: dict[str, Any] = {}
    for _ in range(2):
        found = m.row(store.connection, match_id)
        snapshot = m.session(store.connection, match_id).snapshot()
        who = found.player(snapshot["sideToMove"]).user
        answer = c.move(store.connection, match_id, who, int(snapshot["moves"][0]["move"]))
    store.connection.commit()
    return answer


# ------------------------------------------------------------------ open to a game


class TestOpenToAGame:
    def test_off_by_default_and_listed_for_others_once_ticked(self, store: Scratch) -> None:
        ada, bo, _cy = _three(store)
        seen = ch.people(store.connection, SHORT_HASH, bo)
        assert seen["open"] == [] and seen["mine"] is False
        assert ch.set_open(store.connection, ada, SHORT_HASH, True) is True
        listed = ch.people(store.connection, SHORT_HASH, bo)["open"]
        assert listed == [
            {"handle": "ada", "said": "hasn't played this one yet", "mark": marks.mark(ada)}
        ]
        # Never the reader themself, and their own box reads ticked.
        mine = ch.people(store.connection, SHORT_HASH, ada)
        assert mine["open"] == [] and mine["mine"] is True
        # Per game: ticked on one game is not ticked on another.
        assert ch.people(store.connection, UNRATED_HASH, bo)["open"] == []
        assert ch.open_games(store.connection, ada) == [
            {"game": SHORT_HASH, "name": SHORT["name"]}
        ]

    def test_a_rating_shows_only_once_the_person_has_played(self, store: Scratch) -> None:
        ada, bo, cy = _three(store)
        ch.set_open(store.connection, ada, SHORT_HASH, True)
        _play_out(store, c.start(store.connection, SHORT_HASH, [ada, bo]).id)
        said = ch.people(store.connection, SHORT_HASH, cy)["open"][0]["said"]
        assert re.fullmatch(r"rated \d+ here", said), said

    def test_unticking_and_the_suggestions(self, store: Scratch) -> None:
        ada, bo, cy = _three(store)
        admin.kill_user(store.connection, cy)
        ch.set_open(store.connection, ada, SHORT_HASH, True)
        assert ch.set_open(store.connection, ada, SHORT_HASH, False) is False
        seen = ch.people(store.connection, SHORT_HASH, bo)
        assert seen["open"] == []
        # Every member who can sign in, but the reader and the revoked.
        assert seen["members"] == ["ada"]

    def test_rated_or_practice_is_the_games(self, store: Scratch) -> None:
        ada = store.user("ada")
        assert ch.people(store.connection, SHORT_HASH, ada)["rated"] is True
        assert ch.people(store.connection, UNRATED_HASH, ada)["rated"] is False

    def test_the_pause_hides_the_lists_and_refuses_a_tick(self, store: Scratch) -> None:
        ada, bo, _cy = _three(store)
        ch.set_open(store.connection, ada, SHORT_HASH, True)
        c.set_paused(store.connection, True)
        seen = ch.people(store.connection, SHORT_HASH, bo)
        assert seen["paused"] and seen["pausedSays"] == c.PAUSED_SAID
        assert seen["open"] == [] and seen["members"] == []
        with pytest.raises(m.MatchError, match="paused"):
            ch.set_open(store.connection, bo, SHORT_HASH, True)
        # Unticking always works.
        ch.set_open(store.connection, ada, SHORT_HASH, False)


# --------------------------------------------------------------- challenge by handle


def _seated(store: Scratch, match_id: str) -> tuple[int | None, int | None]:
    found = m.row(store.connection, match_id)
    return found.player(0).user, found.player(1).user


class TestByHandle:
    def test_accepting_starts_the_match_in_the_chosen_seats(self, store: Scratch) -> None:
        ada, bo, _cy = _three(store)
        first = ch.challenge(store.connection, ada, SHORT_HASH, 0, "bo")
        second = ch.challenge(store.connection, ada, UNRATED_HASH, 1, "BO ")
        assert [entry["id"] for entry in ch.for_you(store.connection, bo)] == [
            first.id, second.id
        ]
        one = ch.accept(store.connection, first.id, bo)
        two = ch.accept(store.connection, second.id, bo)
        assert _seated(store, one.id) == (ada, bo)
        assert _seated(store, two.id) == (bo, ada)
        # Rated where the game has a ladder, practice elsewhere: Q570's start decided.
        assert one.mode == m.LEADERBOARD and two.mode == c.PRACTICE
        assert c.between_people(one) and one.running
        accepted = ch.find(store.connection, first.id)
        assert accepted.answer == ch.ACCEPTED and accepted.match == one.id
        assert ch.for_you(store.connection, bo) == []

    def test_the_challenger_and_both_seats_carry_their_marks(self, store: Scratch) -> None:
        """R2: a person named in a challenge or a match's seats carries their own mark,
        made from their account; somebody who left carries none."""
        ada, bo, _cy = _three(store)
        sent = ch.challenge(store.connection, ada, SHORT_HASH, 0, "bo")
        waiting = ch.for_you(store.connection, bo)
        assert waiting[0]["fromMark"] == marks.mark(ada)
        made = ch.accept(store.connection, sent.id, bo)
        seen = c.view(store.connection, m.row(store.connection, made.id), bo)
        assert seen["marks"] == [marks.mark(ada), marks.mark(bo)]
        with store.connection.cursor() as cursor:
            cursor.execute("update users set handle = null where id = %s", (ada,))
        seen = c.view(store.connection, m.row(store.connection, made.id), bo)
        assert seen["marks"] == [None, marks.mark(bo)]

    def test_either_is_drawn_from_the_matchs_seed(self, store: Scratch) -> None:
        ada, bo, _cy = _three(store)
        for _ in range(6):
            sent = ch.challenge(store.connection, ada, UNRATED_HASH, None, "bo")
            made = ch.accept(store.connection, sent.id, bo)
            first = ada if made.seed % 2 == 0 else bo
            assert _seated(store, made.id)[0] == first
            c.cancel(store.connection, made.id, ada)

    def test_declining_says_declined_and_no_more(self, store: Scratch) -> None:
        ada, bo, _cy = _three(store)
        sent = ch.challenge(store.connection, ada, SHORT_HASH, 0, "bo")
        ch.decline(store.connection, sent.id, bo)
        listed = ch.sent_by(store.connection, ada)
        assert [(entry["to"], entry["answer"]) for entry in listed] == [("bo", ch.DECLINED)]
        with pytest.raises(m.MatchError, match="declined"):
            ch.accept(store.connection, sent.id, bo)
        assert store.rows() == 0

    def test_withdrawing_and_who_may_answer(self, store: Scratch) -> None:
        ada, bo, cy = _three(store)
        sent = ch.challenge(store.connection, ada, SHORT_HASH, 0, "bo")
        for outsider in (ada, cy):
            with pytest.raises(KeyError):
                ch.accept(store.connection, sent.id, outsider)
            with pytest.raises(KeyError):
                ch.decline(store.connection, sent.id, outsider)
        with pytest.raises(KeyError):
            ch.withdraw(store.connection, sent.id, bo)
        ch.withdraw(store.connection, sent.id, ada)
        assert ch.sent_by(store.connection, ada) == []
        with pytest.raises(m.MatchError, match="withdrawn"):
            ch.accept(store.connection, sent.id, bo)

    def test_a_challenge_never_expires(self, store: Scratch) -> None:
        ada, bo, _cy = _three(store)
        sent = ch.challenge(store.connection, ada, SHORT_HASH, 0, "bo")
        with store.connection.cursor() as cursor:
            cursor.execute(
                "update challenges set made = made - %s where id = %s",
                (timedelta(days=4000), sent.id),
            )
        # A sweep of everything that sweeps leaves it alone.
        m.sweep(store.connection)
        assert [entry["id"] for entry in ch.for_you(store.connection, bo)] == [sent.id]
        assert ch.accept(store.connection, sent.id, bo).running

    def test_the_refusals(self, store: Scratch) -> None:
        ada, bo, cy = _three(store)
        with pytest.raises(m.MatchError, match="cannot play yourself"):
            ch.challenge(store.connection, ada, SHORT_HASH, 0, "ada")
        with pytest.raises(m.MatchError, match="no member called"):
            ch.challenge(store.connection, ada, SHORT_HASH, 0, "nobody")
        with pytest.raises(m.MatchError, match="isn't on the site"):
            ch.challenge(store.connection, ada, "0" * 16, 0, "bo")
        with pytest.raises(m.MatchError, match="not both"):
            ch.challenge(store.connection, ada, SHORT_HASH, 0, "bo", link=True)
        admin.kill_user(store.connection, cy)
        with pytest.raises(m.MatchError, match="can no longer play"):
            ch.challenge(store.connection, ada, SHORT_HASH, 0, "cy")
        ch.challenge(store.connection, ada, SHORT_HASH, 0, "bo")
        with pytest.raises(m.MatchError, match="already challenged bo"):
            ch.challenge(store.connection, ada, SHORT_HASH, 1, "bo")

    def test_accepting_meets_q570s_refusals_and_leaves_it_waiting(
        self, store: Scratch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ada, bo, cy = _three(store)
        monkeypatch.setattr(quota, "CORRESPONDENCE_AT_ONCE", 1)
        c.start(store.connection, SHORT_HASH, [bo, cy])
        sent = ch.challenge(store.connection, ada, SHORT_HASH, 0, "bo")
        with pytest.raises(m.MatchError, match="bo already has 1 games"):
            ch.accept(store.connection, sent.id, bo)
        assert ch.find(store.connection, sent.id).waiting


class TestTheLimits:
    def test_a_day(self, store: Scratch, monkeypatch: pytest.MonkeyPatch) -> None:
        ada, _bo, _cy = _three(store)
        monkeypatch.setattr(quota, "CHALLENGES_A_DAY", 2)
        ch.challenge(store.connection, ada, SHORT_HASH, 0, "bo")
        made = ch.challenge(store.connection, ada, SHORT_HASH, 0, link=True)
        ch.withdraw(store.connection, made.id, ada)
        # Withdrawn or not, a challenge sent today counts against today.
        with pytest.raises(quota.QuotaError, match="limit of 2"):
            ch.challenge(store.connection, ada, SHORT_HASH, 0, "cy")
        with store.connection.cursor() as cursor:
            cursor.execute("update challenges set made = made - interval '25 hours'")
        ch.challenge(store.connection, ada, SHORT_HASH, 0, "cy")

    def test_waiting_at_once(self, store: Scratch, monkeypatch: pytest.MonkeyPatch) -> None:
        ada, bo, _cy = _three(store)
        monkeypatch.setattr(quota, "CHALLENGES_WAITING", 1)
        sent = ch.challenge(store.connection, ada, SHORT_HASH, 0, "bo")
        with pytest.raises(quota.QuotaError, match="1 challenges waiting"):
            ch.challenge(store.connection, ada, UNRATED_HASH, 0, link=True)
        ch.decline(store.connection, sent.id, bo)
        ch.challenge(store.connection, ada, UNRATED_HASH, 0, link=True)

    def test_admin_lists_both(self) -> None:
        listed = dict(quota.summary())
        assert listed["Challenges one person may send a day"] == str(quota.CHALLENGES_A_DAY)
        assert "Q571" not in "".join(listed.values())
        assert str(quota.CHALLENGES_WAITING) in listed.values()


# ------------------------------------------------------------------------ the link


class TestTheLink:
    def test_opened_by_a_member_it_starts_the_game_once(self, store: Scratch) -> None:
        ada, bo, cy = _three(store)
        link = ch.challenge(store.connection, ada, SHORT_HASH, 1, link=True)
        assert link.token and link.challenged is None
        seen = ch.link_view(store.connection, link.token, bo)
        assert seen["state"] == "waiting" and seen["from"] == "ada"
        assert seen["seat"] == "you'd move first" and seen["rated"] is True
        assert seen["visitor"] is False
        made = ch.accept_link(store.connection, link.token, bo)
        assert _seated(store, made.id) == (bo, ada)
        # A second time: taken, by whom, and the game opens only for its two players.
        again = ch.link_view(store.connection, link.token, cy)
        assert again["state"] == "taken" and again["players"] == ["ada", "bo"]
        assert again["match"] is None
        assert ch.link_view(store.connection, link.token, bo)["match"] == made.id
        with pytest.raises(m.MatchError, match="taken already"):
            ch.accept_link(store.connection, link.token, cy)
        assert store.rows() == 1

    def test_the_challengers_own_and_a_withdrawn_one(self, store: Scratch) -> None:
        ada, bo, _cy = _three(store)
        link = ch.challenge(store.connection, ada, SHORT_HASH, None, link=True)
        assert ch.link_view(store.connection, link.token, ada)["state"] == "yours"
        with pytest.raises(m.MatchError, match="your own challenge link"):
            ch.accept_link(store.connection, link.token, ada)
        listed = ch.sent_by(store.connection, ada)
        assert listed[0]["link"] == "/c/" + str(link.token) and listed[0]["to"] is None
        ch.withdraw(store.connection, link.id, ada)
        assert ch.link_view(store.connection, link.token, bo)["state"] == "withdrawn"
        with pytest.raises(m.MatchError, match="withdrawn"):
            ch.accept_link(store.connection, link.token, bo)

    def test_a_visitor_reads_whose_and_nothing_starts(self, store: Scratch) -> None:
        ada = store.user("ada")
        link = ch.challenge(store.connection, ada, UNRATED_HASH, 0, link=True)
        seen = ch.link_view(store.connection, str(link.token), None)
        assert seen["visitor"] is True and seen["from"] == "ada"
        assert seen["name"] == UNRATED["name"] and seen["rated"] is False
        with pytest.raises(KeyError, match="no challenge at this address"):
            ch.link_view(store.connection, "nothing-here", None)

    def test_the_pause(self, store: Scratch) -> None:
        ada, bo, _cy = _three(store)
        link = ch.challenge(store.connection, ada, SHORT_HASH, 0, link=True)
        named = ch.challenge(store.connection, ada, SHORT_HASH, 0, "bo")
        c.set_paused(store.connection, True)
        assert ch.link_view(store.connection, str(link.token), bo)["state"] == "paused"
        for refused in (
            lambda: ch.accept_link(store.connection, str(link.token), bo),
            lambda: ch.accept(store.connection, named.id, bo),
            lambda: ch.challenge(store.connection, bo, SHORT_HASH, 0, "ada"),
        ):
            with pytest.raises(m.MatchError, match="paused"):
                refused()
        # Nothing was used up by the pause.
        assert ch.find(store.connection, named.id).waiting
        c.set_paused(store.connection, False)
        assert ch.accept(store.connection, named.id, bo).running


# ---------------------------------------------------------------------- the rematch


class TestRematch:
    def test_a_finished_game_offers_the_seats_swapped(self, store: Scratch) -> None:
        ada, bo, cy = _three(store)
        played = c.start(store.connection, SHORT_HASH, [ada, bo])
        with pytest.raises(m.MatchError, match="once this game is over"):
            ch.rematch(store.connection, played.id, ada)
        _play_out(store, played.id)
        with pytest.raises(m.MatchError):
            ch.rematch(store.connection, played.id, cy)
        offered = ch.rematch(store.connection, played.id, bo)
        assert offered.rematch_of == played.id and offered.challenged == ada
        waiting = ch.for_you(store.connection, ada)
        assert waiting[0]["rematch"] is True and waiting[0]["from"] == "bo"
        again = ch.accept(store.connection, offered.id, ada)
        assert _seated(store, again.id) == (bo, ada)

    def test_no_rematch_of_a_cancelled_game(self, store: Scratch) -> None:
        ada, bo, _cy = _three(store)
        called_off = c.start(store.connection, SHORT_HASH, [ada, bo])
        c.cancel(store.connection, called_off.id, bo)
        with pytest.raises(m.MatchError, match="cancelled"):
            ch.rematch(store.connection, called_off.id, ada)


# ------------------------------------------------------------------------- leaving


class TestLeaving:
    @pytest.mark.parametrize("how", ["revoked", "forgotten"])
    def test_challenges_are_withdrawn_and_ticks_removed(
        self, store: Scratch, how: str
    ) -> None:
        ada, bo, cy = _three(store)
        sent = ch.challenge(store.connection, ada, SHORT_HASH, 0, "bo")
        link = ch.challenge(store.connection, ada, SHORT_HASH, 0, link=True)
        received = ch.challenge(store.connection, cy, SHORT_HASH, 0, "ada")
        ch.set_open(store.connection, ada, SHORT_HASH, True)
        if how == "revoked":
            admin.kill_user(store.connection, ada)
        else:
            admin.forget(store.connection, "ada")
        for each in (sent, link, received):
            assert ch.find(store.connection, each.id).answer == ch.WITHDRAWN
        assert ch.people(store.connection, SHORT_HASH, bo)["open"] == []
        assert ch.for_you(store.connection, bo) == []


class TestInvitingAFriendFromAGame:
    """Q731: *Play a person* carries the invitations `/admin` gave the viewer, the same
    allowance as their own page, and says plainly why not when there are none."""

    def test_none_given_some_given_and_all_spent(self, store: Scratch) -> None:
        ada, _bo, _cy = _three(store)
        said = ch.people(store.connection, SHORT_HASH, ada)["invites"]
        assert said == {"left": 0, "said": inviting.NONE_GIVEN, "sent": 0, "under": None}
        assert "Ask the person who runs the site" in inviting.NONE_GIVEN

        inviting.set_to_give(store.connection, ada, 1)
        given = ch.people(store.connection, SHORT_HASH, ada)["invites"]
        assert given == inviting.box(store.connection, ada)
        assert (given["left"], given["said"]) == (1, "You can invite one more person.")

        inviting.send(store.connection, ada, "eve@example.com", "https://x", None)
        spent = ch.people(store.connection, SHORT_HASH, ada)["invites"]
        assert (spent["left"], spent["sent"], spent["said"]) == (0, 1, inviting.NONE_LEFT)

    def test_nothing_while_games_between_people_are_paused(self, store: Scratch) -> None:
        ada, _bo, _cy = _three(store)
        c.set_paused(store.connection, True)
        assert "invites" not in ch.people(store.connection, SHORT_HASH, ada)


# -------------------------------------------------------------------- through the site


@pytest.fixture(scope="module")
def app() -> SiteApp:
    return SiteApp(library.RULESET_DIR, library.RUNS_DIR)


def _as(app: SiteApp, viewer: int | None) -> SiteApp:
    app.viewer = viewer
    return app


class TestThroughTheSite:
    def test_tick_challenge_accept_and_the_lists(self, store: Scratch, app: SiteApp) -> None:
        ada, bo, _cy = _three(store)
        app.connect = store.open
        try:
            ticked = _as(app, ada).post(f"/api/game/{SHORT_HASH}/open", {"on": True})
            assert ticked["open"] is True and ticked["people"]["mine"] is True
            standing = _as(app, bo)._standing(SHORT_HASH)["standing"]
            assert standing["people"]["open"][0]["handle"] == "ada"
            sent = _as(app, bo).post(
                "/api/challenge", {"hash": SHORT_HASH, "seat": "second", "handle": "ada"}
            )
            assert sent["to"] == "ada" and sent["seat"] == "you'd move second"
            waiting = _as(app, ada).get("/api/challenges")
            assert [entry["from"] for entry in waiting["forYou"]] == ["bo"]
            assert waiting["open"] == [{"game": SHORT_HASH, "name": SHORT["name"]}]
            assert _as(app, ada).get("/api/profile")["challenges"]["forYou"] == (
                waiting["forYou"]
            )
            started = _as(app, ada).post(f"/api/challenge/{sent['id']}/accept", {})
            read = _as(app, bo).get(f"/api/session/{started['id']}")
            assert read["correspondence"]["players"] == ["ada", "bo"]
            link = _as(app, ada).post(
                "/api/challenge", {"hash": UNRATED_HASH, "seat": "either", "link": True}
            )
            token = link["link"].removeprefix("/c/")
            assert _as(app, None).get(f"/api/challenge/link/{token}")["visitor"] is True
            took = _as(app, bo).post(f"/api/challenge/link/{token}/accept", {})
            assert c.between_people(m.row(store.connection, took["id"]))
            with pytest.raises(m.MatchError, match="taken already"):
                _as(app, bo).post(f"/api/challenge/{sent['id']}/withdraw", {})
        finally:
            app.viewer = None

    def test_decline_withdraw_and_a_rematch_through_the_routes(
        self, store: Scratch, app: SiteApp
    ) -> None:
        ada, bo, _cy = _three(store)
        app.connect = store.open
        try:
            one = _as(app, ada).post("/api/challenge", {"hash": SHORT_HASH, "handle": "bo"})
            two = _as(app, ada).post("/api/challenge", {"hash": UNRATED_HASH, "handle": "bo"})
            mine = _as(app, bo).post(f"/api/challenge/{one['id']}/decline", {})
            assert [entry["id"] for entry in mine["forYou"]] == [two["id"]]
            theirs = _as(app, ada).post(f"/api/challenge/{two['id']}/withdraw", {})
            assert [entry["answer"] for entry in theirs["sent"]] == [ch.DECLINED]
            played = c.start(store.connection, SHORT_HASH, [ada, bo])
            store.connection.commit()
            _play_out(store, played.id)
            again = _as(app, ada).post("/api/challenge", {"rematch": played.id})
            assert again["to"] == "bo" and again["seat"] == "you'd move second"
            with pytest.raises(m.MatchError, match="first, second or either"):
                _as(app, ada).post(
                    "/api/challenge", {"hash": SHORT_HASH, "handle": "bo", "seat": "top"}
                )
        finally:
            app.viewer = None

    def test_a_visitor_is_asked_to_sign_in(self, store: Scratch, app: SiteApp) -> None:
        app.connect = store.open
        _as(app, None)
        for path in ("/api/challenge", f"/api/game/{SHORT_HASH}/open", "/api/challenge/1/accept",
                     "/api/challenge/link/x/accept"):
            with pytest.raises(SignInFirst):
                app.post(path, {})
        with pytest.raises(SignInFirst):
            app.get("/api/challenges")
        assert "people" not in app._standing(SHORT_HASH)["standing"]


# ------------------------------------------------------- the link, over HTTP, then joined


class TestTheLinkOverHttp:
    """The link works for somebody who opens it, then signs in or joins (§1.4).

    Over HTTP, because the return is a page's address and a cookie: the link page is a
    constant, sign-in and joining are handed the link as data, and the joined person's
    cookie accepts it.
    """

    def test_opened_then_joined_then_accepted(self) -> None:
        from test_site import Sent, _running_site

        with _running_site() as site:
            with site.store.cursor() as cursor:
                cursor.execute(
                    "insert into users (handle, email) values ('ana', 'ana@example.invalid')"
                    " returning id"
                )
                ana = int(cursor.fetchone()[0])
                cursor.execute(
                    "insert into games (hash, spec, name, origin)"
                    " values (%s, %s::jsonb, %s, 'corpus')",
                    (UNRATED_HASH, __import__("json").dumps(UNRATED), UNRATED["name"]),
                )
            link = ch.challenge(site.store, ana, UNRATED_HASH, 0, link=True)
            site.store.commit()
            visitor = site.browser()
            page = visitor.get(f"/c/{link.token}")
            assert page.status == 200
            assert f'"token": "{link.token}"' in page.text
            assert "/static/site/challenge.js" in page.text
            whose = visitor.get(f"/api/challenge/link/{link.token}")
            assert whose.status == 200 and '"from": "ana"' in whose.text
            refused = visitor.post(f"/api/challenge/link/{link.token}/accept")
            assert refused.status == 400 and "Sign in" in refused.text
            door = visitor.get(f"/signin?then=/c/{link.token}")
            assert f'"then": "/c/{link.token}"' in door.text
            invited = site.invite("lee@example.com", Sent())
            joining = visitor.get(f"/join?code={invited.code}&then=/c/{link.token}")
            assert f'"then": "/c/{link.token}"' in joining.text
            made = visitor.post(
                "/api/join",
                {"code": invited.code, "handle": "lee", "password": "a-long-enough-one"},
            )
            assert made.status == 200, made.text
            took = visitor.post(f"/api/challenge/link/{link.token}/accept")
            assert took.status == 200, took.text
            assert visitor.get(f"/c/{link.token}").status == 200

    def test_sign_in_returns_only_to_a_challenge_link(self) -> None:
        assert pages._then("/c/abc_DEF-123") == "/c/abc_DEF-123"
        for elsewhere in ("https://elsewhere.example/c/x", "//elsewhere.example", "/admin",
                          "/c/../admin", "/c/", None, "/c/a b"):
            assert pages._then(elsewhere) is None, elsewhere
        assert '"then": null' in pages.signin("https://elsewhere.example")
        assert routes.LINKS == ("/c/",)
        assert pages.LINK.match("/c/" + "a" * 12)


# ------------------------------------------------------------- the seven tests, the code


def _code(text: str) -> str:
    """Code alone: the comments name what a page must never do, in those words."""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("//")
    )


def _people_source() -> str:
    """The game page's *Play a person* and the landing page's *A challenge for you*."""
    detail = (PACKAGE / "web" / "static" / "page" / "detail.js").read_text(encoding="utf-8")
    start = detail.index("// --------------------------------------------------- play a person")
    end = detail.index("// ---", start + 10)
    browse = (PACKAGE / "web" / "static" / "page" / "browse.js").read_text(encoding="utf-8")
    first = browse.index("// ------------------------------------------- what waits on you")
    last = browse.index("// ---", first + 10)
    return _code(detail[start:end] + browse[first:last])


def _sources() -> dict[str, str]:
    profile = (STATIC / "profile.js").read_text(encoding="utf-8")
    start = profile.index("function drawChallenges")
    return {
        "challenge.js": _code((STATIC / "challenge.js").read_text(encoding="utf-8")),
        "profile.js": _code(profile[start : profile.index("\n}\n", start)]),
        "pages": _people_source(),
        "module": (PACKAGE / "site" / "challenges.py").read_text(encoding="utf-8"),
    }


class TestTheSevenTests:
    def test_nothing_expires_or_counts_down(self) -> None:
        for name, source in _sources().items():
            code = source if name != "module" else re.sub(r'""".*?"""', "", source, flags=re.S)
            for word in ("expires", "expiry", "days left", "time left", "remaining",
                         "deadline", "countdown", "hurry", "lapse"):
                assert word not in code.lower(), (name, word)

    def test_no_badge_count_dot_or_title(self) -> None:
        for name, source in _sources().items():
            if name == "module":
                continue
            for mark in ("--bad", "badge", "document.title", "Notification", "Audio"):
                assert mark not in source, (name, mark)
            assert not re.search(r"\b(red|dot)\b", source), name

    def test_the_box_is_off_by_default(self) -> None:
        migration = (PACKAGE / "site" / "migrations" / "0023_challenges.sql").read_text(
            encoding="utf-8"
        )
        # Off is the absence of a row; nothing writes one but a tick.
        assert "insert into open_to" not in migration
        module = _sources()["module"]
        assert module.count("insert into open_to") == 1

    def test_the_module_works_out_no_position(self) -> None:
        """Rule 3: whose move, what is legal and how it ends stay the engine's."""
        module = _sources()["module"]
        for word in ("legal_moves", "Session(", "side_to_move", "terminal", "outcome",
                     "engine", "_advance"):
            assert word not in re.sub(r'""".*?"""', "", module, flags=re.S), word
        assert "correspondence.start(" in module


def _link_says(found: str) -> Any:
    import json

    from node_session import run

    source = (STATIC / "challenge.js").read_text(encoding="utf-8")
    stub = """
      const el = new Proxy({}, { get: (t, k) => (k === 'textContent' ? '{"token": "t1"}'
        : () => el), set: () => true });
      global.document = { getElementById: () => el, createElement: () => el };
      global.fetch = () => new Promise(() => {});
      global.localStorage = { setItem: () => {}, getItem: () => null };
      const PLAY = '/#/play/', GAME = '/#/game/', HELD_LINK = 'challenge-link';
      function post() {} function say() {}
    """
    done = run(stub + source + f"\nconsole.log(JSON.stringify(linkSays({found})));")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


class TestTheLinkPageSays:
    BASE = ("{from: 'ana', name: 'Short', game: 'h', seat: \"you'd move first\", rated: true,"
            " players: ['ana', 'lee'], rules: ['No clock.'], visitor: false, state: '%s'}")

    def test_a_member_is_offered_accept(self) -> None:
        said = _link_says(self.BASE % "waiting")
        assert said["accept"] is True and said["rules"] == ["No clock."]
        assert said["lines"] == [
            "ana challenges you to Short: you'd move first, and it's rated: the result "
            "moves both of your ratings on it."
        ]

    def test_a_visitor_is_told_whose_and_how_to_sign_in(self) -> None:
        said = _link_says((self.BASE % "waiting").replace("visitor: false", "visitor: true"))
        assert said.get("accept") is None and said["signin"] is True
        assert "challenge from ana" in said["lines"][0]
        assert "by invitation" in said["lines"][1]

    def test_a_second_opening_says_taken(self) -> None:
        said = _link_says(self.BASE % "taken")
        assert said["lines"] == ["This challenge has been taken: ana and lee are playing it."]
        assert said.get("accept") is None


# ------------------------------------------------------ the game page and the landing page

_PEOPLE = """
const GAMEX = { name: 'Short', hash: 'h1', sides: ['north', 'south'], standing: { people: {
  paused: false, pausedSays: null, rated: true, mine: false,
  open: [{ handle: 'ana', said: "hasn't played this one yet" }],
  members: ['ana', 'lee'], rules: ['There is no clock.'] } } };
"""


class TestTheGamePageDraws:
    """*Play a person* and the sheet, run in node over a hand-made game (s1, s2)."""

    def test_the_panel(self) -> None:
        from test_web import _in_page

        panel = html.unescape(_in_page(_PEOPLE, "playPersonHtml(GAMEX)"))
        assert "Play a person" in panel and "Open to a game of Short" in panel
        assert '<b><a href="/players/ana">ana</a></b>' in panel
        assert "hasn't played this one yet" in panel
        assert 'data-challenge="ana"' in panel
        assert '<option value="lee">' in panel and "make a challenge link" in panel
        assert 'id="openTo"' in panel and " checked" not in panel
        assert "Off unless you tick it" in panel and "nothing is" in panel
        # *Start a game* stays the one primary button on the page.
        assert "primary" not in panel

    def test_nobody_open_a_visitor_and_the_pause(self) -> None:
        from test_web import _in_page

        empty = html.unescape(
            _in_page(_PEOPLE + "GAMEX.standing.people.open = [];", "playPersonHtml(GAMEX)")
        )
        assert "Nobody has said they're open to a game of Short yet." in empty
        assert _in_page(_PEOPLE + "GAMEX.standing = {};", "playPersonHtml(GAMEX)") == ""
        paused = _in_page(
            _PEOPLE + "Object.assign(GAMEX.standing.people, { paused: true, mine: true,"
            " pausedSays: 'New games between people are paused for now.' });",
            "playPersonHtml(GAMEX)",
        )
        assert "paused for now" in paused and "challenge" not in paused.lower().replace(
            "challengesheet", ""
        )
        assert 'id="openTo" checked' in paused

    def test_the_sheet_names_the_games_own_sides_and_the_rules(self) -> None:
        from test_web import _in_page

        sheet = html.unescape(_in_page(_PEOPLE, "challengeSheetHtml(GAMEX, 'ana')"))
        assert "Challenge ana to Short" in sheet
        assert "north, moving first" in sheet and "south, moving second" in sheet
        assert "either, drawn when it starts" in sheet
        assert "<li>There is no clock.</li>" in sheet and "rated on Short" in sheet
        link = _in_page(_PEOPLE + "GAMEX.standing.people.rated = false;",
                        "challengeSheetHtml(GAMEX, null)")
        assert "A challenge link to Short" in link and "practice game" in link
        sent = _in_page(_PEOPLE, "challengeSentHtml({ link: '/c/t1' }, 'https://x')")
        assert 'value="https://x/c/t1"' in sent and "works once" in sent
        named = _in_page(_PEOPLE, "challengeSentHtml({ to: 'ana', link: null }, '')")
        assert "Challenge sent." in named and "ana accepts or declines" in named


    def test_a_friend_who_is_not_a_member_yet(self) -> None:
        """Q731. The friend you would invite is who you want to play: with invitations
        left, the form to `/api/invite`; with none, the server's sentence and no form."""
        from test_web import _in_page

        given = (_PEOPLE + "GAMEX.standing.people.invites = { left: 2, sent: 0,"
                 " said: 'You can invite 2 more people.', under: 'They get an email.' };")
        panel = html.unescape(_in_page(given, "playPersonHtml(GAMEX)"))
        assert "A friend who isn't a member yet" in panel
        assert "You can invite 2 more people." in panel and "They get an email." in panel
        assert 'id="inviteForm"' in panel and 'type="email"' in panel
        assert "send the invitation" in panel and "primary" not in panel
        none = (_PEOPLE + "GAMEX.standing.people.invites = { left: 0, sent: 0,"
                f" said: {inviting.NONE_GIVEN!r}, under: null }};")
        panel = html.unescape(_in_page(none, "playPersonHtml(GAMEX)"))
        assert "A friend who isn't a member yet" in panel and inviting.NONE_GIVEN in panel
        assert 'id="inviteForm"' not in panel and "send the invitation" not in panel
        # The same route as your own page, so the same allowance and refusals.
        assert "api('/api/invite', { email:" in _people_source()


class TestTheLandingPageDraws:
    def test_a_challenge_for_you_and_nothing_when_there_is_none(self) -> None:
        from test_web import _in_page

        said = html.unescape(_in_page(
            "",
            "forYouHtml({ forYou: [{ id: 4, from: 'ana', game: 'h1', name: 'Short',"
            " seat: \"you'd move second\", rated: false, rematch: false }] })",
        ))
        assert "A challenge for you" in said and "<b>ana</b> challenged you to" in said
        assert 'data-accept="4"' in said and 'data-decline="4"' in said
        assert "practice" in said
        for nothing in ("{ forYou: [] }", "null"):
            assert _in_page("", f"forYouHtml({nothing})") == ""

    def test_the_finished_game_offers_a_rematch(self) -> None:
        from test_web import _in_page

        bar = _in_page(
            "", "frameHtml(false, 'leaderboard', false, { you: 0, mayCancel: false })"
        )
        assert 'id="rematch" hidden' in bar
        reader = _in_page(
            "", "frameHtml(false, 'leaderboard', false, { you: null, mayCancel: false })"
        )
        assert 'id="rematch"' not in reader
