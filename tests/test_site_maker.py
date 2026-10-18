"""The maker takes challengers, and E11 *Beat the maker* (Q606, `docs/archive/plan-return.md` §3 C4, R10).

A game's designer switches one thing on, on their own page (`users.takes_challengers`, off
by default), and is then named first in *Play a person* on each of their published games as
*‹handle› made this game and takes challengers*. Any other member challenges them there
through Q571's ordinary challenge, which carries no message. A rated win over them on their
own game earns E11, once per game for each winner.

Proved for the four readers the task names: **the maker** (the switch, its refusals, what
they read on their own game), **a challenger** (the line, the challenge, the accepted game,
the badge), **a visitor** (nothing drawn, the route refused) and **a member who is not
eligible** (no published game of their own: the switch is refused; somebody else's game,
a draft or a library game: no line). Beside them, the reader of E11 on hand-made rows, the
words guard on every new sentence, and that neither a challenge nor the switch moves a
game's score.

On specs, never chess (rule 5 of CLAUDE.md): the `dull` variants of `test_site_matches.py`,
made into designed games. The store tests need `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL`
and skip without it, as every site test does; the page checks need node.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import site_words
from test_site_matches import (
    LADDER,
    SHORT,
    SHORT_HASH,
    UNRATED,
    UNRATED_HASH,
    Scratch,
    _dull,
    _scratch,
)

from chess_variant_sandbox.site import admin, awards, badges, rating, routes
from chess_variant_sandbox.site import challenges as ch
from chess_variant_sandbox.site import correspondence as c
from chess_variant_sandbox.site import matches as m
from chess_variant_sandbox.site.server import CHALLENGE_FIELDS, SignInFirst, SiteApp
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.page import marks

REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE = REPOSITORY / "chess_variant_sandbox"
STATIC = PACKAGE / "site" / "static"

DRAFT = _dull(3)
DRAFT_HASH = spec_hash(DRAFT)


def _make_designed(store: Scratch, digest: str, owner: int, *, public: bool = True) -> None:
    """Turn a game the store holds into `owner`'s designed game, on trial or a draft."""
    with store.connection.cursor() as cursor:
        cursor.execute(
            "update games set origin = 'designed', owner = %s, visibility = %s,"
            " published = case when %s then now() end where hash = %s",
            (owner, "public" if public else "unlisted", public, digest),
        )
    store.connection.commit()


@pytest.fixture
def store() -> Iterator[Scratch]:
    """ada's two published games (one rated, one practice) and her draft; bo and cy."""
    with _scratch() as scratch:
        scratch.game(SHORT_HASH, SHORT, LADDER)
        scratch.game(UNRATED_HASH, UNRATED)
        scratch.game(DRAFT_HASH, DRAFT)
        yield scratch


def _people(store: Scratch) -> tuple[int, int, int]:
    ada, bo, cy = store.user("ada"), store.user("bo"), store.user("cy")
    for digest in (SHORT_HASH, UNRATED_HASH):
        _make_designed(store, digest, ada)
    _make_designed(store, DRAFT_HASH, ada, public=False)
    return ada, bo, cy


def _switch(store: Scratch, user: int, on: bool) -> bool:
    found = ch.set_takes_challengers(store.connection, user, on)
    store.connection.commit()
    return found


def _play_out(store: Scratch, match_id: str) -> m.Row:
    """Both players send the engine's first offered move; the short game then ends."""
    for _ in range(2):
        found = m.row(store.connection, match_id)
        snapshot = m.session(store.connection, match_id).snapshot()
        who = found.player(snapshot["sideToMove"]).user
        c.move(store.connection, match_id, who, int(snapshot["moves"][0]["move"]))
    store.connection.commit()
    return m.row(store.connection, match_id)


# --------------------------------------------------------------------- the maker


class TestTheMakersSwitch:
    def test_off_by_default_and_on_at_the_makers_word(self, store: Scratch) -> None:
        ada, bo, _cy = _people(store)
        box = ch.maker_box(store.connection, ada)
        assert box["on"] is False
        # Her published games, oldest first; never the draft.
        assert {game["game"] for game in box["games"]} == {SHORT_HASH, UNRATED_HASH}
        assert ch.people(store.connection, SHORT_HASH, bo)["maker"] is None
        assert _switch(store, ada, True) is True
        assert ch.maker_box(store.connection, ada)["on"] is True

    def test_turned_off_it_is_gone(self, store: Scratch) -> None:
        ada, bo, _cy = _people(store)
        _switch(store, ada, True)
        assert ch.people(store.connection, SHORT_HASH, bo)["maker"] is not None
        assert _switch(store, ada, False) is False
        assert ch.people(store.connection, SHORT_HASH, bo)["maker"] is None

    def test_the_maker_reads_that_they_are_listed_and_is_offered_nobody(
        self, store: Scratch
    ) -> None:
        ada, _bo, _cy = _people(store)
        _switch(store, ada, True)
        mine = ch.people(store.connection, SHORT_HASH, ada)["maker"]
        assert mine["you"] is True and mine["earns"] is None
        assert mine["said"].startswith("You made this game and take challengers")

    def test_the_pause_refuses_switching_on_and_never_switching_off(
        self, store: Scratch
    ) -> None:
        ada, bo, _cy = _people(store)
        _switch(store, ada, True)
        c.set_paused(store.connection, True)
        assert ch.people(store.connection, SHORT_HASH, bo)["maker"] is None
        assert _switch(store, ada, False) is False
        with pytest.raises(m.MatchError, match="paused"):
            ch.set_takes_challengers(store.connection, ada, True)

    def test_leaving_and_revoking_clear_it(self, store: Scratch) -> None:
        ada, bo, _cy = _people(store)
        _switch(store, ada, True)
        admin.kill_user(store.connection, ada)
        store.connection.commit()
        assert ch.people(store.connection, SHORT_HASH, bo)["maker"] is None
        assert ch.takes_challengers(store.connection, ada) is False
        dee = store.user("dee")
        _make_designed(store, UNRATED_HASH, dee)
        _switch(store, dee, True)
        admin.forget(store.connection, "dee")
        store.connection.commit()
        assert ch.takes_challengers(store.connection, dee) is False
        assert ch.people(store.connection, UNRATED_HASH, bo)["maker"] is None


# ----------------------------------------------------- a member who is not eligible


class TestWhoIsNotEligible:
    def test_no_published_game_of_their_own_refuses_the_switch(self, store: Scratch) -> None:
        _ada, bo, _cy = _people(store)
        with pytest.raises(m.MatchError, match="no published game of your own"):
            ch.set_takes_challengers(store.connection, bo, True)
        # Off always works, and the box draws nothing for them.
        assert _switch(store, bo, False) is False
        assert ch.maker_box(store.connection, bo)["games"] == []

    def test_a_draft_alone_is_not_a_published_game(self, store: Scratch) -> None:
        _ada, bo, _cy = _people(store)
        _make_designed(store, DRAFT_HASH, bo, public=False)
        with pytest.raises(m.MatchError, match="no published game of your own"):
            ch.set_takes_challengers(store.connection, bo, True)

    def test_only_the_games_owner_is_its_maker(self, store: Scratch) -> None:
        ada, bo, cy = _people(store)
        _make_designed(store, UNRATED_HASH, bo)
        _switch(store, ada, True)
        _switch(store, bo, True)
        # ada is the maker of the rated game alone now; bo of the practice one.
        assert ch.people(store.connection, SHORT_HASH, cy)["maker"]["handle"] == "ada"
        assert ch.people(store.connection, UNRATED_HASH, cy)["maker"]["handle"] == "bo"

    def test_no_line_on_a_draft_or_a_game_that_came_with_the_site(
        self, store: Scratch
    ) -> None:
        ada, bo, _cy = _people(store)
        _switch(store, ada, True)
        assert ch.people(store.connection, DRAFT_HASH, bo)["maker"] is None
        # A library game the maintainer's account owns has no maker (`origin` decides).
        with store.connection.cursor() as cursor:
            cursor.execute(
                "update games set origin = 'corpus' where hash = %s", (UNRATED_HASH,)
            )
        store.connection.commit()
        assert ch.people(store.connection, UNRATED_HASH, bo)["maker"] is None


# ---------------------------------------------------------------------- a challenger


class TestAChallenger:
    def test_the_line_comes_first_and_names_what_a_rated_win_earns(
        self, store: Scratch
    ) -> None:
        ada, bo, _cy = _people(store)
        _switch(store, ada, True)
        seen = ch.people(store.connection, SHORT_HASH, bo)
        assert seen["maker"] == {
            "handle": "ada",
            "mark": marks.mark(ada),
            "you": False,
            "said": "made this game and takes challengers.",
            "earns": "A win over ada in a rated game here earns the badge Beat the maker.",
        }
        # A practice game earns nothing, and says nothing about a badge.
        assert ch.people(store.connection, UNRATED_HASH, bo)["maker"]["earns"] is None

    def test_the_maker_is_not_listed_twice(self, store: Scratch) -> None:
        ada, bo, _cy = _people(store)
        ch.set_open(store.connection, ada, SHORT_HASH, True)
        assert [p["handle"] for p in ch.people(store.connection, SHORT_HASH, bo)["open"]] == [
            "ada"
        ]
        _switch(store, ada, True)
        seen = ch.people(store.connection, SHORT_HASH, bo)
        assert seen["open"] == [] and seen["maker"]["handle"] == "ada"

    def test_challenged_through_the_ordinary_challenge(self, store: Scratch) -> None:
        ada, bo, _cy = _people(store)
        _switch(store, ada, True)
        sent = ch.challenge(store.connection, bo, SHORT_HASH, 0, "ada")
        store.connection.commit()
        assert [entry["from"] for entry in ch.for_you(store.connection, ada)] == ["bo"]
        made = ch.accept(store.connection, sent.id, ada)
        store.connection.commit()
        assert (made.player(0).user, made.player(1).user) == (bo, ada)
        assert made.mode == m.LEADERBOARD
        # The maker may still decline, as with any challenge.
        again = ch.challenge(store.connection, bo, UNRATED_HASH, None, "ada")
        assert ch.decline(store.connection, again.id, ada).answer == ch.DECLINED


def _won_by(store: Scratch, game: str, players: list[int], winner: int) -> m.Row:
    """A game between `players`, finished as a win for `winner`.

    Through `matches._finish`, the one door every ending goes through (the engine's
    outcome, a resignation, an agreed draw), so the rating and the badges it completes
    land as they do for a played game. The `dull` variants are drawn whoever plays them,
    and E11 is about the verdict, not about how the engine reached it.
    """
    started = c.start(store.connection, game, players)
    m._finish(store.connection, started.id, result=1.0 if players[0] == winner else 0.0,
              reason="goal", left=False)
    store.connection.commit()
    return m.row(store.connection, started.id)


def _e11_on(store: Scratch, game: str) -> list[tuple[int | None, str]]:
    return [(a.holder, a.evidence["match"]) for a in awards.held(store.connection, game=game)
            if a.kind == badges.BEAT_THE_MAKER]


class TestBeatTheMakerInTheStore:
    def test_a_rated_win_over_the_maker_earns_it_and_the_makers_win_does_not(
        self, store: Scratch
    ) -> None:
        """E11's *done when*: a rated match with the owner seated and a win for the
        other person earns it, and the owner's win earns nobody anything."""
        ada, bo, _cy = _people(store)
        made = _won_by(store, SHORT_HASH, [ada, bo], ada)
        assert made.mode == m.LEADERBOARD and made.result == 1.0
        assert _e11_on(store, SHORT_HASH) == []
        won = _won_by(store, SHORT_HASH, [ada, bo], bo)
        assert _e11_on(store, SHORT_HASH) == [(bo, won.id)]
        # Once per game for each winner: a second win earns nothing more.
        _won_by(store, SHORT_HASH, [bo, ada], bo)
        assert _e11_on(store, SHORT_HASH) == [(bo, won.id)]
        # It is bo's, on his page; nothing on ada's.
        assert [a.kind for a in awards.held(store.connection, holder=bo)
                if a.kind == badges.BEAT_THE_MAKER] == [badges.BEAT_THE_MAKER]
        assert [a for a in awards.held(store.connection, holder=ada)
                if a.kind == badges.BEAT_THE_MAKER] == []

    def test_a_draw_earns_nothing(self, store: Scratch) -> None:
        ada, bo, _cy = _people(store)
        for players in ([bo, ada], [ada, bo]):
            finished = _play_out(store, c.start(store.connection, SHORT_HASH, players).id)
            assert finished.result == 0.5
        assert _e11_on(store, SHORT_HASH) == []

    def test_a_practice_game_earns_nothing(self, store: Scratch) -> None:
        ada, bo, _cy = _people(store)
        finished = _won_by(store, UNRATED_HASH, [ada, bo], bo)
        assert finished.mode == m.SANDBOX and finished.result == 0.0
        assert [a for a in awards.held(store.connection, holder=bo)
                if a.kind == badges.BEAT_THE_MAKER] == []


# ---------------------------------------------------------- E11, the reader on rows

_ids = iter(range(10_000))
ADA, BO, CY = 1, 2, 3


def _when(day: int) -> datetime:
    return datetime(2026, 10, 1, tzinfo=timezone.utc) + timedelta(days=day)


def _between(game: str, winner: int, loser: int, day: int, **changed: Any) -> m.Row:
    """A finished rated game between two people, `winner` in seat 0."""
    fields: dict[str, Any] = {
        "id": f"p{next(_ids):04d}", "game": game, "mode": m.LEADERBOARD,
        "seat0": rating.person(winner), "seat1": rating.person(loser), "seed": 0,
        "plies": 20, "result": 1.0, "reason": "checkmate", "left": False, "top": False,
        "top_players": (), "before": (None, None), "source": None, "started": _when(day),
        "moved": None, "finished": _when(day), "deviations": (None, None),
        "rematch_of": None,
    }
    fields.update(changed)
    return m.Row(**fields)


def _design(game: str, owner: int, origin: str = "designed", revoked: bool = False) -> Any:
    return badges.Design(game=game, owner=owner, parent=None, origin=origin,
                         screen_passed=True, created=_when(-10), revoked=revoked)


GAMES = (_design("made", ADA), _design("other", CY), _design("lib", ADA, origin="corpus"),
         _design("killed", ADA, revoked=True))


def _e11(user: int, rows: list[m.Row]) -> list[awards.Award]:
    found = badges.due_from(user, rows, {}, {}, (), GAMES)
    return [award for award in found if award.kind == badges.BEAT_THE_MAKER]


class TestBeatTheMakerReader:
    def test_a_win_over_the_owner_on_their_game_earns_it_once_per_game(self) -> None:
        first = _between("made", BO, ADA, 1)
        found = _e11(BO, [first, _between("made", BO, ADA, 2)])
        assert len(found) == 1
        award = found[0]
        assert (award.holder, award.game, award.earned) == (BO, "made", _when(1))
        assert award.evidence == {"match": first.id, "seat": 0}
        # A second winner earns their own.
        assert len(_e11(CY, [_between("made", CY, ADA, 3)])) == 1

    def test_the_owner_beating_somebody_earns_nothing(self) -> None:
        assert _e11(ADA, [_between("made", ADA, BO, 1)]) == []
        assert _e11(BO, [_between("made", ADA, BO, 1)]) == []

    def test_not_the_owner_not_their_game_not_a_designed_game(self) -> None:
        assert _e11(BO, [_between("made", BO, CY, 1)]) == []      # cy did not make it
        assert _e11(BO, [_between("other", BO, ADA, 1)]) == []    # ada did not make it
        assert _e11(BO, [_between("lib", BO, ADA, 1)]) == []      # came with the site
        assert _e11(BO, [_between("killed", BO, ADA, 1)]) == []   # killed

    def test_a_draw_practice_resignation_or_leaving_earns_nothing(self) -> None:
        for changed in ({"result": 0.5}, {"mode": m.SANDBOX}, {"reason": "resigned"},
                        {"left": True}):
            assert _e11(BO, [_between("made", BO, ADA, 1, **changed)]) == [], changed

    def test_already_held_is_not_offered_again(self) -> None:
        row = _between("made", BO, ADA, 1)
        held = tuple(_e11(BO, [row]))
        again = badges.due_from(BO, [row], {}, {}, held, GAMES)
        assert [a for a in again if a.kind == badges.BEAT_THE_MAKER] == []

    def test_the_badge_names_what_it_reads(self) -> None:
        badge = badges.BADGES[badges.BEAT_THE_MAKER]
        assert badge.name == "Beat the maker"
        for said in ("rated game between two people", "designed", "Once per game",
                     "practice game"):
            assert said in badge.rule, said


# ------------------------------------------------------------ the routes, a visitor


@pytest.fixture(scope="module")
def app() -> SiteApp:
    return SiteApp(library.RULESET_DIR, library.RUNS_DIR)


def _as(app: SiteApp, viewer: int | None) -> SiteApp:
    app.viewer = viewer
    return app


class TestThroughTheSite:
    def test_switch_line_challenge_and_profile(self, store: Scratch, app: SiteApp) -> None:
        ada, bo, cy = _people(store)
        app.connect = store.open
        try:
            assert _as(app, ada).get("/api/profile")["takesChallengers"]["on"] is False
            box = _as(app, ada).post("/api/takes-challengers", {"on": True})
            assert box["on"] is True and box["said"] == ch.MAKER_SAID
            people = _as(app, bo)._standing(SHORT_HASH)["standing"]["people"]
            assert people["maker"]["handle"] == "ada"
            sent = _as(app, bo).post(
                "/api/challenge", {"hash": SHORT_HASH, "seat": "first", "handle": "ada"}
            )
            assert sent["to"] == "ada"
            with pytest.raises(m.MatchError, match="no published game of your own"):
                _as(app, cy).post("/api/takes-challengers", {"on": True})
            assert _as(app, cy).get("/api/profile")["takesChallengers"]["games"] == []
            off = _as(app, ada).post("/api/takes-challengers", {"on": False})
            assert off["on"] is False
            assert _as(app, bo)._standing(SHORT_HASH)["standing"]["people"]["maker"] is None
        finally:
            app.viewer = None

    def test_a_challenge_carries_no_message(self, store: Scratch, app: SiteApp) -> None:
        ada, bo, _cy = _people(store)
        _switch(store, ada, True)
        app.connect = store.open
        try:
            for stray in ("message", "note", "text", "comment"):
                with pytest.raises(m.MatchError, match="nothing else"):
                    _as(app, bo).post(
                        "/api/challenge",
                        {"hash": SHORT_HASH, "handle": "ada", stray: "come and play"},
                    )
            assert ch.for_you(store.connection, ada) == []
            assert {"hash", "seat", "handle", "link", "rematch"} == CHALLENGE_FIELDS
            # Nor can the table hold one.
            with store.connection.cursor() as cursor:
                cursor.execute(
                    "select column_name from information_schema.columns"
                    " where table_name = 'challenges' and table_schema = current_schema()"
                )
                columns = {row[0] for row in cursor.fetchall()}
            assert not columns & {"message", "note", "text", "comment", "body"}
        finally:
            app.viewer = None

    def test_a_visitor_sees_no_line_and_cannot_switch(
        self, store: Scratch, app: SiteApp
    ) -> None:
        ada, _bo, _cy = _people(store)
        _switch(store, ada, True)
        app.connect = store.open
        _as(app, None)
        with pytest.raises(SignInFirst):
            app.post("/api/takes-challengers", {"on": True})
        assert "people" not in app._standing(SHORT_HASH)["standing"]
        visitor = [route for route in routes.OWN if route.path == "/api/takes-challengers"]
        assert visitor and visitor[0].visitor

    def test_neither_the_switch_nor_a_challenge_moves_a_score(
        self, store: Scratch, app: SiteApp
    ) -> None:
        ada, bo, _cy = _people(store)
        app.connect = store.open
        try:
            before = _as(app, bo).get(f"/api/game/{SHORT_HASH}")
            _as(app, ada).post("/api/takes-challengers", {"on": True})
            sent = _as(app, bo).post("/api/challenge", {"hash": SHORT_HASH, "handle": "ada"})
            _as(app, ada).post(f"/api/challenge/{sent['id']}/decline", {})
            after = _as(app, bo).get(f"/api/game/{SHORT_HASH}")
            # Everything the game page is sent but the people's panel, where the line is.
            assert {k: v for k, v in before.items() if k != "standing"} == {
                k: v for k, v in after.items() if k != "standing"
            }
        finally:
            app.viewer = None


# ------------------------------------------------------------------ the words guard


class TestTheWords:
    def test_every_new_sentence_passes_the_guard(self) -> None:
        said = [ch.MAKER_SAID, ch.MAKER_UNDER, ch.NO_GAME_YET,
                "made this game and takes challengers.",
                "You made this game and take challengers: members see you here first.",
                "A win over ada in a rated game here earns the badge Beat the maker.",
                badges.BADGES[badges.BEAT_THE_MAKER].sentence,
                badges.BADGES[badges.BEAT_THE_MAKER].rule]
        page = (PACKAGE / "web" / "static" / "page" / "detail.js").read_text(encoding="utf-8")
        start = page.index("function makerHtml")
        said += site_words.literals(page[start: page.index("\n}\n", start)])
        profile = (STATIC / "profile.js").read_text(encoding="utf-8")
        start = profile.index("function drawTakesChallengers")
        said += site_words.literals(profile[start: profile.index("\n}\n", start)])
        trouble = [(text, site_words.hits(text)) for text in said if site_words.hits(text)]
        assert not trouble, trouble

    def test_the_privacy_notice_says_it(self) -> None:
        from chess_variant_sandbox.site import pages

        notice = pages.privacy()
        assert "taking challengers" in notice and "off unless you switch it on" in notice
        assert "members, and nobody else" in notice and "holds no message" in notice

    def test_the_seven_tests_in_the_new_code(self) -> None:
        """No countdown or lapse, no count, no dot, nothing sent."""
        page = (PACKAGE / "web" / "static" / "page" / "detail.js").read_text(encoding="utf-8")
        start = page.index("function makerHtml")
        profile = (STATIC / "profile.js").read_text(encoding="utf-8")
        first = profile.index("function drawTakesChallengers")
        code = "\n".join(
            line for line in (page[start: page.index("\n}\n", start)]
                              + profile[first: profile.index("\n}\n", first)]).splitlines()
            if not line.lstrip().startswith("//")
        )
        for word in ("expires", "time left", "remaining", "countdown", "lapse",
                     "document.title", "Notification", "--bad", "length}"):
            assert word not in code, word
        assert not re.search(r"\b(red|dot)\b", code)
        migration = (PACKAGE / "site" / "migrations" / "0027_takes_challengers.sql")
        assert "default false" in migration.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- the page

_PEOPLE = """
const GAMEX = { name: 'Short', hash: 'h1', sides: ['north', 'south'], standing: { people: {
  paused: false, pausedSays: null, rated: true, mine: false,
  open: [{ handle: 'lee', said: "hasn't played this one yet" }],
  maker: { handle: 'ana', mark: null, you: false,
           said: 'made this game and takes challengers.',
           earns: 'A win over ana in a rated game here earns the badge Beat the maker.' },
  members: ['ana', 'lee'], rules: ['There is no clock.'] } } };
"""


class TestTheGamePageDraws:
    def test_the_maker_comes_first_with_a_challenge(self) -> None:
        from test_web import _in_page

        panel = html.unescape(_in_page(_PEOPLE, "playPersonHtml(GAMEX)"))
        assert '<b><a href="/players/ana">ana</a></b>' in panel
        assert "made this game and takes challengers." in panel
        assert 'data-challenge="ana"' in panel
        assert "earns the badge Beat the maker" in panel
        assert panel.index('id="makerLine"') < panel.index("Open to a game of Short")
        assert "primary" not in panel

    def test_the_maker_reads_their_own_line_and_nobody_else_sees_one(self) -> None:
        from test_web import _in_page

        own = html.unescape(_in_page(
            _PEOPLE + "Object.assign(GAMEX.standing.people.maker, { you: true, earns: null,"
            " said: 'You made this game and take challengers: members see you here first.' });",
            "playPersonHtml(GAMEX)",
        ))
        assert "You made this game and take challengers" in own
        assert 'data-challenge="ana"' not in own and 'href="/profile"' in own
        none = _in_page(_PEOPLE + "GAMEX.standing.people.maker = null;", "playPersonHtml(GAMEX)")
        assert 'id="makerLine"' not in none

    def test_the_profile_draws_the_switch_only_with_a_game_and_posts_it(self) -> None:
        profile = (STATIC / "profile.js").read_text(encoding="utf-8")
        start = profile.index("function drawTakesChallengers")
        body = profile[start: profile.index("\n}\n", start)]
        assert "if (!box || !(box.games || []).length) return;" in body
        assert "post('/api/takes-challengers', { on: tick.checked })" in body
        assert "drawTakesChallengers(payload.takesChallengers);" in profile
        assert "getElementById('takes-challengers')" in body
