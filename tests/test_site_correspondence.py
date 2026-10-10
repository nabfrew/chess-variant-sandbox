"""A game between two people, played by correspondence (Q570, `docs/plan-return.md` §1).

Four endings, each proved on a spec rather than on chess (rule 5 of CLAUDE.md): the
rules' own (a `dull` variant whose ply limit ends it after one move each), a
resignation, a draw both people agree to, and the loss for the player to move after
`quota.CORRESPONDENCE_IDLE`. Beside them, the parts that make two people's game
different from a game against the computer: whose turn it is is the engine's, a second
rated game against the computer resigns nothing between people, nobody outside a running
game reads it, leaving and revoking resign a person's games, `/admin`'s pause, and a
cancel before both have moved that no reading counts.

Nothing here works out a legal move, a winner or whose move it is: the tests take the
engine's first offered move and read the engine's side to move off the snapshot.

The store tests need `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL` and skip without it, as
every site test does. The page and the metrics checks need nothing.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from test_site_matches import (
    AGENT,
    LADDER,
    SHORT,
    SHORT_HASH,
    UNRATED,
    UNRATED_HASH,
    Scratch,
    _scratch,
)

from chess_variant_sandbox.site import admin, quota
from chess_variant_sandbox.site import correspondence as c
from chess_variant_sandbox.site import matches as m
from chess_variant_sandbox.site import rating as r
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.web import library

REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE = REPOSITORY / "chess_variant_sandbox"


@pytest.fixture
def store() -> Iterator[Scratch]:
    """A store with one rated game (a loaded ladder), one without, and nobody yet."""
    with _scratch() as scratch:
        scratch.game(SHORT_HASH, SHORT, LADDER)
        scratch.game(UNRATED_HASH, UNRATED)
        yield scratch


def _pair(store: Scratch, game: str = SHORT_HASH) -> tuple[m.Row, int, int]:
    ada, bo = store.user("ada"), store.user("bo")
    found = c.start(store.connection, game, [ada, bo])
    store.connection.commit()
    return found, ada, bo


def _snapshot(store: Scratch, match_id: str) -> dict[str, Any]:
    return m.session(store.connection, match_id).snapshot()


def _mover(store: Scratch, match_id: str, ada: int, bo: int) -> int:
    """The person whose move it is, read off the engine's side to move."""
    return (ada, bo)[_snapshot(store, match_id)["sideToMove"]]


def _first_move(store: Scratch, match_id: str) -> int:
    return int(_snapshot(store, match_id)["moves"][0]["move"])


def _move(store: Scratch, match_id: str, ada: int, bo: int) -> dict[str, Any]:
    """The player to move sends the engine's first offered move."""
    who = _mover(store, match_id, ada, bo)
    return c.move(store.connection, match_id, who, _first_move(store, match_id))


# --------------------------------------------------------------------------- starting


class TestStarting:
    def test_rated_where_the_game_has_rated_computer_players_and_practice_elsewhere(
        self, store: Scratch
    ) -> None:
        rated, ada, bo = _pair(store)
        practice = c.start(store.connection, UNRATED_HASH, [bo, ada])
        assert rated.mode == m.LEADERBOARD
        assert practice.mode == c.PRACTICE == m.SANDBOX
        assert c.between_people(rated) and c.between_people(practice)
        assert rated.seat0 == r.person(ada) and rated.seat1 == r.person(bo)

    def test_a_person_cannot_play_themself(self, store: Scratch) -> None:
        ada = store.user("ada")
        with pytest.raises(m.MatchError, match="two different people"):
            c.start(store.connection, SHORT_HASH, [ada, ada])

    def test_an_account_that_cannot_sign_in_is_not_seated(self, store: Scratch) -> None:
        ada, bo = store.user("ada"), store.user("bo")
        admin.kill_user(store.connection, bo)
        with pytest.raises(m.MatchError, match="can no longer play"):
            c.start(store.connection, SHORT_HASH, [ada, bo])

    def test_the_pause_refuses_new_games_and_leaves_running_ones_alone(
        self, store: Scratch
    ) -> None:
        found, ada, bo = _pair(store)
        c.set_paused(store.connection, True, by=ada)
        assert c.paused(store.connection)
        with pytest.raises(m.MatchError, match="paused"):
            c.start(store.connection, SHORT_HASH, [ada, bo])
        # A game in flight carries on, and finishes.
        _move(store, found.id, ada, bo)
        ended = _move(store, found.id, ada, bo)
        assert ended["outcome"] is not None
        c.set_paused(store.connection, False)
        assert c.start(store.connection, SHORT_HASH, [ada, bo]).running

    def test_games_at_once_are_capped_per_person(
        self, store: Scratch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(quota, "CORRESPONDENCE_AT_ONCE", 1)
        _found, ada, bo = _pair(store)
        cy = store.user("cy")
        with pytest.raises(m.MatchError, match="ada already has 1 games"):
            c.start(store.connection, SHORT_HASH, [cy, ada])

    def test_a_rated_game_against_the_computer_resigns_no_game_between_people(
        self, store: Scratch
    ) -> None:
        """§4's *a second rated game resigns the first* is about a person and a computer;
        two friends' games of the same rules run side by side (§1.3)."""
        between, ada, bo = _pair(store)
        second = c.start(store.connection, SHORT_HASH, [bo, ada])
        computer = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], ada)
        again = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], ada)
        assert m.row(store.connection, between.id).running
        assert m.row(store.connection, second.id).running
        assert not m.row(store.connection, computer.id).running, "the computer rule still holds"
        assert again.running


# ---------------------------------------------------------------------------- playing


class TestPlaying:
    def test_only_the_player_the_engine_names_may_move(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        waiting = bo if _mover(store, found.id, ada, bo) == ada else ada
        with pytest.raises(m.MatchError, match="not your move"):
            c.move(store.connection, found.id, waiting, _first_move(store, found.id))
        cy = store.user("cy")
        with pytest.raises(m.MatchError, match="only the two players"):
            c.move(store.connection, found.id, cy, _first_move(store, found.id))
        with pytest.raises(m.MatchError, match="only the two players"):
            c.move(store.connection, found.id, None, _first_move(store, found.id))

    def test_a_move_that_names_nobody_is_refused(self, store: Scratch) -> None:
        """`matches.play` takes no person, so it cannot be how a person's move arrives."""
        found, _ada, _bo = _pair(store)
        with pytest.raises(m.MatchError, match="between two people"):
            m.play(store.connection, found.id, _first_move(store, found.id))

    def test_the_engine_refuses_an_illegal_move(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        legal = {int(move["move"]) for move in _snapshot(store, found.id)["moves"]}
        illegal = next(n for n in range(10_000) if n not in legal)
        with pytest.raises(ValueError, match="not legal"):
            c.move(store.connection, found.id, _mover(store, found.id, ada, bo), illegal)

    def test_each_move_keeps_when_it_landed(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        _move(store, found.id, ada, bo)
        times = c.times_of(store.connection, found.id)
        assert len(times) == 1 and times[0]

    def test_practice_has_no_take_back_and_no_restart(self, store: Scratch) -> None:
        found, ada, bo = _pair(store, UNRATED_HASH)
        _move(store, found.id, ada, bo)
        with pytest.raises(m.MatchError, match="sent moves are final"):
            m.undo(store.connection, found.id)
        with pytest.raises(m.MatchError, match="sent moves are final"):
            m.restart(store.connection, found.id)


# ---------------------------------------------------------------------------- endings


def _rating(store: Scratch, user: int) -> r.Player | None:
    return r.load(store.connection, SHORT_HASH).get(r.person(user))


class TestTheEndings:
    def test_the_rules_end_it_and_both_ratings_move(self, store: Scratch) -> None:
        """The engine's own ending: `max_plies = 2`, one move each."""
        found, ada, bo = _pair(store)
        _move(store, found.id, ada, bo)
        ended = _move(store, found.id, ada, bo)
        assert ended["outcome"] is not None and ended["outcome"]["reason"]
        row = m.row(store.connection, found.id)
        assert not row.running and row.result is not None and not row.left
        assert _rating(store, ada) is not None and _rating(store, ada).matches == 1
        assert _rating(store, bo) is not None and _rating(store, bo).matches == 1
        with pytest.raises(m.MatchError, match="over"):
            c.move(store.connection, found.id, ada, 0)

    def test_a_resignation_is_a_finish_and_not_a_game_left(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        c.resign(store.connection, found.id, bo)
        row = m.row(store.connection, found.id)
        assert row.result == 1.0 and row.reason == "resigned" and not row.left
        assert _rating(store, bo).matches == 1

    def test_a_draw_by_agreement(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        offer = c.offer_draw(store.connection, found.id, ada)
        assert offer == c.Offer(seat=0, ply=0)
        with pytest.raises(m.MatchError, match="no draw offer from the other player"):
            c.accept_draw(store.connection, found.id, ada)
        c.accept_draw(store.connection, found.id, bo)
        row = m.row(store.connection, found.id)
        assert row.result == 0.5 and row.reason == m.AGREED and not row.left
        assert _snapshot(store, found.id)["outcome"] == {"winner": None, "reason": "agreed"}
        assert _rating(store, ada).matches == 1

    def test_an_offer_lapses_when_the_other_player_moves(self, store: Scratch) -> None:
        found, ada, bo = _pair(store, UNRATED_HASH)
        first = _mover(store, found.id, ada, bo)
        second = bo if first == ada else ada
        c.offer_draw(store.connection, found.id, second)
        _move(store, found.id, ada, bo)  # the offerer's opponent plays on
        assert c.offer_of(store.connection, found.id) is None
        c.offer_draw(store.connection, found.id, first)
        _move(store, found.id, ada, bo)  # now the opponent of `first` moves: lapses too
        assert c.offer_of(store.connection, found.id) is None

    def test_the_offerer_moving_keeps_the_offer(self, store: Scratch) -> None:
        found, ada, bo = _pair(store, UNRATED_HASH)
        first = _mover(store, found.id, ada, bo)
        c.offer_draw(store.connection, found.id, first)
        _move(store, found.id, ada, bo)
        assert c.offer_of(store.connection, found.id) is not None

    def test_offering_into_the_other_players_offer_accepts_it(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        c.offer_draw(store.connection, found.id, ada)
        assert c.offer_draw(store.connection, found.id, bo) is None
        assert m.row(store.connection, found.id).reason == m.AGREED

    def test_the_player_to_move_loses_after_five_days(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        _move(store, found.id, ada, bo)
        absent = _mover(store, found.id, ada, bo)
        store.backdate(found.id, timedelta(days=4))
        assert m.sweep(store.connection) == ()
        store.backdate(found.id, timedelta(days=2))
        swept = m.sweep(store.connection)
        assert [row.id for row in swept] == [found.id]
        row = swept[0]
        assert row.left and row.reason == "resigned"
        loser = row.seat_of(r.person(absent))
        assert row.score_for(loser) == 0.0, "the player to move lost, not the one waiting"
        assert _rating(store, ada).matches == 1 and _rating(store, bo).matches == 1

    def test_practice_is_swept_too_and_rated_nowhere(self, store: Scratch) -> None:
        found, ada, bo = _pair(store, UNRATED_HASH)
        store.backdate(found.id, timedelta(days=6))
        assert [row.id for row in m.sweep(store.connection)] == [found.id]
        assert r.load(store.connection, UNRATED_HASH) == {}

    def test_a_game_against_the_computer_keeps_its_thirty_days(self, store: Scratch) -> None:
        ada = store.user("ada")
        computer = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], ada)
        store.backdate(computer.id, timedelta(days=6))
        assert m.sweep(store.connection) == ()

    def test_the_words_hold_the_limit(self) -> None:
        assert timedelta(days=5) == quota.CORRESPONDENCE_IDLE
        assert quota.CORRESPONDENCE_IDLE_SAID == "five days without a move"
        assert quota.CORRESPONDENCE_IDLE_SAID in c.RULES[-1]


class TestCancelling:
    def test_either_may_cancel_until_both_have_moved(self, store: Scratch) -> None:
        found, ada, bo = _pair(store, UNRATED_HASH)
        _move(store, found.id, ada, bo)
        c.cancel(store.connection, found.id, bo)
        row = m.row(store.connection, found.id)
        assert not row.running and row.result is None and row.reason == m.CANCELLED
        assert _snapshot(store, found.id)["outcome"]["reason"] == m.CANCELLED

        later = c.start(store.connection, UNRATED_HASH, [ada, bo])
        _move(store, later.id, ada, bo)
        _move(store, later.id, ada, bo)
        with pytest.raises(m.MatchError, match="can no longer be cancelled"):
            c.cancel(store.connection, later.id, ada)

    def test_no_reading_counts_a_cancelled_game(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        c.cancel(store.connection, found.id, ada)
        assert m.leaderboard_matches(store.connection, SHORT_HASH, running=True) == ()
        assert _rating(store, ada) is None
        assert c.finished_line(store.connection, ada) is None
        with store.connection.cursor() as cursor:
            cursor.execute("select count(*) from finishes where match = %s", (found.id,))
            assert cursor.fetchone()[0] == 0


class TestWhatCountsAsPlayed:
    def test_an_agreed_draw_makes_no_finisher_and_a_rules_ending_does(
        self, store: Scratch
    ) -> None:
        """An agreed draw can be agreed before a move, so it is no finish on the board for
        the shortlist's bar; the rules' own ending is."""
        from chess_variant_sandbox.site import shortlist

        drawn, ada, bo = _pair(store, UNRATED_HASH)
        c.offer_draw(store.connection, drawn.id, ada)
        c.accept_draw(store.connection, drawn.id, bo)
        assert shortlist.finishers(store.connection, UNRATED_HASH) == set()
        played = c.start(store.connection, SHORT_HASH, [ada, bo])
        _move(store, played.id, ada, bo)
        _move(store, played.id, ada, bo)
        assert shortlist.finishers(store.connection, SHORT_HASH) == {ada, bo}

    def test_a_cancelled_game_is_nobodys_try(self, store: Scratch) -> None:
        from chess_variant_sandbox.site import designs

        found, ada, bo = _pair(store, UNRATED_HASH)
        c.cancel(store.connection, found.id, ada)
        assert designs.tries(store.connection, [UNRATED_HASH]) == ()


class TestLeaving:
    def test_revoking_resigns_running_games_as_games_left(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        admin.kill_user(store.connection, bo)
        row = m.row(store.connection, found.id)
        assert row.left and row.score_for(1) == 0.0

    def test_leaving_the_site_does_the_same(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        admin.forget(store.connection, "ada")
        row = m.row(store.connection, found.id)
        assert row.left and row.score_for(0) == 0.0

    def test_the_line_says_what_was_finished(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        assert c.finished_line(store.connection, ada) is None, "never 0 of 0"
        c.resign(store.connection, found.id, ada)
        assert c.finished_line(store.connection, ada) == (
            "Finished the one correspondence game played."
        )
        second = c.start(store.connection, SHORT_HASH, [ada, bo])
        store.backdate(second.id, timedelta(days=6))
        m.sweep(store.connection)
        dropped = m.row(store.connection, second.id)
        quitter = ada if dropped.score_for(0) == 0.0 else bo
        stayer = bo if quitter == ada else ada
        assert c.finished_line(store.connection, quitter) == (
            "Finished 1 of 2 correspondence games played."
        )
        assert c.finished_line(store.connection, stayer) == (
            "Finished all 2 correspondence games played."
        )


# ---------------------------------------------------------------------- through the site


@pytest.fixture(scope="module")
def app() -> SiteApp:
    return SiteApp(library.RULESET_DIR, library.RUNS_DIR)


def _as(app: SiteApp, viewer: int | None) -> SiteApp:
    app.viewer = viewer
    return app


class TestThroughTheSite:
    def test_two_accounts_play_it_to_the_end(self, store: Scratch, app: SiteApp) -> None:
        found, ada, bo = _pair(store)
        app.connect = store.open
        try:
            seen = _as(app, ada).get(f"/api/session/{found.id}")
            block = seen["correspondence"]
            assert block["you"] == 0 and block["players"] == ["ada", "bo"]
            assert block["rated"] is True and block["mayCancel"] is True
            for _ in range(2):
                snapshot = _as(app, ada).get(f"/api/session/{found.id}")
                who = (ada, bo)[snapshot["sideToMove"]]
                answer = _as(app, who).post(
                    f"/api/session/{found.id}/move", {"move": snapshot["moves"][0]["move"]}
                )
            assert answer["outcome"] is not None
            assert len(answer["correspondence"]["times"]) == 2
        finally:
            app.viewer = None
        assert not m.row(store.connection, found.id).running

    def test_draw_resign_and_cancel_through_the_route(
        self, store: Scratch, app: SiteApp
    ) -> None:
        drawn, ada, bo = _pair(store)
        resigned = c.start(store.connection, SHORT_HASH, [bo, ada])
        cancelled = c.start(store.connection, UNRATED_HASH, [ada, bo])
        store.connection.commit()
        app.connect = store.open
        try:
            offered = _as(app, ada).post(f"/api/session/{drawn.id}/draw", {})
            assert offered["correspondence"]["drawOffer"] == 0
            ended = _as(app, bo).post(f"/api/session/{drawn.id}/accept", {})
            assert ended["outcome"] == {"winner": None, "reason": "agreed"}
            gone = _as(app, ada).post(f"/api/session/{resigned.id}/resign", {})
            assert gone["outcome"]["reason"] == "resigned"
            off = _as(app, bo).post(f"/api/session/{cancelled.id}/cancel", {})
            assert off["outcome"]["reason"] == "cancelled"
            with pytest.raises(m.MatchError):
                _as(app, ada).post(f"/api/session/{cancelled.id}/undo", {})
        finally:
            app.viewer = None
        assert m.row(store.connection, resigned.id).score_for(1) == 0.0

    def test_a_running_game_is_its_two_players_alone(
        self, store: Scratch, app: SiteApp
    ) -> None:
        found, ada, bo = _pair(store)
        cy = store.user("cy")
        app.connect = store.open
        try:
            for outsider in (cy, None):
                with pytest.raises(KeyError, match="no game at this address"):
                    _as(app, outsider).get(f"/api/session/{found.id}")
                with pytest.raises(KeyError, match="no game at this address"):
                    _as(app, outsider).post(f"/api/session/{found.id}/resign", {})
            c.resign(store.connection, found.id, ada)
            store.connection.commit()
            # Finished, it is readable by any member, and still by no visitor.
            read = _as(app, cy).get(f"/api/session/{found.id}")
            assert read["correspondence"]["you"] is None
            with pytest.raises(KeyError):
                _as(app, None).get(f"/api/session/{found.id}")
        finally:
            app.viewer = None

    def test_the_profile_lists_the_games_and_whose_move(
        self, store: Scratch, app: SiteApp
    ) -> None:
        found, ada, bo = _pair(store)
        app.connect = store.open
        try:
            page = _as(app, ada).get("/api/profile")
        finally:
            app.viewer = None
        between = page["between"]
        assert [entry["id"] for entry in between["running"]] == [found.id]
        entry = between["running"][0]
        assert entry["opponent"] == "bo"
        assert entry["yourMove"] == (_mover(store, found.id, ada, bo) == ada)
        assert between["finished"] is None
        assert found.id not in [entry["id"] for entry in page["carryOn"]]

    def test_admin_pairs_two_members_and_pauses(self, store: Scratch, app: SiteApp) -> None:
        boss = store.user("boss")
        store.user("ada")
        store.user("bo")
        admin.set_role(store.connection, "boss")
        store.connection.commit()
        app.connect = store.open
        try:
            _as(app, boss)
            app._request.role = admin.ADMIN
            made = app._admin_act(
                {"act": "pair", "hash": SHORT_HASH, "seat0": "ada", "seat1": "bo"}
            )
            app._admin_act({"act": "pause-correspondence"})
            view = app._admin_view()["correspondence"]
            assert view == {"paused": True, "running": 1, "pausedSays": c.PAUSED_SAID,
                        "mailed": 0, "mailCap": quota.MOVE_EMAILS_A_DAY}
            with pytest.raises(m.MatchError, match="paused"):
                app._admin_act(
                    {"act": "pair", "hash": SHORT_HASH, "seat0": "bo", "seat1": "ada"}
                )
            app._admin_act({"act": "resume-correspondence"})
            assert app._admin_view()["correspondence"]["paused"] is False
        finally:
            app.viewer = None
        assert c.between_people(m.row(store.connection, made["id"]))


# ---------------------------------------------------------------------- rule 5, rule 3


class TestTheRules:
    def test_nothing_under_metrics_reads_the_human_store(self) -> None:
        """Rule 5, extended to the store (§1.9): a result between people enters a game's
        rating table and never its score. The scorer cannot reach the store at all."""
        for path in sorted((PACKAGE / "metrics").rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            for name in ("chess_variant_sandbox.site", "correspondence", "draw_offer",
                         "from matches", "web.human"):
                assert name not in source, f"{path.name} names {name}"

    def test_peoples_counts_flow_from_the_site_to_trust_only(self) -> None:
        """Rule 5 again, for `site/people.py` (plan-confidence T3): people's counts go
        site -> trust, never the reverse. `trust.py` takes them as an argument and imports
        nothing of the site, and nothing under `metrics/` names either module."""
        trusted = (PACKAGE / "trust.py").read_text(encoding="utf-8")
        assert "chess_variant_sandbox.site" not in trusted
        assert "from .site" not in trusted and "import site" not in trusted
        for path in sorted((PACKAGE / "metrics").rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            for name in ("site.people", "site import people", "import trust",
                         "trust import"):
                assert name not in source, f"{path.name} names {name}"
        ours = (PACKAGE / "site" / "people.py").read_text(encoding="utf-8")
        assert "from chess_variant_sandbox import trust" in ours

    def test_the_module_works_out_no_position(self) -> None:
        """Rule 3: whose move it is and every legal move are the engine's. The module
        reads `side_to_move` off a replayed state and calls `Session.play`; it never
        builds a move list or reads a board."""
        source = (PACKAGE / "site" / "correspondence.py").read_text(encoding="utf-8")
        assert "legal_moves" not in source and ".board" not in source
        assert "terminal(" not in source


# ----------------------------------------------------------------- the page, no clock


def _between_source() -> str:
    """The match page's code for games between two people (`board.js`'s last section)."""
    board = (PACKAGE / "web" / "static" / "page" / "board.js").read_text(encoding="utf-8")
    start = board.index("// ------------------------------------------------------------- between")
    # The code alone: the comments name what the page must never do, in those words.
    return "\n".join(
        line for line in board[start:].splitlines() if not line.lstrip().startswith("//")
    )


class TestNoCountdown:
    """Tests 2 and 3 of `docs/plan-meta-game.md` on the match page (§1.5, §1.8).

    The time left to move is shown since Q627 (the user's dated exception to test 2), as
    the server's sentence: the page itself still works none out
    (`tests/test_site_time_to_move.py`)."""

    def test_no_time_left_is_ever_worked_out(self) -> None:
        source = _between_source()
        for word in ("days left", "time left", "remaining", "deadline", "expires",
                     "countdown", "hurry", "IDLE"):
            assert word not in source, word
        # The only arithmetic on a time is how long ago something happened.
        assert re.findall(r"Date\.now\(\) - Date\.parse", source)
        assert "Date.parse(stamp) - Date.now()" not in source

    def test_no_red_dot_badge_or_title_count(self) -> None:
        source = _between_source()
        for mark in ("--bad", "badge", "document.title", "Notification", "Audio"):
            assert mark not in source, mark
        for word in ("red", "dot"):
            assert not re.search(rf"\b{word}\b", source), word

    def test_it_asks_only_while_visible_and_only_on_the_other_players_move(self) -> None:
        source = _between_source()
        assert "CORR_EVERY = 15000" in source
        assert "document.visibilityState !== 'visible'" in source
        assert "!corrTheirTurn()" in source


_GAME = """
SESSION = { id: 'g1', ruleset: { sides: ['white', 'black'] }, seats: ['human', 'human'],
  sideToMove: 1, inCheck: false, remaining: 1, outcome: null,
  history: [{ player: 0, text: 'a1-a2' }], moves: [{ move: 7, text: 'b8-b7' }],
  correspondence: { you: 1, players: ['ada', 'bo'], drawOffer: 0, mayCancel: true,
    times: [new Date(Date.now() - 3 * 3600e3).toISOString()],
    rules: ['The player to move loses after five days without a move.'] } };
"""


class TestTheMatchPageDraws:
    """The match page's correspondence parts, run in node over a hand-made snapshot."""

    def test_your_move_says_what_happened_and_never_what_is_left(self) -> None:
        from test_web import _in_page

        said = _in_page(_GAME, "corrStatus()")
        assert "Your move." in said
        assert "ada moved a1-a2, 3 hours ago." in said
        assert "ada offers a draw." in said
        assert "left" not in said and "day" not in said

    def test_a_picked_move_waits_for_send(self) -> None:
        from test_web import _in_page

        held = _in_page(_GAME + "CORR_HELD = 7;", "corrStatus()")
        assert "<b>b8-b7</b> is not sent yet." in held
        # Q740: it says the move can still be changed, the one place a phone says so.
        assert "You can change it until you send it." in held
        # One button sends it, the slot's under the board (Q733), never one in the strip.
        assert 'id="sendHeld"' not in held

    def test_the_other_players_move_and_the_names(self) -> None:
        from test_web import _in_page

        theirs = _in_page(_GAME + "SESSION.correspondence.you = 0;", "corrStatus()")
        assert theirs.startswith("bo's move.") and "You have offered a draw." in theirs
        assert _in_page(_GAME, "seatLabel(1)") == "bo (you)"
        assert _in_page(_GAME, "seatLabel(0)") == "ada"
        assert _in_page(_GAME, "verdictText({ winner: null, reason: 'agreed' })") == (
            "Drawn by agreement."
        )
        assert "counts for nothing" in _in_page(
            _GAME, "verdictText({ winner: null, reason: 'cancelled' })"
        )

    def test_a_seat_with_a_mark_links_to_its_members_page(self) -> None:
        """R1 (Q596): a person's seat names them with their mark and links to their
        page; a seat with no mark (somebody who left) is plain text."""
        from test_web import _in_page

        marked = _GAME + (
            "SESSION.correspondence.marks = [{ d: 'M0 0h1v1h-1z', hue: 1 }, null];"
        )
        linked = _in_page(marked, "seatHtml(0)")
        assert '<a href="/players/ada">' in linked and "ada</a>" in linked
        assert "/players/" not in _in_page(marked, "seatHtml(1)")

    def test_the_bar_offers_a_draw_and_a_cancel_and_no_take_back(self) -> None:
        from test_web import _in_page

        bar = _in_page(_GAME, "frameHtml(false, 'sandbox', false, SESSION.correspondence)")
        for control in ('id="draw"', 'id="cancel"', 'id="resign"', 'id="flip"'):
            assert control in bar, control
        assert 'id="undo"' not in bar
        reader = _in_page(
            _GAME, "frameHtml(false, 'leaderboard', false, { you: null, mayCancel: false })"
        )
        assert 'id="resign"' not in reader and 'id="draw"' not in reader
        # The slot that sends a move is a player's, under the board (Q733).
        assert 'id="corrSend"' in bar and 'id="corrSend"' not in reader
        assert bar.index('id="board"') < bar.index('id="corrSend"') < bar.index('data-frame="controls"')


# A slot the page can write into, and the browser's keeping as a plain map, for the parts
# of Q733 that draw into the frame and keep a move across a visit.
_SLOT = """
const SLOT = { hidden: false, innerHTML: '', held: false,
  classList: { toggle: (name, on) => { if (name === 'held') SLOT.held = !!on; } } };
document.getElementById = id => (id === 'corrSend' ? SLOT : null);
const KEPT = {};
window.localStorage = { getItem: k => (k in KEPT ? KEPT[k] : null),
  setItem: (k, v) => { KEPT[k] = String(v); }, removeItem: k => { delete KEPT[k]; } };
drawBoard = () => {};
"""


class TestTheHeldMove:
    """Q733: a move picked between two people and not sent is unmistakably unfinished --
    on the board, in a slot under it with the frame's one big button, and still there when
    the person comes back -- and the list of games waiting on you says so."""

    def test_the_slot_offers_send_only_while_a_move_is_held(self) -> None:
        from test_web import _in_page

        held = _in_page(_GAME + _SLOT + "CORR_HELD = 7; drawCorrSend();", "SLOT")
        assert held["held"] and not held["hidden"]
        assert 'id="sendHeld"' in held["innerHTML"] and "Send your move" in held["innerHTML"]
        # Q740: a plain button changes it, not a link, and the words say the board does too.
        assert '<button id="changeHeld" class="send-change">Change my move</button>' in held[
            "innerHTML"]
        assert 'id="putBack"' not in held["innerHTML"]
        said = " ".join(held["innerHTML"].split())
        assert "<b>Not sent yet.</b> To change it, tap another square or press" in said
        assert "Change my move</b>." in said

        mine = _in_page(_GAME + _SLOT + "drawCorrSend();", "SLOT")
        assert not mine["held"] and 'id="sendHeld"' not in mine["innerHTML"]
        assert "<b>Your move.</b>" in mine["innerHTML"]
        assert "then press <b>Send your move</b>." in mine["innerHTML"]
        assert "Nothing goes to ada before that." in mine["innerHTML"]

        sent = _in_page(
            _GAME + _SLOT + "SESSION.correspondence.you = 0; drawCorrSend();", "SLOT"
        )
        assert "<b>Your move is sent.</b> It is bo's turn now." in sent["innerHTML"]
        assert 'id="sendHeld"' not in sent["innerHTML"]

        for after in ("SESSION.outcome = { winner: 0, reason: 'resigned' };",
                      "SESSION.correspondence.you = null;"):
            gone = _in_page(_GAME + _SLOT + after + "drawCorrSend();", "SLOT")
            assert gone["hidden"] and gone["innerHTML"] == "", after

    def test_a_held_move_is_kept_until_sent_or_put_back(self) -> None:
        from test_web import _in_page

        kept = _in_page(_GAME + _SLOT + "corrHold(7);", "KEPT")
        assert kept == {
            "chess_variant_sandbox.held.g1": '{"ply":1,"move":7,"text":"b8-b7"}'
        }
        # Back on the page: the same game at the same move gives it back.
        assert _in_page(_GAME + _SLOT + "corrHold(7); CORR_HELD = null;",
                        "corrHeldKept(SESSION)") == 7
        assert _in_page(_GAME + _SLOT + "corrHold(7);", "heldWaiting('g1', 1)") is True
        assert _in_page(_GAME + _SLOT + "corrHold(7);", "heldWaiting('g1', 2)") is False
        # Put back, it is forgotten.
        assert _in_page(_GAME + _SLOT + "corrHold(7); corrPutBack();", "KEPT") == {}
        # A game that moved on, a move the engine no longer offers under those words, the
        # other player's turn or a finished game: forgotten, never given back.
        for changed in (
            "SESSION.history.push({ player: 1, text: 'b8-b7' });",
            "SESSION.moves[0].text = 'b8-b6';",
            "SESSION.sideToMove = 0;",
            "SESSION.outcome = { winner: 0, reason: 'resigned' };",
        ):
            gone = _in_page(_GAME + _SLOT + "corrHold(7);" + changed,
                            "[corrHeldKept(SESSION), KEPT]")
            assert gone == [None, {}], changed

    def test_leaving_with_a_move_held_asks_first(self) -> None:
        from test_web import _in_page

        setup = _GAME + _SLOT + """
        const HEARD = {};
        window.addEventListener = (name, listen) => { HEARD[name] = listen; };
        wireHeldLeaving();
        const asked = () => { const e = { prevented: false, returnValue: null,
          preventDefault() { this.prevented = true; } }; HEARD.beforeunload(e); return e.prevented; };
        """
        assert _in_page(setup, "asked()") is False
        assert _in_page(setup + "CORR_HELD = 7;", "asked()") is True

    def test_the_board_draws_it_dashed_with_the_piece_where_it_would_land(self) -> None:
        """The held move's squares are outlined in dashes, an arrow runs between them and
        the piece stands faded on its landing square; the last move keeps its solid tint."""
        from test_web import _in_page

        board = """
        SESSION.ruleset = Object.assign(SESSION.ruleset, { width: 2, height: 2, nTypes: 1,
          pieceIds: ['P'], icons: [] });
        SESSION.board = [0, 0, 2, 0];
        SESSION.moves = [{ move: 7, from: 2, to: 0, text: 'b8-b7' }];
        FRAME = boardView(2, 2, wrapAxes(null), { files: 0, ranks: 0 }, false);
        CORR_HELD = 7;
        """
        drawn = _in_page(_GAME + board, "corrHeldDrawn()")
        assert drawn["from"] == 2 and drawn["dest"] == 0
        assert drawn["piece"] == {"player": 1, "type": 0, "facing": 0}
        html = _in_page(_GAME + board, "heldHtml(corrHeldDrawn(), 50, 2, 2, null)")
        assert 'data-held="arrow"' in html and "stroke-dasharray" in html
        assert 'data-held="piece" opacity="0.7"' in html and "var(--held)" in html
        outline = _in_page(_GAME + board, "heldOutline(0, 0, 50, true)")
        assert 'data-held="to"' in outline and "stroke-dasharray" in outline
        assert "var(--held-halo)" in outline
        # A promotion stands as the piece it becomes; nothing held, nothing drawn.
        promoted = _in_page(_GAME + board + "SESSION.moves[0].promotion = 3;",
                            "corrHeldDrawn().piece.type")
        assert promoted == 3
        assert _in_page(_GAME + board + "CORR_HELD = null;", "corrHeldDrawn()") is None

    def test_the_waiting_list_says_a_move_was_never_sent(self) -> None:
        from test_web import _in_page

        asked = """
        markHtml = () => '';
        const ASKED = { waiting: [{ id: 'g1', name: 'chess', opponent: 'ada', plies: 1,
          rated: false, moved: new Date().toISOString(), toMove: 'Five days to move.' }] };
        """
        plain = _in_page(_GAME + _SLOT + asked, "waitingHtml(ASKED)")
        assert "Not sent yet" not in plain and ">Your move</button>" in plain
        held = _in_page(_GAME + _SLOT + asked + "corrHold(7);", "waitingHtml(ASKED)")
        assert "<b>Not sent yet:</b>" in held and "on this device" in held
        assert ">Send your move</button>" in held


# Q740's board: three squares a row, two rows; two pieces of the side to move, on 3 and 4.
# The piece on 3 may go to 0 or 1, the one on 4 to 1; squares 2 and 5 are nobody's.
_REPICK = """
SESSION.ruleset = Object.assign(SESSION.ruleset, { width: 3, height: 2, nTypes: 1,
  pieceIds: ['P'], pieceNames: ['pawn'], icons: [] });
SESSION.board = [0, 0, 0, 2, 2, 0];
SESSION.moves = [{ move: 7, from: 3, to: 0, text: 'c1-a1' }, { move: 8, from: 3, to: 1, text: 'c1-b1' },
  { move: 9, from: 4, to: 1, text: 'a2-b1' }];
const ASKED = { waiting: [{ id: 'g1', name: 'chess', opponent: 'ada', plies: 1, rated: false,
  moved: new Date().toISOString(), toMove: 'Five days to move.' }] };
markHtml = () => '';
const state = () => [CORR_HELD, SELECTED, KEPT];
"""


class TestChangingTheHeldMove:
    """Q740: until it is sent, a held move is changed or taken back on the board, by a
    plain button beside *Send your move*, or by Escape; nothing reaches the server, and
    the browser's keeping (and so the list of games waiting on you) follows every change."""

    def _after(self, steps: str):
        from test_web import _in_page

        return _in_page(_GAME + _SLOT + _REPICK + "corrHold(7);" + steps, "state()")

    def test_another_square_of_the_held_piece_replaces_it_at_once(self) -> None:
        held, selected, kept = self._after("clickSquare(1);")
        assert held == 8 and selected is None
        assert kept == {"chess_variant_sandbox.held.g1": '{"ply":1,"move":8,"text":"c1-b1"}'}

    def test_a_tap_on_the_held_move_takes_it_back(self) -> None:
        # Its piece, the square it lands on (where its faded piece stands).
        for square in (3, 0):
            assert self._after(f"clickSquare({square});") == [None, None, {}], square

    def test_another_piece_is_picked_instead(self) -> None:
        assert self._after("clickSquare(4);") == [None, 4, {}]
        # ... and its square makes that piece's move the one held.
        held, _, kept = self._after("clickSquare(4); clickSquare(1);")
        assert held == 9 and '"move":9' in kept["chess_variant_sandbox.held.g1"]

    def test_a_tap_elsewhere_or_while_sending_leaves_it(self) -> None:
        assert self._after("clickSquare(5);")[0] == 7
        assert self._after("clickSquare(2);")[0] == 7
        assert self._after("CORR_SENDING = true; clickSquare(3); clickSquare(1);")[0] == 7

    def test_change_my_move_and_escape_take_it_back(self) -> None:
        from test_web import _in_page

        buttons = """
        const BUTTONS = { changeHeld: {}, sendHeld: {} };
        const HEARD = {};
        window.addEventListener = (name, listen) => { HEARD[name] = listen; };
        document.getElementById = id => (id === 'corrSend' ? SLOT : BUTTONS[id] || null);
        """
        setup = _GAME + _SLOT + _REPICK + buttons + "corrHold(7); drawCorrSend();"
        assert _in_page(setup + "BUTTONS.changeHeld.onclick();", "state()") == [None, None, {}]
        press = "HEARD.keydown({ key: '{key}', target: { tagName: '{tag}' }, preventDefault() {} });"
        escape = press.replace("{key}", "Escape").replace("{tag}", "BODY")
        assert _in_page(setup + escape, "state()") == [None, None, {}]
        typing = press.replace("{key}", "Escape").replace("{tag}", "INPUT")
        assert _in_page(setup + typing, "CORR_HELD") == 7
        other = press.replace("{key}", "a").replace("{tag}", "BODY")
        assert _in_page(setup + other, "CORR_HELD") == 7
        # Taken back, the list of games waiting on you no longer says it was never sent.
        listed = _in_page(setup + escape, "waitingHtml(ASKED)")
        assert "Not sent yet" not in listed and ">Your move</button>" in listed

    def test_nothing_about_a_pick_goes_to_the_server(self) -> None:
        source = _between_source()
        start = source.index("function corrRepick")
        repick = source[start:source.index("function wireCorrHeld")]
        assert "api(" not in repick
        forget = source[source.index("function corrForget"):source.index("function corrPutBack")]
        assert "api(" not in forget


# ----------------------------------------------------------- the wild pieces' next two


class TestTheWildPiecesNextTwo:
    """Q764 (the user, 2026-10-08): in a game between two people the board shows where each
    wild piece steps after the next move and after the one after, as it does against the
    computer, from the core (`engine.coming.coming_phases`), on whichever person's turn
    it is and on the position the store rebuilds."""

    def test_each_turn_carries_both_steps_from_the_core(self, store: Scratch) -> None:
        from test_coming import PATROL

        from chess_variant_sandbox.engine.coming import coming_phases
        from chess_variant_sandbox.spec import spec_hash

        digest = store.game(spec_hash(PATROL), PATROL)
        found, ada, bo = _pair(store, digest)
        for _ in range(4):
            session = m.session(store.connection, found.id)
            snapshot = session.snapshot()
            first, then = coming_phases(session.engine, session.state)
            assert snapshot["coming"] == [s.as_dict() for s in first]
            assert snapshot["comingThen"] == [s.as_dict() for s in then]
            assert snapshot["comingThen"], "the ferret walks after every move"
            for entry in snapshot["history"]:
                assert "comingThen" in entry
            _move(store, found.id, ada, bo)
