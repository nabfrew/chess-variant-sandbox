"""The time left to move in a game between two people, as a plain fact (Q627).

`docs/archive/plan-return.md` §9, the user on 2026-10-02: *"there should be an indication how long
there is left to move ... It would be incredibly annoying for it to be a surprise lose on
time"*, and on the member's own page *"the started games sorted in order descending order
of time left to move, then time left opponent has to move in ascending order"*. The dated
exception to test 2 of `docs/reference/no-dark-patterns.md`: a plain statement, never a countdown.

What is proved, on specs rather than chess (rule 5): the words at the boundary times
against a clock the test sets; that they measure from the instant the five-day rule
measures from, so the sentence and the sweep agree at the limit; the profile's order; and
that the page draws the server's sentence as plain text, with nothing that ticks, flashes
or escalates. The store tests skip without `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL`, as
every site test does; the words, the order and the page checks need nothing.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from test_site_matches import LADDER, SHORT, SHORT_HASH, Scratch, _scratch

from chess_variant_sandbox.site import correspondence as c
from chess_variant_sandbox.site import matches as m
from chess_variant_sandbox.site import quota
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.web import library

PACKAGE = Path(__file__).resolve().parents[1] / "chess_variant_sandbox"

H = timedelta(hours=1)
D = timedelta(days=1)
MINUTE = timedelta(minutes=1)


# ---------------------------------------------------------------------------- the words


class TestTheWords:
    @pytest.mark.parametrize(
        ("left", "said"),
        [
            (5 * D, "5 days"),
            (4 * D + 6 * H, "4 days 6 hours"),
            (4 * D + 6 * H + 59 * MINUTE, "4 days 6 hours"),  # rounded down, never up
            (4 * D + 6 * H - MINUTE, "4 days 5 hours"),
            (D + H, "1 day 1 hour"),
            (D, "1 day"),
            (D - MINUTE, "23 hours"),
            (2 * H, "2 hours"),
            (H, "1 hour"),
            (H - MINUTE, "less than an hour"),
            (MINUTE, "less than an hour"),
        ],
    )
    def test_whole_hours_rounded_down(self, left: timedelta, said: str) -> None:
        assert c.duration_said(left) == said

    def test_no_finer_than_an_hour_so_nothing_ticks(self) -> None:
        """A page that asks every 15 seconds would tick if the words had minutes."""
        for left in (3 * D + 2 * H + 30 * MINUTE, 5 * H + 30 * MINUTE, 40 * MINUTE):
            said = {c.duration_said(left - k * 15 * timedelta(seconds=1)) for k in range(8)}
            assert len(said) == 1, said
            assert "minute" not in said.pop()

    def test_the_sentence_to_each_reader(self) -> None:
        left = 4 * D + 6 * H
        assert c.to_move_said(left) == "You have 4 days 6 hours to move."
        assert c.to_move_said(left, "They") == "They have 4 days 6 hours to move."
        assert c.to_move_said(left, "bo") == "bo has 4 days 6 hours to move."
        assert c.to_move_said(H - MINUTE, "bo") == "bo has less than an hour to move."

    def test_once_the_time_is_up_it_says_so_plainly(self) -> None:
        for left in (timedelta(0), -MINUTE, -D):
            said = c.to_move_said(left, "bo")
            assert said.startswith("bo has no time left to move:"), said
            assert "loss on time" in said

    def test_the_words_pass_the_plain_words_guard(self) -> None:
        from site_words import hits

        for left in (5 * D, 4 * D + 6 * H, D, H - MINUTE, timedelta(0)):
            for who in ("You", "They", "bo"):
                said = c.to_move_said(left, who)
                assert hits(said) == [], (said, hits(said))


# ---------------------------------------------------------------------------- the order


def _entry(ident: str, yours: bool, left: timedelta) -> dict[str, Any]:
    return {"id": ident, "yourMove": yours, "left": left.total_seconds()}


ENTRIES = [
    _entry("theirs-long", False, 4 * D),
    _entry("yours-short", True, 2 * H),
    _entry("theirs-short", False, 3 * H),
    _entry("yours-long", True, 4 * D),
    _entry("yours-mid", True, 2 * D),
]


class TestTheOrder:
    def test_your_move_first_most_urgent_first_then_theirs_least_first(self) -> None:
        """The user, 2026-10-02: most urgent first (*descending* was a typo)."""
        assert [e["id"] for e in c.started_order(ENTRIES)] == [
            "yours-short", "yours-mid", "yours-long", "theirs-short", "theirs-long",
        ]

    def test_a_tie_is_broken_the_same_way_every_time(self) -> None:
        tied = [_entry("b", True, D), _entry("a", True, D)]
        assert [e["id"] for e in c.started_order(tied)] == ["a", "b"]
        assert [e["id"] for e in c.started_order(tied[::-1])] == ["a", "b"]

    def test_the_page_keeps_the_servers_order(self) -> None:
        """The profile draws the list as it arrives; the order lives in one function."""
        source = _code(_function(PACKAGE / "site" / "static" / "profile.js", "drawBetween"))
        assert ".sort(" not in source
        assert "entry.toMove" in source


# ---------------------------------------------------------------------------- the store


@pytest.fixture
def store() -> Iterator[Scratch]:
    with _scratch() as scratch:
        scratch.game(SHORT_HASH, SHORT, LADDER)
        yield scratch


@pytest.fixture
def app() -> SiteApp:
    return SiteApp(library.RULESET_DIR, library.RUNS_DIR)


def _mover(store: Scratch, match_id: str) -> int:
    """The seat to move, read off the engine's side to move."""
    return int(m.session(store.connection, match_id).snapshot()["sideToMove"])


def _first_move(store: Scratch, found: m.Row, people: tuple[int, int]) -> None:
    """The player to move sends the engine's first offered move."""
    snapshot = m.session(store.connection, found.id).snapshot()
    who = people[int(snapshot["sideToMove"])]
    c.move(store.connection, found.id, who, int(snapshot["moves"][0]["move"]))
    store.connection.commit()


class TestAgainstTheClock:
    def test_the_match_page_says_it_to_both_seats(self, store: Scratch) -> None:
        ada, bo = store.user("ada"), store.user("bo")
        found = c.start(store.connection, SHORT_HASH, [ada, bo])
        _first_move(store, found, (ada, bo))
        row = m.row(store.connection, found.id)
        mover = _mover(store, found.id)
        waiting = (ada, bo)[1 - mover]
        to_move = (ada, bo)[mover]
        name = ("ada", "bo")[mover]
        since = row.moved or row.started
        at = since + D + 18 * H - MINUTE  # 3 days 6 hours and a minute left
        mine = c.view(store.connection, row, to_move, mover=mover, at=at)
        theirs = c.view(store.connection, row, waiting, mover=mover, at=at)
        assert mine["toMove"] == "You have 3 days 6 hours to move."
        assert theirs["toMove"] == f"{name} has 3 days 6 hours to move."
        late = c.view(store.connection, row, to_move, mover=mover, at=since + 5 * D)
        assert late["toMove"].startswith("You have no time left to move")
        assert c.view(store.connection, row, to_move)["toMove"] is None  # no mover named

    def test_it_measures_from_the_last_move_or_the_start(self, store: Scratch) -> None:
        ada, bo = store.user("ada"), store.user("bo")
        found = c.start(store.connection, SHORT_HASH, [ada, bo])
        assert c.time_left(found, found.started) == quota.CORRESPONDENCE_IDLE
        _first_move(store, found, (ada, bo))
        moved = m.row(store.connection, found.id)
        assert moved.moved is not None
        assert c.time_left(moved, moved.moved + 2 * D) == 3 * D

    def test_the_words_and_the_rule_agree_at_the_limit(self, store: Scratch) -> None:
        """The idle-loss rule is unchanged, and the sentence meets it where it bites:
        with half an hour left nothing is resigned, half an hour past it is."""
        ada, bo = store.user("ada"), store.user("bo")
        found = c.start(store.connection, SHORT_HASH, [ada, bo])
        store.connection.commit()
        store.backdate(found.id, quota.CORRESPONDENCE_IDLE - 30 * MINUTE)
        row = m.row(store.connection, found.id)
        said = c.to_move_said(c.time_left(row, c.now(store.connection)))
        assert said == "You have less than an hour to move."
        assert m.sweep(store.connection) == ()
        store.backdate(found.id, H)
        row = m.row(store.connection, found.id)
        assert c.to_move_said(c.time_left(row, c.now(store.connection))).startswith(
            "You have no time left"
        )
        assert [swept.id for swept in m.sweep(store.connection)] == [found.id]

    def test_a_finished_game_says_nothing_about_time(self, store: Scratch) -> None:
        ada, bo = store.user("ada"), store.user("bo")
        found = c.start(store.connection, SHORT_HASH, [ada, bo])
        c.resign(store.connection, found.id, ada)
        row = m.row(store.connection, found.id)
        assert c.view(store.connection, row, bo, mover=0, at=row.started)["toMove"] is None


class TestThroughTheSite:
    def test_the_session_read_carries_the_sentence(
        self, store: Scratch, app: SiteApp
    ) -> None:
        ada, bo = store.user("ada"), store.user("bo")
        found = c.start(store.connection, SHORT_HASH, [ada, bo])
        store.connection.commit()
        mover = _mover(store, found.id)
        app.connect = store.open
        try:
            for seat, viewer in enumerate((ada, bo)):
                app.viewer = viewer
                read = app.get(f"/api/session/{found.id}")
                said = read["correspondence"]["toMove"]
                if seat == mover:
                    assert said == "You have 4 days 23 hours to move."
                else:
                    assert said == f"{('ada', 'bo')[mover]} has 4 days 23 hours to move."
        finally:
            app.viewer = None

    def test_the_profile_orders_the_started_games(
        self, store: Scratch, app: SiteApp
    ) -> None:
        ada, bo = store.user("ada"), store.user("bo")
        people = (ada, bo)
        # Five games: the age of each, and whether a move is made in it first.
        plan = [(1 * D, False), (3 * D, False), (2 * D, False), (H, True), (3 * D, True)]
        ages: dict[str, timedelta] = {}
        for age, moved in plan:
            found = c.start(store.connection, SHORT_HASH, [ada, bo])
            store.connection.commit()
            if moved:
                _first_move(store, found, people)
            store.backdate(found.id, age)
            ages[found.id] = age
        app.connect = store.open
        try:
            app.viewer = ada
            running = app.get("/api/profile")["between"]["running"]
        finally:
            app.viewer = None
        yours = {e["id"]: e["yourMove"] for e in running}
        # The engine says whose move each is; ada's own come first, least time left first
        # (most urgent), then bo's, least time left first.
        assert yours == {
            ident: people[_mover(store, ident)] == ada for ident in ages
        }
        first = sorted((i for i in ages if yours[i]), key=lambda i: -ages[i])
        second = sorted((i for i in ages if not yours[i]), key=lambda i: -ages[i])
        assert [e["id"] for e in running] == first + second
        assert first and second, "the plan seats both kinds"
        for entry in running:
            who = "You" if entry["yourMove"] else "They"
            left = quota.CORRESPONDENCE_IDLE - ages[entry["id"]] - MINUTE
            assert entry["toMove"] == c.to_move_said(left, who)


# --------------------------------------------------------------------------- the page


def _function(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index(f"function {name}")
    return text[start : text.index("\n}\n", start)]


def _code(text: str) -> str:
    """Code alone: the comments name what a page must never do, in those words."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))


def _between_source() -> str:
    board = (PACKAGE / "web" / "static" / "page" / "board.js").read_text(encoding="utf-8")
    start = board.index("// ------------------------------------------------------------- between")
    return _code(board[start:])


# Animation, a flash, a colour that escalates: none of it near the time left to move.
ESCALATION = ("animation", "@keyframes", "blink", "pulse", "flash", "urgent", "warn",
              "alarm", "transition", "--bad", "red", "countdown", "tick")

_GAME = """
SESSION = { id: 'g1', ruleset: { sides: ['white', 'black'] }, seats: ['human', 'human'],
  sideToMove: 1, inCheck: false, remaining: 1, outcome: null,
  history: [{ player: 0, text: 'a1-a2' }], moves: [{ move: 7, text: 'b8-b7' }],
  correspondence: { you: 1, players: ['ada', 'bo'], drawOffer: null, mayCancel: true,
    times: [new Date(Date.now() - 3 * 3600e3).toISOString()],
    toMove: 'You have 4 days 6 hours to move.',
    rules: ['The player to move loses after five days without a move.'] } };
"""


class TestThePage:
    def test_the_strip_states_it_as_plain_text(self) -> None:
        from test_web import _in_page

        said = _in_page(_GAME, "corrStatus()")
        assert said == (
            "<b>Your move.</b> ada moved a1-a2, 3 hours ago. You have 4 days 6 hours to move."
        )

    def test_the_waiting_seat_and_a_reader_read_it_too(self) -> None:
        from test_web import _in_page

        theirs = _in_page(
            _GAME + "SESSION.correspondence.you = 0;"
            " SESSION.correspondence.toMove = 'bo has 2 days to move.';",
            "corrStatus()",
        )
        assert theirs == "bo's move. ada moved a1-a2, 3 hours ago. bo has 2 days to move."

    def test_the_markup_has_no_animation_or_escalation(self) -> None:
        """No element, class or style wraps the sentence: it is the strip's own text, and
        nothing in the page's code for these games animates, flashes or turns red."""
        from test_web import _in_page

        for left in ("You have 4 days 6 hours to move.",
                     "You have less than an hour to move.",
                     "You have no time left to move: the game is about to end as a loss on time."):
            said = _in_page(_GAME.replace("You have 4 days 6 hours to move.", left),
                            "corrStatus()")
            tail = said[said.index("3 hours ago.") :]
            assert "<" not in tail and "class" not in tail, tail
        source = _between_source()
        drawn = _code(_function(PACKAGE / "site" / "static" / "profile.js", "drawBetween"))
        for code in (source, drawn):
            for mark in ESCALATION:
                assert not re.search(rf"(?<![\w-]){re.escape(mark)}(?![\w-])", code, re.I), mark

    def test_the_page_works_out_no_time_left(self) -> None:
        """The words are the server's: the page does no sum towards a deadline."""
        source = _between_source()
        assert "Date.parse(stamp) - Date.now()" not in source
        assert "IDLE" not in source and "toMove -" not in source
        assert source.count("between.toMove") == 2
