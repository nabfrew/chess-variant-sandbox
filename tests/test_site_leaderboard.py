"""A game's leaderboard on its page (L2 of `docs/plan-leaderboard.md`, §2.1, §2.5, §3).

Q2 was decided on 2026-10-10: everyone who has finished a rated game on a game is on its
leaderboard, a number still settling marked `?`. `site/leaderboard.py` reads the game's one
rating table (`rating.load`) and `SiteApp._standing` sends it as `standing.leaderboard`;
the page draws the rows in the order they came (`detail.js::leaderboardHtml`).

Held here, on specs rather than chess (rule 5): who is a row and who is not (a computer
player by the picker's name, a person by handle once they have finished a rated game; never
a practice-only player, a person who has only practised, or a person row with no finished
game), the order and the marks (strongest first, `?` while settling, the strongest computer
players, a player no longer offered, the reader's own row), the counts, and each state of
§2.5 (rated with nobody yet, not rated yet on a library game and on a member's). The rows
are proved on tables built by hand first, then through the route over a real store, where a
finished rated match puts its person on the table. The page's half is drawn in node.

The store tests need `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL` and skip without it, as every
site test does.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
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
    _play_out,
    _scratch,
)
from test_web import _in_page

from chess_variant_sandbox.site import identity, rating
from chess_variant_sandbox.site import leaderboard as board
from chess_variant_sandbox.site import matches as m
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.web import library, opponents

NET = "minimax:net=aaaaaaaaaaa1,nodes=6400"
"""A learned player's agent string: practice only (Q683), never a row."""

ROSTER = [
    *opponents.ladder(),
    opponents.Opponent(agent=NET, label="the learned player", group="learned", store="learned",
                       name="the learned player", practice_only=True),
]
NAMES = {entry.agent: entry.name for entry in ROSTER}


def _computer(agent: str, score: float, games: int = 0) -> rating.Player:
    return rating.Player(rating.computer(agent), score, 100.0, 0.0, games)


def _person(user: int, score: float, deviation: float, games: int) -> rating.Player:
    return rating.Player(rating.person(user), score, deviation, 0.06, games)


def _table(*players: rating.Player) -> dict[rating.PlayerKey, rating.Player]:
    return {player.key: player for player in players}


TABLE = _table(
    _computer("random", 820.0),
    _computer("greedy", 1100.0, games=2),
    _computer("minimax:nodes=6400", 1946.0, games=5),
    _computer("minimax:nodes=200000", 2302.4, games=3),
    _computer("minimax:nodes=3200", 1500.0, games=1),  # offered no longer
    _computer(NET, 2600.0),  # practice only: never a row, never the strongest
    _person(1, 1712.6, 90.0, games=14),  # established
    _person(2, 1640.2, 210.0, games=2),  # still settling
    _person(3, 1500.0, 350.0, games=0),  # a seat, and no finished game
)
NAMED = {1: "ada", 2: "bo", 3: "cy"}


def _rows(viewer: int | None = None, table: Any = TABLE) -> list[dict[str, Any]]:
    return board.rows(table, named=NAMED, roster=ROSTER, viewer=viewer)


class TestTheRows:
    def test_strongest_first_with_people_and_computer_players_together(self) -> None:
        assert [row["name"] for row in _rows()] == [
            "hardest", "hard", "ada", "bo", "the 3,200-position player", "greedy", "random",
        ]
        assert [row["rating"] for row in _rows()] == [2302, 1946, 1713, 1640, 1500, 1100, 820]

    def test_a_practice_only_player_and_a_person_with_no_rated_game_are_absent(self) -> None:
        names = {row["name"] for row in _rows()}
        assert "the learned player" not in names and "cy" not in names
        assert NET not in json.dumps(_rows())
        # Absent from the strongest too: a practice-only player is nobody's target.
        assert [row["name"] for row in _rows() if row["top"]] == ["hardest"]

    def test_a_settling_number_is_marked_and_an_established_one_is_not(self) -> None:
        rows = {row["name"]: row for row in _rows()}
        assert rows["bo"]["settling"] is True and rows["ada"]["settling"] is False
        # A computer player is established from its seed, whatever its count.
        assert not any(row["settling"] for row in rows.values() if row["computer"])

    def test_the_count_is_rated_games_here(self) -> None:
        rows = {row["name"]: row for row in _rows()}
        assert rows["ada"]["games"] == 14 and rows["hard"]["games"] == 5
        assert rows["random"]["games"] == 0

    def test_a_player_no_longer_offered_keeps_its_row_named_not_spelled(self) -> None:
        gone = [row for row in _rows() if row["gone"]]
        assert [row["name"] for row in gone] == ["the 3,200-position player"]
        assert "nodes=" not in json.dumps(_rows())

    def test_the_readers_own_row_is_marked_and_only_theirs(self) -> None:
        assert [row["name"] for row in _rows(viewer=2) if row["you"]] == ["bo"]
        assert not any(row["you"] for row in _rows(viewer=3))
        assert not any(row["you"] for row in _rows())

    def test_a_tie_at_the_top_marks_each_of_the_tied(self) -> None:
        tied = _table(_computer("minimax:nodes=6400", 2000.0),
                      _computer("minimax:nodes=200000", 2003.0), _computer("random", 900.0))
        assert [row["name"] for row in _rows(table=tied) if row["top"]] == ["hardest", "hard"]

    def test_somebody_who_left_reads_as_a_former_player(self) -> None:
        table = _table(_computer("random", 900.0), _person(9, 1600.0, 200.0, games=1))
        assert [row["name"] for row in _rows(table=table)] == [identity.FORMER, "random"]


class TestTheStates:
    def test_rated_and_played(self) -> None:
        shown = board.shown(TABLE, named=NAMED, roster=ROSTER, viewer=None, library=True)
        assert shown["rated"] is True and shown["people"] == 2 and shown["waiting"] == []
        # Offered, rateable and with no row: named under the table, never given a number.
        assert shown["unrated"] == ["easy", "medium", "tree search"]

    def test_rated_and_nobody_yet(self) -> None:
        table = _table(_computer("random", 900.0), _computer("greedy", 1200.0))
        shown = board.shown(table, named={}, roster=ROSTER, viewer=7, library=True)
        assert shown["rated"] is True and shown["people"] == 0
        assert [row["name"] for row in shown["rows"]] == ["greedy", "random"]

    def test_not_rated_yet_names_the_players_weakest_first_and_no_numbers(self) -> None:
        shown = board.shown({}, named={}, roster=ROSTER, viewer=None, library=True)
        assert shown["rated"] is False and shown["rows"] == []
        assert shown["waiting"] == [
            "random", "greedy", "easy", "medium", "hard", "hardest", "tree search",
        ]
        assert "the learned player" not in shown["waiting"]
        assert shown["library"] is True


# ------------------------------------------------------------------------- the page


def _drawn(leaderboard: dict[str, Any], **more: Any) -> str:
    game = {"standing": {"leaderboard": leaderboard}, **more}
    return " ".join(str(_in_page(
        "ACCOUNTS = true; VIEWER = { handle: 'bo' };",
        f"leaderboardHtml({json.dumps(game)})")).split())


class TestThePage:
    def _board(self, viewer: int | None = None) -> dict[str, Any]:
        return board.shown(TABLE, named=NAMED, roster=ROSTER, viewer=viewer, library=True)

    def test_the_rows_are_drawn_in_the_order_sent_with_their_marks(self) -> None:
        html = _drawn(self._board(viewer=2))
        order = [html.index(name) for name in (">hardest ", ">hard<", ">ada<", ">bo<")]
        assert order == sorted(order)
        assert "(the one to beat)" in html and "(no longer offered)" in html
        assert '<tr class="you">' in html and "(you)" in html
        assert '<b>1640</b><span title="still settling">?</span>' in html
        assert "A number with a ? is still settling" in html
        assert "Nobody has played a rated game here yet" not in html

    def test_the_page_sorts_nothing(self) -> None:
        """Reversed rows come out reversed: the order is the server's."""
        sent = self._board()
        sent["rows"] = list(reversed(sent["rows"]))
        html = _drawn(sent)
        assert html.index(">random<") < html.index(">hardest ")

    def test_nobody_yet_says_where_a_first_game_puts_you(self) -> None:
        shown = board.shown(_table(_computer("random", 900.0)), named={}, roster=ROSTER,
                            viewer=None, library=True)
        html = _drawn(shown)
        assert ("Nobody has played a rated game here yet. Your first one puts you on this"
                " table, starting at 1500.") in html
        assert "Every computer player on this game, strongest first" in html

    def test_not_rated_yet_on_a_library_game_and_on_a_members(self) -> None:
        waiting = board.shown({}, named={}, roster=ROSTER, viewer=None, library=True)
        html = _drawn(waiting)
        assert ("This game's computer players have not been rated yet, so every game here"
                " is practice for now. They are rated on our own machine; this usually takes"
                " a day.") in html
        assert "<b>" not in html and 'data-gloss="ratedHere"' not in html
        assert html.index(">random<") < html.index(">greedy<") < html.index(">hardest<")
        member = _drawn({**waiting, "library": False})
        assert "practice for now." in member and "usually takes a day" not in member

    def test_the_local_application_keeps_its_own_panel(self) -> None:
        """No accounts, no leaderboard: `boardOf` is null there whatever is sent."""
        assert _in_page("ACCOUNTS = false;",
                        "boardOf({ standing: { leaderboard: { rated: true } } })") is None


# ------------------------------------------------------------------- over the store


@pytest.fixture
def store() -> Iterator[Scratch]:
    """One rated game (a loaded ladder), one not rated, and a library game not rated."""
    with _scratch() as scratch:
        scratch.game(SHORT_HASH, SHORT, LADDER)
        scratch.game(UNRATED_HASH, UNRATED)
        yield scratch


@pytest.fixture(scope="module")
def app() -> SiteApp:
    return SiteApp(library.RULESET_DIR, library.RUNS_DIR)


def _standing(app: SiteApp, store: Scratch, digest: str, viewer: int | None) -> Any:
    app.connect = store.open
    app.viewer = viewer
    try:
        return app._standing(digest)["standing"]["leaderboard"]
    finally:
        app.viewer = None


class TestThroughTheRoute:
    def test_a_rated_game_nobody_has_played_is_its_computer_players(
        self, store: Scratch, app: SiteApp
    ) -> None:
        shown = _standing(app, store, SHORT_HASH, None)
        assert shown["rated"] is True and shown["people"] == 0
        assert [row["name"] for row in shown["rows"]] == [
            NAMES[AGENT], "greedy", "random"]
        assert [row["top"] for row in shown["rows"]] == [True, False, False]
        assert shown["library"] is False  # a game the library on disk does not hold

    def test_a_finished_rated_game_puts_its_person_on_the_table(
        self, store: Scratch, app: SiteApp
    ) -> None:
        ada, bo = store.user("ada"), store.user("bo")
        played = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], ada)
        _play_out(store, played.id)
        # Bo only practised: a sandbox game moves no table and puts nobody on it.
        practised = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], bo)
        _play_out(store, practised.id)
        # Cy has a row and no finished game (a seat the table holds): not a row either.
        cy = store.user("cy")
        rating.store(store.connection, SHORT_HASH, [rating.newcomer(rating.person(cy))])
        store.connection.commit()

        shown = _standing(app, store, SHORT_HASH, ada)
        people = [row for row in shown["rows"] if not row["computer"]]
        assert [row["name"] for row in people] == ["ada"]
        assert people[0]["games"] == 1 and people[0]["settling"] is True
        assert people[0]["you"] is True and shown["people"] == 1
        opponent = next(row for row in shown["rows"] if row["name"] == NAMES[AGENT])
        assert opponent["games"] == 1
        # In the table's order, strongest first, whatever the result was.
        ratings = [row["rating"] for row in shown["rows"]]
        assert ratings == sorted(ratings, reverse=True)
        # Read by somebody else, nobody's row is theirs.
        assert not any(row["you"] for row in _standing(app, store, SHORT_HASH, bo)["rows"])

    def test_a_practice_only_row_in_the_table_is_never_shown(
        self, store: Scratch, app: SiteApp
    ) -> None:
        rating.store(store.connection, SHORT_HASH, [_computer(NET, 2900.0)])
        store.connection.commit()
        shown = _standing(app, store, SHORT_HASH, None)
        assert NET not in json.dumps(shown) and shown["rows"][0]["name"] == NAMES[AGENT]
        assert [row["name"] for row in shown["rows"] if row["top"]] == [NAMES[AGENT]]

    def test_a_game_not_rated_yet_names_its_players_and_says_whose_machine(
        self, store: Scratch, app: SiteApp
    ) -> None:
        shown = _standing(app, store, UNRATED_HASH, None)
        assert shown["rated"] is False and shown["rows"] == []
        assert shown["waiting"][:3] == ["random", "greedy", "easy"]
        assert shown["library"] is False
        spec = json.loads((library.RULESET_DIR / "dull.json").read_text(encoding="utf-8"))
        digest = store.game(spec_hash(spec), spec)
        shown = _standing(app, store, digest, None)
        assert shown["rated"] is False and shown["library"] is True
