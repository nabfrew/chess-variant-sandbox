"""The launch set as one constant, and the front page's featured game (Q654, Q564).

`docs/plan-launch-set.md` §3 and §5.1, as the user answered them on 2026-10-03: the nine and
the reserves live in `web/launch.py` alone, pinned by name and hash; the front page features
the five a chess player can start without reading the rules, one a day by the calendar,
with the gentlest opponent and the next one up a tap away; *reckoning* is the first game of
the month.

The pure half needs nothing. `TestTheFrontPage` reads a running site and so needs the test
database (`tests/test_site.py` has the recipe); it skips without one.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
import site_words

from chess_variant_sandbox import refresher
from chess_variant_sandbox.web import launch, library, opponents

LADDER = [entry.as_dict() for entry in opponents.ladder()]


class _Game:
    def __init__(self, digest: str, name: str, ratings: Any = None) -> None:
        self.hash = digest
        self.name = name
        self.ratings = ratings


def _the_library() -> list[library.Game]:
    return library.discover(library.RULESET_DIR, library.RUNS_DIR, library.ARCHIVE_DIR)


class TestTheLaunchSet:
    def test_the_nine_in_the_plans_order(self) -> None:
        assert [entry.name for entry in launch.LAUNCH_SET] == [
            "dobutsu", "los alamos", "cylinder chess", "three-check", "crazyhouse",
            "quarry", "reckoning", "wolves", "hex glinski hexagon",
        ]
        assert [entry.name for entry in launch.RESERVES] == [
            "intransitive", "atomic", "one-way cylinder chess", "succession", "shogi",
            "xiangqi",
        ]

    def test_every_pinned_game_is_in_the_library_under_its_name(self) -> None:
        """Fails, naming the game, when a launch or reserve game's rules were edited (a new
        hash), when it was renamed, or when its file is gone: the set never follows a name
        to a different game by itself."""
        found = launch.resolve((*launch.LAUNCH_SET, *launch.RESERVES), _the_library())
        assert [game.hash for game in found] == [
            entry.hash for entry in (*launch.LAUNCH_SET, *launch.RESERVES)
        ]

    def test_a_game_that_disappears_fails_loudly_by_name(self) -> None:
        games = [_Game(entry.hash, entry.name) for entry in launch.LAUNCH_SET[1:]]
        games.append(_Game(launch.LAUNCH_SET[2].hash + "x", "elsewhere"))
        renamed = [_Game(launch.LAUNCH_SET[0].hash, "lion catch"), *games]
        with pytest.raises(launch.LaunchSetMissing, match="dobutsu"):
            launch.resolve(launch.LAUNCH_SET, games)
        with pytest.raises(launch.LaunchSetMissing, match="called 'lion catch' now"):
            launch.resolve(launch.LAUNCH_SET, renamed)

    def test_the_refresher_ranks_the_same_nine(self) -> None:
        assert tuple(entry.name for entry in launch.LAUNCH_SET) == refresher.LAUNCH_SET

    def test_the_featured_five_and_the_first_month_are_launch_games(self) -> None:
        names = [entry.name for entry in launch.LAUNCH_SET]
        # The user's answer 6 A, in the order it gave them.
        assert launch.FEATURED == (
            "cylinder chess", "three-check", "quarry", "los alamos", "crazyhouse",
        )
        assert set(launch.FEATURED) <= set(names)
        assert launch.FIRST_MONTH_GAME == "reckoning"
        assert launch.FIRST_MONTH_GAME in names

    def test_every_launch_game_says_how_it_differs_in_a_visitors_words(self) -> None:
        for entry in launch.LAUNCH_SET:
            assert entry.sentence.endswith("."), entry.name
            assert site_words.hits(entry.sentence) == [], entry.name


class TestTheRotation:
    def test_it_starts_at_cylinder_chess(self) -> None:
        assert launch.featured_name(launch.ROTATION_START) == "cylinder chess"

    def test_one_a_day_in_order_and_round_again(self) -> None:
        days = [launch.ROTATION_START + timedelta(days=n) for n in range(11)]
        shown = [launch.featured_name(day) for day in days]
        assert shown[:5] == list(launch.FEATURED)
        assert shown[5:10] == list(launch.FEATURED)
        assert shown[10] == "cylinder chess"

    def test_a_day_before_the_start_is_still_in_the_cycle(self) -> None:
        before = launch.ROTATION_START - timedelta(days=1)
        assert launch.featured_name(before) == "crazyhouse"

    def test_the_same_day_is_the_same_game(self) -> None:
        day = date(2027, 2, 14)
        assert launch.featured_name(day) == launch.featured_name(date(2027, 2, 14))

    def test_a_withdrawn_game_passes_its_day_to_the_next(self) -> None:
        offered = [_Game(entry.hash, entry.name) for entry in launch.LAUNCH_SET
                   if entry.name != "cylinder chess"]
        step = launch.first_step(offered, LADDER, launch.ROTATION_START)
        assert step is not None and step["name"] == "three-check"
        assert launch.first_step([], LADDER, launch.ROTATION_START) is None

    def test_the_launch_cards_keep_the_order_and_drop_the_withdrawn(self) -> None:
        offered = [_Game(entry.hash, entry.name) for entry in reversed(launch.LAUNCH_SET)
                   if entry.name != "wolves"]
        names = [card["name"] for card in launch.launch_cards(offered)]
        assert names == [entry.name for entry in launch.LAUNCH_SET if entry.name != "wolves"]


class TestTheFirstOpponents:
    def test_without_a_rating_table_the_ladders_order(self) -> None:
        chosen = launch.opponents(LADDER, None)
        assert chosen is not None
        assert chosen["gentle"]["agent"] == "greedy"
        assert chosen["harder"]["agent"] == "minimax:nodes=400"
        assert chosen["basis"] == "ladder"

    def test_with_one_the_tables_order(self) -> None:
        """K3's example on *intransitive*: tree search below easy, greedy above both."""
        table = {"standings": [
            {"name": "random", "rating": -900.0},
            {"name": "mcts:nodes=1600", "rating": -345.0},
            {"name": "minimax:nodes=400", "rating": -12.0},
            {"name": "greedy", "rating": 25.0},
            {"name": "minimax:nodes=200000", "rating": 900.0},
        ]}
        chosen = launch.opponents(LADDER, table)
        assert chosen is not None
        assert chosen["gentle"] == {"agent": "mcts:nodes=1600", "name": "tree search"}
        assert chosen["harder"] == {"agent": "minimax:nodes=400", "name": "easy"}
        assert chosen["basis"] == "ratings"

    def test_never_random_a_person_or_the_capped_hardest(self) -> None:
        only = [entry for entry in LADDER
                if entry["agent"] in ("human", "random", "minimax:nodes=200000")]
        assert launch.opponents(only, None) is None


class TestTheFirstStepOnThePage:
    def test_the_page_draws_what_the_server_chose(self) -> None:
        from page_text import PAGE

        assert "function firstStepHtml(step)" in PAGE
        assert "Play the computer" in PAGE
        assert "A harder one: " in PAGE
        # The page picks no game and no opponent: it reads both off `firstStep`.
        assert "step.opponents.gentle" in PAGE
        assert "featured_name" not in PAGE


class TestTheFrontDoor:
    """Q559 (L1, `docs/plan-launch.md` §3.1): the site's `/` opens with the first step, the
    three lines of `docs/plan-understanding.md` §1.2, the launch set, then *All games*."""

    LINES = (
        "People invent chess-like games here, and anyone can play them.",
        "The computer plays each one hundreds of times and scores how interesting it is: a "
        "best estimate, which improves as people play.",
        "Play this one now",
        "Pick one and play it now",
        'build your own</a>.',
    )

    @staticmethod
    def _flat(text: str) -> str:
        return " ".join(text.split())

    def test_the_three_lines_word_for_word(self) -> None:
        from page_text import PAGE

        flat = self._flat(PAGE)
        for line in self.LINES:
            assert line in flat, line
        # *thousands* is true of no game (§1.2's note): the lines say *hundreds*.
        assert "thousands of times" not in flat

    def test_the_lines_pass_the_words_guard(self) -> None:
        for line in self.LINES:
            assert site_words.hits(line) == [], line

    def test_the_launch_set_then_all_games_behind_a_door(self) -> None:
        from page_text import PAGE

        assert "function frontDoorHtml(data)" in PAGE
        assert "Games to start with" in PAGE
        assert "All games (${LIBRARY.length})" in PAGE
        # Shut at first, and only where the server sent a launch set (the site): the local
        # application keeps its list open.
        assert "all: false," in PAGE
        assert "const door = Array.isArray(data.launchSet);" in PAGE
        # The page names no game of its own: the cards are the server's.
        assert "launchSetHtml(data.launchSet)" in PAGE
        for entry in launch.LAUNCH_SET:
            assert f"'{entry.name}'" not in PAGE


class TestThisMonthsGame:
    """Q603 (C2, `docs/plan-return.md` §3; `docs/plan-launch-set.md` §5.2): one game a month,
    *reckoning* first, then the rotation as data, by the calendar month."""

    def test_reckoning_first_then_quarry(self) -> None:
        assert launch.MONTHLY[0] == launch.FIRST_MONTH_GAME == "reckoning"
        assert launch.month_name(date(2026, 10, 1)) == "reckoning"
        assert launch.month_name(date(2026, 10, 31)) == "reckoning"
        assert launch.month_name(date(2026, 11, 1)) == "quarry"
        assert launch.month_name(date(2027, 1, 15)) == "hex glinski hexagon"

    def test_each_of_the_nine_takes_a_month_in_turn(self) -> None:
        assert sorted(launch.MONTHLY) == sorted(entry.name for entry in launch.LAUNCH_SET)
        months = [date(2026 + (9 + k) // 12, (9 + k) % 12 + 1, 1) for k in range(18)]
        named = [launch.month_name(day) for day in months]
        assert named[:9] == list(launch.MONTHLY)
        assert named[9:] == list(launch.MONTHLY)

    def test_a_withdrawn_game_passes_its_month_on(self) -> None:
        october = date(2026, 10, 4)
        everything = [_Game(entry.hash, entry.name) for entry in launch.LAUNCH_SET]
        month = launch.month_game(everything, october)
        assert month is not None
        assert month["name"] == "reckoning"
        assert month["month"] == "October 2026"
        assert month["sentence"] == launch.by_name("reckoning").sentence
        without = [game for game in everything if game.name != "reckoning"]
        passed = launch.month_game(without, october)
        assert passed is not None and passed["name"] == "quarry"
        assert launch.month_game([], october) is None

    def test_its_words_say_why_and_how_and_pass_the_guard(self) -> None:
        month = launch.month_game([_Game(e.hash, e.name) for e in launch.LAUNCH_SET],
                                  date(2026, 10, 4))
        assert month is not None
        assert "correspondence" in month["why"]
        assert "first of each month" in month["how"]
        for where, words in site_words.strings(month):
            assert site_words.hits(words) == [], (where, words)
        # A calendar month, never time left (test 2), and nothing earned (test 6).
        for words in (month["why"], month["how"]):
            assert not re.search(r"\b(left|ends in|days? to go|reward|earn)", words, re.I)

    def test_the_page_draws_what_the_server_chose(self) -> None:
        from page_text import PAGE

        assert "function monthGameHtml(month)" in PAGE
        assert "monthGameHtml(data.monthGame)" in PAGE
        assert "This month's game" in PAGE
        assert "month_name" not in PAGE


# ------------------------------------------------------------------ the running site


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """A running site over the five featured games and one game outside the set."""
    from test_site_admin import SITE_URL, _running_site

    from chess_variant_sandbox.site import mail
    from chess_variant_sandbox.spec import load_spec, spec_hash

    monkeypatch.setenv(mail.URL_VAR, SITE_URL)
    monkeypatch.delenv(mail.KEY_VAR, raising=False)
    for name in ("rulesets", "runs", "archive"):
        (tmp_path / name).mkdir()
    files = [library.RULESET_DIR / f"{name}.json"
             for name in ("cylinder", "threecheck", "losalamos", "crazyhouse", "dull")]
    files.append(library.ARCHIVE_DIR / "quarry.json")
    for path in files:
        shutil.copy(path, tmp_path / "rulesets")
    with _running_site(tmp_path) as running:
        for path in files:
            copied = tmp_path / "rulesets" / path.name
            running.game(spec_hash(load_spec(copied)), json.loads(copied.read_text("utf-8")))
        yield running


class TestTheFrontPage:
    def test_the_days_game_with_its_gentlest_opponent(
        self, served: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(launch, "today", lambda: launch.ROTATION_START + timedelta(days=2))
        payload = served.browser().get("/api/library").payload
        step = payload["firstStep"]
        assert step["name"] == "quarry"
        assert step["hash"] == launch.by_name("quarry").hash
        assert step["sentence"] == launch.by_name("quarry").sentence
        # No measurement in this library, so the ladder's own order.
        assert step["opponents"]["gentle"]["name"] == "greedy"
        assert step["opponents"]["harder"]["name"] == "easy"
        assert step["preview"]["width"] == 8 and step["preview"]["obstacles"]
        assert [card["name"] for card in payload["launchSet"]] == [
            "los alamos", "cylinder chess", "three-check", "crazyhouse", "quarry",
        ]
        for where, words in site_words.strings({"firstStep": step}):
            assert site_words.hits(words) == [], (where, words)

    def test_this_months_game_is_sent(
        self, served: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # *reckoning* is not in this library, so October passes to *quarry*, which is.
        monkeypatch.setattr(launch, "today", lambda: date(2026, 10, 4))
        month = served.browser().get("/api/library").payload["monthGame"]
        assert month["name"] == "quarry"
        assert month["hash"] == launch.by_name("quarry").hash
        assert month["month"] == "October 2026"

    def test_the_next_day_the_next_game(
        self, served: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(launch, "today", lambda: launch.ROTATION_START + timedelta(days=3))
        assert served.browser().get("/api/library").payload["firstStep"]["name"] == "los alamos"

    def test_a_visitor_can_start_it(self, served: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(launch, "today", lambda: launch.ROTATION_START)
        visitor = served.browser()
        step = visitor.get("/api/library").payload["firstStep"]
        assert step["name"] == "cylinder chess"
        for opponent in (step["opponents"]["gentle"], step["opponents"]["harder"]):
            started = visitor.post("/api/session", {
                "hash": step["hash"], "seats": ["human", opponent["agent"]],
                "mode": "sandbox",
            })
            assert started.status == 200, started.text
