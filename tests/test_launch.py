"""The launch set as one constant, and the front page's featured game (Q654, Q564).

`docs/reference/opponent-line.md` §3 (planned in `docs/archive/plan-launch-set.md` §3 and §5.1), as the user answered them on 2026-10-03: the nine and
the reserves live in `web/launch.py` alone, pinned by name and hash; the front page features
the five a chess player can start without reading the rules, one a day by the calendar,
with the gentlest opponent and the next one up a tap away; *reckoning* is the first game of
the month.

The pure half needs nothing. `TestTheFrontPage` reads a running site and so needs the test
database (`tests/test_site.py` has the recipe); it skips without one.
"""

from __future__ import annotations

import hashlib
import html
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
            "quarry", "reckoning", "wolves", "glinski hexagonal chess",
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
    """Q564's first step, on the landing page's picks row since Q788."""

    def test_the_page_draws_what_the_server_chose(self) -> None:
        from page_text import PAGE

        from chess_variant_sandbox.web import landing

        assert "function wireTodaysGame(panel, step)" in PAGE
        assert "Play the computer" in landing.HTML
        assert "A harder one: " in landing.HTML
        assert html.escape(landing.TODAY_LABEL) in landing.HTML
        # The page picks no game and no opponent: it reads both off `firstStep`.
        assert "step.opponents.gentle" in PAGE
        assert "featured_name" not in PAGE


class TestTheGamesPage:
    """Q788 (the user, 2026-10-09): `/game` repeats none of the landing page's highlights and
    opens straight on the search and the filters; Q559's front door and its *All games* door
    are gone, on both servers."""

    def test_no_front_door_and_no_door_to_the_list(self) -> None:
        from page_text import PAGE

        for gone in ("function frontDoorHtml(", "function launchSetHtml(",
                     "function firstStepHtml(", "function aboutLinesHtml(",
                     "Games to start with", "All games (${LIBRARY.length})", "BROWSE.all",
                     'id="everything" hidden', "${shut ? ' hidden' : ''}"):
            assert gone not in PAGE, gone

    def test_the_library_view_opens_on_the_search_and_filters(self) -> None:
        from page_text import PAGE

        body = PAGE[PAGE.index("async function showLibrary()"):]
        body = body[:body.index("\n}\n")]
        assert "controlsHtml()" in body and "tabsHtml()" in body
        # No highlight of the landing page is drawn here.
        for highlight in ("firstStep", "launchSet", "monthGame", "newGames", "spotlight",
                          "madeHere"):
            assert highlight not in body, highlight
        # The page names no game of its own.
        for entry in launch.LAUNCH_SET:
            assert f"'{entry.name}'" not in PAGE


class _Dated:
    """A listed game with what `library.added` reads: where its file is and its spec."""

    def __init__(self, name: str, added: str | None, source: str = "archived",
                 description: str = "A game. More words.") -> None:
        self.hash = hashlib.sha256(name.encode()).hexdigest()[:12]
        self.name = name
        self.source = source
        self.description = description
        self.spec: dict[str, Any] = {"provenance": {} if added is None else {"added": added}}


class TestTheNewGamesAndTheSpotlight:
    """Q788's shelf and *Intransitive*; Q826: the shelf derived from the day each game joined."""

    DAY = date(2026, 10, 10)

    def test_a_game_is_new_for_the_window_after_the_day_it_joined(self) -> None:
        window = launch.NEW_WINDOW_DAYS
        games = [
            _Dated("today", self.DAY.isoformat()),
            _Dated("last day", (self.DAY - timedelta(days=window - 1)).isoformat()),
            _Dated("just past", (self.DAY - timedelta(days=window)).isoformat()),
            _Dated("tomorrow", (self.DAY + timedelta(days=1)).isoformat()),
            _Dated("undated", None),
        ]
        cards = launch.new_cards(games, self.DAY)
        assert [card["name"] for card in cards] == ["today", "last day"]
        assert all(card["fresh"] for card in cards)
        assert cards[0]["added"] == "10 October 2026" and cards[0]["day"] == "2026-10-10"
        assert cards[0]["sentence"] == "A game."

    def test_a_game_added_tomorrow_is_on_the_shelf_tomorrow_with_no_edit(self) -> None:
        tomorrow = self.DAY + timedelta(days=1)
        games = [_Dated("old", "2026-10-01"), _Dated("tomorrow", tomorrow.isoformat())]
        assert [card["name"] for card in launch.new_cards(games, self.DAY)] == ["old"]
        assert [card["name"] for card in launch.new_cards(games, tomorrow)] == [
            "tomorrow", "old"]

    def test_newest_day_first_then_by_name_and_at_most_the_shelf(self) -> None:
        games = [_Dated(f"game {letter}", "2026-10-09") for letter in "hgfedcba"]
        games += [_Dated("zed", "2026-10-10"), _Dated("Alpha", "2026-10-10")]
        cards = launch.new_cards(games, self.DAY)
        assert len(cards) == launch.NEW_SHOWN
        assert [card["name"] for card in cards][:4] == ["Alpha", "zed", "game a", "game b"]

    def test_when_nothing_is_new_the_most_recent_marked_not_fresh(self) -> None:
        games = [_Dated(f"g{n}", f"2026-08-{10 + n:02d}") for n in range(6)]
        cards = launch.new_cards(games, self.DAY)
        assert [card["name"] for card in cards] == ["g5", "g4", "g3", "g2"][
            :launch.NEW_FALLBACK]
        assert not any(card["fresh"] for card in cards)
        assert launch.new_cards([], self.DAY) == []

    def test_only_a_library_file_says_when_it_joined(self) -> None:
        """A member's draft or a design run's candidate may carry any `provenance.added`;
        only the library's two directories speak for theirs."""
        games = [_Dated("drafted", "2026-10-10", source="designed"),
                 _Dated("evolved", "2026-10-10", source="evolved"),
                 _Dated("kept", "2026-10-10", source="committed")]
        assert [card["name"] for card in launch.new_cards(games, self.DAY)] == ["kept"]

    def test_the_library_today_shows_what_joined_on_the_ninth_and_tenth(self) -> None:
        """The shelf over the real library on the day Q826 landed: eight of the games that
        joined on 9 and 10 October, the tenth's first, each in the library."""
        found = _the_library()
        cards = launch.new_cards(found, self.DAY)
        assert len(cards) == launch.NEW_SHOWN and all(card["fresh"] for card in cards)
        assert {card["day"] for card in cards} <= {"2026-10-10", "2026-10-09"}
        assert cards[0]["day"] == "2026-10-10"
        listed = {game.hash for game in found}
        assert all(card["hash"] in listed for card in cards)
        # On the ninth, the ninth's own games (fifteen joined, so the shelf is full of
        # them); three weeks later nothing is new and the shelf says so.
        ninth = launch.new_cards(found, date(2026, 10, 9))
        assert len(ninth) == launch.NEW_SHOWN
        assert {card["day"] for card in ninth} == {"2026-10-09"}
        later = launch.new_cards(found, date(2026, 11, 1))
        assert len(later) == launch.NEW_FALLBACK and not any(card["fresh"] for card in later)

    def test_the_shelfs_link_opens_every_game_newest_first(self) -> None:
        """A day can add more than the shelf holds; the link is the rest of them."""
        from page_text import PAGE

        from chess_variant_sandbox.web import landing

        assert f"data-newest>{html.escape(landing.NEW_ALL)}</a>" in landing.HTML
        body = PAGE[PAGE.index("function fillNewShelf(data)"):]
        body = body[:body.index("\n}\n")]
        assert "BROWSE.sort = 'added'" in body and "BROWSE.descending = true" in body
        assert "ORDER_SET = true" in body and "goPath(GAMES_PATH)" in body

    def test_their_words_are_a_visitors(self) -> None:
        for card in launch.new_cards(_the_library(), self.DAY):
            assert card["sentence"].endswith("."), card
            assert site_words.hits(card["sentence"]) == [], card
        for words in (launch.SPOTLIGHT.sentence, launch.SPOTLIGHT_WHY):
            assert words.endswith("."), words
            assert site_words.hits(words) == [], words

    def test_the_spotlight_is_in_the_library(self) -> None:
        found = launch.resolve((launch.SPOTLIGHT,), _the_library())
        assert [game.hash for game in found] == [launch.SPOTLIGHT.hash]
        assert launch.SPOTLIGHT.name == "intransitive"
        assert launch.SPOTLIGHT.hash == launch.by_name("intransitive").hash
        assert launch.spotlight_card([]) is None
        spot = launch.spotlight_card([_Game(launch.SPOTLIGHT.hash, "intransitive")])
        assert spot is not None and spot["why"] == launch.SPOTLIGHT_WHY

    def test_the_page_draws_no_number_of_its_own(self) -> None:
        """A new game is drawn with the library card's own pill, which reads *not
        measured* when nobody has measured it and marks a stale score provisional."""
        from page_text import PAGE

        from chess_variant_sandbox.web import landing

        body = PAGE[PAGE.index("function fillNewShelf(data)"):]
        body = body[:body.index("\n}\n")]
        assert "scoreTag(game)" in body
        assert "fmt(" not in body and ".index" not in body
        assert 'id="landing-new"' in landing.HTML
        assert 'id="landingSpotlight"' in landing.HTML
        assert landing.NEW_HEADING in landing.HTML

    def test_an_older_shelf_is_never_called_new(self) -> None:
        """With no card `fresh`, the heading and the lead take their `data-older` words."""
        from page_text import PAGE

        from chess_variant_sandbox.web import landing

        body = PAGE[PAGE.index("function fillNewShelf(data)"):]
        body = body[:body.index("\n}\n")]
        assert "card.fresh" in body and "dataset.older" in body
        assert f'data-older="{html.escape(landing.NEW_OLDER_HEADING)}"' in landing.HTML
        assert f'data-older="{html.escape(landing.NEW_OLDER_LEAD)}"' in landing.HTML
        assert "new" not in landing.NEW_OLDER_HEADING.lower()
        for words in (landing.NEW_LEAD, landing.NEW_OLDER_LEAD, landing.NEW_OLDER_HEADING):
            assert site_words.hits(words) == [], words


class TestThisMonthsGame:
    """Q603 (C2, `docs/archive/plan-return.md` §3; `docs/reference/opponent-line.md` §3.1): one game a month,
    *reckoning* first, then the rotation as data, by the calendar month."""

    def test_reckoning_first_then_quarry(self) -> None:
        assert launch.MONTHLY[0] == launch.FIRST_MONTH_GAME == "reckoning"
        assert launch.month_name(date(2026, 10, 1)) == "reckoning"
        assert launch.month_name(date(2026, 10, 31)) == "reckoning"
        assert launch.month_name(date(2026, 11, 1)) == "quarry"
        assert launch.month_name(date(2027, 1, 15)) == "glinski hexagonal chess"

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

        from chess_variant_sandbox.web import landing

        # On the landing page's picks row since Q788.
        assert "fill('landingMonth', data && data.monthGame" in PAGE
        assert html.escape(landing.MONTH_LABEL) in landing.HTML
        assert "month_name" not in PAGE


# ------------------------------------------------------------------ the running site


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """A running site over the five featured games, one game outside the set, one of the
    newest games and the spotlight."""
    from test_site_admin import SITE_URL, _running_site

    from chess_variant_sandbox.site import mail
    from chess_variant_sandbox.spec import load_spec, spec_hash

    monkeypatch.setenv(mail.URL_VAR, SITE_URL)
    monkeypatch.delenv(mail.KEY_VAR, raising=False)
    for name in ("rulesets", "runs", "archive"):
        (tmp_path / name).mkdir()
    files = [library.RULESET_DIR / f"{name}.json"
             for name in ("cylinder", "threecheck", "losalamos", "crazyhouse", "dull",
                          "intransitive")]
    files.append(library.ARCHIVE_DIR / "quarry.json")
    files.append(library.ARCHIVE_DIR / "rifle.json")
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

    def test_the_new_games_and_spotlight_are_sent_over_what_is_offered(
        self, served: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Of this library only *rifle* joined in the fortnight before 10 October, so it
        # alone is sent: never a game the listing does not offer, and never an old one.
        monkeypatch.setattr(launch, "today", lambda: date(2026, 10, 10))
        payload = served.browser().get("/api/library").payload
        assert [card["name"] for card in payload["newGames"]] == ["rifle"]
        assert payload["newGames"][0]["added"] == "9 October 2026"
        assert payload["newGames"][0]["fresh"] is True
        # Every card carries the day its game joined, which *date added* sorts on.
        days = {card["name"]: card["added"] for card in payload["games"]}
        assert days["rifle"] == "2026-10-09" and days["cylinder chess"] == "2026-08-28"
        assert payload["spotlight"]["name"] == "intransitive"
        assert payload["spotlight"]["hash"] == launch.SPOTLIGHT.hash
        for where, words in site_words.strings(
                {"newGames": payload["newGames"], "spotlight": payload["spotlight"]}):
            assert site_words.hits(words) == [], (where, words)

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
