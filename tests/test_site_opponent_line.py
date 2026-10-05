"""The opponent line (Q653) and the ladder row (Q602) on a game's page.

`docs/plan-launch-set.md` §2.4 and `docs/plan-return.md` §3 C3. What is held down here:

- **The three states**, from people's counts alone, at the thresholds pre-registered on
  2026-10-04 (`opponent_line.BEATEN_PEOPLE`, `HELD_GAMES`, `HELD_PEOPLE`), each boundary
  stated on both sides; a game nobody has played, and a game with no rated play, read
  *not enough games yet*. No word says a tester passed or won: the person test is scrapped.
- **The words guard**: no visitor's word holds a phrase of `site_words.PHRASES`.
- **It moves nothing**: the trust line read with the same counts is the same with or
  without the opponent line, and nothing under `metrics/` or `trust` imports it.
- **The ladder row**: every computer player on the table, weakest first, each with this
  person's best result (*won*, *drew*, *not yet*), from their finished rated games.
- **The routes and the page** (the route tests need a database, as `test_site_unbeaten`'s
  do): the standing carries both, the page draws the line under the computer players'
  table, apart from the trust line, and the row in the *Unbeaten* panel.

Results are stated, never derived: nothing here works out an outcome.
"""

from __future__ import annotations

import ast
import html as html_text
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import site_words
from test_site_unbeaten import (
    ALPHA,
    ALPHA_HASH,
    BETA,
    BETA_HASH,
    BOTTOM,
    CLEAR_LADDER,
    GAMMA,
    GAMMA_HASH,
    MIDDLE,
    TIED_LADDER,
    TOP,
    Scratch,
    _scratch,
)

from chess_variant_sandbox import trust
from chess_variant_sandbox.site import opponent_line as ol
from chess_variant_sandbox.site import unbeaten
from chess_variant_sandbox.site.server import SiteApp

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def store() -> Iterator[Scratch]:
    """`test_site_unbeaten`'s three games: a clear top, a tied top, and no ratings."""
    with _scratch() as scratch:
        scratch.game(ALPHA_HASH, ALPHA, CLEAR_LADDER)
        scratch.game(BETA_HASH, BETA, TIED_LADDER)
        scratch.game(GAMMA_HASH, GAMMA)
        yield scratch


@pytest.fixture
def app(store: Scratch, tmp_path: Path) -> SiteApp:
    """The site over the scratch store, as `test_site_unbeaten`'s `app` builds it."""
    rulesets = tmp_path / "rulesets"
    rulesets.mkdir()
    for spec in (ALPHA, BETA, GAMMA):
        (rulesets / f"{spec['name']}.json").write_text(json.dumps(spec), encoding="utf-8")
    for name in ("runs", "archive"):
        (tmp_path / name).mkdir()
    return SiteApp(rulesets, tmp_path / "runs", tmp_path / "heuristics",
                   tmp_path / "archive", store.open)


def _people(games: int, people: int, cleared: int, latest: str | None = None) -> trust.People:
    return trust.People(games=games, people=people, cleared=cleared, latest=latest)


def _words(line: dict[str, Any]) -> list[str]:
    return [line["state"], line["said"], *line["more"]]


# ------------------------------------------------------------------ the three states


class TestTheStates:
    def test_the_thresholds_are_the_pre_registered_ones(self) -> None:
        """§2.4, fixed 2026-10-04; 20 by 5 are the trust line's own numbers."""
        assert (ol.BEATEN_PEOPLE, ol.HELD_GAMES, ol.HELD_PEOPLE) == (1, 20, 5)
        assert (
            trust.RULE.people_games, trust.RULE.people_people) == (ol.HELD_GAMES, ol.HELD_PEOPLE)

    def test_nobody_has_played_it(self) -> None:
        line = ol.shown(_people(0, 0, 0))
        assert line["state"] == ol.EARLY
        assert "No rated games against the strongest computer player yet." in line["more"]
        assert ol.shown(None) == line

    def test_a_game_with_no_rated_play(self) -> None:
        line = ol.shown(_people(0, 0, 0), rated=False)
        assert line["state"] == ol.EARLY
        assert "no ratings here yet" in line["more"][0]

    def test_one_persons_win_makes_it_beaten(self) -> None:
        line = ol.shown(_people(1, 1, 1, "2026-10-03"))
        assert line["state"] == ol.BEATEN
        assert line["said"].startswith("People have beaten the strongest computer player")
        assert "1 person has beaten it." in line["more"]
        assert "1 rated game against the strongest computer player, by 1 person, the "\
            "latest in October 2026." in line["more"]

    def test_a_win_outranks_any_number_of_games(self) -> None:
        assert ol.state(_people(500, 80, 1)) == ol.BEATEN

    def test_held_needs_both_the_games_and_the_people(self) -> None:
        assert ol.state(_people(20, 5, 0)) == ol.HELD
        assert ol.state(_people(19, 5, 0)) == ol.EARLY
        assert ol.state(_people(20, 4, 0)) == ol.EARLY
        assert ol.state(_people(200, 4, 0)) == ol.EARLY
        line = ol.shown(_people(25, 6, 0, "2026-09-11"))
        assert line["state"] == ol.HELD
        assert "Nobody has beaten it yet." in line["more"]
        assert "25 rated games against the strongest computer player, by 6 people, the "\
            "latest in September 2026." in line["more"]

    def test_the_rule_is_said_under_more(self) -> None:
        for people in (_people(0, 0, 0), _people(25, 6, 0), _people(3, 2, 2)):
            assert ol.shown(people)["more"][-1] == ol.RULE_SAID
        assert "20 rated games" in ol.RULE_SAID and "5 different people" in ol.RULE_SAID

    def test_every_state_is_reachable_and_says_its_own_sentence(self) -> None:
        said = {ol.shown(p)["state"]: ol.shown(p)["said"]
                for p in (_people(0, 0, 0), _people(20, 5, 0), _people(2, 1, 1))}
        assert set(said) == set(ol.STATES)
        assert len(set(said.values())) == 3


# ------------------------------------------------------------------ the words


class TestTheWords:
    PEOPLE = (_people(0, 0, 0), _people(1, 1, 0, "2026-10-01"), _people(1, 1, 1),
              _people(20, 5, 0, "2026-10-02"), _people(9, 4, 2, "x"), _people(40, 12, 3))

    def test_no_visitors_word_holds_a_guarded_phrase(self) -> None:
        said = [word for people in self.PEOPLE for rated in (True, False)
                for word in _words(ol.shown(people, rated=rated))]
        trouble = [(text, site_words.hits(text)) for text in said if site_words.hits(text)]
        assert not trouble, trouble[:5]

    def test_no_word_says_a_tester_passed_or_won(self) -> None:
        """The person test was scrapped on 2026-10-03 (plan §8)."""
        said = " ".join(word for people in self.PEOPLE
                        for word in _words(ol.shown(people))).lower()
        for gone in (r"\btest", r"\bpass", "unbeatable", "strong player played"):
            assert not re.search(gone, said), gone


# ------------------------------------------------------------------ it moves nothing


class TestItMovesNothing:
    def test_nothing_that_scores_or_trusts_imports_it(self) -> None:
        package = ROOT / "chess_variant_sandbox"
        readers = []
        for path in [package / "trust.py", *(package / "metrics").rglob("*.py")]:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = ([alias.name for alias in node.names]
                         if isinstance(node, (ast.Import, ast.ImportFrom)) else [])
                module = getattr(node, "module", None) or ""
                if "opponent_line" in module or "opponent_line" in names:
                    readers.append(path.name)
        assert not readers, readers

    def test_the_trust_reading_is_the_same_whatever_the_line_says(self) -> None:
        """The line reads the counts; the trust level reads them too, and is not told."""
        source = (ROOT / "chess_variant_sandbox" / "site" / "opponent_line.py").read_text(
            encoding="utf-8")
        tree = ast.parse(source)
        called = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name) and node.func.value.id == "trust"}
        assert called <= {"People", "month"}, called


# ------------------------------------------------------------------ the ladder row


def _row(game: str, user: int, agent: str, result: float, seat: int = 0) -> Any:
    from datetime import datetime, timezone

    from chess_variant_sandbox.site import matches as m
    from chess_variant_sandbox.site import rating

    seats = [rating.person(user), rating.computer(agent)]
    if seat:
        seats.reverse()
    return m.Row(
        id=f"{agent}-{result}-{seat}", game=game, mode=m.LEADERBOARD,
        seat0=seats[0], seat1=seats[1], seed=0, plies=10,
        result=result if seat == 0 else 1.0 - result, reason="checkmate", left=False,
        top=False, top_players=(), before=(None, None), source=None,
        started=datetime(2026, 9, 1, tzinfo=timezone.utc), moved=None,
        finished=datetime(2026, 9, 2, tzinfo=timezone.utc),
    )


class TestTheLadderRow:
    def test_weakest_first_and_marked_by_the_best_result(self) -> None:
        from chess_variant_sandbox.site import rating

        table = {
            rating.computer(agent): rating.Player(rating.computer(agent), value, 50.0, 0.06, 24)
            for agent, value in ((TOP, 1900.0), (MIDDLE, 1500.0), (BOTTOM, 1100.0))
        }
        rows = [_row("g", 7, BOTTOM, 0.0), _row("g", 7, BOTTOM, 1.0, seat=1),
                _row("g", 7, MIDDLE, 0.5), _row("g", 7, MIDDLE, 0.0),
                _row("g", 8, TOP, 1.0), _row("other", 7, TOP, 1.0)]
        assert unbeaten.ladder_from("g", 7, rows, table) == (
            (BOTTOM, unbeaten.LADDER_WON),
            (MIDDLE, unbeaten.LADDER_DREW),
            (TOP, unbeaten.LADDER_NOT_YET),
        )

    def test_no_table_no_row(self) -> None:
        assert unbeaten.ladder_from("g", 7, [_row("g", 7, TOP, 1.0)], {}) == ()


# ------------------------------------------------------------------ the routes


class TestTheRoutes:
    def test_a_game_nobody_has_played_says_not_enough_games_yet(
        self, app: SiteApp, store: Scratch
    ) -> None:
        standing = app.get(f"/api/game/{ALPHA_HASH}")["standing"]
        assert standing["opponentLine"]["state"] == ol.EARLY
        assert "yourLadder" not in standing, "a signed-out visitor got somebody's row"

    def test_a_game_with_no_ratings_says_not_enough_games_yet(
        self, app: SiteApp, store: Scratch
    ) -> None:
        app.viewer = store.user("ada")
        standing = app.get(f"/api/game/{GAMMA_HASH}")["standing"]
        assert standing["opponentLine"]["state"] == ol.EARLY
        assert standing["yourLadder"] == []

    def test_a_persons_win_over_the_strongest_makes_it_beaten_and_the_score_stays(
        self, app: SiteApp, store: Scratch
    ) -> None:
        before = app.get(f"/api/game/{ALPHA_HASH}")
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, TOP, result=1.0)
        after = app.get(f"/api/game/{ALPHA_HASH}")
        assert after["standing"]["opponentLine"]["state"] == ol.BEATEN
        assert after.get("score") == before.get("score")
        assert after.get("trust") == before.get("trust")

    def test_held_once_twenty_games_by_five_people_have_not_beaten_it(
        self, app: SiteApp, store: Scratch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chess_variant_sandbox.site import people

        monkeypatch.setattr(people, "counts", lambda connection, game: _people(20, 5, 0))
        line = app.get(f"/api/game/{ALPHA_HASH}")["standing"]["opponentLine"]
        assert line["state"] == ol.HELD

    def test_the_ladder_row_carries_this_persons_marks(
        self, app: SiteApp, store: Scratch
    ) -> None:
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, BOTTOM, result=1.0)
        store.play(ALPHA_HASH, ada, MIDDLE, result=0.5)
        store.play(ALPHA_HASH, ada, TOP, result=0.0)
        app.viewer = ada
        row = app.get(f"/api/game/{ALPHA_HASH}")["standing"]["yourLadder"]
        assert [(step["agent"], step["mark"]) for step in row] == [
            (BOTTOM, "won"), (MIDDLE, "drew"), (TOP, "not yet")]
        assert [step["name"] for step in row] == ["random", "easy", "medium"]
        for step in row:
            assert not site_words.hits(f"{step['name']}: {step['mark']}")


# ------------------------------------------------------------------ the page


LINE = {"state": ol.HELD, "said": "Nobody has beaten it here so far.",
        "more": ["25 rated games against the strongest computer player, by 6 people.",
                 ol.RULE_SAID]}
LADDER = [{"agent": "random", "name": "random", "mark": "won"},
          {"agent": "minimax:nodes=400", "name": "easy", "mark": "drew"},
          {"agent": "minimax:nodes=200000", "name": "hardest", "mark": "not yet"}]


class TestThePage:
    def _drawn(self, **standing: Any) -> str:
        from test_web import _game_payload, _page_on

        game = _game_payload("dddddddddddd", site=True)
        game["standing"].update(standing)
        game["trust"] = {"level": "checked", "said": "A trust sentence.", "more": []}
        return str(_page_on(True, "#/game/dddddddddddd", game)["html"])

    def test_the_line_is_under_the_computer_players_and_apart_from_the_trust_line(
        self
    ) -> None:
        html = self._drawn(opponentLine=LINE)
        line = html.index('id="opponentLine"')
        assert html.index('id="strength"') < line < html.index("How it scored")
        assert "<strong>Held so far.</strong> Nobody has beaten it here so far." in html
        for said in LINE["more"]:
            assert html_text.escape(said).replace("&#x27;", "&#39;") in html
        between = html[line:html.index("How it scored")]
        assert 'class="trust"' not in between, "the trust line is the score's, drawn apart"

    def test_no_line_sent_none_drawn(self) -> None:
        assert 'id="opponentLine"' not in self._drawn()

    def test_the_ladder_row_is_drawn_weakest_first_in_the_unbeaten_panel(self) -> None:
        html = self._drawn(yourLadder=LADDER)
        row = html.index('class="ladderRow"')
        assert html.index("Unbeaten") < row < html.index('id="strength"')
        drawn = html[row:]
        assert drawn.index("random: won") < drawn.index("easy: drew") \
            < drawn.index("hardest: not yet")
        assert 'class="tag won"' in drawn and 'class="tag notYet"' in drawn

    def test_no_row_without_steps(self) -> None:
        assert 'class="ladderRow"' not in self._drawn(yourLadder=[])
