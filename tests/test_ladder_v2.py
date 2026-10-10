"""The picker ladder since L3: its stamp, its kept games, the top-up and the move-aside.

`docs/plan-leaderboard.md` §2.2. A ladder is `web.opponents.Ratings`' run over a game's
picker. Since L3 it says what it rated, on what build and when, and keeps its games beside
it, so a newly offered player is added by playing its own games against the spine alone
and the fit is redone over the old games and the new. These play real, tiny ladders on
gardner (one game a pairing, three cheap players) -- well under a second each -- because
what is being held is what the run writes, and only a run writes it.

The site's side (the reseed, the anchor) is `tests/test_site_rating.py` and
`tests/test_site_load.py`; the command line is `tests/test_deploy.py`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.spec.canonical import load_spec, spec_hash
from chess_variant_sandbox.tournament.ratings import rate
from chess_variant_sandbox.tournament.runner import schedule
from chess_variant_sandbox.web import opponents

RULESETS = Path(__file__).resolve().parent.parent / "rulesets"
SPEC = load_spec(RULESETS / "gardner.json")
DIGEST = spec_hash(SPEC)

# A spine of three (the cheapest of `opponents.SPINE`) and one cheap challenger to add.
FIRST = ("random", "greedy", "minimax:nodes=400")
NEWCOMER = "minimax:nodes=100"


def _roster(agents: tuple[str, ...]) -> list[opponents.Opponent]:
    return [opponents.Opponent(agent=agent, label=agent, group="") for agent in agents]


def _run(runs: Path, agents: tuple[str, ...], top_up: bool = False) -> opponents.RatingRun:
    run = opponents.Ratings(runs).start(SPEC, DIGEST, "gardner", _roster(agents), games=1,
                                        top_up=top_up)
    deadline = time.monotonic() + 60
    while run.state == "running" and time.monotonic() < deadline:
        time.sleep(0.02)
    return run


def _stored(runs: Path) -> dict[str, Any]:
    return json.loads(opponents.stored_path(DIGEST, runs).read_text(encoding="utf-8"))


@pytest.fixture
def dated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(opponents, "_today", lambda: "2026-10-10")
    monkeypatch.setattr(opponents, "_built", lambda: "f" * 12)


class TestTheStamp:
    def test_a_ladder_says_what_it_rated_on_what_build_and_when(
        self, tmp_path: Path, dated: None
    ) -> None:
        run = _run(tmp_path, FIRST)
        assert run.state == "finished", run.error
        table = _stored(tmp_path)
        assert table["format"] == opponents.LADDER_FORMAT
        assert table["roster"] == list(FIRST)
        assert table["spine"] == list(FIRST)
        assert (table["rated"], table["built"]) == ("2026-10-10", "f" * 12)
        assert table["games_file"] == f"{DIGEST}.games.jsonl"
        assert table["top_ups"] == []

    def test_its_games_are_kept_beside_it_and_refit_to_the_same_table(
        self, tmp_path: Path, dated: None
    ) -> None:
        """The kept games are the games the fit read: refitting them gives the table."""
        run = _run(tmp_path, FIRST)
        kept = opponents.read_games(DIGEST, tmp_path)
        assert kept is not None and len(kept) == run.total == _stored(tmp_path)["games"]
        assert {record.agents for record in kept} == {
            (first, second) for first in FIRST for second in FIRST if first != second
        }
        refit = {row["name"]: row["rating"] for row in rate(kept).as_dict()["standings"]}
        assert refit == {row["name"]: row["rating"] for row in _stored(tmp_path)["standings"]}
        # Never under a name that globs as a measured corpus.
        assert not list(tmp_path.rglob("tournament.json"))
        assert not list(tmp_path.rglob("games.jsonl"))

    def test_an_unreadable_line_makes_the_kept_games_unreadable_not_shorter(
        self, tmp_path: Path, dated: None
    ) -> None:
        _run(tmp_path, FIRST)
        path = opponents.games_path(DIGEST, tmp_path)
        path.write_text(path.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
        assert opponents.read_games(DIGEST, tmp_path) is None


class TestTheTopUp:
    def test_a_newcomer_plays_the_spine_only_and_the_fit_is_redone_over_all(
        self, tmp_path: Path, dated: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = _run(tmp_path, FIRST)
        assert first.state == "finished", first.error
        old = opponents.read_games(DIGEST, tmp_path)
        assert old is not None

        played: list[Any] = []
        real = opponents.run_tournament

        def spy(spec: Any, config: Any, *args: Any, **kwargs: Any) -> Any:
            played.append(config)
            return real(spec, config, *args, **kwargs)

        monkeypatch.setattr(opponents, "run_tournament", spy)
        monkeypatch.setattr(opponents, "_today", lambda: "2026-10-11")
        run = _run(tmp_path, (*FIRST, NEWCOMER), top_up=True)
        assert run.state == "finished", run.error

        # Only the newcomer against each spine player, each seat, at the ladder's own
        # games per pairing; seeds past every kept one.
        (config,) = played
        games = schedule(config)
        assert run.total == len(games) == 2 * len(FIRST)
        assert all(NEWCOMER in (first, second) for _, first, second in games)
        assert min(seed for seed, _, _ in games) > max(record.seed for record in old)

        table = _stored(tmp_path)
        kept = opponents.read_games(DIGEST, tmp_path)
        assert kept is not None and len(kept) == len(old) + len(games) == table["games"]
        refit = {row["name"]: row["rating"] for row in rate(kept).as_dict()["standings"]}
        assert {row["name"]: row["rating"] for row in table["standings"]} == refit
        assert NEWCOMER in refit
        # The ladder's own stamp stands, its older games still being most of the fit; the
        # top-up says who joined, when and on what build.
        assert table["rated"] == "2026-10-10"
        assert table["roster"] == [*FIRST, NEWCOMER]
        assert table["top_ups"] == [{"players": [NEWCOMER], "games": len(games),
                                     "seed": config.seed, "rated": "2026-10-11",
                                     "built": "f" * 12}]

    def test_a_top_up_is_a_pure_function_of_the_ladder_it_extends(
        self, tmp_path: Path, dated: None
    ) -> None:
        """Seeds fixed before a thread starts: the same ladder topped up twice is one table."""
        tables = []
        for name in ("one", "two"):
            runs = tmp_path / name
            _run(runs, FIRST)
            assert _run(runs, (*FIRST, NEWCOMER), top_up=True).state == "finished"
            tables.append(_stored(runs))
        assert tables[0] == tables[1]

    def test_nothing_new_offered_is_nothing_played(self, tmp_path: Path, dated: None) -> None:
        _run(tmp_path, FIRST)
        before = opponents.stored_path(DIGEST, tmp_path).read_bytes()
        run = _run(tmp_path, FIRST, top_up=True)
        assert run.state == "finished" and run.total == 0
        assert opponents.stored_path(DIGEST, tmp_path).read_bytes() == before

    def test_a_player_no_longer_offered_keeps_its_row(self, tmp_path: Path, dated: None) -> None:
        """Its games are still in the fit, and people's games against it are history."""
        _run(tmp_path, (*FIRST, "minimax:nodes=50"))
        run = _run(tmp_path, (*FIRST, NEWCOMER), top_up=True)
        assert run.state == "finished", run.error
        names = {row["name"] for row in _stored(tmp_path)["standings"]}
        assert names == {*FIRST, "minimax:nodes=50", NEWCOMER}

    @pytest.mark.parametrize("damage", ["unstamped", "no games", "short games", "new spine"])
    def test_a_ladder_that_cannot_take_one_fails_at_once_and_writes_nothing(
        self, tmp_path: Path, dated: None, damage: str
    ) -> None:
        _run(tmp_path, FIRST[:2] if damage == "new spine" else FIRST)
        path = opponents.stored_path(DIGEST, tmp_path)
        kept = opponents.games_path(DIGEST, tmp_path)
        if damage == "unstamped":
            table = json.loads(path.read_text(encoding="utf-8"))
            del table["format"]
            path.write_text(json.dumps(table), encoding="utf-8")
        elif damage == "no games":
            kept.unlink()
        elif damage == "short games":
            kept.write_text(kept.read_text(encoding="utf-8").split("\n", 1)[1],
                            encoding="utf-8")
        before = path.read_bytes()
        run = _run(tmp_path, (*FIRST, NEWCOMER), top_up=True)
        assert run.state == "failed" and "--rerate" in run.error
        assert path.read_bytes() == before

    def test_a_top_up_with_no_ladder_is_refused(self, tmp_path: Path) -> None:
        run = _run(tmp_path, FIRST, top_up=True)
        assert run.state == "failed" and "no ladder" in run.error
        assert not opponents.stored_path(DIGEST, tmp_path).exists()


class TestTheMoveAside:
    def _ladder(self, runs: Path, text: str = "{}") -> None:
        path = opponents.stored_path(DIGEST, runs)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        opponents.games_path(DIGEST, runs).write_text("games\n", encoding="utf-8")

    def test_it_moves_both_files_and_never_writes_over_an_earlier_move(
        self, tmp_path: Path
    ) -> None:
        self._ladder(tmp_path, "first")
        aside = opponents.move_aside(DIGEST, tmp_path, "2026-10-10")
        assert aside == tmp_path / "opponents" / "before-2026-10-10"
        assert not opponents.stored_path(DIGEST, tmp_path).exists()
        assert (aside / f"{DIGEST}.json").read_text(encoding="utf-8") == "first"
        assert (aside / f"{DIGEST}.games.jsonl").exists()

        self._ladder(tmp_path, "second")
        again = opponents.move_aside(DIGEST, tmp_path, "2026-10-10")
        assert again == tmp_path / "opponents" / "before-2026-10-10.2"
        assert (aside / f"{DIGEST}.json").read_text(encoding="utf-8") == "first"
        assert (again / f"{DIGEST}.json").read_text(encoding="utf-8") == "second"

    def test_restore_puts_it_back_where_nothing_took_its_place(self, tmp_path: Path) -> None:
        self._ladder(tmp_path, "kept")
        aside = opponents.move_aside(DIGEST, tmp_path, "2026-10-10")
        assert aside is not None
        opponents.restore(DIGEST, tmp_path, aside)
        assert opponents.stored_path(DIGEST, tmp_path).read_text(encoding="utf-8") == "kept"
        assert opponents.games_path(DIGEST, tmp_path).exists()
        assert not aside.exists()

    def test_no_ladder_is_nothing_to_move(self, tmp_path: Path) -> None:
        assert opponents.move_aside(DIGEST, tmp_path, "2026-10-10") is None


class TestOldPlayers:
    def test_a_ladder_rated_before_a_marked_change_has_old_players(self) -> None:
        latest = opponents.PLAYER_CHANGES[-1][0]
        assert opponents.changed_since("2026-09-01")
        assert opponents.changed_since(latest) == ()
        assert opponents.changed_since(None) == opponents.PLAYER_CHANGES

    def test_the_stamp_is_read_before_the_files_date(self, tmp_path: Path) -> None:
        path = tmp_path / "ladder.json"
        path.write_text("{}", encoding="utf-8")
        assert opponents.ladder_date({"rated": "2026-09-01"}, path) == "2026-09-01"
        assert opponents.ladder_date({}, path) is not None
        assert opponents.ladder_date({}, tmp_path / "absent.json") is None

    def test_the_marked_changes_are_dated_and_in_order(self) -> None:
        dates = [date for date, _ in opponents.PLAYER_CHANGES]
        assert dates == sorted(dates)
        assert all(len(date) == 10 and what for date, what in opponents.PLAYER_CHANGES)
