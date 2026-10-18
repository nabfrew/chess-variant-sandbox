"""The site's small readers and the local server's three endpoints that nothing ran
(archive/plan-tests F9, Q751b).

No database: these are the pieces of `site/` that are pure (the load report's words, the
level-to-tab mapping, the one-run basis, the playability child) and the three `web/server.py`
methods reached through `App.get` / `App.post`, with the thread that does the work replaced by
a recorder so that what is held is the endpoint's own promise: how the request body is
clamped, defaulted and handed on.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from test_evolving import _finish

from chess_variant_sandbox import corpus
from chess_variant_sandbox.loop import LoopConfig
from chess_variant_sandbox.site import load as site_load
from chess_variant_sandbox.site import provisional, screening, stages
from chess_variant_sandbox.site import server as site_server
from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.web import evolving
from chess_variant_sandbox.web.server import App

RULESETS = Path(__file__).resolve().parent.parent / "rulesets"
DULL = RULESETS / "dull.json"


class TestLoadReport:
    def change(self, added: int = 0, updated: int = 0, unchanged: int = 0) -> site_load.Change:
        return site_load.Change(added=added, updated=updated, unchanged=unchanged)

    def test_a_load_that_wrote_nothing_says_so_first(self) -> None:
        report = site_load.Report(games=self.change(unchanged=3))
        lines = report.summary().splitlines()
        assert lines[0] == "nothing changed"
        assert "  games        3 row(s): 0 added, 0 updated" in lines

    def test_the_head_counts_every_table_that_wrote(self) -> None:
        report = site_load.Report(
            games=self.change(added=2), measurements=self.change(updated=1),
            matches=self.change(added=4), ratings=self.change(unchanged=9),
        )
        assert report.written == 7
        assert report.summary().splitlines()[0] == "7 row(s) written"

    def test_every_table_has_its_line_in_the_order_it_is_loaded(self) -> None:
        names = [line.split()[0] for line in site_load.Report().summary().splitlines()[1:]]
        assert names == ["games", "retired", "moved", "rested", "measurements", "ratings",
                         "matches", "links"]

    def test_the_notes_and_the_files_that_would_not_load_come_last_each_in_its_own_words(
        self,
    ) -> None:
        report = site_load.Report(notes=["skipped abc: deleted"], skipped=["bad.json: not json"])
        lines = report.summary().splitlines()
        assert lines[-2:] == ["  skipped abc: deleted", "  skipped: bad.json: not json"]


class TestStageTab:
    @pytest.mark.parametrize("level", [lv for entry in stages.TABS for lv in entry["levels"]])
    def test_a_listed_level_is_in_the_tab_that_lists_it(self, level: str) -> None:
        tab = stages.tab(level)
        assert tab is not None
        assert level in next(e for e in stages.TABS if e["key"] == tab)["levels"]

    @pytest.mark.parametrize("level", [stages.DRAFT, stages.RESTING, stages.RETIRED, "nonsense"])
    def test_a_level_no_tab_lists_has_none(self, level: str) -> None:
        assert stages.tab(level) is None

    def test_the_library_and_trial_tabs_are_these_keys(self) -> None:
        assert stages.tab(stages.LIBRARY) == stages.tab(stages.BASELINE) == "library"
        assert stages.tab(stages.ON_TRIAL) == "trial"


class TestOneRunBasis:
    spec = load_spec(DULL)

    def test_is_the_basis_a_one_run_measurement_of_this_game_would_carry(self) -> None:
        digest = spec_hash(self.spec)
        expected = corpus.current_basis(digest, spec=self.spec, seeds=corpus.ONE_SEED)
        for reader in (provisional.one_run, site_server._one_run_basis):
            assert reader(digest, self.spec) == expected

    def test_it_is_a_one_run_basis_and_not_the_librarys(self) -> None:
        digest = spec_hash(self.spec)
        library = corpus.current_basis(digest, spec=self.spec)
        one = provisional.one_run(digest, self.spec)
        assert one is not None and one != library
        assert one.seeds == corpus.ONE_SEED and library.seeds == corpus.LIBRARY_SEEDS

    @pytest.mark.parametrize("nothing", [None, "rules", [], 3])
    def test_a_row_without_rules_has_no_basis_at_all(self, nothing: Any) -> None:
        assert provisional.one_run("h", nothing) is None
        assert site_server._one_run_basis("h", nothing) is None


class TestScreeningChild:
    """The check a save runs in a child process: its verdict, and the pipe it speaks."""

    def test_a_playable_game_is_screened_after_it_is_smoked(self) -> None:
        verdict = screening.verdict(load_spec(DULL), games=4, smoke_seconds=5.0, workers=1)
        assert verdict["passed"] is True and verdict["reasons"] == []
        assert verdict["rung"] == "screen"
        assert verdict["smoke"]["passed"] is True
        assert verdict["stats"]["games"] > 0

    def test_a_game_that_fails_smoke_never_reaches_the_screen(self) -> None:
        spec = load_spec(DULL)
        spec["rules"]["max_plies"] = 4
        verdict = screening.verdict(spec, games=4, smoke_seconds=5.0, workers=1)
        assert verdict["passed"] is False and verdict["rung"] == "smoke"
        assert verdict["stats"] == {}
        assert verdict["reasons"] == verdict["smoke"]["reasons"] != []

    def test_the_child_reads_one_draft_on_stdin_and_writes_one_verdict_on_stdout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = load_spec(DULL)
        asked = {"spec": spec, "games": "4", "smoke_seconds": "5", "workers": "1"}
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(asked)))
        monkeypatch.setattr(sys, "stdout", out)
        screening._child()
        written = json.loads(out.getvalue())
        assert written == screening.verdict(spec, games=4, smoke_seconds=5.0, workers=1)

    def test_the_drafts_budget_and_the_one_core_reach_both_rungs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A thread count never changes an answer, so only the wiring can show it."""
        import importlib
        from types import SimpleNamespace

        import chess_variant_sandbox.loop.gates as gates

        # `loop/__init__` re-exports a function named `screen`, which hides the module
        screen_module = importlib.import_module("chess_variant_sandbox.loop.screen")

        seen: dict[str, dict[str, Any]] = {}

        def smoke(spec: Any, **kwargs: Any) -> Any:
            seen["smoke"] = kwargs
            return SimpleNamespace(passed=True, reasons=[], as_dict=lambda: {})

        def screen(spec: Any, **kwargs: Any) -> Any:
            seen["screen"] = kwargs
            return SimpleNamespace(passed=True, reasons=[], stats={})

        monkeypatch.setattr(gates, "smoke_test", smoke)
        monkeypatch.setattr(screen_module, "screen", screen)
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(
            {"spec": {}, "games": "9", "smoke_seconds": "2.5", "workers": "3"})))
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        screening._child()
        assert seen["smoke"] == {"games": 9, "time_budget": 2.5, "workers": 3}
        assert seen["screen"] == {"workers": 3}

    def test_run_as_a_script_it_is_the_same_child(self) -> None:
        """What the parent process starts: this file, with the draft on stdin."""
        import subprocess

        asked = {"spec": load_spec(DULL), "games": 4, "smoke_seconds": 5.0, "workers": 1}
        done = subprocess.run(  # noqa: S603 - our own interpreter and our own file
            [sys.executable, screening.__file__], input=json.dumps(asked), text=True,
            capture_output=True, check=True, timeout=60,
        )
        verdict = json.loads(done.stdout)
        assert verdict["passed"] is True and verdict["rung"] == "screen"


@pytest.fixture
def app(tmp_path: Path) -> tuple[App, str]:
    rulesets = tmp_path / "rulesets"
    rulesets.mkdir()
    (rulesets / "dull.json").write_text(DULL.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "runs").mkdir()
    (tmp_path / "archive").mkdir()
    return (App(rulesets, tmp_path / "runs", archive_dir=tmp_path / "archive"),
            spec_hash(load_spec(DULL)))


class RatingRecorder:
    """Stands in for `Ratings`: remembers what `start` was handed, rates nothing."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def start(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)

        class Run:
            @staticmethod
            def as_dict() -> dict[str, Any]:
                return {"state": "running", "name": kwargs["name"]}

        return Run()

    def stored(self, digest: str) -> None:
        return None


class TestStartRating:
    def test_the_picker_is_rated_on_the_games_own_spec_and_roster(
        self, app: tuple[App, str]
    ) -> None:
        server, digest = app
        server.ratings = RatingRecorder()  # type: ignore[assignment]
        answer = server.post("/api/rate", {"hash": digest})
        assert answer == {"state": "running", "name": "dull"}
        [call] = server.ratings.calls  # type: ignore[attr-defined]
        assert call["ruleset_hash"] == digest and call["name"] == "dull"
        assert call["spec"] == server.game(digest).spec
        assert call["roster"] == server.roster(digest)
        assert (call["games"], call["seed"]) == (2, 0)

    @pytest.mark.parametrize(("asked", "games"), [(0, 1), (-5, 1), (3, 3), (6, 6), (50, 6)])
    def test_games_per_pairing_is_kept_between_one_and_six(
        self, app: tuple[App, str], asked: int, games: int
    ) -> None:
        server, digest = app
        server.ratings = RatingRecorder()  # type: ignore[assignment]
        server.post("/api/rate", {"hash": digest, "games": asked, "seed": "7"})
        [call] = server.ratings.calls  # type: ignore[attr-defined]
        assert call["games"] == games and call["seed"] == 7

    def test_an_unknown_game_is_refused_before_anything_starts(
        self, app: tuple[App, str]
    ) -> None:
        server, _ = app
        server.ratings = RatingRecorder()  # type: ignore[assignment]
        with pytest.raises(LookupError):
            server.post("/api/rate", {"hash": "feedfeedfeed"})
        assert server.ratings.calls == []  # type: ignore[attr-defined]


class TestStartRun:
    @pytest.fixture
    def handed(self, monkeypatch: pytest.MonkeyPatch) -> list[LoopConfig]:
        configs: list[LoopConfig] = []

        def evolve(config: LoopConfig, report: Any) -> None:
            configs.append(config)
            report("one generation")

        monkeypatch.setattr(evolving, "evolve", evolve)
        return configs

    def test_a_design_run_starts_from_the_chosen_game_in_a_directory_of_its_own(
        self, app: tuple[App, str], handed: list[LoopConfig]
    ) -> None:
        server, digest = app
        shown = server.post("/api/run", {"hash": digest, "theme": "  more captures  ",
                                         "generations": 3, "judge": False})
        _finish(shown["id"])
        [config] = handed
        game = server.game(digest)
        assert config.seeds == (game.path,)
        assert config.theme == "more captures"
        assert (config.generations, config.judge) == (3, False)
        assert config.directory == server.runs_dir / "design" / shown["id"]
        assert shown["parent"] == "dull" and shown["theme"] == "more captures"
        assert server.get(f"/api/run/{shown['id']}")["log"][-1] == "one generation"

    def test_the_defaults_are_four_generations_and_the_judge_on(
        self, app: tuple[App, str], handed: list[LoopConfig]
    ) -> None:
        server, digest = app
        shown = server.post("/api/run", {"hash": digest})
        _finish(shown["id"])
        assert (handed[0].generations, handed[0].judge, handed[0].theme) == (4, True, "")

    @pytest.mark.parametrize(("asked", "kept"), [(0, 1), (-2, 1), (40, 40), (500, 40)])
    def test_generations_are_kept_between_one_and_forty(
        self, app: tuple[App, str], handed: list[LoopConfig], asked: int, kept: int
    ) -> None:
        server, digest = app
        shown = server.post("/api/run", {"hash": digest, "generations": asked})
        _finish(shown["id"])
        assert handed[-1].generations == kept

    def test_each_run_has_its_own_id(self, app: tuple[App, str], handed: list[LoopConfig]) -> None:
        server, digest = app
        ids = {server.post("/api/run", {"hash": digest})["id"] for _ in range(3)}
        for run_id in ids:
            _finish(run_id)
        assert len(ids) == 3


class TestGraph:
    def test_the_library_map_has_a_node_for_each_game_and_none_for_anything_else(
        self, tmp_path: Path
    ) -> None:
        rulesets = tmp_path / "rulesets"
        rulesets.mkdir()
        names = ("dull", "gardner")
        for name in names:
            (rulesets / f"{name}.json").write_text(
                (RULESETS / f"{name}.json").read_text(encoding="utf-8"), encoding="utf-8")
        server = App(rulesets, tmp_path / "runs", archive_dir=tmp_path / "archive")
        graph = server.graph()
        assert {node["name"] for node in graph["nodes"]} == set(names)
        assert server.get("/api/graph") == graph, "the route is this method's answer"
        assert (tmp_path / "runs" / "graph").is_dir(), "the pairs are cached beside the runs"

    def test_a_game_from_a_design_run_is_listed_and_is_not_a_node(self, tmp_path: Path) -> None:
        """A candidate is not in the library, so the map does not draw it."""
        from chess_variant_sandbox.loop.archive import Archive, Entry

        rulesets = tmp_path / "rulesets"
        rulesets.mkdir()
        (rulesets / "dull.json").write_text(DULL.read_text(encoding="utf-8"), encoding="utf-8")
        child = load_spec(RULESETS / "gardner.json")
        digest = spec_hash(child)
        Archive(tmp_path / "runs" / "design").record(
            Entry(hash=digest, name="a candidate", generation=1, parent=None,
                  status="evaluated", index=0.5, objective=0.5, viable=True),
            child,
        )
        server = App(rulesets, tmp_path / "runs", archive_dir=tmp_path / "archive")
        listed = {game.hash: game.source for game in server.refresh()}
        assert listed[digest] in ("designed", "evolved")
        assert {n["hash"] for n in server.graph()["nodes"]} == {spec_hash(load_spec(DULL))}
