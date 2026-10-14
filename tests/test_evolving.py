"""`web/evolving.py`: a design run started from the browser, on a thread, and watched.

The local server's *design a game from this one* (`POST /api/run`, `GET /api/run/<id>`)
is `Runs.start` and `Runs.get`. The loop itself is not under test here (it spends money);
`evolve` is replaced by a stand-in that records what it was handed and reports through the
callback it was given, so what is held is the wrapper's own promise: the run is configured
from the chosen game and nothing else, its report reaches the page, and a run that dies says
why in words the page can show (Q751, plan-tests F9: none of this had run).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.loop import LoopConfig
from chess_variant_sandbox.web import evolving
from chess_variant_sandbox.web.evolving import LOG_LINES, Runs

# Two budgets of one kind, the least a roster may be (`LoopConfig` refuses less).
ROSTER = ("minimax:nodes=400", "minimax:nodes=1600")


def _finish(run_id: str) -> None:
    """Wait for the run's thread, which `Runs` names after the run."""
    for thread in threading.enumerate():
        if thread.name == f"evolve-{run_id}":
            thread.join(timeout=10)
            assert not thread.is_alive()


@pytest.fixture
def handed(monkeypatch: pytest.MonkeyPatch) -> list[LoopConfig]:
    """`evolve` replaced: it records its config and reports two lines, one of them two."""
    configs: list[LoopConfig] = []

    def evolve(config: LoopConfig, report: Any) -> None:
        configs.append(config)
        report("generation 1 of 2")
        report("proposed: a\nkept: b")

    monkeypatch.setattr(evolving, "evolve", evolve)
    return configs


def test_a_run_is_configured_from_the_chosen_game(handed: list[LoopConfig], tmp_path: Path) -> None:
    runs = Runs()
    spec = tmp_path / "chess.json"
    run = runs.start("r1", spec, "chess", "more drops", tmp_path / "design" / "r1", 3,
                     judge=False, games_per_pairing=5)
    _finish("r1")
    (config,) = handed
    assert config.seeds == (spec,)
    assert config.generations == 3
    assert config.theme == "more drops"
    assert config.judge is False
    assert config.games_per_pairing == 5
    assert config.directory == tmp_path / "design" / "r1"
    # No roster named: the loop's own, so the starting point is measured on the roster the
    # run's children will be.
    assert config.agents == LoopConfig().agents
    assert run.state == "finished" and run.error == ""


def test_a_named_roster_replaces_the_loops(handed: list[LoopConfig], tmp_path: Path) -> None:
    Runs().start("r2", tmp_path / "x.json", "x", "", tmp_path, 1, agents=ROSTER)
    _finish("r2")
    assert handed[0].agents == ROSTER


def test_the_report_reaches_the_page_line_by_line(handed: list[LoopConfig], tmp_path: Path) -> None:
    runs = Runs()
    runs.start("r3", tmp_path / "x.json", "dull", "", tmp_path, 2)
    _finish("r3")
    shown = runs.get("r3").as_dict()
    assert shown["log"] == ["starting from dull", "theme: (none given)",
                            "generation 1 of 2", "proposed: a", "kept: b"]
    assert shown["state"] == "finished"
    assert shown["parent"] == "dull" and shown["generations"] == 2
    # The listing is the same runs without their logs, which can be long.
    (listed,) = runs.all()
    assert "log" not in listed
    assert listed == {k: v for k, v in shown.items() if k != "log"}


def test_a_run_that_dies_says_why(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def evolve(config: LoopConfig, report: Any) -> None:
        report("generation 1 of 4")
        raise RuntimeError("no API key")

    monkeypatch.setattr(evolving, "evolve", evolve)
    runs = Runs()
    runs.start("r4", tmp_path / "x.json", "x", "t", tmp_path, 4)
    _finish("r4")
    run = runs.get("r4")
    assert run.state == "failed"
    assert run.error == "RuntimeError: no API key"
    assert "RuntimeError: no API key" in run.lines
    assert run.lines.index("generation 1 of 4") < run.lines.index("RuntimeError: no API key")


def test_an_unknown_run_is_a_key_error_the_server_turns_into_a_404() -> None:
    with pytest.raises(KeyError, match="no such run"):
        Runs().get("nope")


def test_the_log_keeps_only_its_window() -> None:
    run = evolving.Run(id="r", directory=Path("."), theme="", parent="p", generations=1)
    for n in range(LOG_LINES + 10):
        run.report(f"line {n}")
    assert len(run.lines) == LOG_LINES
    assert run.lines[0] == "line 10"
    run.report("")
    assert run.lines[-1] == ""
