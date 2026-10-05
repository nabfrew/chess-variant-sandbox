"""The per-move accuracy report (Q701, `docs/reference/win-probability.md` §1, *The report*)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from chess_variant_sandbox import accuracy
from chess_variant_sandbox.accuracy import Judgement, judge, move_accuracy, render, word_for
from chess_variant_sandbox.cli import main
from chess_variant_sandbox.winprob import (
    CALIBRATED,
    ROUGH,
    TOO_FEW,
    Basis,
    Calibration,
    GameReading,
    Position,
    Ruler,
    WinFit,
    chances,
)

ROOT = Path(__file__).resolve().parents[1]
RULER = Ruler(nodes=1, max_depth=1, goal=0.0, immune=0.0, unit=1.0, decided_at=1e5, build="")


def _fit(verdict: str = CALIBRATED, k: float = 1.0) -> WinFit:
    basis = Basis("winprob-1", "fake", 0, "d", 10, 300, 10, 300, 0, None, {})
    table = Calibration((), 300, 10, 0.0, 0.1, 0.2, verdict, ())
    return WinFit(k=k, ruler=RULER, basis=basis, calibration=table)


def _game() -> GameReading:
    positions = (
        Position(0, 0, 5, 0.0, 0.0, 1),  # best kept: no loss
        Position(1, 1, 4, 1.0, -1.0, 2),  # a real loss
        Position(2, 0, 1, -0.5, -0.5, 3),  # the only move
        Position(3, 1, 3, 2e5, 0.0, 4),  # threw away a win the search saw
    )
    return GameReading(0, 0, ("a", "b"), 1, positions)


def test_the_words_follow_the_registered_bands() -> None:
    assert word_for(0.0) == word_for(0.0199) == "best or as good"
    assert word_for(0.02) == word_for(0.0499) == "good"
    assert word_for(0.05) == "a slip"
    assert word_for(0.10) == "a mistake"
    assert word_for(0.20) == word_for(1.0) == "a blunder"


def test_a_move_scores_by_the_chances_it_gave_away() -> None:
    assert move_accuracy(0.0) == 100.0
    assert move_accuracy(0.10) == pytest.approx(36.79, abs=0.01)
    assert move_accuracy(-0.5) == 100.0


def test_a_report_reads_each_move_through_the_fit() -> None:
    report = judge(_game(), _fit())
    kept, lost, only, thrown = report.moves
    assert kept.loss == 0.0 and kept.word == "best or as good"
    assert lost.before == pytest.approx(chances(1.0, 1.0))
    assert lost.loss == pytest.approx(chances(1.0, 1.0) - chances(1.0, -1.0))
    assert lost.word == "a blunder"
    assert only.forced and only.word == "the only move" and only.accuracy is None
    assert thrown.before == 1.0 and thrown.loss == pytest.approx(0.5)
    # The only move counts toward nothing; each side's mean is over its own choices.
    assert report.accuracy(0) == 100.0
    assert report.accuracy(1) == pytest.approx((move_accuracy(lost.loss) + move_accuracy(0.5)) / 2)
    assert report.count(1, "a blunder") == 2


def test_a_side_with_no_choice_has_no_accuracy() -> None:
    game = GameReading(0, 0, ("a", "b"), 0, (Position(0, 0, 1, 0.0, 0.0, 0),))
    report = judge(game, _fit())
    assert report.accuracy(0) is None and report.accuracy(1) is None
    assert "never had a choice" in render(report)


def test_the_report_carries_its_fit_basis() -> None:
    payload = judge(_game(), _fit()).as_dict()
    assert payload["fit"]["basis"]["method"] == "winprob-1"
    assert payload["fit"]["calibration"]["verdict"] == CALIBRATED
    assert len(payload["moves"]) == 4


@pytest.mark.parametrize("verdict", [CALIBRATED, ROUGH, TOO_FEW])
def test_the_words_stay_plain(verdict: str) -> None:
    text = render(judge(_game(), _fit(verdict)), "a game").lower()
    for word in accuracy.FORBIDDEN:
        assert not re.search(rf"\b{word}\b", text), word
    assert ("rough guide" in text) == (verdict != CALIBRATED)
    assert "first player" in text and "second player" in text
    assert "draw counted as half" in text


def test_a_judgement_with_texts_names_the_better_move() -> None:
    game = GameReading(0, 0, ("a", "b"), 1, (Position(0, 0, 2, 1.0, -1.0, 1),))
    report = judge(game, _fit(), [("x1", "y2")])
    assert "Move 1, first player x1: a blunder" in render(report)
    assert "(the best was y2)" in render(report)
    assert isinstance(report.moves[0], Judgement)


def test_the_command_judges_a_game_the_core_played(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end on the real core: a tiny run of a committed ruleset, then `accuracy`.

    Too few games for a verdict, on purpose: the bound needs 200 held-out positions,
    and the command must say so rather than present the chances as checked.
    """
    ruleset = ROOT / "rulesets" / "gardner.json"
    run = tmp_path / "run"
    assert (
        main(
            [
                "tournament",
                str(ruleset),
                "--agents",
                "random",
                "greedy",
                "--games",
                "2",
                "--workers",
                "1",
                "--out",
                str(run),
            ]
        )
        == 0
    )
    games = (run / "games.jsonl").read_text(encoding="utf-8")
    capsys.readouterr()

    out = tmp_path / "report.json"
    assert (
        main(["accuracy", str(run), "--ruleset", str(ruleset), "--game", "1", "--out", str(out)])
        == 0
    )
    printed = capsys.readouterr().out
    assert "rough guide" in printed and "(too few)" in printed
    payload = json.loads(out.read_text(encoding="utf-8"))
    first_game = json.loads(games.splitlines()[1])
    assert len(payload["moves"]) == len(first_game["moves"])
    assert payload["fit"]["basis"]["ruleset_hash"] == first_game["ruleset_hash"]
    assert payload["fit"]["basis"]["ruler"]["build"]
    # It reads the run and writes only where it was told to.
    assert (run / "games.jsonl").read_text(encoding="utf-8") == games
    assert sorted(p.name for p in run.iterdir()) == sorted(["games.jsonl", "tournament.json"])
