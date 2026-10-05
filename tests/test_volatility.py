"""Revision volatility (EV6, Q690; `docs/plan-evolving-score.md` §7.1, pre-registered).

What these hold down, on fixture histories written straight to disk:

- **The reading**: the spread and the largest step across the newest `VOLATILITY_K`
  revisions at the newest weights version, the present `canonical.json` last.
- **Fewer than two revisions say so**: no history, no score, one score; never an error and
  never a zero.
- **A weights change breaks the series**: revisions before it are counted and left out,
  and no step is computed across it.
- **It is shown to the maintainer only**: `/admin` and the local scores panel draw it,
  a game page's payload carries none of it, and it moves nothing (no metric or trust rule
  reads it).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import history, measurement

GAME = "a" * 12


def _present(runs: Path, index: float | None, weights: str = "v12",
             low: float | None = None, high: float | None = None) -> None:
    directory = runs / GAME
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "canonical.json").write_text(json.dumps(
        {"index": index, "ci_low": low, "ci_high": high, "weights_version": weights}),
        encoding="utf-8")


def _was(runs: Path, n: int, index: float | None, weights: str = "v12",
         moved: tuple[str, ...] = ("engine",), replaced: str = "2026-10-01") -> None:
    entry = history.entry_for(
        n, {"index": index, "runs": [{}], "weights_version": weights, "basis": None},
        None, "2026-09-01", "replay", ("why",))
    data = {**entry.as_dict(), "moved": list(moved), "replaced": replaced}
    history.append(GAME, runs, history.Entry.from_dict(data))


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    return tmp_path / "runs"


class TestFewerThanTwo:
    def test_nothing_stored_says_so(self, runs: Path) -> None:
        found = history.volatility(GAME, runs)
        assert found["revisions"] == 0
        assert found["spread"] is None and found["largest_step"] is None
        assert found["said"] == "No score stored, so nothing to compare."

    def test_one_score_says_there_is_nothing_to_compare_and_is_never_zero(
        self, runs: Path
    ) -> None:
        _present(runs, 0.6, low=0.58, high=0.62)
        found = history.volatility(GAME, runs)
        assert found["revisions"] == 1
        assert found["spread"] is None and found["largest_step"] is None
        assert found["said"].startswith("One score at weights v12, so nothing to compare yet.")
        assert "±0.020" in found["said"]
        assert "0.000" not in found["said"]

    def test_a_present_file_with_no_score_is_counted_not_compared(self, runs: Path) -> None:
        _was(runs, 1, 0.5)
        _present(runs, None)
        found = history.volatility(GAME, runs)
        assert found["revisions"] == 1 and found["unscored"] == 1
        assert found["spread"] is None
        assert "1 with no score" in found["said"]


class TestTheReading:
    def test_two_revisions(self, runs: Path) -> None:
        _was(runs, 1, 0.50, moved=("agents",), replaced="2026-10-02")
        _present(runs, 0.53, low=0.52, high=0.54)
        found = history.volatility(GAME, runs)
        assert found["revisions"] == 2
        assert found["spread"] == pytest.approx(0.03)
        assert found["largest_step"] == pytest.approx(0.03)
        assert found["step_moved"] == ["agents"] and found["step_replaced"] == "2026-10-02"
        assert found["interval_half"] == pytest.approx(0.01)
        assert found["said"] == (
            "Moved 0.030 across its last 2 scores at weights v12 (largest single change "
            "0.030, when the players changed). Its interval is ±0.010.")

    def test_many_revisions_read_the_newest_k(self, runs: Path) -> None:
        # Seven revisions; the oldest two (0.10, 0.90) fall outside the newest five.
        values = [0.10, 0.90, 0.50, 0.52, 0.47, 0.55]
        for n, value in enumerate(values, start=1):
            _was(runs, n, value, moved=("metrics",), replaced=f"2026-10-0{n}")
        _present(runs, 0.54)
        found = history.volatility(GAME, runs)
        assert history.VOLATILITY_K == 5
        assert found["revisions"] == 5
        assert found["spread"] == pytest.approx(0.55 - 0.47)
        assert found["largest_step"] == pytest.approx(0.08)  # 0.47 -> 0.55
        assert found["step_replaced"] == "2026-10-05"
        assert "its metrics changed" in found["said"]
        assert "interval" not in found["said"]  # no interval stored, none invented

    def test_the_order_of_lines_on_disk_does_not_matter(self, runs: Path) -> None:
        _was(runs, 2, 0.60)
        _was(runs, 1, 0.40)
        _present(runs, 0.50)
        found = history.volatility(GAME, runs)
        assert found["spread"] == pytest.approx(0.20)
        assert found["largest_step"] == pytest.approx(0.20)  # 0.40 -> 0.60


class TestAWeightsChange:
    def test_revisions_before_a_weights_change_are_left_out_and_counted(
        self, runs: Path
    ) -> None:
        _was(runs, 1, 0.30, weights="v11")
        _was(runs, 2, 0.31, weights="v11", moved=("weights_version",))
        _was(runs, 3, 0.70, weights="v12")
        _present(runs, 0.72)
        found = history.volatility(GAME, runs)
        assert found["revisions"] == 2 and found["other_weights"] == 2
        assert found["spread"] == pytest.approx(0.02)  # never 0.42 across the scales
        assert found["largest_step"] == pytest.approx(0.02)
        assert "Not compared: 2 earlier at other weights." in found["said"]

    def test_the_series_is_the_trailing_run_only(self, runs: Path) -> None:
        # v12, then v11, then v12 again: only the last run counts.
        _was(runs, 1, 0.90, weights="v12")
        _was(runs, 2, 0.30, weights="v11")
        _present(runs, 0.50, weights="v12")
        found = history.volatility(GAME, runs)
        assert found["revisions"] == 1 and found["other_weights"] == 2
        assert found["spread"] is None
        assert found["said"].startswith("One score at weights v12")

    def test_a_score_just_re_scored_to_new_weights_has_nothing_to_compare(
        self, runs: Path
    ) -> None:
        _was(runs, 1, 0.40, weights="v11", moved=("weights_version",))
        _present(runs, 0.70, weights="v12")
        found = history.volatility(GAME, runs)
        assert found["revisions"] == 1 and found["spread"] is None
        assert "1 earlier at other weights" in found["said"]


class TestRows:
    def test_every_named_game_is_listed_by_name(self, runs: Path) -> None:
        _present(runs, 0.6)
        rows = history.volatility_rows(runs, {GAME: "beta", "b" * 12: "alpha"})
        assert [row["name"] for row in rows] == ["alpha", "beta"]
        assert rows[0]["said"].startswith("No score stored")


class TestWhereItIsShown:
    def test_admin_draws_it(self) -> None:
        from chess_variant_sandbox.site import pages

        page = pages.admin("me")
        assert 'id="score-volatility"' in page
        script = (measurement.ROOT / "chess_variant_sandbox/site/static/admin.js").read_text(
            encoding="utf-8")
        assert "scoreVolatility(payload.volatility || [])" in script

    def test_the_local_scores_panel_draws_it(self) -> None:
        from page_text import PAGE

        assert "function volatilityHtml(rows)" in PAGE
        assert "${volatilityHtml(s.volatility)}" in PAGE

    def test_the_local_scores_payload_carries_it(self, tmp_path: Path) -> None:
        from chess_variant_sandbox.web.server import App

        server = App(ruleset_dir=measurement.ROOT / "rulesets", runs_dir=tmp_path / "runs",
                     archive_dir=tmp_path / "archive")
        payload: dict[str, Any] = server.scoring()
        assert payload["games"]
        assert len(payload["volatility"]) == len(payload["games"])
        for row in payload["volatility"]:
            assert row["said"] == "No score stored, so nothing to compare."

    def test_a_game_page_carries_none_of_it(self) -> None:
        from chess_variant_sandbox.web import library

        source = (measurement.ROOT / "chess_variant_sandbox/web/library.py").read_text(
            encoding="utf-8")
        assert "volatility" not in source
        assert callable(library.detail)


class TestItMovesNothing:
    def test_no_metric_or_trust_rule_reads_it(self) -> None:
        package = measurement.ROOT / "chess_variant_sandbox"
        for path in [*(package / "metrics").rglob("*.py"), package / "trust.py",
                     package / "scoring.py", package / "measurement.py"]:
            if path.exists():
                text = path.read_text(encoding="utf-8")
                assert "history.volatility" not in text and "VOLATILITY_K" not in text, path
                assert "import history" not in text and "history import" not in text, path
