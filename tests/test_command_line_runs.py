"""Q416: every index the command line prints says how many runs it rests on.

`CLAUDE.md`'s invariant, *a score carries its basis ... every number says which*, in the
page's words: *as the one run*, *as the mean of four runs*, and the pill's *1 run*,
*4 runs*. Q339 made the site and the local app say it; these are the commands.
`chess-variant-sandbox leaderboard` is held in `tests/test_loop.py`, beside the archive it prints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import cli
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS
from chess_variant_sandbox.metrics.index import IndexResult
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament import TournamentConfig, run_tournament

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"


def _result(**overrides: Any) -> IndexResult:
    fields: dict[str, Any] = {
        "ruleset_hash": "abc123", "ruleset_name": "a game", "weights_version": "v10",
        "total": 0.5, "viable": True, "violations": [], "scores": {"depth": 0.5},
        "contributions": {"depth": 0.5}, "raw": {"depth": 0.5},
        "families": {"depth": 0.5},
        "interval": {"sd": 0.02, "ci_low": 0.46, "ci_high": 0.54},
    }
    fields.update(overrides)
    return IndexResult(**fields)


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> Path:
    """One small tournament's games, which is one run."""
    played = run_tournament(
        load_spec(RULESETS / "gardner.json"),
        TournamentConfig(agents=("random", "greedy"), games_per_pairing=1, seed=0,
                         workers=1),
    )
    return Path(played.save(tmp_path_factory.mktemp("score") / "run"))


class TestScore:
    def test_the_headline_says_the_one_run(self) -> None:
        """`IndexResult.summary` says the count itself.

        Q416 put it in from `cli.py`, because the line is `metrics/index.py`'s and moving
        it would have restaled every library score for a sentence. `v11` C3 re-scored the
        library anyway, so the count moved into the line and the wrapper went.
        """
        head = _result().summary().partition("\n")[0]
        assert head == (
            "a game [abc123]  index=0.500 +/-0.020 [0.460, 0.540] as the one run "
            "(objective 0.500, weights v10)"
        )
        assert not hasattr(cli, "_as_the_one_run")

    def test_a_placeholder_total_prints_no_score(self) -> None:
        """A 0.0 that stands for no total is never printed as a number.

        Both refusals in `metrics/index.py` leave `contributions` and `families` empty
        and `total` at 0.0, which `IndexResult.scored` reads. The headline says *no
        score*, and the violations below it say why.
        """
        placeholder = _result(
            total=0.0, viable=False, contributions={}, families={}, interval=None,
            violations=["this roster cannot read this ruleset: the ladder is flat"],
        )
        assert not placeholder.scored
        said = placeholder.summary()
        assert said.partition("\n")[0] == "a game [abc123]  no score (weights v10)"
        assert "0.000" not in said and "index=" not in said and "objective" not in said
        assert "NOT VIABLE" in said and "the ladder is flat" in said
        assert "by family" not in said
        # The components that were measured are still facts, and still printed.
        assert "weakest: depth 0.50" in said

    def test_the_command_prints_the_one_run(self, run, capsys, monkeypatch) -> None:
        """The command prints `summary`'s headline as it stands, count and all."""
        monkeypatch.setattr(cli, "score_ruleset", lambda *args, **kwargs: _result())
        cli.main(["score", str(run), "--ruleset", str(RULESETS / "gardner.json"),
                  "--no-interval"])
        assert capsys.readouterr().out.splitlines()[0] == _result().summary().splitlines()[0]

    def test_the_command_prints_no_number_for_a_placeholder(self, run, capsys) -> None:
        """Scored for real, and a corpus this small is refused, so no number is printed.

        Two seats and no analysis pass fail the gates and miss five of `v11`'s
        components under `v11`. Before `v11` C3 this printed ``index=0.000 as the one run``.
        """
        cli.main(["score", str(run), "--ruleset", str(RULESETS / "gardner.json"),
                  "--no-interval"])
        out = capsys.readouterr().out
        assert out.splitlines()[0].endswith(f"]  no score (weights {DEFAULT_WEIGHTS})"), out
        assert "index=" not in out and "0.000 as the one run" not in out


def _scored(directory: Path, total: float, weights: str = "v10") -> None:
    """An `index-<weights>.json` as `calibrate` reads it, and nothing else."""
    directory.mkdir(parents=True)
    (directory / f"index-{weights}.json").write_text(json.dumps({
        "ruleset_name": directory.name, "weights_version": weights, "total": total,
        "viable": True, "interval": {"sd": 0.01}, "raw": {}, "unreadable": [],
    }), encoding="utf-8")


class TestCalibrate:
    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        """Three directories that say three different things about their runs."""
        _scored(tmp_path / "played", 0.6)
        (tmp_path / "played" / "games.jsonl").write_text("", encoding="utf-8")
        _scored(tmp_path / "measured", 0.5)
        (tmp_path / "measured" / "canonical.json").write_text(json.dumps({
            "seeds": [0, 1000, 2000, 3000], "weights_version": "v10",
        }), encoding="utf-8")
        _scored(tmp_path / "unsaid", 0.4)
        return tmp_path

    def test_the_count_is_read_off_what_the_directory_holds(self, root) -> None:
        index = {"weights_version": "v10"}
        assert cli._runs_behind(root / "played", index) == 1
        assert cli._runs_behind(root / "measured", index) == 4
        assert cli._runs_behind(root / "unsaid", index) == 0
        # A measurement's seeds are believed only under the weights its index was taken
        # under: an older index file beside a newer `canonical.json` says nothing.
        assert cli._runs_behind(root / "measured", {"weights_version": "v9"}) == 0

    def test_each_row_says_its_count_or_shows_no_number(
        self, root, capsys, monkeypatch
    ) -> None:
        from chess_variant_sandbox.metrics import calibration

        class Report:
            def summary(self) -> str:
                return "the report"

        monkeypatch.setattr(calibration, "calibrate", lambda entries, weights_version: Report())
        assert cli.main(["calibrate", str(root), "--weights", "v10"]) == 0
        rows = {
            line.split()[0]: line
            for line in capsys.readouterr().out.split("+/-  runs\n", 1)[1].splitlines()
        }
        assert rows["played"].endswith("0.600 0.010  1 run")
        assert rows["measured"].endswith("0.500 0.010  4 runs")
        assert rows["unsaid"].endswith("no score")
        assert "0.400" not in rows["unsaid"]
