"""`v12`: every curve end names its source, and the band checks the scale without placing it.

`docs/archive/plan-evolving-score.md` §4 (EV4, Q689) and `docs/reference/v12-calibration.md`, whose
pre-registration fixed the domain, gate and meaning ends before the reference read. Each
assertion is a fact about the files, so a later edit that moved an end without a new version,
or put the band back in charge of one, fails here.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import cli
from chess_variant_sandbox.metrics import WeightsError, load_weights, parse_weights
from chess_variant_sandbox.metrics.calibration import (
    DOMAIN,
    Measured,
    band_quantiles,
    check_ends,
)
from chess_variant_sandbox.metrics.scoring import DEFAULT_WEIGHTS, SOURCE_KINDS, WEIGHTS_DIR

# The ends `v12-calibration.md`'s pre-registration fixed from the domain, a gate or a
# sentence, before any quantile was read. The reference ends are the read's, pinned below.
FIXED = {
    ("first_player_advantage", "good_at"): (0.0, "domain"),
    ("first_player_advantage", "zero_at"): (0.5, "meaning"),
    ("stall_rate", "good_at"): (0.0, "domain"),
    ("stall_rate", "zero_at"): (0.7, "gate"),
    ("mean_branching", "zero_low"): (1.5, "gate"),
    ("ply_spread", "zero_at"): (0.0, "meaning"),
    ("position_divergence", "zero_at"): (0.0, "domain"),
    ("position_divergence", "good_at"): (1.0, "domain"),
    ("blunder_recoverability", "zero_low"): (0.0, "domain"),
    ("blunder_recoverability", "good_low"): (0.25, "meaning"),
    ("blunder_recoverability", "good_high"): (0.75, "meaning"),
    ("blunder_recoverability", "zero_high"): (1.0, "domain"),
    ("decision_criticality", "zero_low"): (0.0, "domain"),
    ("decision_criticality", "zero_high"): (1.0, "domain"),
    ("piece_utilisation", "good_at"): (1.0, "domain"),
    ("skill_gradient", "zero_at"): (8.0, "gate"),
    ("instance_participation", "zero_low"): (0.0, "domain"),
    ("instance_participation", "zero_high"): (1.0, "domain"),
    ("lead_conversion", "zero_low"): (0.5, "meaning"),
    ("lead_conversion", "zero_high"): (1.0, "domain"),
}
# The read (`v12-calibration.md`, *The read* and *A1, read*), pinned.
READ = {
    ("mean_branching", "good_low"): 9.6242,
    ("mean_branching", "good_high"): 56.3717,
    ("mean_branching", "zero_high"): 112.7434,
    ("ply_spread", "good_at"): 0.9707,
    ("decision_criticality", "good_low"): 0.2954,
    ("decision_criticality", "good_high"): 0.6535,
    ("piece_utilisation", "zero_at"): 0.735,
    ("skill_gradient", "good_at"): 201.9833,
    ("instance_participation", "good_low"): 0.5139,
    ("instance_participation", "good_high"): 0.8847,
    ("lead_conversion", "good_low"): 0.6164,
    ("lead_conversion", "good_high"): 0.9357,
}


def _raw(version: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((WEIGHTS_DIR / f"{version}.json").read_text(encoding="utf-8"))
    return data


class TestVersionTwelve:
    def test_it_is_the_default(self) -> None:
        assert DEFAULT_WEIGHTS == "v12"

    def test_only_the_ends_moved_from_v11(self) -> None:
        """So a `v11` to `v12` move of any score is the scale changing source (§4.4)."""
        eleven, twelve = load_weights("v11"), load_weights("v12")
        assert twelve.components() == eleven.components()
        assert twelve.weights == eleven.weights
        assert twelve.families == eleven.families
        assert {n: c.kind for n, c in twelve.curves.items()} == {
            n: c.kind for n, c in eleven.curves.items()
        }
        assert twelve.gates == eleven.gates
        assert twelve.llm_weight == eleven.llm_weight
        assert twelve.optional == eleven.optional
        for name, config in _raw("v12")["components"].items():
            for key in ("roster_rho", "evidence"):
                assert config.get(key) == _raw("v11")["components"][name].get(key), name

    def test_every_end_names_one_source(self) -> None:
        weights = load_weights("v12")
        assert set(weights.sources) == set(weights.components())
        for name, curve in weights.curves.items():
            assert set(weights.sources[name]) == set(curve.points), name
            assert set(weights.sources[name].values()) <= set(SOURCE_KINDS), name

    def test_the_pre_registered_ends(self) -> None:
        weights = load_weights("v12")
        every = {
            (name, end) for name, curve in weights.curves.items() for end in curve.points
        }
        assert every == set(FIXED) | set(READ)
        for (name, end), (value, kind) in FIXED.items():
            assert weights.curves[name].points[end] == value, (name, end)
            assert weights.sources[name][end] == kind, (name, end)
        for (name, end), value in READ.items():
            assert weights.curves[name].points[end] == value, (name, end)
            assert weights.sources[name][end] == "reference", (name, end)

    def test_derived_reference_ends_follow_their_rule(self) -> None:
        points = {n: c.points for n, c in load_weights("v12").curves.items()}
        branching = points["mean_branching"]
        assert branching["zero_high"] == pytest.approx(2 * branching["good_high"], abs=1e-9)
        # The low quantile less the distance between the quantiles: 0.8591 - (0.9832 - 0.8591).
        assert points["piece_utilisation"]["zero_at"] == pytest.approx(
            0.8591 - (0.9832 - 0.8591), abs=5e-5
        )

    def test_domain_ends_lie_on_the_domain(self) -> None:
        weights = load_weights("v12")
        for name, kinds in weights.sources.items():
            low, high = DOMAIN[name]
            for end, kind in kinds.items():
                if kind == "domain":
                    value = weights.curves[name].points[end]
                    assert value in (low, high), (name, end)

    def test_no_end_reads_the_corpus_when_the_file_loads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `reference` end is a pinned number, never a read: loading `v12` from a
        directory with no `runs/` gives exactly the file's numbers."""
        monkeypatch.chdir(tmp_path)
        weights = load_weights("v12")
        for name, config in _raw("v12")["components"].items():
            assert weights.curves[name].points == {
                k: float(v) for k, v in config["points"].items()
            }, name
        assert not (tmp_path / "runs").exists()


class TestParseWeightsChecksSources:
    @staticmethod
    def _data() -> dict[str, Any]:
        return copy.deepcopy(_raw("v12"))

    def test_v12_parses(self) -> None:
        assert parse_weights(self._data()).version == "v12"

    def test_some_curves_sourced_and_some_not_is_refused(self) -> None:
        data = self._data()
        del data["components"]["stall_rate"]["sources"]
        with pytest.raises(WeightsError, match="source for some curves"):
            parse_weights(data)

    def test_an_end_with_no_source_is_refused(self) -> None:
        data = self._data()
        del data["components"]["stall_rate"]["sources"]["zero_at"]
        with pytest.raises(WeightsError, match="one source for each"):
            parse_weights(data)

    def test_an_unknown_kind_is_refused(self) -> None:
        data = self._data()
        data["components"]["stall_rate"]["sources"]["zero_at"]["source"] = "band"
        with pytest.raises(WeightsError, match="unknown source"):
            parse_weights(data)

    def test_an_end_with_no_sentence_is_refused(self) -> None:
        data = self._data()
        data["components"]["stall_rate"]["sources"]["zero_at"]["why"] = " "
        with pytest.raises(WeightsError, match="no sentence"):
            parse_weights(data)

    def test_a_gate_end_off_its_gate_is_refused(self) -> None:
        data = self._data()
        data["components"]["stall_rate"]["points"]["zero_at"] = 0.65
        with pytest.raises(WeightsError, match="no threshold at that value"):
            parse_weights(data)

    def test_a_gate_end_naming_no_gate_is_refused(self) -> None:
        data = self._data()
        data["components"]["stall_rate"]["sources"]["zero_at"]["gate"] = "stall_rate"
        with pytest.raises(WeightsError, match="no threshold at that value"):
            parse_weights(data)


class TestTheBandChecksTheScale:
    @staticmethod
    def _band() -> list[Measured]:
        ladder = {"ladder_monotonicity": 1.0}
        return [
            Measured("low", "human", {**ladder, "ply_spread": 0.1, "stall_rate": 0.8}),
            Measured("mid", "human", {**ladder, "ply_spread": 0.5, "stall_rate": 0.1}),
            Measured("top", "human", {**ladder, "ply_spread": 1.2, "stall_rate": 0.0}),
            Measured("dull", "control", {**ladder, "ply_spread": 0.0, "stall_rate": 1.0}),
            Measured("blind", "human", {"ladder_monotonicity": 0.2, "ply_spread": 2.0}),
        ]

    def test_it_reports_plateau_zero_and_ends_passed(self) -> None:
        checks = {c.component: c for c in check_ends(self._band(), load_weights("v12"))}
        spread = checks["ply_spread"]
        assert spread.games == 3 and spread.unreadable == ["blind"]
        assert spread.plateau == 1 and spread.zero == 0
        assert spread.past == {"zero_at": [], "good_at": ["top"]}
        assert spread.sources == {"zero_at": "meaning", "good_at": "reference"}
        stall = checks["stall_rate"]
        assert stall.zero == 1 and stall.past == {"good_at": [], "zero_at": ["low"]}

    def test_quantiles_are_the_kth_from_each_end(self) -> None:
        values = [float(v) for v in range(1, 21)]
        assert band_quantiles(values) == (2.0, 19.0)
        assert band_quantiles([5.0, 1.0, 3.0]) == (1.0, 5.0)
        assert band_quantiles([]) is None

    def test_anchor_writes_nothing_and_has_no_write(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            cli.main(["anchor", str(tmp_path), "--write"])
        capsys.readouterr()
        for name, total, raw in (("ref-a", 0.8, 0.5), ("ref-b", 0.7, 1.4)):
            run = tmp_path / name
            run.mkdir()
            (run / "index-v11.json").write_text(json.dumps({
                "ruleset_name": name, "total": total, "viable": True,
                "weights_version": "v11",
                "raw": {"ladder_monotonicity": 1.0, "ply_spread": raw},
            }))
        before = {p.name: p.read_bytes() for p in WEIGHTS_DIR.glob("*.json")}
        assert cli.main(["anchor", str(tmp_path), "--json"]) == 0
        report = {c["component"]: c for c in json.loads(capsys.readouterr().out)}
        assert report["ply_spread"]["past"]["good_at"] == ["ref-b"]
        assert {p.name: p.read_bytes() for p in WEIGHTS_DIR.glob("*.json")} == before

    def test_a_library_run_is_read_once(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """In `runs/`, a `<hash>-<seed>` run beside its `<hash>` measurement is not a game."""
        for name in ("abc", "abc-0", "abc-1000"):
            run = tmp_path / name
            run.mkdir()
            (run / "index-v12.json").write_text(json.dumps({
                "ruleset_name": "a game", "total": 0.8, "viable": True,
                "weights_version": "v12",
                "raw": {"ladder_monotonicity": 1.0, "ply_spread": 0.5},
            }))
        (tmp_path / "abc" / "canonical.json").write_text("{}")
        assert cli.main(["anchor", str(tmp_path), "--json"]) == 0
        report = {c["component"]: c for c in json.loads(capsys.readouterr().out)}
        assert report["ply_spread"]["games"] == 1
