"""`metrics.index` pinned on hand-built measurements and weights (Q1145).

Every weights version here is made up inline, with identity curves (``higher_better``
from 0 to 1), so a component's score is its raw value and every expected number below is
arithmetic a reader can redo by hand. Raw values carry six decimals so that rounding to
four is told apart from rounding to three or five.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chess_variant_sandbox.metrics import index
from chess_variant_sandbox.metrics.components import Components
from chess_variant_sandbox.metrics.index import IndexResult, score_components
from chess_variant_sandbox.metrics.scoring import Curve, Gate, Weights

_IDENTITY = Curve(kind="higher_better", points={"zero_at": 0.0, "good_at": 1.0})


def _weights(
    weights: dict[str, float],
    families: dict[str, str],
    gates: tuple[Gate, ...] = (),
    optional: frozenset[str] = frozenset(),
    llm_weight: float = 0.0,
) -> Weights:
    return Weights(
        version="vt",
        families=families,
        weights=weights,
        curves={name: _IDENTITY for name in weights},
        gates=gates,
        llm_weight=llm_weight,
        optional=optional,
    )


# a and b scored; c optional and in family f1 beside a.
_ABC = _weights(
    {"a": 1.0, "b": 2.0, "c": 1.0},
    {"a": "f1", "b": "f2", "c": "f1"},
    optional=frozenset({"c"}),
    llm_weight=0.25,
)


class TestScoreComponents:
    def test_a_scored_result_pinned_whole(self) -> None:
        components = Components(raw={"a": 0.123456, "b": 0.654321, "x": 0.987654})
        result = score_components("h", "g", components, _ABC, llm_score=0.5, llm_notes="n")
        # total: families f1 = 0.123456 and f2 = 0.654321, family weights 2 and 2, so the
        # geometric mean is sqrt(0.123456 * 0.654321) = 0.284218.
        # contributions over the scored weight 3; arithmetic (a + 2b) / 3 = 0.477366;
        # combined 0.75 * 0.284218 + 0.25 * 0.5 = 0.338163.
        assert result.as_dict() == {
            "ruleset_hash": "h",
            "ruleset_name": "g",
            "weights_version": "vt",
            "total": 0.2842,
            "viable": True,
            "violations": [],
            "scores": {"a": 0.1235, "b": 0.6543},
            "contributions": {"a": 0.0412, "b": 0.4362},
            "raw": {"a": 0.1235, "b": 0.6543, "x": 0.9877},
            "families": {"f1": 0.1235, "f2": 0.6543},
            "diagnostics": {"arithmetic_total": 0.4774, "unmeasured": ["c"]},
            "unreadable": (),
            "llm_score": 0.5,
            "llm_notes": "n",
            "combined": 0.3382,
            "interval": None,
            "measured": True,
        }

    def test_a_measured_optional_component_is_not_reported_unmeasured(self) -> None:
        components = Components(raw={"a": 0.2, "b": 0.4, "c": 0.6})
        result = score_components("h", "g", components, _ABC)
        assert "unmeasured" not in result.diagnostics
        assert result.combined is None

    def test_an_absent_component_with_a_failed_gate_is_a_refusal_pinned_whole(self) -> None:
        weights = _weights(
            {"a": 1.0, "d": 1.0},
            {"a": "f1", "d": "f2"},
            gates=(Gate(component="a", minimum=0.5, reason="r"),),
        )
        components = Components(raw={"a": 0.123456, "x": 0.987654})
        result = score_components("h", "g", components, weights)
        assert result.as_dict() == {
            "ruleset_hash": "h",
            "ruleset_name": "g",
            "weights_version": "vt",
            "total": 0.0,
            "viable": False,
            "violations": [
                "a=0.123 below minimum 0.5 (r)",
                "could not be measured on this corpus: d",
            ],
            "scores": {"a": 0.1235},
            "contributions": {},
            "raw": {"a": 0.1235, "x": 0.9877},
            "families": {},
            "diagnostics": {},
            "unreadable": (),
            "llm_score": None,
            "llm_notes": "",
            "combined": None,
            "interval": None,
            "measured": True,
        }

    def test_a_zero_gradient_is_not_inverted(self) -> None:
        weights = _weights({"skill_gradient": 1.0, "a": 1.0}, {"skill_gradient": "f1", "a": "f2"})
        components = Components(raw={"skill_gradient": 0.0, "a": 0.5, "ladder_monotonicity": 0.9})
        result = score_components("h", "g", components, weights)
        assert result.unreadable == ()
        assert result.scored


class TestUnreadable:
    _W = _weights({"skill_gradient": 1.0, "a": 1.0}, {"skill_gradient": "f1", "a": "f2"})

    def test_an_inverted_ladder_at_the_monotonicity_line_pinned_whole(self) -> None:
        # ladder_monotonicity exactly at MEASURABLE_LADDER (0.5) is not flat, so the reason
        # given is the inverted gradient.
        components = Components(
            raw={"skill_gradient": -0.3, "a": 0.123456, "ladder_monotonicity": 0.5,
                 "x": 0.987654}
        )
        result = score_components("h", "g", components, self._W)
        assert result.as_dict() == {
            "ruleset_hash": "h",
            "ruleset_name": "g",
            "weights_version": "vt",
            "total": 0.0,
            "viable": False,
            "violations": [
                "this roster cannot read this ruleset: a larger search rates *worse* here "
                "(-0.3 Elo per doubling), so skill_gradient would describe the players "
                "rather than the game"
            ],
            "scores": {"skill_gradient": 0.0, "a": 0.1235},
            "contributions": {},
            "raw": {"skill_gradient": -0.3, "a": 0.1235, "ladder_monotonicity": 0.5,
                    "x": 0.9877},
            "families": {},
            "diagnostics": {},
            "unreadable": ("skill_gradient",),
            "llm_score": None,
            "llm_notes": "",
            "combined": None,
            "interval": None,
            "measured": False,
        }

    def test_an_absent_monotonicity_never_prints_a_negative_share(self) -> None:
        components = Components(raw={"skill_gradient": -0.3, "a": 0.5})
        result = score_components("h", "g", components, self._W)
        (sentence,) = result.violations
        assert "-100%" not in sentence


def _result(total: float = 0.5, interval: dict[str, Any] | None = None,
            scores: dict[str, float] | None = None) -> IndexResult:
    return IndexResult(
        ruleset_hash="h", ruleset_name="g", weights_version="vt", total=total,
        viable=True, violations=[], scores=scores or {}, contributions={"a": 1.0},
        raw={}, families={}, interval=interval,
    )


def _band(low: float, high: float, **extra: float) -> dict[str, Any]:
    return {"sd": 0.01, "ci_low": low, "ci_high": high, **extra}


class TestIndexResult:
    def test_sd_without_an_interval_is_zero(self) -> None:
        assert _result().sd == 0.0
        assert _result(interval=_band(0.4, 0.6)).sd == 0.01

    def test_without_intervals_separation_is_strictly_the_headline(self) -> None:
        assert _result(0.6).separated_from(_result(0.5)) is True
        assert _result(0.5).separated_from(_result(0.5)) is False

    def test_one_interval_missing_reads_the_headline(self) -> None:
        mine = _result(0.6, _band(0.55, 0.65))
        assert mine.separated_from(_result(0.5)) is True

    def test_separated_intervals_without_sweep_counts_separate(self) -> None:
        mine = _result(0.6, _band(0.55, 0.65))
        theirs = _result(0.5, _band(0.45, 0.54))
        assert mine.separated_from(theirs) is True

    def test_sweeps_at_the_warning_line_refuse_either_side(self) -> None:
        mine = _result(0.6, _band(0.55, 0.65, deterministic_pairings=0.8))
        theirs = _result(0.5, _band(0.45, 0.54))
        assert mine.separated_from(theirs) is False
        assert _result(0.6, _band(0.55, 0.65)).separated_from(
            _result(0.5, _band(0.45, 0.54, deterministic_pairings=0.8))
        ) is False
        # Just under the line separates.
        just_under = _result(0.6, _band(0.55, 0.65, deterministic_pairings=0.79))
        assert just_under.separated_from(theirs) is True

    def test_sweeps_refuse_even_a_lower_headline(self) -> None:
        mine = _result(0.4, _band(0.55, 0.65, deterministic_pairings=0.9))
        assert mine.separated_from(_result(0.5, _band(0.45, 0.54))) is False

    def test_equal_headlines_never_separate(self) -> None:
        mine = _result(0.5, _band(0.55, 0.65))
        assert mine.separated_from(_result(0.5, _band(0.45, 0.54))) is False

    def test_weakest_and_strongest_take_three_in_order(self) -> None:
        result = _result(scores={"p": 0.1, "q": 0.5, "r": 0.3, "s": 0.9, "t": 0.7})
        assert result.weakest() == [("p", 0.1), ("r", 0.3), ("q", 0.5)]
        assert result.strongest() == [("s", 0.9), ("t", 0.7), ("q", 0.5)]

    def test_save_makes_its_directories_and_indents_by_two(self, tmp_path: Path) -> None:
        path = _result().save(tmp_path / "one" / "two" / "index.json")
        text = path.read_text(encoding="utf-8")
        assert text.startswith('{\n  "ruleset_hash": "h",\n  "ruleset_name": "g",')
        assert json.loads(text)["total"] == 0.5


def test_score_ruleset_names_the_result_after_the_first_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        index, "compute_components",
        lambda *args, **_kwargs: Components(raw={"a": 0.2, "b": 0.4}),
    )
    records: Any = [
        SimpleNamespace(ruleset_hash=f"h{i}", ruleset_name=f"n{i}") for i in range(3)
    ]
    result = index.score_ruleset({}, records, ratings=None, weights=_ABC,  # type: ignore[arg-type]
                                 interval=False)
    assert (result.ruleset_hash, result.ruleset_name) == ("h0", "n0")
    named = index.score_ruleset({"name": "spec"}, records, ratings=None,  # type: ignore[arg-type]
                                weights=_ABC, interval=False)
    assert named.ruleset_name == "spec"
