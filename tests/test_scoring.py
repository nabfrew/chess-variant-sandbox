"""The scoring configuration's arithmetic, gates and parse refusals, on weights built inline."""

from __future__ import annotations

import dataclasses
import math

import pytest

from chess_variant_sandbox.metrics.scoring import (
    Curve,
    Gate,
    Weights,
    WeightsError,
    _ramp_down,
    _ramp_up,
    load_weights,
    parse_weights,
)

STEP = Curve(kind="higher_better", points={"zero_at": 0.0, "good_at": 1.0})


def _weights(weights: dict[str, float], gates: tuple[Gate, ...] = ()) -> Weights:
    return Weights(
        version="t",
        families={name: f"family-{name}" for name in weights},
        weights=weights,
        curves={name: STEP for name in weights},
        gates=gates,
        llm_weight=0.0,
    )


def _component(**extra: object) -> dict[str, object]:
    return {"curve": "higher_better", "points": {"zero_at": 0, "good_at": 1}, **extra}


def test_curve_gate_and_weights_are_immutable() -> None:
    # Load-bearing: a loaded weights version is shared and names a leaderboard; nothing
    # downstream may edit one in place.
    with pytest.raises(dataclasses.FrozenInstanceError):
        STEP.kind = "lower_better"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        Gate(component="a").minimum = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        _weights({"a": 1.0}).llm_weight = 0.5  # type: ignore[misc]


def test_gate_bounds_are_inclusive() -> None:
    gate = Gate(component="x", minimum=0.2, maximum=0.8, reason="r")
    assert gate.check(0.2) is None
    assert gate.check(0.8) is None
    assert gate.check(0.19) == "x=0.190 below minimum 0.2 (r)"
    assert gate.check(0.81) == "x=0.810 above maximum 0.8 (r)"


def test_combine_and_arithmetic_total_on_unequal_weights() -> None:
    weights = _weights({"a": 1.0, "b": 3.0})
    total, contributions = weights.combine({"a": 0.5, "b": 1.0})
    assert total == pytest.approx(0.5**0.25)
    assert contributions == {"a": 0.125, "b": 0.75}
    assert weights.arithmetic_total({"a": 0.5, "b": 1.0}) == 0.875


def test_arithmetic_total_of_nothing_is_zero_and_of_a_light_weight_is_its_score() -> None:
    assert _weights({"a": 1.0}).arithmetic_total({}) == 0.0
    assert _weights({"a": 0.5}).arithmetic_total({"a": 0.8}) == 0.8


def test_check_gates_reads_every_gate_after_an_unmeasured_one() -> None:
    gates = (Gate(component="unmeasured", minimum=0.0), Gate(component="a", minimum=0.9, reason="r"))
    assert _weights({"a": 1.0}, gates).check_gates({"a": 0.5}) == [
        "unmeasured was not measured",
        "a=0.500 below minimum 0.9 (r)",
    ]


def test_load_weights_names_the_versions_it_has() -> None:
    with pytest.raises(WeightsError, match=r"no weights version 'no-such'; available: .*\bv12\b"):
        load_weights("no-such")


def test_parse_weights_refusals() -> None:
    with pytest.raises(WeightsError, match="declare no components"):
        parse_weights({"version": "t", "components": {}})
    with pytest.raises(WeightsError, match="must carry a positive weight"):
        parse_weights({"version": "t", "components": {"a": _component()}})
    for bad in (-0.5, 1.0, 1.5):
        with pytest.raises(WeightsError, match="llm_weight must be at least 0 and below 1"):
            parse_weights({"components": {"a": _component(weight=1)}, "llm_weight": bad})
    assert parse_weights({"components": {"a": _component(weight=1)}, "llm_weight": 0.0}).llm_weight == 0.0


def test_ramps_with_coincident_ends_are_steps() -> None:
    # A curve's points may coincide (_validate_points allows equal neighbours): a step.
    assert [_ramp_up(v, 3.0, 3.0) for v in (2.0, 3.0, 4.0)] == [0.0, 1.0, 1.0]
    assert [_ramp_down(v, 3.0, 3.0) for v in (2.0, 3.0, 4.0)] == [1.0, 1.0, 0.0]
    assert math.isclose(_ramp_up(2.5, 2.0, 3.0), 0.5)
