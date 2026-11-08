"""`v13`: `v12` on the shifted measuring ladder, every reference end re-read and nothing else moved.

`docs/plan-ladder-shift.md` and `docs/reference/v13-calibration.md` (Q1102's pre-registration,
Q1104's read). Each assertion is a fact about the files: a later edit that moved an end without
a new version, or moved anything but a reference end from `v12`, fails here.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from chess_variant_sandbox.metrics import load_weights
from chess_variant_sandbox.metrics.scoring import DEFAULT_WEIGHTS, WEIGHTS_DIR

# The read (`v13-calibration.md`, *Phase C*), pinned.
READ = {
    ("mean_branching", "good_low"): 8.9219,
    ("mean_branching", "good_high"): 57.5673,
    ("mean_branching", "zero_high"): 115.1346,
    ("ply_spread", "good_at"): 0.9261,
    ("decision_criticality", "good_low"): 0.353,
    ("decision_criticality", "good_high"): 0.6649,
    ("piece_utilisation", "zero_at"): 0.6885,
    ("skill_gradient", "good_at"): 155.58,
    ("instance_participation", "good_low"): 0.5137,
    ("instance_participation", "good_high"): 0.9139,
    ("lead_conversion", "good_low"): 0.567,
    ("lead_conversion", "good_high"): 0.9191,
}


def _raw(version: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((WEIGHTS_DIR / f"{version}.json").read_text(encoding="utf-8"))
    return data


def test_it_is_the_default() -> None:
    assert DEFAULT_WEIGHTS == "v13"


def test_only_the_reference_ends_moved_from_v12() -> None:
    twelve, thirteen = load_weights("v12"), load_weights("v13")
    assert thirteen.components() == twelve.components()
    assert thirteen.weights == twelve.weights
    assert thirteen.families == twelve.families
    assert thirteen.gates == twelve.gates
    assert thirteen.llm_weight == twelve.llm_weight
    assert thirteen.optional == twelve.optional
    assert thirteen.sources == twelve.sources
    for name, curve in thirteen.curves.items():
        assert curve.kind == twelve.curves[name].kind, name
        for end, value in curve.points.items():
            if thirteen.sources[name][end] != "reference":
                assert value == twelve.curves[name].points[end], (name, end)


def test_the_read_is_pinned() -> None:
    weights = load_weights("v13")
    reference = {
        (name, end) for name, kinds in weights.sources.items()
        for end, kind in kinds.items() if kind == "reference"
    }
    assert reference == set(READ)
    for (name, end), value in READ.items():
        assert weights.curves[name].points[end] == value, (name, end)


def test_every_reference_end_names_its_game_and_the_record() -> None:
    for name, config in _raw("v13")["components"].items():
        for end, source in config["sources"].items():
            if source["source"] == "reference" and end != "zero_high":
                assert "v13-calibration.md" in source["why"], (name, end)


def test_derived_reference_ends_follow_their_rule() -> None:
    points = {n: c.points for n, c in load_weights("v13").curves.items()}
    branching = points["mean_branching"]
    assert branching["zero_high"] == pytest.approx(2 * branching["good_high"], abs=1e-9)
    # The low quantile less the distance between the quantiles: 0.8418 - (0.9951 - 0.8418).
    assert points["piece_utilisation"]["zero_at"] == pytest.approx(
        0.8418 - (0.9951 - 0.8418), abs=5e-5
    )
