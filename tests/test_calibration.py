"""`metrics/calibration.py`: the held-out slice, the report's every field, the curve-end
suggestion, and the domain clamp, pinned to literals built by hand (Q1145).

`tests/test_index_quality.py` covers the behaviour in prose-sized pieces; these pin whole
outputs, so a changed rounding, default, bound or comparison shows as a changed value.
`tests/test_calibration_anchor.py` holds the rest of `anchor_curves` and `check_ends`.
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

from chess_variant_sandbox.metrics.calibration import (
    Measured,
    ScoredEntry,
    _clamp,
    anchor_curves,
    calibrate,
    held_out,
    suggest_curve_ends,
)


def _first_byte(name: str) -> int:
    return hashlib.sha256(f"v8:{name}".encode()).digest()[0]


def test_held_out_boundary_is_the_first_byte_below_a_third_of_256():
    # The first byte b is held out when b / 256 < 1/3, so 85 is in and 86 is out.
    assert (_first_byte("g325"), _first_byte("g154"), _first_byte("g3")) == (85, 84, 86)
    assert held_out("g325") is True
    assert held_out("g154") is True
    assert held_out("g3") is False


def test_scored_entry_is_frozen_and_hashable():
    entry = ScoredEntry(name="a", total=0.5, viable=True, label="human")
    assert hash(entry) == hash(ScoredEntry(name="a", total=0.5, viable=True, label="human"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.total = 0.6  # type: ignore[misc]


def _e(name, total, label, *, viable=True, base="", sd=None, ladder=1.0):
    if sd is None:  # leave the dataclass default in play
        return ScoredEntry(name=name, total=total, viable=viable, label=label, base=base,
                           ladder=ladder)
    return ScoredEntry(name=name, total=total, viable=viable, label=label, base=base,
                       sd=sd, ladder=ladder)


def _corpus_a() -> list[ScoredEntry]:
    return [
        # the band: three measurable, viable humans
        _e("h1", 0.123456, "human"),  # default sd
        _e("h2", 0.654321, "human", sd=0.123456),
        _e("h3", 0.333337, "human", sd=0.2),
        # humans out of the band
        _e("hg", 0.9, "human", viable=False),  # gated
        _e("edge", 0.95, "human", viable=False, ladder=0.5),  # ladder exactly at the bar: gated
        _e("hu", 0.97, "human", ladder=0.4),  # unreadable
        # negatives
        _e("w_gated", 0.9, "worse", viable=False),
        _e("w_floor", 0.123456, "worse"),  # exactly on the floor: misplaced
        _e("w_low", 0.05, "worse"),
        # nulls and their bases
        _e("bA", 1.5, "control", sd=0.25),
        _e("nA", 0.5, "same", base="bA", sd=0.25),  # gap 1.0 == threshold 1.0: stable
        _e("bB", 0.5, "control"),
        _e("nB", 0.53, "same", base="bB"),  # gap 0.03 < 0.04: stable
        _e("bC", 0.5, "control"),
        _e("nC", 0.55, "same", base="bC"),  # gap 0.05 > 0.04: unstable
        _e("bD", 0.5, "control"),  # default sd
        _e("nD", 0.56, "same", base="bD", sd=0.05),  # gap 0.06 < 0.1: stable
        _e("bE", 0.0, "control"),
        _e("nE", 1.234567, "same", base="bE", sd=1.0),  # gap 1.234567 < 2.0: stable
    ]


def test_calibrate_pins_every_field():
    report = calibrate(_corpus_a(), "vt")
    assert report.as_dict() == {
        "weights_version": "vt",
        "separation": 0.0,
        "sign_accuracy": 0.6667,
        "null_stability": 1.2346,
        "human_spread": 0.5309,
        "human_floor": 0.1235,
        "human_ceiling": 0.6543,
        "misplaced": ["w_floor"],
        "unstable_nulls": ["nC"],
        "unmeasurable": ["hu"],
        "gated_humans": ["edge", "hg"],
        "detail": {
            "humans": 3,
            "worse": 3,
            "nulls": 5,
            "gated_out": ["w_gated"],
            "null_gaps": {"nA": 1.0, "nB": 0.03, "nC": 0.05, "nD": 0.06, "nE": 1.2346},
            "human_median": 0.3333,
            "mean_sd": 0.1617,
        },
    }


def test_calibrate_summary_lists_every_group():
    assert calibrate(_corpus_a(), "vt").summary() == "\n".join([
        "calibration of weights vt",
        "  human band      0.123 - 0.654 (spread 0.531; smaller is better)",
        "  separation      +0.000  (human floor minus best known-bad mutant; positive is the point)",
        "  sign accuracy   67% of known-bad mutants below the human floor",
        "  null stability  1.235 worst gap between a null and its base (over 5 nulls)",
        "  known-bad mutants inside the human band:",
        "    - w_floor",
        "  nulls that moved further than the sampling interval:",
        "    - nC",
        "  real games the gates rejected, so out of the band:",
        "    - edge",
        "    - hg",
        "  held out of the band -- this roster cannot play them, so their scores describe the agents:",
        "    - hu",
    ])


def test_calibrate_with_no_negatives_and_no_sampling_interval():
    report = calibrate([
        _e("x", 0.25, "human", sd=0.0),
        _e("y", 0.75, "human", sd=0.0),
        _e("b", 0.4, "control"),
        _e("n", 0.4, "same", base="b"),
    ], "vt")
    # with no negatives the worst case is -1, so separation is the floor plus one
    assert report.separation == 1.25
    assert report.sign_accuracy == 0.0
    assert report.detail["mean_sd"] == 0.0
    assert report.misplaced == []


def test_calibrate_scores_a_gated_negative_at_minus_one():
    report = calibrate([
        _e("x", 0.123456, "human"),
        _e("g", 0.9, "worse", viable=False),
        _e("b", 0.4, "control"),
        _e("n", 0.4, "same", base="b"),
    ], "vt")
    assert report.separation == 1.1235
    assert report.sign_accuracy == 1.0
    assert report.detail["gated_out"] == ["g"]


_HUMANS = [0.876543, 0.2, 0.111111, 0.3, 0.987654, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85]


def test_suggest_curve_ends_default_quantile():
    ends = suggest_curve_ends({"human": _HUMANS, "worse": [0.234567, 0.05, 0.123456]})
    assert ends == {
        "human_low": 0.1111,
        "human_high": 0.9877,
        "human_quantile": 0.8765,  # int(0.9 * 12) = 10th of 0..11
        "negative_high": 0.2346,
        "negative_median": 0.1235,
    }


def test_suggest_curve_ends_quantile_one_is_the_top():
    ends = suggest_curve_ends({"human": _HUMANS, "worse": [0.05]}, quantile=1.0)
    assert ends["human_quantile"] == 0.9877


def test_suggest_curve_ends_without_negatives():
    assert suggest_curve_ends({"human": [0.5]}) == {
        "human_low": 0.5,
        "human_high": 0.5,
        "human_quantile": 0.5,
        "negative_high": 0.0,
        "negative_median": 0.0,
    }
    assert suggest_curve_ends({"worse": [0.1]}) == {}


def test_anchor_curves_noise_floor_is_three_sds():
    # Load-bearing: MIN_SPAN_IN_NOISE sets the span when noise exceeds the band, and
    # anchor_curves re-derives v3-v11's ends with it.
    corpus = [Measured(f"h{i}", "human", {"probe_metric": v})
              for i, v in enumerate([0.4, 0.44, 0.48, 0.52, 0.56, 0.6])]
    points, dropped = anchor_curves(corpus, {"probe_metric": {"curve": "higher_better"}},
                                    noise={"probe_metric": 0.5})
    # good = 0.6 + 0.35 * 0.2 = 0.67; span floor = max(3 * 0.5, 2 * 0.2) = 1.5
    assert points == {"probe_metric": {"good_at": 0.67, "zero_at": -0.83}}
    assert dropped == {}


def test_anchor_curves_degenerate_band_falls_back_to_a_fixed_width():
    corpus = [Measured(f"h{i}", "human", {"flat_metric": 0.5}) for i in range(6)]
    points, _ = anchor_curves(corpus, {"flat_metric": {"curve": "band"}})
    # width 1e-6 rounds away, so both zero ends fall back to 0.05 beyond the good ends
    assert points == {"flat_metric": {
        "good_low": 0.5, "good_high": 0.5, "zero_low": 0.45, "zero_high": 0.55,
    }}


def test_domain_clamps():
    # Load-bearing: DOMAIN bounds every end anchor_curves places (v3-v11 re-derivation).
    got = {name: [_clamp(name, v) for v in (-0.5, 0.5, 1.5)]
           for name in ("rule_yield", "first_player_advantage", "mean_branching",
                        "quick_kill_rate", "ply_spread")}
    assert got == {
        "rule_yield": [0.0, 0.5, 1.5],
        "first_player_advantage": [0.0, 0.5, 1.0],
        "mean_branching": [0.0, 0.5, 1.5],
        "quick_kill_rate": [0.0, 0.5, 1.0],
        "ply_spread": [0.0, 0.5, 1.5],
    }

