"""The web glossary's prose does not recompute anything, so nothing catches a stale
number in it except a test that reads the actual words.

Q486: `HOW_IT_WORKS` quoted `v10`'s calibration band (43 adjacent pairs, zero
separating) as if it were current, well after `v11` and the 2026-09-26 replay moved
those figures. This pins the current reading (`docs/reference/batch-0926-replay.md`,
*Calibration, re-read (Q478)*: 4 of 59 adjacent pairs separate) so a future replay that
moves the figure again has to touch this file rather than leave the page lying.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.metrics import scoring
from chess_variant_sandbox.metrics.glossary import (
    HOW_IT_WORKS,
    MEASUREMENTS,
    Measurement,
    _roster_rho,
    explain,
)


def _body(heading: str) -> str:
    for entry in HOW_IT_WORKS:
        if entry["heading"] == heading:
            return entry["body"]
    raise AssertionError(f"no HOW_IT_WORKS entry titled {heading!r}")


class TestCalibrationFigures:
    def test_adjacent_pairs_figure_is_the_current_one(self) -> None:
        body = _body("The interval is wider than the gaps, not most of them")
        assert "59 adjacent" in body
        assert "4 of" in body or "only 4" in body

    def test_stale_v10_figures_are_gone(self) -> None:
        # v10's band (44 members, 43 adjacent pairs, none separating) is not v11's, in
        # this entry or in any other: the stale figure was quoted wherever the band was.
        for entry in HOW_IT_WORKS:
            body = entry["body"]
            assert "43 adjacent" not in body, entry["heading"]
            assert "44 members" not in body, entry["heading"]
        assert "is zero" not in _body("The interval is wider than the gaps, not most of them")

    def test_figure_is_dated(self) -> None:
        # A number this volatile (it moves with every replay) is only honest if it
        # says when it was read, per the CLAUDE.md instruction to date a claim that
        # cannot practically be kept live.
        body = _body("The interval is wider than the gaps, not most of them")
        assert "2026-09-26" in body


class TestTheSitesWords:
    """W1 (`docs/archive/plan-understanding.md` §3.3, A18-A22): the site's words for the score's
    parts and its account of the method, beside the maintainer's, which stay as written."""

    # The maintainer's terms a visitor's line must not carry (A20, A22, §3.2).
    MAINTAINERS = ("analysis pass", "bootstrap", "full window", "built-in evaluator",
                   "fitted objective", "entropy", "forced blunder", "sampling sd", "mutant",
                   "corpus", "roster", "ruleset", "elo", "plies", "gate", "ladder", "index",
                   "harness", "seat", "weighted geometric mean", "pooled", "half-width",
                   "re-anchor", "weights", "control", "budget", "2026-")

    def _clean(self, text: str, where: str) -> None:
        lowered = text.lower()
        for term in self.MAINTAINERS:
            assert term not in lowered, (where, term)

    def test_each_entry_names_a_measurement_and_only_words(self) -> None:
        from chess_variant_sandbox.metrics.glossary import MEASUREMENTS, SITE_WORDS

        assert set(SITE_WORDS) <= set(MEASUREMENTS)
        for name, words in SITE_WORDS.items():
            assert words and set(words) <= {"title", "what", "reading", "units"}, name
            for field, text in words.items():
                assert text != getattr(MEASUREMENTS[name], field), (name, field)
                self._clean(text, f"{name}.{field}")

    def test_every_scored_part_has_a_visitors_reading_and_title(self) -> None:
        """Under the current scoring rules, what the site draws for each part: its title,
        what it is and how to read it, merged as the page merges them."""
        from chess_variant_sandbox.metrics.glossary import explain

        guide = explain()
        for name, entry in guide["components"].items():
            shown = {**entry, **entry["site"]}
            for field in ("title", "what", "reading", "units"):
                self._clean(str(shown.get(field, "")), f"{name}.{field}")
            if entry["curve"]["kind"] == "band":
                # A band has two bad ends, and the visitor's line says so too.
                reading = shown["reading"].lower()
                assert "middle" in reading or "too" in reading, name
        components = guide["components"]
        assert components["skill_gradient"]["site"]["title"] == "does thinking longer win"
        assert components["stall_rate"]["site"]["title"] == "games that never finish"

    def test_the_method_is_told_twice_in_one_order(self) -> None:
        from chess_variant_sandbox.metrics.glossary import (
            HOW_IT_WORKS,
            SITE_HOW_IT_WORKS,
            explain,
        )

        assert len(SITE_HOW_IT_WORKS) == len(HOW_IT_WORKS)
        for entry in SITE_HOW_IT_WORKS:
            assert set(entry) == {"heading", "body"}
            self._clean(entry["heading"] + " " + entry["body"], entry["heading"])
        said = " ".join(entry["body"] for entry in SITE_HOW_IT_WORKS)
        assert "multiplied together" in said and "deliberately dull test game" in said
        assert "never as a place" in said and "five times in six" in said
        assert "scores under different versions are never compared" in said
        assert explain()["howItWorksSite"] == SITE_HOW_IT_WORKS

    def test_the_families_say_sides_and_no_harness(self) -> None:
        from chess_variant_sandbox.metrics.glossary import FAMILIES

        assert "two sides" in FAMILIES["balance"]
        assert "harness" not in FAMILIES["depth"]


def _write(directory: Path, version: str, data: Any) -> None:
    text = data if isinstance(data, str) else json.dumps(data)
    (directory / f"{version}.json").write_text(text, encoding="utf-8")


class TestExplainOnAMadeUpVersion:
    """`explain` and `_roster_rho` against a weights directory written inline (Q1145)."""

    def test_weights_are_shown_to_four_places(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(scoring, "WEIGHTS_DIR", tmp_path)
        curve = {"curve": "higher_better", "points": {"zero_at": 0.0, "good_at": 1.0}}
        _write(tmp_path, "vt", {"version": "vt", "components": {
            "p_one": {**curve, "weight": 0.123456, "family": "f1", "roster_rho": 0.7},
            "p_two": {**curve, "weight": 0.654321, "family": "f2"},
        }})
        guide = explain("vt")
        assert guide["families"] == {
            "f1": {"means": "", "weight": 0.1235},
            "f2": {"means": "", "weight": 0.6543},
        }
        assert guide["components"]["p_one"]["weight"] == 0.1235
        assert guide["components"]["p_one"]["roster_rho"] == 0.7
        assert guide["components"]["p_two"]["weight"] == 0.6543
        assert "roster_rho" not in guide["components"]["p_two"]

    def test_roster_rho_skips_a_malformed_entry_and_keeps_reading(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(scoring, "WEIGHTS_DIR", tmp_path)
        _write(tmp_path, "vt", {"components": {
            "p": 3, "q": {"roster_rho": 0.4}, "r": {"roster_rho": True},
        }})
        assert _roster_rho("vt") == {"q": 0.4}

    def test_roster_rho_of_a_missing_or_unparseable_file_is_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(scoring, "WEIGHTS_DIR", tmp_path)
        assert _roster_rho("absent") == {}
        _write(tmp_path, "broken", "{not json")
        assert _roster_rho("broken") == {}


def test_a_measurement_is_immutable_and_hashable() -> None:
    # MEASUREMENTS holds one shared instance per component that every `explain` reads;
    # `frozen` is what keeps a caller from editing the glossary in place.
    entry = Measurement(title="t", what="w", how="h", reading="r")
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.title = "changed"  # type: ignore[misc]
    assert hash(entry) == hash(Measurement(title="t", what="w", how="h", reading="r"))
    assert MEASUREMENTS
