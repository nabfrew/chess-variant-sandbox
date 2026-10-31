"""A game's short description (Q1085a): the spec's `summary` and `provenance.note`.

The user (2026-11-01): *"The descriptions are generally too wordy."* A game's page now leads
with a few plain lines saying what makes the game itself, then one line on where it came
from where that is of interest, then the full rules folded shut. The lines are text in the
spec, outside the mechanics like `description`: writing one moves no game's name, the
compiler and Fairy-Stockfish's translation never read it, and the validator only checks it
is a few short lines.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.spec.schema import (
    SUMMARY_MAX_LINE,
    SUMMARY_MAX_LINES,
    provenance_note,
    summary_lines,
    validate_spec,
)
from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.page import game as drawing
from tests.page_text import PAGE

ROOT = Path(__file__).resolve().parent.parent
# The pattern games Q1085a wrote, one of each kind of rule (Q1085b wrote the rest).
PATTERN = (
    "archive/the-free-companies.json", "rulesets/duck-chess.json",
    "rulesets/rock-paper-shove.json", "rulesets/seirawan.json", "rulesets/crazyhouse.json",
)


def _load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _without_text(spec: dict[str, Any]) -> dict[str, Any]:
    bare = copy.deepcopy(spec)
    bare.pop("summary", None)
    if isinstance(bare.get("provenance"), dict):
        bare["provenance"].pop("note", None)
    return bare


def _every_spec() -> list[Path]:
    return sorted((ROOT / "rulesets").glob("*.json")) + sorted((ROOT / "archive").glob("*.json"))


class TestTheTextMovesNothing:
    @pytest.mark.parametrize("relative", PATTERN)
    def test_a_pattern_game_carries_a_summary_and_is_valid(self, relative: str) -> None:
        spec = _load(relative)
        assert summary_lines(spec), relative
        assert validate_spec(spec) == []

    @pytest.mark.parametrize("relative", PATTERN)
    def test_the_summary_moves_no_hash(self, relative: str) -> None:
        spec = _load(relative)
        assert spec_hash(spec) == spec_hash(_without_text(spec))

    def test_writing_one_into_a_game_moves_no_hash(self) -> None:
        spec = _load("rulesets/chess.json")
        before = spec_hash(spec)
        spec["summary"] = ["A line.", "Another line."]
        spec.setdefault("provenance", {})["note"] = "Where it came from."
        assert validate_spec(spec) == []
        assert spec_hash(spec) == before

    def test_every_library_summary_is_valid(self) -> None:
        for path in _every_spec():
            spec = json.loads(path.read_text(encoding="utf-8"))
            if "summary" in spec or provenance_note(spec):
                assert validate_spec(spec) == [], path.name

    def test_every_library_game_carries_one_and_it_moves_no_hash(self) -> None:
        """Q1085b: every game under rulesets/ and archive/ leads with a summary."""
        for path in _every_spec():
            spec = json.loads(path.read_text(encoding="utf-8"))
            assert summary_lines(spec), path.name
            assert spec_hash(spec) == spec_hash(_without_text(spec)), path.name

    @pytest.mark.parametrize("relative", ("rulesets/crazyhouse.json", "rulesets/duck-chess.json"))
    def test_the_fairy_translation_ignores_it(self, relative: str) -> None:
        spec = _load(relative)
        try:
            with_text = translate(spec)
        except Untranslatable:
            pytest.skip("this build does not translate the game")
        assert with_text == translate(_without_text(spec))


class TestTheValidatorReadsAFewShortLines:
    @pytest.mark.parametrize("summary, says", [
        ("one string", "list of short lines"),
        ([], "list of short lines"),
        (["fine", ""], "non-empty strings"),
        (["fine", 3], "non-empty strings"),
        (["x"] * (SUMMARY_MAX_LINES + 1), f"at most {SUMMARY_MAX_LINES} lines"),
        (["x" * (SUMMARY_MAX_LINE + 1)], f"at most {SUMMARY_MAX_LINE} characters"),
    ])
    def test_a_wrong_summary_is_refused_by_name(self, summary: Any, says: str) -> None:
        spec = _load("rulesets/chess.json")
        spec["summary"] = summary
        problems = validate_spec(spec)
        assert any(says in problem for problem in problems), problems

    def test_a_note_is_text(self) -> None:
        spec = _load("rulesets/chess.json")
        spec["provenance"] = {"note": 7}
        assert "provenance.note must be a string" in validate_spec(spec)


class TestThePageLeadsWithIt:
    def test_the_detail_carries_the_lines_and_the_note(self) -> None:
        spec = _load("rulesets/duck-chess.json")
        game = library.Game(hash=spec_hash(spec), name=spec["name"],
                            description=spec["description"], path=ROOT / "rulesets",
                            source="committed", spec=spec)
        said = library.detail(game, [])
        assert said["summaryLines"] == spec["summary"]
        assert said["provenanceNote"] == spec["provenance"]["note"]

    def test_a_game_without_one_carries_empty_text(self) -> None:
        # Every library game has one since Q1085b, so chess stands in with its text taken out.
        spec = _without_text(_load("rulesets/chess.json"))
        assert summary_lines(spec) == [] and provenance_note(spec) == ""

    def test_the_page_draws_lead_then_full_rules_then_play(self) -> None:
        """In order, with nothing between: the short description, the note (both in
        `leadHtml`), then the full rules, then the play panel."""
        lead = PAGE.index("${leadHtml(game)}")
        rules = PAGE.index("${fullRulesHtml(game)}")
        play = PAGE.index('<div class="gameTop">', rules)
        assert lead < rules < play
        assert PAGE[lead:play].replace("${leadHtml(game)}", "").replace(
            "${fullRulesHtml(game)}", "").strip() == ""

    def test_the_full_rules_start_folded_on_every_screen(self) -> None:
        start = PAGE.index("function fullRulesHtml")
        body = PAGE[start:PAGE.index("\n}\n", start)]
        assert '<details class="sect fullRules" data-sect="rules">' in body
        assert " open" not in body
        # The section it replaced, far down the page, is gone.
        assert "sectionHtml('rules'" not in PAGE

    def test_the_server_drawn_page_leads_the_same_way(self) -> None:
        spec = _load("archive/the-free-companies.json")
        drawn = drawing.Drawn(hash="f858b7a63dea", name=spec["name"],
                              description=spec["description"], spec=spec, score=None,
                              stale=False, near=(), indexable=True)
        html = drawing.view(drawn)
        first = html.index('<ul class="gameLead">')
        folded = html.index('<details class="sect fullRules">')
        assert first < folded < html.index('class="drawnScore"')
        assert "Hire the mercenaries in the middle." in html[first:folded]
        assert "<details class=\"sect fullRules\" open" not in html
        # The search result's line reads the first line, not the longer description.
        assert "Hire the mercenaries in the middle." in drawing.meta_description(drawn)

    def test_the_server_drawn_page_without_one_is_as_it_was(self) -> None:
        spec = _without_text(_load("rulesets/chess.json"))
        drawn = drawing.Drawn(hash="831c77e75650", name=spec["name"],
                              description=spec["description"], spec=spec, score=None,
                              stale=False, near=(), indexable=True)
        html = drawing.view(drawn)
        assert "gameLead" not in html and "fullRules" not in html
        assert '<p class="gameAbout">' in html
