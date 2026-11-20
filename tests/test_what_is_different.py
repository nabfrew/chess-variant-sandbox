"""R-W2b2 (`docs/plan-website.md` *What the play page lists first*): what is different, first.

A game's detail carries `differentLines` (its `summary`, else lines derived from its own rule
lines and `relate`), `differentFrom` (the reference game's name, read from its spec) and
`ruleLines` with the game's own lines first (`ruleLinesOwn` of them). The game's own lines
are a plain difference of `describe_rules` against the reference's, with the reference's
clocks put back, so nothing here names a rule. The games are read from their files; queen
bomb is the plan's named case.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from page_text import PAGE

from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.spec.describe import describe_rules
from chess_variant_sandbox.web import library

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "rulesets" / library.REFERENCE_FILE


def _spec(relative: str) -> dict[str, Any]:
    spec = load_spec(ROOT / relative)
    assert isinstance(spec, dict)
    return spec


def _game(spec: dict[str, Any], where: str = "rulesets") -> library.Game:
    return library.Game(hash=spec_hash(spec), name=spec["name"],
                        description=spec.get("description", ""), path=ROOT / where,
                        source="committed", spec=spec)


def _reference_game() -> library.Game:
    spec = _spec(f"rulesets/{library.REFERENCE_FILE}")
    return library.Game(hash=spec_hash(spec), name=spec["name"],
                        description=spec.get("description", ""), path=REFERENCE,
                        source="committed", spec=spec)


def _without_summary(spec: dict[str, Any]) -> dict[str, Any]:
    bare = copy.deepcopy(spec)
    bare.pop("summary", None)
    return bare


class TestTheRuleLinesLeadWithTheGamesOwn:
    def test_queen_bombs_first_rule_line_is_the_blasts(self) -> None:
        spec = _spec("archive/queen-bomb.json")
        said = library.detail(_game(spec, "archive"), [_reference_game()])
        # The blast's line is the one rule line queen bomb gives and the reference does not.
        reference = set(describe_rules(_spec(f"rulesets/{library.REFERENCE_FILE}")))
        (blast,) = [line for line in describe_rules(spec) if line not in reference]
        assert "destroys" in blast
        assert said["ruleLines"][0] == blast
        assert said["ruleLinesOwn"] == 1
        assert sorted(said["ruleLines"]) == sorted(describe_rules(spec)), "reordered, not cut"

    def test_a_clock_is_not_a_rule_of_the_games_own(self) -> None:
        # Gardner differs in its ply limit's line alone; with the reference's clocks put back
        # that line is gone, so none of its lines leads.
        spec = _spec("rulesets/gardner.json")
        reference = _spec(f"rulesets/{library.REFERENCE_FILE}")
        assert set(describe_rules(spec)) - set(describe_rules(reference)), "it differs somewhere"
        assert library.own_rule_lines(spec, reference) == []

    def test_the_reference_itself_has_none_and_differs_from_nothing(self) -> None:
        said = library.detail(_reference_game(), [_reference_game()])
        assert said["ruleLinesOwn"] == 0
        assert said["differentLines"] == [] and said["differentFrom"] is None


class TestWhatIsDifferent:
    def test_a_game_with_a_summary_leads_with_it_and_names_the_reference(self) -> None:
        spec = _spec("archive/queen-bomb.json")
        said = library.detail(_game(spec, "archive"), [_reference_game()])
        assert said["differentLines"] == spec["summary"]
        assert said["differentFrom"] == _spec(f"rulesets/{library.REFERENCE_FILE}")["name"]

    def test_a_spec_with_no_summary_still_gets_lines_from_its_rules(self) -> None:
        spec = _without_summary(_spec("archive/queen-bomb.json"))
        said = library.detail(_game(spec, "archive"), [_reference_game()])
        assert said["differentLines"], "derived, never empty"
        assert said["differentLines"][0] == said["ruleLines"][0], "its own rule line first"

    def test_a_spec_with_no_summary_whose_difference_is_a_piece_gets_relates_line(self) -> None:
        # Berolina's difference is a piece's move, which no rule line says: the line is
        # `relate`'s sentence about the piece, as a sentence.
        spec = _without_summary(_spec("rulesets/berolina.json"))
        reference = _spec(f"rulesets/{library.REFERENCE_FILE}")
        assert library.own_rule_lines(spec, reference) == []
        lines = library.different_lines(spec, reference)
        assert lines and all(line[:1].isupper() and line.endswith(".") for line in lines)

    def test_a_draft_asked_without_the_library_reads_the_checkouts_reference(self) -> None:
        spec = _spec("archive/queen-bomb.json")
        said = library.detail(_game(spec, "archive"))
        assert said["differentFrom"] and said["ruleLinesOwn"] == 1


class TestThePlayPage:
    def test_the_rules_tab_has_its_two_headings_from_the_payload(self) -> None:
        assert "What is different from ${esc(game.differentFrom)}" in PAGE
        assert "As in ${esc(game.differentFrom" in PAGE
        assert "game.ruleLinesOwn" in PAGE

    def test_the_line_above_the_board_shows_until_the_first_move(self) -> None:
        assert 'id="different" data-frame="different"' in PAGE
        assert "(game.differentLines || [])[0]" in PAGE
        assert "!(SESSION.history || []).length && !AT && !SESSION.outcome" in PAGE
        assert "data-all-rules" in PAGE

    def test_the_panel_opens_on_the_rules_then_turns_to_the_moves(self) -> None:
        assert "TAB = opensOnRules(snapshot, how) ? 'rules' : 'moves';" in PAGE
        assert "if (moved) turnToMoves();" in PAGE
        assert "OPEN_TAB" not in PAGE, "K3's first-game switch is folded into opensOnRules"
