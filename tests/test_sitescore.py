"""A score's own sentences in a visitor's words (`web/sitescore.py`; W3, rows G29-G31).

The scoring writes for the person who runs it. Each shape of sentence it writes is held
here to a site sentence that carries the same numbers and none of the maintainer's terms,
so a refusal on a game's page reads *quick kills, worse half came to 0.538* and never
*quick_kill_rate_worst_half=0.538 above maximum 0.5*.
"""

from __future__ import annotations

import copy
import re

import site_words

from chess_variant_sandbox.metrics.glossary import explain
from chess_variant_sandbox.web import library, opponents, sitescore

# One of each shape `metrics/` writes, as stored in a library game's `canonical.json`.
SHAPES = {
    "gate below": "skill_gradient=4.200 below minimum 8.0 (doubling the thinking budget must "
                  "be worth at least a few Elo, or skill does not exist here)",
    "gate above": "quick_kill_rate_worst_half=0.538 above maximum 0.5 (a game where half of "
                  "all contests are over in a handful of moves is decided by an opening trap)",
    "unreadable": "this roster cannot read this ruleset: the rating ladder orders the agents "
                  "on only 40% of its budget steps, so lead_conversion, skill_gradient would "
                  "describe the players rather than the game",
    "unreadable, worse": "this roster cannot read this ruleset: a larger search rates *worse* "
                         "here (-3.0 Elo per doubling), so skill_gradient would describe the "
                         "players rather than the game",
    "absent": "could not be measured on this corpus: blunder_recoverability",
    "unmeasured": "search_advantage was not measured",
    "seat, one run": "top_three_seat_decided_chance=3.13e-04 below minimum 0.0005: the first "
                     "seat went 7-29 with 0 draws over the 36 games between the three "
                     "strongest rungs (800, 1600, 6400 nodes), one chance in 3,199 under a "
                     "fair seat (a game the three strongest rungs win from one seat)",
    "seat, pooled": "seat_decided_chance=5.37e-12 below minimum 0.001: the first seat took "
                    "100% of the 64 decided games among the 64 played between the two "
                    "strongest rungs (1600, 6400 nodes) over 4 runs, which rules out a seat "
                    "held to 67% or less (a game the two strongest rungs win from one seat)",
}

# What a visitor must never read in one of these, beyond the guard's own list.
MAINTAINERS = re.compile(r"_|\brungs?\b|\bseat\b|\bElo\b|\bladder\b|\bbudget|\broster\b|"
                         r"\bnodes\b|\bminimum\b|\bmaximum\b|\bruleset\b", re.I)


def _clean(text: str) -> None:
    assert text and text != sitescore.FAILED, text
    assert not site_words.hits(text, tuple(site_words.PHRASES)), text
    assert not MAINTAINERS.search(text), text


class TestEachRefusalReadsAsAVisitorsSentence:
    def test_every_shape_the_scoring_writes_has_site_words(self) -> None:
        for name, text in SHAPES.items():
            said = sitescore.violation(text)
            _clean(said)
            # A test's measured number survives the rewording. A seat refusal's chance does
            # not: its counts say the same thing to a visitor.
            for number in re.findall(r"=(\d+\.\d+) ", text):
                assert number in said, (name, said)

    def test_the_numbers_and_the_side_are_the_stored_ones(self) -> None:
        said = sitescore.violation(SHAPES["seat, one run"])
        assert "second side" in said and "29 games to 7" in said
        said = sitescore.violation(SHAPES["gate below"])
        assert said.startswith("does thinking longer win came to 4.200 rating points")
        assert "under the 8.0 rating points it must reach" in said
        said = sitescore.violation(SHAPES["unreadable"])
        assert "two of the things measured" in said and "this run has no score" in said
        assert "did worse with more thinking" in sitescore.violation(SHAPES["unreadable, worse"])

    def test_a_pooled_refusal_is_named_by_its_place_among_the_runs(self) -> None:
        text = "run 2000: " + SHAPES["gate above"]
        said = sitescore.violation(text, [0, 1000, 2000, 3000])
        assert said.startswith("run 3 of 4: quick kills, worse half came to 0.538")

    def test_a_sentence_nobody_has_seen_is_never_passed_through(self) -> None:
        assert sitescore.violation("new_gate=1.0 behaves oddly in the corpus") == (
            sitescore.FAILED)

    def test_the_split_and_the_lean_in_counts(self) -> None:
        verdict = {"accept": 3, "refuse": 1, "split": "Three of its four runs accept it and"
                                                      " one refuses it, so it stays in."}
        assert sitescore.split(verdict, True) == (
            "Three of its four runs pass it and one rules it out, so it stays in.")
        tie = {"accept": 2, "refuse": 2, "split": "x"}
        assert sitescore.split(tie, False).endswith("A tie counts against it, so it is ruled"
                                                    " out.")
        assert sitescore.split({"accept": 4, "refuse": 0, "split": ""}, True) == ""
        band = {"component": "seat_decided_chance", "wins": 41, "losses": 20, "decided": 61,
                "favoured": "first", "share": 0.6721, "lean": "the first seat took 41"}
        seat = {"tolerated_share": 0.6667, "refuses": False, "lean": "x",
                "bands": {"top_rung_seat": band}}
        said = sitescore.lean(seat)
        assert said == ("the two strongest computer players won 41 of their 61 decided games"
                        " from the first side, more than the two in three allowed, though"
                        " not by enough to rule the game out")
        _clean(said)
        assert sitescore.lean({**seat, "refuses": True}) == ""

    def test_a_worded_score_carries_both_and_leaves_the_stored_one_alone(self) -> None:
        score = {
            "seeds": [0, 1000, 2000, 3000], "viable": False,
            "verdict": {"accept": 2, "refuse": 2, "split": "Two refuse."},
            "violations": ["Two refuse.", "run 1000: " + SHAPES["gate above"]],
            "runs": [{"seed": 1000, "viable": False, "violations": [SHAPES["gate above"]]}],
            "seat": {"refuses": False, "lean": ""},
        }
        before = copy.deepcopy(score)
        worded = sitescore.worded(score)
        assert score == before
        assert worded is not None
        assert worded["violations"] == before["violations"]
        assert worded["site"]["violations"][0].startswith("Two of its four runs pass it")
        assert worded["site"]["violations"][1].startswith("run 2 of 4: quick kills")
        assert worded["runs"][0]["site"]["violations"][0].startswith("quick kills")
        assert worded["verdict"]["site"]["split"].endswith("so it is ruled out.")
        assert worded["seat"]["site"] == {"lean": ""}
        assert sitescore.worded(None) is None


class TestTheTestsAGameMustPass:
    def test_every_test_of_the_current_rules_has_a_visitors_name_and_reason(self) -> None:
        guide = sitescore.guide(explain())
        assert guide is not None and guide["gates"]
        for gate in guide["gates"]:
            site = gate["site"]
            assert site["title"] and site["reason"], gate["component"]
            _clean(site["title"])
            _clean(site["reason"])
        titles = {gate["component"]: gate["site"]["title"] for gate in guide["gates"]}
        assert titles["search_advantage"] == "thinking beats random moves"
        assert titles["seat_decided_chance"] == "won by one side"
        assert titles["skill_gradient"] == "does thinking longer win"

    def test_the_game_page_payload_carries_them(self) -> None:
        score = {"weights_version": explain()["version"]}
        guide = library.score_guide(score)
        assert guide is not None and all("site" in gate for gate in guide["gates"])


class TestNamesAndLikeness:
    def test_a_ladder_row_is_named_for_a_person(self) -> None:
        """G17: the ladder that scored a game plays its own fitted players."""
        assert opponents.table_name("minimax:fit=98f172aeac49,nodes=6400") == (
            "the 6,400-position player")
        assert opponents.table_name("minimax:nodes=1600") == "medium"
        assert opponents.table_name("random") == "random"
        assert opponents.table_name("nonsense") == "the computer"

    def test_what_differs_in_a_visitors_words(self) -> None:
        """G40: *mechanism* is a way of moving, a square a starting square."""
        counts = {"rules": 1, "settings": 1, "mechanisms": 6, "pieces": 0, "squares": 28,
                  "books": 1}
        assert library.likeness_words(counts) == (
            "1 rule, 1 rule set differently, 6 ways of moving, 28 starting squares, 1 limit"
            " or price")
        assert library.likeness_words({"pieces": 1}) == "no rules, 1 piece"

    def test_a_refused_player_is_named_and_no_more_on_the_site(self) -> None:
        """G19: the file and the repair command are the maintainer's."""
        said = opponents.unavailable("own piece values")
        assert said == "One computer player, own piece values, is unavailable on this game for now."
        _clean(said)
