"""`v15` (R-S17i, `docs/reference/wrong-no.md` §12): the quick-kill gate becomes a warning.

`v15` is `v14` to the digit but for one line: `quick_kill_rate_worst_half` at most 0.5 leaves
the gates for `warnings`. A warning is read as a gate is and written beside the run's
violations as a sentence of its own; no vote reads it, so a warned run votes as if the
sentence were absent, and the page says it beside the score in R-S17a's words. Everything
here is synthetic: hand-built measurements and run readings. Nothing plays a game.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from chess_variant_sandbox.metrics import glossary
from chess_variant_sandbox.metrics.components import Components
from chess_variant_sandbox.metrics.index import score_components
from chess_variant_sandbox.metrics.scoring import (
    DEFAULT_WEIGHTS,
    WEIGHTS_DIR,
    WeightsError,
    load_weights,
    parse_weights,
)
from chess_variant_sandbox.metrics.seeds import VOTE_BLIND_ABSTAINS, RunReading, combine
from chess_variant_sandbox.web import landing, sitescore
from tests.page_text import PAGE

QUICK_KILL = "quick_kill_rate_worst_half"
# `v14.json` as R-S6d made it current: a version is never edited.
V14_SHA256 = "9d7b43073d3660170c14b42ed36a005143c332af9022a393b85dad18ff2dfde3"


def _json(version: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((WEIGHTS_DIR / f"{version}.json").read_text(encoding="utf-8"))
    return data


V15 = load_weights("v15")


class TestTheVersion:
    def test_v15_is_current_and_v14_is_kept_unedited(self) -> None:
        assert DEFAULT_WEIGHTS == "v15"
        assert V15.version == "v15"
        text = (WEIGHTS_DIR / "v14.json").read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(text).hexdigest() == V14_SHA256
        assert load_weights("v14").warnings == ()

    def test_it_is_v14_to_the_digit_but_for_the_one_line(self) -> None:
        v14, v15 = _json("v14"), _json("v15")
        rest = {"version", "notes", "gates", "warnings"}
        assert {k: v for k, v in v14.items() if k not in rest} == {
            k: v for k, v in v15.items() if k not in rest
        }
        moved = [g for g in v14["gates"] if g["component"] == QUICK_KILL]
        assert v15["warnings"] == moved
        assert v15["gates"] == [g for g in v14["gates"] if g["component"] != QUICK_KILL]
        assert len(v15["gates"]) == 6

    def test_a_line_is_never_both_a_gate_and_a_warning(self) -> None:
        data = _json("v15")
        data["gates"] = [*data["gates"], *data["warnings"]]
        with pytest.raises(WeightsError, match="both gate and warn"):
            parse_weights(data)


def _passing_raw() -> dict[str, float]:
    """A measurement every gate of `v15` passes, read off the gates."""
    raw = {name: 0.5 for name in V15.components()}
    for gate in V15.gates:
        raw[gate.component] = (gate.minimum + abs(gate.minimum) + 1.0
                               if gate.minimum is not None else (gate.maximum or 0.0) / 2)
    raw["ladder_monotonicity"] = 1.0
    raw[QUICK_KILL] = 0.1
    return raw


class TestAWarningRefusesAndDeductsNothing:
    def test_a_quick_kill_run_is_viable_with_the_same_total_and_a_warning(self) -> None:
        clean = score_components("h", "g", Components(raw=_passing_raw()), V15)
        raw = {**_passing_raw(), QUICK_KILL: 0.77}
        warned = score_components("h", "g", Components(raw=raw), V15)
        assert clean.viable and warned.viable
        assert warned.total == clean.total and warned.scores == clean.scores
        assert clean.warnings == []
        assert len(warned.warnings) == 1 and warned.warnings[0].startswith(f"{QUICK_KILL}=0.770")
        assert not any(text.startswith(QUICK_KILL) for text in warned.violations)
        assert "WARNING" in warned.summary()

    def test_under_v14_the_same_run_was_refused(self) -> None:
        raw = {**_passing_raw(), QUICK_KILL: 0.77}
        v14 = score_components("h", "g", Components(raw=raw), load_weights("v14"))
        assert not v14.viable and v14.warnings == []

    def test_a_blind_run_says_the_warning_and_still_abstains(self) -> None:
        raw = {**_passing_raw(), QUICK_KILL: 1.0, "ladder_monotonicity": 0.2}
        result = score_components("h", "g", Components(raw=raw), V15)
        assert result.measured is False and result.warnings

    def test_an_unmeasured_line_warns_nothing(self) -> None:
        assert V15.check_warnings({}) == []


def _run(seed: int, warned: bool, viable: bool = True) -> RunReading:
    sentence = f"{QUICK_KILL}=0.773 above maximum 0.5 (a reason)"
    return RunReading(
        seed=seed, scored=True, total=0.7, sd=0.01, viable=viable,
        violations=() if viable else ("stall_rate_worst_half=0.900 above maximum 0.7 (why)",),
        warnings=(sentence,) if warned else (),
    )


class TestAWarnedRunVotesAsIfTheSentenceWereAbsent:
    @pytest.mark.parametrize("pattern", ["AAAA", "AAAR", "AARR", "ARRR", "RRRR"])
    def test_the_verdict_and_the_number_are_the_unwarned_runs(self, pattern: str) -> None:
        seeds = (0, 1000, 2000, 3000)
        plain = [_run(s, False, p == "A") for s, p in zip(seeds, pattern, strict=True)]
        warned = [_run(s, True, p == "A") for s, p in zip(seeds, pattern, strict=True)]
        a = combine("h", "g", "v15", plain, VOTE_BLIND_ABSTAINS)
        b = combine("h", "g", "v15", warned, VOTE_BLIND_ABSTAINS)
        assert (a.viable, a.measured, a.verdict, a.index, a.violations) == (
            b.viable, b.measured, b.verdict, b.index, b.violations)
        assert a.warnings == ()
        assert b.warnings[0]["component"] == QUICK_KILL
        assert (b.warnings[0]["runs"], b.warnings[0]["of"]) == (4, 4)
        assert b.warnings[0]["sentences"][0].startswith(f"run 0: {QUICK_KILL}=")
        assert all(run["warnings"] for run in b.runs)

    def test_the_count_says_how_many_runs_crossed_it(self) -> None:
        runs = [_run(0, True), _run(1000, False), _run(2000, True), _run(3000, False)]
        found = combine("h", "g", "v15", runs, VOTE_BLIND_ABSTAINS)
        assert found.viable
        assert (found.warnings[0]["runs"], found.warnings[0]["of"]) == (2, 4)
        assert len(found.detail["warnings"]) == 2


class TestThePageShowsTheWarning:
    def _score(self, crossed: int) -> dict[str, Any]:
        sentences = [f"run {seed}: {QUICK_KILL}=0.773 above maximum 0.5 (a reason)"
                     for seed in (0, 1000, 2000, 3000)[:crossed]]
        return {"viable": True, "measured": True, "seeds": [0, 1000, 2000, 3000],
                "violations": [], "runs": [],
                "warnings": [{"component": QUICK_KILL, "runs": crossed, "of": 4,
                              "sentences": sentences}]}

    def test_the_words_name_the_test_the_runs_and_the_players(self) -> None:
        said = sitescore.warning(self._score(4))
        assert said is not None
        assert "“quick kills, worse half” on all four of its runs" in said["head"]
        assert "as these computer players play it" in said["head"]
        assert "no longer rules a game out" in said["head"]
        assert "counts against nothing" in said["note"]
        assert len(said["lines"]) == 4 and said["lines"][0].startswith("run 1 of 4: ")
        assert said["lines"][0].endswith("over the 0.5 it warns above")
        assert "on two of its four runs" in sitescore.warning(self._score(2))["head"]

    def test_a_score_without_warnings_has_none_and_worded_carries_it(self) -> None:
        assert sitescore.warning({"warnings": []}) is None
        assert sitescore.warning(None) is None
        worded = sitescore.worded(self._score(4))
        assert worded is not None and worded["warning"]["checks"][0]["runs"] == 4
        assert worded["refusal"] is None

    def test_the_page_draws_it_beside_the_score(self) -> None:
        assert "function warningHtml(warning)" in PAGE
        assert "${warningHtml(score.warning)}" in PAGE
        assert 'data-warning="1"' in PAGE
        assert "warned: ['Warning'" in PAGE
        assert "warningText(game.warning)" in PAGE

    def test_the_guide_lists_it_as_a_warning_and_not_as_a_gate(self) -> None:
        explained = glossary.explain("v15")
        assert QUICK_KILL not in [gate["component"] for gate in explained["gates"]]
        assert [w["component"] for w in explained["warnings"]] == [QUICK_KILL]
        guided = sitescore.guide(explained)
        assert guided is not None
        assert guided["warnings"][0]["site"]["title"] == "quick kills, worse half"
        assert all(QUICK_KILL not in line and "12 turns" not in line
                   for line in landing.rule_outs("v15"))
