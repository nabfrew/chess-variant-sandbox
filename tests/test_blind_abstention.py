"""The blind-run abstention, rule (e) of `docs/plan-teacher.md` §10.16.4 (RS3).

A run whose roster could not read it neither accepts nor refuses, unless it fails a gate
it *could* read; a game is viable or not viable only when every way its abstaining runs
could have voted gives the same verdict, and *not measured* otherwise. Everything here is
synthetic: hand-built measurements and run readings, scored under the current weights or
combined by `metrics.seeds`. Nothing plays a game and nothing names one.
"""

from __future__ import annotations

import itertools

import pytest

from chess_variant_sandbox import corpus
from chess_variant_sandbox.metrics.components import Components
from chess_variant_sandbox.metrics.index import (
    LADDER_DEPENDENT,
    NOT_MEASURED,
    NOT_VIABLE,
    VIABLE,
    score_components,
    verdict_word,
)
from chess_variant_sandbox.metrics.scoring import DEFAULT_WEIGHTS, load_weights
from chess_variant_sandbox.metrics.seeds import (
    VOTE_BLIND_ABSTAINS,
    VOTE_POOLED_SEAT,
    VOTE_RULES,
    RunReading,
    combine,
    verdict,
)
from chess_variant_sandbox.web import sitescore

WEIGHTS = load_weights(DEFAULT_WEIGHTS)


def _passing_raw() -> dict[str, float]:
    """A measurement every gate of the current weights passes, read off the gates."""
    raw = {name: 0.5 for name in WEIGHTS.components()}
    for gate in WEIGHTS.gates:
        if gate.minimum is not None:
            raw[gate.component] = gate.minimum + abs(gate.minimum) + 1.0
        elif gate.maximum is not None:
            raw[gate.component] = gate.maximum / 2
    raw["ladder_monotonicity"] = 1.0
    return raw


def _blind(**changes: float) -> Components:
    """A run whose ladder orders nothing, so the ladder's components are withheld."""
    raw = _passing_raw()
    raw["ladder_monotonicity"] = 0.2
    raw.update(changes)
    return Components(raw=raw, diagnostics={"games": 256})


def _readable_gate():
    """One of the current weights' gates a blind run can still read."""
    return next(gate for gate in WEIGHTS.gates if gate.component not in LADDER_DEPENDENT)


def _failing(gate) -> float:
    if gate.minimum is not None:
        return gate.minimum - abs(gate.minimum) - 1.0
    return gate.maximum * 2 + 1.0


class TestABlindRunReadsItsReadableGates:
    """§10.16.4 point 1: one run, scored by `index._unreadable_result`."""

    def test_the_fixture_passes_every_gate_when_the_ladder_is_read(self) -> None:
        result = score_components("h", "g", Components(raw=_passing_raw()), WEIGHTS)
        assert result.unreadable == ()
        assert result.viable and result.measured

    def test_a_blind_run_that_fails_nothing_it_can_read_abstains(self) -> None:
        result = score_components("h", "g", _blind(), WEIGHTS)
        assert result.unreadable
        assert not result.viable
        assert result.measured is False
        assert result.verdict_word == NOT_MEASURED
        assert result.total == 0.0 and not result.scored
        assert len(result.violations) == 1
        assert result.violations[0].startswith("this roster cannot read this ruleset")
        assert "NOT MEASURED" in result.summary() and "NOT VIABLE" not in result.summary()

    def test_a_blind_run_refuses_on_a_gate_it_could_read(self) -> None:
        gate = _readable_gate()
        result = score_components("h", "g", _blind(**{gate.component: _failing(gate)}), WEIGHTS)
        assert not result.viable
        assert result.measured is True
        assert result.verdict_word == NOT_VIABLE
        # The gate's own sentence, then the *cannot read* one beside it.
        assert result.violations[0].startswith(f"{gate.component}=")
        assert result.violations[-1].startswith("this roster cannot read this ruleset")

    def test_a_ladder_gate_failing_is_not_read_on_a_blind_run(self) -> None:
        """`skill_gradient` below its gate, and negative: it is withheld, so it abstains."""
        ladder_gates = [gate for gate in WEIGHTS.gates if gate.component in LADDER_DEPENDENT]
        assert ladder_gates, "the current weights read no ladder gate; the test means nothing"
        changes = {gate.component: _failing(gate) for gate in ladder_gates}
        result = score_components("h", "g", _blind(**changes), WEIGHTS)
        assert result.measured is False

    def test_a_gate_the_corpus_did_not_supply_is_not_read(self) -> None:
        gate = _readable_gate()
        components = _blind()
        del components.raw[gate.component]
        assert score_components("h", "g", components, WEIGHTS).measured is False


def _accept(seed: int) -> RunReading:
    return RunReading(seed=seed, scored=True, total=0.6, sd=0.02, viable=True,
                      interval={"mean": 0.6, "sd": 0.02, "ci_low": 0.56, "ci_high": 0.64})


def _refuse(seed: int) -> RunReading:
    return RunReading(seed=seed, scored=True, total=0.4, sd=0.02, viable=False,
                      violations=("stall_rate_worst_half=0.900 above maximum 0.7 (x)",),
                      interval={"mean": 0.4, "sd": 0.02, "ci_low": 0.36, "ci_high": 0.44})


def _abstain(seed: int) -> RunReading:
    return RunReading(seed=seed, scored=False, total=None, sd=None, viable=False,
                      violations=("this roster cannot read this ruleset: x, so "
                                  "skill_gradient would describe the players rather than "
                                  "the game",),
                      unreadable=("skill_gradient",), measured=False)


def _blind_refuse(seed: int) -> RunReading:
    return RunReading(seed=seed, scored=False, total=None, sd=None, viable=False,
                      violations=("decisive_rate_worst_half=0.000 below minimum 0.2 (x)",
                                  "this roster cannot read this ruleset: x, so "
                                  "skill_gradient would describe the players rather than "
                                  "the game"),
                      unreadable=("skill_gradient",), measured=True)


_KINDS = {"A": _accept, "R": _refuse, "B": _abstain, "X": _blind_refuse}


def _runs(pattern: str) -> list[RunReading]:
    return [_KINDS[kind](1000 * at) for at, kind in enumerate(pattern)]


class TestTheGameSVote:
    """§10.16.4 point 2: A accepts, R refusals, B abstentions."""

    def test_the_rule_is_in_force_and_named(self) -> None:
        assert corpus.VERDICT_RULE == VOTE_BLIND_ABSTAINS
        assert VOTE_BLIND_ABSTAINS in VOTE_RULES
        assert VOTE_BLIND_ABSTAINS == "vote-tie-refuses+pooled-seat+blind-abstains"

    @pytest.mark.parametrize("pattern", ["".join(p) for n in (1, 2, 3, 4)
                                         for p in itertools.product("AR", repeat=n)])
    def test_with_no_blind_run_it_is_the_older_vote_exactly(self, pattern: str) -> None:
        old = combine("h", "g", "v12", _runs(pattern), VOTE_POOLED_SEAT)
        new = combine("h", "g", "v12", _runs(pattern), VOTE_BLIND_ABSTAINS)
        assert (new.viable, new.measured, new.violations, new.index) == (
            old.viable, True, old.violations, old.index)
        assert new.verdict.split == old.verdict.split

    @pytest.mark.parametrize("pattern", ["".join(p) for n in (1, 2, 3, 4)
                                         for p in itertools.product("ARB", repeat=n)])
    def test_a_verdict_only_where_every_abstainer_s_vote_agrees(self, pattern: str) -> None:
        """The completion rule, against brute force over the abstainers' possible votes."""
        found = verdict(_runs(pattern), VOTE_BLIND_ABSTAINS)
        accept, refuse = pattern.count("A"), pattern.count("R")
        outcomes = {
            accept + extra > refuse + (pattern.count("B") - extra)
            for extra in range(pattern.count("B") + 1)
        }
        assert (found.accept, found.refuse, found.abstain) == (
            accept, refuse, pattern.count("B"))
        if outcomes == {True}:
            assert found.word == VIABLE
        elif outcomes == {False}:
            assert found.word == NOT_VIABLE
        else:
            assert found.word == NOT_MEASURED

    def test_one_accept_and_three_blind_is_not_measured(self) -> None:
        found = combine("h", "g", "v12", _runs("ABBB"), VOTE_BLIND_ABSTAINS)
        assert found.word == NOT_MEASURED
        assert not found.viable and found.measured is False
        assert found.index is None and found.totals_used == 1
        assert found.violations[0].startswith("Not measured: ")
        assert found.verdict.split == (
            "Of its four runs, one accepts it and three could not be read, so it is not "
            "measured.")
        assert found.detail["measured"] is False
        assert all("measured" in run for run in found.runs)

    def test_the_same_runs_under_the_older_rule_are_not_viable(self) -> None:
        found = combine("h", "g", "v12", _runs("ABBB"), VOTE_POOLED_SEAT)
        assert found.word == NOT_VIABLE

    def test_blind_runs_that_cannot_swing_it_leave_a_verdict(self) -> None:
        assert combine("h", "g", "v12", _runs("AAAB"), VOTE_BLIND_ABSTAINS).word == VIABLE
        # Two refusals and two abstentions: even if both accepted, a tie refuses.
        found = combine("h", "g", "v12", _runs("RRBB"), VOTE_BLIND_ABSTAINS)
        assert found.word == NOT_VIABLE and found.measured is True

    def test_blind_runs_that_could_swing_it_are_not_measured(self) -> None:
        assert combine("h", "g", "v12", _runs("AARB"), VOTE_BLIND_ABSTAINS).word == NOT_MEASURED
        assert combine("h", "g", "v12", _runs("BBBB"), VOTE_BLIND_ABSTAINS).word == NOT_MEASURED

    def test_a_blind_run_refusing_on_a_readable_gate_is_a_refusal(self) -> None:
        found = combine("h", "g", "v12", _runs("XXXX"), VOTE_BLIND_ABSTAINS)
        assert found.word == NOT_VIABLE
        assert (found.verdict.refuse, found.verdict.abstain) == (4, 0)
        assert any("decisive_rate_worst_half" in line for line in found.violations)
        swung = combine("h", "g", "v12", _runs("AXBB"), VOTE_BLIND_ABSTAINS)
        assert swung.word == NOT_MEASURED

    def test_one_blind_run_alone_is_not_measured(self) -> None:
        """§10.16.4 point 4: a visitor's lone blind run."""
        found = combine("h", "g", "v12", _runs("B"), VOTE_BLIND_ABSTAINS)
        assert found.word == NOT_MEASURED
        assert found.violations[0] == (
            "Not measured: its computer players could not tell a stronger search from a "
            "weaker one on this run.")
        assert combine("h", "g", "v12", _runs("X"), VOTE_BLIND_ABSTAINS).word == NOT_VIABLE

    def test_the_pooled_seat_refuses_on_top(self) -> None:
        """§10.16.4 point 3: a seat refusal is a verdict whatever the abstentions."""
        def seat(run: int) -> dict:
            # A clean sweep a run, every game different: `test_seeds`' shape.
            return {"top_rung_seat": {
                "rungs": [1600, 6400], "games": 16, "wins": 16, "losses": 0, "draws": 0,
                "distinct": 16, "transcripts": [f"b{run}-{game}" for game in range(16)],
            }}

        runs = [
            RunReading(**{**_KINDS[kind](1000 * at).__dict__, "seat": seat(at)})
            for at, kind in enumerate("ABBB")
        ]
        found = combine("h", "g", "v12", runs, VOTE_BLIND_ABSTAINS)
        assert found.seat.get("refuses")
        assert found.word == NOT_VIABLE and found.measured is True
        assert not any(line.startswith("Not measured") for line in found.violations)
        assert not any("so it is not measured" in line for line in found.violations)


class TestTheWords:
    """§10.16.4 point 5: every reader says *not measured*, never *ruled out*."""

    def test_verdict_word_reads_an_older_file_as_measured(self) -> None:
        assert verdict_word(True) == VIABLE
        assert verdict_word(False) == NOT_VIABLE
        assert verdict_word(False, None) == NOT_VIABLE
        assert verdict_word(False, False) == NOT_MEASURED

    def test_the_site_split_says_not_measured(self) -> None:
        found = combine("h", "g", "v12", _runs("ABBB"), VOTE_BLIND_ABSTAINS)
        said = sitescore.split(found.verdict.as_dict(), found.viable)
        assert said == ("Of its four runs, one passes it and three could not be read, so it "
                        "is not measured.")
        assert "ruled out" not in said

    def test_the_site_words_the_not_measured_sentence(self) -> None:
        found = combine("h", "g", "v12", _runs("ABBB"), VOTE_BLIND_ABSTAINS)
        said = sitescore.violation(found.violations[0])
        assert said.startswith("not measured: its computer players could not tell")
        assert said != sitescore.FAILED

    def test_the_older_split_words_are_unchanged(self) -> None:
        old = {"accept": 3, "refuse": 1, "split": "x"}
        assert sitescore.split(old, True) == (
            "Three of its four runs pass it and one rules it out, so it stays in.")


class TestTheReadersSayIt:
    """The verdict's readers outside `metrics`: each says *not measured*, none *ruled out*."""

    def test_the_proposer_hears_no_verdict_and_no_rejection(self) -> None:
        from chess_variant_sandbox.llm.feedback import describe

        text = describe(score_components("h", "g", _blind(), WEIGHTS), {"pieces": []})
        assert "not measured" in text and "rejected outright" not in text
        gate = _readable_gate()
        refused = score_components("h", "g", _blind(**{gate.component: _failing(gate)}), WEIGHTS)
        assert "still rejected" in describe(refused, {"pieces": []})

    def test_the_archive_says_not_measured_and_not_disqualified(self) -> None:
        from chess_variant_sandbox.llm.feedback import describe_failures
        from chess_variant_sandbox.loop import Entry

        entry = Entry(hash="h", name="g", generation=0, parent="p", status="evaluated",
                      index=None, viable=False, measured=False, notes="x")
        assert entry.scored_as() == "not measured"
        assert "disqualified" not in describe_failures([entry])
        assert "no verdict" in describe_failures([entry])

    def test_a_stored_score_s_flag_reaches_the_status_and_the_promotion_refusal(self) -> None:
        status = corpus.ScoringStatus(hash="h", name="g", stage="corpus", state="current",
                                      needs="", reason="", measured=False)
        assert status.as_dict()["measured"] is False

    def test_the_page_draws_it_and_never_as_ruled_out(self) -> None:
        from page_text import PAGE

        assert "function isNotMeasured(" in PAGE
        assert "function notMeasuredSentence(" in PAGE
        # The page's readers of `viable === false` that would say *ruled out* ask first.
        assert PAGE.count("isNotMeasured(") >= 10
