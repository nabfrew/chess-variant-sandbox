"""The whole design loop, driven by a stubbed model.

The point of this file is that everything between the model call and the archive is
testable without credentials: proposal validation, the cheap gate, the tournament, the
index, the judge blend, lineage, deduplication and parent selection. Only the two
network calls are stubbed, and they are stubbed with fixed JSON rather than mocked
behaviour, so the test fails if the surrounding pipeline breaks -- which is the part
that can break silently.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from chess_variant_sandbox.llm.client import JUDGE_MODEL, PROPOSAL_MODEL
from chess_variant_sandbox.loop import Archive, Entry, LoopConfig, evolve
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS

GOOD_SPEC = {
    "schema_version": 1,
    "name": "lancers",
    "description": "A small skirmish game.",
    "board": {"width": 5, "height": 5},
    "pieces": [
        {"id": "K", "name": "captain", "royal": True, "value": 0, "moves": [
            {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"}
        ]},
        {"id": "L", "name": "lancer", "value": 5, "moves": [
            {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both"}
        ]},
    ],
    "setup": {"mirror": True, "placements": [
        {"piece": "K", "squares": ["c1"]},
        {"piece": "L", "squares": ["a1", "e1"]},
    ]},
    "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 120,
              "no_progress_draw": 40},
}

# What the judge is asked for now: which of two games is better, not how good one is.
# "first" and "second" name a *position* in the prompt, which is exactly why every
# comparison is made twice with the pair swapped.
PREFERS_FIRST = {"better": "first", "because": "The first has more to decide."}
PREFERS_SECOND = {"better": "second", "because": "The second has more to decide."}
NO_PREFERENCE = {"better": "neither", "because": "They are very close."}


class StubModel:
    """Returns queued responses. Records what it was shown, so prompts can be asserted."""

    def __init__(
        self, responses: list[dict[str, Any]], model: str = PROPOSAL_MODEL
    ) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.systems: list[Any] = []
        # A stub stands for `Claude`, and `Claude` knows which model it is. That was
        # cosmetic until `cached_system` began asking: whether a prefix is worth a cache
        # breakpoint is a fact about the model, so a stub that cannot say which one it is
        # cannot exercise the branch that decides it.
        self.model = model

    def structured(self, system, user, schema, max_tokens=16000) -> dict[str, Any]:
        self.prompts.append(user)
        self.systems.append(system)
        return self.responses.pop(0) if self.responses else copy.deepcopy(GOOD_SPEC)


def _config(tmp_path, **overrides) -> LoopConfig:
    settings: dict[str, Any] = {
        "generations": 1,
        # Two minimax budgets: the loop refuses a roster that cannot measure
        # how much stronger a player gets when allowed to think longer. Kept to the
        # smallest roster that satisfies it, since these run on every commit.
        # Spelled `threats=0`, the search before Q502i: one game a pairing is a coin
        # flip on whether the larger budget wins, the fixture's candidates pass their
        # gates on the flip that search gives, and these tests are about the loop, not
        # the search. Q502i's default flipped it to non-viable.
        "agents": ("random", "minimax:nodes=80,threats=0", "minimax:nodes=320,threats=0"),
        "games_per_pairing": 1,
        "workers": 1,
        "smoke_games": 12,
        # These tests are about the model path, so the free local operator is switched
        # off unless a test asks for it. Left at its default, half the generations would
        # never reach the stub model and the assertions about prompts would be about
        # nothing. `TestMechanicalOperator` covers the other half.
        "mechanical_rate": 0.0,
        "directory": tmp_path,
    }
    settings.update(overrides)
    return LoopConfig(**settings)


def test_a_good_proposal_runs_the_whole_pipeline(tmp_path) -> None:
    proposer = StubModel([copy.deepcopy(GOOD_SPEC)])
    judge = StubModel([], model=JUDGE_MODEL)

    archive = evolve(_config(tmp_path), proposer=proposer, judge_client=judge, report=lambda _: None)

    assert len(archive.entries) == 1
    entry = archive.entries[0]
    assert entry.status == "evaluated"
    assert entry.name == "lancers"
    assert entry.objective is not None
    # Nothing to compare a first candidate against, so the judge is not called at all.
    assert judge.prompts == []
    # And no opinion enters the score: the headline *is* the objective measurement now.
    assert entry.index == entry.objective
    assert entry.judge is None
    assert entry.weights_version == DEFAULT_WEIGHTS
    # One tournament at the run's one seed, recorded, so the number can say it is one
    # run's wherever it is shown (Q416).
    assert entry.seeds == [_config(tmp_path).seed]
    assert archive.load_spec(entry.hash) is not None
    assert (tmp_path / "index" / f"{entry.hash}.json").exists()
    assert (tmp_path / "tournaments" / entry.hash / "games.jsonl").exists()
    # The default weights ask for engine-derived components, so the loop has to run the
    # second pass. Without this the loop would be optimising a strict subset of the
    # index the corpus was calibrated against.
    assert (tmp_path / "tournaments" / entry.hash / "analysis.json").exists()
    # A candidate with no parent did not fail to improve on anything, and the two cases
    # have to stay distinguishable.
    assert entry.separated is None


def test_an_invalid_proposal_is_retried_once_then_archived(tmp_path) -> None:
    broken = copy.deepcopy(GOOD_SPEC)
    broken["board"] = {"width": 99, "height": 99}
    proposer = StubModel([broken, copy.deepcopy(broken)])

    archive = evolve(
        _config(tmp_path, judge=False), proposer=proposer, report=lambda _: None
    )

    assert archive.entries[0].status == "invalid"
    assert len(proposer.prompts) == 2
    # The retry has to carry the specific errors, or it is just a second guess.
    assert "rejected by the validator" in proposer.prompts[1]
    assert "board.width" in proposer.prompts[1]


def test_a_degenerate_proposal_is_rejected_before_any_tournament(tmp_path) -> None:
    stuck = copy.deepcopy(GOOD_SPEC)
    stuck["pieces"] = [
        {"id": "K", "royal": True, "value": 0, "moves": [
            {"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "move"}
        ]}
    ]
    stuck["setup"] = {"mirror": True, "placements": [{"piece": "K", "squares": ["c1"]}]}
    stuck["rules"] = {"win": "royal_capture", "max_plies": 40, "no_progress_draw": 40}

    archive = evolve(
        _config(tmp_path, smoke_games=40, judge=False),
        proposer=StubModel([stuck]),
        report=lambda _: None,
    )

    assert archive.entries[0].status == "rejected"
    assert not (tmp_path / "tournaments").exists()


def test_a_renamed_repeat_is_caught_before_it_costs_a_tournament(tmp_path) -> None:
    """The reason the content hash ignores name and description."""
    renamed = copy.deepcopy(GOOD_SPEC)
    renamed["name"] = "a completely different game"
    renamed["description"] = "a bold new direction"

    archive = evolve(
        _config(tmp_path, generations=2),
        proposer=StubModel([copy.deepcopy(GOOD_SPEC), renamed]),
        judge_client=StubModel(
            [dict(NO_PREFERENCE), dict(NO_PREFERENCE)], model=JUDGE_MODEL
        ),
        report=lambda _: None,
    )

    assert [e.status for e in archive.entries] == ["evaluated", "duplicate"]


def test_the_second_generation_revises_the_first(tmp_path) -> None:
    variant = copy.deepcopy(GOOD_SPEC)
    variant["name"] = "lancers II"
    variant["board"] = {"width": 6, "height": 6}
    proposer = StubModel([copy.deepcopy(GOOD_SPEC), variant])
    lines: list[str] = []

    archive = evolve(
        _config(tmp_path, generations=2, explore_rate=0.0),
        proposer=proposer,
        judge_client=StubModel(
            [dict(NO_PREFERENCE), dict(NO_PREFERENCE)], model=JUDGE_MODEL
        ),
        report=lines.append,
    )

    # Every number the run prints says it is one run's (`v11` C3): each candidate's
    # headline, and the parent's number on the line that says what is being revised.
    first = archive.entries[0]
    headlines = [line for line in lines if " index=" in line]
    assert len(headlines) == 2
    assert all(" as the one run (objective " in line for line in headlines)
    assert f"revising lancers [{first.hash}] ({first.scored_as()})" in lines
    assert " as the one run" in first.scored_as()

    revision_prompt = proposer.prompts[1]
    assert "Revise the game below" in revision_prompt
    assert "lancers" in revision_prompt
    # Observations from real games must reach the proposer...
    assert "when simulated" in revision_prompt
    # ...but never the scoring function. This is the guarantee the design rests on.
    for term in ("skill_gradient", "rule_mass", "weights", "index"):
        assert term not in revision_prompt


def test_a_candidate_with_no_score_keeps_no_number(tmp_path, monkeypatch) -> None:
    """A placeholder total is printed as *no score* and archived as no index (`v11` C3).

    The first candidate is scored as it is. The second is given the shape both refusals
    in `metrics/index.py` return, a 0.0 total with nothing combined, which is what a
    roster that cannot read a ruleset produces. Before C3 the loop printed it as
    ``index=0.000``, stored 0.0 as the entry's index, and compared it with its parent.
    """
    import dataclasses

    from chess_variant_sandbox.loop import orchestrator

    real = orchestrator.score_ruleset
    calls: list[str] = []

    def second_is_refused(spec, *args, **kwargs):
        result = real(spec, *args, **kwargs)
        calls.append(spec["name"])
        if len(calls) == 1:
            return result
        return dataclasses.replace(
            result, total=0.0, viable=False, contributions={}, families={},
            combined=None, interval=None,
            violations=["this roster cannot read this ruleset: stubbed"],
        )

    monkeypatch.setattr(orchestrator, "score_ruleset", second_is_refused)
    variant = copy.deepcopy(GOOD_SPEC)
    variant["name"] = "lancers II"
    variant["board"] = {"width": 6, "height": 6}
    lines: list[str] = []

    archive = evolve(
        _config(tmp_path, generations=2, explore_rate=0.0, judge=False),
        proposer=StubModel([copy.deepcopy(GOOD_SPEC), variant]),
        report=lines.append,
    )

    assert calls == ["lancers", "lancers II"]
    refused = archive.entries[1]
    assert refused.status == "evaluated" and not refused.viable
    assert refused.index is None and refused.objective is None
    assert refused.separated is None
    assert refused.scored_as() == "no score"
    # `summary()` is reported whole, so its headline is the first line of one report.
    heads = [line.splitlines()[0] for line in lines if line.startswith("  lancers II [")]
    assert heads == [f"  lancers II [{refused.hash}]  no score (weights {DEFAULT_WEIGHTS})"]
    assert "  no score, so it is not compared with its parent" in lines
    assert not [line for line in lines if "lancers II" in line and "0.000" in line]


def test_the_run_resumes_where_it_stopped(tmp_path) -> None:
    common = {"report": lambda _: None}
    config = _config(tmp_path, judge=False)
    evolve(config, proposer=StubModel([copy.deepcopy(GOOD_SPEC)]), **common)

    variant = copy.deepcopy(GOOD_SPEC)
    variant["board"] = {"width": 6, "height": 6}
    evolve(_config(tmp_path, judge=False), proposer=StubModel([variant]), **common)

    archive = Archive(tmp_path)
    assert [e.generation for e in archive.entries] == [0, 1]


def _relabelled(generation: int) -> dict[str, Any]:
    """The same game under different labels: a different hash, an identical ruleset.

    Piece *names* are part of the spec and therefore of its content hash, but nothing
    downstream compiles them -- ``compile.py`` keys on ``id`` -- so a renamed spec plays
    move for move identically. This is `mutants.py`'s ``null-renamed`` transform with a
    per-generation label so each clone survives the duplicate check and costs a real
    tournament.
    """
    clone = copy.deepcopy(GOOD_SPEC)
    clone["name"] = f"lancers mark {generation}"
    for index, piece in enumerate(clone["pieces"]):
        piece["name"] = f"piece {index}-{generation}"
    return clone


def test_a_run_of_clones_reports_no_improvement(tmp_path) -> None:
    """The acceptance test for interval-aware selection.

    Every candidate is the same game wearing a different label, so there is nothing to
    find. A loop that ranks by the point estimate will still show a leaderboard that
    climbs, because the top of three samples of one distribution is above the first
    sample of it. The loop must instead report that no revision beat its parent, and no
    entry may claim separation.
    """
    proposer = StubModel([_relabelled(n) for n in range(3)])

    archive = evolve(
        _config(tmp_path, generations=3, explore_rate=0.0, judge=False),
        proposer=proposer,
        report=lambda _: None,
    )

    evaluated = [e for e in archive.entries if e.status == "evaluated"]
    assert len(evaluated) == 3, [e.status for e in archive.entries]

    # Nothing separated from anything, in either direction.
    assert not any(e.separated for e in evaluated)
    assert archive.top_group() == archive.leaderboard()
    assert "has not yet found an improvement" in archive.summary()

    # And the scores really are the same game measured three times: identical, since a
    # rename cannot reach a single measurement.
    assert len({round(e.objective or 0.0, 6) for e in evaluated}) == 1


def test_selection_is_uniform_over_the_tied_group(tmp_path) -> None:
    """Ties are broken by nothing that correlates with the index.

    The old rule weighted parents by ``index ** 3``, which sharpened an ordering whose
    gaps are smaller than its own error bars. With overlapping intervals every viable
    entry is a candidate parent, and each must be reachable.
    """
    import random

    from chess_variant_sandbox.loop.orchestrator import _choose_parent

    archive = Archive(tmp_path)
    for digest, index in (("a", 0.40), ("b", 0.80), ("c", 0.55)):
        archive.record(
            Entry(
                hash=digest, name=digest, generation=0, parent=None, status="evaluated",
                index=index, objective=index, viable=True, index_sd=0.12,
                # Interval half-widths of this order are what the reference corpus
                # actually shows against a human band 0.14 wide, so the three overlap.
                index_ci_low=index - 0.25, index_ci_high=index + 0.25,
            )
        )

    config = _config(tmp_path, explore_rate=0.0)
    rng = random.Random(0)
    chosen = {(_choose_parent(archive, config, rng) or Entry("", "", 0, None, "x")).hash
              for _ in range(200)}
    assert chosen == {"a", "b", "c"}


def test_a_clearly_better_candidate_is_still_selected(tmp_path) -> None:
    """Tied does not mean everything is tied. A real gap must still win.

    Without this the change would read as "selection pressure removed", which is not
    what it says: pressure is removed *where the evidence cannot support it*.
    """
    import random

    from chess_variant_sandbox.loop.orchestrator import _choose_parent

    archive = Archive(tmp_path)
    archive.record(Entry(hash="weak", name="w", generation=0, parent=None,
                         status="evaluated", index=0.30, objective=0.30, viable=True,
                         index_sd=0.01, index_ci_low=0.28, index_ci_high=0.32))
    archive.record(Entry(hash="strong", name="s", generation=1, parent=None,
                         status="evaluated", index=0.80, objective=0.80, viable=True,
                         index_sd=0.01, index_ci_low=0.78, index_ci_high=0.82))

    config = _config(tmp_path, explore_rate=0.0)
    rng = random.Random(0)
    chosen = {_choose_parent(archive, config, rng).hash for _ in range(50)}
    assert chosen == {"strong"}


class TestMechanicalOperator:
    """The free half of the operator set, driving the loop with no model at all.

    An LLM call costs more than the tournament that follows it, so spending it on a
    single-step local edit is the worst available allocation -- and until the loop had a
    local operator, every candidate was one. These check that the cheap path is a real
    path: it produces candidates, they are evaluated identically to proposed ones, and
    the run needs no credentials.
    """

    def test_a_run_needs_no_model_at_all(self, tmp_path) -> None:
        import json

        spec_path = tmp_path / "seed.json"
        spec_path.write_text(json.dumps(GOOD_SPEC), encoding="utf-8")

        archive = evolve(
            _config(
                tmp_path / "run",
                generations=2,
                explore_rate=0.0,
                judge=False,
                mechanical_rate=1.0,
                seeds=(spec_path,),
            ),
            # No proposer, and none may be constructed: a run of local edits that
            # demanded an API key would be a local operator in name only.
            proposer=None,
            report=lambda _: None,
        )

        evaluated = [e for e in archive.entries if e.status == "evaluated"]
        assert len(evaluated) >= 2, [(e.status, e.notes[:40]) for e in archive.entries]
        children = [e for e in archive.entries if e.parent]
        assert children, "no candidate was made from the starting point"
        for child in children:
            # Every child says which operator made it, including one that turned out to
            # be a game already in the archive -- a mechanical edit can undo an earlier
            # one, and "this operator keeps rediscovering what we have" is exactly the
            # kind of thing the record should be able to answer.
            assert child.detail.get("operator")
            assert child.detail.get("edit")

    def test_a_candidate_records_what_made_it_and_what_played_it(self, tmp_path) -> None:
        """Read off the seed, which is measured before any generation runs, so none does."""
        import json

        spec_path = tmp_path / "seed.json"
        spec_path.write_text(json.dumps(GOOD_SPEC), encoding="utf-8")

        archive = evolve(
            _config(
                tmp_path / "run",
                generations=0,
                explore_rate=0.0,
                judge=False,
                mechanical_rate=1.0,
                seeds=(spec_path,),
            ),
            proposer=None,
            report=lambda _: None,
        )

        seeded = archive.entries[0]
        # Free to record, and it makes a run's cost profile reconstructable: the two
        # engines differ by two orders of magnitude, so which one played a candidate is
        # part of what that candidate cost.
        assert seeded.detail.get("backend") in ("core", "reference")
        assert "screen" in seeded.detail


class TestBatchedGenerations:
    """A generation makes several children, gates them all, and spends one tournament.

    The loop used to evaluate one candidate per generation, so the whole tournament
    budget went on whatever a single draw produced. Both halves of the fix were already
    paid for: a mechanical child costs a validation and a screen costs 0.70s, against
    minutes for the rated roster.
    """

    @staticmethod
    def _seeded(tmp_path, **overrides):
        import json

        tmp_path.mkdir(parents=True, exist_ok=True)
        spec_path = tmp_path / "seed.json"
        spec_path.write_text(json.dumps(GOOD_SPEC), encoding="utf-8")
        settings = dict(
            generations=1, explore_rate=0.0, judge=False, mechanical_rate=1.0,
            seeds=(spec_path,),
        )
        settings.update(overrides)
        return evolve(
            _config(tmp_path / "run", **settings), proposer=None, report=lambda _: None
        )

    @pytest.fixture(scope="class")
    @classmethod
    def batch_of_six(cls, tmp_path_factory: pytest.TempPathFactory) -> Archive:
        """One generation, six children, one promoted: read by two tests, run once."""
        return cls._seeded(tmp_path_factory.mktemp("batch-of-six"), batch=6, promote=1)

    def test_a_generation_gates_a_batch_and_measures_one_of_it(
        self, batch_of_six: Archive
    ) -> None:
        archive = batch_of_six

        children = [e for e in archive.entries if e.parent]
        # One tournament for the generation, as before. The rest of the batch either
        # failed a cheap rung and is recorded, or survived and was left for the operator
        # to rediscover -- never a second tournament.
        measured = [e for e in children if e.status == "evaluated"]
        assert len(measured) == 1, [(e.status, e.name) for e in children]
        assert all(e.status in ("evaluated", "rejected", "duplicate") for e in children)

    def test_promoting_more_than_one_costs_more_than_one_tournament(self, tmp_path) -> None:
        archive = self._seeded(tmp_path, batch=6, promote=3)
        measured = [
            e for e in archive.entries if e.parent and e.status == "evaluated"
        ]
        assert 1 < len(measured) <= 3

    def test_the_batch_size_is_checked_rather_than_silently_clamped(self) -> None:
        import pytest

        from chess_variant_sandbox.loop import LoopConfig

        with pytest.raises(ValueError, match="batch"):
            LoopConfig(batch=0)
        with pytest.raises(ValueError, match="promote"):
            LoopConfig(batch=4, promote=5)

    def test_a_batch_does_not_change_what_a_candidate_scores(self, tmp_path) -> None:
        """The seed discipline, which batching must not disturb.

        One seed for a whole run, so identical rules play identical games and two
        candidates stay comparable. A batch draws more children from the same stream,
        which changes *which* candidates appear -- it must not change what any of them
        measures. So a child both runs measure has to score the same in both.

        The first child of a batch is the one draw a batch of one makes, and promoting
        the whole batch measures it, after the gates have run on all of its siblings.
        This used to compare only the seed, which is measured before any batch is drawn
        and so could not have moved (Q377).

        **The run seed is searched for, not fixed.** The comparison needs the batch of
        one's single child to pass the gates, and which child the first draw makes depends
        on how many operators there are: Q452's `promote-on-leaving` turned seed 0's first
        draw into a `restrict-a-capture` child the gates reject, and the test then compared
        nothing. So it takes the first seed whose first child is measured, which any
        operator set has within a few. The search starts at 3, the first such seed under
        Q452's 34 operators, so the test costs two runs until the next operator moves it.
        """

        def measured(archive: Archive) -> dict[str, Entry]:
            return {e.hash: e for e in archive.entries if e.parent and e.status == "evaluated"}

        for seed in range(3, 11):
            alone = measured(self._seeded(tmp_path / f"a{seed}", batch=1, seed=seed))
            if alone:
                break
        assert alone, "no run seed in eight measured the first child; the gates moved"
        many = self._seeded(tmp_path / f"b{seed}", batch=3, promote=3, seed=seed)
        batched = measured(many)
        shared = alone.keys() & batched.keys()
        assert shared, ("no child was measured by both runs, so nothing is compared",
                        sorted(alone), sorted(batched))
        for digest in shared:
            assert alone[digest].index == batched[digest].index
            assert alone[digest].index_sd == batched[digest].index_sd

    def test_every_candidate_records_where_its_time_went(self, batch_of_six: Archive) -> None:
        """The measurement stage 9b is required to take before it builds anything.

        The archive is the only record of a run's cost once the run is over, and the
        conclusion it produced was that the tournament -- the thing 9b proposes to
        optimise -- is about one per cent of a candidate.
        """
        measured = [e for e in batch_of_six.entries if e.status == "evaluated"]
        assert len(measured) == 2  # the seed and the one promoted child
        for entry in measured:
            seconds = entry.detail["seconds"]
            for phase in ("smoke", "screen", "tournament", "analysis", "score"):
                assert phase in seconds, (phase, sorted(seconds))
                assert seconds[phase] >= 0.0


class TestPairwiseJudge:
    """The judge in the loop: an ordering, not a term in the score.

    An absolute judgement entered the headline at a quarter weight, which made it the
    ranker inside a band the objective index resolves to about 0.05. It is now asked
    which of two games is better, both ways round, and the ordering is used only where
    the interval says the index cannot choose.
    """

    def test_a_second_candidate_is_compared_with_the_first(self, tmp_path) -> None:
        from chess_variant_sandbox.loop.preference import PreferenceLog

        variant = copy.deepcopy(GOOD_SPEC)
        variant["name"] = "lancers II"
        variant["board"] = {"width": 6, "height": 6}
        # Prefers the game shown first, then the one shown second: the same game both
        # times, which is what makes it a preference rather than a position.
        judge = StubModel([dict(PREFERS_FIRST), dict(PREFERS_SECOND)], model=JUDGE_MODEL)

        archive = evolve(
            _config(tmp_path, generations=2, explore_rate=0.0),
            proposer=StubModel([copy.deepcopy(GOOD_SPEC), variant]),
            judge_client=judge,
            report=lambda _: None,
        )

        log = PreferenceLog(tmp_path)
        assert len(log.comparisons) == 1
        comparison = log.comparisons[0]
        assert comparison.agreed
        # The second candidate was shown first, so a preference for "first" then
        # "second" is a preference for it.
        assert comparison.winner == [e.hash for e in archive.entries][1]
        # Two calls for one comparison, which is the whole cost of the defence against
        # position bias.
        assert len(judge.prompts) == 2

    def test_a_flipped_preference_is_recorded_as_a_tie(self, tmp_path) -> None:
        from chess_variant_sandbox.loop.preference import PreferenceLog

        variant = copy.deepcopy(GOOD_SPEC)
        variant["name"] = "lancers II"
        variant["board"] = {"width": 6, "height": 6}
        judge = StubModel([dict(PREFERS_FIRST), dict(PREFERS_FIRST)], model=JUDGE_MODEL)

        evolve(
            _config(tmp_path, generations=2, explore_rate=0.0),
            proposer=StubModel([copy.deepcopy(GOOD_SPEC), variant]),
            judge_client=judge,
            report=lambda _: None,
        )

        comparison = PreferenceLog(tmp_path).comparisons[0]
        assert not comparison.agreed
        assert comparison.winner is None
        assert PreferenceLog(tmp_path).table().flip_rate == 1.0

    def test_an_ordering_too_thin_to_trust_decides_nothing(self, tmp_path) -> None:
        """One comparison is one opinion, and selection stays uniform over the tied group."""
        import random

        from chess_variant_sandbox.llm.compare import Comparison
        from chess_variant_sandbox.loop.orchestrator import _choose_parent
        from chess_variant_sandbox.loop.preference import PreferenceLog

        archive = Archive(tmp_path)
        for digest, index in (("a", 0.40), ("b", 0.44)):
            archive.record(
                Entry(
                    hash=digest, name=digest, generation=0, parent=None,
                    status="evaluated", index=index, objective=index, viable=True,
                    index_sd=0.12, index_ci_low=index - 0.25, index_ci_high=index + 0.25,
                )
            )
        log = PreferenceLog(tmp_path)
        log.record(Comparison(first="a", second="b", winner="a", agreed=True))

        rng = random.Random(0)
        config = _config(tmp_path, explore_rate=0.0)
        chosen = {_choose_parent(archive, config, rng, log).hash for _ in range(60)}
        assert chosen == {"a", "b"}, "one opinion must not decide a tie"

    def test_a_settled_ordering_breaks_the_tie(self, tmp_path) -> None:
        """And when the comparisons do support it, the preferred game is the parent.

        This is the direction that interval-aware selection gave up: with overlapping
        intervals every viable candidate is a candidate parent, and something that is
        not the index has to choose between them.
        """
        import random

        from chess_variant_sandbox.llm.compare import Comparison
        from chess_variant_sandbox.loop.orchestrator import _choose_parent
        from chess_variant_sandbox.loop.preference import PreferenceLog

        archive = Archive(tmp_path)
        for digest, index in (("a", 0.40), ("b", 0.44)):
            archive.record(
                Entry(
                    hash=digest, name=digest, generation=0, parent=None,
                    status="evaluated", index=index, objective=index, viable=True,
                    index_sd=0.12, index_ci_low=index - 0.25, index_ci_high=index + 0.25,
                )
            )
        log = PreferenceLog(tmp_path)
        for _ in range(8):
            log.record(Comparison(first="a", second="b", winner="a", agreed=True))

        rng = random.Random(0)
        config = _config(tmp_path, explore_rate=0.0)
        chosen = {_choose_parent(archive, config, rng, log).hash for _ in range(30)}
        assert chosen == {"a"}, "a settled preference is the tie-break"


class TestTheRequestTheLoopActuallySends:
    """The stub above hands `structured` a schema and never sees a request.

    That is precisely the gap the judge died in: `Claude.structured` built the request
    inside the one method no offline test can call, sent adaptive thinking and an effort
    to every model, and `claude-haiku-4-5` accepts neither -- so every comparison in
    every design run answered 400, and this file passed throughout.

    So this class stands a fake *transport* where the SDK client goes, and refuses what
    the API refuses. Everything above the transport is the real thing: the real `Claude`,
    the real `compare_rulesets`, the real orchestrator.
    """

    class Transport:
        """Stands in for `anthropic.Anthropic`, and answers 400 the way it does."""

        REFUSES = {
            # The thinking half is measured -- it is the 400 that revealed the bug. The
            # effort half is the published support matrix, and is refused here for the
            # same reason it is withheld in `client.MODEL_WIRE`: a parameter a judge
            # does not need is not worth being wrong about.
            "claude-haiku-4-5": ("thinking", "effort"),
            "claude-opus-5": (),
        }

        def __init__(self, replies: list[dict[str, Any]]) -> None:
            self.replies = list(replies)
            self.requests: list[dict[str, Any]] = []

        @property
        def messages(self) -> TestTheRequestTheLoopActuallySends.Transport:
            return self

        def stream(self, **request: Any):  # noqa: ANN401 - mirrors the SDK's own shape
            self.requests.append(request)
            model = request["model"]
            refused = self.REFUSES.get(model, ())
            if "thinking" in refused and "thinking" in request:
                raise RuntimeError(
                    f"400 invalid_request_error: adaptive thinking is not supported on {model}"
                )
            if "effort" in refused and "effort" in request.get("output_config", {}):
                raise RuntimeError(
                    f"400 invalid_request_error: output_config.effort is not supported on {model}"
                )
            return self._reply(self.replies.pop(0) if self.replies else {})

        @staticmethod
        def _reply(payload: dict[str, Any]):  # noqa: ANN205 - a context manager, as the SDK's is
            import contextlib
            import json as _json
            from types import SimpleNamespace

            message = SimpleNamespace(
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                stop_reason="end_turn",
                stop_details=None,
                content=[SimpleNamespace(type="text", text=_json.dumps(payload))],
            )

            @contextlib.contextmanager
            def opened():
                yield SimpleNamespace(get_final_message=lambda: message)

            return opened()

    def _client(self, monkeypatch, model: str, effort: str, replies: list[dict[str, Any]]):
        import pytest

        anthropic = pytest.importorskip("anthropic")
        from chess_variant_sandbox.llm.client import Claude

        transport = self.Transport(replies)
        monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: transport)
        return Claude(model=model, effort=effort), transport

    def test_a_comparison_judges_without_a_refused_parameter(self, monkeypatch) -> None:
        """The judge's request, built by the code that ships.

        `compare_rulesets` is what the loop calls with its judge client, and it asks in
        both orders -- the defence against position bias, and the thing that had never
        once run -- so two requests have to reach the transport. That the loop reaches
        `compare_rulesets` at all, with two candidates, is
        `TestPairwiseJudge::test_a_second_candidate_is_compared_with_the_first`; what only
        this test holds is the real `Claude` building the request, so it no longer plays a
        second two-generation `evolve` to get there (Q377b).
        """
        from chess_variant_sandbox.llm.client import JUDGE_MODEL
        from chess_variant_sandbox.llm.compare import compare_rulesets

        variant = copy.deepcopy(GOOD_SPEC)
        variant["name"] = "lancers II"
        variant["board"] = {"width": 6, "height": 6}

        judge, transport = self._client(
            monkeypatch, JUDGE_MODEL, "medium", [dict(PREFERS_FIRST), dict(PREFERS_SECOND)]
        )

        comparison = compare_rulesets(GOOD_SPEC, variant, [], [], client=judge)

        assert len(transport.requests) == 2, "the judge never reached the transport"
        for request in transport.requests:
            assert request["model"] == JUDGE_MODEL
            assert "thinking" not in request
            assert "effort" not in request["output_config"]
            assert request["output_config"]["format"]["type"] == "json_schema"

        assert comparison.agreed and comparison.winner is not None

    def test_the_proposer_keeps_the_parameters_its_model_does_take(
        self, tmp_path, monkeypatch
    ) -> None:
        """The other direction: the fix must not cost the expensive call its thinking.

        And it must not quietly cost the proposal its *contract* either. The rule spec
        schema has no response-format form -- the grammar compiler refuses 67 optional
        properties -- so the request carries no format block, and the schema has to be in
        the system prompt where the model can still read it. A run where neither is true
        is a proposer with no contract at all.
        """
        from chess_variant_sandbox.llm.client import PROPOSAL_MODEL

        proposer, transport = self._client(
            monkeypatch, PROPOSAL_MODEL, "high", [copy.deepcopy(GOOD_SPEC)]
        )

        archive = evolve(
            _config(tmp_path, judge=False), proposer=proposer, report=lambda _: None
        )

        assert [e.status for e in archive.entries] == ["evaluated"]
        request = transport.requests[0]
        assert request["thinking"] == {"type": "adaptive"}
        assert request["output_config"]["effort"] == "high"
        assert "format" not in request["output_config"]

        stable = request["system"][0]["text"]
        assert '"schema_version"' in stable and '"additionalProperties": false' in stable
        assert "Return one complete ruleset as a single JSON object" in transport.requests[0][
            "messages"
        ][0]["content"]

    def test_a_reply_cut_off_at_max_tokens_is_reported_as_one(self, monkeypatch) -> None:
        """Not as a JSON delimiter error out of the middle of a generation.

        A proposal spends tokens on thinking before the JSON starts, so this is the
        failure a tight cap produces, and `json.loads` describes it as a missing brace.
        """
        from types import SimpleNamespace

        import pytest

        from chess_variant_sandbox.llm.client import PROPOSAL_MODEL, LLMUnavailable

        client, transport = self._client(monkeypatch, PROPOSAL_MODEL, "high", [])

        def truncated(**request: Any):
            import contextlib

            message = SimpleNamespace(
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                stop_reason="max_tokens",
                stop_details=None,
                content=[SimpleNamespace(type="text", text='{"name": "half a ru')],
            )

            @contextlib.contextmanager
            def opened():
                yield SimpleNamespace(get_final_message=lambda: message)

            return opened()

        monkeypatch.setattr(transport, "stream", truncated)
        with pytest.raises(LLMUnavailable, match="cut off at max_tokens"):
            client.structured(system="s", user="u", schema={"type": "object"}, max_tokens=64)


def test_a_model_free_run_does_not_draw_a_generation_it_cannot_use(tmp_path) -> None:
    """"Propose from scratch" needs the operator that can.

    At `--mechanical-rate 1.0` roughly `explore_rate` of every run's generations chose
    no parent, had no model to start fresh with, and produced nothing -- a third of a
    run doing nothing, reported as a line of prose. The draw itself is unchanged, so a
    run *with* a model explores exactly as often as before.
    """
    import json

    spec_path = tmp_path / "seed.json"
    spec_path.write_text(json.dumps(GOOD_SPEC), encoding="utf-8")

    archive = evolve(
        _config(
            tmp_path / "run",
            generations=4,
            # The default, which is what made the old behaviour a third of a run.
            explore_rate=0.3,
            judge=False,
            mechanical_rate=1.0,
            seeds=(spec_path,),
            batch=2,
        ),
        proposer=None,
        report=lambda _: None,
    )

    children = [entry for entry in archive.entries if entry.parent]
    generations = {entry.generation for entry in children}
    assert generations == {1, 2, 3, 4}, sorted(generations)
