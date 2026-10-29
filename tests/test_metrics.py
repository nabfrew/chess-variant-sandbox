"""The interestingness index.

The tests that matter most here are the ones about *how* components combine, not about
any individual measurement. An index is easy to build and easy to make worthless, and
the two failure modes both showed up during development: cheap components drowning
expensive ones, and a broken game averaging its way to a respectable score.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_variant_sandbox.metrics import (
    WeightsError,
    load_weights,
    parse_weights,
    score_components,
)
from chess_variant_sandbox.metrics.components import Components
from chess_variant_sandbox.metrics.scoring import DEFAULT_WEIGHTS, Curve

# The committed rulesets that declare no capture matrix, on which one must be inert.
_WITHOUT_A_MATRIX = [
    path
    for path in sorted((Path(__file__).resolve().parents[1] / "rulesets").glob("*.json"))
    if not any("captures" in piece for piece in json.loads(path.read_text(encoding="utf-8"))["pieces"])
]


class TestCurves:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.0, 1.0), (30.0, 1.0), (90.0, 0.5), (150.0, 0.0), (400.0, 0.0)],
    )
    def test_lower_better(self, value: float, expected: float) -> None:
        curve = Curve("lower_better", {"good_at": 30, "zero_at": 150})
        assert curve.score(value) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.0, 0.0), (0.5, 0.0), (0.7, 0.5), (0.9, 1.0), (1.0, 1.0)],
    )
    def test_higher_better(self, value: float, expected: float) -> None:
        curve = Curve("higher_better", {"zero_at": 0.5, "good_at": 0.9})
        assert curve.score(value) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.0, 0.0), (8.0, 0.0), (19.0, 0.5), (30.0, 1.0), (100.0, 1.0), (270.0, 0.5), (400.0, 0.0)],
    )
    def test_band_falls_off_at_both_ends(self, value: float, expected: float) -> None:
        """Too much is as bad as too little, for length, branching and swinginess alike."""
        curve = Curve("band", {"zero_low": 8, "good_low": 30, "good_high": 140, "zero_high": 400})
        assert curve.score(value) == pytest.approx(expected, abs=0.02)


class TestWeightsFile:
    def test_the_shipped_versions_load(self) -> None:
        for version in ("v1", "v2"):
            weights = load_weights(version)
            assert weights.version == version
            assert weights.weights

    def test_an_unknown_version_names_what_is_available(self) -> None:
        with pytest.raises(WeightsError, match="available"):
            load_weights("v99")

    def test_a_curve_with_points_out_of_order_is_rejected(self) -> None:
        with pytest.raises(WeightsError, match="out of order"):
            parse_weights(
                {
                    "version": "bad",
                    "components": {
                        "x": {
                            "weight": 1,
                            "curve": "higher_better",
                            "points": {"zero_at": 10, "good_at": 2},
                        }
                    },
                }
            )

    def test_a_zero_weight_component_is_rejected(self) -> None:
        with pytest.raises(WeightsError, match="positive weight"):
            parse_weights(
                {
                    "version": "bad",
                    "components": {
                        "x": {
                            "weight": 0,
                            "curve": "higher_better",
                            "points": {"zero_at": 0, "good_at": 1},
                        }
                    },
                }
            )


def _weights(llm_weight: float = 0.0):
    return parse_weights(
        {
            "version": "test",
            "llm_weight": llm_weight,
            "components": {
                "cheap_a": {"family": "cheap", "weight": 0.4, "curve": "higher_better",
                            "points": {"zero_at": 0, "good_at": 1}},
                "cheap_b": {"family": "cheap", "weight": 0.4, "curve": "higher_better",
                            "points": {"zero_at": 0, "good_at": 1}},
                "hard": {"family": "hard", "weight": 0.4, "curve": "higher_better",
                         "points": {"zero_at": 0, "good_at": 1}},
            },
            "gates": [
                {"component": "hard", "minimum": 0.1, "reason": "must have some of this"}
            ],
        }
    )


def _components(**raw: float) -> Components:
    return Components(raw=raw, diagnostics={"games": 10})


class TestCombination:
    def test_a_failing_family_cannot_be_averaged_away(self) -> None:
        """The bug that made the dull control score 0.64 under the first weights.

        Two families, one perfect and one nearly zero. An arithmetic mean calls that a
        respectable two-thirds; the geometric mean across families calls it what it is.
        """
        weights = _weights()
        scores = weights.score_all(_components(cheap_a=1.0, cheap_b=1.0, hard=0.05).raw)
        geometric, _ = weights.combine(scores)
        arithmetic = weights.arithmetic_total(scores)
        assert arithmetic > 0.65
        assert geometric < 0.40

    def test_a_balanced_candidate_beats_a_lopsided_one_of_equal_mean(self) -> None:
        weights = _weights()
        balanced = weights.combine(weights.score_all(_components(cheap_a=0.6, cheap_b=0.6, hard=0.6).raw))[0]
        lopsided = weights.combine(weights.score_all(_components(cheap_a=1.0, cheap_b=1.0, hard=0.2).raw))[0]
        assert balanced > lopsided

    def test_a_single_zero_family_leaves_a_usable_gradient(self) -> None:
        """A hard zero would flatten every broken candidate onto the same score.

        The search still needs to tell a nearly-good game from a hopeless one.
        """
        weights = _weights()
        worse = weights.combine(weights.score_all(_components(cheap_a=0.1, cheap_b=0.1, hard=0.0).raw))[0]
        better = weights.combine(weights.score_all(_components(cheap_a=0.9, cheap_b=0.9, hard=0.0).raw))[0]
        assert 0.0 < worse < better


class TestGatesAndResults:
    def test_a_gate_failure_marks_a_result_unviable(self) -> None:
        weights = _weights()
        result = score_components("h", "broken", _components(cheap_a=1.0, cheap_b=1.0, hard=0.0), weights)
        assert not result.viable
        assert "must have some of this" in result.violations[0]

    def test_a_missing_measurement_is_an_error_not_a_silent_reweight(self) -> None:
        """Dropping a component would change what the index means without changing its version."""
        weights = _weights()
        with pytest.raises(WeightsError, match="not present"):
            weights.score_all({"cheap_a": 1.0, "cheap_b": 1.0})

    def test_the_weights_version_travels_with_the_score(self) -> None:
        result = score_components("h", "g", _components(cheap_a=0.5, cheap_b=0.5, hard=0.5), _weights())
        assert result.weights_version == "test"

    def test_the_judge_blends_in_at_the_declared_weight_only(self) -> None:
        weights = _weights(llm_weight=0.25)
        components = _components(cheap_a=0.4, cheap_b=0.4, hard=0.4)
        without = score_components("h", "g", components, weights)
        with_judge = score_components("h", "g", components, weights, llm_score=1.0)
        assert with_judge.total == without.total  # the objective half is untouched
        assert with_judge.combined == pytest.approx(0.75 * without.total + 0.25)
        assert with_judge.headline == with_judge.combined

    def test_the_weakest_components_are_reported_for_steering(self) -> None:
        result = score_components(
            "h", "g", _components(cheap_a=0.9, cheap_b=0.8, hard=0.1), _weights()
        )
        assert result.weakest(1)[0][0] == "hard"

    def test_a_result_serialises_to_json(self) -> None:
        result = score_components("h", "g", _components(cheap_a=0.5, cheap_b=0.5, hard=0.5), _weights())
        assert json.loads(json.dumps(result.as_dict()))["weights_version"] == "test"


class TestStageTwelveMeasurements:
    """The seven candidates from stage 12, now `docs/archive/plan-stages-9-16.md`.

    Measured, reported in `docs/reference/stage-12-measurements.md`, and weighted by
    nothing. These tests hold the properties the report's conclusions rest on -- if one
    of them stops being true the report is stale and its recommendations are void.
    """

    # The seven, and what the report concluded about each. Stage 12 measured them and
    # weighted none; stage 10 took three of them on that evidence and left four alone.
    TAKEN = {"progress_irreversibility", "instance_participation", "lead_conversion"}
    NOT_TAKEN = {
        "capture_availability",     # -0.84 with piece_utilisation, on raw values
        "position_revisit_rate",    # spread narrower than the mandated curve span
        "threat_asymmetry",         # re-read after schema 3; still unweighted
        "traversal_time",           # a diagnostic; scoring it asserts slower is better
    }

    def test_nothing_the_report_rejected_is_scored_anywhere(self) -> None:
        """The half of stage 12's promise that outlives stage 12.

        A component the report declined is not a component waiting for a version to be
        quietly added to. Moving one of these into a weights file means writing down
        what changed about the evidence.
        """
        from chess_variant_sandbox.metrics.scoring import WEIGHTS_DIR

        for path in sorted(WEIGHTS_DIR.glob("*.json")):
            scored = set(load_weights(path.stem).components())
            assert not (scored & self.NOT_TAKEN), (
                f"{path.stem} scores {scored & self.NOT_TAKEN}"
            )

    def test_the_survivors_arrive_in_v5_and_no_earlier(self) -> None:
        """Weights are versioned and a re-anchor is a bump, never an edit.

        Every score records the version that produced it, so the guarantee that makes a
        leaderboard readable is that v3 and v4 still mean what they meant. A component
        appearing in an older file would silently change every number computed under it.
        """
        for version in ("v1", "v2", "v3", "v4"):
            scored = set(load_weights(version).components())
            assert not (scored & self.TAKEN), f"{version} scores {scored & self.TAKEN}"
        for version in ("v5", "v6", "v7"):
            assert set(load_weights(version).components()) >= self.TAKEN

    def test_the_irreversible_series_is_the_one_the_record_has_always_carried(self) -> None:
        """A property, not a copy. Renaming the JSON key orphans every persisted run,
        because `from_dict` does `cls(**payload)`."""
        from chess_variant_sandbox.tournament.record import GameRecord

        record = GameRecord(
            ruleset_hash="h", ruleset_name="g", seed=0, agents=("a", "b"),
            moves=[], winner=None, reason="draw", plies=3, captures=[0, 1, 0],
        )
        assert record.irreversible is record.captures
        assert "irreversible" not in record.as_dict()
        assert record.as_dict()["captures"] == [0, 1, 0]

    def test_threat_asymmetry_is_zero_when_every_piece_threatens_alike(self) -> None:
        """The "fifteen rooks" case, which is the whole reason this measurement exists.

        The video's target design is a game where every piece *moves* identically and
        which is interesting anyway, because the capture relation carries the
        asymmetry. A move-set-diversity component would score that as degenerate; this
        one scores the uniform-threat case as degenerate and leaves the other alone.
        """
        from pathlib import Path

        from chess_variant_sandbox.engine.compile import compile_spec
        from chess_variant_sandbox.metrics.components import _threat_asymmetry
        from chess_variant_sandbox.spec import load_spec

        rulesets = Path(__file__).resolve().parents[1] / "rulesets"
        assert _threat_asymmetry(compile_spec(load_spec(rulesets / "dull.json"))) == 0.0
        assert _threat_asymmetry(compile_spec(load_spec(rulesets / "chess.json"))) > 0.0

    def test_threat_asymmetry_reads_the_capture_matrix(self) -> None:
        """Stage 12's "re-read it then", collected.

        The component was written for exactly one design -- pieces that move alike and
        differ only in what they may take -- and until the matrix was wired in it scored
        that design 0.000, indistinguishable from the fifteen-rook control. It is the
        corpus maximum now, above every real game, which is what CLAUDE.md has claimed
        for it since schema 3.

        Still weighted by nothing. The re-read was the promise; adopting it is a weights
        version, and the only games that exercise it are six negatives by one designer.
        """
        from pathlib import Path

        from chess_variant_sandbox.engine.compile import compile_spec
        from chess_variant_sandbox.metrics.components import _threat_asymmetry
        from chess_variant_sandbox.spec import load_spec

        rulesets = Path(__file__).resolve().parents[1] / "rulesets"
        cycle = _threat_asymmetry(compile_spec(load_spec(rulesets / "rps-v1.json")))
        assert cycle == 0.5
        assert cycle > _threat_asymmetry(compile_spec(load_spec(rulesets / "chess.json")))
        assert _threat_asymmetry(compile_spec(load_spec(rulesets / "dull.json"))) == 0.0

    # One test per committed ruleset without a matrix (Q749): the walk was one test of
    # 3.8 s in the gate.
    @pytest.mark.parametrize("path", _WITHOUT_A_MATRIX, ids=lambda path: path.stem)
    def test_a_capture_matrix_is_the_only_thing_that_moved(self, path: Path) -> None:
        """Inert on every ruleset that does not declare one.

        Not a nicety: the whole corpus is scored under versions that predate this, and a
        component that shifted on chess would invalidate a stage-12 report row that no
        longer has the games to be re-derived from.
        """
        import json

        from chess_variant_sandbox.engine.compile import compile_spec
        from chess_variant_sandbox.metrics.components import _threat_asymmetry
        from chess_variant_sandbox.spec import load_spec

        spec = json.loads(path.read_text(encoding="utf-8"))
        assert _threat_asymmetry(compile_spec(load_spec(path))) == pytest.approx(
            _geometry_only_asymmetry(spec)
        ), path.name

    def test_traversal_time_reads_movement_and_not_capture(self) -> None:
        """Found by measuring: a ruleset with captures switched off has no capture
        geometry at all, and reading reach off the capture tables scored it as
        instantaneous when the truth is that it never crosses the board."""
        from chess_variant_sandbox.engine.compile import MODE_CAPTURE, MODE_MOVE, compile_spec
        from chess_variant_sandbox.metrics.components import _displacements, _traversal_time

        move_only = {
            "schema_version": 1,
            "name": "no captures",
            "board": {"width": 8, "height": 8},
            "pieces": [
                {"id": "K", "royal": True,
                 "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]],
                            "symmetry": "rotate4", "mode": "move"}]}
            ],
            "setup": {"mirror": True, "placements": [{"piece": "K", "squares": ["a1"]}]},
            "rules": {"win": "checkmate"},
        }
        compiled = compile_spec(move_only)
        assert not any(_displacements(compiled, MODE_CAPTURE))
        assert any(_displacements(compiled, MODE_MOVE))
        assert _traversal_time(compiled) == pytest.approx(7.0)

    def test_lead_conversion_is_absent_rather_than_zero_without_enough_evidence(self) -> None:
        """A corpus of draws has nothing to say about whether a lead converts, and 0.0
        would say the opposite of nothing. `score_components` already handles a
        component a corpus cannot supply."""
        from chess_variant_sandbox.metrics.components import _lead_conversion
        from chess_variant_sandbox.tournament.record import GameRecord

        drawn = [
            GameRecord(ruleset_hash="h", ruleset_name="g", seed=i, agents=("a", "b"),
                       moves=[], winner=None, reason="draw", plies=40,
                       material=[0.0] * 40)
            for i in range(20)
        ]
        assert _lead_conversion(drawn)[0] is None

        decisive = [
            GameRecord(ruleset_hash="h", ruleset_name="g", seed=i, agents=("a", "b"),
                       moves=[], winner=0, reason="checkmate", plies=40,
                       material=[3.0] * 40)
            for i in range(20)
        ]
        value, detail = _lead_conversion(decisive)
        assert value == 1.0, "seat 0 was ahead and seat 0 won, every time"
        assert detail["qualifying_games"] == 20


class TestSpecFactsAreMeasuredOnce:
    """The rules-only measurements are memoised, and memoising them changed nothing.

    ``bootstrap_index`` re-measures a ruleset 200 times over corpora resampled from one
    tournament, and three of the measurements cannot move between resamples because
    they never read a record. Recomputing them was 116.7 ms of every one of those calls
    on chess -- over half the cost of scoring a ruleset.
    """

    def _chess(self):
        from pathlib import Path

        from chess_variant_sandbox.spec import load_spec

        return load_spec(Path(__file__).resolve().parents[1] / "rulesets" / "chess.json")

    def test_the_same_spec_is_measured_once(self) -> None:
        from chess_variant_sandbox.metrics.components import spec_facts

        assert spec_facts(self._chess()) is spec_facts(self._chess())

    def test_the_memo_holds_what_the_direct_measurement_holds(self) -> None:
        """The invisibility claim, checked rather than asserted."""
        from chess_variant_sandbox.engine.compile import compile_spec
        from chess_variant_sandbox.metrics.components import (
            _threat_asymmetry,
            _traversal_time,
            spec_facts,
        )
        from chess_variant_sandbox.spec.complexity import measure_complexity

        spec = self._chess()
        compiled = compile_spec(spec)
        facts = spec_facts(spec)
        assert facts.complexity == measure_complexity(spec)
        assert facts.threat_asymmetry == _threat_asymmetry(compiled)
        assert facts.traversal_time == _traversal_time(compiled)

    def test_the_key_is_the_whole_spec_and_not_its_mechanics_hash(self) -> None:
        """Two specs differing only in flavour text share a mechanics hash and are the
        same game -- but ``description_words`` is a complexity count that reads the
        description, so keying on the hash would report one spec's prose for another.
        """
        from chess_variant_sandbox.metrics.components import spec_facts

        spec = self._chess()
        retold = dict(spec, description="the same game, described in eight fewer words")
        assert spec_facts(spec) is not spec_facts(retold)
        assert spec_facts(retold).complexity.description_words == 8
        assert spec_facts(spec).complexity.description_words == len(
            str(spec["description"]).split()
        )

    def test_the_cache_is_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A design run compiles thousands of distinct specs in one process, and every
        entry holds that spec's whole canonical JSON.

        The bound is shrunk to two, and the cache emptied, so three times the bound is six
        specs rather than twenty-four (Q377); the eviction is the same code either way.
        """
        from collections import OrderedDict

        from chess_variant_sandbox.metrics import components

        monkeypatch.setattr(components, "_FACTS_CACHE_SIZE", 2)
        monkeypatch.setattr(components, "_FACTS", OrderedDict())
        spec = self._chess()
        for index in range(components._FACTS_CACHE_SIZE * 3):
            components.spec_facts(dict(spec, description=f"variation {index}"))
        assert len(components._FACTS) == components._FACTS_CACHE_SIZE


class TestTheRosterCanBeTheProblem:
    """`ladder_monotonicity` promoted from a diagnostic to a gate on the *instrument*.

    Antichess is the worked example. Its 6400-node search rates level with a random
    mover and its 400-node search finishes last, because the spec-derived evaluator
    maximises material in a game won by shedding it. Every skill number for such a
    ruleset describes the agents, and scoring it as a shallow game would be the same
    error as letting the cheap gate reject shatranj for being slow.

    The distinction this holds down is between "bad game" and "wrong instrument", and it
    has to survive in three places: the total, the violation text, and the prose the
    proposer is shown.
    """

    def _measured(self, monotonicity: float) -> Components:
        raw = {name: 0.5 for name in load_weights(DEFAULT_WEIGHTS).components()}
        raw["ladder_monotonicity"] = monotonicity
        raw.update({"mean_plies": 60, "decisive_rate": 0.7, "quick_kill_rate": 0.0})
        # Every gate has to have something to read, or the result is unviable for a
        # reason this class is not about. The worst-half keys are what v10's outcome
        # gates read; on a single set of games they are the pooled values.
        raw.update({
            "mean_plies_worst_half": 60,
            "decisive_rate_worst_half": 0.7,
            "quick_kill_rate_worst_half": 0.0,
            "stall_rate_worst_half": 0.5,
        })
        return Components(raw=raw, diagnostics={"games": 144})

    def test_an_ordered_ladder_scores_normally(self) -> None:
        result = score_components(
            "h", "readable", self._measured(1.0), load_weights(DEFAULT_WEIGHTS)
        )
        assert result.unreadable == ()
        assert result.total > 0.0

    def test_a_ladder_that_orders_nothing_withholds_the_depth_components(self) -> None:
        result = score_components(
            "h", "unreadable", self._measured(0.33), load_weights(DEFAULT_WEIGHTS)
        )
        from chess_variant_sandbox.metrics.calibration import SKILL_DERIVED

        # Every ladder-derived component the *current* version scores, rather than a
        # list of names: v10 drops two of the three v9 had, and a test naming them would
        # have gone quiet rather than red.
        scored = set(load_weights(DEFAULT_WEIGHTS).components())
        assert SKILL_DERIVED & scored, "no ladder-derived component left to withhold"
        assert (SKILL_DERIVED & scored) <= set(result.unreadable)
        assert "skill_gradient" in result.unreadable
        # ...and nothing that does not depend on the ladder.
        assert "rule_yield" not in result.unreadable
        assert "blunder_recoverability" not in result.unreadable, (
            "forcing one arbitrary move does not need the ladder to be ordered"
        )

    def test_it_is_reported_as_unmeasured_rather_than_as_a_bad_game(self) -> None:
        """No partial total, deliberately.

        A geometric mean over six families is a different quantity from one over seven,
        so scoring what is left would invite exactly the comparison it cannot support --
        and the loop would read the smaller number as "worse game" when what happened is
        "wrong instrument".
        """
        result = score_components(
            "h", "unreadable", self._measured(0.33), load_weights(DEFAULT_WEIGHTS)
        )
        assert not result.viable
        assert result.families == {}
        assert result.contributions == {}
        assert any("cannot read this ruleset" in v for v in result.violations)
        # The measurements that *were* taken are still reported: the next question is
        # always "how badly?".
        assert result.scores

    def test_the_proposer_is_told_the_instrument_failed_not_the_design(self) -> None:
        from chess_variant_sandbox.llm.feedback import describe

        result = score_components(
            "h", "unreadable", self._measured(0.33), load_weights(DEFAULT_WEIGHTS)
        )
        text = describe(result, {"pieces": [{"id": "K", "name": "king"}]})
        assert "rejected outright" not in text
        assert "different kind of opponent" in text
        # And the leak invariant still holds on this path: no component names.
        for component in load_weights(DEFAULT_WEIGHTS).components():
            assert component not in text.lower()
            assert component.replace("_", " ") not in text.lower()

    def test_a_larger_search_rating_worse_is_the_instrument_too(self) -> None:
        """The case a share of rising steps cannot see, found by widening the roster.

        `ladder_monotonicity` counts how many budget steps rose. Under a ladder with no
        relationship at all that share averages 0.5, so a threshold at 0.5 admits half
        of pure noise -- and going from four budgets to six pushed antichess from 0.33
        to 0.60, straight over the line, while its `skill_gradient` stayed at -0.56.

        A larger search rating *worse* cannot be a property of a game. It is the
        spec-derived evaluator maximising material in a game won by shedding it.
        """
        from chess_variant_sandbox.metrics.index import withheld_components

        weights = load_weights(DEFAULT_WEIGHTS)
        raw = {name: 0.5 for name in weights.components()}
        readable = {**raw, "ladder_monotonicity": 0.6, "skill_gradient": 40.0}
        inverted = {**raw, "ladder_monotonicity": 0.6, "skill_gradient": -0.56}
        assert not withheld_components(Components(raw=readable), weights)
        assert withheld_components(Components(raw=inverted), weights)

    def test_antichess_is_the_ruleset_this_was_written_for(self) -> None:
        """Held to the real corpus rather than to a synthetic one, because the threshold
        is only interesting if a real ruleset sits on the wrong side of it."""
        from chess_variant_sandbox.metrics.index import withheld_components

        raw = {name: 0.5 for name in load_weights(DEFAULT_WEIGHTS).components()}
        weights = load_weights(DEFAULT_WEIGHTS)
        for value, expected in ((0.33, True), (0.5, False), (1.0, False)):
            components = Components(raw={**raw, "ladder_monotonicity": value})
            assert bool(withheld_components(components, weights)) is expected


def _glossary_records():
    """A synthetic corpus wide enough to emit every raw key.

    Three searching rungs, because the top-three seat reading needs a band of three that
    all search and stays silent on a shorter ladder. A two-agent corpus leaves its two
    keys unmeasured and the glossary then reads as if it explained something that does
    not exist.
    """
    from chess_variant_sandbox.tournament.record import GameRecord

    ladder = ("minimax:nodes=100", "minimax:nodes=200", "minimax:nodes=400")
    return [
        GameRecord(
            ruleset_hash="h", ruleset_name="chess", seed=seed,
            agents=(ladder[seed % 3], ladder[(seed + 1) % 3]), moves=list(range(30)),
            winner=seed % 2, reason="checkmate", plies=30,
            branching=[20] * 30, material=[0.0] * 30,
            # `captures` is the per-ply irreversibility flag; see GameRecord.
            captures=[seed % 2] * 30,
        )
        for seed in range(6)
    ]


class TestGlossary:
    """What the numbers mean. The index has sixteen components and, to anybody who has
    not read this repository, sixteen unlabelled numbers with a seventeenth on top."""

    def test_every_scored_component_is_explained(self) -> None:
        """In every shipped version, not just the current one.

        A component arrives in a weights file and a glossary entry is written by
        somebody looking at it. The version that gets forgotten is the old one, which
        still has scores on disk that the interface will happily render.
        """
        from pathlib import Path

        from chess_variant_sandbox.metrics.glossary import MEASUREMENTS
        from chess_variant_sandbox.metrics.scoring import WEIGHTS_DIR, load_weights

        for path in sorted(Path(WEIGHTS_DIR).glob("*.json")):
            weights = load_weights(path.stem)
            missing = [name for name in weights.components() if name not in MEASUREMENTS]
            assert not missing, f"{path.stem} scores {missing} with nothing to say about it"
            gated = [g.component for g in weights.gates if g.component not in MEASUREMENTS]
            assert not gated, f"{path.stem} gates on {gated} with nothing to say about it"
            families = {weights.families[name] for name in weights.components()}
            from chess_variant_sandbox.metrics.glossary import FAMILIES

            assert not families - set(FAMILIES), f"{path.stem}: unexplained families"

    def test_every_measurement_a_corpus_produces_is_explained(self, chess_spec) -> None:
        """The other direction, and the one that matters for the raw block.

        A page showing the raw measurements shows the scored ones, the gated ones and
        the pure diagnostics side by side, and a reader cannot tell which is which
        without being told. So the glossary has to cover what `compute_components`
        actually emits, not only what some weights file happens to score.
        """
        from chess_variant_sandbox.metrics.components import compute_components
        from chess_variant_sandbox.metrics.glossary import MEASUREMENTS
        from chess_variant_sandbox.tournament.ratings import rate

        records = _glossary_records()
        ratings = rate(records)
        from chess_variant_sandbox.metrics.components import KEYED_COMPONENTS
        from chess_variant_sandbox.tournament.analysis import ANALYSIS_COMPONENTS
        from chess_variant_sandbox.tournament.heuristics import HEURISTIC_COMPONENTS

        produced = set(compute_components(chess_spec, records, ratings).raw)
        # The second and third passes are separate producers and a short synthetic
        # corpus reaches neither, so they are named rather than measured here. Leaving
        # the analysis pass out would have let all six of its components go unexplained;
        # the heuristic pass needs a committed store on top of that. The two that read
        # the pass's position keys are named in `metrics` rather than beside the six.
        produced |= set(ANALYSIS_COMPONENTS) | set(KEYED_COMPONENTS)
        produced |= set(HEURISTIC_COMPONENTS)
        missing = sorted(name for name in produced if name not in MEASUREMENTS)
        assert not missing, f"measured but unexplained: {missing}"

    def test_the_glossary_names_nothing_that_does_not_exist(self, chess_spec) -> None:
        """A stale entry is worse than a missing one: it describes a rule that is gone."""
        from pathlib import Path

        from chess_variant_sandbox.metrics.components import KEYED_COMPONENTS, compute_components
        from chess_variant_sandbox.metrics.glossary import MEASUREMENTS
        from chess_variant_sandbox.metrics.scoring import WEIGHTS_DIR, load_weights
        from chess_variant_sandbox.tournament.analysis import ANALYSIS_COMPONENTS
        from chess_variant_sandbox.tournament.heuristics import HEURISTIC_COMPONENTS
        from chess_variant_sandbox.tournament.ratings import rate

        records = _glossary_records()

        produced = set(
            compute_components(chess_spec, records, rate(records)).raw
        ) | set(ANALYSIS_COMPONENTS) | set(KEYED_COMPONENTS) | set(HEURISTIC_COMPONENTS)
        for path in Path(WEIGHTS_DIR).glob("*.json"):
            produced |= set(load_weights(path.stem).components())

        stale = sorted(set(MEASUREMENTS) - produced)
        assert not stale, f"glossary explains measurements nothing produces: {stale}"

    def test_the_method_is_explained_at_one_run_and_at_four(self) -> None:
        """`docs/archive/plan-four-seeds.md` §7: both readings of the interval, never one hedge.

        A library score is the mean of four runs and a visitor's own is one run, and the
        two intervals are different claims. The four-run one is the sampling error of the
        mean, and it carries Q343's figure for a fresh set of four (about five times in
        six); the one-run one carries the replay figure. Saying only one of them would make
        the other a lie.
        """
        from chess_variant_sandbox.metrics.glossary import explain

        said = " ".join(
            entry["heading"] + " " + entry["body"]
            for entry in explain()["howItWorks"]
        ).lower()
        assert "one run" in said and "mean of four runs" in said
        assert "within four of these standard deviations" in said
        assert "four more seeds" in said and "five times in six" in said
        # Never the pool. Scoring the four runs' games as one corpus is a different
        # quantity and would need its own weights version.
        assert "pooled" in said

    def test_the_direction_is_read_from_the_weights_and_never_asserted(self) -> None:
        """The one thing a glossary must not do is restate the curve.

        `mean_branching` and `progress_irreversibility` are bands: a glossary calling
        either "more is better" would be telling a reader to optimise something no
        player wants, which is precisely what the weights files exist to stop. So the
        shape comes out of the file, and the prose is checked for not contradicting it.
        """
        from chess_variant_sandbox.metrics.glossary import explain

        report = explain("v5")
        bands = [
            name for name, entry in report["components"].items()
            if entry["curve"]["kind"] == "band"
        ]
        assert "mean_branching" in bands and "progress_irreversibility" in bands
        for name in bands:
            reading = report["components"][name]["reading"].lower()
            assert "higher is better" not in reading
            # A band has two bad ends, and the prose has to say what both of them are
            # -- "high is good" on a band is the exact error the weights files exist
            # to prevent, and silence about the top end reads as exactly that.
            assert "band" in reading or "too" in reading or "optional" in reading, name

        for name, entry in report["components"].items():
            assert entry["curve"]["points"], name
            assert entry["what"] and entry["how"] and entry["reading"], name

    def test_the_glossary_is_not_shown_to_the_proposing_model(self) -> None:
        """The index stated in prose is the leak in its purest form.

        `tests/test_loop.py` holds the brief and the per-candidate feedback to the rule
        that no component name reaches the proposer. This module is every component
        name with an explanation attached, so the guard worth having is structural: no
        module under `chess_variant_sandbox/llm/` may import it.
        """
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "chess_variant_sandbox" / "llm"
        for path in sorted(root.glob("*.py")):
            assert "glossary" not in path.read_text(encoding="utf-8"), path.name


def _geometry_only_asymmetry(spec: dict) -> float:
    """`_threat_asymmetry` as it read before schema 3's matrix was wired in.

    The oracle for the inertness check rather than a second implementation of the
    measurement: it exists to answer "did this value move", and nothing calls it outside
    that test.
    """
    from chess_variant_sandbox.engine.compile import MODE_CAPTURE, compile_spec
    from chess_variant_sandbox.metrics.components import _displacements

    reach = _displacements(compile_spec(spec), MODE_CAPTURE)
    if len(reach) < 2:
        return 0.0

    def one_way(attacker: int, victim: int) -> bool:
        answers = {(-x, y) for x, y in reach[victim]}
        return any((-dx, -dy) not in answers for dx, dy in reach[attacker])

    pairs = [(a, b) for a in range(len(reach)) for b in range(len(reach)) if a != b]
    return sum(1 for a, b in pairs if one_way(a, b) and not one_way(b, a)) / len(pairs)


class TestHowDeepEachRungSaw:
    """Stage 27: the depth read-out, which is a diagnostic and must stay one.

    A node budget is not a fixed amount of search -- at 6,400 nodes the ``dull`` control
    completes 8.5 plies and xiangqi completes 2.0 -- so the depth family compares games
    on ladders of very different reach. Reporting that is free; gating on it is a
    threshold picked to catch one ruleset, and the plan says so out loud.
    """

    def _records(self) -> list:
        from chess_variant_sandbox.tournament.record import GameRecord

        return [
            GameRecord(
                "h", "n", seed, ("minimax:nodes=400", "random"), [1, 2], None, "test", 2,
                depth_sum=(6, 0), searches=(2, 0),
            )
            for seed in range(3)
        ]

    def test_a_rung_reports_plies_per_search_and_the_sample_behind_it(self) -> None:
        from chess_variant_sandbox.metrics.components import _depth_by_agent

        table = _depth_by_agent(self._records())
        assert table["minimax:nodes=400"] == {"mean_depth": 3.0, "searches": 6}

    def test_an_agent_that_does_not_deepen_is_absent_rather_than_zero(self) -> None:
        """"Searched nothing" and "searched and completed no ply" are different claims.

        Only the second says anything about the ruleset, and a random mover reported at
        0.00 plies beside a minimax rung that genuinely never finished one iteration
        would put the two in the same column.
        """
        from chess_variant_sandbox.metrics.components import _depth_by_agent

        assert "random" not in _depth_by_agent(self._records())

    def test_it_is_scored_by_no_weights_version(self) -> None:
        from chess_variant_sandbox.metrics.scoring import WEIGHTS_DIR

        for path in sorted(WEIGHTS_DIR.glob("*.json")):
            assert "depth_by_agent" not in set(load_weights(path.stem).components())
            assert "mean_depth" not in set(load_weights(path.stem).components())


def _ladder_record(agents: tuple[str, str], plies: int, winner: int | None, reason: str):
    """One game between two named seats, carrying only what the outcome gates read."""
    from chess_variant_sandbox.tournament.record import GameRecord

    return GameRecord(
        ruleset_hash="h",
        ruleset_name="two-games",
        seed=0,
        agents=agents,
        moves=list(range(plies)),
        winner=winner,
        reason=reason,
        plies=plies,
        branching=[20] * plies,
        material=[0.0] * plies,
        captures=[0] * plies,
    )


class TestTheGatesReadTheWorseHalfOfTheLadder:
    """The split denominator: `docs/reference/roster-confound.md` sections 3 and 4.

    `bastion` is the committed broken control and it measured `viable: true` under `v9`
    at 0.7103. It is two games, not one. Against the rungs that cannot see its three-ply
    trap it is an opening trap -- 80 quick kills in 132 games, past a maximum of 0.5.
    Against the rungs that can, it is a dead shuffle -- 64 games with no quick kill in
    them at all. Pooled, the same 80 kills read 0.408 and pass, the trap games supply
    the decisiveness that rescues the stall gate, and a mean length of 117.7 plies
    describes no game that was ever played. **Each failure is the other one's alibi.**

    The corpus below is that shape built from nothing rather than replayed from that
    run, so what is under test is the denominator and not one ruleset: on the weak half
    eight quick kills and four long wins, on the strong half nine stalls and three wins.
    Every one of the four gates passes on the pooled set, and two of them -- one from
    each half -- fire on the worst half.
    """

    WEAK_PAIRS = (
        ("greedy", "minimax:nodes=100"),
        ("minimax:nodes=100", "minimax:nodes=200"),
        ("minimax:nodes=200", "greedy"),
        # Puts the top rung on the ladder while still standing in the weak half: the
        # split reads the *weaker* seat, so a 6400-node search against `greedy` is a
        # game one of the two players cannot see the trap in.
        ("greedy", "minimax:nodes=6400"),
    )
    STRONG_PAIRS = (
        ("minimax:nodes=400", "minimax:nodes=800"),
        ("minimax:nodes=800", "minimax:nodes=1600"),
        ("minimax:nodes=1600", "minimax:nodes=6400"),
        ("minimax:nodes=400", "minimax:nodes=6400"),
    )

    def _corpus(self) -> list:
        records = []
        for index in range(12):
            pair = self.WEAK_PAIRS[index % len(self.WEAK_PAIRS)]
            plies, reason = (3, "checkmate") if index < 8 else (100, "checkmate")
            records.append(_ladder_record(pair, plies, index % 2, reason))
        for index in range(12):
            pair = self.STRONG_PAIRS[index % len(self.STRONG_PAIRS)]
            if index < 9:
                records.append(_ladder_record(pair, 250, None, "ply_limit"))
            else:
                records.append(_ladder_record(pair, 250, index % 2, "checkmate"))
        # Dropped before the split, exactly as they are dropped before every other
        # graded measurement: the weak seat is `random`, not a small budget.
        records += [
            _ladder_record(("random", "minimax:nodes=6400"), 4, 1, "checkmate")
            for _ in range(20)
        ]
        return records

    def _raw(self, chess_spec) -> dict[str, float]:
        from chess_variant_sandbox.metrics.components import compute_components
        from chess_variant_sandbox.tournament.ratings import rate

        records = self._corpus()
        return compute_components(chess_spec, records, rate(records)).raw

    def _outcome_violations(self, raw: dict[str, float], version: str) -> set[str]:
        """Which of the four outcome gates fire, under one weights version."""
        from chess_variant_sandbox.metrics.components import WORST_HALF_RATES, WORST_HALF_SUFFIX

        names = set(WORST_HALF_RATES) | {
            name + WORST_HALF_SUFFIX for name in WORST_HALF_RATES
        }
        return {
            gate.component
            for gate in load_weights(version).gates
            if gate.component in names and gate.check(raw[gate.component]) is not None
        }

    def test_the_pooled_reading_passes_every_outcome_gate(self, chess_spec) -> None:
        """The alibi, stated as the thing v9 actually does."""
        assert self._outcome_violations(self._raw(chess_spec), "v9") == set()

    def test_each_half_fails_a_gate_the_other_half_passes(self, chess_spec) -> None:
        assert self._outcome_violations(self._raw(chess_spec), "v10") == {
            "quick_kill_rate_worst_half",
            "stall_rate_worst_half",
        }

    def test_the_worst_half_is_chosen_per_gate_and_not_once_for_all_four(
        self, chess_spec
    ) -> None:
        """The two gates that fire read *different* halves, which is the whole point.

        Choosing one half and reading all four rates on it would catch whichever failure
        happened to be on that side and let the other one through -- and which side that
        is depends on the ruleset, not on the instrument.
        """
        from chess_variant_sandbox.metrics.components import compute_components
        from chess_variant_sandbox.tournament.ratings import rate

        records = self._corpus()
        detail = compute_components(
            chess_spec, records, rate(records)
        ).diagnostics["ladder_split"]
        assert detail["worst_half"]["quick_kill_rate"] == "weak"
        assert detail["worst_half"]["stall_rate"] == "strong"

    def test_the_split_is_the_median_rung_with_greedy_at_the_bottom(
        self, chess_spec
    ) -> None:
        """Pre-registered as English -- *the median rung of the graded ladder* -- and
        this is what that English resolves to on the canonical roster: `greedy` plus six
        searching budgets is seven rungs, and the fourth is 400."""
        from chess_variant_sandbox.metrics.components import compute_components
        from chess_variant_sandbox.tournament.ratings import rate

        records = self._corpus()
        detail = compute_components(
            chess_spec, records, rate(records)
        ).diagnostics["ladder_split"]
        assert detail["rungs"] == [0, 100, 200, 400, 800, 1600, 6400]
        assert detail["split_rung"] == 400
        assert detail["games"] == {"weak": 12, "strong": 12}

    def test_a_game_is_strong_when_its_weaker_seat_is_at_or_above_the_midpoint(
        self,
    ) -> None:
        """"Can both players see it?", not "can either?". A 6400-node search against
        `greedy` is a weak-half game, because one of the two seats is not searching."""
        from chess_variant_sandbox.metrics.components import _worst_half_rates

        records = [
            _ladder_record(("greedy", "minimax:nodes=6400"), 3, 0, "checkmate"),
            _ladder_record(("minimax:nodes=400", "minimax:nodes=400"), 300, None, "ply_limit"),
            _ladder_record(("minimax:nodes=100", "minimax:nodes=200"), 3, 0, "checkmate"),
            _ladder_record(("minimax:nodes=800", "minimax:nodes=1600"), 300, None, "ply_limit"),
        ]
        _, detail = _worst_half_rates(records)
        assert detail["split_rung"] == 400
        assert detail["games"] == {"weak": 2, "strong": 2}
        assert detail["rates"]["weak"]["quick_kill_rate"] == 1.0
        assert detail["rates"]["strong"]["stall_rate"] == 1.0

    def test_a_ladder_with_one_rung_falls_back_to_the_pooled_set(self) -> None:
        """A corpus the split cannot read gates exactly as it did before.

        The alternative is a gate reading an empty denominator, which would disqualify a
        ruleset for the shape of the schedule rather than for anything about the game.
        """
        from chess_variant_sandbox.metrics.components import _outcome_rates, _worst_half_rates

        records = [
            _ladder_record(("minimax:nodes=400", "minimax:nodes=400"), 3, 0, "checkmate")
            for _ in range(6)
        ]
        raw, detail = _worst_half_rates(records)
        assert detail["split_rung"] is None
        pooled = _outcome_rates(records)
        for name, value in pooled.items():
            assert raw[name + "_worst_half"] == value

    def test_the_split_reports_the_depth_each_rung_and_each_halfs_weaker_seat_completed(
        self,
    ) -> None:
        """Q529's groundwork: the budget split is a proxy for *both players complete enough
        plies to see it* (`roster-confound.md` section 4), so the detail carries the
        thing meant beside the proxy. The depth is the core's (`depth_sum`, `searches`);
        this only divides. `greedy` does not deepen, so a half whose weaker seat is
        `greedy` reads ``None``, never zero, and a tied pairing pools both seats."""
        from dataclasses import replace

        from chess_variant_sandbox.metrics.components import _worst_half_rates

        def searched(agents, depth_sum, searches):
            record = _ladder_record(agents, 3, 0, "checkmate")
            return replace(record, depth_sum=depth_sum, searches=searches)

        records = [
            searched(("greedy", "minimax:nodes=6400"), (0, 40), (0, 10)),
            searched(("minimax:nodes=400", "minimax:nodes=400"), (20, 30), (10, 10)),
            searched(("minimax:nodes=100", "minimax:nodes=200"), (5, 15), (10, 10)),
            searched(("minimax:nodes=800", "minimax:nodes=1600"), (30, 25), (10, 10)),
        ]
        _, detail = _worst_half_rates(records)
        assert detail["rung_depth"] == {
            "100": 0.5, "200": 1.5, "400": 2.5, "800": 3.0, "1600": 2.5, "6400": 4.0,
        }
        assert "0" not in detail["rung_depth"]
        # weak: greedy (no searches) and the 100-node seat; strong: both 400s and the 800.
        assert detail["weaker_seat_depth"] == {"weak": 0.5, "strong": round(80 / 30, 3)}

        only_greedy = [searched(("greedy", "minimax:nodes=6400"), (0, 40), (0, 10))]
        assert _worst_half_rates(only_greedy)[1]["weaker_seat_depth"] == {"pooled": None}

    def test_the_depth_in_the_split_moves_no_gate(self) -> None:
        """Groundwork, not a gate: the four worst-half rates read the same games and give
        the same numbers whether or not any seat recorded a search."""
        from dataclasses import replace

        from chess_variant_sandbox.metrics.components import _worst_half_rates

        bare = [
            _ladder_record(("minimax:nodes=100", "minimax:nodes=6400"), 3, 0, "checkmate"),
            _ladder_record(("minimax:nodes=400", "minimax:nodes=800"), 300, None, "ply_limit"),
        ]
        deep = [replace(r, depth_sum=(9, 99), searches=(3, 3)) for r in bare]
        assert _worst_half_rates(bare)[0] == _worst_half_rates(deep)[0]

    def test_every_worst_half_gate_is_read_in_the_direction_that_can_fail_it(
        self,
    ) -> None:
        """The one way this change could be silently wrong.

        A `_worst_half` key reports the higher of the two halves for a rate that is bad
        when high and the lower for a rate that is bad when low. If a weights file gated
        one of them the other way round it would be reading the *safer* half, and the
        split gate would be weaker than the pooled gate it replaced -- which is the
        opposite of the point, and would look exactly like a working change.
        """
        from chess_variant_sandbox.metrics.components import WORST_HALF_RATES, WORST_HALF_SUFFIX
        from chess_variant_sandbox.metrics.scoring import WEIGHTS_DIR

        seen = set()
        for path in sorted(WEIGHTS_DIR.glob("*.json")):
            for gate in load_weights(path.stem).gates:
                if not gate.component.endswith(WORST_HALF_SUFFIX):
                    continue
                base = gate.component[: -len(WORST_HALF_SUFFIX)]
                assert base in WORST_HALF_RATES, (
                    f"{path.stem} gates on {gate.component}, which nothing measures"
                )
                seen.add(base)
                if WORST_HALF_RATES[base] == "high":
                    assert gate.maximum is not None and gate.minimum is None, (
                        f"{path.stem}: {base} is worst when high and must be gated above"
                    )
                else:
                    assert gate.minimum is not None and gate.maximum is None, (
                        f"{path.stem}: {base} is worst when low and must be gated below"
                    )
        assert seen == set(WORST_HALF_RATES), (
            "a worst-half key nothing gates on is a measurement with no reader"
        )

    def test_every_pairing_lies_inside_one_half_so_the_resample_is_already_right(
        self,
    ) -> None:
        """Why the split needs no bootstrap code, only a bootstrap report.

        `bootstrap_index` stratifies on `record.agents` and draws `len(games)` from each
        stratum, so a resampled corpus has exactly the pairings the real one had. A
        pairing's weaker seat is a property of the pairing, so each half's game count is
        the same in every resample and a worst-half rate resamples correctly by
        construction. If a pairing could straddle the split, the halves would be
        random-sized between resamples and the interval would be measuring the schedule.

        What *does* need saying in the report rather than in the code: the statistic is
        the worse of two noisy halves, so its own resampled sd carries the selection the
        `max` performs. That is the right error bar for it, because it is the quantity
        the gate reads -- but it is not either half's sd and must not be quoted as one.
        """
        from chess_variant_sandbox.metrics.components import _budget

        strata: dict[tuple[str, str], set[int]] = {}
        for record in self._corpus():
            side = min(_budget(agent) for agent in record.agents)
            strata.setdefault(record.agents, set()).add(side)
        assert all(len(sides) == 1 for sides in strata.values())

    def test_the_scored_stall_rate_keeps_its_pooled_denominator(self) -> None:
        """`stall_rate` is scored *and* gated, and only the gate moves.

        Moving a scored component's denominator would move the curve that was anchored
        on the old one, and `v10` anchors nothing. So the component keeps the pooled
        value and the gate reads the worst half beside it.
        """
        for version in ("v9", "v10"):
            weights = load_weights(version)
            assert "stall_rate" in weights.weights
            assert "stall_rate_worst_half" not in weights.weights
        assert load_weights("v9").curves["stall_rate"].points == (
            load_weights("v10").curves["stall_rate"].points
        )


class TestVersionTenIsNotAReAnchor:
    """`v10` is `v9`'s curves to the digit, three components lighter.

    Two claims the campaign's pre-registered prediction rests on, and both are
    mechanical rather than editorial: no curve end moved, and no family total moved. If
    either were false the prediction would be checking the implementation against a
    corpus scored under different curves, and a miss would be unreadable.
    """

    DROPPED = ("progress_irreversibility", "gradient_persistence", "draw_slope")

    def test_exactly_three_components_left_and_they_are_the_named_three(self) -> None:
        nine = set(load_weights("v9").components())
        ten = set(load_weights("v10").components())
        assert nine - ten == set(self.DROPPED)
        assert ten - nine == set()
        assert len(ten) == 13

    def test_every_surviving_curve_is_v9s_to_the_digit(self) -> None:
        nine, ten = load_weights("v9"), load_weights("v10")
        for name in ten.components():
            assert ten.curves[name] == nine.curves[name], name

    def test_the_weight_goes_to_the_family_and_not_across_families(self) -> None:
        """The disposition rule, section 8.1, pre-registered before its table.

        Deleting the three lines instead would have cut depth from 0.2079 to 0.1386,
        pacing to 0.0792 and drama to 0.0891 -- a re-weighting *across* families that
        nobody argued for, and one the geometric mean reads as exponents.
        """
        nine, ten = load_weights("v9").family_weights(), load_weights("v10").family_weights()
        assert set(nine) == set(ten)
        for family in nine:
            assert ten[family] == pytest.approx(nine[family], abs=5e-7), family

    def test_the_survivors_share_the_dropped_weight_pro_rata(self) -> None:
        nine, ten = load_weights("v9"), load_weights("v10")
        for family in ("pacing", "drama", "depth"):
            members = [n for n in ten.components() if ten.families[n] == family]
            factor = nine.family_weights()[family] / sum(nine.weights[n] for n in members)
            for name in members:
                assert ten.weights[name] == pytest.approx(
                    nine.weights[name] * factor, abs=5e-7
                ), name

    def test_the_dropped_components_are_still_measured_and_still_explained(
        self, chess_spec
    ) -> None:
        """A component leaves a weights file; it does not leave the instrument.

        `v9` is still on disk and still has to be re-derivable, and the raw numbers are
        what a later stage would need to check whether a definition fix rehabilitates
        either of the two roster-bound ones.
        """
        from chess_variant_sandbox.metrics.components import compute_components
        from chess_variant_sandbox.metrics.glossary import MEASUREMENTS
        from chess_variant_sandbox.tournament.ratings import rate

        records = [
            _ladder_record((f"minimax:nodes={budget}", "minimax:nodes=800"), 40, 0, "checkmate")
            for budget in (100, 200, 400, 800, 1600, 6400)
            for _ in range(4)
        ]
        raw = compute_components(chess_spec, records, rate(records)).raw
        for name in self.DROPPED:
            assert name in raw or name == "progress_irreversibility", name
            assert name in MEASUREMENTS, name


class TestVersionElevenIsAReAnchor:
    """`v11` is the opposite claim to `v10`'s: its ends moved, and so did its design.

    `docs/archive/plan-variety.md` §5.4, C2. Rule yield and the elegance family leave; the variety
    pair gives way to `position_divergence`, alone at the family's 0.099; every other
    weight and curve shape is `v10`'s; every curve end was re-placed by `chess-variant-sandbox
    anchor` on `runs/v11`. Each assertion is a mechanical fact about the two files, so a
    later edit that quietly re-weighted a family, or put a curve end back, fails here.
    """

    DROPPED = ("rule_yield", "opening_diversity", "distinct_game_rate")
    SIX = ("balance", "decisions", "depth", "drama", "pacing", "variety")
    # The curves the anchor moved, read off the file it wrote (`docs/reference/
    # v11-calibration.md`, *C2, the anchor*). `stall_rate` and `lead_conversion` came out at
    # `v10`'s ends to the digit: same band extremes and same negatives' median.
    MOVED = frozenset({
        "blunder_recoverability", "decision_criticality", "first_player_advantage",
        "instance_participation", "mean_branching", "piece_utilisation", "ply_spread",
        "skill_gradient",
    })

    def test_it_was_the_default_until_v12(self) -> None:
        """`v12` (EV4) gave every end a source and took the default; `v11` stays on disk,
        because a score records the version that made it."""
        assert DEFAULT_WEIGHTS == "v12"
        assert load_weights("v11").version == "v11"
        assert load_weights("v11").sources == {}

    def test_rule_yield_and_the_pair_are_gone_and_one_component_arrived(self) -> None:
        ten = set(load_weights("v10").components())
        eleven = set(load_weights("v11").components())
        assert ten - eleven == set(self.DROPPED)
        assert eleven - ten == {"position_divergence"}
        assert len(eleven) == 11

    def test_six_families_and_variety_is_one_component(self) -> None:
        weights = load_weights("v11")
        assert sorted(weights.family_weights()) == list(self.SIX)
        variety = [n for n in weights.components() if weights.families[n] == "variety"]
        assert variety == ["position_divergence"]
        assert weights.weights["position_divergence"] == pytest.approx(0.099)
        assert weights.curves["position_divergence"].kind == "higher_better"

    def test_every_surviving_weight_and_curve_shape_is_v10s(self) -> None:
        """So the five families that stay keep `v10`'s totals, and `combine` divides by
        their sum -- the renormalisation by 1/0.9009 is the whole redistribution."""
        ten, eleven = load_weights("v10"), load_weights("v11")
        for name in set(ten.components()) & set(eleven.components()):
            assert eleven.weights[name] == ten.weights[name], name
            assert eleven.families[name] == ten.families[name], name
            assert eleven.curves[name].kind == ten.curves[name].kind, name
        assert eleven.optional == ten.optional
        tens, elevens = ten.family_weights(), eleven.family_weights()
        for family in self.SIX:
            assert elevens[family] == pytest.approx(tens[family], abs=5e-7), family
        assert sum(elevens.values()) == pytest.approx(0.9009, abs=5e-7)

    def test_the_ends_moved(self) -> None:
        ten, eleven = load_weights("v10"), load_weights("v11")
        shared = set(ten.components()) & set(eleven.components())
        moved = {n for n in shared if eleven.curves[n] != ten.curves[n]}
        assert moved == self.MOVED

    def test_position_divergences_ends_are_its_domain(self) -> None:
        """What C1 read off `anchor_curves` before the anchor ran: a band from 0.214 to
        0.983 is wide enough that `good_at` and the span floor both clamp."""
        from chess_variant_sandbox.metrics.calibration import DOMAIN

        points = load_weights("v11").curves["position_divergence"].points
        assert DOMAIN["position_divergence"] == (0.0, 1.0)
        assert points == {"good_at": 1.0, "zero_at": 0.0}

    def test_the_gates_are_v10s(self) -> None:
        assert load_weights("v11").gates == load_weights("v10").gates


class TestPieceUtilisationDenominator:
    """`piece_utilisation` divides by the types a player can hold (Q205).

    A type only the wild phase places has no player to move it and is not charged. A
    held type that never moves still is, which is how `mutant-idle-army` is caught.
    """

    @staticmethod
    def _utilisation(spec: dict, moves_by_type: list[int]) -> float:
        from chess_variant_sandbox.metrics.components import _analysis_components
        from chess_variant_sandbox.tournament.analysis import CorpusAnalysis, GameAnalysis

        game = GameAnalysis(seed=0, agents=("a", "b"), moves_by_type=moves_by_type)
        return _analysis_components(CorpusAnalysis(games=[game]), {}, spec)[
            "piece_utilisation"
        ]

    @staticmethod
    def _wild_spec() -> dict:
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "archive" / "wolves.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_a_type_only_the_wild_phase_places_is_not_held(self) -> None:
        from chess_variant_sandbox.spec.schema import pieces_a_player_can_hold

        spec = self._wild_spec()
        wild = {npc["piece"] for npc in spec["rules"]["npcs"]}
        placed = {p["piece"] for p in spec["setup"]["placements"]}
        held = pieces_a_player_can_hold(spec)

        assert wild and not wild & placed
        assert not wild & held
        assert placed <= held

    def test_a_type_only_the_wild_phase_places_is_not_charged(self) -> None:
        spec = self._wild_spec()
        wild = {npc["piece"] for npc in spec["rules"]["npcs"]}
        counts = [0 if p["id"] in wild else 10 for p in spec["pieces"]]

        assert self._utilisation(spec, counts) == pytest.approx(1.0)

    def test_a_wild_piece_captured_into_a_hand_is_held(self) -> None:
        from chess_variant_sandbox.spec.schema import pieces_a_player_can_hold

        spec = self._wild_spec()
        spec["rules"]["drops"] = {"from_captures": True}
        wild = {npc["piece"] for npc in spec["rules"]["npcs"]}

        assert wild <= pieces_a_player_can_hold(spec)

    def test_an_idle_army_is_still_charged(self, chess_spec) -> None:
        """The confined types are held and never move, so they stay in as zeros."""
        from chess_variant_sandbox.metrics.components import _normalised_entropy
        from chess_variant_sandbox.metrics.mutants import MUTANTS
        from chess_variant_sandbox.spec.schema import pieces_a_player_can_hold

        mutant = next(m for m in MUTANTS if m.name == "mutant-idle-army")
        assert mutant.expectation == "worse"
        spec = mutant.build(chess_spec)
        confined = {p["id"] for p in spec["pieces"] if "region" in p}
        assert confined

        assert pieces_a_player_can_hold(spec) == {p["id"] for p in spec["pieces"]}
        counts = [0 if p["id"] in confined else 10 for p in spec["pieces"]]
        measured = self._utilisation(spec, counts)
        assert measured < 1.0
        assert measured == pytest.approx(_normalised_entropy(counts))


_VARIETY_NAMES = ("prefix_divergence", "position_divergence", "event_divergence")
_VARIETY_AGENTS = ("minimax:nodes=100", "minimax:nodes=200")


def _variety_game(
    seed: int, moves: list[int], keys: list[int], irreversible: list[int] | None = None
):
    """One graded game and its analysis, with the position keys and events stated."""
    from chess_variant_sandbox.tournament.analysis import GameAnalysis
    from chess_variant_sandbox.tournament.record import GameRecord

    record = GameRecord(
        ruleset_hash="h", ruleset_name="variety", seed=seed, agents=_VARIETY_AGENTS,
        moves=moves, winner=None, reason="fixture", plies=len(moves),
        branching=[20] * len(moves), material=[0.0] * len(moves),
        captures=irreversible if irreversible is not None else [1] * len(moves),
    )
    game = GameAnalysis(
        seed=seed, agents=_VARIETY_AGENTS, walked_plies=len(keys), position_keys=keys
    )
    return record, game


def _variety(pairs) -> tuple[dict[str, float], dict]:
    """The three variety readings of a list of ``_variety_game`` pairs."""
    from chess_variant_sandbox.metrics import components
    from chess_variant_sandbox.tournament.analysis import CorpusAnalysis

    records = [record for record, _ in pairs]
    diagnostics: dict = {}
    raw = {"prefix_divergence": components._prefix_divergence(records)}
    raw.update(
        components._keyed_components(
            records, CorpusAnalysis(games=[game for _, game in pairs]), diagnostics
        )
    )
    return raw, diagnostics


def _commutative_corpus(games: int = 16, plies: int = 24, seed: int = 3):
    """Games in which a position is how many of each of three moves have been played.

    So two orders of the same moves transpose, positions are shared between games in
    part, and every distance is somewhere between the ends. A move of kind 0 is the one
    that cannot be undone.
    """
    import random

    rng = random.Random(seed)
    pairs = []
    for index in range(games):
        moves = [rng.randrange(3) for _ in range(plies)]
        counts = [0, 0, 0]
        keys = []
        for move in moves:
            counts[move] += 1
            keys.append(counts[0] * 10_000 + counts[1] * 100 + counts[2])
        pairs.append(_variety_game(index, moves, keys, [int(m == 0) for m in moves]))
    return pairs


class TestTheVarietyComponents:
    """`docs/archive/plan-variety.md` §3.1 to §3.3 and §5.4 A1.

    Each is a mean over graded games of a distance to the nearest *other* game, where a
    game is an identity rather than a draw. §3.2 reads the positions a game passed
    through, §3.3 only those a move that cannot be undone reached, and §3.1 the move
    list, as the comparison §3.6 asks for.
    """

    def test_a_doubled_corpus_reads_the_same(self, chess_spec) -> None:
        """Every game twice is every identity twice: no copy becomes a neighbour."""
        from chess_variant_sandbox.metrics.components import compute_components
        from chess_variant_sandbox.tournament.analysis import CorpusAnalysis
        from chess_variant_sandbox.tournament.ratings import rate

        pairs = _commutative_corpus()
        records = [record for record, _ in pairs]
        games = [game for _, game in pairs]
        once = compute_components(
            chess_spec, records, rate(records), CorpusAnalysis(games=games)
        ).raw
        doubled = records + records
        twice = compute_components(
            chess_spec, doubled, rate(doubled), CorpusAnalysis(games=games + games)
        ).raw

        for name in _VARIETY_NAMES:
            assert 0.0 < once[name] < 1.0, name
            assert twice[name] == once[name], name

    def test_a_replay_under_another_seed_is_a_second_game_at_distance_zero(self) -> None:
        """Two identities with one move list are the same game played twice."""
        first = _variety_game(0, [1, 2, 3], [10, 20, 30])
        replay = _variety_game(1, [1, 2, 3], [10, 20, 30])
        other = _variety_game(2, [4, 5, 6], [40, 50, 60])

        raw, _ = _variety([first, replay, other])
        for name in _VARIETY_NAMES:
            assert raw[name] == pytest.approx(1 / 3), name

    def test_two_orderings_of_one_position_set_read_zero(self) -> None:
        """A transposition is the same game to the position forms and not to the moves."""
        pairs = [
            _variety_game(0, [1, 2, 3], [10, 20, 30]),
            _variety_game(1, [2, 1, 3], [20, 10, 30]),
        ]
        raw, _ = _variety(pairs)

        assert raw["position_divergence"] == 0.0
        assert raw["event_divergence"] == 0.0
        assert raw["prefix_divergence"] == 1.0

    def test_a_game_with_no_event_scores_zero_on_event_divergence(self) -> None:
        """B and C share two of six events; A shares every position with B and no event."""
        eventless = _variety_game(0, [1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0])
        b = _variety_game(1, [5, 6, 7, 8], [1, 2, 3, 4])
        c = _variety_game(2, [9, 10, 11, 12], [3, 4, 5, 6])

        raw, diagnostics = _variety([eventless, b, c])
        assert raw["event_divergence"] == pytest.approx((0.0 + 2 / 3 + 2 / 3) / 3)
        assert diagnostics["eventless_games"] == 1
        # The positions it did pass through are still read by the position form.
        assert raw["position_divergence"] == pytest.approx((0.0 + 0.0 + 2 / 3) / 3)

    def test_a_game_with_no_event_is_nobodys_neighbour(self) -> None:
        """B and C read each other with A present exactly as without it, and a lone
        eventful game among eventless ones has no neighbour at all."""
        eventless = _variety_game(0, [1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0])
        b = _variety_game(1, [5, 6, 7, 8], [1, 2, 3, 4])
        c = _variety_game(2, [9, 10, 11, 12], [3, 4, 5, 6])
        also_eventless = _variety_game(3, [13, 14, 15, 16], [5, 6, 7, 8], [0, 0, 0, 0])

        with_it, _ = _variety([eventless, b, c])
        without, _ = _variety([b, c])
        assert with_it["event_divergence"] * 3 == pytest.approx(without["event_divergence"] * 2)

        alone, diagnostics = _variety([eventless, b, also_eventless])
        assert alone["event_divergence"] == 0.0
        assert diagnostics["eventless_games"] == 2

    def test_a_corpus_without_its_series_reads_no_events(self) -> None:
        """An empty irreversible series is a series nobody collected, not a game with no
        events, so the event form is withheld rather than read as zero."""
        pairs = [
            _variety_game(0, [1, 2, 3], [10, 20, 30], []),
            _variety_game(1, [2, 1, 3], [20, 10, 40], []),
        ]
        raw, _ = _variety(pairs)
        assert "position_divergence" in raw
        assert "event_divergence" not in raw

    def test_the_position_forms_need_the_keys_the_pass_persists(self, chess_spec) -> None:
        from chess_variant_sandbox.metrics.components import KEYED_COMPONENTS, compute_components
        from chess_variant_sandbox.tournament.analysis import CorpusAnalysis, GameAnalysis
        from chess_variant_sandbox.tournament.ratings import rate

        pairs = _commutative_corpus(games=6)
        records = [record for record, _ in pairs]
        # An `analysis.json` written before the field: every game walked, none has keys.
        keyless = CorpusAnalysis(
            games=[
                GameAnalysis(seed=g.seed, agents=g.agents, walked_plies=g.walked_plies)
                for _, g in pairs
            ]
        )
        for analysis in (None, keyless):
            raw = compute_components(chess_spec, records, rate(records), analysis).raw
            assert "prefix_divergence" in raw
            assert not set(KEYED_COMPONENTS) & set(raw)

    def test_a_stored_run_with_no_keys_names_the_pass(self, chess_spec) -> None:
        """A version that scores a position form, read on a run whose pass predates the
        keys, says which command supplies them rather than reporting a bare gap."""
        from chess_variant_sandbox.metrics.components import compute_components
        from chess_variant_sandbox.tournament.analysis import CorpusAnalysis, GameAnalysis
        from chess_variant_sandbox.tournament.ratings import rate

        weights = parse_weights(
            {
                "version": "keyed",
                "components": {
                    name: {"family": "variety", "weight": 1.0, "curve": "higher_better",
                           "points": {"zero_at": 0.2, "good_at": 0.8}}
                    for name in ("position_divergence", "event_divergence")
                },
            }
        )
        pairs = _commutative_corpus(games=6)
        records = [record for record, _ in pairs]
        keyless = CorpusAnalysis(
            games=[
                GameAnalysis(seed=g.seed, agents=g.agents, walked_plies=g.walked_plies)
                for _, g in pairs
            ]
        )
        measured = compute_components(chess_spec, records, rate(records), keyless)
        with pytest.raises(WeightsError, match="`chess-variant-sandbox analyse <run>`") as raised:
            score_components("h", "variety", measured, weights)
        assert "position_divergence" in str(raised.value)
        assert "predates the position keys" in str(raised.value)

        keyed = CorpusAnalysis(games=[game for _, game in pairs])
        measured = compute_components(chess_spec, records, rate(records), keyed)
        assert score_components("h", "variety", measured, weights).scored

    def test_they_are_declared_outside_the_analysis_fingerprint(self) -> None:
        """Listing them in `tournament/analysis.py` would move the `analysis` fingerprint
        and make the re-score every stored run owes a re-analysis (§5.4, A1)."""
        from chess_variant_sandbox.metrics.components import KEYED_COMPONENTS
        from chess_variant_sandbox.tournament.analysis import ANALYSIS_COMPONENTS

        assert set(KEYED_COMPONENTS) == {"position_divergence", "event_divergence"}
        assert not set(_VARIETY_NAMES) & set(ANALYSIS_COMPONENTS)

    def test_they_move_no_ten_total(self) -> None:
        """`v10` weighs none of them, so its total reads to the digit with or without."""
        weights = load_weights("v10")
        raw = {name: 0.5 for name in weights.components()}
        without = score_components("h", "g", Components(raw=dict(raw)), weights)
        with_them = score_components(
            "h", "g", Components(raw={**raw, **dict.fromkeys(_VARIETY_NAMES, 0.9)}), weights
        )
        assert with_them.total == without.total
        assert with_them.contributions == without.contributions
        assert with_them.families == without.families

    def test_the_distance_table_never_changes_an_answer(self) -> None:
        """A read from a table another call built equals a read from one built for it.

        Includes a table first built from a subset and then grown, which is how a
        bootstrap that never read its whole corpus first fills one.
        """
        import random

        from chess_variant_sandbox.metrics import components

        pairs = _commutative_corpus(games=24)
        rng = random.Random(1)
        draws = [rng.choices(pairs, k=len(pairs)) for _ in range(6)] + [pairs]

        components._TABLES.clear()
        warm = [_variety(draw)[0] for draw in draws]
        cold = []
        for draw in draws:
            components._TABLES.clear()
            cold.append(_variety(draw)[0])
        assert warm == cold
        assert all(len(tables) <= components._TABLES_PER_MEASURE
                   for tables in components._TABLES.values())


def _scripted(outcomes: list):
    """The top pair's winners, one per game in schedule order."""
    remaining = iter(outcomes)
    return lambda game: next(remaining)


class TestTheSeatGateRefusesASeatDecidedGame:
    """The seat gate: `docs/reference/metrics-and-calibration-record.md`, Q235.

    `tunnel-vertical` is a forced first-seat win for the roster from `nodes=1600` up and
    measured viable under `v10` at 0.7791. `first_player_advantage` read 0.0000 on it,
    because the top two rungs' self-play went 8 of 8 to the first seat and the rungs
    below went 8 of 24, cancelling to the digit. The one stored reading that showed it
    was the first seat's 16-0 over the games between the two strongest rungs.

    The corpora below are that shape built from nothing: the whole canonical schedule,
    with the stronger rung winning every mismatched game whoever opens, self-play
    alternating, and the top pair's sixteen games set by each test.
    """

    ROSTER = ("greedy",) + tuple(
        f"minimax:nodes={budget}" for budget in (100, 200, 400, 800, 1600, 6400)
    )
    TOP = ("minimax:nodes=1600", "minimax:nodes=6400")

    def _corpus(self, top_pair, roster=ROSTER, top=TOP, games=4) -> list:
        """Every ordered pairing, `games` times; `top_pair(game)` decides the top pair.

        Every game a different move list, the way varied players' games are: since Q953a
        the test counts each different decided game once, and `_ladder_record` alone gives
        every game one list.
        """
        from chess_variant_sandbox.metrics.components import _budget

        records = []
        for first in roster:
            for second in roster:
                for game in range(games):
                    if first in top and second in top:
                        winner = top_pair(game)
                    elif first == second:
                        winner = game % 2
                    else:
                        winner = 0 if _budget(first) > _budget(second) else 1
                    reason = "checkmate" if winner is not None else "stalemate"
                    record = _ladder_record((first, second), 60, winner, reason)
                    record.moves = [1000 + len(records), *record.moves[1:]]
                    records.append(record)
        return records

    def _measure(self, chess_spec, records):
        from chess_variant_sandbox.metrics.components import compute_components
        from chess_variant_sandbox.tournament.ratings import rate

        return compute_components(chess_spec, records, rate(records))

    def test_the_sign_test_is_the_exact_two_sided_binomial(self) -> None:
        """The figures the record's section 3 quotes, and the marginal ones."""
        from chess_variant_sandbox.metrics.seat import SEAT_DECIDED_ALPHA, sign_test

        assert sign_test(16, 0) == pytest.approx(2 / 65536)
        assert sign_test(10, 1) == pytest.approx(24 / 2048)
        assert sign_test(13, 3) == pytest.approx(2 * 697 / 65536)
        assert sign_test(3, 13) == sign_test(13, 3)
        assert sign_test(0, 0) == 1.0
        assert sign_test(4, 4) == 1.0
        # 14-1 and 11-0 sit at 1/1024, under the line by an exact dyadic margin; 10-0 at
        # 1/512 passes. On sixteen games the gate reads a sweep and nothing narrower.
        assert sign_test(14, 1) == sign_test(11, 0) == 1 / 1024 < SEAT_DECIDED_ALPHA
        assert sign_test(10, 0) == 1 / 512 > SEAT_DECIDED_ALPHA

    def _script_below_the_pair(self, records, outcomes) -> list:
        """Set the twenty top-three games that are not top-pair games.

        The bands overlap, so a top-three split is the pair's plus these. Left alone by
        `_corpus` they run 10-10 and say nothing.
        """
        from chess_variant_sandbox.metrics.components import _budget

        remaining = iter(outcomes)
        for record in records:
            budgets = {_budget(name) for name in record.agents}
            if 800 in budgets and budgets <= {800, 1600, 6400}:
                record.winner = next(remaining)
        return records

    def test_a_first_seat_sweep_of_the_top_pair_is_refused(self, chess_spec) -> None:
        from chess_variant_sandbox.metrics.seat import (
            SEAT_DECIDED_CHANCE,
            TOP_RUNG_SEAT_DETAIL,
            seat_violations,
        )

        components = self._measure(chess_spec, self._corpus(lambda game: 0))
        assert components.raw[SEAT_DECIDED_CHANCE] == pytest.approx(2 / 65536)
        assert components.raw["top_rung_first_seat_score"] == 1.0
        detail = components.diagnostics[TOP_RUNG_SEAT_DETAIL]
        assert detail["rungs"] == [1600, 6400]
        assert (detail["games"], detail["wins"], detail["draws"], detail["losses"]) == (
            16, 16, 0, 0,
        )
        violations = seat_violations(components.raw, components.diagnostics)
        assert len(violations) == 1
        assert violations[0].startswith(SEAT_DECIDED_CHANCE)
        assert "16-0" in violations[0] and "32,768" in violations[0]

        result = score_components("h", "g", components, load_weights("v10"))
        assert not result.viable
        assert any(text.startswith(SEAT_DECIDED_CHANCE) for text in result.violations)

    def test_a_second_seat_sweep_is_refused_too(self, chess_spec) -> None:
        """Two-sided. A game the second seat always wins is as seat-decided as one the
        first seat always wins, and the strength interaction can lean either way."""
        from chess_variant_sandbox.metrics.seat import seat_violations

        components = self._measure(chess_spec, self._corpus(lambda game: 1))
        assert components.raw["top_rung_first_seat_score"] == 0.0
        assert seat_violations(components.raw, components.diagnostics)

    def test_the_pair_alone_still_misses_a_ten_to_one_split(self, chess_spec) -> None:
        """10 W / 1 L / 5 D on the top pair is one chance in 85, and the pair's gate
        refuses at one in a thousand. That miss is why the wider reading exists; the
        games below the pair are even here, so only the pair has anything to say."""
        from chess_variant_sandbox.metrics.seat import SEAT_DECIDED_CHANCE, seat_violations

        components = self._measure(
            chess_spec, self._corpus(_scripted([0] * 10 + [1] + [None] * 5))
        )
        detail = components.diagnostics["top_rung_seat"]
        assert (detail["wins"], detail["draws"], detail["losses"]) == (10, 5, 1)
        assert components.raw[SEAT_DECIDED_CHANCE] == pytest.approx(24 / 2048)
        assert seat_violations(components.raw, components.diagnostics) == []

    def test_the_wide_reading_catches_the_lean_the_pair_cannot(self, chess_spec) -> None:
        """The shape the widening was adopted for. The pair goes 10-1 with five draws
        and passes; the same lean continued through the third rung reads 25-1 over 36
        games, which is one chance in 1.2 million and refused."""
        from chess_variant_sandbox.metrics.seat import (
            SEAT_DECIDED_CHANCE,
            TOP_THREE_SEAT_DECIDED_CHANCE,
            TOP_THREE_SEAT_DETAIL,
            seat_violations,
        )

        records = self._script_below_the_pair(
            self._corpus(_scripted([0] * 10 + [1] + [None] * 5)),
            [0] * 15 + [None] * 5,
        )
        components = self._measure(chess_spec, records)
        detail = components.diagnostics[TOP_THREE_SEAT_DETAIL]
        assert detail["rungs"] == [800, 1600, 6400]
        assert (detail["games"], detail["wins"], detail["draws"], detail["losses"]) == (
            36, 25, 10, 1,
        )
        assert components.raw[TOP_THREE_SEAT_DECIDED_CHANCE] == pytest.approx(
            2 * 27 / 2**26
        )

        violations = seat_violations(components.raw, components.diagnostics)
        assert len(violations) == 1
        assert violations[0].startswith(TOP_THREE_SEAT_DECIDED_CHANCE)
        assert "25-1" in violations[0] and "three strongest rungs" in violations[0]
        assert "800, 1600, 6400 nodes" in violations[0]

        result = score_components("h", "g", components, load_weights("v10"))
        assert not result.viable
        assert not any(text.startswith(SEAT_DECIDED_CHANCE + "=") for text in result.violations)

    def test_the_two_readings_keep_their_own_thresholds(self) -> None:
        """The pair stays where it was set and the wide reading comes in at half of it.
        Reading both at 0.0005 would have moved the pair's own verdicts, and a built
        threshold is not edited after seeing what it caught."""
        from chess_variant_sandbox.metrics.seat import (
            SEAT_DECIDED_ALPHA,
            TOP_THREE_ALPHA,
            sign_test,
        )

        assert SEAT_DECIDED_ALPHA == 0.001
        assert TOP_THREE_ALPHA == 0.0005
        # On the pair's 16 games 11-0 refuses and 12-0 would be the line at 0.0005, so
        # the two thresholds are not interchangeable on the same denominator.
        assert sign_test(11, 0) < SEAT_DECIDED_ALPHA
        assert sign_test(11, 0) > TOP_THREE_ALPHA
        # On 36 games 29-7 refuses and 28-8 does not; on 29 decided, 25-4 and not 24-5.
        assert sign_test(29, 7) < TOP_THREE_ALPHA < sign_test(28, 8)
        assert sign_test(25, 4) < TOP_THREE_ALPHA < sign_test(24, 5)

    def test_a_second_seat_lean_is_refused_by_the_wide_reading_too(
        self, chess_spec
    ) -> None:
        """Two-sided on both denominators."""
        from chess_variant_sandbox.metrics.seat import (
            TOP_THREE_SEAT_DECIDED_CHANCE,
            seat_violations,
        )

        records = self._script_below_the_pair(
            self._corpus(_scripted([1] * 10 + [0] + [None] * 5)),
            [1] * 15 + [None] * 5,
        )
        components = self._measure(chess_spec, records)
        assert components.diagnostics["top_three_seat"]["losses"] == 25
        violations = seat_violations(components.raw, components.diagnostics)
        assert [v.split("=")[0] for v in violations] == [TOP_THREE_SEAT_DECIDED_CHANCE]

    def test_a_sweep_below_the_third_rung_is_read_by_neither(self, chess_spec) -> None:
        """The seat effect the gates exist for appears only above a budget, and every
        rung added below dilutes it. Both bands stop at the third rung from the top."""
        from chess_variant_sandbox.metrics.seat import seat_violations

        records = self._corpus(lambda game: game % 2)
        lower = {"minimax:nodes=100", "minimax:nodes=200", "minimax:nodes=400"}
        for record in records:
            if set(record.agents) <= lower:
                record.winner = 0
        components = self._measure(chess_spec, records)
        assert components.diagnostics["top_rung_seat"]["wins"] == 8
        assert components.diagnostics["top_three_seat"]["wins"] == 18
        assert seat_violations(components.raw, components.diagnostics) == []

    def test_the_loop_roster_reads_its_own_top_pair_alone(self, chess_spec) -> None:
        """`DEFAULT_AGENTS` is `greedy` plus 400 and 1600 at three games a pairing, so
        the top pair is twelve games and only 12-0 or 11-0 can fail. Its third rung from
        the top is `greedy`, which searches nothing, so no wide reading is emitted: the
        threshold was set for a band of three searching rungs."""
        from chess_variant_sandbox.metrics.seat import (
            TOP_THREE_SEAT_DECIDED_CHANCE,
            TOP_THREE_SEAT_DETAIL,
            seat_violations,
        )

        roster = ("greedy", "minimax:nodes=400", "minimax:nodes=1600")
        top = ("minimax:nodes=400", "minimax:nodes=1600")
        sweep = self._measure(
            chess_spec, self._corpus(lambda game: 0, roster=roster, top=top, games=3)
        )
        assert sweep.diagnostics["top_rung_seat"]["rungs"] == [400, 1600]
        assert sweep.diagnostics["top_rung_seat"]["games"] == 12
        assert TOP_THREE_SEAT_DECIDED_CHANCE not in sweep.raw
        assert TOP_THREE_SEAT_DETAIL not in sweep.diagnostics
        assert len(seat_violations(sweep.raw, sweep.diagnostics)) == 1

        near = self._measure(
            chess_spec,
            self._corpus(_scripted([0] * 11 + [1]), roster=roster, top=top, games=3),
        )
        assert seat_violations(near.raw, near.diagnostics) == []

    def test_an_unmeasured_seat_is_silent_not_a_verdict(self) -> None:
        """Like `withheld_components`: a hand-built `Components` has not read the seat."""
        from chess_variant_sandbox.metrics.seat import seat_violations

        assert seat_violations({}, {}) == []
        result = score_components(
            "h", "g", _components(cheap_a=0.9, cheap_b=0.9, hard=0.9), _weights()
        )
        assert result.viable

    def test_the_explain_page_lists_both_gates_beside_the_weights_own(self) -> None:
        from chess_variant_sandbox.metrics.glossary import explain
        from chess_variant_sandbox.metrics.seat import (
            SEAT_DECIDED_ALPHA,
            SEAT_DECIDED_CHANCE,
            TOP_THREE_ALPHA,
            TOP_THREE_SEAT_DECIDED_CHANCE,
        )

        gates = {gate["component"]: gate for gate in explain(DEFAULT_WEIGHTS)["gates"]}
        assert gates[SEAT_DECIDED_CHANCE]["minimum"] == SEAT_DECIDED_ALPHA
        assert gates[SEAT_DECIDED_CHANCE]["title"] == "seat-decided chance"
        assert gates[TOP_THREE_SEAT_DECIDED_CHANCE]["minimum"] == TOP_THREE_ALPHA
        assert (
            gates[TOP_THREE_SEAT_DECIDED_CHANCE]["title"]
            == "seat-decided chance, top three"
        )


_SEAT_RUNS = iter(range(10**9))


def _seat_run(
    wins: int,
    losses: int,
    draws: int = 0,
    wide: tuple | None = None,
    transcripts: list[str] | None = None,
) -> dict:
    """One run's seat diagnostics: the pair's split, and the top three's when given.

    Every decided game a different game unless `transcripts` says which the pair's are,
    and different from every other call's, the way independent runs' games are. The pair's
    first `wins` transcripts are the first seat's wins, as `_seat_split` stores them.
    """
    from chess_variant_sandbox.metrics.seat import TOP_RUNG_SEAT_DETAIL, TOP_THREE_SEAT_DETAIL

    run = next(_SEAT_RUNS)
    pair = transcripts if transcripts is not None else [
        f"pair-{run}-{game}" for game in range(wins + losses)
    ]
    won = set(pair[:wins])
    found = {
        TOP_RUNG_SEAT_DETAIL: {
            "rungs": [1600, 6400],
            "games": wins + losses + draws,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "distinct": len(set(pair)),
            "transcripts": sorted(set(pair)),
            "first_seat_transcripts": sorted(won),
        }
    }
    if wide is not None:
        wide_wins, wide_losses, wide_draws = wide
        every = [f"wide-{run}-{game}" for game in range(wide_wins + wide_losses)]
        found[TOP_THREE_SEAT_DETAIL] = {
            "rungs": [800, 1600, 6400],
            "games": wide_wins + wide_losses + wide_draws,
            "wins": wide_wins,
            "losses": wide_losses,
            "draws": wide_draws,
            "distinct": wide_wins + wide_losses,
            "transcripts": every,
            "first_seat_transcripts": every[:wide_wins],
        }
    return found


class TestTheSeatCheckStatedAsAShare:
    """The restated seat check: `docs/reference/seat-effect-size.md` §2, §3 and §5.

    The built form tests an *even* seat, so the same p-value refuses a narrower split
    every time a run is added -- 94% on one run's sixteen pair games, 72% on four runs'
    sixty-four, 61% on sixteen runs'. A library game is measured four times and will be
    measured more, so what the check means would keep moving. Replacing the even seat in
    the null with a tolerated share fixes the meaning and leaves the sample buying power
    alone. The splits below are the library's own, from §11's table.
    """

    def test_at_an_even_seat_the_share_test_is_the_sign_test(self) -> None:
        """The built form is one member of this family, not a different statistic."""
        from chess_variant_sandbox.metrics.seat import share_test, sign_test

        for wins, losses in ((16, 0), (10, 1), (13, 3), (0, 0), (4, 4), (32, 32)):
            assert share_test(wins, losses, 0.5) == pytest.approx(sign_test(wins, losses))

    def test_one_run_cannot_refuse_even_a_clean_sweep(self) -> None:
        """Which is why the reading is pooled. Sixteen decided games at a tolerated
        share of two in three carry less evidence than the pair's alpha asks for, so a
        16-0 run passes and the four runs' 64-0 does not."""
        from chess_variant_sandbox.metrics.seat import SEAT_DECIDED_ALPHA, share_test

        assert share_test(16, 0) > SEAT_DECIDED_ALPHA
        assert share_test(18, 0) < SEAT_DECIDED_ALPHA
        assert share_test(64, 0) < SEAT_DECIDED_ALPHA

    def test_a_game_inside_the_tolerated_share_is_not_refused_by_more_runs(self) -> None:
        """The defect the restated form exists to fix, stated as a test.

        A game whose seat takes two of every three decided games is refused by the built
        form as soon as enough runs are pooled, and by the restated form at no sample
        size. `chess` itself reads 20-39 on its pooled pair.
        """
        from chess_variant_sandbox.metrics.seat import SEAT_DECIDED_ALPHA, share_test, sign_test

        for runs in (4, 8, 16, 64):
            wins, losses = 16 * runs // 3, 32 * runs // 3
            assert share_test(wins, losses) > SEAT_DECIDED_ALPHA
        assert sign_test(16 * 8 // 3, 32 * 8 // 3) < SEAT_DECIDED_ALPHA

    def test_the_five_swept_games_are_refused_and_the_leans_are_not(self) -> None:
        """§11's table, as arithmetic. The five games every reading refuses read 91% or
        more of their decided pair games to one seat; the five it passes read 84% or
        less, `weathervane half-turn` the nearest of them."""
        from chess_variant_sandbox.metrics.seat import pooled_seat_reading

        swept = {
            "chess sweeping bishops": ((64, 0, 0), (144, 0, 0)),
            "tunnel chess vertical": ((60, 0, 4), (95, 43, 6)),
            "gardner": ((39, 2, 23), (100, 2, 42)),
            "rps race": ((61, 3, 0), (128, 16, 0)),
            "hierarchy extended": ((58, 6, 0), (118, 26, 0)),
        }
        leaning = {
            "weathervane half-turn": ((10, 51, 3), (32, 107, 5)),
            "torus chess": ((35, 8, 21), (75, 32, 37)),
            "crazyhouse": ((14, 50, 0), (47, 96, 1)),
            "king of the hill": ((46, 16, 2), (75, 67, 2)),
            "hierarchy rise throw": ((28, 12, 24), (57, 24, 63)),
        }
        for name, (pair, wide) in swept.items():
            assert pooled_seat_reading([_seat_run(*pair, wide=wide)])["refuses"], name
        for name, (pair, wide) in leaning.items():
            assert not pooled_seat_reading([_seat_run(*pair, wide=wide)])["refuses"], name

    def test_the_bands_are_pooled_over_the_runs_that_measured_them(self) -> None:
        """Four runs of sixteen pair games are sixty-four, and a run that emitted no
        wide band leaves the wide band reading the runs that did."""
        from chess_variant_sandbox.metrics.seat import (
            TOP_RUNG_SEAT_DETAIL,
            TOP_THREE_SEAT_DETAIL,
            pooled_seat_reading,
        )

        runs = [_seat_run(12, 3, 1, wide=(28, 7, 1)) for _ in range(3)]
        runs.append(_seat_run(12, 3, 1))
        bands = pooled_seat_reading(runs)["bands"]
        assert bands[TOP_RUNG_SEAT_DETAIL]["wins"] == 48
        assert bands[TOP_RUNG_SEAT_DETAIL]["decided"] == 60
        assert bands[TOP_RUNG_SEAT_DETAIL]["runs"] == 4
        assert bands[TOP_THREE_SEAT_DETAIL]["wins"] == 84
        assert bands[TOP_THREE_SEAT_DETAIL]["runs"] == 3

    def test_a_band_no_run_measured_is_absent_rather_than_passed(self) -> None:
        """A roster whose third rung searches nothing emits no wide reading, and absence
        is not a verdict -- the same rule `seat_violations` already keeps."""
        from chess_variant_sandbox.metrics.seat import TOP_THREE_SEAT_DETAIL, pooled_seat_reading

        reading = pooled_seat_reading([_seat_run(8, 4, 4) for _ in range(4)])
        assert TOP_THREE_SEAT_DETAIL not in reading["bands"]
        assert pooled_seat_reading([]) == {
            "tolerated_share": pytest.approx(0.667, abs=0.0005),
            "bands": {},
            "refuses": False,
            "violations": [],
            "lean": "",
        }

    def test_a_lean_is_printed_and_refuses_nothing(self) -> None:
        """`weathervane half-turn`, the boundary game: its second seat takes five decided
        pair games in six, which is past what the check tolerates and short of a sweep.
        It is ranked, with the split printed beside its number."""
        from chess_variant_sandbox.metrics.seat import pooled_seat_reading

        reading = pooled_seat_reading([_seat_run(10, 51, 3, wide=(32, 107, 5))])
        assert not reading["refuses"]
        assert "second seat took 51 of the 61 decided games" in reading["lean"]
        assert "84%" in reading["lean"]

    def test_exactly_two_in_three_is_inside_the_line(self) -> None:
        """The null is every share at or under the tolerated one, so a split of exactly two
        in three is tolerated and prints no lean (Q625). `atomic` and `intransitive` stored
        42 of 63 on their pair, whose share rounds to 0.6667 and read *"67%, more than the
        67%"*; one game further is past the line and is printed."""
        from chess_variant_sandbox.metrics.seat import POOLED_TOLERATED_SHARE, pooled_seat_reading

        assert POOLED_TOLERATED_SHARE == 2 / 3
        for took, lost in ((42, 21), (21, 42), (200, 100), (2000, 1000)):
            reading = pooled_seat_reading([_seat_run(took, lost, 1)])
            assert not reading["refuses"]
            assert reading["lean"] == "", (took, lost)
        assert pooled_seat_reading([_seat_run(43, 21, 1)])["lean"] != ""
        assert pooled_seat_reading([_seat_run(2001, 1000, 1)])["lean"] != ""

    def test_a_band_too_small_to_refuse_a_sweep_prints_no_lean(self) -> None:
        """A share off three decided games says the game draws, not that it leans.

        The floor is the sample at which a clean sweep *would* be refused, so it comes
        from the tolerated share and the band's alpha rather than being a number of its
        own. `rps repaint` reads 0-2 on its pooled pair and must say nothing.
        """
        from chess_variant_sandbox.metrics.seat import pooled_seat_reading

        assert pooled_seat_reading([_seat_run(0, 2, 62)])["lean"] == ""
        assert pooled_seat_reading([_seat_run(10, 5, 49)])["lean"] == ""
        assert pooled_seat_reading([_seat_run(14, 4, 46)])["lean"] != ""

    def test_a_refused_game_has_no_lean_however_its_other_band_read(self) -> None:
        """`tunnel chess vertical` sweeps its pair 60-0 and reads 69% on its top three.
        It is refused, and a refusal is not a thing to print a lean beside."""
        from chess_variant_sandbox.metrics.seat import pooled_seat_reading

        reading = pooled_seat_reading([_seat_run(60, 0, 4, wide=(95, 43, 6))])
        assert reading["refuses"]
        assert reading["lean"] == ""

    def test_a_refusal_quotes_the_pooled_split_and_the_runs_it_rests_on(self) -> None:
        from chess_variant_sandbox.metrics.seat import SEAT_DECIDED_CHANCE, pooled_seat_reading

        reading = pooled_seat_reading([_seat_run(15, 0, 1) for _ in range(4)])
        assert len(reading["violations"]) == 1
        said = reading["violations"][0]
        assert said.startswith(f"{SEAT_DECIDED_CHANCE}=")
        assert "first seat took 100% of the 60 decided games" in said
        assert "over 4 runs" in said

    def test_a_seat_sentence_is_told_from_every_other_refusal(self) -> None:
        """What lets the combiner drop a run's seat vote without dropping its others."""
        from chess_variant_sandbox.metrics.seat import is_seat_violation, pooled_seat_reading

        said = pooled_seat_reading([_seat_run(15, 0, 1) for _ in range(4)])["violations"]
        assert all(is_seat_violation(line) for line in said)
        assert not is_seat_violation("mean_plies=6.0 below minimum 8 (a game that ends)")



def _band_record(agents: tuple[str, str], moves: list[int], winner: int | None, seed: int):
    """One game between two rungs, with the move list the distinct count reads."""
    from chess_variant_sandbox.tournament.record import GameRecord

    return GameRecord(
        ruleset_hash="h",
        ruleset_name="replays",
        seed=seed,
        agents=agents,
        moves=moves,
        winner=winner,
        reason="checkmate" if winner is not None else "stalemate",
        plies=len(moves),
        branching=[20] * len(moves),
        material=[0.0] * len(moves),
        captures=[0] * len(moves),
    )


class TestTheSeatBandCountsDistinctGames:
    """`docs/reference/seat-effect-size.md` §15: *a distinct-transcript floor belongs in
    any lean measurement*. `king of the hill` played 200 games at equal strength and
    produced 11 transcripts, so a band's decided-game count can be a handful of games
    replayed. The count is taken off the stored records, draws out, each move list once,
    and it is the floor the printed lean reads. Since Q953a the refusal reads it too: the
    sign test counts each different decided game once (`repeated-games.md` §6)."""

    PAIR = ("minimax:nodes=1600", "minimax:nodes=6400")

    def _records(self, seed: int) -> list:
        """Seven games in the pair band: five decided on three move lists, two drawn."""
        first, second = self.PAIR
        games = [
            ((first, second), [1, 2, 3], 0),
            ((second, first), [1, 2, 3], 0),
            ((first, first), [1, 2, 3], 0),
            ((second, second), [1, 2, 4], 0),
            ((first, second), [5], 1),
            ((second, first), [7, 8], None),
            ((first, first), [9], None),
        ]
        return [
            _band_record(agents, moves, winner, seed + index)
            for index, (agents, moves, winner) in enumerate(games)
        ]

    def _details(self, records, tmp_path, seed: int) -> dict:
        """The seat splits of one run, read the way a run's are: off its stored games."""
        from chess_variant_sandbox.metrics.components import _seat_readings
        from chess_variant_sandbox.metrics.seat import seat_details
        from chess_variant_sandbox.tournament.record import read_records, write_records

        path = write_records(records, tmp_path / f"games-{seed}.jsonl")
        _raw, details = _seat_readings(list(read_records(path)))
        return seat_details(details)

    def test_a_replayed_game_is_counted_once_and_a_draw_not_at_all(self, tmp_path) -> None:
        from chess_variant_sandbox.metrics.seat import TOP_RUNG_SEAT_DETAIL

        band = self._details(self._records(0), tmp_path, 0)[TOP_RUNG_SEAT_DETAIL]
        assert (band["wins"], band["losses"], band["draws"]) == (4, 1, 2)
        assert band["distinct"] == 3
        assert len(band["transcripts"]) == 3

    def test_runs_that_replay_each_others_games_are_counted_once_pooled(
        self, tmp_path
    ) -> None:
        """Four seeds of two deterministic players are one game four times. Summing each
        run's own count would read twelve distinct games; the union reads three."""
        from chess_variant_sandbox.metrics.seat import TOP_RUNG_SEAT_DETAIL, pooled_seat_reading

        runs = [self._details(self._records(seed), tmp_path, seed) for seed in range(4)]
        band = pooled_seat_reading(runs)["bands"][TOP_RUNG_SEAT_DETAIL]
        assert band["decided"] == 20
        assert band["distinct"] == 3

    def test_a_lean_off_a_few_games_replayed_is_not_printed(self) -> None:
        """30-10 is 75%, past the tolerated share, on enough decided games that a sweep
        would be refused. On eleven distinct games it is not, and the lean says nothing;
        on twenty it is, and the lean says how many of the forty were different."""
        from chess_variant_sandbox.metrics.seat import pooled_seat_reading

        def lean(distinct: int) -> str:
            won = 3 * distinct // 4
            transcripts = [f"won-{game % won}" for game in range(30)] + [
                f"lost-{game % (distinct - won)}" for game in range(10)
            ]
            return pooled_seat_reading([_seat_run(30, 10, 4, transcripts=transcripts)])["lean"]

        assert lean(11) == ""
        assert "30 of the 40 decided games (20 of them distinct)" in lean(20)
        assert "30 of the 40 decided games between" in lean(40)

    def test_a_run_without_transcripts_prints_no_lean(self) -> None:
        """A run stored before the count cannot say how many of its games were different,
        and a floor it cannot read is not a floor it passed."""
        from chess_variant_sandbox.metrics.seat import TOP_RUNG_SEAT_DETAIL, pooled_seat_reading

        old = _seat_run(30, 10, 4)
        for key in ("distinct", "transcripts"):
            del old[TOP_RUNG_SEAT_DETAIL][key]
        reading = pooled_seat_reading([old, _seat_run(30, 10, 4)])
        assert reading["bands"][TOP_RUNG_SEAT_DETAIL]["distinct"] is None
        assert reading["lean"] == ""

    def test_the_refusal_reads_each_different_game_once(self) -> None:
        """Q953a reverses Q534 (`repeated-games.md` §4.3, §6). A sweep of 60 decided games
        is refused when they are 60 games and not when they are one game played 60 times:
        a line replayed is one observation of what its pairing does. Runs stored before the
        split was kept are tested as played, and the band says which it read."""
        from chess_variant_sandbox.metrics.seat import TOP_RUNG_SEAT_DETAIL, pooled_seat_reading

        many = pooled_seat_reading([_seat_run(15, 0, 1) for _ in range(4)])
        one = pooled_seat_reading([_seat_run(15, 0, 1, transcripts=["same"]) for _ in range(4)])
        old = [_seat_run(15, 0, 1) for _ in range(4)]
        for run in old:
            del run[TOP_RUNG_SEAT_DETAIL]["first_seat_transcripts"]
        unknown = pooled_seat_reading(old)

        assert many["bands"][TOP_RUNG_SEAT_DETAIL]["distinct"] == 60
        assert many["refuses"] is True
        assert "60 decided games, each different game counted once" in many["violations"][0]
        band = one["bands"][TOP_RUNG_SEAT_DETAIL]
        assert (band["wins"], band["decided"], band["distinct"]) == (60, 60, 1)
        assert (band["distinct_wins"], band["distinct_losses"]) == (1, 0)
        assert band["counted"] == "distinct"
        assert one["refuses"] is False
        assert unknown["refuses"] is True
        assert unknown["bands"][TOP_RUNG_SEAT_DETAIL]["counted"] == "as played"
        assert unknown["bands"][TOP_RUNG_SEAT_DETAIL]["distinct_wins"] is None

    def test_the_band_keeps_both_splits(self) -> None:
        """SP2's top two, §4.3's table: 58-2 as played, 16-2 on its different games, which
        is one chance in 30 at the tolerated share and passes. The split as played stays
        in the band beside the one tested, and the lean quotes it."""
        from chess_variant_sandbox.metrics.seat import TOP_RUNG_SEAT_DETAIL, pooled_seat_reading

        transcripts = [f"won-{game % 16}" for game in range(58)] + ["lost-0", "lost-1"]
        reading = pooled_seat_reading([_seat_run(58, 2, 4, transcripts=transcripts)])
        band = reading["bands"][TOP_RUNG_SEAT_DETAIL]
        assert (band["wins"], band["losses"], band["decided"]) == (58, 2, 60)
        assert (band["distinct_wins"], band["distinct_losses"], band["distinct"]) == (16, 2, 18)
        assert band["chance"] == pytest.approx(3.26e-02, abs=5e-05)
        assert not reading["refuses"]
        assert "58 of the 60 decided games (18 of them distinct)" in reading["lean"]

    def test_a_transcript_won_in_one_run_is_won_in_all(self) -> None:
        """The pooled split is the union of the runs' transcripts, split by the union of
        the ones the first seat won: four runs of one line are one win."""
        from chess_variant_sandbox.metrics.seat import TOP_RUNG_SEAT_DETAIL, pooled_seat_reading

        runs = [_seat_run(3, 1, 0, transcripts=["a", "b", "c", "d"]) for _ in range(4)]
        band = pooled_seat_reading(runs)["bands"][TOP_RUNG_SEAT_DETAIL]
        assert (band["wins"], band["losses"]) == (12, 4)
        assert (band["distinct_wins"], band["distinct_losses"]) == (3, 1)

    def test_one_runs_own_seat_gate_reads_it_too(self, tmp_path) -> None:
        """The built per-run form, which a visitor's single run is decided by, tests the
        distinct split as well: sixteen copies of one game swept are one win, and sixteen
        different games swept are refused. Both splits are in the detail and the refusal
        quotes the one it tested."""
        from chess_variant_sandbox.metrics.components import _seat_readings
        from chess_variant_sandbox.metrics.seat import TOP_RUNG_SEAT_DETAIL, seat_violations

        first, second = self.PAIR
        pairings = [(first, second), (second, first), (first, first), (second, second)]

        def violations(moves_of) -> tuple[list[str], dict]:
            records = [
                _band_record(pairings[game % 4], moves_of(game), 0, game) for game in range(16)
            ]
            raw, details = _seat_readings(records)
            return seat_violations(raw, details), details[TOP_RUNG_SEAT_DETAIL]

        replayed, one = violations(lambda game: [1, 2, 3])
        varied, sixteen = violations(lambda game: [1, 2, game])
        assert (one["distinct"], sixteen["distinct"]) == (1, 16)
        assert (one["wins"], one["distinct_wins"], one["distinct_losses"]) == (16, 1, 0)
        assert replayed == []
        assert len(varied) == 1
        assert "16-0 with each different game counted once" in varied[0]

    def test_a_forced_line_played_by_inert_players_passes_the_seat(self) -> None:
        """§4.3's weakness, pinned (`docs/bugs.md`): a band whose players replay one forced
        first-seat line is one decided game however often it is played, so no number of
        runs can refuse it on the seat. Only the length and quick-kill gates can refuse
        such a game, and a forced line longer than their horizons passes them too."""
        from chess_variant_sandbox.metrics.seat import pooled_seat_reading

        forced = [_seat_run(16, 0, 0, transcripts=["the line"] * 16) for _ in range(64)]
        reading = pooled_seat_reading(forced)
        assert all(band["decided"] == 1024 for band in reading["bands"].values())
        assert reading["refuses"] is False


class TestWhatAPlayerCanHoldIsReadOffEachRoute:
    """Q922: `pieces_a_player_can_hold`, one route at a time, on inline specs.

    The walk reads only `pieces`, `setup.placements` and `rules`, so each case names the
    ids a player owns from ply zero (`A`) and the one route that adds a second type.
    """

    @staticmethod
    def _spec(rules: dict, *, placed: tuple[str, ...] = ("A",), npcs: tuple[str, ...] = ()) -> dict:
        ids = ("A", "B", "C", "D", "S")
        return {
            "pieces": [{"id": i} for i in ids],
            "setup": {"placements": [{"piece": p, "square": "a1"} for p in placed]},
            "rules": {**rules, **({"npcs": [{"piece": n} for n in npcs]} if npcs else {})},
        }

    CASES = {
        "a held piece promotes": ({"promotion": [{"piece": "A", "to": ["B"]}]}, {"A", "B"}),
        "a lone promotion rule, not a list": ({"promotion": {"piece": "A", "to": ["B"]}}, {"A", "B"}),
        "an unheld piece promotes, which adds nothing": (
            {"promotion": [{"piece": "C", "to": ["B"]}]},
            {"A"},
        ),
        "promotion chains through its own target": (
            {"promotion": [{"piece": "A", "to": ["B"]}, {"piece": "B", "to": ["C"]}]},
            {"A", "B", "C"},
        ),
        "a held piece is struck down a chain": (
            {"strike": {"effect": "demote", "chain": [{"from": "A", "to": "B"}]}},
            {"A", "B"},
        ),
        "an unheld piece's chain link adds nothing": (
            {"strike": {"effect": "demote", "chain": [{"from": "C", "to": "B"}]}},
            {"A"},
        ),
        "a contact between two standing pieces changes the mover": (
            {"contacts": {"links": [{"mover": "A", "onto": "S", "mover_to": "C"}]},
             "drops": {"stock": {"S": 1}, "from_captures": False}},
            {"A", "S", "C"},
        ),
        "a contact with an absent partner changes nothing": (
            {"contacts": {"links": [{"mover": "A", "onto": "D", "mover_to": "C"}]}},
            {"A"},
        ),
        "a contact whose partner stands only by stock still fires": (
            {"contacts": {"links": [{"mover": "S", "onto": "A", "onto_to": "C"}]},
             "drops": {"stock": {"S": 1}, "from_captures": False}},
            {"A", "S", "C"},
        ),
    }

    @pytest.mark.parametrize("label", sorted(CASES))
    def test_each_route(self, label: str) -> None:
        from chess_variant_sandbox.spec.reachability import pieces_a_player_can_hold

        rules, expected = self.CASES[label]
        assert pieces_a_player_can_hold(self._spec(rules)) == expected

    @pytest.mark.parametrize(
        ("drops", "filled"),
        [
            ({"stock": {"S": 1}}, True),
            ({"stock": {"S": 1}, "from_captures": True}, True),
            ({"stock": {"S": 1}, "from_captures": False}, False),
            ({}, False),
        ],
        ids=["default", "explicit-true", "explicit-false", "empty"],
    )
    def test_a_hand_that_fills_by_default_holds_what_it_captures(
        self, drops: dict, filled: bool
    ) -> None:
        """`from_captures` defaults to true, so a wild piece taken lands in a hand."""
        from chess_variant_sandbox.spec.reachability import pieces_a_player_can_hold

        held = pieces_a_player_can_hold(self._spec({"drops": drops}, npcs=("D",)))
        assert ("D" in held) is filled
