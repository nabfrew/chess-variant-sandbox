"""Tests about the index rather than about any game.

Everything in ``test_metrics.py`` asks whether the scoring machinery does what it says.
These ask whether what it says is worth saying. The distinction earned its own file the
hard way: v2 passed every mechanical test it had while carrying 0.06 of weight on a
component with zero variance across the entire reference corpus, two components
correlated at +0.93, and an elegance term that scored both deliberately broken controls
above every real game. Nothing was broken. Everything was working exactly as written,
and the written thing was wrong.

So these tests are assertions about the *shape* of a weights file, checked against the
reference corpus, and they are what stops that class of defect from coming back
unnoticed in v4.
"""

from __future__ import annotations

import copy
import json
import os
import statistics
from pathlib import Path

import pytest

from chess_variant_sandbox.metrics.calibration import (
    EXCLUDED_FROM_BAND,
    ScoredEntry,
    calibrate,
)
from chess_variant_sandbox.metrics.scoring import DEFAULT_WEIGHTS, WEIGHTS_DIR, load_weights
from chess_variant_sandbox.spec.complexity import _RULE_COSTS
from chess_variant_sandbox.spec.schema import _RULES_KEYS


def _main_checkout() -> Path:
    """The checkout whose `runs/` holds the corpora.

    A worktree has no `runs/` of its own, so the audit read nothing there and skipped in
    every worker. Its `.git` is a one-line file, `gitdir: <main>/.git/worktrees/<name>`,
    which names the main checkout without a subprocess.
    """
    repository = Path(__file__).resolve().parents[1]
    pointer = repository / ".git"
    if pointer.is_file():
        gitdir = Path(pointer.read_text(encoding="utf-8").partition(":")[2].strip())
        if gitdir.parent.name == "worktrees" and gitdir.parents[1].name == ".git":
            return gitdir.parents[2]
    return repository


RUNS = _main_checkout() / "runs"

# Anchors and probes, not band members. Excluded wherever the question is "does this
# behave like a real ruleset?" -- the controls because they were built not to, the RPS
# probes because their own designer rejected them. Aliased rather than repeated: a name
# added to one copy and not the other silently widens the human band.
CONTROL_RULESETS = EXCLUDED_FROM_BAND

# A component whose score barely moves across a corpus of wildly different games is not
# measuring those games. It may still be a fine gate.
# Weights versions still in use, read off the weights directory rather than listed here.
# v1 and v2 are kept so their recorded scores stay reproducible; their component sets are
# history. A hand-maintained tuple went stale the moment `v9` was cut -- silently, and in
# the direction that exempts the *default* version from every invariant below, which is
# the one version that must not be exempt.
_HISTORICAL_VERSIONS = ("v1", "v2")
_LIVE_VERSIONS = tuple(
    sorted(
        (p.stem for p in WEIGHTS_DIR.glob("*.json") if p.stem not in _HISTORICAL_VERSIONS),
        key=lambda name: int(name.lstrip("v")),
    )
)

MIN_SPREAD = 0.05
# Two components this correlated are one component charged for twice, and its noise is
# counted twice with it. The limit was placed 0.05 above the closest pair's v3 reading and
# has never been calibrated; it stays where it is, because moving it after a miss would
# fit the check to the miss.
MAX_CORRELATION = 0.85
# Pairs known to read above the limit, recorded rather than tuned away. Every other pair
# is still held to the limit, and an entry here must leave when its pair falls back under
# it or one of the two leaves the weights file: the test fails on a stale entry.
#
# **Emptied by `v11` (`docs/archive/plan-variety.md` §5.4, C2).** The one entry was the variety
# family's two components, `distinct_game_rate` and `opening_diversity`, which agreed on
# what is not a game (raw correlation 0.96 over the negatives) and disagreed inside the
# band (0.71), and crossed 0.85 when the calibration corpus grew after v9 was anchored
# (docs/reference/component-audit-v3-pin.md). `v11` scores neither: `position_divergence`
# replaced both, and C1's table read it at 0.86 and 0.88 against them, so neither could
# have stayed beside it. The pair stays recorded for `v10`, whose audit no longer runs as
# the default's.
KNOWN_DUPLICATED: frozenset[frozenset[str]] = frozenset()
# How far a component may score the negatives *above* the real games before it counts
# as pushing the wrong way. Slightly negative is tolerated: a component guarding a
# failure the mutant corpus happens not to contain will look mildly backwards without
# being wrong. Deleting it on that evidence would be fitting the index to the negatives.
MAX_BACKWARDS_GAP = -0.15

# What makes one directory of `RUNS` a labelled corpus rather than a pile of runs. Named so
# `tests/test_calibration_corpus.py` can hold a three-entry corpus to this reader.
MIN_CORPUS_RUNS = 8
MIN_LABELLED_RUNS = 4


# Subtrees of `RUNS` that hold history, never a corpus: `history.deliver` keeps the score it
# replaces under `<hash>/previous/<n>/`, and a replay keeps the runs it supersedes under
# `replaced/`. Together they are over half the files in `runs/` (23k of 51k on the main
# checkout, 2.7 s of walking against 0.7 s without them), and the audit paid that walk
# twice per run of the file. A corpus is a directory a campaign wrote its fresh runs into.
_HISTORY_DIRECTORIES = frozenset({"previous", "replaced"})


def _scored_files(directory: Path, version: str) -> list[Path]:
    """Every `index-<version>.json` under `directory`, history subtrees left unvisited."""
    name = f"index-{version}.json"
    found = []
    for here, children, files in os.walk(directory):
        children[:] = [child for child in children if child not in _HISTORY_DIRECTORIES]
        if name in files:
            found.append(Path(here) / name)
    return sorted(found)


def _corpus(version: str) -> Path:
    """The labelled corpus scored under one weights version: one subdirectory of `RUNS`.

    Every arm needs the negatives. A real variant is meant to plateau near the top, so on
    a corpus of real games alone a well-anchored band component reads as a constant and
    the constant arm fails the design for working (`lead_conversion`: 61 of 62 canonical
    runs at exactly 1.0), and the backwards arm has no `worse` runs at all. Globbing the
    whole tree instead pooled the calibration corpus with canonical re-scores of the same
    rulesets and passed on the mixture. So the corpus is exactly one directory holding at
    least eight scored runs of which at least four carry a label. None skips; two or more
    fails, because picking one silently would be the pooling artefact by another route.
    """
    candidates = []
    for directory in sorted(RUNS.iterdir()) if RUNS.is_dir() else []:
        if not directory.is_dir():
            continue
        scored = _scored_files(directory, version)
        labelled = [p for p in scored if (p.parent / "mutant.json").exists()]
        if len(scored) >= MIN_CORPUS_RUNS and len(labelled) >= MIN_LABELLED_RUNS:
            candidates.append(directory)
    if not candidates:
        pytest.skip(f"needs a labelled corpus scored under {version} in {RUNS}; found none")
    if len(candidates) > 1:
        pytest.fail(
            f"{len(candidates)} labelled corpora are scored under {version}: "
            f"{[c.name for c in candidates]}. The audit reads one; pooling them was the "
            "artefact that let a v10 audit pass on the whole tree."
        )
    return candidates[0]


def _labelled_runs(version: str) -> list[dict]:
    """Scored runs with the label the calibration corpus assigns them."""
    controls = CONTROL_RULESETS
    found = []
    for path in _scored_files(_corpus(version), version):
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle)
        meta = path.parent / "mutant.json"
        if meta.exists():
            with open(meta, encoding="utf-8") as handle:
                result["_label"] = json.load(handle)["expectation"]
        else:
            name = str(result["ruleset_name"]).lower()
            result["_label"] = "control" if name in controls else "human"
        found.append(result)
    return found


class TestEveryScoredComponentIsAnchorable:
    """A component the anchoring cannot clamp gets curve ends outside its own range.

    Found the hard way on the first v5 anchoring. `progress_irreversibility` is a share
    in [0, 1] and it is a band, which is the combination that makes the omission visible:
    the span floors push a band's zero ends outward, and with no domain to clamp against
    `zero_low` came out at -0.64. A zero end the measurement can never reach is a
    component that can never score badly -- silent, and exactly the shape of thing that
    survives a review because every number in the file looks plausible.
    """

    def test_every_scored_component_declares_its_domain(self) -> None:
        from chess_variant_sandbox.metrics.calibration import DOMAIN

        for version in _LIVE_VERSIONS:
            for name in sorted(load_weights(version).components()):
                assert name in DOMAIN, (
                    f"{name} is scored under {version} and has no DOMAIN entry, so "
                    "`chess-variant-sandbox anchor` cannot clamp its curve ends to the range the "
                    "measurement can actually produce"
                )

    def test_a_bounded_components_curve_ends_stay_inside_its_bounds(self) -> None:
        """The check on the anchored file itself, not on the table behind it."""
        from chess_variant_sandbox.metrics.calibration import DOMAIN
        from chess_variant_sandbox.metrics.scoring import WEIGHTS_DIR

        for version in _LIVE_VERSIONS:
            config = json.loads(
                (WEIGHTS_DIR / f"{version}.json").read_text(encoding="utf-8")
            )
            for name, component in config["components"].items():
                low, high = DOMAIN.get(name, (None, None))
                for edge, value in component["points"].items():
                    if low is not None:
                        assert value >= low, f"{version}/{name}/{edge} = {value} < {low}"
                    if high is not None:
                        assert value <= high, f"{version}/{name}/{edge} = {value} > {high}"


class TestRuleCostTable:
    """Every rule the schema accepts has to cost something, or nothing is elegant.

    This is the test that would have caught the real defect: ``measure_complexity``
    counted five hardcoded rule keys, so every rule added to the schema afterwards was
    free. A spec could take on drops, regions or a bare-royal win and measure exactly as
    complex as one without them. Nothing failed, the engine played it, and the elegance
    term quietly rewarded it -- and rule mass is now a *denominator*, so an uncounted
    feature inflates the ratio rather than merely shifting a level.
    """

    def test_every_schema_rule_has_a_declared_cost(self) -> None:
        uncosted = sorted(_RULES_KEYS - set(_RULE_COSTS))
        assert not uncosted, (
            f"rules accepted by the schema but invisible to rule mass: {uncosted}. "
            "Add a row to _RULE_COSTS -- zero is a legitimate cost for bookkeeping, "
            "but it has to be written down rather than left out."
        )

    def test_no_cost_for_a_rule_the_schema_does_not_accept(self) -> None:
        stale = sorted(set(_RULE_COSTS) - _RULES_KEYS)
        assert not stale, f"rule costs for keys the schema no longer has: {stale}"


class TestComponentAudit:
    """Every weighted component must earn its weight on the reference corpus.

    Skipped when no labelled corpus has been scored under the version being checked,
    because the alternative is a test that passes by having no evidence -- which is how
    the defects it looks for got in. The version is the live one: pinned to `v3`, the
    audit stayed green for six versions while guarding an index nothing scored under.
    """

    version = DEFAULT_WEIGHTS

    @pytest.fixture(scope="class")
    @classmethod
    def scores(cls) -> list[dict]:
        # Labelled once: the corpus is found by walking `runs/`, the one slow step, and
        # the arms that rank negatives against real games need the labels as well.
        return _labelled_runs(cls.version)

    def test_no_component_is_a_constant(self, scores: list[dict]) -> None:
        weights = load_weights(self.version)
        dead = []
        for name in weights.components():
            values = [r["scores"][name] for r in scores if name in r["scores"]]
            if len(values) > 1 and statistics.pstdev(values) < MIN_SPREAD:
                dead.append((name, round(statistics.pstdev(values), 4)))
        assert not dead, (
            f"components carrying weight but not varying across the corpus: {dead}. "
            "A constant cannot rank anything; move it to a gate or remove it."
        )

    def test_no_two_components_are_the_same_measurement(self, scores: list[dict]) -> None:
        """Checked on the raw measurements, deliberately, not on the scores.

        Scores saturate. Two genuinely independent measurements that both read 1.00 for
        every healthy game and near zero for every broken one will correlate above 0.9
        in score space while their raw values correlate at 0.3 -- which is an artifact of
        the curves, not double counting. Running this on scores flagged three such pairs
        and would have had someone delete a component that was doing real work. The
        defect this test exists for is two components *measuring the same thing*, and
        that is a question about the measurements.
        """
        weights = load_weights(self.version)
        names = weights.components()
        columns = {
            name: [r["raw"][name] for r in scores if name in r["raw"]]
            for name in names
        }
        duplicated = []
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                left, right = columns[first], columns[second]
                if len(left) != len(right) or len(left) < 4:
                    continue
                if statistics.pstdev(left) == 0 or statistics.pstdev(right) == 0:
                    continue
                correlation = statistics.correlation(left, right)
                if abs(correlation) > MAX_CORRELATION:
                    duplicated.append((first, second, round(correlation, 3)))
        found = {frozenset(pair[:2]) for pair in duplicated}
        new = [pair for pair in duplicated if frozenset(pair[:2]) not in KNOWN_DUPLICATED]
        assert not new, (
            f"components that are one measurement counted twice: {new}. "
            "Their weights add, and so does their noise."
        )
        stale = [sorted(pair) for pair in KNOWN_DUPLICATED - found]
        assert not stale, (
            f"recorded pairs no longer above {MAX_CORRELATION} under {self.version}: "
            f"{stale}. Remove them from KNOWN_DUPLICATED; a record that outlives its "
            "miss is an exemption."
        )

    def test_every_component_scores_broken_games_below_real_ones(
        self, scores: list[dict]
    ) -> None:
        """The test that replaced a pinning check, because pinning is not the defect.

        Real variants plateauing near the top is the design: they are all good games,
        and an index that confidently ranks them is claiming a resolution the sampling
        interval says it does not have. What a component may *not* do is score the
        known-bad rulesets at or above the real ones -- that is weight pushing the wrong
        way, and no amount of headroom fixes it.

        A small negative gap is tolerated for components whose specific failure the
        mutant corpus does not contain; a large one is not.
        """
        weights = load_weights(self.version)
        labelled = scores
        humans = [r for r in labelled if r["_label"] == "human"
                  and r["raw"].get("ladder_monotonicity", 1.0) >= 0.5]
        broken = [r for r in labelled if r["_label"] in ("worse", "control")]
        if len(humans) < 6 or len(broken) < 4:
            pytest.skip("needs a labelled corpus with both real games and negatives")

        backwards = []
        for name in weights.components():
            good = [r["scores"][name] for r in humans if name in r["scores"]]
            bad = [r["scores"][name] for r in broken if name in r["scores"]]
            if not good or not bad:
                continue
            gap = statistics.fmean(good) - statistics.fmean(bad)
            if gap < MAX_BACKWARDS_GAP:
                backwards.append((name, round(gap, 3)))
        assert not backwards, (
            f"components scoring broken rulesets above real ones: {backwards}. "
            "Their weight is pushing the index the wrong way."
        )


class TestBootstrap:
    """An index score is an estimate, and an estimate without an interval misleads."""

    def _records(self, chess, count: int = 12):
        from chess_variant_sandbox.tournament.play import play_game

        pairs = [("minimax:nodes=150", "minimax:nodes=150"), ("random", "minimax:nodes=150")]
        return [
            play_game(chess, pairs[seed % 2], seed)
            for seed in range(count)
        ]

    def test_returns_nothing_rather_than_faking_precision(self, chess_spec, chess) -> None:
        """A missing interval is reported as missing. A zero would read as exact."""
        from chess_variant_sandbox.metrics.bootstrap import bootstrap_index

        assert bootstrap_index(chess_spec, self._records(chess, 4), load_weights("v2")) is None

    def test_produces_an_interval_that_contains_its_own_mean(self, chess_spec, chess) -> None:
        from chess_variant_sandbox.metrics.bootstrap import bootstrap_index

        result = bootstrap_index(
            chess_spec, self._records(chess), load_weights("v2"), resamples=40
        )
        assert result is not None
        assert result.ci_low <= result.mean <= result.ci_high
        assert result.sd >= 0.0
        assert result.width >= 0.0

    def test_separation_needs_non_overlapping_intervals(self) -> None:
        """The test that keeps a design loop from climbing its own sampling noise."""
        from chess_variant_sandbox.metrics.bootstrap import BootstrapResult

        def band(low: float, high: float) -> BootstrapResult:
            return BootstrapResult(
                resamples=100,
                mean=(low + high) / 2,
                sd=(high - low) / 4,
                ci_low=low,
                ci_high=high,
                deterministic_pairings=0.0,
                component_sd={},
            )

        clearly_better = band(0.80, 0.90)
        clearly_worse = band(0.50, 0.60)
        overlapping = band(0.76, 0.86)

        assert clearly_better.separates(clearly_worse)
        # 0.85 against 0.81 looks like an improvement and is not one: the intervals
        # overlap, so the loop must not treat it as progress.
        assert not clearly_better.separates(overlapping)


class TestMutantCorpus:
    """The negatives. Without them there is nothing to calibrate a zero end against."""

    def _bases(self) -> dict:
        from chess_variant_sandbox.spec import load_spec

        rulesets = Path(__file__).resolve().parents[1] / "rulesets"
        return {
            load_spec(p)["name"]: load_spec(p)
            for p in rulesets.glob("*.json")
        }

    def test_every_mutant_is_a_valid_spec(self) -> None:
        """A mutant that will not build tests nothing at all."""
        from chess_variant_sandbox.metrics.mutants import build_mutants
        from chess_variant_sandbox.spec.schema import validate_spec

        broken = []
        for mutant, spec in build_mutants(self._bases()):
            errors = validate_spec(spec)
            if errors:
                broken.append((mutant.name, errors[0]))
        assert not broken, f"mutants that do not validate: {broken}"

    def test_every_mutant_differs_from_its_base(self) -> None:
        """The hash covers mechanics only, so a mutant that changed nothing collides."""
        from chess_variant_sandbox.metrics.mutants import build_mutants
        from chess_variant_sandbox.spec import spec_hash

        bases = self._bases()
        for mutant, spec in build_mutants(bases):
            if mutant.expectation == "same":
                continue
            assert spec_hash(spec) != spec_hash(bases[mutant.base]), (
                f"{mutant.name} produced the same mechanics as {mutant.base}"
            )

    def test_the_corpus_carries_all_three_labels(self) -> None:
        from chess_variant_sandbox.metrics.mutants import EXPECTATIONS, MUTANTS

        present = {m.expectation for m in MUTANTS}
        assert present == set(EXPECTATIONS)
        assert sum(1 for m in MUTANTS if m.expectation == "worse") >= 8
        assert sum(1 for m in MUTANTS if m.expectation == "same") >= 2

    def test_the_idle_wanderer_is_its_base_with_a_file_to_shuffle_on(
        self, chess_spec, chess
    ) -> None:
        """The variety fixture of `docs/archive/plan-variety.md` §3.6 changes nothing but
        the wanderer.

        Along a game of the base, every position of the mutant offers the base's legal
        moves, shifted one file, plus the wanderer's own, and no other piece may ever
        land on the two new files. Named after chess only because it is the base.
        """
        from chess_variant_sandbox.engine import Engine
        from chess_variant_sandbox.engine.moves import (
            encode_move,
            move_from,
            move_landing,
            move_promotion,
            move_to,
            move_type,
        )
        from chess_variant_sandbox.metrics.mutants import MUTANTS
        from chess_variant_sandbox.tournament.play import play_game

        mutant = next(m for m in MUTANTS if m.name == "null-idle-wanderer")
        assert (mutant.base, mutant.expectation) == ("chess", "same")
        wide = Engine(mutant.build(chess_spec))
        width = chess.spec.width
        assert wide.spec.width == width + 2

        def shift(move: int) -> int:
            def square(index: int) -> int:
                return (index // width) * (width + 2) + index % width + 1

            return encode_move(
                square(move_from(move)),
                square(move_to(move)),
                move_type(move),
                move_promotion(move),
                move_landing(move),
            )

        record = play_game(chess, ("greedy", "greedy"), 3)
        assert record.plies > 20
        narrow_state, wide_state = chess.initial_state(), wide.initial_state()
        shuffles = 0
        for move in record.moves:
            legal = wide.legal_moves(wide_state)

            # Empty only while the mover is in check.
            wanderer = {m for m in legal if move_from(m) % (width + 2) == 0}
            shuffles += bool(wanderer)
            assert {m for m in legal if m not in wanderer} == {
                shift(m) for m in chess.legal_moves(narrow_state)
            }
            narrow_state = chess.apply(narrow_state, move)
            wide_state = wide.apply(wide_state, shift(move))
        assert shuffles > record.plies // 2

    def test_mutants_are_built_from_more_than_one_base(self) -> None:
        """Otherwise every conclusion is a fact about one ruleset."""
        from chess_variant_sandbox.metrics.mutants import MUTANTS

        assert len({m.base for m in MUTANTS}) >= 2

    def test_the_negatives_span_several_geometries(self) -> None:
        """The `worse` set alone, because it is the set that carries the weight.

        `sign_accuracy` counts over it and `anchor_curves` places every `zero_at` on
        the median of its raws, so negatives drawn from one family of boards answer
        "where does not-a-game begin" for that family and no other. Chess and gardner
        share an army, a topology and a win condition; four more bases were added in
        Q49 precisely so that this assertion could be made.
        """
        from chess_variant_sandbox.metrics.mutants import MUTANTS

        bases = {m.base for m in MUTANTS if m.expectation == "worse"}
        assert len(bases) >= 6, f"the negatives descend from only {sorted(bases)}"

    def test_flatten_army_is_a_null_on_a_game_whose_pieces_already_move_alike(
        self,
    ) -> None:
        """Which is why the territory base takes `no_captures` instead.

        Rock, paper and scissors move identically and the asymmetry is entirely in
        `captures`, so flattening the army returns the base spec unchanged. Asserting
        it keeps the reason in the suite rather than only in a comment: if a future
        edit made the transform bite there, the mutant it would produce is a different
        claim and would need a fresh label.
        """
        from chess_variant_sandbox.metrics.mutants import flatten_army
        from chess_variant_sandbox.spec import spec_hash

        base = self._bases()["rps territory"]
        assert spec_hash(flatten_army(copy.deepcopy(base))) == spec_hash(base)


class TestCalibrationReport:
    """The four numbers that say whether an index is worth trusting."""

    def _entry(self, name, total, label, base="", viable=True, sd=0.02):
        return ScoredEntry(name=name, total=total, viable=viable, label=label, base=base, sd=sd)

    def _corpus(self, worst_bad=0.55):
        return [
            self._entry("alpha", 0.86, "human"),
            self._entry("beta", 0.80, "human"),
            self._entry("gamma", 0.78, "human"),
            self._entry("broken-one", worst_bad, "worse"),
            self._entry("broken-two", 0.40, "worse"),
            self._entry("null-alpha", 0.85, "same", base="alpha"),
        ]

    def test_separation_is_the_gap_to_the_best_known_bad(self) -> None:
        report = calibrate(self._corpus(), "vtest")
        assert report.separation == pytest.approx(0.78 - 0.55)
        assert report.sign_accuracy == 1.0
        assert not report.misplaced

    def test_a_negative_inside_the_human_band_is_named(self) -> None:
        """The point of the corpus: a miss says which defect the index cannot see."""
        report = calibrate(self._corpus(worst_bad=0.82), "vtest")
        assert report.separation < 0
        assert report.misplaced == ["broken-one"]
        assert report.sign_accuracy == pytest.approx(0.5)

    def test_a_gated_out_mutant_counts_as_separated(self) -> None:
        """Gates and scores are two halves of one verdict.

        A broken game the gates disqualify has been noticed. Scoring it on its total
        alone would punish the gates for working, which is exactly backwards.
        """
        corpus = self._corpus(worst_bad=0.95)
        corpus[3] = self._entry("broken-one", 0.95, "worse", viable=False)
        report = calibrate(corpus, "vtest")
        assert report.sign_accuracy == 1.0
        assert report.detail["gated_out"] == ["broken-one"]

    def test_human_spread_is_reported_so_it_can_be_kept_small(self) -> None:
        """Small is the goal. Twenty classics are all good games.

        An index that confidently ranks them claims a resolution the sampling interval
        says it does not have.
        """
        report = calibrate(self._corpus(), "vtest")
        assert report.human_spread == pytest.approx(0.86 - 0.78)

    def test_a_null_that_moves_within_the_noise_is_stable(self) -> None:
        report = calibrate(self._corpus(), "vtest")
        assert report.null_stability == pytest.approx(0.01)
        assert not report.unstable_nulls

    def test_a_null_that_moves_further_than_the_noise_is_flagged(self) -> None:
        """A rename must not move the score. If it does, the index is reading the seed."""
        corpus = self._corpus()
        corpus[-1] = self._entry("null-alpha", 0.60, "same", base="alpha")
        report = calibrate(corpus, "vtest")
        assert report.unstable_nulls == ["null-alpha"]

    def test_a_corpus_with_no_null_refuses_to_report_null_stability(self) -> None:
        """Q71: over zero nulls the worst gap reads 0.000, a perfect score by luck."""
        corpus = [e for e in self._corpus() if e.label != "same"]
        with pytest.raises(ValueError, match=r"null_stability.*`same` entry"):
            calibrate(corpus, "vtest")

    def test_a_null_that_names_no_base_is_refused_by_name(self) -> None:
        """How `runs/v9` read 0.000: all four nulls carry `"base": ""`.

        One unpaired null among paired ones refuses too. Dropping it would report the
        worst gap over the nulls that remain, which says nothing about the one missing.
        """
        corpus = self._corpus() + [self._entry("null-beta", 0.80, "same", base="")]
        with pytest.raises(ValueError, match=r"1 of 2 name no base.*null-beta \(base ''\)"):
            calibrate(corpus, "vtest")
        corpus[-1] = self._entry("null-beta", 0.80, "same", base="not in the corpus")
        with pytest.raises(ValueError, match="null-beta"):
            calibrate(corpus, "vtest")

    def test_the_summary_says_how_many_nulls_the_number_is_over(self) -> None:
        assert "(over 1 nulls)" in calibrate(self._corpus(), "vtest").summary()

    def test_curve_ends_anchor_low_on_negatives_and_high_past_every_human(self) -> None:
        """The anti-overfitting rule, made mechanical.

        The good end goes beyond the *whole* band rather than beyond any one game in
        it. Anchoring on a single reference game, however canonical, builds a
        similarity-to-that-game score and calls it quality.
        """
        from chess_variant_sandbox.metrics.calibration import suggest_curve_ends

        ends = suggest_curve_ends({"human": [0.5, 0.7, 0.9], "worse": [0.1, 0.2, 0.3]})
        assert ends["negative_high"] == 0.3
        assert ends["human_high"] == 0.9
        assert ends["human_low"] == 0.5
        assert ends["negative_high"] < ends["human_low"]

    def test_a_game_the_roster_cannot_play_is_held_out_of_the_band(self) -> None:
        """Antichess is the case this exists for, and it is not a scoring question.

        Its evaluator maximises material in a game won by shedding it, so the 6400-node
        search rates level with a random mover and the 400-node search finishes last.
        The near-zero skill gradient that follows is a fact about the agents. Letting it
        set the human floor would calibrate the scale against the roster's blind spot
        and quietly widen every margin that depends on it.
        """
        corpus = self._corpus()
        corpus.append(
            ScoredEntry(name="unreadable", total=0.30, viable=False, label="human",
                        sd=0.02, ladder=0.33)
        )
        report = calibrate(corpus, "vtest")
        assert report.unmeasurable == ["unreadable"]
        assert report.human_floor == pytest.approx(0.78)
        assert report.separation == pytest.approx(0.78 - 0.55)

    def test_a_clean_ladder_stays_in_the_band(self) -> None:
        corpus = self._corpus()
        corpus.append(
            ScoredEntry(name="ordinary", total=0.70, viable=True, label="human",
                        sd=0.02, ladder=1.0)
        )
        report = calibrate(corpus, "vtest")
        assert not report.unmeasurable
        assert report.human_floor == pytest.approx(0.70)


class TestCheapGate:
    """The gate must not reject games people actually play.

    It used to. At a 0.90 stall threshold it threw out courier and shatranj, and the
    labelled corpus showed why: stall rate under *random* play does not separate a game
    from a non-game. Makruk stalls 87% of the time and chess 79%, while the deliberately
    uniform army stalls 20%. Random movers are bad at forcing mate and get worse as
    material grows, so the threshold was closest to firing on the big, rich, slow games
    most worth keeping.
    """

    @pytest.mark.slow
    def test_every_committed_ruleset_survives_the_gate(self) -> None:
        from chess_variant_sandbox.loop.gates import smoke_test
        from chess_variant_sandbox.spec import load_spec

        rulesets = Path(__file__).resolve().parents[1] / "rulesets"
        rejected = []
        for path in sorted(rulesets.glob("*.json")):
            spec = load_spec(path)
            # The controls are excluded, and only the controls. They are deliberately
            # degenerate -- `dull` has no reachable win under random play at all -- so
            # the gate catching them is the gate working. The claim being tested is
            # narrower and more useful: it must not reject a game people actually play.
            if spec["name"] in CONTROL_RULESETS:
                continue
            result = smoke_test(spec, games=60, seed=0, time_budget=10.0)
            if not result.passed:
                rejected.append((spec["name"], result.reasons[0]))
        assert not rejected, (
            f"the cheap gate rejects committed rulesets: {rejected}. "
            "Anything needing competent play belongs in the index, not here."
        )

    @pytest.mark.slow
    def test_a_ruleset_with_no_reachable_win_is_still_caught(self) -> None:
        """Relaxing the stall threshold must not blunt the check that does work."""
        from chess_variant_sandbox.loop.gates import smoke_test
        from chess_variant_sandbox.metrics.mutants import build_mutants
        from chess_variant_sandbox.spec import load_spec

        rulesets = Path(__file__).resolve().parents[1] / "rulesets"
        bases = {load_spec(p)["name"]: load_spec(p) for p in rulesets.glob("*.json")}
        wanted = {"mutant-no-captures", "mutant-bare-kings"}
        checked = 0
        for mutant, spec in build_mutants(bases):
            if mutant.name not in wanted:
                continue
            checked += 1
            assert not smoke_test(spec, games=60, seed=0, time_budget=10.0).passed, (
                f"{mutant.name} should not survive the cheap gate"
            )
        assert checked == len(wanted)


# The seed-0 runs Q235 pre-registered against, as the library replay of 2026-10-01 kept
# them when it replaced them (`docs/reference/replay-2026-09-29.md`). Every replay before it
# re-played these two games' top rungs move for move; the threat-aware build does not, so the
# pins below read the kept games, never whatever the library holds now.
SEAT_PINNED_RUNS = RUNS / "replay-2026-09-29" / "replaced"


class TestTheSeatGateOnTheStoredCorpus:
    """Q235's pre-registration, held against the stored canonical runs (record, §4).

    What is pinned is the gate's reading of particular recorded games, a regression pin on
    `metrics/seat.py`, not a claim about either game as the library measures it today. On
    the library's current runs (2026-10-01, threat-aware rungs) neither game draws a seat
    finding: gardner's pair reads 7-7 with 2 draws and tunnel-vertical's 9-7, and both entered
    the band (`replay-2026-09-29.md`, *Verdicts that flipped*).

    Read-only, off the main checkout's `runs/`, and skipped where a run is absent. The
    stored run is found the way `docs/reference/seat-decided.py` finds it: by the
    ruleset hash the Q226 duel records carry.
    """

    def _components(self, name: str):
        from chess_variant_sandbox.metrics.components import compute_components
        from chess_variant_sandbox.spec import load_spec
        from chess_variant_sandbox.tournament.ratings import rate
        from chess_variant_sandbox.tournament.record import read_records

        duel = RUNS / "q226-duel" / name / "games.jsonl"
        if not duel.exists():
            pytest.skip(f"needs {duel}")
        with open(duel, encoding="utf-8") as handle:
            digest = json.loads(handle.readline())["ruleset_hash"]
        games = SEAT_PINNED_RUNS / f"{digest}-0" / "games.jsonl"
        if not games.exists():
            pytest.skip(f"needs {games}")
        records = list(read_records(games))
        # `tunnel-vertical` left the library on 2026-10-12 (Q868, `docs/archive/plan-library-audit.md`);
        # its spec stays beside this file, byte for byte, for the stored games it played.
        here = Path(__file__).resolve().parent
        retired = here / f"_retired_{name.replace('-', '_')}.json"
        spec = load_spec(retired if retired.exists() else here.parent / "rulesets" / f"{name}.json")
        return compute_components(spec, records, rate(records))

    def test_tunnel_vertical_s_stored_sweep_is_few_games_replayed(self) -> None:
        """The pair swept 16-0, which refused it until Q953a. Its top three read 24-12-0
        at 0.065, so the pair's refusal stood alone. The sixteen were eight different
        games, and since Q953a the test reads each once
        (`repeated-games.md` §6): 8-0 is one chance in 128 and passes one run's pair."""
        from chess_variant_sandbox.metrics.seat import TOP_RUNG_SEAT_DETAIL, seat_violations

        components = self._components("tunnel-vertical")
        detail = components.diagnostics[TOP_RUNG_SEAT_DETAIL]
        assert (detail["wins"], detail["draws"], detail["losses"]) == (16, 0, 0)
        assert (detail["distinct_wins"], detail["distinct_losses"]) == (8, 0)
        assert seat_violations(components.raw, components.diagnostics) == []

    def test_gardner_is_refused_by_the_wide_reading_and_not_by_the_pair(self) -> None:
        """The flip the widening was adopted for, on gardner's own stored games. Its
        pair goes 10-1 with 5 draws, one chance in 85, which the pair's gate passes at
        one in a thousand. Its top three go 25-1 with 10 draws, one chance in 1.2
        million. A 100-game duel at `6400` v `51200` gave the first seat 96 of 100."""
        from chess_variant_sandbox.metrics.seat import (
            SEAT_DECIDED_CHANCE,
            TOP_RUNG_SEAT_DETAIL,
            TOP_THREE_SEAT_DECIDED_CHANCE,
            TOP_THREE_SEAT_DETAIL,
            seat_violations,
        )

        components = self._components("gardner")
        pair = components.diagnostics[TOP_RUNG_SEAT_DETAIL]
        assert (pair["wins"], pair["draws"], pair["losses"]) == (10, 5, 1)
        assert components.raw[SEAT_DECIDED_CHANCE] == pytest.approx(24 / 2048)

        wide = components.diagnostics[TOP_THREE_SEAT_DETAIL]
        assert wide["rungs"] == [800, 1600, 6400]
        assert (wide["games"], wide["wins"], wide["draws"], wide["losses"]) == (
            36, 25, 10, 1,
        )
        assert components.raw[TOP_THREE_SEAT_DECIDED_CHANCE] == pytest.approx(
            2 * 27 / 2**26
        )
        violations = seat_violations(components.raw, components.diagnostics)
        assert [v.split("=")[0] for v in violations] == [TOP_THREE_SEAT_DECIDED_CHANCE]


# --------------------------------------------------------------------------
# The mutant transforms, pinned to what they do (Q740, from the Q734m survivors in
# `spec/transform.py`). Each runs on chess because chess is the one small spec whose
# every piece and placement is written out; what is asserted is the transform's
# contract, so the expected values are written here and not read from the code.
# --------------------------------------------------------------------------


def _chess() -> dict:
    from chess_variant_sandbox.spec import load_spec

    return copy.deepcopy(load_spec(Path(__file__).resolve().parents[1] / "rulesets" / "chess.json"))


def _piece(spec: dict, piece_id: str) -> dict:
    return next(p for p in spec["pieces"] if p["id"] == piece_id)


def _squares(spec: dict, piece_id: str) -> list[str]:
    return next(p["squares"] for p in spec["setup"]["placements"] if p["piece"] == piece_id)


def _tiny(pieces: list[dict], placements: list[tuple[str, list[str]]]) -> dict:
    """Just enough spec for the picking helpers, which never validate."""
    return {
        "pieces": pieces,
        "setup": {"placements": [{"piece": p, "squares": s} for p, s in placements]},
    }


class TestADistinctSeatCannotRefuseAForcedLine:
    """Q953a's cost, pinned (`docs/reference/repeated-games.md` §4.3, `docs/bugs.md`).

    Since Q953a the seat check counts each different decided game once. Sweeping bishop
    chess's first player mates in one, and its two strongest rungs play that one game in
    every run, so its top two are one distinct game and the seat refuses nothing. It stays
    out because its length and quick-kill gates refuse it on every run. Read-only, off the
    main checkout's library runs, and skipped where they are absent.
    """

    HASH = "42bd7d06828d"

    def test_sweeping_bishop_chess_stays_refused_by_its_other_gates(self) -> None:
        from chess_variant_sandbox.metrics.components import _seat_readings, compute_components
        from chess_variant_sandbox.metrics.index import gate_violations
        from chess_variant_sandbox.metrics.seat import pooled_seat_reading, seat_details
        from chess_variant_sandbox.spec import load_spec
        from chess_variant_sandbox.tournament.analysis import CorpusAnalysis
        from chess_variant_sandbox.tournament.ratings import rate
        from chess_variant_sandbox.tournament.record import read_records

        spec = load_spec(Path(__file__).resolve().parent.parent / "archive"
                         / "chess-sweeping-bishops.json")
        details, refusals = [], []
        for seed in (0, 1000, 2000, 3000):
            run = RUNS / f"{self.HASH}-{seed}"
            if not (run / "games.jsonl").exists() or not (run / "analysis.json").exists():
                pytest.skip(f"needs {run}")
            records = list(read_records(run / "games.jsonl"))
            details.append(seat_details(_seat_readings(records)[1]))
            components = compute_components(
                spec, records, rate(records), CorpusAnalysis.load(run / "analysis.json")
            )
            weights = load_weights(DEFAULT_WEIGHTS)
            refusals.append({
                text.split("=")[0] for text in gate_violations(components, weights)
            })

        pooled = pooled_seat_reading(details)
        top = pooled["bands"]["top_rung_seat"]
        assert (top["wins"], top["losses"]) == (64, 0)
        assert (top["distinct_wins"], top["distinct_losses"]) == (1, 0)
        assert pooled["refuses"] is False
        for found in refusals:
            assert {"mean_plies_worst_half", "quick_kill_rate_worst_half"} <= found

class TestPickingHelpers:
    def test_the_most_numerous_piece_of_chess_is_the_pawn(self) -> None:
        from chess_variant_sandbox.spec.transform import most_numerous

        assert most_numerous(_chess()) == "P"

    def test_the_royal_is_never_the_most_numerous(self) -> None:
        from chess_variant_sandbox.spec.transform import most_numerous

        spec = _tiny(
            [{"id": "K", "royal": True}, {"id": "A"}],
            [("K", ["a1", "b1", "c1"]), ("A", ["d1"])],
        )
        assert most_numerous(spec) == "A"

    def test_most_numerous_has_an_answer_only_when_something_is_placed(self) -> None:
        from chess_variant_sandbox.spec.transform import most_numerous

        assert most_numerous(_tiny([{"id": "A"}], [])) is None
        only_royal = _tiny([{"id": "K", "royal": True}], [("K", ["a1"])])
        assert most_numerous(only_royal) is None

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            # an undeclared value ranks below 0.5 and above a value of 0 (declared)
            ([("A", None), ("B", 0.5)], "B"),
            ([("A", None), ("B", 0)], "A"),
            ([("A", 2), ("B", 7), ("C", 3)], "B"),
            ([("A", 4), ("B", 4)], "A"),
        ],
    )
    def test_strongest_ranks_by_declared_value(self, values, expected) -> None:
        from chess_variant_sandbox.spec.transform import strongest

        pieces = [{"id": i} if v is None else {"id": i, "value": v} for i, v in values]
        assert strongest({"pieces": pieces}) == expected

    def test_the_strongest_piece_is_never_the_royal(self) -> None:
        from chess_variant_sandbox.spec.transform import strongest

        spec = {"pieces": [{"id": "K", "royal": True, "value": 99}, {"id": "A", "value": 1}]}
        assert strongest(spec) == "A"
        assert strongest({"pieces": [{"id": "K", "royal": True}]}) is None
        assert strongest({}) is None


class TestPiecewiseMutants:
    def test_set_moves_to_gives_every_non_royal_piece_the_source_moves(self) -> None:
        from chess_variant_sandbox.spec.transform import set_moves_to

        original = _chess()
        spec = set_moves_to(copy.deepcopy(original), "N")
        knight_moves = _piece(original, "N")["moves"]
        for piece in spec["pieces"]:
            if piece["id"] == "K":
                assert piece["moves"] == _piece(original, "K")["moves"]
                assert piece["value"] == 0
            else:
                assert piece["moves"] == knight_moves
                assert piece["value"] == 3
        # separate copies, not one shared list
        assert _piece(spec, "P")["moves"] is not _piece(spec, "R")["moves"]
        assert "promotion" not in spec["rules"]
        assert "en_passant" not in spec["rules"]
        assert "castling" in spec["rules"]

    def test_set_moves_to_writes_value_one_when_the_source_declares_none(self) -> None:
        from chess_variant_sandbox.spec.transform import set_moves_to

        spec = _chess()
        del _piece(spec, "N")["value"]
        set_moves_to(spec, "N")
        assert {p["id"]: p["value"] for p in spec["pieces"]} == {
            "P": 1, "N": 1, "B": 1, "R": 1, "Q": 1, "K": 0,
        }

    def test_set_moves_to_an_unknown_piece_changes_nothing(self) -> None:
        from chess_variant_sandbox.spec.transform import set_moves_to

        assert set_moves_to(_chess(), "Z") == _chess()
        assert set_moves_to(_chess(), None) == _chess()

    def test_omnipotent_piece_rewrites_the_strongest_and_only_that(self) -> None:
        from chess_variant_sandbox.spec.transform import omnipotent_piece

        original = _chess()
        spec = omnipotent_piece(copy.deepcopy(original))
        assert _piece(spec, "Q")["moves"] == [
            {"kind": "slide", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"},
            {
                "kind": "leap",
                "vectors": [[1, 2], [1, 3], [2, 3]],
                "symmetry": "dihedral8",
                "mode": "both",
            },
        ]
        assert _piece(spec, "Q")["value"] == 40
        for piece in original["pieces"]:
            if piece["id"] != "Q":
                assert _piece(spec, piece["id"]) == piece
        assert spec["rules"] == original["rules"]

    def test_widen_royal_makes_the_king_a_slider_and_drops_castling(self) -> None:
        from chess_variant_sandbox.spec.transform import widen_royal

        original = _chess()
        spec = widen_royal(copy.deepcopy(original), 6)
        assert _piece(spec, "K")["moves"] == [
            {"kind": "slide", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4",
             "mode": "both", "range": 6},
        ]
        for piece in original["pieces"]:
            if piece["id"] != "K":
                assert _piece(spec, piece["id"]) == piece
        assert "castling" not in spec["rules"]
        assert spec["rules"]["en_passant"] is True

    def test_widen_royal_default_range_and_a_spec_without_rules(self) -> None:
        from chess_variant_sandbox.spec.transform import widen_royal

        spec = widen_royal(_chess())
        assert _piece(spec, "K")["moves"][0]["range"] == 6
        bare = {"pieces": [{"id": "K", "royal": True, "moves": [{"kind": "leap"}]}]}
        assert widen_royal(bare, 3)["pieces"][0]["moves"] == [{"kind": "slide", "range": 3}]
        assert "rules" not in bare

    def test_widen_royal_leaves_a_castling_that_is_not_a_table(self) -> None:
        from chess_variant_sandbox.spec.transform import widen_royal

        spec = _chess()
        spec["rules"]["castling"] = True
        assert widen_royal(spec)["rules"]["castling"] is True

    def test_rename_everything_numbers_the_pieces_in_declaration_order(self) -> None:
        from chess_variant_sandbox.spec.transform import rename_everything

        spec = rename_everything(_chess())
        assert [(p["id"], p["name"]) for p in spec["pieces"]] == [
            ("P", "piece 1"), ("N", "piece 2"), ("B", "piece 3"),
            ("R", "piece 4"), ("Q", "piece 5"), ("K", "piece 6"),
        ]

    def test_reverse_backrank_mirrors_every_placement_left_to_right(self) -> None:
        from chess_variant_sandbox.spec.transform import reverse_backrank

        spec = reverse_backrank(_chess())
        assert _squares(spec, "Q") == ["e1"]
        assert _squares(spec, "K") == ["d1"]
        assert _squares(spec, "N") == ["b1", "g1"]
        assert _squares(spec, "R") == ["a1", "h1"]
        assert _squares(spec, "B") == ["c1", "f1"]
        assert _squares(spec, "P") == [f"{f}2" for f in "abcdefgh"]
        assert "castling" not in spec["rules"]

    def test_reverse_backrank_reads_the_board_width(self) -> None:
        from chess_variant_sandbox.spec.transform import reverse_backrank

        spec = {
            "board": {"width": 5},
            "setup": {"placements": [{"piece": "A", "squares": ["a1", "b3"]}]},
        }
        assert reverse_backrank(spec)["setup"]["placements"][0]["squares"] == ["d3", "e1"]


class TestPromoteEarly:
    def test_chess_promotes_two_ranks_past_the_pawn_start(self) -> None:
        from chess_variant_sandbox.spec.transform import promote_early

        spec = promote_early(_chess())
        assert spec["rules"]["promotion"]["rank"] == 3
        assert promote_early(_chess(), 4)["rules"]["promotion"]["rank"] == 5

    def test_the_rank_follows_where_the_pieces_are_placed(self) -> None:
        from chess_variant_sandbox.spec.transform import promote_early

        spec = _chess()
        _squares(spec, "P")[:] = [f"{f}4" for f in "abcdefgh"]
        assert promote_early(spec)["rules"]["promotion"]["rank"] == 5

    def test_the_highest_placed_square_decides(self) -> None:
        from chess_variant_sandbox.spec.transform import promote_early

        spec = _chess()
        _squares(spec, "P")[:] = ["a2", "b4", "c3"]
        assert promote_early(spec)["rules"]["promotion"]["rank"] == 5

    def test_an_unplaced_promoter_falls_back_to_rank_one(self) -> None:
        from chess_variant_sandbox.spec.transform import promote_early

        spec = _chess()
        spec["setup"]["placements"] = [
            p for p in spec["setup"]["placements"] if p["piece"] != "P"
        ]
        assert promote_early(spec)["rules"]["promotion"]["rank"] == 3

    @pytest.mark.parametrize(
        ("pawn_square", "height", "expected"),
        [("a6", 8, 7), ("a7", 8, 7), ("a5", 8, 6), ("a4", 8, 5), ("a4", 6, 5), ("a2", 3, 2)],
    )
    def test_the_rank_never_leaves_the_board(self, pawn_square, height, expected) -> None:
        from chess_variant_sandbox.spec.transform import promote_early

        spec = _chess()
        _squares(spec, "P")[:] = [pawn_square]
        spec["board"]["height"] = height
        assert promote_early(spec)["rules"]["promotion"]["rank"] == expected

    def test_a_board_without_a_height_is_eight_high(self) -> None:
        from chess_variant_sandbox.spec.transform import promote_early

        spec = _chess()
        _squares(spec, "P")[:] = ["a7"]
        del spec["board"]["height"]
        assert promote_early(spec)["rules"]["promotion"]["rank"] == 7

    def test_a_spec_with_no_promotion_table_is_returned_alone(self) -> None:
        from chess_variant_sandbox.spec.transform import promote_early

        spec = _chess()
        spec["rules"]["promotion"] = "last"
        assert promote_early(spec) == spec
        assert promote_early({}) == {}


class TestStampsAndConfinement:
    def test_captures_come_back_leaves_the_stamp_and_writes_a_capture_hand(self) -> None:
        from chess_variant_sandbox.spec.transform import captures_come_back

        # An edit writes no stamp since Q1008c: nothing reads it to decide what a spec may use.
        spec = captures_come_back(dict(_chess(), schema_version=1))
        assert spec["schema_version"] == 1
        assert spec["rules"]["drops"] == {"from_captures": True}

    def test_captures_come_back_never_lowers_a_higher_version(self) -> None:
        from chess_variant_sandbox.spec.transform import captures_come_back

        spec = _chess()
        spec["schema_version"] = 9
        assert captures_come_back(spec)["schema_version"] == 9

    def test_the_idle_army_confines_exactly_the_pieces_not_named(self) -> None:
        from chess_variant_sandbox.spec.transform import confine_all_but

        original = dict(_chess(), schema_version=1)  # chess's stamp before Q1008b
        spec = confine_all_but(copy.deepcopy(original), "P", "K", "Q")
        assert spec["schema_version"] == 1
        regions = {p["id"]: p.get("region") for p in spec["pieces"]}
        assert regions == {
            "P": None, "K": None, "Q": None,
            "N": ["b1", "g1"], "B": ["c1", "f1"], "R": ["a1", "h1"],
        }

    def test_the_idle_army_as_the_mutant_builds_it(self) -> None:
        from chess_variant_sandbox.spec.transform import (
            confine_all_but,
            most_numerous,
            royal_id,
            strongest,
        )

        spec = _chess()
        spec = confine_all_but(spec, most_numerous(spec), royal_id(spec), strongest(spec))
        assert sorted(p["id"] for p in spec["pieces"] if "region" in p) == ["B", "N", "R"]

    def test_confinement_skips_unplaced_pieces_and_none_names(self) -> None:
        from chess_variant_sandbox.spec.transform import confine_all_but

        spec = _chess()
        spec["setup"]["placements"] = [
            p for p in spec["setup"]["placements"] if p["piece"] != "B"
        ]
        confine_all_but(spec, None, "P")
        regions = {p["id"]: p.get("region") for p in spec["pieces"]}
        assert regions["B"] is None and regions["P"] is None
        assert regions["Q"] == ["d1"]

    def test_confinement_never_lowers_a_higher_version(self) -> None:
        from chess_variant_sandbox.spec.transform import confine_all_but

        spec = _chess()
        spec["schema_version"] = 7
        assert confine_all_but(spec, "P")["schema_version"] == 7
