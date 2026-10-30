"""The interestingness index: one number, and everything needed to argue with it.

An index that reports only a total is not usable in a design loop. When a candidate
scores 0.71 the next question is always *why*, and the answer has to be specific
enough to steer the next proposal. So every result carries the raw measurements, the
per-component scores, each component's contribution to the total, and any gate it
failed.

The LLM judgement is blended in here rather than folded into a component, and only at
a declared weight. It is a different kind of evidence -- an opinion about whether a
game is worth playing, formed without seeing any of these numbers -- and keeping it
separable means the objective index can always be read on its own.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from chess_variant_sandbox.metrics.bootstrap import (
    DETERMINISTIC_PAIRING_WARN,
    BootstrapResult,
    bootstrap_index,
    clears,
)
from chess_variant_sandbox.metrics.calibration import MEASURABLE_LADDER
from chess_variant_sandbox.metrics.components import (
    KEYED_COMPONENTS,
    Components,
    compute_components,
)
from chess_variant_sandbox.metrics.scoring import (
    DEFAULT_WEIGHTS,
    Weights,
    WeightsError,
    load_weights,
)
from chess_variant_sandbox.metrics.seat import seat_violations
from chess_variant_sandbox.tournament.analysis import ANALYSIS_COMPONENTS, CorpusAnalysis
from chess_variant_sandbox.tournament.heuristics import HeuristicReport
from chess_variant_sandbox.tournament.ratings import RatingTable
from chess_variant_sandbox.tournament.record import GameRecord

# Every component only the analysis pass can supply: the six it names itself, and the
# two that read the position keys it persists, which `metrics.components` names so that
# declaring them moves no `analysis` fingerprint.
_FROM_THE_PASS = frozenset(ANALYSIS_COMPONENTS) | frozenset(KEYED_COMPONENTS)

# The third verdict, *not measured* (plan-teacher §10.16.4 point 5): this roster on this
# game produced no verdict. It is current, not provisional, and it carries no number, so
# it enters no band, gap or leaderboard comparison. Readers print these words.
VIABLE = "viable"
NOT_VIABLE = "not viable"
NOT_MEASURED = "not measured"
NOT_MEASURED_MANY = (
    "its computer players could not tell a stronger search from a weaker one on most of "
    "its runs"
)
NOT_MEASURED_ONE = (
    "its computer players could not tell a stronger search from a weaker one on this run"
)


def verdict_word(viable: bool | None, measured: bool | None = True) -> str:
    """The verdict in the words every reader shows: `viable`, `not viable`, `not measured`.

    ``measured`` is False only for the third outcome; a stored file that predates it has
    no such key, and reads as measured, which is what its verdict was taken as.
    """
    if measured is False:
        return NOT_MEASURED
    return VIABLE if viable else NOT_VIABLE


@dataclass
class IndexResult:
    """A scored ruleset."""

    ruleset_hash: str
    ruleset_name: str
    weights_version: str
    total: float
    viable: bool
    violations: list[str]
    scores: dict[str, float]
    contributions: dict[str, float]
    raw: dict[str, float]
    families: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # Components withheld because the *instrument* could not read this ruleset, not
    # because the ruleset was bad at them. Empty for almost everything.
    unreadable: tuple[str, ...] = ()
    llm_score: float | None = None
    llm_notes: str = ""
    combined: float | None = None
    # How much of ``total`` is measurement and how much is luck. Absent when the corpus
    # was too small to resample, and absent is reported as absent -- a zero here would
    # read as "perfectly precise", which is the one thing it never is.
    interval: dict[str, Any] | None = None
    # False for one result alone: a run the roster could not read whose readable gates all
    # passed (`_unreadable_result`, plan-teacher §10.16.4 point 1). It neither accepts nor
    # refuses, so its word is *not measured*; `viable` stays False beside it because it
    # was not admitted, and every reader asks this first. A refusal is always measured.
    measured: bool = True

    @property
    def verdict_word(self) -> str:
        """`viable`, `not viable` or `not measured`: the run's own verdict."""
        return verdict_word(self.viable, self.measured)

    @property
    def sd(self) -> float:
        """Sampling standard deviation of the total, or 0.0 if it was not measured."""
        return float(self.interval["sd"]) if self.interval else 0.0

    @property
    def scored(self) -> bool:
        """Whether ``total`` is a score or a placeholder.

        Both refusals below -- a roster that cannot read the ruleset, and a component the
        corpus could not supply -- set ``total`` to 0.0 and leave ``contributions``
        empty, because there is no combination to report. A real score always has one
        contribution per weighted component. So the emptiness is the signal, and a reader
        that averages totals asks this before it reads one: a placeholder 0.0 published
        as a number is the one failure `metrics.seeds` exists to prevent.
        """
        return bool(self.contributions)

    def separated_from(self, other: IndexResult) -> bool:
        """Whether this score is better than another by more than the noise in both.

        The test that keeps a design loop honest. Without it a run will happily accept
        a candidate 0.02 above its parent when 0.02 is a third of the interval, and
        climb sampling error for as many generations as it is given.

        Two conditions, not one. The interval is resampled from the *objective* total,
        while ``headline`` blends in a judge score that carries noise of its own and is
        not resampled at all. Requiring the headline to be higher as well means a
        judge-driven difference can never be reported as separated on the strength of
        an interval that never saw it.
        """
        if not self.interval or not other.interval:
            return self.headline > other.headline
        # An interval built on deterministic sweeps is not evidence, and the two tests
        # have to agree about that or a loop built on one while reporting the other
        # would disagree with itself -- which is the whole reason `clears` is one
        # function. See `BootstrapResult.beats` for what the artifact was.
        sweeps = max(
            float(self.interval.get("deterministic_pairings", 0.0)),
            float(other.interval.get("deterministic_pairings", 0.0)),
        )
        if sweeps >= DETERMINISTIC_PAIRING_WARN:
            return False
        return self.headline > other.headline and clears(
            float(self.interval["ci_low"]), float(other.interval["ci_high"])
        )

    @property
    def headline(self) -> float:
        """The number to rank by: blended if a judgement is present, objective if not."""
        return self.combined if self.combined is not None else self.total

    def weakest(self, count: int = 3) -> list[tuple[str, float]]:
        """The lowest-scoring components -- the specific thing to fix next."""
        return sorted(self.scores.items(), key=lambda kv: kv[1])[:count]

    def strongest(self, count: int = 3) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda kv: -kv[1])[:count]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.as_dict(), handle, indent=2)
        return path

    def summary(self) -> str:
        """The result in a few lines, headed by its number and the runs it rests on.

        An `IndexResult` scores one tournament's games, so its number is always one
        run's, and the headline says so in the page's words (`web/page/core.py`'s
        `runsPhrase`): *as the one run*. Several runs become one number only in
        `metrics.seeds`, and its readers say *as the mean of four runs*.

        A result that is not `scored` holds a 0.0 that stands for no total at all, so its
        headline says *no score* and prints no number: `NOT VIABLE` below says why.
        """
        head = f"{self.ruleset_name} [{self.ruleset_hash}]  "
        if self.scored:
            band = ""
            if self.interval:
                band = (
                    f" +/-{self.interval['sd']:.3f} "
                    f"[{self.interval['ci_low']:.3f}, {self.interval['ci_high']:.3f}]"
                )
            head += (
                f"index={self.headline:.3f}{band} as the one run "
                f"(objective {self.total:.3f}, weights {self.weights_version})"
            )
        else:
            head += f"no score (weights {self.weights_version})"
        lines = [head]
        if not self.measured:
            lines.append(f"  NOT MEASURED: {NOT_MEASURED_ONE}")
            lines.extend(f"    - {violation}" for violation in self.violations)
        elif not self.viable:
            lines.append("  NOT VIABLE:")
            lines.extend(f"    - {violation}" for violation in self.violations)
        if self.families:
            lines.append("  by family: " + ", ".join(
                f"{name} {value:.2f}" for name, value in sorted(self.families.items())
            ))
        if self.scores:
            lines.append("  weakest: " + ", ".join(
                f"{name} {value:.2f}" for name, value in self.weakest()
            ))
        if self.llm_score is not None:
            lines.append(f"  judge: {self.llm_score:.2f}")
        return "\n".join(lines)


def score_ruleset(
    spec: dict[str, Any],
    records: list[GameRecord],
    ratings: RatingTable,
    weights: Weights | str = DEFAULT_WEIGHTS,
    llm_score: float | None = None,
    llm_notes: str = "",
    analysis: CorpusAnalysis | None = None,
    heuristics: HeuristicReport | None = None,
    interval: bool = True,
) -> IndexResult:
    """Measure a ruleset and score it under one weights version.

    ``interval`` resamples the corpus to put an error bar on the total. It is on by
    default because a score without one invites a comparison the evidence does not
    support, and it is cheap against the tournament it reads: **2.7 s a run**, measured
    over 53 canonical runs on 2026-09-06 (Q48) under an exclusive lease. "A fraction of
    a second" is what this used to say and it was wrong by an order of magnitude;
    ``components.SpecFacts`` quotes 41.8 s for the same stage, which was true before the
    memoisation that docstring describes and has not been true since.

    ``heuristics`` carries the third pass. Nothing it produces is scored under any
    weights version, so it changes no total -- it travels in ``raw`` and in the
    diagnostics, beside ``ladder_monotonicity``, which is the other measurement here
    that describes the instrument rather than the game.
    """
    if isinstance(weights, str):
        weights = load_weights(weights)

    components = compute_components(spec, records, ratings, analysis, heuristics)
    resampled = (
        bootstrap_index(
            spec,
            records,
            weights,
            components_of_interest=("skill_gradient",),
            analysis=analysis,
        )
        if interval
        else None
    )
    return score_components(
        spec_hash_value=records[0].ruleset_hash,
        name=spec.get("name", records[0].ruleset_name),
        components=components,
        weights=weights,
        llm_score=llm_score,
        llm_notes=llm_notes,
        resampled=resampled,
    )


# Components whose meaning depends on the rating ladder ordering the agents by
# strength. If it does not, every one of them describes the *agents* rather than the
# game, and reporting them as properties of the design is the measurement lying about
# its own scope.
#
# The depth family by construction, plus `draw_slope`, which is decisiveness measured
# *as a function of strength* and is therefore the same claim in a different family.
# `blunder_recoverability` is deliberately not here: it forces one arbitrary move and
# asks whether the game is still a game, which does not require the ladder to be
# ordered -- only that both seats play the same way afterwards, which they do.
LADDER_DEPENDENT = frozenset({
    "skill_gradient",
    "gradient_persistence",
    "rating_spread",
    "search_advantage",
    "draw_slope",
})

# Components that assume material is what a player is trying to accumulate. Antichess is
# the standing counterexample and any cyclic-capture game from schema 3 is another: a
# lead in material there is not a lead. Withheld on the same evidence, because a ladder
# the evaluator cannot order is the symptom of an evaluator pointed the wrong way.
MATERIAL_DEPENDENT = frozenset({"lead_conversion"})


def withheld_components(components: Components, weights: Weights) -> tuple[str, ...]:
    """Which scored components this corpus cannot support a claim about.

    `ladder_monotonicity` is a diagnostic and never a component: it reports the share of
    budget steps on which the larger search actually rated higher. Antichess measures
    0.33 -- its 6400-node search rates level with a random mover, because the
    spec-derived evaluator maximises material in a game won by shedding it. Scoring that
    as a shallow game would be the same error as letting the cheap gate reject shatranj
    for being slow.

    So the number becomes a gate on the *instrument* rather than on the game: below
    `MEASURABLE_LADDER`, every component derived from the ladder is withheld and the
    ruleset is reported as unmeasured rather than as bad.
    """
    monotonicity = components.raw.get("ladder_monotonicity")
    gradient = components.raw.get("skill_gradient")
    flat = monotonicity is not None and monotonicity < MEASURABLE_LADDER
    # A *negative* gradient is the unambiguous case, and it is the one a share of rising
    # steps cannot see. `ladder_monotonicity` counts how many steps rose; under a ladder
    # with no relationship at all that share averages 0.5, so a threshold at 0.5 admits
    # half of pure noise -- and widening the roster from four budgets to six pushed
    # antichess from 0.33 to 0.60, straight over the line, while its `skill_gradient`
    # stayed at -0.56.
    #
    # Searching *harder* rating *worse* cannot be a property of a game. It is the
    # spec-derived evaluator maximising material in a game won by shedding it, which is
    # the diagnosis this project already had for antichess before any of this ran.
    inverted = gradient is not None and gradient < 0.0
    if not (flat or inverted):
        return ()
    at_risk = LADDER_DEPENDENT | MATERIAL_DEPENDENT
    return tuple(sorted(name for name in weights.weights if name in at_risk))


def _unreadable_result(
    spec_hash_value: str,
    name: str,
    components: Components,
    weights: Weights,
    unreadable: tuple[str, ...],
    llm_score: float | None,
    llm_notes: str,
) -> IndexResult:
    """A ruleset the roster cannot read: no total, and no verdict about the design.

    Not scored on what is left, and that is the point. A geometric mean over six
    families is a different quantity from one over seven, so a partial total would
    invite exactly the comparison it cannot support -- and the loop would read the
    smaller number as "worse game" when what happened is "wrong instrument".

    The components that *were* measured are still reported, because they are still
    facts, and because the next reader's question is going to be "how badly?".

    **The blind-run abstention** (plan-teacher §10.16.4 point 1, rule (e)). Gates
    disqualify, and a gate read on a component this instrument *can* read is a fact
    whatever happened to the ladder. So the run's readable gates are checked: the weights'
    gates whose component is neither withheld here nor in `LADDER_DEPENDENT`, and that
    the corpus measured. Any failure refuses the run, in the usual gate sentence, with the
    *cannot read* sentence beside it. Otherwise the run abstains: `measured` is False, and
    its word is *not measured*. The seat gates are not among the readable ones: they are
    not the weights version's gates, and they read the ladder's top rungs by budget; a
    library game's pooled seat still refuses on top (`seeds.combine`).
    """
    partial = {
        component: weights.curves[component].score(components.raw[component])
        for component in weights.weights
        if component in components.raw
    }
    monotonicity = components.raw.get("ladder_monotonicity", 0.0)
    gradient = components.raw.get("skill_gradient", 0.0)
    why = (
        f"the rating ladder orders the agents on only {monotonicity:.0%} of its budget "
        "steps"
        if monotonicity < MEASURABLE_LADDER
        else f"a larger search rates *worse* here ({gradient:.1f} Elo per doubling)"
    )
    refusals = readable_gate_violations(components, weights, unreadable)
    return IndexResult(
        ruleset_hash=spec_hash_value,
        ruleset_name=name,
        weights_version=weights.version,
        total=0.0,
        viable=False,
        violations=refusals + [
            f"this roster cannot read this ruleset: {why}, so "
            f"{', '.join(unreadable)} would describe the players rather than the game"
        ],
        measured=bool(refusals),
        scores={k: round(v, 4) for k, v in partial.items()},
        contributions={},
        raw={k: round(v, 4) for k, v in components.raw.items()},
        families={},
        diagnostics=dict(components.diagnostics),
        unreadable=unreadable,
        llm_score=llm_score,
        llm_notes=llm_notes,
    )


def readable_gate_violations(
    components: Components, weights: Weights, unreadable: tuple[str, ...]
) -> list[str]:
    """The failures of the gates a blind run can still read (§10.16.4 point 1).

    A gate whose component is withheld, or derives from the ladder, describes the
    players here and is not read. A gate whose component the corpus did not supply was
    not read either: a gate it could not read is not one it failed.
    """
    blind = set(unreadable) | LADDER_DEPENDENT
    readable = [
        gate for gate in weights.gates
        if gate.component not in blind and gate.component in components.raw
    ]
    failures = (gate.check(components.raw[gate.component]) for gate in readable)
    return [failure for failure in failures if failure]


def gate_violations(components: Components, weights: Weights) -> list[str]:
    """The weights' gates, then the seat gates.

    The seat gates stand under every weights version because they read no curve: they are
    findings about this roster's strongest rungs and these games, the same standing as
    ``withheld_components``. They live in code rather than in a weights file for the same
    reason the fingerprint covers ``metrics/*.py`` and not ``weights/*.json``: a gate
    added to a JSON file would change verdicts without staling a single stored score.
    """
    violations = weights.check_gates(components.raw)
    violations.extend(seat_violations(components.raw, components.diagnostics))
    return violations


def score_components(
    spec_hash_value: str,
    name: str,
    components: Components,
    weights: Weights,
    llm_score: float | None = None,
    llm_notes: str = "",
    resampled: BootstrapResult | None = None,
) -> IndexResult:
    """Score already-computed measurements. Re-scoring never re-simulates.

    A component the corpus could not supply is not scored as zero. Usually that means a
    weights version is asking for something the run predates, and raising by name is the
    right answer. But there is one case where it is a fact about the ruleset: a game that
    ends in five plies offers no position to force a blunder into, so
    ``blunder_recoverability`` has nothing to measure. Such a ruleset always fails a gate
    too -- a five-ply game is below the ``mean_plies`` minimum -- so it is reported as
    disqualified, with the unmeasurable component named among its violations, rather than
    crashing the scorer or being handed a fabricated zero.
    """
    unreadable = withheld_components(components, weights)
    if unreadable:
        return _unreadable_result(
            spec_hash_value, name, components, weights, unreadable, llm_score, llm_notes
        )

    absent = sorted(
        name
        for name in weights.weights
        if name not in components.raw and name not in weights.optional
    )
    unmeasured = sorted(
        name
        for name in weights.optional
        if name in weights.weights and name not in components.raw
    )
    if absent:
        violations = gate_violations(components, weights)
        if not violations:
            # Nothing wrong with the ruleset: the corpus is simply missing a
            # measurement this version needs. That is a pipeline error, not a verdict.
            fix = ""
            if all(name in _FROM_THE_PASS for name in absent):
                # Not a corpus that is too old to score -- a corpus that has not had its
                # second pass run. That pass replays the games already played and needs
                # no new ones, so saying which command produces the missing numbers is
                # the whole difference between an error and a dead end.
                fix = (
                    ". These come from the engine-derived second pass: run "
                    "`chess-variant-sandbox analyse <run>`, which replays the games already played "
                    "and needs no new ones"
                )
                if any(name in KEYED_COMPONENTS for name in absent):
                    # The pass may have run and still not supply them: an
                    # `analysis.json` written before it persisted the position keys.
                    fix += ", and rewrites an `analysis.json` that predates the position keys"
            raise WeightsError(
                f"weights {weights.version} expect components not present in the "
                f"measurements: {', '.join(absent)}{fix}"
            )
        violations.append(
            f"could not be measured on this corpus: {', '.join(absent)}"
        )
        partial = {
            name: weights.curves[name].score(components.raw[name])
            for name in weights.weights
            if name in components.raw
        }
        return IndexResult(
            ruleset_hash=spec_hash_value,
            ruleset_name=name,
            weights_version=weights.version,
            total=0.0,
            viable=False,
            violations=violations,
            scores={k: round(v, 4) for k, v in partial.items()},
            contributions={},
            raw={k: round(v, 4) for k, v in components.raw.items()},
            families={},
            diagnostics=dict(components.diagnostics),
            llm_score=llm_score,
            llm_notes=llm_notes,
        )

    scores = weights.score_all(components.raw)
    total, contributions = weights.combine(scores)
    violations = gate_violations(components, weights)
    families = weights.family_scores(scores)

    combined = None
    if llm_score is not None:
        combined = (1.0 - weights.llm_weight) * total + weights.llm_weight * llm_score

    diagnostics = dict(components.diagnostics)
    diagnostics["arithmetic_total"] = round(weights.arithmetic_total(scores), 4)
    if unmeasured:
        # Recorded rather than hidden. The family keeps its full weight and is scored on
        # what could be measured, which is the right answer when the evidence is absent
        # rather than bad -- but a reader comparing two totals deserves to know that one
        # of them rests on fifteen measurements and the other on sixteen.
        diagnostics["unmeasured"] = unmeasured

    return IndexResult(
        ruleset_hash=spec_hash_value,
        ruleset_name=name,
        weights_version=weights.version,
        total=round(total, 4),
        viable=not violations,
        violations=violations,
        scores={k: round(v, 4) for k, v in scores.items()},
        contributions={k: round(v, 4) for k, v in contributions.items()},
        raw={k: round(v, 4) for k, v in components.raw.items()},
        families={k: round(v, 4) for k, v in families.items()},
        diagnostics=diagnostics,
        llm_score=llm_score,
        llm_notes=llm_notes,
        combined=round(combined, 4) if combined is not None else None,
        interval=resampled.as_dict() if resampled is not None else None,
    )
