"""Turning raw measurements into scores, under versioned weights.

Three rules govern this module, and they exist because an index is very easy to make
useless without noticing.

**Weights are versioned and travel with every score.** A version is a revision of the one
live scorer, and exactly one is current (`DEFAULT_WEIGHTS`). A score is an evolving
estimate: worked out under the current version it is *current*; worked out under an
earlier one it is *provisional* -- still shown, marked with the version it came from and
its date, never shown as current and never compared with a current score or placed on the
current leaderboard (`docs/archive/plan-evolving-score.md` §3). So the version is recorded on the
result, not on the run, and the basis stamp decides which a reader holds.

**Most components are bands, not directions.** Game length, branching factor, and
swinginess are all things where more is better until suddenly it is much worse. A
monotonic "higher is better" curve on any of them tells the design loop to maximise
something no player wants.

**Curves are calibrated against the reference rulesets, not the current generation.**
If the scale were fitted to whatever the search has produced so far it would drift
upward as the search progresses, and every generation would look like an improvement.

**Since `v12`, every curve end says where it comes from** (`SOURCE_KINDS`): the limit the
quantity cannot pass, the gate on the same measurement, a one-sentence meaning, or a quantile
of the published band games read once and pinned. The band checks the scale (`calibrate`)
and places none of it, so the scale does not move when the set of reference games does
(`docs/archive/plan-evolving-score.md` §4).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WEIGHTS_DIR = Path(__file__).parent / "weights"

# Bump this when a new calibration lands. Old versions stay on disk: a score computed
# under v1 is a different quantity from one computed under v2, and being able to
# reproduce the old number is what makes that statement checkable rather than a claim.
#
# Every caller that does not name a version reads this one, which is the point: while
# this said "v2" the design loop was evolving against the weights version whose four
# defects v3 was built to fix, and the CLI defaulted one subcommand to v3 and two to v2,
# so a corpus could be scored under a mix of versions with nothing recording which.
# Components asked for by the current version come from the analysis pass, so anything
# that scores under it has to run that pass first -- see ``tournament/analysis.py``.
#
# v5 is the first re-anchoring since v3, run against a corpus that contains games schema
# 3 and 4 made expressible. A wider roster (six minimax budgets), three components folded
# in from the stage-12 measurement report, and every curve end re-placed mechanically by
# `chess-variant-sandbox anchor` from a corpus the held-out slice was kept out of. Human band
# 0.731-0.862, separation -0.027, sign accuracy 92%, null stability 0.000; full report in
# docs/reference/v5-calibration.md.
#
# v4 (v3 with ``llm_weight`` at zero) and v3 are kept because a score records the version
# that produced it and versions are not edited. What moved between them was only where
# the model's opinion enters -- as a pairwise ordering breaking ties the interval cannot,
# rather than as an absolute score blended into the headline.
# v6 is v5 re-anchored, not re-designed: same sixteen components, same families, same
# weights, ten of sixteen curve ends re-placed mechanically. What forced it was the
# agents -- stage 17 gave the evaluator terms for territory, a score at the clock, a
# count of checks and running out of a piece, so nine of the fifty-eight corpus entries
# play differently.
#
# It calibrates *worse* than v5 on every headline number and all of it is one ruleset:
# `antichess` was gated out under v5 because the roster could not read it, and joins the
# band at 0.686, the floor. Excluding it, v6 is 0.728-0.854 with separation -0.028
# against v5's 0.731-0.862 and -0.027. Full report in docs/reference/v6-calibration.md.
#
# v7 is v6 re-anchored on the same terms -- same components, same families, same weights,
# fifteen of sixteen curve ends re-placed mechanically -- and what forced it is stage 17
# phase C: bounds in the transposition table, quiescence, killers and history ordering,
# and the terminal check hoisted above the table for every ruleset. A change to the
# *search* moves every game in the corpus, so unlike phase B this one has no inert set.
#
# It calibrates worse again, and worse by more, and the decomposition is in
# docs/reference/v7-calibration.md: of 0.516 lost separation, 0.391 is `racingkings`
# no longer failing the stall gate -- a game whose win condition the evaluator provably
# cannot price, entering the band at 0.280 and setting the floor there -- and the rest is
# `mutant-instant-promotion`, the standing weakness under v5 and v6 alike, rising to
# 0.866 and second place in the corpus.
#
# It is the default anyway, for the same forward-looking reason v6 was, and the reason is
# stronger here: the loop scores candidates played by *today's* agents, phase C changed
# what those agents do in every ruleset, and v6's curves are now fitted to games nothing
# in this repository can still produce. Keeping an older version as the default would be
# the error `DEFAULT_WEIGHTS` sitting at `v2` used to be, with no inert set to soften it.
# v10 is v9's curves to the digit and is not a re-anchor. Two changes: the four outcome
# gates read the worse half of the budget ladder rather than the pooled graded set, at a
# midpoint pre-registered before the corpus was swept; and three components leave, each
# dropped weight going to its own family's survivors pro rata so that no family total
# and no exponent moves. What forced it is one control measuring `viable` while being
# two different games -- an opening trap against the rungs that cannot see it and a dead
# shuffle against the rungs that can, each half's failure the other half's alibi. The
# drops are `progress_irreversibility` (inert: 41 of 43 band members score exactly 1.0)
# and `gradient_persistence` and `draw_slope` (roster-bound: rank rho -0.09 and -0.61
# between two rosters, against a threshold of 0.50 fixed before the table was computed).
# Losing three terms near 1.0 from a geometric mean lowers every total by a mean 0.045,
# so v10's scale is not v9's and the two must not share a leaderboard. Full report in
# docs/reference/v10-calibration.md.
#
# v11 is a re-anchor and a re-design at once (docs/archive/plan-variety.md §5.4, C2). Rule yield
# and its elegance family leave the score on the user's decision of 2026-09-23; the
# variety pair (`opening_diversity`, `distinct_game_rate`) gives way to
# `position_divergence`, alone in the family at its 0.099, adopted by C1's pre-registered
# table; every other weight is v10's, and `combine` renormalises the six families by
# their own total (0.9009). Every curve end was then re-placed by `chess-variant-sandbox
# anchor` on `runs/v11` -- the first anchor on the live roster, and the first under
# `piece_utilisation`'s held-type denominator, the identity bootstrap, the hand at h = 1
# and 48 blunder trials. Those changes land together and their effects on a score cannot
# be told apart. Full report in docs/reference/v11-calibration.md.
#
# v12 is v11 with every curve end given a source, and nothing else (docs/archive/plan-evolving-score.md
# §4, EV4; the user's decision of 2026-10-03). Each end is a domain limit, a gate, a stated
# meaning or a quantile of the 20 published band games read once from the v11 library and
# pinned; `anchor` no longer writes ends. Every weight, family, curve kind and gate is v11's,
# so a v11-to-v12 move is the scale changing source. Pre-registration, read and grading in
# docs/reference/v12-calibration.md.
DEFAULT_WEIGHTS = "v12"

# Floor applied to a family score before it enters the geometric mean. Without it a
# single zeroed family collapses the index to zero for every candidate that has one,
# and the design loop loses the ability to tell a nearly-good game from a hopeless one.
FLOOR = 0.02

CURVE_KINDS = ("band", "lower_better", "higher_better")

# Where a curve end comes from (`docs/archive/plan-evolving-score.md` §4.2). A weights file either
# names one for every end of every curve, as `v12` onward do, or for none, as the versions
# whose ends `anchor` placed from the band do; never for some.
#
# * ``domain``: the limit the quantity cannot pass. Changes never.
# * ``gate``: the gate on the same measurement, so the zero end sits where the gate already
#   says *not a game*. It names the gate, and must equal its threshold.
# * ``meaning``: a stated reading of the quantity, in one sentence.
# * ``reference``: a quantile of the published band games, read once and pinned as a number.
#   Nothing reads the corpus when a file loads; the number is the file's.
SOURCE_KINDS = ("domain", "gate", "meaning", "reference")


class WeightsError(Exception):
    """Raised when a weights file is malformed. Never silently defaulted."""


@dataclass(frozen=True)
class Curve:
    """How one raw measurement maps onto a 0-1 score."""

    kind: str
    points: dict[str, float]

    def score(self, value: float) -> float:
        if self.kind == "lower_better":
            return _ramp_down(value, self.points["good_at"], self.points["zero_at"])
        if self.kind == "higher_better":
            return _ramp_up(value, self.points["zero_at"], self.points["good_at"])
        rising = _ramp_up(value, self.points["zero_low"], self.points["good_low"])
        falling = _ramp_down(value, self.points["good_high"], self.points["zero_high"])
        return min(rising, falling)


@dataclass(frozen=True)
class Gate:
    """A hard viability condition. Failing one does not lower a score -- it disqualifies.

    Separate from the weighted sum on purpose: a game that never finishes, or that a
    searching player cannot beat a random one at, is not a mediocre game. Letting a
    strong showing elsewhere average such a failure away is exactly how a broken
    ruleset reaches the top of a leaderboard.
    """

    component: str
    minimum: float | None = None
    maximum: float | None = None
    reason: str = ""

    def check(self, value: float) -> str | None:
        if self.minimum is not None and value < self.minimum:
            return f"{self.component}={value:.3f} below minimum {self.minimum} ({self.reason})"
        if self.maximum is not None and value > self.maximum:
            return f"{self.component}={value:.3f} above maximum {self.maximum} ({self.reason})"
        return None


@dataclass(frozen=True)
class Weights:
    """One complete, versioned scoring configuration."""

    version: str
    families: dict[str, str]
    weights: dict[str, float]
    curves: dict[str, Curve]
    gates: tuple[Gate, ...]
    llm_weight: float
    # Components a corpus may legitimately fail to supply, as opposed to ones a corpus
    # is too old to have. `lead_conversion` is the case that forced the distinction: it
    # needs enough decisive games carrying a real material edge to say anything, and a
    # short run simply may not have them. That is not a pipeline error and it is not a
    # verdict about the ruleset -- it is an absence of evidence, and the family it
    # belongs to is scored on what could be measured.
    #
    # Deliberately narrow. A component that is *usually* absent has no business carrying
    # weight, and one that is absent for a *reason* -- a five-ply game offering nowhere
    # to force a blunder -- is already caught by the gate that reason trips.
    optional: frozenset[str] = frozenset()
    notes: str = ""
    # Each component's curve ends and the kind of source each one names (`SOURCE_KINDS`),
    # empty for a version whose ends the band placed.
    sources: dict[str, dict[str, str]] = field(default_factory=dict)

    def components(self) -> list[str]:
        return sorted(self.weights)

    def score_all(self, raw: dict[str, float]) -> dict[str, float]:
        """Score every configured component. Missing measurements are an error.

        A quietly absent component would shift the weighted mean onto whatever
        remains, changing the meaning of the index without changing its version.
        """
        missing = [
            name
            for name in self.weights
            if name not in raw and name not in self.optional
        ]
        if missing:
            raise WeightsError(
                f"weights {self.version} expect components not present in the "
                f"measurements: {', '.join(sorted(missing))}"
            )
        return {
            name: self.curves[name].score(raw[name])
            for name in self.weights
            if name in raw
        }

    def family_scores(self, scores: dict[str, float]) -> dict[str, float]:
        """Collapse components into families by weighted arithmetic mean.

        Within a family the components really are substitutable -- there is more than
        one way for a game to be well balanced -- so averaging them is the right move.
        """
        totals: dict[str, float] = {}
        weights: dict[str, float] = {}
        for name, score in scores.items():
            family = self.families.get(name, "other")
            totals[family] = totals.get(family, 0.0) + self.weights[name] * score
            weights[family] = weights.get(family, 0.0) + self.weights[name]
        return {family: totals[family] / weights[family] for family in totals}

    def family_weights(self) -> dict[str, float]:
        weights: dict[str, float] = {}
        for name, weight in self.weights.items():
            family = self.families.get(name, "other")
            weights[family] = weights.get(family, 0.0) + weight
        return weights

    def combine(self, scores: dict[str, float]) -> tuple[float, dict[str, float]]:
        """Weighted **geometric** mean across families, plus per-component contributions.

        Arithmetic across families was the first thing tried and it is wrong. It lets a
        game be brilliant at the cheap things and hopeless at the expensive one and
        still land mid-table: the deliberately dull control scored 0.64 that way,
        because simplicity, variety and drama all max out trivially on a game with
        nothing in it, outvoting a near-zero on depth.

        Families are not substitutable. Elegance cannot make up for having no
        strategy, so the combination has to be one where a bad family drags the whole
        score down rather than being averaged away. A geometric mean does exactly
        that. Scores are floored just above zero so a single failed family leaves a
        usable gradient for the search instead of flattening everything to zero.
        """
        family_scores = self.family_scores(scores)
        family_weights = self.family_weights()
        total_weight = sum(family_weights.values())

        log_total = sum(
            family_weights[family] * math.log(max(score, FLOOR))
            for family, score in family_scores.items()
        )
        total = math.exp(log_total / total_weight)

        # Over what was *scored* rather than over what was declared: an optional
        # component the corpus could not supply has no score to contribute, and
        # dividing by the full declared weight would quietly report every other
        # contribution as smaller than it was.
        component_weight = sum(self.weights[name] for name in scores)
        contributions = {
            name: self.weights[name] * scores[name] / component_weight for name in scores
        }
        return total, contributions

    def arithmetic_total(self, scores: dict[str, float]) -> float:
        """The flat weighted mean, kept as a diagnostic.

        Useful for seeing how much the geometric combination is penalising: a large
        gap between the two means one family is dragging everything.
        """
        total_weight = sum(self.weights[name] for name in scores)
        if total_weight <= 0:
            return 0.0
        return sum(self.weights[name] * scores[name] for name in scores) / total_weight

    def check_gates(self, raw: dict[str, float]) -> list[str]:
        violations = []
        for gate in self.gates:
            if gate.component not in raw:
                violations.append(f"{gate.component} was not measured")
                continue
            failure = gate.check(raw[gate.component])
            if failure:
                violations.append(failure)
        return violations


def load_weights(version: str = DEFAULT_WEIGHTS) -> Weights:
    """Load a weights file by version name."""
    path = WEIGHTS_DIR / f"{version}.json"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in WEIGHTS_DIR.glob("*.json"))) or "none"
        raise WeightsError(f"no weights version {version!r}; available: {available}")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return parse_weights(data, fallback_version=version)


def parse_weights(data: dict[str, Any], fallback_version: str = "?") -> Weights:
    version = data.get("version", fallback_version)
    raw_components = data.get("components")
    if not isinstance(raw_components, dict) or not raw_components:
        raise WeightsError(f"weights {version} declare no components")

    weights: dict[str, float] = {}
    curves: dict[str, Curve] = {}
    families: dict[str, str] = {}
    optional: set[str] = set()
    sources: dict[str, dict[str, str]] = {}
    gate_values = _gate_values(data.get("gates", []))

    for name, config in raw_components.items():
        if not isinstance(config, dict):
            raise WeightsError(f"component {name!r} must be an object")
        kind = config.get("curve")
        if kind not in CURVE_KINDS:
            raise WeightsError(f"component {name!r} has unknown curve {kind!r}")
        points = {k: float(v) for k, v in config.get("points", {}).items()}
        _validate_points(name, kind, points)

        weight = float(config.get("weight", 0.0))
        if weight <= 0:
            raise WeightsError(f"component {name!r} must carry a positive weight")
        weights[name] = weight
        curves[name] = Curve(kind=kind, points=points)
        families[name] = str(config.get("family", "other"))
        if config.get("optional"):
            optional.add(name)
        if "sources" in config:
            sources[name] = _parse_sources(name, config["sources"], points, gate_values)

    if sources and set(sources) != set(raw_components):
        unsourced = ", ".join(sorted(set(raw_components) - set(sources)))
        raise WeightsError(
            f"weights {version} name a source for some curves and not for {unsourced}"
        )

    gates = tuple(
        Gate(
            component=g["component"],
            minimum=g.get("minimum"),
            maximum=g.get("maximum"),
            reason=g.get("reason", ""),
        )
        for g in data.get("gates", [])
    )

    llm_weight = float(data.get("llm_weight", 0.0))
    if not 0.0 <= llm_weight < 1.0:
        raise WeightsError("llm_weight must be at least 0 and below 1")

    return Weights(
        version=version,
        families=families,
        weights=weights,
        curves=curves,
        gates=gates,
        llm_weight=llm_weight,
        optional=frozenset(optional),
        notes=str(data.get("notes", "")),
        sources=sources,
    )


def _gate_values(gates: Any) -> dict[str, set[float]]:
    """Every threshold each gated component carries, minimum or maximum."""
    values: dict[str, set[float]] = {}
    for gate in gates if isinstance(gates, list) else []:
        for bound in ("minimum", "maximum"):
            if gate.get(bound) is not None:
                values.setdefault(str(gate["component"]), set()).add(float(gate[bound]))
    return values


def _parse_sources(
    name: str, raw: Any, points: dict[str, float], gate_values: dict[str, set[float]]
) -> dict[str, str]:
    """Check one curve's sources against its ends, and return end -> kind.

    Every end has exactly one source with a sentence or rule (`why`), and a ``gate`` end
    names a gate in the same file whose threshold it equals: an end that says it sits where
    the gate does and has drifted from it is the one error this can catch mechanically.
    """
    if not isinstance(raw, dict) or set(raw) != set(points):
        raise WeightsError(
            f"component {name!r} must name one source for each of its ends "
            f"({', '.join(sorted(points))})"
        )
    kinds: dict[str, str] = {}
    for end, entry in raw.items():
        kind = entry.get("source") if isinstance(entry, dict) else None
        if kind not in SOURCE_KINDS:
            raise WeightsError(f"component {name!r} end {end} has unknown source {kind!r}")
        if not str(entry.get("why", "")).strip():
            raise WeightsError(f"component {name!r} end {end} gives no sentence for its source")
        if kind == "gate":
            gate = str(entry.get("gate", ""))
            if points[end] not in gate_values.get(gate, set()):
                raise WeightsError(
                    f"component {name!r} end {end}={points[end]} names gate {gate!r}, "
                    "which has no threshold at that value"
                )
        kinds[end] = str(kind)
    return kinds


def _validate_points(name: str, kind: str, points: dict[str, float]) -> None:
    required = {
        "band": ("zero_low", "good_low", "good_high", "zero_high"),
        "lower_better": ("good_at", "zero_at"),
        "higher_better": ("zero_at", "good_at"),
    }[kind]
    missing = [key for key in required if key not in points]
    if missing:
        raise WeightsError(f"component {name!r} curve is missing {', '.join(missing)}")

    # Every curve's required keys are listed in the order their values must ascend.
    ordered = [points[key] for key in required]
    if any(a > b for a, b in zip(ordered, ordered[1:], strict=False)):
        raise WeightsError(f"component {name!r} curve points are out of order: {points}")


def _ramp_up(value: float, zero_at: float, good_at: float) -> float:
    if good_at <= zero_at:
        return 1.0 if value >= good_at else 0.0
    return max(0.0, min(1.0, (value - zero_at) / (good_at - zero_at)))


def _ramp_down(value: float, good_at: float, zero_at: float) -> float:
    if zero_at <= good_at:
        return 1.0 if value <= good_at else 0.0
    return max(0.0, min(1.0, (zero_at - value) / (zero_at - good_at)))
