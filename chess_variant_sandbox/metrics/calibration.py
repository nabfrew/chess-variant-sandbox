"""Scoring the index itself.

An index is a hypothesis about what makes a game good, and like any hypothesis it can
be wrong in ways its own output will not reveal. Reading a leaderboard tells you what
the index believes; it does not tell you whether the belief is worth holding. What is
needed is a set of numbers *about* the index, computed against a corpus whose labels
are known in advance, and that is what this module produces.

Four measurements, and each one names a distinct way for an index to fail.

``separation``
    The margin between the worst human variant and the best known-bad mutant. Positive
    and wide is the whole job: everything a designer would recognise as broken must
    score below everything a designer would recognise as a game. Negative means the
    index cannot tell the difference, and the specific mutant that crossed the line
    says which defect it is blind to.

``sign_accuracy``
    The share of ``worse`` mutants that actually land below the human floor. Separation
    is a worst case; this is the average one, and the two fail differently. An index can
    have a healthy margin and still misplace a third of the negatives.

``null_stability``
    The largest gap between a ``same`` mutant and its own base. These pairs differ in
    spec and in hash but not in play, so any gap at all is the index reading noise.
    Compared against the sampling interval: a null that moves further than the bootstrap
    sd is a real defect, one that moves less is the measurement breathing. It is refused
    rather than reported when the corpus has no null, or a null names no base in it: over
    zero pairs the worst gap reads 0.000 whatever the index does (Q71).

    Read the current 0.000 precisely. Renaming and reordering leave every raw component
    bit-identical -- reordering does change 33 of 144 move sequences, but only in the
    random-vs-random games the measurements exclude. So the number certifies that
    spec-level relabelling cannot reach the index, which is worth certifying, and does
    not yet demonstrate stability under different *competent* play. No mechanical
    transform can, because anything strong players respond to is a change to the game.

``human_spread``
    The width of the band the real games occupy -- **to be kept small, not large**. This
    is the number most easily mistaken for a fault. Twenty published variants that
    people have played for centuries are all good games; an index that confidently ranks
    them is claiming a resolution it does not have, and the bootstrap interval says it
    does not have it. The job is to separate games from non-games and to leave room
    above the band for something genuinely better, not to sort the classics.

The reason these are computed rather than argued about: v2's curve ends were set by
judgement against a corpus of ten positives and two negatives, and three of its four
worst properties -- dead weight, double counting, an inverted elegance term -- survived
that judgement unnoticed. Numbers with known labels are what makes the difference
checkable.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

# Below this share of rising steps, the rating ladder is not ordering the agents by
# strength, and nothing derived from that ordering describes the game.
MEASURABLE_LADDER = 0.5

# Committed rulesets that are in the corpus on purpose and are **not** evidence about
# where the human band sits. Held here rather than in two places, because a name added
# to one copy and not the other silently widens the band.
#
# Two kinds, and they are not the same kind.
#
# *Controls* were built to be degenerate: `dull` has no strategy in it and `bastion`
# hides a three-ply forced mate. They anchor the low end.
#
# *Probes* are real designs by a real designer who rejected every one of them and said
# why -- `docs/reference/rps-versions.md`. That makes them a labelled set with
# independent verdicts, which is the one thing the mutant corpus can never be. They are
# still all negatives, so they can show that a named failure is *found* and can never
# show that a good game is *accepted*, and six deliberately-flawed games inside the band
# would drag every calibration anchor down with them.
CONTROL_RULESETS = frozenset({"dull", "bastion"})
PROBE_RULESETS = frozenset({
    "rps annihilation", "rps race", "hierarchy chess",
    "rps territory", "rps repaint", "rps both endings",
})
EXCLUDED_FROM_BAND = CONTROL_RULESETS | PROBE_RULESETS

# The share of the labelled negatives held out of curve anchoring, so that separation
# can be reported on mutants no curve was fitted against.
#
# The rule the plan states is "hold a slice out chosen *before* any component changed",
# and the only way to make that claim checkable later is to derive the slice from the
# names rather than record a list somebody could have edited after seeing the scores.
# `held_out` hashes the name with a fixed salt, which is deterministic, auditable, and
# demonstrably not a choice: change the salt and the whole slice changes, so there is no
# way to nudge one mutant across the line without moving every other one too.
HELDOUT_SHARE = 1 / 3
HELDOUT_SALT = "v8"


def held_out(name: str) -> bool:
    """Whether this labelled ruleset is in the held-out slice.

    Deterministic in the name, so the same mutant is held out of every run and the
    membership cannot drift with the corpus. Anchoring reads the complement; the
    calibration report reads both and quotes the held-out numbers, because separation
    measured on the mutants the curves were fitted against is not separation.
    """
    import hashlib

    digest = hashlib.sha256(f"{HELDOUT_SALT}:{name}".encode()).digest()
    return digest[0] / 256 < HELDOUT_SHARE


@dataclass(frozen=True)
class ScoredEntry:
    """One ruleset's place in a calibration corpus."""

    name: str
    total: float
    viable: bool
    label: str  # human | control | worse | same | probe
    base: str = ""
    sd: float = 0.0
    # Share of budget steps on which the bigger search rated higher. A human variant
    # that fails this is not a bad game -- it is a game this roster cannot play, and
    # letting it set the human floor would calibrate the scale against the agents'
    # blind spot rather than against the games.
    ladder: float = 1.0
    # Whether the *index* withheld components for this ruleset. The authority on
    # readability is `metrics/index.py`, and this used to be re-derived here from the
    # ladder alone -- which diverged the moment the index gained a second test. Under
    # the six-budget roster antichess reads 0.60 on the ladder and -0.56 Elo per
    # doubling: the index withholds, the ladder share does not, and the disagreement put
    # a withheld 0.000 into the human band and dragged the floor down with it.
    withheld: bool = False

    @property
    def measurable(self) -> bool:
        return not self.withheld and self.ladder >= MEASURABLE_LADDER


@dataclass
class CalibrationReport:
    """How well an index separates what it is supposed to separate."""

    weights_version: str
    separation: float
    sign_accuracy: float
    null_stability: float
    human_spread: float
    human_floor: float
    human_ceiling: float
    misplaced: list[str] = field(default_factory=list)
    unstable_nulls: list[str] = field(default_factory=list)
    unmeasurable: list[str] = field(default_factory=list)
    # Real games the gates rejected. Out of the band for the same reason a rejected
    # mutant counts as separated: the verdict is a verdict, and applying it in one
    # direction only would be reading it twice.
    gated_humans: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"calibration of weights {self.weights_version}",
            f"  human band      {self.human_floor:.3f} - {self.human_ceiling:.3f} "
            f"(spread {self.human_spread:.3f}; smaller is better)",
            f"  separation      {self.separation:+.3f}  "
            f"(human floor minus best known-bad mutant; positive is the point)",
            f"  sign accuracy   {self.sign_accuracy:.0%} of known-bad mutants below the human floor",
            f"  null stability  {self.null_stability:.3f} worst gap between a null and its base "
            f"(over {len(self.detail.get('null_gaps', {}))} nulls)",
        ]
        if self.misplaced:
            lines.append("  known-bad mutants inside the human band:")
            lines.extend(f"    - {name}" for name in self.misplaced)
        if self.unstable_nulls:
            lines.append("  nulls that moved further than the sampling interval:")
            lines.extend(f"    - {name}" for name in self.unstable_nulls)
        if self.gated_humans:
            lines.append("  real games the gates rejected, so out of the band:")
            lines.extend(f"    - {name}" for name in self.gated_humans)
        if self.unmeasurable:
            lines.append(
                "  held out of the band -- this roster cannot play them, so their "
                "scores describe the agents:"
            )
            lines.extend(f"    - {name}" for name in self.unmeasurable)
        return "\n".join(lines)


def calibrate(entries: list[ScoredEntry], weights_version: str) -> CalibrationReport:
    """Compute the four index-quality numbers from a scored calibration corpus.

    A mutant that failed its gates is *correctly* rejected and counts as separated: the
    gate and the score are two halves of one verdict, and an index that disqualifies a
    broken game has not failed to notice it. Scoring such a mutant on its total alone
    would punish the gates for working.
    """
    # The band is where real games the index *accepts* sit, and both halves of that
    # matter. A ruleset the roster cannot read is excluded because its numbers describe
    # the agents. A ruleset the gates reject is excluded because it has been
    # disqualified -- which is exactly how a rejected *mutant* is treated two blocks
    # below, where a gated-out negative counts as separated by construction. Letting a
    # gated-out real game set the floor while a gated-out mutant is scored at -1 would
    # be applying the same verdict in opposite directions.
    #
    # It was a gated-out real game that made this visible: widening the roster from four
    # budgets to six pushed racing kings' `ladder_monotonicity` from 0.33 to 0.60, so it
    # stopped being filtered as unreadable and set the human floor at 0.483 -- below five
    # of the negatives, which made separation meaningless.
    humans = [e for e in entries if e.label == "human" and e.measurable and e.viable]
    unreadable = [e for e in entries if e.label == "human" and not e.measurable]
    rejected = [
        e for e in entries if e.label == "human" and e.measurable and not e.viable
    ]
    worse = [e for e in entries if e.label == "worse"]
    nulls = [e for e in entries if e.label == "same"]
    by_name = {e.name: e for e in entries}

    if not humans:
        raise ValueError("a calibration corpus needs at least one human variant")
    refusal = _nulls_refusal(nulls, by_name)
    if refusal:
        raise ValueError(refusal)

    floor = min(e.total for e in humans)
    ceiling = max(e.total for e in humans)

    # A gated-out mutant is separated by construction; give it the floor so it neither
    # flatters nor penalises the margin.
    def effective(entry: ScoredEntry) -> float:
        return -1.0 if not entry.viable else entry.total

    worst_case = max((effective(e) for e in worse), default=-1.0)
    correct = sum(1 for e in worse if effective(e) < floor)

    gaps: dict[str, float] = {}
    for null in nulls:
        base = by_name.get(null.base)
        if base is not None:
            gaps[null.name] = abs(null.total - base.total)

    # A null that moves less than the combined sampling noise of the two runs has not
    # moved: it is the same game measured twice. Compared against the sum rather than
    # either alone, because both scores carry their own interval.
    unstable = [
        name
        for name, gap in gaps.items()
        if gap > max(by_name[name].sd + by_name.get(by_name[name].base, by_name[name]).sd, 0.02) * 2
    ]

    return CalibrationReport(
        weights_version=weights_version,
        separation=round(floor - worst_case, 4),
        sign_accuracy=round(correct / len(worse), 4) if worse else 0.0,
        null_stability=round(max(gaps.values()), 4) if gaps else 0.0,
        human_spread=round(ceiling - floor, 4),
        human_floor=round(floor, 4),
        human_ceiling=round(ceiling, 4),
        misplaced=sorted(e.name for e in worse if effective(e) >= floor),
        unstable_nulls=sorted(unstable),
        unmeasurable=sorted(e.name for e in unreadable),
        gated_humans=sorted(e.name for e in rejected),
        detail={
            "humans": len(humans),
            "worse": len(worse),
            "nulls": len(nulls),
            "gated_out": sorted(e.name for e in worse if not e.viable),
            "null_gaps": {k: round(v, 4) for k, v in sorted(gaps.items())},
            "human_median": round(statistics.median([e.total for e in humans]), 4),
            "mean_sd": round(
                statistics.fmean([e.sd for e in humans if e.sd > 0] or [0.0]), 4
            ),
        },
    )


def _nulls_refusal(nulls: list[ScoredEntry], by_name: dict[str, ScoredEntry]) -> str:
    """Why null stability cannot be reported over these nulls, or ``""`` when it can.

    Q71. `runs/v9`'s four `same` entries all carry `"base": ""`, so no null found its base,
    the gaps were empty, and `null_stability` read 0.000 -- a perfect score over zero
    nulls, right only by luck. A null is read against its base or not at all, so every
    `same` entry has to name a base the corpus holds, and there has to be at least one.
    A null that lost its base is refused rather than dropped: the worst gap over the
    nulls that remain is the same number over fewer nulls, and says nothing about the
    one that went missing.
    """
    if not nulls:
        return (
            "null_stability needs at least one `same` entry, and this corpus has none: "
            "a worst gap over zero nulls reads 0.000 whatever the index does (Q71)"
        )
    unpaired = [null for null in nulls if null.base not in by_name]
    if not unpaired:
        return ""
    named = ", ".join(f"{null.name} (base {null.base!r})" for null in unpaired)
    return (
        f"null_stability reads each `same` entry against its base, and {len(unpaired)} "
        f"of {len(nulls)} name no base in this corpus: {named}. Rebuild the corpus with "
        "`chess-variant-sandbox calibration-corpus`, which writes every base (Q71)"
    )


def suggest_curve_ends(
    values_by_label: dict[str, list[float]], quantile: float = 0.9
) -> dict[str, float]:
    """Where a component's curve ends belong, read off the labelled corpus.

    The anti-overfitting rule made mechanical. ``zero_at`` is anchored on the
    **negatives** -- the boundary of "not a game" is wherever the known-bad things
    actually sit -- and ``good_at`` is placed beyond the **whole** human band rather
    than beyond any single game in it.

    That asymmetry is the point. Anchoring the good end on one reference game, however
    canonical, builds a similarity-to-that-game score and calls it quality; every real
    variant then reads as a deviation. Anchoring it past all of them means every real
    game plateaus near the top, the index declines to rank things it cannot resolve, and
    the resolution that remains lives above the band, where a genuinely new candidate
    would have to earn it.
    """
    humans = sorted(values_by_label.get("human", []))
    negatives = sorted(values_by_label.get("worse", []))
    if not humans:
        return {}

    index = min(int(quantile * len(humans)), len(humans) - 1)
    return {
        "human_low": round(humans[0], 4),
        "human_high": round(humans[-1], 4),
        "human_quantile": round(humans[index], 4),
        "negative_high": round(negatives[-1], 4) if negatives else 0.0,
        "negative_median": round(statistics.median(negatives), 4) if negatives else 0.0,
    }


# --------------------------------------------------------------- curve anchoring

# Curves read off the rating ladder. A single inverted budget step is enough to make one
# of these meaningless for that ruleset, so they are anchored only on clean ladders.
SKILL_DERIVED = frozenset({"skill_gradient", "gradient_persistence", "draw_slope"})

# How far past the top of the human band the good end goes, as a share of band width.
HEADROOM = 0.35

# A curve may be no sharper than the measurement is precise, nor sharper than the band
# it is scoring. Both floors earned their place: without the first, a component measured
# to +/-0.11 got a band 0.35 wide and scored mostly its own error; without the second,
# ``piece_utilisation`` got a curve running 0.898 to 0.997 and the corpus of good games
# was spread across it by sampling noise, pushing two real variants below both controls.
MIN_SPAN_IN_NOISE = 3.0
MIN_SPAN_IN_BANDS = 2.0

# Bounds a curve end may not leave, because the quantity cannot go there.
DOMAIN: dict[str, tuple[float, float | None]] = {
    "rule_yield": (0.0, None),
    "first_player_advantage": (0.0, 1.0),
    "stall_rate": (0.0, 1.0),
    "mean_branching": (0.0, None),
    "quick_kill_rate": (0.0, 1.0),
    "ply_spread": (0.0, None),
    "opening_diversity": (0.0, 1.0),
    "distinct_game_rate": (0.0, 1.0),
    "blunder_recoverability": (0.0, 1.0),
    "decision_criticality": (0.0, 1.0),
    "piece_utilisation": (0.0, 1.0),
    "skill_gradient": (0.0, None),
    "gradient_persistence": (0.0, 2.5),
    "draw_slope": (0.0, 2.0),
    # Schema-3 era arrivals. All three are shares and all three are bands, which is the
    # combination that makes a missing entry here visible: the span floors push a band's
    # zero ends outward, and with no domain to clamp against they land outside the range
    # the measurement can produce. The first v5 anchoring did exactly that --
    # `progress_irreversibility` came out with `zero_low` at -0.64 -- which means the
    # zero end is unreachable and the component can never score badly.
    "progress_irreversibility": (0.0, 1.0),
    "instance_participation": (0.0, 1.0),
    "lead_conversion": (0.0, 1.0),
    # `v11`'s variety component (`docs/archive/plan-variety.md` §5.4 C2). A mean Jaccard distance
    # between position sets, so a share; `higher_better` on a band as wide as this one
    # pushes `good_at` past 1.0 and `zero_at` below 0.0, and both clamp here.
    "position_divergence": (0.0, 1.0),
}


@dataclass(frozen=True)
class Measured:
    """One ruleset's raw measurements, with the label and ladder needed to read them."""

    name: str
    label: str
    raw: dict[str, float]

    @property
    def ladder(self) -> float:
        return float(self.raw.get("ladder_monotonicity", 1.0))


def _clamp(name: str, value: float) -> float:
    low, high = DOMAIN.get(name, (None, None))
    if low is not None:
        value = max(value, low)
    if high is not None:
        value = min(value, high)
    return round(value, 4)


def anchor_curves(
    corpus: list[Measured],
    components: dict[str, dict[str, Any]],
    noise: dict[str, float] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    """Place every curve's ends from the labelled corpus, mechanically.

    **How `v3` to `v11` were placed, and no longer how a version is.** Since `v12` every end
    names its source and the band places none of them (`docs/archive/plan-evolving-score.md` §4);
    `anchor` reports with `check_ends` and writes nothing. This stays so the band-placed
    versions' ends can be re-derived (`docs/reference/negatives-widened.py` reads it).

    Returns the new points per component, and which rulesets were held out of each.

    The rules, and why each one is not negotiable:

    * ``zero_at`` comes from the negatives. Where "not a game" begins is an empirical
      question and the corpus answers it.
    * ``good_at`` goes past the *whole* human band. Anchoring it on one reference game,
      however canonical, builds a similarity-to-that-game score and calls it quality.
      The consequence -- real variants plateauing near the top -- is the intended
      behaviour, not a loss of resolution: they are all good games, and the resolution
      that remains lives above the band where a genuinely better candidate would earn it.
    * A ruleset whose rating ladder is not monotone in budget is left out, and the
      skill-derived curves demand a perfectly clean ladder. Not because such a ruleset
      scores badly -- because the roster cannot play it, so its numbers describe the
      agents.
    """
    noise = noise or {}
    points: dict[str, dict[str, float]] = {}
    held_out: dict[str, list[str]] = {}

    for name, config in components.items():
        need = 0.99 if name in SKILL_DERIVED else MEASURABLE_LADDER
        human: list[float] = []
        dropped: list[str] = []
        for entry in corpus:
            if entry.label != "human" or name not in entry.raw:
                continue
            if entry.ladder < need:
                dropped.append(entry.name)
                continue
            human.append(entry.raw[name])
        negatives = sorted(
            e.raw[name] for e in corpus if e.label in ("worse", "control") and name in e.raw
        )
        if len(human) < 6:
            continue

        human.sort()
        if dropped:
            held_out[name] = sorted(dropped)

        low, high = human[0], human[-1]
        width = max(high - low, 1e-6)
        pad = width * HEADROOM
        midpoint = statistics.median(negatives) if negatives else None
        floor = max(MIN_SPAN_IN_NOISE * noise.get(name, 0.0), MIN_SPAN_IN_BANDS * width)

        kind = config["curve"]
        if kind == "higher_better":
            good = high + pad
            target = midpoint if midpoint is not None else low - pad
            points[name] = {
                "good_at": _clamp(name, good),
                "zero_at": _clamp(name, min(target, good - floor)),
            }
        elif kind == "lower_better":
            target = midpoint if midpoint is not None else high + pad
            points[name] = {
                "good_at": _clamp(name, low),
                "zero_at": _clamp(name, max(target, low + floor)),
            }
        else:
            band = {
                "good_low": _clamp(name, low),
                "good_high": _clamp(name, high),
                "zero_low": _clamp(name, low - floor / 2),
                "zero_high": _clamp(name, high + floor / 2),
            }
            # A band whose zero end lands on its good end scores everything zero.
            if band["zero_low"] >= band["good_low"]:
                band["zero_low"] = _clamp(name, band["good_low"] - max(width, 0.05))
            if band["zero_high"] <= band["good_high"]:
                band["zero_high"] = _clamp(name, band["good_high"] + max(width, 0.05))
            points[name] = band

    return points, held_out


@dataclass(frozen=True)
class EndCheck:
    """Where the band sits against one curve's pinned ends (`anchor`'s report since `v12`).

    ``past`` names, for each end, the band games beyond it on the side the curve is not
    good: below a low end, above a high end, past a ``higher_better`` good end (plateaued)
    or a ``lower_better`` one (likewise). ``low`` and ``high`` are the band's quantiles at
    the rule `v12`'s reference ends were read by, the *k*-th lowest and highest at
    *k* = ⌈0.1·*n*⌉, so a check reads the band the way the scale was read.
    """

    component: str
    kind: str
    ends: dict[str, float]
    sources: dict[str, str]
    games: int
    plateau: int
    zero: int
    low: float | None
    high: float | None
    past: dict[str, list[str]]
    unreadable: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def band_quantiles(values: list[float]) -> tuple[float, float] | None:
    """The *k*-th lowest and *k*-th highest of ``values``, *k* = ⌈0.1·*n*⌉; ``None`` if empty."""
    import math

    if not values:
        return None
    ordered = sorted(values)
    k = max(1, math.ceil(0.1 * len(ordered)))
    return ordered[k - 1], ordered[len(ordered) - k]


def check_ends(corpus: list[Measured], weights: Any) -> list[EndCheck]:
    """Read the band against a weights version's ends. Places nothing and writes nothing.

    Since `v12` the band checks the scale and no longer sets it
    (`docs/archive/plan-evolving-score.md` §4.2): this is what the next revision reads to see which
    games plateau, which fall to zero, and which ends a band game now passes. Only the
    `human` entries count, and for each component only those whose ladder the roster can
    read, by the rule ``anchor_curves`` used (0.99 for the skill-derived curves).
    """
    checks: list[EndCheck] = []
    for name in weights.components():
        curve = weights.curves[name]
        need = 0.99 if name in SKILL_DERIVED else MEASURABLE_LADDER
        band: list[tuple[float, str]] = []
        unreadable: list[str] = []
        for entry in corpus:
            if entry.label != "human" or name not in entry.raw:
                continue
            if entry.ladder < need:
                unreadable.append(entry.name)
                continue
            band.append((float(entry.raw[name]), entry.name))
        ends = dict(curve.points)
        past: dict[str, list[str]] = {}
        for end, value in ends.items():
            beyond = _beyond(curve.kind, end)
            past[end] = sorted(n for v, n in band if (v < value if beyond < 0 else v > value))
        scores = [curve.score(v) for v, _ in band]
        quantiles = band_quantiles([v for v, _ in band])
        checks.append(EndCheck(
            component=name, kind=curve.kind, ends=ends,
            sources=dict(weights.sources.get(name, {})), games=len(band),
            plateau=sum(1 for s in scores if s >= 1.0), zero=sum(1 for s in scores if s <= 0.0),
            low=None if quantiles is None else round(quantiles[0], 4),
            high=None if quantiles is None else round(quantiles[1], 4),
            past=past, unreadable=sorted(unreadable),
        ))
    return checks


def _beyond(kind: str, end: str) -> int:
    """-1 if a game is past this end when below it, +1 when above it."""
    if kind == "band":
        return -1 if end in ("zero_low", "good_low") else 1
    if kind == "higher_better":
        return -1 if end == "zero_at" else 1
    return 1 if end == "zero_at" else -1
