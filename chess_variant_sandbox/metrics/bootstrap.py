"""How much of an index score is measurement, and how much is luck.

The index is computed from a finite tournament, so it is an estimate, and an estimate
without an interval invites two specific mistakes. The first is reading a leaderboard
ordering that the evidence does not support. The second is worse and belongs to the
design loop: accepting a candidate because it scored 0.02 above its parent, when 0.02
is a third of the noise. A loop that does that will climb sampling error for as many
generations as it is given, and nothing in the pipeline would say so.

The interval comes from resampling the games that were actually played, **stratified by
agent pairing**. The schedule is fixed by ``TournamentConfig`` -- every ordered pair
gets the same number of games -- so a resample that ignored the pairing structure would
produce corpora the tournament could never have produced, with some matchups
over-represented and others missing. Resampling within each pairing preserves the
design and asks the only question worth asking: had the dice fallen differently, what
else might this ruleset have scored?

Two properties of the result are worth knowing before reading one:

**A resample holds copies, and only a count of distinct games can see them.** Four games
drawn with replacement hold 2.73 distinct games on average. A component that averages or
sums over games only reweights under copies. A component that counts distinct games
would read each copy as a repeat, so ``distinct_game_rate`` and ``opening_diversity``
read one entry per game identity (``components._one_per_game``). Before that rule the
copies pulled ``distinct_game_rate`` about 0.29 below its point value, which moved
chess's interval mean from its 0.794 point total to 0.769. That gap was this bias, not
Jensen's inequality.

**The bootstrap mean can still sit off the point estimate, in either direction.** A
clamped curve is flat past its ends, so a component pinned at a curve end can only move
inward under resampling. And a count read on fewer distinct games reads slightly high,
because fewer games have fewer chances to collide: ``opening_diversity`` averages +0.02
over ``runs/v9``, and chess's interval mean is 0.801 against its 0.794 point. The point
estimate is not wrong, but it is not the centre of the interval, and the two should not
be reported as though it were.

**It captures sampling noise, not seed noise.** Resampling asks what else *these* games
could have added up to. It cannot ask what a different draw of games entirely would
have looked like, because those games were never played. A fresh run of the same game
differs from this one by two runs' sampling error, and it lands within **four** of these
sds about 19 times in 20 (`docs/reference/run-to-run-spread.md` §1). For anything
published as a baseline, run several seeds as well.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from chess_variant_sandbox.metrics.components import GameReadings, compute_components
from chess_variant_sandbox.metrics.scoring import Weights
from chess_variant_sandbox.tournament.analysis import CorpusAnalysis
from chess_variant_sandbox.tournament.ratings import rate
from chess_variant_sandbox.tournament.record import GameRecord

# Enough that the standard deviation is stable to about two decimal places, few enough
# that scoring a run stays a sub-second operation. The interval is a guard rail, not a
# publication-grade confidence statement.
DEFAULT_RESAMPLES = 200

# A pairing whose games all ended identically contributes nothing to the resample: every
# draw from it is the same corpus. That is not a bug, but when it holds for *most*
# pairings the resulting interval is an underestimate, because the Bradley-Terry fit is
# then pinned by its prior rather than by evidence.
#
# Above this share, an interval is reported *and disregarded*: `BootstrapResult.beats`
# refuses to declare a separation on the strength of it. The number was measured and
# then went unused for a version, which is how courier came to carry a `skill_gradient`
# sd of exactly 0.0 with nothing acting on it.
DETERMINISTIC_PAIRING_WARN = 0.8


def stratum_correction(sizes: Iterable[int]) -> float:
    """What a stratified resample's sd has to be multiplied by to be the right size.

    A stratum of n games is drawn from those n games as though they were the whole
    population, so every resample carries the plug-in variance, which is (n-1)/n of the
    sample variance. Each stratum therefore owes sqrt(n/(n-1)) on the sd -- 1.155 at the
    canonical four games a pairing. Four replays of the whole library measured the
    published sd 14% short and found nothing beyond this factor
    (`docs/reference/run-to-run-spread.md` §1).

    Pooling the strata into one factor weights each by its size, which is its share of
    the variance when every game carries the same influence. That is exact when the
    strata are all one size, which the canonical schedule makes them.

    A stratum of one cannot be corrected. It resamples to itself, contributes no
    variance to correct, and n/(n-1) is undefined, so it is left out of both sums and
    the interval stays short by whatever that one game carries. All strata of one give
    1.0, because then there is nothing to correct.
    """
    usable = [n for n in sizes if n >= 2]
    if not usable:
        return 1.0
    return math.sqrt(sum(n * n / (n - 1) for n in usable) / sum(usable))


@dataclass(frozen=True)
class BootstrapResult:
    """The sampling distribution of one ruleset's index score."""

    resamples: int
    mean: float
    sd: float
    ci_low: float
    ci_high: float
    deterministic_pairings: float
    component_sd: dict[str, float]
    component_mean: dict[str, float] = field(default_factory=dict)

    # The factor already applied to `sd`, to `component_sd` and to the interval's two
    # ends, so a stored interval says what it was corrected by rather than leaving a
    # reader to infer it from the schedule.
    stratum_correction: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low

    def beats(self, other: BootstrapResult) -> bool:
        """Whether this score is *higher* than another by more than the noise in both.

        Refused outright when either side's interval rests on a corpus of deterministic
        sweeps, and that refusal is the fix for a specific artifact rather than caution
        in general. A nonparametric bootstrap resamples the games that were played; a
        pairing whose four games all ended the same way resamples to itself every time
        and contributes exactly zero variance. Courier's `skill_gradient` had a
        bootstrap sd of **exactly 0.0** for this reason, with 84% of its pairings
        deterministic: a perfectly separated ladder yielding a confident number that
        contains no information.

        The temptation is to fix it at the source by fabricating the games the sample
        does not contain -- a Jeffreys prior on each pairing's rate, say. That would be
        worse: the fabricated result has to arrive attached to a *game*, with plies and
        material and a move list, and every other component would then be measured on
        games nobody played.

        So it is fixed at the consumer, where the honest statement is available: an
        interval built on sweeps is not evidence, and "we cannot tell" is the right
        answer rather than "certainly better". It matters because of which direction the
        error runs -- a design loop rewarded for narrow intervals would learn to produce
        rulesets whose ladders sweep, and that is a property of the *roster* being easy
        to separate rather than of the game being good.
        """
        if (
            max(self.deterministic_pairings, other.deterministic_pairings)
            >= DETERMINISTIC_PAIRING_WARN
        ):
            return False
        return clears(self.ci_low, other.ci_high)

    def separates(self, other: BootstrapResult) -> bool:
        """Whether two scores differ by more than their intervals allow, either way.

        The test the design loop needs: is this candidate *actually* different from that
        one, or is the difference inside the noise of both?
        """
        return self.beats(other) or other.beats(self)


def clears(low_end: float, high_end: float) -> bool:
    """The whole separation test, in one place.

    There were two of these -- ``BootstrapResult.separates``, symmetric, and
    ``IndexResult.separated_from``, directional -- and a loop built on one while
    reporting the other would have disagreed with itself. Both now read this.

    Note what the test costs. Non-overlapping 95% intervals is a *conservative*
    comparison, roughly alpha = 0.006 rather than 0.05, so the failure mode is
    under-detecting real gains rather than accepting fake ones. On the reference corpus,
    with half widths of 0.02-0.10 against a human band 0.14 wide, almost nothing
    separates -- which is the honest answer and not a defect in the test.
    """
    return low_end > high_end


def bootstrap_index(
    spec: dict[str, Any],
    records: list[GameRecord],
    weights: Weights,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    components_of_interest: tuple[str, ...] = (),
    analysis: CorpusAnalysis | None = None,
) -> BootstrapResult | None:
    """Resample a finished tournament and re-score it, many times.

    ``analysis`` must be passed whenever the weights ask for a component the second pass
    supplies, and it is resampled in step with the records rather than held fixed. Two
    reasons, and the first one cost a debugging session: re-measuring without it makes
    every resample raise on the missing component, so the interval silently disappears
    and a score prints as though it were exact. The second is the real one -- the
    position-derived components are estimates from a sample of plies like everything
    else here, and freezing them would understate the interval by exactly the amount
    they contribute.

    Returns ``None`` when there is nothing to resample -- too few games, or a corpus
    that cannot be scored at all. A missing interval is reported as missing rather than
    faked with a zero, which would read as "perfectly precise".
    """
    if len(records) < 8:
        return None

    strata: dict[tuple[str, str], list[GameRecord]] = defaultdict(list)
    for record in records:
        strata[record.agents].append(record)

    # Analyses are keyed by the game they describe, so a resampled corpus can carry the
    # matching analyses rather than the original ones.
    by_game = {}
    if analysis is not None:
        by_game = {(g.seed, tuple(g.agents)): g for g in analysis.games}

    # Every resample holds these same record objects, so each game is read once.
    readings = GameReadings()
    rng = random.Random(seed)
    totals: list[float] = []
    tracked: dict[str, list[float]] = {name: [] for name in components_of_interest}

    for _ in range(resamples):
        resampled: list[GameRecord] = []
        for games in strata.values():
            resampled.extend(rng.choices(games, k=len(games)))
        drawn = None
        if analysis is not None:
            matched = [
                by_game[(r.seed, r.agents)]
                for r in resampled
                if (r.seed, r.agents) in by_game
            ]
            trials = (
                rng.choices(analysis.blunders, k=len(analysis.blunders))
                if analysis.blunders
                else []
            )
            drawn = CorpusAnalysis(games=matched, blunders=trials)
        try:
            measured = compute_components(
                spec, resampled, rate(resampled), drawn, readings=readings
            )
            scores = weights.score_all(measured.raw)
            total, _ = weights.combine(scores)
        except Exception:
            # A resample can be degenerate -- every game from one agent, no self-play,
            # a rating table that will not fit. Dropping it biases the interval very
            # slightly towards the well-behaved, which is the conservative direction.
            continue
        totals.append(total)
        for name in tracked:
            if name in measured.raw:
                tracked[name].append(measured.raw[name])

    if len(totals) < 2:
        return None

    totals.sort()
    deterministic = sum(
        1 for games in strata.values() if len({_outcome_key(g) for g in games}) == 1
    ) / max(len(strata), 1)

    # The resample reads each stratum as its own population, so every spread it reports
    # is short by the same factor. Widening the two ends about the resample's own mean
    # keeps the interval's shape; the mean itself is a location and does not move.
    correction = stratum_correction(len(games) for games in strata.values())
    centre = statistics.fmean(totals)

    return BootstrapResult(
        resamples=len(totals),
        mean=round(centre, 4),
        sd=round(statistics.pstdev(totals) * correction, 4),
        ci_low=round(centre - correction * (centre - _percentile(totals, 0.025)), 4),
        ci_high=round(centre + correction * (_percentile(totals, 0.975) - centre), 4),
        deterministic_pairings=round(deterministic, 3),
        component_sd={
            name: round(statistics.pstdev(values) * correction, 4)
            for name, values in tracked.items()
            if len(values) >= 2
        },
        component_mean={
            name: round(statistics.fmean(values), 4)
            for name, values in tracked.items()
            if len(values) >= 2
        },
        stratum_correction=round(correction, 4),
    )


def _outcome_key(record: GameRecord) -> tuple[int | None, int]:
    return (record.winner, record.plies)


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def component_noise(
    corpus: list[tuple[dict[str, Any], list[GameRecord], Any]],
    weights: Weights,
    components: tuple[str, ...],
    resamples: int = 60,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """How much of each component's corpus spread is real, and how much is resampling.

    The honest input to a weight. A component whose spread across wildly different games
    is barely larger than its own error bar cannot rank them, however good the idea
    behind it is -- and this is the only way to find that out, because a plausible
    measurement with a plausible spread looks exactly like a useful one until it is
    resampled. Under v2 it would have said that ``first_player_advantage``, carrying
    0.10 of the weight, was measured to a precision of 1.9 times its own spread.

    Returns ``corpus_sd``, ``noise_sd`` and their ratio per component.
    """
    values: dict[str, list[float]] = {name: [] for name in components}
    noise: dict[str, list[float]] = {name: [] for name in components}

    for spec, records, analysis in corpus:
        measured = compute_components(spec, records, rate(records), analysis)
        for name in components:
            if name in measured.raw:
                values[name].append(measured.raw[name])
        drawn = bootstrap_index(
            spec,
            records,
            weights,
            resamples=resamples,
            seed=seed,
            components_of_interest=components,
            analysis=analysis,
        )
        if drawn is None:
            continue
        for name, sd in drawn.component_sd.items():
            noise[name].append(sd)

    summary: dict[str, dict[str, float]] = {}
    for name in components:
        if len(values[name]) < 4 or not noise[name]:
            continue
        spread = statistics.pstdev(values[name])
        error = statistics.fmean(noise[name])
        summary[name] = {
            "corpus_sd": round(spread, 4),
            "noise_sd": round(error, 4),
            "ratio": round(spread / error, 2) if error > 0 else float("inf"),
        }
    return summary
