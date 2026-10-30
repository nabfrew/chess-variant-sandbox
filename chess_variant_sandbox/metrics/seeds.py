"""Several runs of one game, read as one measurement.

A run is a tournament from one seed. `docs/reference/run-to-run-spread.md` measured what
changes between two of them: the total moves by about 0.025 on the corpus, and four of
71 games change their *verdict* depending on which seed they were played from. So a
library score is the mean of four runs and a visitor's own design is one run, and both
are the same shape -- a visitor's score is the case n = 1.

This module is the only place that turns several runs into one number, one verdict and
one ladder. `docs/archive/plan-four-seeds.md` §2, §4 and §5 are the design; what follows
is the part of it that is arithmetic.

**A mean, never a pool.** Each run is scored alone on its own 256 games, so every total
is on the scale the weights were anchored on. Scoring the four runs' 1,024 games as one
corpus is a different quantity -- the rating prior pulls less and `skill_gradient` reads
13.7% steeper -- and it would need a weights version. Nothing here does that.

**The interval comes from the runs' own bootstraps, not from the spread of the totals.**
Four totals give a standard deviation on three degrees of freedom, whose own nine-in-ten
range is 0.34 to 1.61 of the truth, and every separation test downstream would become a
lottery. Each run's bootstrap rests on 200 resamples of 256 games (`DEFAULT_RESAMPLES`;
every stored run says 200) and is right in the pool: under `v12` the between-run sd over
the bootstraps' own is R = 0.93, 0.83 to 1.03, which contains 1
(`docs/reference/run-to-run-spread-published-sd.md`). So the runs' half-widths are added
in quadrature instead. That is exact when each run's error is normal, it keeps a lopsided
interval lopsided, and at one run it returns that run's own interval to the digit.

**The seat is the one check the runs do not vote on.** Every other gate reads one run's
256 games and the runs take a majority. The seat check reads 16 games a run, on which it
can refuse only a clean sweep, so voting on it makes the bar a function of how drawish
the game is. Instead the four runs' games are read once, together, against a tolerated
share -- `metrics/seat.py`, and `docs/reference/seat-effect-size.md` for why. A one-run
measurement has nothing to pool and keeps the per-run form.

**A run either has a total or has a sentence saying why not.** A run the roster cannot
read, and a run whose scorer raised, carry no number: they are never averaged, never
published as 0.0, and never filled in from another run. A mean needs `MIN_TOTALS` of
four, for the same reason a verdict does: a game that loses two ladders of four is one the
roster reads about half the time, and a mean over its readable runs describes the roster's
good days more than the game. The survivors' excess over the game's expected total is
`p x gap` (how often the ladder fails times how far a failed run sits below a readable one),
measured at 0.0014 to 0.0075 on the one library case and not removed by a fourth run;
`docs/reference/three-run-lean.md` §6-7.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from chess_variant_sandbox.metrics.index import (
    NOT_MEASURED_MANY,
    NOT_MEASURED_ONE,
    IndexResult,
    verdict_word,
)
from chess_variant_sandbox.metrics.seat import is_seat_violation, pooled_seat_reading, seat_details

# What to do when a game's runs disagree. All three are implemented, so the answer to
# `docs/decisions.md`'s open question is a one-line edit in `corpus.py` and a re-score,
# rather than a stage. `corpus.VERDICT_RULE` names the one in force and the basis stamps
# it, so a file made under another reads stale by name.
VOTE_TIE_REFUSES = "vote-tie-refuses"
VOTE_TIE_ADMITS = "vote-tie-admits"

# The vote, with the seat check taken out of it and read once over the runs pooled. The
# user's decision of 2026-09-19, *sweeps only*: a seat refusal at four runs is a claim
# about a share of decided games and not about one run's 16 (`metrics/seat.py`,
# `docs/reference/seat-effect-size.md` §13). The vote still decides every other check,
# and at one run this rule is `VOTE_TIE_REFUSES` exactly.
VOTE_POOLED_SEAT = "vote-tie-refuses+pooled-seat"

# As `VOTE_POOLED_SEAT`, but a run the roster could not read and that failed no gate it
# could read abstains rather than refusing (plan-teacher §10.16.4, rule (e), registered
# 2026-10-29). A game is viable or not viable only when every way its abstaining runs
# could have voted gives the same verdict, and *not measured* otherwise. With no
# abstention it is `VOTE_POOLED_SEAT` exactly.
VOTE_BLIND_ABSTAINS = "vote-tie-refuses+pooled-seat+blind-abstains"

VOTE_RULES = (VOTE_TIE_REFUSES, VOTE_TIE_ADMITS, VOTE_POOLED_SEAT, VOTE_BLIND_ABSTAINS)

# How many of a game's runs must carry a total before their mean is published. A number
# needs what a verdict needs. Under `vote-tie-admits` a game can be viable on two runs of
# four, so an answer of B lowers this to 2 in the same edit.
# What three protects is the reading, not an unbiased mean: the excess of a three-run
# mean is `p x gap`, 0.0014 to 0.0075 on `intransitive`, and a fourth run would not shrink
# it (`docs/reference/three-run-lean.md` §6-7; the mark is `trust.py`'s `ladder.partly`).
MIN_TOTALS = 3


def minimum_totals(runs: int) -> int:
    """How many scored runs a game of this many runs needs before it has an index.

    `MIN_TOTALS` is the rule for a library game. A visitor's design is one run, and one
    run must be scored -- there is no majority to take.
    """
    return min(MIN_TOTALS, max(runs, 1))


@dataclass(frozen=True)
class RunReading:
    """One run of one game, as the combiner reads it.

    Built from a live `IndexResult` or from a stored `index-<weights>.json`, so a run
    just played and a run read as it stands arrive here in the same shape. It carries no
    basis of its own: every claim about a run is in the game's one file, which `_store`
    stamps once for all of them.
    """

    seed: int
    # Whether this run produced a score at all. A placeholder is not a score, and this is
    # the one thing the combiner asks before it averages anything.
    scored: bool
    total: float | None
    sd: float | None
    viable: bool
    violations: tuple[str, ...] = ()
    unreadable: tuple[str, ...] = ()
    scores: Mapping[str, float] = field(default_factory=dict)
    raw: Mapping[str, float] = field(default_factory=dict)
    contributions: Mapping[str, float] = field(default_factory=dict)
    families: Mapping[str, float] = field(default_factory=dict)
    interval: Mapping[str, Any] | None = None
    standings: tuple[Mapping[str, Any], ...] = ()
    games: int = 0
    # This run's seat splits, the diagnostics keys `metrics/seat.py` names. Carried
    # because the pooled seat reading is the one thing about a run that is read across
    # runs rather than averaged, and a run whose scorer raised carries none.
    seat: Mapping[str, Any] = field(default_factory=dict)
    # How this run's records came to be here: played, or adopted from a campaign and
    # re-derived. Recorded per run because it is the only claim that is not the game's.
    origin: str = "played"
    # False for a blind run that failed no gate it could read: it abstains under
    # `VOTE_BLIND_ABSTAINS` (`IndexResult.measured`). A stored run without the key reads
    # as measured, which is how its vote was counted when it was stored.
    measured: bool = True

    @classmethod
    def from_result(
        cls,
        seed: int,
        result: IndexResult,
        standings: Sequence[Mapping[str, Any]] = (),
        games: int = 0,
        origin: str = "played",
    ) -> RunReading:
        """A run that was scored, whatever the scorer made of it."""
        return cls(
            seed=seed,
            scored=result.scored,
            total=result.headline if result.scored else None,
            sd=result.sd if result.scored else None,
            viable=bool(result.viable),
            violations=tuple(result.violations),
            unreadable=tuple(result.unreadable),
            scores=dict(result.scores),
            raw=dict(result.raw),
            contributions=dict(result.contributions),
            families=dict(result.families),
            interval=dict(result.interval) if result.interval else None,
            standings=tuple(dict(standing) for standing in standings),
            games=games,
            origin=origin,
            seat=seat_details(result.diagnostics),
            measured=bool(result.measured),
        )

    @classmethod
    def unscored(
        cls,
        seed: int,
        why: str,
        standings: Sequence[Mapping[str, Any]] = (),
        games: int = 0,
        origin: str = "played",
    ) -> RunReading:
        """A run whose scorer raised, carrying the sentence instead of a number.

        The other runs are unaffected. A game with no blunder trials raises under `v10`
        today, and the honest reading of that is *this run has no total*, not zero.
        """
        return cls(
            seed=seed,
            scored=False,
            total=None,
            sd=None,
            viable=False,
            violations=(why,),
            standings=tuple(dict(standing) for standing in standings),
            games=games,
            origin=origin,
        )

    def as_dict(self) -> dict[str, Any]:
        """The `runs[]` entry in the game's file."""
        return {
            "seed": self.seed,
            "index": self.total,
            "sd": self.sd,
            "viable": self.viable,
            "violations": list(self.violations),
            "unreadable": list(self.unreadable),
            "measured": self.measured,
            "origin": self.origin,
        }


@dataclass(frozen=True)
class Verdict:
    """What a game's runs decided together, and how they split."""

    rule: str
    viable: bool
    accept: int
    refuse: int
    # The split in one sentence, or `""` when the runs agreed or there is only one.
    split: str = ""
    # Runs that neither accepted nor refused (`VOTE_BLIND_ABSTAINS`), and whether the
    # vote reached a verdict at all. `measured` False is *not measured*: `viable` is
    # False beside it because the game was not admitted, not because it was ruled out.
    abstain: int = 0
    measured: bool = True

    @property
    def word(self) -> str:
        return verdict_word(self.viable, self.measured)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "accept": self.accept,
            "refuse": self.refuse,
            "abstain": self.abstain,
            "measured": self.measured,
            "split": self.split,
        }


def pools_the_seat(rule: str, runs: int) -> bool:
    """Whether this measurement reads the seat check over its runs pooled.

    Only `VOTE_POOLED_SEAT`, and only where there is more than one run to pool. A
    visitor's design is one run of 16 pair games, on which the tolerated share cannot
    refuse even a clean sweep, so that measurement keeps the built per-run form and
    reads exactly as it did before (`docs/reference/seat-effect-size.md` §3).
    """
    return rule in (VOTE_POOLED_SEAT, VOTE_BLIND_ABSTAINS) and runs > 1


def abstains(run: RunReading, rule: str) -> bool:
    """Whether this run neither accepts nor refuses under this rule.

    Only under `VOTE_BLIND_ABSTAINS`, and only a blind run that failed no gate it could
    read (`IndexResult.measured`). Under every older rule it refuses, as it always did.
    """
    return rule == VOTE_BLIND_ABSTAINS and not run.measured


def verdict(runs: Sequence[RunReading], rule: str = VOTE_TIE_REFUSES) -> Verdict:
    """Whether the game stays in, over the runs' own verdicts.

    One function, because a verdict combined in two places is two rules. A run with no
    total votes to refuse: a gate it could not reach is not a gate it passed. A run the
    roster could not read refuses too under the older rules. Under `VOTE_BLIND_ABSTAINS`
    it refuses only on a gate it could read, and otherwise abstains (plan-teacher
    §10.16.4 point 2): with A accepts, R refusals and B abstentions, the game is viable iff
    A > R + B, not viable iff A + B <= R (the tie refuses either way), and *not measured*
    in between. At B = 0 that is the older vote exactly.

    Under `VOTE_POOLED_SEAT` a run's own seat refusal is not counted here, because the
    seat is decided once over the runs pooled and counting it twice would let four runs
    refuse a game the pooled reading accepts. Every other check still votes.

    At one run every rule returns that run's own verdict, so nothing about a visitor's
    score turns on which rule is in force.
    """
    if rule not in VOTE_RULES:
        raise ValueError(f"unknown verdict rule {rule!r}; it is one of {', '.join(VOTE_RULES)}")
    pooled = pools_the_seat(rule, len(runs))
    abstain = sum(1 for run in runs if abstains(run, rule))
    accept = sum(
        1 for run in runs
        if not abstains(run, rule) and (_viable_but_for_seat(run) if pooled else run.viable)
    )
    refuse = len(runs) - accept - abstain
    if rule == VOTE_TIE_ADMITS:
        viable = accept >= refuse
        measured = True
    else:
        viable = accept > refuse + abstain
        measured = viable or accept + abstain <= refuse
    return Verdict(
        rule=rule,
        viable=viable,
        accept=accept,
        refuse=refuse,
        split=_split(accept, refuse, viable, abstain, measured),
        abstain=abstain,
        measured=measured,
    )


def _viable_but_for_seat(run: RunReading) -> bool:
    """This run's verdict with its seat sentences struck out.

    A run that scored and whose only refusals were seat refusals votes to accept: the
    pooled reading answers that question for the whole measurement. A run that carries
    no total still votes to refuse, because a check it never reached is not one it
    passed, which is what `unscored` already meant.
    """
    if not run.scored:
        return False
    return not any(not is_seat_violation(text) for text in run.violations)


_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight")


def _word(count: int) -> str:
    return _WORDS[count] if count < len(_WORDS) else str(count)


def _split(
    accept: int, refuse: int, viable: bool, abstain: int = 0, measured: bool = True
) -> str:
    """The disagreement in one sentence, or nothing to say.

    Nothing to say covers the two cases a reader does not need told: every run agreed, or
    there was only one. A run that abstained is a disagreement worth a sentence.
    """
    total = accept + refuse + abstain
    if total < 2 or (not abstain and (not accept or not refuse)):
        return ""
    if not measured:
        ending = "it is not measured"
    else:
        ending = "it stays in" if viable else "it is ruled out"
    if abstain:
        parts = []
        if accept:
            parts.append(f"{_word(accept)} {'accepts' if accept == 1 else 'accept'} it")
        if refuse:
            parts.append(f"{_word(refuse)} {'refuses' if refuse == 1 else 'refuse'} it")
        parts.append(f"{_word(abstain)} could not be read")
        said = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
        if abstain == total:
            said = "none could be read"
        return f"Of its {_word(total)} runs, {said}, so {ending}."
    head = (
        f"{_word(accept).capitalize()} of its {_word(total)} runs "
        f"{'accepts' if accept == 1 else 'accept'} it and {_word(refuse)} "
        f"{'refuses' if refuse == 1 else 'refuse'} it"
    )
    if accept == refuse:
        # Read off what the tie did rather than off the rule's name, so a rule that
        # changes some other part of the vote cannot silently flip this word.
        tie = "admits" if viable else "refuses"
        return f"{head}. A tie {tie}, so {ending}."
    return f"{head}, so {ending}."


@dataclass(frozen=True)
class Combined:
    """Several runs of one game as the one measurement every reader trusts."""

    seeds: tuple[int, ...]
    index: float | None
    totals_used: int
    sd: float | None
    ci_low: float | None
    ci_high: float | None
    viable: bool
    verdict: Verdict
    violations: tuple[str, ...]
    # The seat check over the runs pooled: both bands' splits, the share each one gives
    # its favoured seat, and whether that refuses the game. Published whether or not it
    # refuses, because a lean short of a sweep is a thing a reader wants printed beside
    # the score. Empty where no run measured a seat at all.
    seat: dict[str, Any]
    families: dict[str, float]
    standings: list[dict[str, Any]]
    games: int
    runs: list[dict[str, Any]]
    # The mean's own `index-<weights>.json`, in `IndexResult`'s keys so that every reader
    # joining on the measurement's directory opens it unchanged.
    detail: dict[str, Any]
    # False for the third outcome, *not measured* (`Verdict.measured`); `viable` is False
    # beside it. A pooled seat refusal is always measured: it reads games actually played.
    measured: bool = True

    @property
    def word(self) -> str:
        return verdict_word(self.viable, self.measured)


def combine(
    ruleset_hash: str,
    ruleset_name: str,
    weights_version: str,
    runs: Sequence[RunReading],
    rule: str = VOTE_TIE_REFUSES,
) -> Combined:
    """One game's runs as one measurement.

    The runs are sorted by seed first and every aggregate is taken in that order, so the
    answer is a property of the runs and not of the order they were combined in --
    floating-point addition is not associative, and a file that differed by a digit
    depending on which worker finished first would be a determinism bug wearing a
    rounding error.
    """
    ordered = sorted(runs, key=lambda run: run.seed)
    if not ordered:
        raise ValueError("a measurement needs at least one run")

    usable = [run for run in ordered if run.scored and run.total is not None]
    # What the display keys are averaged over. The number-bearing keys are the usable
    # runs' alone; the components a placeholder run *did* measure are still facts, and a
    # game whose every run was withheld should still show how far off it was.
    reported = usable or ordered

    pooled = pools_the_seat(rule, len(ordered))
    seat = pooled_seat_reading([run.seat for run in ordered if run.seat]) if pooled else {}

    decided = verdict(ordered, rule)
    enough = len(usable) >= minimum_totals(len(ordered))
    index = round(statistics.fmean(run.total or 0.0 for run in usable), 4) if enough else None

    interval = _interval(usable) if enough else None
    sd = interval["sd"] if interval else None

    families = _means(run.families for run in reported)
    viable = decided.viable and not seat.get("refuses")
    # A pooled seat refusal makes the game not viable whatever the abstentions were
    # (§10.16.4 point 3), so it is a verdict, not *not measured*.
    measured = decided.measured or bool(seat.get("refuses"))
    violations = _violations(ordered, decided, seat, pooled)

    detail = {
        "ruleset_hash": ruleset_hash,
        "ruleset_name": ruleset_name,
        "weights_version": weights_version,
        # The mean of the runs' totals by definition, and never recomputed from the mean
        # component scores: families combine geometrically, so the two differ in the
        # third digit and only one of them is what the runs measured.
        "total": index,
        "viable": viable,
        "measured": measured,
        "violations": list(violations),
        "scores": _means(run.scores for run in reported),
        "contributions": _means(run.contributions for run in reported),
        "raw": _means(run.raw for run in reported),
        "families": dict(families),
        # Empty on purpose. No page reads it, and each run's diagnostics stay in the run's
        # own index file, where they describe the corpus they were taken over.
        "diagnostics": {},
        "unreadable": list(_unreadable(reported)),
        "llm_score": None,
        "llm_notes": "",
        "combined": None,
        "interval": interval,
    }

    return Combined(
        seeds=tuple(run.seed for run in ordered),
        index=index,
        totals_used=len(usable),
        sd=sd,
        ci_low=interval["ci_low"] if interval else None,
        ci_high=interval["ci_high"] if interval else None,
        viable=viable,
        verdict=decided,
        violations=violations,
        seat=dict(seat),
        families=families,
        standings=_ladder(ordered),
        games=sum(run.games for run in ordered),
        runs=[run.as_dict() for run in ordered],
        detail=detail,
        measured=measured,
    )


def _violations(
    runs: Sequence[RunReading],
    decided: Verdict,
    seat: Mapping[str, Any],
    pooled: bool,
) -> tuple[str, ...]:
    """Why the game was ruled out, in the runs' own words.

    Empty when it is viable. At one run the run's own sentences stand unchanged, which is
    what a visitor's score has always said; where several runs disagreed the split comes
    first and each refusing run's sentences are named by seed, because "which run" is the
    first question a reader of a 3-1 has.

    Under the pooled seat rule the seat speaks once, in the pooled reading's own words,
    and the per-run seat sentences are dropped: each of those describes 16 games and the
    refusal rests on all 64.

    A game *not measured* is not ruled out, but its reader still asks why there is no
    verdict: the sentences say so first (`NOT_MEASURED_MANY`, or `NOT_MEASURED_ONE` at
    one run), then each run that did not accept, by seed, in its own words.
    """
    pooled_seat = tuple(str(line) for line in (seat.get("violations") or ())) if pooled else ()
    if decided.viable and not pooled_seat:
        return ()

    lines = [decided.split] if decided.split and not decided.viable else []
    if not decided.measured and not seat.get("refuses"):
        why = NOT_MEASURED_MANY if len(runs) > 1 else NOT_MEASURED_ONE
        lines.insert(0, f"Not measured: {why}.")
    elif not decided.measured and lines:
        # The vote reached no verdict, and the pooled seat refused on top (§10.16.4
        # point 3): the game is ruled out, so its split must not say *not measured*.
        lines[0] = lines[0].replace("so it is not measured", "so the vote is undecided")
    for run in runs:
        vote = _viable_but_for_seat(run) if pooled else run.viable
        if vote or decided.viable:
            continue
        prefix = f"run {run.seed}: " if len(runs) > 1 else ""
        said = [text for text in run.violations if not (pooled and is_seat_violation(text))]
        lines.extend(prefix + violation for violation in said)
    return tuple(lines) + pooled_seat


def _interval(usable: Sequence[RunReading]) -> dict[str, Any] | None:
    """The mean's interval, from the runs' own bootstraps.

    Each end combines the runs' half-widths in quadrature about the mean of their
    bootstrap centres. Independent errors add that way, and a run's interval is centred on
    its bootstrap mean today, so at one run this returns that run's interval to the digit.

    `deterministic_pairings` is the mean of the runs' shares rather than the worst of
    them: taking the maximum would refuse more intervals as seeds are added without any
    game changing, which is the defect pooling had.
    """
    intervals = [run.interval for run in usable if run.interval]
    if len(intervals) != len(usable) or not intervals:
        # One run without a bootstrap makes the mean's interval a claim about a different
        # number of runs than the mean is. Absent is reported as absent.
        return None

    count = len(intervals)
    centre = statistics.fmean(float(found["mean"]) for found in intervals)
    low = math.sqrt(
        sum((float(f["mean"]) - float(f["ci_low"])) ** 2 for f in intervals)
    ) / count
    high = math.sqrt(
        sum((float(f["ci_high"]) - float(f["mean"])) ** 2 for f in intervals)
    ) / count
    sd = math.sqrt(sum(float(f["sd"]) ** 2 for f in intervals)) / count
    return {
        "mean": round(centre, 4),
        "sd": round(sd, 4),
        "ci_low": round(centre - low, 4),
        "ci_high": round(centre + high, 4),
        "deterministic_pairings": round(
            statistics.fmean(float(f.get("deterministic_pairings", 0.0)) for f in intervals), 3
        ),
        "stratum_correction": round(
            statistics.fmean(float(f.get("stratum_correction", 1.0)) for f in intervals), 4
        ),
        "runs": count,
        # Per run, not in total. The runs all resample the same number of times, and
        # reporting a sum would read as a precision nobody bought.
        "resamples": int(intervals[0].get("resamples", 0)),
    }


def _means(tables: Any) -> dict[str, float]:
    """Per-key mean over a sequence of tables, in first-appearance order.

    A key one run measured and another did not is averaged over the runs that have it.
    Empty tables contribute nothing, so a placeholder beside three real runs cannot pull
    a component towards zero.
    """
    gathered: dict[str, list[float]] = {}
    for table in tables:
        for key, value in table.items():
            gathered.setdefault(key, []).append(float(value))
    return {key: round(statistics.fmean(values), 4) for key, values in gathered.items()}


def _unreadable(runs: Sequence[RunReading]) -> tuple[str, ...]:
    """Components no run in this set could read.

    The intersection, not the union: a component three runs measured is a component the
    instrument can read here, and one run's withheld ladder does not take it away.
    """
    withheld: set[str] | None = None
    for run in runs:
        withheld = set(run.unreadable) if withheld is None else withheld & set(run.unreadable)
    return tuple(sorted(withheld or ()))


def _ladder(runs: Sequence[RunReading]) -> list[dict[str, Any]]:
    """The mean ladder: each agent's mean rating, with its games summed.

    Over every run, not only the usable ones, because a run whose score was withheld still
    played its games and rated its players. A slope is linear in the ratings, so the mean
    ladder is the one ladder consistent with the mean `skill_gradient` -- one fit over the
    pooled games is a different quantity, and it reads 13.7% steeper.
    """
    ratings: dict[str, list[float]] = {}
    games: dict[str, int] = {}
    for run in runs:
        for standing in run.standings:
            name = str(standing.get("name", ""))
            ratings.setdefault(name, []).append(float(standing.get("rating", 0.0)))
            games[name] = games.get(name, 0) + int(standing.get("games", 0) or 0)
    ordered = sorted(
        ((name, statistics.fmean(values)) for name, values in ratings.items()),
        key=lambda pair: -pair[1],
    )
    return [
        {"name": name, "rating": rating, "games": games[name]}
        for name, rating in ordered
    ]
