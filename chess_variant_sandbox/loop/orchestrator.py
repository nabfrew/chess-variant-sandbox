"""The design loop: propose, gate, simulate, score, judge, record, repeat.

The ordering is a cost ladder. Schema validation is free, the smoke test costs a
second, the tournament costs minutes and the judge costs money -- so each stage only
runs on what survived the cheaper one. Most proposals die at validation or the smoke
test, which is what makes a long run affordable.

Parent selection is deliberately not pure hill-climbing. Always revising the current
best converges on a local dead end quickly, because the model's revisions of a game
tend to look like that game. A share of generations start from nothing at all, and the
rest sample from the top few rather than always the top one.

A score taken here is one run on `DEFAULT_AGENTS` at three games a pairing, and a library
score is the mean of four runs on the corpus roster at four. The two are different
quantities and never share a column: this one is `designIndex`, never `index`, and the
rung that turns a candidate into the second is `chess-variant-sandbox test`.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_variant_sandbox.llm.client import JUDGE_MODEL, PROPOSAL_MODEL, Claude, LLMUnavailable
from chess_variant_sandbox.llm.compare import compare_rulesets
from chess_variant_sandbox.llm.feedback import describe, describe_failures
from chess_variant_sandbox.llm.propose import propose_ruleset
from chess_variant_sandbox.loop.archive import Archive, Entry
from chess_variant_sandbox.loop.gates import smoke_test
from chess_variant_sandbox.loop.grid import describe_cell, descriptor, label
from chess_variant_sandbox.loop.operators import mutate
from chess_variant_sandbox.loop.preference import PreferenceLog
from chess_variant_sandbox.loop.screen import screen
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS, load_weights, score_ruleset
from chess_variant_sandbox.metrics.bootstrap import clears
from chess_variant_sandbox.spec.canonical import load_spec, spec_hash
from chess_variant_sandbox.tournament import TournamentConfig, run_tournament
from chess_variant_sandbox.tournament.analysis import analyse_corpus

DEFAULT_AGENTS = ("random", "greedy", "minimax:nodes=400", "minimax:nodes=1600")

# Design language for the weakest family, so a revision has a direction. These say what
# to work on, never how it is scored -- see llm/feedback.py for where that line sits.
_FOCUS = {
    "depth": "make the positions reward thinking ahead -- the best move should be findable but not obvious",
    # Was "even out the two seats, and make an advantage something a good player can
    # convert". The second half was a prose statement of `lead_conversion`, which was
    # measured and unweighted when it was written and is scored from v5 onward. A
    # scored component's procedure does not belong in what the proposer is shown.
    "balance": "even out the two seats, so that neither is the one you would rather have",
    # Was "get the length of a game and the number of options per turn into a
    # comfortable range", which states the *band* two scored components are curved on.
    # A brief may name the quality it wants; naming the target is handing over the
    # shape of the objective function, which is the half of it that can be gamed.
    "pacing": "give the game room to develop -- a turn should offer a real choice, and a position should have somewhere to go",
    "variety": "make it possible to play this game in more than one way",
    "drama": "let the advantage change hands during a game rather than being settled early",
    # No "elegance". It said "cut the rules back", and the user took rule yield out of the
    # score (2026-09-23, `docs/archive/plan-variety.md` §5.4): the designer is neither told about
    # few rules nor rewarded for them. A `v10` score still carries the family, so
    # `_focus_for` passes over any family with no advice here.
}


@dataclass
class LoopConfig:
    generations: int = 8
    agents: tuple[str, ...] = DEFAULT_AGENTS
    games_per_pairing: int = 3
    workers: int = 0
    explore_rate: float = 0.3
    top_k: int = 5
    smoke_games: int = 150
    # One seed for the whole run, not one per generation.
    #
    # Every candidate is measured on the same draw of games, which is what makes two
    # candidates comparable at all: with a per-generation seed, re-evaluating the *same*
    # ruleset returns a different score, and the loop cannot tell that difference from a
    # design improvement. Measured on a three-generation run of relabelled clones -- the
    # same game three times -- a per-generation seed produced 0.372, 0.515 and 0.318,
    # a spread of 0.20 on a corpus whose within-run interval was near zero. Seat streams
    # already exclude the ruleset hash, so identical rules under one seed play identical
    # games and score identically, which is the property the loop needs.
    #
    # What this does *not* buy is variance reduction between a parent and a child: their
    # legal-move sets differ, so the streams decorrelate at the first differing decision.
    # And it does not estimate seed-to-seed variation, which needs several seeds per
    # candidate and is a budget question, not a selection one.
    seed: int = 0
    weights: str = DEFAULT_WEIGHTS
    # Whether to ask the model which of two games is better. It replaces the absolute
    # 0-1 judgement that used to be blended into the headline at a quarter weight: an
    # absolute score from a model is noisy and poorly calibrated, and at that weight it
    # became the de facto ranker inside a band the objective index resolves to about
    # 0.05. A preference needs no scale, and it is used only where the index admits it
    # cannot choose -- as the tie-break, never as a term.
    judge: bool = True
    # How often a model proposal is aimed at an empty neighbouring cell of the
    # behaviour grid rather than at whatever the measurements suggest. This is the half
    # of quality-diversity search that only works with a model operator: an ordinary
    # mutation lands where it lands, and a model can be *told* the kind of game to make.
    target_rate: float = 0.4
    # How many incumbents a new candidate is compared against. Comparisons are O(n^2) if
    # allowed to be, and each one is two model calls, so a candidate meets the top of its
    # own leaderboard and nothing else.
    comparisons: int = 2
    # What to explore, in the designer's own words. Reaches the proposer alongside the
    # focus the measurements suggest, and is recorded on every candidate the run
    # produces -- a game's provenance is not complete without what its author was
    # asked for.
    theme: str = ""
    # Rulesets to start from. Each is measured by the same path a proposal is, because
    # a starting point scored under a different roster is not comparable to anything
    # the run will produce.
    seeds: tuple[Path, ...] = ()
    # Share of generations that make their candidate with a mechanical edit rather than
    # a model call. The two operators are good at different things and cost differently:
    # a mechanical child is free and local, a model call is expensive and can add
    # something that was not there. Spending the expensive one on a single-step tweak is
    # the worst available allocation, which is why the default is a half-and-half split
    # rather than the model doing everything. At 1.0 nothing is proposed by a model;
    # with ``judge`` off as well, the run contacts no model at all and needs no
    # credentials.
    mechanical_rate: float = 0.5
    # How many children a generation produces, and how many of the survivors reach the
    # rated roster.
    #
    # The loop used to evaluate one candidate per generation, which put the whole
    # tournament budget on whatever a single draw produced. Measured on the
    # ten-generation demo run that meant ten full tournaments buying ten candidates,
    # and the interval said nothing separated any of them. Both halves of the fix were
    # already paid for: `loop/operators.py` makes a valid local edit for the price of a
    # validation, and `loop/screen.py` sorts one at 0.70s. A batch of eight therefore
    # costs seconds and buys the rated roster a far better candidate to spend minutes
    # on.
    #
    # `promote` stays at one so the *cost* of a generation is unchanged. Raising it is
    # how a run trades generations for breadth, and it is a budget decision rather than
    # a search one.
    batch: int = 8
    promote: int = 1
    directory: Path = field(default_factory=lambda: Path("runs/design"))

    def __post_init__(self) -> None:
        _require_measurable_ladder(self.agents)
        if self.batch < 1:
            raise ValueError("batch must be at least 1")
        if not 1 <= self.promote <= self.batch:
            raise ValueError("promote must be between 1 and batch")

    def as_dict(self) -> dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items()}
        data["directory"] = str(self.directory)
        data["agents"] = list(self.agents)
        data["seeds"] = [str(path) for path in self.seeds]
        return data


def evolve(
    config: LoopConfig,
    proposer: Claude | None = None,
    judge_client: Claude | None = None,
    report: Callable[[str], None] = print,
) -> Archive:
    """Run the design loop and return the archive it wrote to."""
    archive = Archive(config.directory)
    weights = load_weights(config.weights)
    rng = random.Random(config.seed)

    # Which clients this run has to build for itself, asked before it builds them: a
    # caller that supplied its own has its own arrangements, and the warning below is
    # about *this* environment's credentials.
    needs = [
        name
        for name, wanted in (
            ("proposals", proposer is None and config.mechanical_rate < 1.0),
            ("judging", judge_client is None and config.judge),
        )
        if wanted
    ]
    if proposer is None and config.mechanical_rate < 1.0:
        proposer = Claude(model=PROPOSAL_MODEL, effort="high")
    if config.judge and judge_client is None:
        judge_client = Claude(model=JUDGE_MODEL, effort="medium")
    _warn_if_unconfigured(config, needs, report)
    preferences = PreferenceLog(config.directory)

    _seed_archive(config, archive, weights, report)

    start_generation = archive.next_generation()
    for offset in range(config.generations):
        generation = start_generation + offset
        report(f"\n--- generation {generation} ---")
        try:
            _run_generation(
                generation, config, archive, weights, rng, proposer, judge_client,
                preferences, report,
            )
        except LLMUnavailable as error:
            report(f"stopping: {error}")
            break

    report("\n" + archive.summary())
    report("\n" + _cost_profile(archive))
    for line in _model_profile(proposer, judge_client):
        report(line)
    table = preferences.table()
    if table.comparisons:
        report(
            f"  the judge made {table.comparisons} comparisons and reversed itself on "
            f"{table.flip_rate:.0%} of them when the pair was swapped"
        )
    return archive


def _model_profile(*clients: Claude | None) -> list[str]:
    """What the run asked of each model, in tokens.

    The other half of `_cost_profile`, and the half that is actually money: a proposal
    costs more than the tournament that follows it, and until now a run said where its
    *seconds* went and nothing at all about that. `Claude.usage` has been accumulating
    this the whole time with no one to report it.

    Tokens rather than a currency, on purpose. A price belongs to a moment and this file
    would carry a stale one forever; the model is named, so anyone can multiply.
    """
    lines = []
    for client in clients:
        usage = getattr(client, "usage", None)
        if usage is None or not usage.calls:
            continue
        # Both cache meters, never just the read. A run reporting nothing from cache is
        # either one whose prefix was too short to cache -- free, and fine -- or one that
        # paid to open an entry on every call and let it expire unread, which costs more
        # than not caching at all. The read alone cannot tell those apart, and which one
        # a run is is the only thing that says whether `CACHE_TTL` is set right.
        cached = ""
        if usage.cache_write_tokens or usage.cache_read_tokens:
            cached = (
                f", cache {usage.cache_read_tokens:,} read / "
                f"{usage.cache_write_tokens:,} written"
            )
            if not usage.cache_read_tokens:
                cached += " (nothing read back -- the entries expired unused)"
        lines.append(
            f"  {getattr(client, 'model', '?')}: {usage.calls} call(s), "
            f"{usage.input_tokens:,} in, {usage.output_tokens:,} out{cached}"
        )
    return lines


def _warn_if_unconfigured(
    config: LoopConfig,
    needs: list[str],
    report: Callable[[str], None],
) -> None:
    """Say up front that a run which needs a model has not found one.

    A warning rather than a refusal, and the asymmetry is deliberate: the SDK resolves
    credentials several ways this check does not know about -- a named profile, a
    federation, an environment this function has never heard of -- so a false negative
    is likely enough that refusing to start would eventually stop a run that would have
    worked. What it costs to be wrong the other way is one line of prose.

    Worth saying at all because the first model call comes *after* every starting point
    has been measured, so the alternative is minutes of tournaments and then a
    credentials error. It is also the first caller `Claude.is_configured` has ever had.
    """
    if not needs or Claude.is_configured():
        return
    report(
        f"warning: no Anthropic credentials found, and this run uses the model for "
        f"{' and '.join(needs)}. Set ANTHROPIC_API_KEY or run `ant auth login` -- or "
        f"run without a model at all with --mechanical-rate 1.0 --no-judge"
    )
    if config.mechanical_rate < 1.0:
        report("  (the run will stop at the first proposal)")


def _cost_profile(archive: Archive) -> str:
    """Where the run's time went, rung by rung.

    Printed rather than merely stored, because the decision it informs is whether the
    next optimisation is worth building at all. Stage 9b's case rests on
    pool-construction overhead dominating a population of candidates; if the profile
    says the time is somewhere else, the honest answer is to say so and stop.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for entry in archive.entries:
        for phase, value in (entry.detail or {}).get("seconds", {}).items():
            totals[phase] = totals.get(phase, 0.0) + float(value)
            counts[phase] = counts.get(phase, 0) + 1
    if not totals:
        return "no timings recorded"
    overall = sum(totals.values())
    lines = [f"where the time went ({overall:.0f}s across {len(archive.entries)} candidates)"]
    for phase, value in sorted(totals.items(), key=lambda kv: -kv[1]):
        lines.append(
            f"  {phase:<12} {value:>8.1f}s  {value / overall:>5.0%}  "
            f"over {counts[phase]} candidates, {value / counts[phase]:.2f}s each"
        )
    return "\n".join(lines)


def _seed_archive(
    config: LoopConfig,
    archive: Archive,
    weights: Any,
    report: Callable[[str], None],
) -> None:
    """Measure the rulesets this run was told to start from, once.

    A run seeded with an existing game needs that game *in the archive* before it can be
    chosen as a parent -- and needs it measured on this run's roster, not on whatever
    roster produced the number someone quoted. Already-seen specs are skipped, so a
    resumed run does not pay for its starting point twice.
    """
    for path in config.seeds:
        spec = load_spec(path)
        digest = spec_hash(spec)
        if digest in archive.seen_hashes():
            continue
        report(f"\n--- starting point: {spec.get('name', path)} [{digest}] ---")
        evaluate(
            spec=spec,
            generation=archive.next_generation(),
            parent=None,
            parent_entry=None,
            config=config,
            archive=archive,
            weights=weights,
            judge_client=None,
            preferences=PreferenceLog(config.directory),
            report=report,
        )


def _run_generation(
    generation: int,
    config: LoopConfig,
    archive: Archive,
    weights: Any,
    rng: random.Random,
    proposer: Claude | None,
    judge_client: Claude | None,
    preferences: PreferenceLog,
    report: Callable[[str], None],
) -> None:
    # "Start from nothing" is a thing only the model operator can do, so a run without
    # one must not draw it. It used to: at `--mechanical-rate 1.0` roughly
    # `explore_rate` of every run's generations chose no parent, found no model to
    # propose from scratch, and produced nothing at all.
    parent_entry = _choose_parent(
        archive, config, rng, preferences, explore=proposer is not None
    )
    parent_spec = archive.load_spec(parent_entry.hash) if parent_entry else None
    feedback = parent_entry.notes if parent_entry else ""
    goal = _focus_for(parent_entry, weights) if parent_entry else ""
    target = _target_cell(archive, parent_entry, config, rng)
    if target is not None:
        # Ordinary MAP-Elites mutates blindly and hopes to land in an empty cell. A model
        # can be told the cell, which is the specific reason a behaviour grid and a
        # learned operator are a better pairing than either half suggests alone.
        goal = (
            f"Aim for a game of this kind, which nothing in the archive has reached yet: "
            f"{describe_cell(target)}. Keep what is good about the game below, but the "
            f"result should genuinely be that kind of game."
        )

    if config.theme:
        # The designer's brief first, the measurements' suggestion second. They are
        # different kinds of instruction and the run was started for the first one.
        goal = f"{config.theme}\n\nAlso worth attention: {goal}" if goal else config.theme

    if parent_entry:
        # The parent's number as the archive shows it, with the runs it rests on.
        report(f"revising {parent_entry.name} [{parent_entry.hash}] ({parent_entry.scored_as()})")
    else:
        report("proposing from scratch")

    candidates = _produce(
        generation, config, archive, rng, proposer, parent_entry, parent_spec,
        feedback, goal, report,
    )
    if not candidates:
        report("  nothing to evaluate this generation")
        return

    # Everything the generation produced meets the cheap rungs before any of it meets
    # the expensive one. That is the whole of stage 9a: the gates already cost under two
    # seconds each against minutes for the rated roster, and the loop was spending the
    # rated roster on whichever single candidate the draw happened to produce. Measured
    # on the ten-generation demo, ten full tournaments bought ten candidates and the
    # interval said nothing separated any of them.
    gate_started = time.perf_counter()
    survivors: list[tuple[Gated, str | None, dict[str, Any]]] = []
    for spec, parent_hash, origin in candidates:
        gated = run_gates(spec, config, report)
        if gated.passed:
            survivors.append((gated, parent_hash, origin))
        else:
            archive.record(
                _rejection(spec, generation, parent_hash, gated, config, origin),
                spec=spec,
            )
    gate_seconds = time.perf_counter() - gate_started
    report(
        f"  {len(survivors)} of {len(candidates)} survived the cheap rungs "
        f"in {gate_seconds:.1f}s"
    )

    chosen = _promote(survivors, archive, rng, config, report)
    for gated, parent_hash, origin in chosen:
        measure(
            spec=gated.spec,
            gated=gated,
            # The whole batch's gate cost, charged to the candidates that got a
            # tournament out of it. A survivor that was not promoted leaves no archive
            # entry -- deliberately, so the operator can rediscover it later -- which
            # means its seconds have nowhere else to go, and a profile that quietly
            # dropped them would understate the price of batching.
            batch_seconds=gate_seconds / len(chosen),
            generation=generation,
            parent=parent_hash,
            parent_entry=parent_entry,
            config=config,
            archive=archive,
            weights=weights,
            judge_client=judge_client,
            preferences=preferences,
            report=report,
            origin=origin,
        )


def _produce(
    generation: int,
    config: LoopConfig,
    archive: Archive,
    rng: random.Random,
    proposer: Claude | None,
    parent_entry: Entry | None,
    parent_spec: dict[str, Any] | None,
    feedback: str,
    goal: str,
    report: Callable[[str], None],
) -> list[tuple[dict[str, Any], str | None, dict[str, Any]]]:
    """This generation's children: at most one model call, and free siblings.

    **Only the free operator batches**, and that is the cost structure rather than an
    omission. A model call costs more than the tournament that follows it, so a batch of
    model proposals would spend the expensive resource to save the cheap one -- the
    exact allocation `mechanical_rate` exists to avoid. A mechanical child costs a
    validation. So a generation that drew the mechanical operator makes ``batch`` of
    them and a generation that drew the model makes one proposal, and both then meet the
    same rungs in the same order.

    Duplicates are filtered here rather than downstream. A local edit can easily
    reproduce a game the archive already holds -- and within one batch, two draws can
    easily reproduce each other -- and a duplicate that reaches the gates costs a
    smoke test and a screen to discover what a hash lookup already knew.
    """
    produced: list[tuple[dict[str, Any], str | None, dict[str, Any]]] = []
    seen = set(archive.seen_hashes())

    def keep(spec: dict[str, Any], parent: str | None, origin: dict[str, Any]) -> bool:
        digest = spec_hash(spec)
        if digest in seen:
            archive.record(
                Entry(
                    hash=digest,
                    name=spec["name"],
                    generation=generation,
                    parent=parent,
                    status="duplicate",
                    notes="mechanically identical to an earlier candidate",
                    detail=dict(origin),
                ),
                spec=spec,
            )
            return False
        seen.add(digest)
        produced.append((spec, parent, origin))
        return True

    # The free operator first, where this generation drew it and there is a parent to
    # edit. It cannot invent a piece or a win condition -- that is what the expensive one
    # is for -- but it is the right instrument for a neighbourhood, and it leaves the
    # model's budget for the jumps only it can make.
    mechanical = parent_spec is not None and rng.random() < config.mechanical_rate
    if mechanical:
        assert parent_spec is not None
        for _ in range(config.batch):
            # The mechanical operator aims at *any* empty neighbouring cell rather than
            # at the one named for the model. "Anywhere new nearby" is the question the
            # archive is asking, and it is the one a local edit has a chance of
            # answering.
            mutation = mutate(
                parent_spec,
                rng,
                targets=_open_neighbours(archive, parent_entry),
                avoid=seen,
            )
            if mutation is None:
                break
            landed = (
                " -- and it reaches a kind of game the archive did not have"
                if descriptor(mutation.spec) != descriptor(parent_spec)
                else ""
            )
            if keep(
                mutation.spec,
                parent_entry.hash if parent_entry else None,
                {"operator": mutation.operator, "edit": mutation.summary},
            ):
                report(
                    f"  mechanical edit [{mutation.operator}]: {mutation.summary}{landed}"
                )

    if mechanical and produced:
        return produced
    if mechanical:
        report("  no local edit of this ruleset was valid; asking the model instead")

    if proposer is None:
        if not produced:
            report("  no model available and no local edit found")
        return produced

    proposal = propose_ruleset(
        client=proposer,
        parent=parent_spec,
        feedback=feedback,
        avoid=sorted(seen),
        goal=goal,
        # What was already tried from this parent and failed. The archive has always
        # kept these and the proposer has never seen them: ``avoid`` is a list of
        # hashes, which says that something was tried and nothing about what happened.
        failures=describe_failures(
            archive.failed_children(parent_entry.hash)
            if parent_entry
            else archive.failures(4)
        ),
    )

    if not proposal.accepted:
        report(f"  rejected by the validator after {proposal.attempts} attempts")
        archive.record(
            Entry(
                hash=f"invalid-{generation}",
                name="(invalid)",
                generation=generation,
                parent=proposal.parent_hash,
                status="invalid",
                notes="; ".join(proposal.errors[:6]),
            )
        )
        return produced

    spec = proposal.spec
    assert spec is not None
    if keep(spec, proposal.parent_hash, {"operator": "model"}):
        report(f"  proposed {spec['name']} [{spec_hash(spec)}]")
    return produced


def _promote(
    survivors: list[tuple[Gated, str | None, dict[str, Any]]],
    archive: Archive,
    rng: random.Random,
    config: LoopConfig,
    report: Callable[[str], None],
) -> list[tuple[Gated, str | None, dict[str, Any]]]:
    """Which screened children get a full roster, and on what grounds.

    **Not on the screen's numbers.** `loop/screen.py` says in its own docstring that it
    must not acquire an opinion about depth, elegance or variety; ranking a batch on its
    decisive rate or its mean length would give it one through the back door, and a
    second ranker with no calibration behind it is exactly what the pairwise judge
    replaced. Gates disqualify, they do not deduct.

    The grounds are coverage, which is what this search is actually for. A candidate's
    grid cell is spec-derived, free, and known before anything plays it, so a child that
    reaches a cell the archive has not filled goes first. Everything after that is
    uniform: the index cannot separate these candidates -- it has not measured them --
    and inventing an order would be pretending otherwise.
    """
    if len(survivors) <= config.promote:
        return survivors

    filled = {cell for cell in archive.grid().elites}
    fresh = [
        row for row in survivors if tuple(descriptor(row[0].spec)) not in filled
    ]
    rest = [row for row in survivors if row not in fresh]
    rng.shuffle(fresh)
    rng.shuffle(rest)
    chosen = (fresh + rest)[: config.promote]
    if fresh:
        report(
            f"  promoting {len(chosen)}: {min(len(fresh), config.promote)} reaching a "
            "cell the archive has not filled"
        )
    return chosen


def evaluate(
    spec: dict[str, Any],
    generation: int,
    parent: str | None,
    parent_entry: Entry | None,
    config: LoopConfig,
    archive: Archive,
    weights: Any,
    judge_client: Claude | None,
    preferences: PreferenceLog,
    report: Callable[[str], None],
    origin: dict[str, Any] | None = None,
) -> Entry:
    """Gate, play, analyse, score and record one candidate.

    Separate from proposing it so that a *starting point* can be measured by exactly the
    path a proposal is measured by. A run seeded with an existing ruleset that had been
    scored somewhere else would be comparing candidates against a number produced by a
    different roster, which is the one comparison the archive must never contain.
    """
    gated = run_gates(spec, config, report)
    if not gated.passed:
        return archive.record(
            _rejection(spec, generation, parent, gated, config, origin), spec=spec
        )
    return measure(
        spec=spec,
        gated=gated,
        generation=generation,
        parent=parent,
        parent_entry=parent_entry,
        config=config,
        archive=archive,
        weights=weights,
        judge_client=judge_client,
        preferences=preferences,
        report=report,
        origin=origin,
    )


@dataclass(frozen=True)
class Gated:
    """What the two cheap rungs said about one candidate.

    Split out of ``evaluate`` so a whole batch of children can be gated before any of
    them is given a tournament. That is the only structural change stage 9a needed: the
    gates were already cheap enough to run on a dozen candidates -- a smoke test and a
    screen together cost under two seconds against minutes for the rated roster -- and
    the loop was spending the expensive rung on whatever the single draw produced.
    """

    spec: dict[str, Any]
    smoke: Any
    screen: Any | None
    passed: bool
    # Seconds per rung, carried onto the archive entry. Free to record, and it is the
    # only way to answer "where did a run's time actually go" after the run is over --
    # which is the measurement stage 9b is required to take before it builds anything.
    seconds: dict[str, float] = field(default_factory=dict)

    @property
    def reasons(self) -> list[str]:
        if not self.smoke.passed:
            return list(self.smoke.reasons)
        return list(self.screen.reasons) if self.screen is not None else []


def run_gates(
    spec: dict[str, Any], config: LoopConfig, report: Callable[[str], None]
) -> Gated:
    """Smoke, then screen. Cheap enough to run on everything a generation produced."""
    name = spec.get("name", "?")
    started = time.perf_counter()
    smoke = smoke_test(
        spec,
        games=config.smoke_games,
        seed=config.seed,
        workers=config.workers,
    )
    seconds = {"smoke": time.perf_counter() - started}
    if not smoke.passed:
        report(f"  {name}: failed the smoke test:")
        for reason in smoke.reasons:
            report(f"    - {reason}")
        return Gated(
            spec=spec, smoke=smoke, screen=None, passed=False, seconds=seconds
        )

    # The middle rung. Random playouts cannot tell whether *searching* wins a game, and
    # the rated roster costs minutes to find out; eight games between two small budgets
    # answer it in under a second. Measured across the committed corpus it keeps every
    # real ruleset and rejects the deliberately shallow control.
    started = time.perf_counter()
    screened = screen(spec, seed=config.seed, workers=config.workers)
    seconds["screen"] = time.perf_counter() - started
    if not screened.passed:
        report(f"  {name}: failed the cheap screen:")
        for reason in screened.reasons:
            report(f"    - {reason}")
        return Gated(
            spec=spec, smoke=smoke, screen=screened, passed=False, seconds=seconds
        )

    report(f"  {name}: smoke {smoke.stats}; screen {screened.stats}")
    return Gated(
        spec=spec, smoke=smoke, screen=screened, passed=True, seconds=seconds
    )


def _rejection(
    spec: dict[str, Any],
    generation: int,
    parent: str | None,
    gated: Gated,
    config: LoopConfig,
    origin: dict[str, Any] | None,
) -> Entry:
    """One archive entry for a candidate the cheap rungs threw out.

    Recorded rather than dropped, and that matters more now than it did: a batch
    produces several rejects per generation, and every one of them is both a hash the
    loop must not pay to evaluate twice and a failure the proposer is shown next time.
    """
    rung = "the tournament" if gated.screen is None else "the full roster"
    detail: dict[str, Any] = {
        "smoke": gated.smoke.as_dict(),
        "theme": config.theme,
        "seconds": {k: round(v, 3) for k, v in gated.seconds.items()},
        **(origin or {}),
    }
    if gated.screen is not None:
        detail["screen"] = gated.screen.as_dict()
    return Entry(
        hash=spec_hash(spec),
        name=spec["name"],
        generation=generation,
        parent=parent,
        status="rejected",
        notes=f"Rejected before {rung}: " + "; ".join(gated.reasons),
        detail=detail,
    )


def measure(
    spec: dict[str, Any],
    gated: Gated,
    generation: int,
    parent: str | None,
    parent_entry: Entry | None,
    config: LoopConfig,
    archive: Archive,
    weights: Any,
    judge_client: Claude | None,
    preferences: PreferenceLog,
    report: Callable[[str], None],
    origin: dict[str, Any] | None = None,
    batch_seconds: float = 0.0,
) -> Entry:
    """The expensive rung: a full roster, the analysis pass, a score and a comparison."""
    digest = spec_hash(spec)
    screened = gated.screen
    assert screened is not None

    seconds = dict(gated.seconds)
    if batch_seconds:
        # This candidate's own two rungs are already in there; the rest of the batch's
        # is the price of having chosen it.
        seconds["batch"] = batch_seconds - seconds.get("smoke", 0.0) - seconds.get(
            "screen", 0.0
        )
    started = time.perf_counter()
    tournament = run_tournament(
        spec,
        TournamentConfig(
            agents=config.agents,
            games_per_pairing=config.games_per_pairing,
            seed=config.seed,
            workers=config.workers,
        ),
    )
    seconds["tournament"] = time.perf_counter() - started
    report(f"  {len(tournament.records)} games in {seconds['tournament']:.0f}s")

    # The second pass, before scoring rather than after. Three of the components the
    # current weights ask for -- decision_criticality, piece_utilisation,
    # blunder_recoverability -- are engine-derived and come from here, so a loop that
    # skipped it would be optimising a strict subset of the index the corpus was
    # calibrated against, and every candidate would be disqualified for a measurement
    # the run simply never took.
    started = time.perf_counter()
    analysis = analyse_corpus(
        spec,
        tournament.records,
        seed=config.seed,
        workers=config.workers,
    )
    analysis.save(config.directory / "tournaments" / digest / "analysis.json")
    seconds["analysis"] = time.perf_counter() - started

    started = time.perf_counter()
    result = score_ruleset(
        spec,
        tournament.records,
        tournament.ratings,
        weights=weights,
        analysis=analysis,
    )
    seconds["score"] = time.perf_counter() - started
    result.save(config.directory / "index" / f"{digest}.json")
    tournament.save(config.directory / "tournaments" / digest)

    report(f"  {result.summary()}")

    # A result that is not `scored` holds a 0.0 that stands for no total, so nothing of
    # it is kept as a number: the entry records no index, the comparison with the parent
    # is not made, and every reader of the archive shows *no score* for it.
    scored = result.scored
    separated = _separated_from_parent(result, parent_entry) if scored else None
    if parent_entry is not None:
        if not scored:
            report("  no score, so it is not compared with its parent")
        else:
            report(
                "  beats its parent by more than the noise in both"
                if separated
                else "  not separated from its parent: the difference is inside the interval"
            )

    entry = archive.record(
        Entry(
            hash=digest,
            name=spec["name"],
            generation=generation,
            parent=parent,
            status="evaluated",
            index=result.headline if scored else None,
            index_sd=result.sd,
            index_ci_low=result.interval["ci_low"] if result.interval else None,
            index_ci_high=result.interval["ci_high"] if result.interval else None,
            # The one tournament above, at the run's one seed: the number is one run's.
            seeds=[config.seed],
            separated=separated,
            objective=result.total if scored else None,

            viable=result.viable,
            weights_version=result.weights_version,
            cell=label(descriptor(spec)),
            notes=describe(result, spec),
            detail={
                "families": result.families,
                "scores": result.scores,
                "violations": result.violations,

                # What the run was told to explore, kept with the candidate rather than
                # only in the run's configuration: an archive read later should say why
                # this game exists, and "the loop was asked for siege games" is half of
                # that answer.
                "theme": config.theme,
                "screen": screened.as_dict(),
                # Which implementation played the games. Free to record, and it makes a
                # run's cost profile reconstructable afterwards -- the reference
                # interpreter is two orders of magnitude slower than the core, so a
                # candidate that lands on it costs a different amount to explore, and
                # nothing used to say which did.
                "backend": _backend(spec),
                # Where this candidate's time went, rung by rung. The archive is the
                # only record of it once a run is over, and "measure before building"
                # needs something to measure.
                "seconds": {k: round(v, 3) for k, v in seconds.items()},
                **(origin or {}),
            },
        ),
        spec=spec,
    )

    if judge_client is not None and result.viable:
        # Timed and reported, but deliberately *not* written onto the entry. The archive
        # is an append-only log and rewriting a line to add a number would break that
        # for a figure the profile does not need: judging is model latency and money,
        # not compute, and the question the profile exists to answer -- whether
        # population evaluation in the core is worth building -- is entirely about
        # compute. It happens after the entry is safely on disk, which is also worth
        # keeping.
        started = time.perf_counter()
        _compare_with_incumbents(
            spec, tournament.records, entry, config, archive, preferences,
            judge_client, report,
        )
        report(f"  judged in {time.perf_counter() - started:.0f}s")
    return entry


def _compare_with_incumbents(
    spec: dict[str, Any],
    records: list[Any],
    entry: Entry,
    config: LoopConfig,
    archive: Archive,
    preferences: PreferenceLog,
    judge_client: Claude,
    report: Callable[[str], None],
) -> None:
    """Ask the judge which of two games is better, against the top of the board.

    Only against a handful of incumbents: comparisons are O(n^2) if allowed to be, and
    each one is two model calls -- both orders, because a preference that flips when the
    pair is swapped is a non-result rather than a weak result.

    The ordering this builds is used as a tie-break and nothing else. It is not blended
    into a score, and the score is not shown to the judge, so the two pieces of evidence
    stay independent -- which is the only reason a tie-break from an opinion is worth
    having at all.
    """
    from chess_variant_sandbox.tournament.record import read_records

    # Drawn from the leaderboard rather than from the *tied* group. The tied group is
    # what the ordering is later used to break, so drawing the opponents from it would
    # mean no comparison is ever made in the one case that matters -- and none at all in
    # runs where the index does separate its candidates, leaving the ordering empty for
    # the first tie that arrives.
    opponents = [
        other
        for other in archive.leaderboard(config.top_k)
        if other.hash != entry.hash and not preferences.seen(entry.hash, other.hash)
    ][: config.comparisons]

    for other in opponents:
        other_spec = archive.load_spec(other.hash)
        games_path = config.directory / "tournaments" / other.hash / "games.jsonl"
        if other_spec is None or not games_path.exists():
            continue
        try:
            comparison = compare_rulesets(
                spec,
                other_spec,
                records,
                list(read_records(games_path)),
                client=judge_client,
            )
        except LLMUnavailable as error:
            report(f"  judge unavailable ({error}); no comparison made")
            return
        preferences.record(comparison)
        if comparison.winner is None:
            verdict = "no preference" + ("" if comparison.agreed else " (it flipped when swapped)")
        else:
            preferred = entry.name if comparison.winner == entry.hash else other.name
            verdict = f"prefers {preferred}"
        report(f"  judge vs {other.name}: {verdict}")


def _backend(spec: dict[str, Any]) -> str:
    """Which engine a ruleset runs on, by name rather than by timing it."""
    from chess_variant_sandbox.engine.bridge import rust_supports
    from chess_variant_sandbox.engine.compile import compile_spec

    try:
        return "reference" if rust_supports(compile_spec(spec)) else "core"
    except Exception:  # noqa: BLE001 - a spec that will not compile has no backend
        return "unknown"


def _require_measurable_ladder(agents: tuple[str, ...]) -> None:
    """Refuse a roster that cannot produce the loop's most important measurement.

    Strategic depth is read as Elo gained per doubling of the thinking budget, which
    needs at least two budgets of one agent kind. With a single budget the slope is
    silently zero, every candidate trips the depth gate, and the run reports that
    every game invented is shallow -- a statement about the roster, not the games.
    """
    budgets: dict[str, set[int]] = {}
    for agent in agents:
        match = re.search(r"nodes=(\d+)", agent)
        if match:
            budgets.setdefault(agent.split(":")[0], set()).add(int(match.group(1)))

    if not any(len(values) >= 2 for values in budgets.values()):
        raise ValueError(
            "the agent roster cannot measure strategic depth: it needs at least two "
            "node budgets of the same agent kind, for example "
            "'minimax:nodes=400' and 'minimax:nodes=1600'. "
            f"Got: {', '.join(agents)}"
        )


def _choose_parent(
    archive: Archive,
    config: LoopConfig,
    rng: random.Random,
    preferences: PreferenceLog | None = None,
    explore: bool = True,
) -> Entry | None:
    """Pick what to revise, or nothing to start fresh.

    ``explore`` is false where nothing can act on a fresh start -- a run with no model
    operator. The draw still happens either way, so which candidates a seed produces
    does not depend on whether the run has credentials.

    Uniform over the tied top group -- every viable candidate the leader does not beat
    by more than the noise in both.

    This used to weight by the *cube* of the index, which sharpened exactly the
    ordering the interval says is not real: the bootstrap sd of a score is 0.01-0.05
    against a human band 0.14 wide, so a cubed weighting mostly amplified sampling
    error and would have climbed it for as many generations as it was given.

    Be clear about what replaces it. At these interval widths almost nothing separates,
    so the tied group is usually the whole board and selection is close to uniform
    random over the viable set. That is not a loss -- it turns a fake optimiser into an
    honest random search over a filtered space, which has a real expected yield -- but
    direction does not come back until the judge supplies a tie-break the index cannot.
    Ties are broken uniformly rather than by any near-index quantity, because anything
    else smuggles the discredited ordering back in under a different name.
    """
    if rng.random() < config.explore_rate and explore:
        return None

    table = preferences.table() if preferences is not None else None

    # Uniform over the elites of the behaviour grid: one candidate parent per *kind* of
    # game, rather than the top of a ranked board. That is the quality-diversity move,
    # and it is what stops eighty generations being spent polishing one local optimum --
    # which is the failure a ranked board invites when the gaps between its rows are
    # smaller than their error bars.
    #
    # The index still decides, twice: it gates viability, and inside a cell it picks the
    # elite (see ``grid.build``). What it no longer does is order two games of different
    # kinds against each other, which is the comparison its interval cannot support.
    elites = list(archive.grid(prefer=table).elites.values())
    if elites:
        return rng.choice(elites)

    # Nothing has a cell yet -- an archive written before the grid existed. Fall back to
    # the tied top group, and let the judge's ordering break the tie where it is settled.
    group = archive.top_group(config.top_k)
    if not group:
        return None
    if table is not None and len(group) > 1:
        best = table.ranked([entry.hash for entry in group])[0]
        if all(best == entry.hash or table.beats(best, entry.hash) for entry in group):
            return next(entry for entry in group if entry.hash == best)
    return rng.choice(group)


def _open_neighbours(archive: Archive, parent: Entry | None) -> list[tuple[int, ...]]:
    """Empty cells one step from the parent's, for a mechanical edit to try to reach."""
    if parent is None or not parent.cell:
        return []
    from chess_variant_sandbox.loop.grid import parse

    return archive.grid().empty_neighbours(parse(parent.cell))


def _target_cell(
    archive: Archive,
    parent: Entry | None,
    config: LoopConfig,
    rng: random.Random,
) -> tuple[int, ...] | None:
    """An empty cell one step from the parent's, or ``None`` to leave the aim alone.

    One step on one axis, because a neighbour two changes away is a different game
    rather than a variation on this one, and the instruction has to be something a
    revision of *this* ruleset could plausibly satisfy.
    """
    if parent is None or not parent.cell or rng.random() >= config.target_rate:
        return None
    from chess_variant_sandbox.loop.grid import parse

    empty = archive.grid().empty_neighbours(parse(parent.cell))
    return rng.choice(empty) if empty else None


def _separated_from_parent(result: Any, parent: Entry | None) -> bool | None:
    """Did this candidate actually beat what it was revised from?

    ``None`` when there was no parent, which is different from ``False`` and has to stay
    different: a candidate proposed from scratch did not fail to improve on anything.
    """
    if parent is None or parent.index is None:
        return None
    if result.interval is None or parent.index_ci_high is None:
        return bool(result.headline > parent.index)
    return bool(result.headline > parent.index) and clears(
        float(result.interval["ci_low"]), float(parent.index_ci_high)
    )


def _focus_for(entry: Entry, weights: Any) -> str:
    families = entry.detail.get("families") if entry.detail else None
    advised = {name: value for name, value in (families or {}).items() if name in _FOCUS}
    if not advised:
        return ""
    weakest = min(advised, key=lambda name: advised[name])
    return _FOCUS[weakest]
