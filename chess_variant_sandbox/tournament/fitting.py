"""Running the fit: play the games, harvest the positions, regress, and check.

`agents/fitted.py` holds the procedure's constants, the solver and the store; this is
the pass that executes it, and it lives under `tournament/` for the same reason the
analysis pass does -- it plays games, and `agents/` must not.

Four steps, and the fourth is the one that makes the whole thing arguable:

1. **Play `S` games** on a fixed cheap roster at seed 0. Fixed in
   ``agents/fitted.FIT_AGENTS`` and hashed into the procedure digest, so "the games the
   fit read" is part of the player's definition rather than of whoever ran it.
2. **Harvest `(features, result)`** from every sampled ply. Texel's trick: the label is
   the *game's* outcome, so the regression learns which positional facts precede
   winning rather than which ones some evaluator liked.
3. **Regress.** Convex, cheap, no model calls, and it needs no new games beyond step 1.
4. **Check it against the objective it replaces**, at equal budget, and refuse it by
   name if it loses. A fit that does not beat the built-in evaluator is a failed fit and
   falls back to it -- recorded, not hidden.

**The check may be curtailed, exactly, and only when asked** (``curtail=True``,
``docs/reference/compute-economy.md`` §4.2). ``acceptance`` then reads the match's games one at a time
in a fixed order of the schedule and stops at the first prefix after which every result of
the unplayed games gives the same verdict (``duel.settled``). The verdict is the full
match's by construction; the stored score and ``accept_games`` are the prefix's, so the
store says "decided after ``accept_games`` of ``2 x ACCEPT_GAMES_PER_PAIRING``". It is off
by default because the sliced measurement (``slices.gather_verdict``) plays all 48 and the
site holds a visitor's sliced fit to the server's field by field, score and game count
included: a curtailed store would be a different file from the one that path derives.

**Why fitting rather than proposing.** Stage 23 asked a model for weight vectors over
this same vocabulary. They cleared the random-vector null and lost to a free hand-written
list, and they were near-random where it counted: mobility at 0.667 of material where the
built-in objective uses 0.029. That is evidence that *proposing* weights is the wrong
operation, not that the vocabulary is the wrong substrate. Fitting is the operation the
substrate was always for, and it costs no model call.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_variant_sandbox.agents import native_spec
from chess_variant_sandbox.agents.features import VOCABULARY, FeatureEvaluator, feature_context
from chess_variant_sandbox.agents.fitted import (
    ACCEPT_GAMES_PER_PAIRING,
    ACCEPT_NODES,
    ACCEPT_SEED,
    FIT_AGENTS,
    FIT_GAMES_PER_PAIRING,
    FIT_SEED,
    FIT_SKIP_PLIES,
    FIT_STRIDE,
    FittedObjective,
    fit_store,
    load_fit,
    logistic_fit,
    procedure_digest,
    register_fit,
    stale_verdict,
    store_path,
    unit,
)
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.tournament import runner
from chess_variant_sandbox.tournament.duel import settled
from chess_variant_sandbox.tournament.heuristics import head_to_head
from chess_variant_sandbox.tournament.play import replay
from chess_variant_sandbox.tournament.record import GameRecord
from chess_variant_sandbox.tournament.runner import TournamentConfig, run_tournament

Sample = tuple[tuple[float, ...], float]


def harvest(engine: Engine, records: list[GameRecord]) -> list[Sample]:
    """Positions and the result of the game they came from, from seat 0's point of view.

    **One fixed perspective, not the side to move's.** Every term in the vocabulary is
    already signed from ``player``'s view, so reading them all from seat 0 and labelling
    with seat 0's result makes the sign of a weight mean one thing across the whole
    sample. Taking the mover's view instead would fold "whose turn is it" into every
    term, and the fit would spend weight on tempo.

    The opening is skipped and the rest is strided. Both are in the procedure digest:
    the first because every game of a ruleset shares its opening and its label is just
    the game's result, so those rows teach the base rate; the second because two
    consecutive plies are nearly the same position with the same label and counting them
    twice overstates how much evidence there is.
    """
    context = feature_context(engine.spec)
    # The weights are irrelevant to `features`, which reads every term whatever they
    # say; ones rather than zeros only because the constructor is entitled to assume a
    # vector that names a player.
    reader = FeatureEvaluator(context, (1.0,) * len(VOCABULARY))
    samples: list[Sample] = []
    for record in records:
        label = record.score_for(0)
        states = replay(engine, record)
        # The terminal position is dropped: it has no legal moves, so `mobility` reads
        # zero for a reason that is about the game being over rather than about the
        # position, and every finished game would push the same row.
        for ply in range(FIT_SKIP_PLIES, len(states) - 1, FIT_STRIDE):
            state = states[ply]
            move_count = len(engine.legal_moves(state))
            samples.append((reader.features(engine, state, 0, move_count), label))
    return samples


def ensure_fit(
    spec: dict[str, Any],
    store: str | Path | None = None,
    workers: int = 0,
) -> FittedObjective:
    """This ruleset's verdict under the current procedure, fitting it if there is none.

    The store is the fit's home and a stored verdict is never re-derived: the procedure
    is deterministic in the spec, so re-running it would produce the same vector at the
    cost of the acceptance match. What this adds over `fit_ruleset` is that a caller who
    only needs *a* verdict -- `corpus.measure`, before it plays a candidate on a roster
    that carries `fit=` -- does not have to know whether one exists yet.

    **"Deterministic in the spec" is true of the procedure and not of the engine**, which
    is what the reuse used to assume without saying. The vector is regressed from games
    the core played and the acceptance match is played by the core, so a stored verdict
    made under a `rust/src/` that has since moved is a verdict about a different set of
    games -- and the roster then plays a vector accepted by an engine that no longer
    exists, under an agent string that says nothing about it.

    So the reuse stays and the silence goes. A mismatch is written to stderr once, here,
    where the decision to reuse is actually taken; refusing would break every command
    that touches an older ruleset, and refitting unasked would spend minutes on a caller
    that only wanted a number. `FittedObjective.build` argues both halves at length. A
    store made or reproduced under the whole `reproduction_basis` -- this core, and this
    game's tables, players and runner as they are now -- is this core's answer, so it is
    reused without the line; any other gets `unverified_reason`'s, a store made on this
    core under tables that have since moved included (Q892).

    **This is the only writer whose caller does not name a path**, which is why the
    default is resolved here rather than bound into the signature: `corpus.measure` calls
    it with a spec and nothing else, so whatever `fit_store()` says is where a fit made
    on the way to a measurement lands. `fitted.STORE_ENV` says why that matters.
    """
    store = fit_store() if store is None else store
    engine = Engine(spec)
    procedure = procedure_digest()
    stored = load_fit(engine.spec.hash, procedure, store)
    if stored is not None:
        register_fit(stored, replace=True)
        drifted = unverified_reason(stored, store, spec)
        if drifted is not None:
            # stderr rather than a `report` callback: this is a warning about the thing
            # being returned rather than progress, every caller should see it, and the one
            # caller that has a reporter today (`corpus.measure`) would have to remember
            # to pass it. One line per measurement, not per game -- `resolve_fit` is the
            # per-game path and stays quiet.
            print(f"reusing {store_path(engine.spec.hash, procedure, store)}: {drifted}",
                  file=sys.stderr)
        return stored
    verdict = fit_ruleset(spec, workers=workers)
    save_fit(verdict, spec, store)
    return verdict


def fit_ruleset(
    spec: dict[str, Any],
    workers: int = 0,
    verify: bool = True,
    curtail: bool = False,
) -> FittedObjective:
    """Fit this ruleset's objective, check it, and return the verdict.

    Deterministic in ``spec`` alone: the roster, the game count, the seed, the sampling
    and the solver are all constants of the procedure. Two machines running this produce
    the same vector, and that is what makes a fitted player reproducible from the
    repository rather than from somebody's cache.

    ``verify=False`` skips the acceptance match and marks the fit accepted. It exists
    for tests and for the one legitimate case -- inspecting a vector before paying for
    the match -- and a store written that way is a store whose central claim is
    unchecked, which is why the flag is not reachable from the CLI.

    ``curtail`` stops the acceptance match once its verdict is settled (``acceptance``):
    the same verdict, the prefix's score, and ``accept_games`` the games it was decided
    after. The procedure digest does not move, since the verdict function is the same.
    """
    engine = Engine(spec)
    procedure = procedure_digest()

    result = run_tournament(
        spec,
        TournamentConfig(
            agents=FIT_AGENTS,
            games_per_pairing=FIT_GAMES_PER_PAIRING,
            seed=FIT_SEED,
            workers=workers,
            collect_series=False,
        ),
    )
    samples = harvest(engine, result.records)

    # A ruleset whose games end inside the opening skip yields no rows at all --
    # `mutant-cramped-board` and `mutant-gardner-uniform` are both over in fewer than
    # `FIT_SKIP_PLIES` plies. That is a fit that cannot be *made*, which is a different
    # thing from one that was made and lost, and both are refusals: the store records an
    # empty vector, `resolve_fit(...).weights` returns nothing, and the agent plays the
    # built-in objective. Raising instead would mean a roster carrying `fit=` could not
    # play a corpus that contains one such entry, which is how a measurement stops being
    # about the corpus and starts being about which entries the tooling can reach.
    if not samples:
        return _record(
            engine.spec.hash, procedure, (0.0,) * len(VOCABULARY), False, 0.0, 0, 0, 0.0
        )

    weights, _intercept, loss = logistic_fit(samples)
    vector = unit(weights)

    # A vector of zeros is a player that scores every move identically and therefore
    # moves at random. That is a legitimate null control and never a fit, so it is
    # refused here rather than reaching the match -- where it would lose, correctly, but
    # for a reason the number could not tell you.
    if not any(vector):
        return _record(
            engine.spec.hash, procedure, vector, False, 0.0, 0, len(samples), loss
        )

    provisional = _record(
        engine.spec.hash, procedure, vector, True, 0.0, 0, len(samples), loss
    )
    register_fit(provisional, replace=True)
    if not verify:
        return provisional

    match = acceptance(
        spec,
        f"minimax:fit={procedure},nodes={ACCEPT_NODES}",
        f"minimax:nodes={ACCEPT_NODES}",
        ACCEPT_GAMES_PER_PAIRING,
        ACCEPT_SEED,
        workers,
        curtail=curtail,
    )
    final = _record(
        engine.spec.hash,
        procedure,
        vector,
        match.accepted,
        match.score,
        match.decided_after,
        len(samples),
        loss,
    )
    register_fit(final, replace=True)
    return final


# ------------------------------------------------------------------ the acceptance match


def beats(scores: Sequence[float]) -> bool:
    """The acceptance bar over the match's games, each 0, 1/2 or 1 for the candidate.

    Strictly better. A fitted objective that draws with the one it replaces has not earned
    a place in the roster, and "beats" is the word the bar was written in. The score is
    ``head_to_head``'s, rounding included, so the full match is one computation either way.
    """
    return _match_score(scores) > 0.5


def _match_score(scores: Sequence[float]) -> float:
    return round(sum(scores) / len(scores), 4) if scores else 0.5


@dataclass(frozen=True)
class Acceptance:
    """The acceptance match's verdict, its score, and how many of its games decided it.

    ``score`` is over the first ``decided_after`` games of the reading order, all
    ``planned`` of them unless the match was curtailed. A curtailed score never sits on the
    other side of the bar from the verdict: an accept is decided only once more than half
    the planned points are won, which is more than half the prefix's, and a refusal only
    once at least half are lost.
    """

    accepted: bool
    score: float
    decided_after: int
    planned: int
    # The kept games, in reading order. Out of equality: a record carries its wall time.
    records: tuple[GameRecord, ...] = field(default=(), compare=False)


PlayGames = Callable[[list[tuple[int, str, str]]], list[GameRecord]]
"""A block of schedule entries ``(seed, first, second)`` to their records, in order."""


def reading_order(tasks: Sequence[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
    """The order a curtailed match reads its schedule in: the two colours alternately.

    The schedule plays every game with the candidate first, then every game with it
    second. Read in that order, any prefix shorter than half the match is one colour's, and
    its score would carry the first move's advantage. Alternating the halves keeps every
    even prefix balanced. It is a fixed permutation of the schedule, so the stopping point
    still depends on nothing but the schedule.
    """
    half, odd = divmod(len(tasks), 2)
    if odd:
        raise ValueError(f"an acceptance match of {len(tasks)} games is not both colours")
    order: list[tuple[int, str, str]] = []
    for first, second in zip(tasks[:half], tasks[half:], strict=True):
        order.extend((first, second))
    return order


def acceptance(
    spec: dict[str, Any],
    candidate: str,
    reference: str,
    games_per_pairing: int,
    seed: int,
    workers: int = 0,
    curtail: bool = False,
    play: PlayGames | None = None,
) -> Acceptance:
    """``candidate`` against ``reference`` at both colours, and whether it ``beats`` it.

    Uncurtailed this is ``head_to_head``, unchanged. Curtailed (reference/compute-economy.md §4.2), the same
    schedule is read in ``reading_order`` and played a batch of the worker count at a time
    (by ``play``, or the runner); after each batch the games are read one by one and the
    first prefix ``settled`` answers is the decision, the rest of that batch discarded. So
    the verdict is the full match's whatever the unplayed games would have been, and the
    stopping point, the kept games and the score are functions of the schedule alone, the
    same at any worker count.

    ``settled`` was written for pairs, scored in quarters, and serves single games as it
    stands: a game's 0, 1/2 or 1 is a whole number of quarters, and ``beats`` reads only
    the sum, so it meets both of ``duel.Verdict``'s conditions. The quarter completions
    ``settled`` tries include every game completion, among them the all-lost and all-won
    ones, and a verdict monotone in the sum agrees with every completion as soon as it
    agrees with those two; so its answer is exactly the games' answer (held to brute force
    over every game completion in ``tests/test_fit_curtail.py``).
    """
    if not curtail:
        score = head_to_head(spec, candidate, reference, games_per_pairing, seed, workers)
        planned = 2 * games_per_pairing
        return Acceptance(score > 0.5, score, planned, planned)

    config = TournamentConfig(
        agents=(candidate, reference),
        games_per_pairing=games_per_pairing,
        seed=seed,
        workers=workers,
        include_self_play=False,
        collect_series=False,
    )
    order = reading_order(runner.schedule(config))
    batch = config.resolved_workers()
    if play is None:
        engine = Engine(spec)
        natives = [native_spec(agent, engine.spec.hash) for agent in config.agents]

        def standard(block: list[tuple[int, str, str]]) -> list[GameRecord]:
            # The runner's own player over a block of its schedule, as `checkpoint` reads
            # it: `runner.py` is in the engine fingerprint set, so it is read, never edited.
            return runner._play(engine, config, block, natives, False)

        play = standard

    scores: list[float] = []
    kept: list[GameRecord] = []
    while True:
        block = order[len(scores):len(scores) + batch]
        records = play(block)
        if [(record.seed, *record.agents) for record in records] != block:
            raise ValueError("a batch of the acceptance match came back as other games")
        for record in records:
            scores.append(record.score_for(0 if record.agents[0] == candidate else 1))
            kept.append(record)
            call = settled(scores, len(order), beats)
            if call is not None:
                return Acceptance(
                    call, _match_score(scores), len(scores), len(order), tuple(kept)
                )


def _record(
    ruleset_hash: str,
    procedure: str,
    vector: tuple[float, ...],
    accepted: bool,
    score: float,
    games: int,
    samples: int,
    loss: float,
) -> FittedObjective:
    """One store entry, so the provisional and the final one cannot drift apart."""
    from chess_variant_sandbox.measurement import built_fingerprint

    return FittedObjective(
        ruleset_hash=ruleset_hash,
        procedure=procedure,
        vector=vector,
        accepted=accepted,
        accept_score=round(score, 4),
        accept_games=games,
        # Stamped on the one constructor rather than at the top of the pass, so it names
        # the core that played the acceptance match -- which is the number a reader is
        # trusting -- and so the refusals that never reach a match carry it too.
        # `valuing._entry` does exactly this and for the same reason.
        build=built_fingerprint(),
        samples=samples,
        loss=round(loss, 6),
    )


# ------------------------------------------------------------------ reproduction (Q80b)
#
# `docs/reference/refit-repair.md` is the design, and `fit-reproduction.md` (Q231, Q270)
# the method and its record. A store says which core made it and nothing about what else
# its games read, so on any other footing the question "does this verdict still hold?"
# has one answer: re-derive it and compare. That costs a fit -- the regression is
# bit-deterministic in its rows, so what can move is the fit games and the acceptance
# match, which is all of `fit_ruleset` -- and the saving is the *replay* a reproducing fit
# no longer owes, not the refit.
#
# **What a fit's games read** (Q892) is `reproduction_basis`: the core, this game's
# compiled tables, what the runner hands the core for the fit's players, and the runner's
# own source. Each can move without the others -- a compiler edit moves one game's tables
# and no core (Q880 moved *quarry*'s, `docs/reference/replay-2026-10-13b.md`), an edit to
# `agents/evaluation.py` moves the piece values the core is handed -- so a sidecar vouches
# for the whole basis, and a store counts as verified on a core only where an entry for
# that core matches every other part now. Until Q892 it recorded cores alone, and a
# compiler-only change left *quarry*'s store unrederived until the next core arrived.
#
# What a check finds is written down so it is paid once per basis:
#
# - **reproduces**: the store is untouched and the basis joins the sidecar's
#   `verified_under` (and the core its `reproduced_on`, the history). The store is not
#   re-stamped: its `build` names the core whose games the numbers in it came from, and a
#   second core that agrees is a different fact.
# - **does not**: the new fit replaces the store, exactly as `fit --force` would, and the
#   sidecar, which vouched for a store that no longer exists, is replaced by one naming
#   the new store's maker. So is the sidecar of any store a writer here makes.
#
# **A sidecar written before Q892 vouches for nothing**, maker included: it has no
# `verified_under`, and neither it nor the store records the tables a check ran on. Nor
# can `runs/` prove them: a replay's stamp names the tables it was played on, but its
# `_fit_once` played nothing where the core was already listed -- which is exactly the
# *quarry* case. So every such store is checked once, and `reproduced_on` is kept as the
# record it is.
#
# The sidecar sits in a subdirectory because `fitted.load_all` globs `<store>/*.json`, and
# it is not a field of the store because `FittedObjective` lives in `agents/`, which was in
# `measurement.ENGINE_SOURCES` when this was written (since Q631 a field the core is not
# handed stales nothing; `Basis.players`), and because a new field would move every
# store's digest and the site's field-by-field check of a visitor's sliced fit.

# Q231's pre-registered tolerance. The same rows through the same solver give
# bit-identical weights (Q109 G2, Q129), so it absorbs float noise and never a changed game.
VECTOR_TOLERANCE = 1e-9

REPRODUCED_DIR = "reproduced"

# What a check did, in the order `check_fit` tries them.
MADE_HERE = "made here"
REPRODUCED_BEFORE = "reproduced before"
REPRODUCED = "reproduced"
REFIT = "refit"
FITTED = "fitted"

# The players whose native specs the fit's games are played with: the fit roster and the
# acceptance match's reference. The candidate is the store's own vector, which the
# sidecar's `store_digest` already ties it to.
REPRODUCTION_PLAYERS: tuple[str, ...] = (*FIT_AGENTS, f"minimax:nodes={ACCEPT_NODES}")

# The parts of `reproduction_basis`, in the order a sentence names them.
BASIS_PARTS = ("core", "tables", "players", "runner")


def _loaded_core() -> str:
    from chess_variant_sandbox.measurement import built_fingerprint

    return built_fingerprint()


def reproduction_basis(spec: dict[str, Any]) -> dict[str, str]:
    """Everything a fit of this ruleset's games reads that can move on its own.

    - ``core``: the loaded build, which walks the tables and plays every seat.
    - ``tables``: `measurement.tables_fingerprint`, what the compiler made of this game.
    - ``players``: `measurement.players_fingerprint` over `REPRODUCTION_PLAYERS`, the
      native specs and piece values the runner hands the core. Where it cannot say, the
      agents' source stands in, as it does for `Basis.players`.
    - ``runner``: the `engine` source set (`tournament/runner.py`, which fixes every
      seed, and `play.py`), the half of `Basis.engine` that is not the core.

    The harvest's own reading of a position is not a separate part: `agents/features.py`
    is the oracle `eval.rs` is held to term for term (`tests/test_backends.py`), so a term
    that reads differently moves the core with it. The regression is pure arithmetic over
    the rows and the procedure digest pins its constants.
    """
    from chess_variant_sandbox.measurement import (
        players_fingerprint,
        source_fingerprint,
        tables_fingerprint,
    )

    players = players_fingerprint(spec, REPRODUCTION_PLAYERS)
    return {
        "core": _loaded_core(),
        "tables": tables_fingerprint(spec),
        "players": players or "agents:" + source_fingerprint("agents"),
        "runner": source_fingerprint("engine"),
    }


def store_digest(fit: FittedObjective) -> str:
    """The digest of everything a store says, which is what a sidecar vouches for.

    Over the store's canonical JSON rather than the file's bytes, so a checkout that
    rewrites line endings does not orphan a sidecar. Every field is in it -- the core that
    made it, the score, the loss -- because the claim "this store reproduces on core X" is
    about that file, and a refit that changes any of it is a different file.
    """
    canonical = json.dumps(fit.as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def sidecar_path(ruleset_hash: str, procedure: str, root: str | Path) -> Path:
    """Where the bases a store was made or reproduced under are recorded, beside it."""
    return Path(root) / REPRODUCED_DIR / f"{ruleset_hash}-{procedure}.json"


def _sidecar(fit: FittedObjective, root: str | Path) -> dict[str, Any]:
    """The sidecar's record for this store, or an empty one if it vouches for another.

    A sidecar written for a different store -- one since refit by hand, say -- vouches for
    nothing: the digest is what ties the two files together.
    """
    path = sidecar_path(fit.ruleset_hash, fit.procedure, root)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("store_digest") != store_digest(fit):
        return {}
    return data


def reproduced_on(fit: FittedObjective, root: str | Path) -> tuple[str, ...]:
    """The cores a check re-derived this store on, other than its maker: the history.

    Not a verification. An entry written before Q892 says only that the store reproduced
    on that core at the tables of the day; `verified_cores` is what a reader trusts.
    """
    return tuple(str(core) for core in _sidecar(fit, root).get("reproduced_on", ()))


def verified_under(fit: FittedObjective, root: str | Path) -> tuple[dict[str, str], ...]:
    """Every basis this store was made or reproduced under, as the sidecar records them.

    Empty for a sidecar written before Q892, which records cores and nothing else.
    """
    found = _sidecar(fit, root).get("verified_under", ())
    return tuple(
        {part: str(entry.get(part, "")) for part in BASIS_PARTS}
        for entry in found
        if isinstance(entry, dict)
    )


def _matches(entry: dict[str, str], basis: dict[str, str]) -> bool:
    """Every part but the core agrees: the store is that core's answer on this footing."""
    return all(entry[part] == basis[part] for part in BASIS_PARTS if part != "core")


def verified_cores(
    fit: FittedObjective, root: str | Path, spec: dict[str, Any] | None
) -> frozenset[str]:
    """Every core this store is known to be the answer on, under this game as it is now.

    A core counts where the sidecar records a basis for it whose tables, players and runner
    are today's: the maker's, written by whoever made the store here, or a check's. A store
    whose sidecar predates Q892, or that has none, is verified nowhere -- absent is unknown,
    never agreement, as `fitted.stale_verdict` reads an unstamped store. ``spec`` ``None``
    cannot compare the tables and answers the same, the safe direction
    `corpus.current_basis` takes for its own tables.
    """
    entries = verified_under(fit, root)
    if spec is None or not entries:
        return frozenset()
    basis = reproduction_basis(spec)
    return frozenset(entry["core"] for entry in entries if _matches(entry, basis))


def _is_verified(fit: FittedObjective, root: str | Path, basis: dict[str, str]) -> bool:
    return any(entry == basis for entry in verified_under(fit, root))


def basis_gap(
    fit: FittedObjective, root: str | Path, basis: dict[str, str]
) -> tuple[bool, tuple[str, ...]]:
    """What keeps this store from being verified under ``basis`` (Q894), as parts to name.

    ``(False, ())`` where the sidecar records no basis at all -- a store saved before
    Q892, by hand, or with no sidecar -- so nothing says what its games read. Otherwise
    ``(True, parts)``: the `BASIS_PARTS` where the newest entry in `verified_under` differs
    from ``basis``, in `BASIS_PARTS` order. Empty parts with a record means the store is
    verified (or an older entry is), which callers have already ruled out. The sentences
    that name these are written where they are shown, `measurement.Basis.stale_reasons` and
    `cli._reproduction`; the parts are only ever read from here.
    """
    entries = verified_under(fit, root)
    if not entries:
        return False, ()
    newest = entries[-1]
    return True, tuple(part for part in BASIS_PARTS if newest[part] != basis[part])


def unverified_reason(
    fit: FittedObjective, root: str | Path, spec: dict[str, Any]
) -> str | None:
    """Why this store may not be the loaded core's answer for this game now, if it may not.

    `fitted.stale_verdict`'s sentence where the core moved; otherwise the parts of the
    basis that differ from the newest one recorded on this core, or that none is.
    """
    basis = reproduction_basis(spec)
    if _is_verified(fit, root, basis):
        return None
    drifted = stale_verdict(fit)
    if drifted is not None:
        return drifted
    here = [entry for entry in verified_under(fit, root) if entry["core"] == basis["core"]]
    if not here:
        return (
            "this fit was made on the loaded core, but nothing records the tables, players "
            "and runner its games read; `fit --check` re-derives it"
        )
    moved = [part for part in BASIS_PARTS if here[-1][part] != basis[part]]
    return (
        f"this fit was made or reproduced on the loaded core under other {', '.join(moved)}, "
        "so the verdict may no longer hold; `fit --check` re-derives it"
    )


def record_made(fit: FittedObjective, spec: dict[str, Any], root: str | Path) -> Path:
    """Write the sidecar a store just made here deserves: its maker's whole basis.

    Called by every writer in this module and by `chess-variant-sandbox fit`, right after
    the store is saved, so a store made on the loaded core is verified without being made
    twice. Any older sidecar vouched for the store this one replaces and goes.
    """
    return _write_sidecar(fit, root, [], [reproduction_basis(spec)])


def save_fit(fit: FittedObjective, spec: dict[str, Any], root: str | Path) -> Path:
    """Save a store made here and record its maker's basis beside it."""
    path = store_path(fit.ruleset_hash, fit.procedure, root)
    fit.save(path)
    record_made(fit, spec, root)
    return path


def differences(stored: FittedObjective, fresh: FittedObjective) -> list[str]:
    """Q231's three tests, each failure as a sentence; empty when the fit reproduces.

    Samples, vector and verdict. The accept score is not one of them: a score that moves
    without crossing the bar leaves the roster playing the same objective, so it still
    reproduces (`fit-reproduction.md`, "What reproduces means").
    """
    failures = []
    if stored.samples != fresh.samples:
        failures.append(f"samples {stored.samples} -> {fresh.samples}")
    moved = max(
        (abs(a - b) for a, b in zip(stored.vector, fresh.vector, strict=True)),
        default=0.0,
    )
    if moved > VECTOR_TOLERANCE:
        distance = math.dist(stored.vector, fresh.vector)
        failures.append(
            f"vector moves {distance:.4g} (largest component {moved:.4g})"
        )
    if stored.accepted != fresh.accepted:
        failures.append(
            f"verdict {_verdict(stored)} -> {_verdict(fresh)}"
        )
    return failures


def _verdict(fit: FittedObjective) -> str:
    return f"{'accepted' if fit.accepted else 'refused'} {fit.accept_score:.4f}"


@dataclass(frozen=True)
class Reproduction:
    """What `check_fit` found and did.

    ``fit`` is the verdict in the store afterwards, which is what the roster will play;
    ``stored`` the one it found (``None`` if there was none) and ``fresh`` the one it
    derived (``None`` when it played nothing).
    """

    outcome: str
    core: str
    path: Path
    fit: FittedObjective
    stored: FittedObjective | None
    fresh: FittedObjective | None
    failures: tuple[str, ...] = ()
    sidecar: Path | None = None
    # What kept the stored fit from being verified on the loaded core (`basis_gap`),
    # read before anything was played: ``recorded`` False where the sidecar names no
    # basis, else ``differs`` the parts of the newest one that are not today's.
    recorded: bool = True
    differs: tuple[str, ...] = ()


def check_fit(
    spec: dict[str, Any],
    store: str | Path | None = None,
    workers: int = 0,
) -> Reproduction:
    """Make this ruleset's stored fit the loaded core's answer, as cheaply as that can be.

    - Already made or reproduced under the whole `reproduction_basis` -- the loaded core,
      this game's tables, the players and the runner -- nothing is played.
    - Otherwise the procedure runs here and meets Q231's three tests (`differences`). A fit
      that reproduces adds the basis to the sidecar; one that does not is replaced by the
      new fit, as `fit --force` would, with its maker's basis beside it.
    - No store at all: the ruleset is fitted, since a fresh fit is this core's answer by
      construction and the check's promise is only that the store is, on return.
    """
    store = fit_store() if store is None else store
    basis = reproduction_basis(spec)
    core = basis["core"]
    ruleset_hash = Engine(spec).spec.hash
    procedure = procedure_digest()
    path = store_path(ruleset_hash, procedure, store)
    stored = load_fit(ruleset_hash, procedure, store)

    if stored is not None:
        register_fit(stored, replace=True)
        if _is_verified(stored, store, basis):
            sidecar = sidecar_path(ruleset_hash, procedure, store)
            if stored.build == core:
                return Reproduction(MADE_HERE, core, path, stored, stored, None, sidecar=sidecar)
            return Reproduction(
                REPRODUCED_BEFORE, core, path, stored, stored, None, sidecar=sidecar
            )

    recorded, differs = (True, ()) if stored is None else basis_gap(stored, store, basis)
    fresh = fit_ruleset(spec, workers=workers)
    if stored is None:
        _replace_store(fresh, path, store, basis)
        return Reproduction(FITTED, core, path, fresh, None, fresh)

    failures = tuple(differences(stored, fresh))
    if failures:
        _replace_store(fresh, path, store, basis)
        return Reproduction(
            REFIT, core, path, fresh, stored, fresh, failures,
            recorded=recorded, differs=differs,
        )

    # The store stands, and the registry holds it rather than the fresh copy, whose score
    # may differ in a way the tests above deliberately allow.
    register_fit(stored, replace=True)
    sidecar = _add_basis(stored, basis, store)
    return Reproduction(
        REPRODUCED, core, path, stored, stored, fresh, sidecar=sidecar,
        recorded=recorded, differs=differs,
    )


def _replace_store(
    fresh: FittedObjective, path: Path, store: str | Path, basis: dict[str, str]
) -> None:
    fresh.save(path)
    _write_sidecar(fresh, store, [], [basis])


def _add_basis(stored: FittedObjective, basis: dict[str, str], store: str | Path) -> Path:
    cores = list(reproduced_on(stored, store))
    if basis["core"] != stored.build and basis["core"] not in cores:
        cores.append(basis["core"])
    entries = [entry for entry in verified_under(stored, store) if entry != basis]
    return _write_sidecar(stored, store, cores, [*entries, basis])


def _write_sidecar(
    fit: FittedObjective,
    store: str | Path,
    cores: list[str],
    entries: list[dict[str, str]],
) -> Path:
    path = sidecar_path(fit.ruleset_hash, fit.procedure, store)
    record = {
        "store_digest": store_digest(fit),
        "made_on": fit.build,
        "reproduced_on": cores,
        "verified_under": [{part: entry[part] for part in BASIS_PARTS} for entry in entries],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    return path
