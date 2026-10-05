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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    store that `check_fit` has reproduced on the loaded core is this core's answer, so it
    is reused without the line.

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
        drifted = stale_verdict(stored)
        if drifted is not None and _loaded_core() not in verified_cores(stored, store):
            # stderr rather than a `report` callback: this is a warning about the thing
            # being returned rather than progress, every caller should see it, and the one
            # caller that has a reporter today (`corpus.measure`) would have to remember
            # to pass it. One line per measurement, not per game -- `resolve_fit` is the
            # per-game path and stays quiet.
            print(f"reusing {store_path(engine.spec.hash, procedure, store)}: {drifted}",
                  file=sys.stderr)
        return stored
    verdict = fit_ruleset(spec, workers=workers)
    verdict.save(store_path(engine.spec.hash, procedure, store))
    return verdict


def fit_ruleset(
    spec: dict[str, Any],
    workers: int = 0,
    verify: bool = True,
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

    score = head_to_head(
        spec,
        f"minimax:fit={procedure},nodes={ACCEPT_NODES}",
        f"minimax:nodes={ACCEPT_NODES}",
        ACCEPT_GAMES_PER_PAIRING,
        ACCEPT_SEED,
        workers,
    )
    # Strictly better. A fitted objective that draws with the one it replaces has not
    # earned a place in the roster, and "beats" is the word the bar was written in.
    final = _record(
        engine.spec.hash,
        procedure,
        vector,
        score > 0.5,
        score,
        2 * ACCEPT_GAMES_PER_PAIRING,
        len(samples),
        loss,
    )
    register_fit(final, replace=True)
    return final


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
# the method and its record. A stored fit names the core that made it and nothing else,
# so on any other core the question "does this verdict still hold?" has had one answer:
# re-derive it and compare. That costs a fit -- the regression is bit-deterministic in
# its rows, so what depends on the core is the fit games and the acceptance match, which
# is all of `fit_ruleset` -- and the saving is the *replay* a reproducing fit no longer
# owes, not the refit.
#
# What a check finds is written down so it is paid once per core:
#
# - **reproduces**: the store is untouched and the loaded core joins the sidecar's
#   `reproduced_on`. The store is not re-stamped: its `build` names the core whose games
#   the numbers in it came from, and a second core that agrees is a different fact.
# - **does not**: the new fit replaces the store, exactly as `fit --force` would, and the
#   sidecar, which vouched for a store that no longer exists, goes with it.
#
# The sidecar sits in a subdirectory because `fitted.load_all` globs `<store>/*.json`, and
# it is not a field of the store because `FittedObjective` lives in `agents/`, which was in
# `measurement.ENGINE_SOURCES` when this was written (since Q631 a field the core is not
# handed stales nothing; `Basis.players`).

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


def _loaded_core() -> str:
    from chess_variant_sandbox.measurement import built_fingerprint

    return built_fingerprint()


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
    """Where the cores a store was reproduced on are recorded, beside the store."""
    return Path(root) / REPRODUCED_DIR / f"{ruleset_hash}-{procedure}.json"


def reproduced_on(fit: FittedObjective, root: str | Path) -> tuple[str, ...]:
    """The cores a check found this store reproduces on, other than the one that made it.

    A sidecar written for a different store -- one since refit by hand, say -- vouches for
    nothing and reads as empty: the digest is what ties the two files together.
    """
    path = sidecar_path(fit.ruleset_hash, fit.procedure, root)
    if not path.exists():
        return ()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("store_digest") != store_digest(fit):
        return ()
    return tuple(str(core) for core in data.get("reproduced_on", ()))


def verified_cores(fit: FittedObjective, root: str | Path) -> frozenset[str]:
    """Every core this store is known to be the answer on: its maker and the reproduced.

    An unstamped store has no maker to count, and `fitted.stale_verdict`'s reading holds:
    absent is unknown, never agreement.
    """
    made = {fit.build} if fit.build else set()
    return frozenset(made | set(reproduced_on(fit, root)))


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


def check_fit(
    spec: dict[str, Any],
    store: str | Path | None = None,
    workers: int = 0,
) -> Reproduction:
    """Make this ruleset's stored fit the loaded core's answer, as cheaply as that can be.

    - Made on the loaded core, or already reproduced on it: nothing is played.
    - Otherwise the procedure runs on this core and meets Q231's three tests
      (`differences`). A fit that reproduces adds the core to the sidecar; one that does
      not is replaced by the new fit, as `fit --force` would.
    - No store at all: the ruleset is fitted, since a fresh fit is this core's answer by
      construction and the check's promise is only that the store is, on return.
    """
    store = fit_store() if store is None else store
    core = _loaded_core()
    ruleset_hash = Engine(spec).spec.hash
    procedure = procedure_digest()
    path = store_path(ruleset_hash, procedure, store)
    stored = load_fit(ruleset_hash, procedure, store)

    if stored is not None:
        register_fit(stored, replace=True)
        if stored.build == core:
            return Reproduction(MADE_HERE, core, path, stored, stored, None)
        if core in reproduced_on(stored, store):
            return Reproduction(
                REPRODUCED_BEFORE, core, path, stored, stored, None,
                sidecar=sidecar_path(ruleset_hash, procedure, store),
            )

    fresh = fit_ruleset(spec, workers=workers)
    if stored is None:
        _replace_store(fresh, path, store)
        return Reproduction(FITTED, core, path, fresh, None, fresh)

    failures = tuple(differences(stored, fresh))
    if failures:
        _replace_store(fresh, path, store)
        return Reproduction(REFIT, core, path, fresh, stored, fresh, failures)

    # The store stands, and the registry holds it rather than the fresh copy, whose score
    # may differ in a way the tests above deliberately allow.
    register_fit(stored, replace=True)
    sidecar = _add_core(stored, core, store)
    return Reproduction(REPRODUCED, core, path, stored, stored, fresh, sidecar=sidecar)


def _replace_store(fresh: FittedObjective, path: Path, store: str | Path) -> None:
    fresh.save(path)
    sidecar_path(fresh.ruleset_hash, fresh.procedure, store).unlink(missing_ok=True)


def _add_core(stored: FittedObjective, core: str, store: str | Path) -> Path:
    path = sidecar_path(stored.ruleset_hash, stored.procedure, store)
    cores = list(reproduced_on(stored, store))
    if core not in cores:
        cores.append(core)
    record = {
        "store_digest": store_digest(stored),
        "made_on": stored.build,
        "reproduced_on": cores,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    return path
