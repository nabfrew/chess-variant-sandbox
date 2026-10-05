"""A weight vector nobody wrote: fitted from games, by a procedure that is the identity.

`agents/heuristics.py` holds vectors somebody *stated* -- a name, a sentence and
fifteen numbers -- and the rule about them is absolute: a heuristic is per-ruleset by
construction, so the same digest is not the same player on two boards, so it must never
enter the rated roster. This module is the one narrow exception the project argues for,
and the argument turns on a distinction rather than on an exemption.

**What has to be comparable across rulesets is the player definition, not the vector.**

    `minimax:nodes=N`, weights fitted by procedure P from S games at seed 0

is a single deterministic, ruleset-independent, content-hashable definition, in exactly
the way `minimax:nodes=N` with a spec-derived evaluator already is. Two rulesets' fitted
players are the same player in the only sense the ladder needs, because the *procedure*
is what the name carries -- `fit=<procedure digest>`, identical on every board -- while
the vector it produces is derived from the rules the way the piece values already are.

The status quo is not a neutral alternative. A ladder that is competent on chess-likes
and incompetent on everything won by something other than capture is *itself* a rung
that means different things on different games; that is the defect
`docs/reference/ladder-strength.md` §6 measures. Whether a fitted ladder is more
comparable or less is an empirical question, and `accepted` below is what makes it
answerable with a measurement rather than with a position.

**A fit that did not work is refused by name.** Every store records the head-to-head
score of the fitted objective against the built-in one at equal budget, and a fit that
does not beat it falls back to the built-in objective and says so -- the way the
heuristic pass reports **unmeasured** rather than guessing. A silent bad fit is a
differently-blind ladder wearing a better name, which is worse than the blindness it
replaces because it is harder to see.

Three properties are carried over from `heuristics.py` unchanged, and for the same
reasons:

* **Resolution never defaults.** An unregistered `(procedure, ruleset)` raises. A silent
  fallback would produce a differently-playing agent under an unchanged name.
* **Scale is normalised away.** A search's argmax is invariant under multiplying every
  weight by a positive constant, so the vector is stored at unit length.
* **The store is a cache with a verified key.** A hand-edited file fails at load rather
  than quietly meaning something else.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_variant_sandbox.agents.features import VOCABULARY, VOCABULARY_VERSION
from chess_variant_sandbox.spec.canonical import HASH_LENGTH

STORE_DIR = "fitted"

# Where a fit is written when the caller did not say. Read at call time, never bound
# into a default argument, because the whole point is that it can be moved.
#
# **The store is tracked, and until Q91 the test suite wrote into it.** `ensure_fit` is
# the one writer with no path from its caller -- `corpus.measure` calls it, and
# `tests/test_corpus.py` measures a five-square fixture -- so a green run left a fit for
# a board no corpus has in a directory git watches, and `fitted/ee26d2322244-*.json` is
# one somebody then committed. A fit records what objective an opponent was given for
# one game, which makes its provenance the whole of its value; a store anything can
# write into by accident has no provenance. `tests/conftest.py` points this at a
# temporary directory for the session, and asserts at the end that the repository's
# stores did not move.
#
# Reads are deliberately *not* redirected. `load_fit(hash)` with no root still resolves
# `STORE_DIR`, so a test that asks whether chess has a committed fit still gets the
# committed answer rather than silently turning into a skip.
STORE_ENV = "CHESS_VARIANT_SANDBOX_FIT_STORE"


def fit_store() -> str:
    """The store `ensure_fit` writes to: `$CHESS_VARIANT_SANDBOX_FIT_STORE`, or `fitted/`."""
    return os.environ.get(STORE_ENV) or STORE_DIR

# ------------------------------------------------------------------ the procedure
#
# Everything below is frozen and hashed into the agent's name. Changing any of it is a
# different player, which is why it lives here as constants rather than as arguments: a
# procedure passed in at the call site is a procedure that can differ between two
# rulesets in the same run, and then the rung is not a rung.

# Bumped whenever the procedure changes in a way the constants below do not capture --
# the sampling rule, the loss, the perspective the features are taken from. The same
# kind of number as `SCHEMA_VERSION` and `VOCABULARY_VERSION`.
PROCEDURE_VERSION = 1

# The games the fit reads. Both orderings of both agents, so a seat-dependent bias
# cannot become a weight. `random` is deliberately absent: its games explore the state
# space and their *outcomes* are noise, and an outcome is the label here.
FIT_AGENTS: tuple[str, ...] = ("greedy", "minimax:nodes=400")
FIT_GAMES_PER_PAIRING = 12
FIT_SEED = 0

# Which plies become samples. The opening is skipped because every game shares it and
# its label is just the game's result, which would teach the regression the base rate
# and nothing else. The stride keeps consecutive, nearly identical positions from
# counting as independent evidence.
FIT_SKIP_PLIES = 6
FIT_STRIDE = 3

# Ridge, and it is doing two jobs. It keeps a separable sample from sending a weight to
# infinity, and it drives a term that is identically zero on this ruleset to *exactly*
# zero -- which is what the fixed-length vocabulary means by "absent", and what makes a
# fitted vector for chess and one for a territory game the same kind of object.
FIT_RIDGE = 1.0
FIT_ITERATIONS = 60
FIT_TOLERANCE = 1e-9

# The bar a fit has to clear, and it is a match rather than a loss number. A lower
# training loss is not evidence that a player is better; beating the objective it
# replaces, at the same budget, is.
#
# The budget is the top rated rung, because that is the rung every skill component
# reads. Both colours are played, so this is 48 games, whose binomial standard error is
# about 0.07 -- which is the honest resolution of the test and is reported beside every
# verdict rather than turned into a margin. A margin chosen here would be a margin
# chosen while looking at which rulesets it excludes.
ACCEPT_NODES = 6_400
ACCEPT_GAMES_PER_PAIRING = 24
ACCEPT_SEED = 0


class FitError(ValueError):
    """A fit, or a store of one, that cannot be trusted to name one player."""


def procedure_digest() -> str:
    """The content hash of the procedure, which is what the agent's name carries.

    Ruleset-independent by construction: nothing about any particular game enters it.
    That is the whole of 28.2 -- two rulesets' fitted players share this string, so the
    rating table sees one player and the ladder stays a ladder.

    **There is exactly one procedure, and adding a key to this blob moves the roster.**
    ``corpus.CANONICAL_AGENTS`` embeds this string, so a change here moves every stored
    ``Basis`` and forces a corpus replay and a weights re-anchor. Stage 30B added a
    second procedure by putting a key here *only when it was selected*, precisely so the
    default would not move; it lost its measurement and was deleted, and this is back to
    one branch. It hashes to ``98f172aeac49`` and ``tests/test_fitted.py`` pins it there
    so a future edit cannot move it by accident. It was ``16a4528b0d53`` until Q10 added
    ``facing_alignment`` and bumped ``VOCABULARY_VERSION``, which is a key here; the
    verdicts filed under the old digest are that procedure's evidence and stay readable
    beside the new ones, which is what two halves in ``store_path`` are for.
    """
    stated: dict[str, Any] = {
        "procedure_version": PROCEDURE_VERSION,
        "vocabulary_version": VOCABULARY_VERSION,
        "agents": list(FIT_AGENTS),
        "games_per_pairing": FIT_GAMES_PER_PAIRING,
        "seed": FIT_SEED,
        "skip_plies": FIT_SKIP_PLIES,
        "stride": FIT_STRIDE,
        "ridge": FIT_RIDGE,
        "iterations": FIT_ITERATIONS,
        "tolerance": FIT_TOLERANCE,
        "accept": [ACCEPT_NODES, ACCEPT_GAMES_PER_PAIRING, ACCEPT_SEED],
    }
    blob = json.dumps(stated, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:HASH_LENGTH]


# ------------------------------------------------------------------ the regression


def logistic_fit(
    samples: list[tuple[tuple[float, ...], float]],
    ridge: float = FIT_RIDGE,
    iterations: int = FIT_ITERATIONS,
    tolerance: float = FIT_TOLERANCE,
) -> tuple[tuple[float, ...], float, float]:
    """Texel-style: regress the game's result on the features of its positions.

    Returns ``(weights, intercept, mean loss)``. Convex, so there is one answer and the
    solver's only job is to find it; Newton with a backtracking line search does that in
    a handful of iterations on sixteen parameters.

    **The intercept is fitted and then discarded.** A constant added to every move's
    score cannot change an argmax, so it is not part of the player -- but leaving it out
    of the *fit* would make the regression spend a real feature on the base rate, which
    on a first-player-advantaged game is a term with nothing to do with the position.

    ``y`` is the result from one fixed side's point of view and a draw is ``0.5``. Cross
    entropy takes a fractional target without special-casing, which is the right reading:
    a draw is evidence for neither side and the loss says exactly that.
    """
    if not samples:
        raise FitError("a fit needs positions; none were sampled")

    width = len(samples[0][0])
    size = width + 1  # the intercept rides in the last slot
    weights = [0.0] * size

    def loss(w: list[float]) -> float:
        total = 0.0
        for features, y in samples:
            z = _dot(w, features)
            # log(1 + exp(z)) written so neither tail overflows.
            softplus = z + math.log1p(math.exp(-z)) if z > 0 else math.log1p(math.exp(z))
            total += softplus - y * z
        # `fsum`: the line search compares these, and float `sum` is version-dependent.
        penalty = 0.5 * ridge * math.fsum(v * v for v in w[:width])
        return (total + penalty) / len(samples)

    current = loss(weights)
    for _ in range(iterations):
        gradient = [0.0] * size
        hessian = [[0.0] * size for _ in range(size)]
        for features, y in samples:
            z = _dot(weights, features)
            p = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
            residual = p - y
            variance = p * (1.0 - p)
            for i in range(size):
                xi = features[i] if i < width else 1.0
                if xi == 0.0:
                    continue
                gradient[i] += residual * xi
                row = hessian[i]
                for j in range(i, size):
                    xj = features[j] if j < width else 1.0
                    row[j] += variance * xi * xj
        for i in range(width):
            gradient[i] += ridge * weights[i]
            hessian[i][i] += ridge
        # A curvature floor on the intercept only: every other diagonal already carries
        # the ridge, and a sample with no variation would otherwise leave the system
        # singular in exactly one direction.
        hessian[width][width] += 1e-9
        for i in range(size):
            for j in range(i):
                hessian[i][j] = hessian[j][i]

        step = _solve(hessian, gradient)
        if step is None:
            break

        # Backtracking. Newton on a convex problem overshoots only when the quadratic
        # model is poor, and halving until the loss falls is the whole of the fix.
        scale = 1.0
        improved = False
        for _ in range(30):
            trial = [weights[i] - scale * step[i] for i in range(size)]
            value = loss(trial)
            if value <= current:
                moved = max(abs(scale * s) for s in step)
                weights, current, improved = trial, value, True
                break
            scale *= 0.5
        if not improved or moved < tolerance:
            break

    return tuple(weights[:width]), weights[width], current


def _dot(w: list[float], features: tuple[float, ...]) -> float:
    total = w[len(features)]  # the intercept
    for i, x in enumerate(features):
        if x != 0.0:
            total += w[i] * x
    return total


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting. ``None`` when the system is singular.

    Sixteen by sixteen, once per Newton step, in a project with no dependencies. A
    singular system is not an error here -- it means the sample says nothing about some
    direction -- so it stops the iteration rather than raising.
    """
    size = len(vector)
    rows = [list(matrix[i]) + [vector[i]] for i in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(rows[r][column]))
        if abs(rows[pivot][column]) < 1e-12:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        for below in range(column + 1, size):
            factor = rows[below][column] / scale
            if factor == 0.0:
                continue
            for k in range(column, size + 1):
                rows[below][k] -= factor * rows[column][k]
    out = [0.0] * size
    for column in reversed(range(size)):
        total = rows[column][size]
        for k in range(column + 1, size):
            total -= rows[column][k] * out[k]
        out[column] = total / rows[column][column]
    return out


def unit(weights: tuple[float, ...]) -> tuple[float, ...]:
    """The vector at unit length. Identity is direction; scale is not part of a player.

    ``math.fsum``, as in ``features.builtin_vector``: a stored vector must not depend on
    whether CPython's float ``sum`` is compensated (3.12 on) or naive (3.11) (Q448).
    """
    norm = math.sqrt(math.fsum(w * w for w in weights))
    if norm == 0.0:
        return weights
    return tuple(round(w / norm, 6) + 0.0 for w in weights)


# ------------------------------------------------------------------ the store


@dataclass(frozen=True)
class FittedObjective:
    """One ruleset's fitted vector, and everything needed to distrust it.

    ``accepted`` is the whole of 28.3. A fit that does not beat the built-in objective
    at equal budget is kept, reported and *not played*: ``weights`` resolves to the empty
    tuple, which is the signal every agent already understands as "use the built-in
    evaluator". The vector stays in the file so a failure can be read rather than
    inferred from an absence.
    """

    ruleset_hash: str
    procedure: str
    vector: tuple[float, ...]
    accepted: bool
    accept_score: float
    samples: int
    loss: float
    accept_games: int = 0
    # The core this verdict was made on: `_core.BUILD_FINGERPRINT` at fit time.
    #
    # Neither the procedure digest nor the ruleset hash covers the engine, and the engine
    # decides the games -- twice over here, which is more than it decides for a searched
    # valuation. The vector is regressed from records the core played, and the acceptance
    # match that turned it into a *verdict* is played by the core as well. So without this
    # a store is a set of numbers with no build behind them, which is `measurement.Basis`'s
    # own argument one level down and `Valuation.build`'s verbatim.
    #
    # It was added after three separate `rust/src/` changes landed in one day underneath a
    # store that could not say which of them it predated. `Valuation.build` was added for
    # the same reason after `main` moved 289 lines of `game.rs` mid-stage, and there the
    # only thing that separated "the verdicts still hold" from "a bug in the checker says
    # they do not" was somebody finding the bug. This field is what makes the question
    # answerable without re-playing anything.
    #
    # **Reported, never enforced, and never silently re-derived.** `ensure_fit` reuses a
    # stored verdict on a mismatch: refusing one would break every command that touches an
    # older ruleset, and re-fitting on its own would spend minutes nobody asked for. What a
    # mismatch invalidates is the accept/refuse -- which decides whether the roster plays
    # this vector or the built-in objective under an agent string that does not change
    # either way -- so saying so is the honest response. `stale_verdict` is the sentence.
    # A *measurement* does not reuse silently since Q80d: `corpus.refresh` re-derives a
    # fit this core has not made or reproduced before it trusts the games it played.
    #
    # Empty means a store written before this field existed, and reads as *unknown* rather
    # than as agreement -- the safe direction, exactly as an absent `Basis` is stale.
    # Nothing on disk was back-stamped: a fingerprint written by a later hand is
    # indistinguishable to every future reader from an invented one, which is the rule
    # `CLAUDE.md` already states about hand-writing a `canonical.json`.
    build: str = ""
    procedure_version: int = PROCEDURE_VERSION
    vocabulary_version: int = VOCABULARY_VERSION

    def __post_init__(self) -> None:
        if len(self.vector) != len(VOCABULARY):
            raise FitError(
                f"a fit under vocabulary {VOCABULARY_VERSION} has {len(VOCABULARY)} "
                f"weights, got {len(self.vector)}"
            )
        for index, weight in enumerate(self.vector):
            if not math.isfinite(weight):
                raise FitError(f"{VOCABULARY[index]} is not a finite weight")

    @property
    def weights(self) -> tuple[float, ...]:
        """What the agent actually plays: the vector, or the built-in objective."""
        return self.vector if self.accepted else ()

    @property
    def accept_error(self) -> float:
        """The standard error of the acceptance score, so a reader can see the noise.

        A verdict is binary because the bar is binary, and this is what says which
        verdicts the sample could actually resolve. Not used to decide anything: a
        margin applied here would be a threshold chosen after seeing which rulesets it
        excludes, which is the mistake the mutant corpus already caught once.
        """
        if self.accept_games <= 0:
            return 0.0
        return math.sqrt(0.25 / self.accept_games)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ruleset_hash": self.ruleset_hash,
            "procedure": self.procedure,
            "procedure_version": self.procedure_version,
            "vocabulary_version": self.vocabulary_version,
            # Written out redundantly so a stale file diagnoses itself rather than
            # silently meaning something else.
            "vocabulary": list(VOCABULARY),
            "accepted": self.accepted,
            "accept_score": self.accept_score,
            "accept_games": self.accept_games,
            "build": self.build,
            "samples": self.samples,
            "loss": self.loss,
            "weights": list(self.vector),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FittedObjective:
        version = int(data.get("vocabulary_version", -1))
        if version != VOCABULARY_VERSION:
            raise FitError(
                f"this fit was made against vocabulary {version} and this is "
                f"{VOCABULARY_VERSION}; refit rather than reusing it"
            )
        stored_vocabulary = data.get("vocabulary")
        if stored_vocabulary is not None and tuple(stored_vocabulary) != VOCABULARY:
            raise FitError(
                "the stored vocabulary does not match this one, though the versions "
                "agree; one of them was edited without a bump"
            )
        procedure_version = int(data.get("procedure_version", -1))
        if procedure_version != PROCEDURE_VERSION:
            raise FitError(
                f"this fit was made by procedure version {procedure_version} and this "
                f"is {PROCEDURE_VERSION}; refit rather than reusing it"
            )
        return cls(
            ruleset_hash=str(data["ruleset_hash"]),
            procedure=str(data["procedure"]),
            vector=tuple(float(w) for w in data["weights"]),
            accepted=bool(data["accepted"]),
            accept_score=float(data.get("accept_score", 0.0)),
            accept_games=int(data.get("accept_games", 0)),
            # Absent is *unknown*, never the current build. Defaulting it to what is
            # loaded now would let exactly the stores this field exists to distinguish
            # pass as agreeing with the engine that is about to play them.
            build=str(data.get("build", "")),
            samples=int(data.get("samples", 0)),
            loss=float(data.get("loss", 0.0)),
            procedure_version=procedure_version,
            vocabulary_version=version,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.as_dict(), handle, indent=2)
            handle.write("\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> FittedObjective:
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


# Keyed by `(procedure, ruleset_hash)`, module level, mirroring `heuristics._HEURISTICS`.
# Two keys rather than one because the procedure is the half the agent's name carries and
# the ruleset is the half it deliberately does not.
_FITTED: dict[tuple[str, str], FittedObjective] = {}


def register_fit(fit: FittedObjective, replace: bool = False) -> None:
    """Make a fit resolvable.

    ``replace`` exists for one caller and it is not a convenience. The acceptance test
    plays the fitted agent *by its real name*, which means the fit has to be resolvable
    before its verdict is known -- so the fitter registers a provisional entry, plays the
    match, and registers the verdict over it. Testing a different spelling of the same
    vector would leave the resolution path itself untested, which is where a fitted
    player is most likely to be quietly wrong.
    """
    key = (fit.procedure, fit.ruleset_hash)
    if replace:
        _FITTED[key] = fit
    else:
        _FITTED.setdefault(key, fit)


def registered_fits() -> dict[tuple[str, str], FittedObjective]:
    return dict(_FITTED)


def normalise_procedure(digest: Any) -> str:
    """A procedure digest as it came back out of an agent string.

    ``parse_agent`` coerces an option value to an ``int`` when it looks like one, so an
    all-digit digest arrives with its leading zeros gone. Same repair as
    ``heuristics.normalise_digest``, and the same reason it cannot be skipped: one digest
    in sixteen starts with a zero.
    """
    return str(digest).zfill(HASH_LENGTH)


def resolve_fit(procedure: Any, ruleset_hash: str) -> FittedObjective:
    """The fit a `(procedure, ruleset)` pair names, or an error naming what is missing."""
    procedure = normalise_procedure(procedure)
    if not ruleset_hash:
        raise FitError(
            f"a fitted agent needs to know which ruleset it is playing; procedure "
            f"{procedure!r} was resolved without one. This is a caller that has not "
            f"passed its ruleset hash through, not a missing file."
        )
    fit = _FITTED.get((procedure, ruleset_hash))
    if fit is None:
        # Read the committed store before giving up. Not a default -- a missing file
        # still raises below -- but the store is the fit's home, and requiring every
        # caller to load it first would be a rule the analysis pass cannot keep: it
        # spawns fresh processes that import this module from disk and inherit no
        # registry, so a game with a fitted agent in it would resolve in the parent and
        # fail in the child.
        fit = load_fit(ruleset_hash, procedure)
    if fit is None:
        raise FitError(
            f"no fit for procedure {procedure!r} on ruleset {ruleset_hash!r} is loaded; "
            f"run `chess-variant-sandbox fit` for that ruleset. A game played by a fitted agent is "
            f"reproducible only with its store in hand, exactly as a ruleset hash needs "
            f"rulesets/"
        )
    return fit


def store_path(ruleset_hash: str, procedure: str, root: str | Path = STORE_DIR) -> Path:
    """Where one ruleset's fit under one procedure lives.

    Both halves are in the filename because a procedure bump must not overwrite the
    evidence of what the previous one produced -- a campaign's numbers are only readable
    beside the fit that played them.
    """
    return Path(root) / f"{ruleset_hash}-{procedure}.json"


def load_fit(
    ruleset_hash: str, procedure: str | None = None, root: str | Path = STORE_DIR
) -> FittedObjective | None:
    """The stored fit for a ruleset, registered and ready, or ``None`` if there is none.

    Absent is reported as absent. A missing store means the fit has not been run for this
    ruleset; it is neither an error nor a verdict, and it becomes an error only when an
    agent that needs it is asked to play.
    """
    procedure = procedure or procedure_digest()
    path = store_path(ruleset_hash, procedure, root)
    if not path.exists():
        return None
    fit = FittedObjective.load(path)
    if fit.procedure != procedure or fit.ruleset_hash != ruleset_hash:
        raise FitError(
            f"{path} holds a fit for {fit.ruleset_hash}/{fit.procedure} and is filed "
            f"under {ruleset_hash}/{procedure}; the file has been renamed"
        )
    register_fit(fit)
    return fit


def load_all(root: str | Path = STORE_DIR) -> int:
    """Register every fit on disk that this procedure can read. Returns how many.

    A file written by a different procedure or vocabulary is skipped rather than
    refused: an old store sitting beside a new one is the normal state of a bumped
    procedure, and it is exactly what makes the previous campaign's numbers still
    readable.
    """
    directory = Path(root)
    if not directory.is_dir():
        return 0
    loaded = 0
    for path in sorted(directory.glob("*.json")):
        try:
            fit = FittedObjective.load(path)
        except FitError:
            continue
        register_fit(fit)
        loaded += 1
    return loaded


def stale_verdict(fit: FittedObjective) -> str | None:
    """Why this fit's verdict describes a different engine, if it does.

    Reported, never enforced, and the wording of `valuations.stale_verdict` because it is
    the same argument. A stored vector is still a legal player and still the vector the
    regression found; what a mismatch invalidates is the **verdict** -- the head-to-head
    that decided whether the roster plays it or falls back to the built-in objective --
    so the honest response is to say so and let a caller decide whether to pay for a
    refit.

    **The core is not a `measurement.Basis` field; the player is** (Q80c,
    `docs/reference/refit-repair.md`). `Basis.fit` records which objective the roster
    played -- the built-in, or a digest of the procedure and the vector -- and a store
    neither made nor reproduced on the loaded core reads `stale (refit)`. The repair is
    the fourth, `refit`, which `corpus.refresh` runs first: `tournament.fitting.check_fit`
    re-derives the fit and either records the core in the store's sidecar or replaces the
    store, and only a changed player costs a replay. This sentence stays for `fit` and
    `ensure_fit`, which still reuse a store and say so.
    """
    from chess_variant_sandbox.measurement import built_fingerprint

    if not fit.build:
        return (
            "this fit records no core build, so which engine played the games it was "
            "regressed from and the match that accepted it is unknown; refit to find out"
        )
    current = built_fingerprint()
    if fit.build == current:
        return None
    return (
        f"this fit was made on core {fit.build} and the loaded one is {current}; the "
        "engine decides the games, so the accept/refuse verdict may no longer hold"
    )


def fitted_weights(options: dict[str, Any], ruleset_hash: str) -> tuple[float, ...]:
    """The weight vector an agent's ``fit=`` option names, or ``()`` for the built-in.

    An empty tuple is the signal that no fitted objective is in play, which is what keeps
    every committed number where it is -- and it is also what a *refused* fit resolves
    to, so a fitted agent on a ruleset the fit failed on is the built-in agent, by name
    and in the record.
    """
    procedure = options.get("fit")
    if procedure is None:
        return ()
    return resolve_fit(procedure, ruleset_hash).weights
