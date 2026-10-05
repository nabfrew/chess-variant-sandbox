"""What the pieces are worth, searched rather than derived, fitted or remembered.

Three ways of getting a valuation have now been tried in this project and this is the
fourth, so it is worth being precise about what makes it different.

* **Derived** -- average reach on an empty board, blended with the declared value. A fact
  about the rules, cheap, and it cannot flatter a ruleset it has seen. It is also
  sometimes wrong by a lot: xiangqi's table prices the region-locked elephant at a sixth
  of a soldier and runs from 1.25 to 74.
* **Declared** -- what the spec says. An unvalidated opinion, used at ``DECLARED_WEIGHT``
  as a weak prior and never alone.
* **Fitted** -- stage 30B, a Texel-style regression of the game's *result* on the
  position. Deleted. It got better at predicting the games and worse at playing them:
  training loss improved on 37 of 41 rulesets while play fell on 27.
* **Searched** -- this. Hill-climb the table against the only thing anybody actually
  wants, which is *playing strength against the table it replaces*, one piece at a time.

The fourth is the third's post-mortem taken seriously. What went wrong there was the
objective, not the parameterisation, so this optimises the objective directly; and what
made that impossible before was that a 48-game win rate cannot resolve the 0.02 steps a
hill-climb walks through, so this runs against ``tournament/duel.py`` instead, whose
error is *measured* and which cancels outright everything two nearly-identical players
agree about.

**Identity is the procedure, exactly as it is for a fit.** ``values=<digest>`` names how
the table was found, not the table; the digest is the same string on every board and the
table it produces is per-ruleset, which is the distinction ``agents/fitted.py`` argues at
length and this module inherits whole. The store is keyed by the *search* half of the
procedure alone, so a later change to how a valuation is used cannot invalidate the
expensive part.

**A search that does not beat the derived table is refused by name**, and the refusal is
stored. ``Valuation.values`` returns the empty tuple for a refused search, which is the
signal every agent already reads as "use the table the rules imply". Stage 30B's whole
value was that its failure was legible; a search that quietly shipped a worse table would
be a differently-blind player wearing a better name.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_variant_sandbox.agents.features import VOCABULARY
from chess_variant_sandbox.spec.canonical import HASH_LENGTH
from chess_variant_sandbox.spec.schema import MAX_PIECE_TYPES

STORE_DIR = "valuations"

# Bumped when the search changes in a way the constants below do not capture -- the
# order pieces are visited in, what counts as an improvement, the shape of a step. The
# same kind of number as ``fitted.PROCEDURE_VERSION``.
PROCEDURE_VERSION = 1


class ValuationError(ValueError):
    """A valuation, or a store of one, that cannot be trusted to name one player."""


@dataclass(frozen=True)
class SearchProcedure:
    """Everything about how a table is found. Hashed, and it is the store's key.

    Frozen constants rather than call-site arguments, for the reason ``fitted.py`` gives:
    a procedure that can differ between two rulesets in one run is not a procedure, and
    the player it names is not one player.
    """

    version: int = PROCEDURE_VERSION
    # The budget the search optimises *for*. A valuation is not budget-independent --
    # `runs/stage30/pilots.txt` measured the sign of the hand-versus-derived effect
    # flipping between 400 and 25,600 nodes on xiangqi -- so the budget is part of the
    # procedure rather than of whoever ran it.
    nodes: int = 6_400
    # Seed-pairs per comparison, so 96 games. Twice what the fit's acceptance test spends
    # on a single verdict, which is exactly the ceiling Stage A's exit set for buying
    # resolution, and it is spent because the pairing did not buy as much as that exit
    # assumed: the measured standard error of one comparison is about 0.054 at 24 pairs
    # and about 0.039 at 48.
    pairs: int = 48
    # The seed the whole search runs on. Fixed, deliberately: holding the randomness
    # still across every comparison is what "common random numbers" means, and it is
    # what makes two candidates differing in one piece comparable at all. The price is
    # that the search can fit this seed set, which is why the acceptance match uses
    # another one.
    seed: int = 0
    # Multiplicative steps, coarse first. A step is a ratio rather than an increment
    # because the derived tables span two orders of magnitude within one ruleset, so a
    # fixed increment would be a rounding error on the chariot and a doubling on the
    # soldier.
    steps: tuple[float, ...] = (1.6, 1.25)
    # Passes over the pieces at one step size before giving up on it.
    sweeps: int = 3
    # A ceiling on the whole search, so a ruleset with many piece types cannot cost an
    # unbounded number of games. Reaching it is recorded rather than hidden.
    max_comparisons: int = 120
    # What counts as an improvement, in units of the comparison's own measured standard
    # error.
    #
    # Two, not one, and the reason is a measurement rather than caution. A search of this
    # shape runs sixty to a hundred comparisons, so a one-sigma bar accepts a false step
    # roughly one time in six *per comparison* and the incumbent walks uphill on noise.
    # Two sigma at 48 pairs is a bar of about 0.078, and the per-piece effects that
    # actually exist on the ruleset this was built for are 0.10 to 0.36 -- so the bar
    # separates them by a wide margin rather than by a hair. On a ruleset where no such
    # effect exists it accepts nothing, which is the right answer and is what chess gets.
    accept_sigma: float = 2.0

    # The acceptance match, against the derived table, at the search budget.
    #
    # **A different seed from the search.** The search holds one seed set fixed on
    # purpose and can therefore fit it, so a verdict taken on that same set would be the
    # training score. This is the held-out one, and it is the number every claim about
    # this stage rests on.
    accept_nodes: int = 6_400
    accept_pairs: int = 48
    accept_seed: int = 90_001

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "nodes": self.nodes,
            "pairs": self.pairs,
            "seed": self.seed,
            "steps": list(self.steps),
            "sweeps": self.sweeps,
            "max_comparisons": self.max_comparisons,
            "accept_sigma": self.accept_sigma,
            "accept": [self.accept_nodes, self.accept_pairs, self.accept_seed],
        }

    @property
    def digest(self) -> str:
        """The content hash of the search. Ruleset-independent by construction."""
        blob = json.dumps(
            self.as_dict(), separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:HASH_LENGTH]


# The one search anybody runs. A second one is a second constant here, never an argument
# threaded through the pass -- and `tests/test_valuations.py` pins this digest so a
# future edit cannot move the store's key by accident.
SEARCH = SearchProcedure()


@dataclass(frozen=True)
class Valuation:
    """One ruleset's searched table, and everything needed to distrust it.

    ``accepted`` carries the same weight it does on a fit. A table that does not beat the
    derived one at equal budget is kept, reported and *not played*: ``values`` resolves
    to the empty tuple, which every agent already reads as the derived table. The vector
    stays in the file so a failure can be read rather than inferred from an absence.
    """

    ruleset_hash: str
    procedure: str
    # Positional, and meaningless without its labels -- which is why they are stored
    # beside it, exactly as `HeuristicSet` stores them.
    piece_ids: tuple[str, ...]
    vector: tuple[float, ...]
    # What the search started from, on the same scale. Kept because every claim here is
    # relative to it and a reader should not have to recompile the ruleset to see it.
    derived: tuple[float, ...]
    # The weight vector this player carries, which is `features.builtin_vector` for this
    # ruleset and is *not* a choice this stage made.
    #
    # It has to be stored, and the reason is a design decision in the core rather than a
    # convenience. `eval::Evaluator` reads the derived table and only the derived table
    # -- `Objective::new` sends an empty weight vector there and the boundary refuses
    # values without weights, deliberately, so that "the built-in objective" always means
    # the arithmetic every committed number came from. A player with an opinion about the
    # pieces is therefore a `FeatureEvaluator`, and the vector that makes one behave like
    # the built-in objective is ruleset-dependent: it multiplies back by the opening
    # material and the square count. Resolution holds a ruleset *hash* and cannot
    # recompute it, so the search writes down what it played.
    #
    # It cannot drift, because it is a pure function of the compiled spec and the spec is
    # what the hash covers.
    weights: tuple[float, ...]
    accepted: bool
    accept_score: float
    accept_error: float
    accept_games: int
    # The same table against the *ladder* agent -- `minimax:nodes=N`, the player every
    # other number in the project is about -- rather than against the derived table on
    # this player's own code path.
    #
    # Two numbers because they answer two questions and only the first is attributable.
    # `accept_score` differs from its reference in the valuation and in nothing else, so
    # a win there is a win for the values. This one additionally crosses from the feature
    # evaluator to the built-in one, whose decisions `builtin_vector` reproduces but
    # whose bits it does not, so a difference here is the values *plus* that. Gating on
    # the second would be gating on two things at once; reporting only the first would
    # hide whether the thing anybody would actually play got better.
    ladder_score: float = 0.5
    ladder_error: float = 0.0
    # How much the search cost and how far it got, so "no improvement found" and "ran out
    # of comparisons" are different readings of the same refusal.
    comparisons: int = 0
    accepted_steps: int = 0
    exhausted: bool = False
    # The core this verdict was measured on: `_core.BUILD_FINGERPRINT` at search time.
    #
    # Neither the procedure digest nor the ruleset hash covers the engine, and the engine
    # decides the games -- so without this a store is a set of numbers with no way to say
    # which build produced them. That is `measurement.Basis`'s own argument one level down.
    #
    # It was added after `main` changed 289 lines of `game.rs` mid-stage and there was no
    # way to tell, from the store alone, whether the verdicts in it still described the
    # engine. Re-playing them said yes; a bug in the *checker* briefly said otherwise, and
    # the only thing that separated those two readings was somebody finding the bug. This
    # field is what makes the question answerable without re-playing anything.
    #
    # Empty means a store written before this field existed, which is reported as unknown
    # rather than as agreement -- the safe direction, exactly as an absent `Basis` is
    # treated as stale.
    build: str = ""
    # Plies completed per search, measured at a few budgets by `_depth_reached`'s probe.
    # Nothing in this stage reads it; it is what a depth-aware valuation would need, and
    # measuring it while the games are already being played costs a fraction of one game.
    plies: dict[str, float] = field(default_factory=dict)
    procedure_version: int = PROCEDURE_VERSION

    def __post_init__(self) -> None:
        if len(self.vector) != len(self.derived):
            raise ValuationError(
                f"a searched valuation has {len(self.vector)} entries and the derived "
                f"table it replaces has {len(self.derived)}; one of them is for a "
                "different game"
            )
        if len(self.vector) > MAX_PIECE_TYPES:
            raise ValuationError(
                f"a valuation has at most {MAX_PIECE_TYPES} entries, which is the "
                f"validator's own cap on piece types, got {len(self.vector)}"
            )
        if self.piece_ids and len(self.piece_ids) != len(self.vector):
            raise ValuationError(
                f"a valuation prices {len(self.vector)} types and names "
                f"{len(self.piece_ids)}; the labels and the numbers disagree"
            )
        for index, value in enumerate(self.vector):
            if not math.isfinite(value):
                raise ValuationError(f"piece {index} is not worth a finite amount")

    @property
    def values(self) -> tuple[float, ...]:
        """What the agent actually plays: the searched table, or the derived one."""
        return self.vector if self.accepted else ()

    @property
    def playing_weights(self) -> tuple[float, ...]:
        """The weights the agent actually plays, and empty for a refused search.

        Both halves fall back together, which is the only fallback that is really a
        fallback: an empty weight vector *and* an empty value vector is `minimax:nodes=N`
        to the bit, while the built-in vector with no values is a player whose decisions
        merely agree with it. A refused search has to leave the ordinary agent behind,
        not a close relative of one.
        """
        return self.weights if self.accepted else ()

    @property
    def ratios(self) -> tuple[float, ...]:
        """The searched table as a multiple of the derived one, which is what moved.

        Absolute values on the derived scale are hard to read across rulesets -- xiangqi's
        run to 74 -- and the ratio is what the search actually chose. A derived entry of
        zero is a royal slot, which the search never touches, and reads 1.0.
        """
        return tuple(
            (value / base) if base else 1.0
            for value, base in zip(self.vector, self.derived, strict=True)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ruleset_hash": self.ruleset_hash,
            "procedure": self.procedure,
            "procedure_version": self.procedure_version,
            "piece_ids": list(self.piece_ids),
            "accepted": self.accepted,
            "accept_score": self.accept_score,
            "accept_error": self.accept_error,
            "accept_games": self.accept_games,
            "ladder_score": self.ladder_score,
            "ladder_error": self.ladder_error,
            "comparisons": self.comparisons,
            "accepted_steps": self.accepted_steps,
            "exhausted": self.exhausted,
            "plies": dict(self.plies),
            "build": self.build,
            "values": list(self.vector),
            "derived": list(self.derived),
            "weights": list(self.weights),
            "ratios": [round(r, 4) for r in self.ratios],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Valuation:
        version = int(data.get("procedure_version", -1))
        if version != PROCEDURE_VERSION:
            raise ValuationError(
                f"this valuation was searched by procedure version {version} and this "
                f"is {PROCEDURE_VERSION}; search again rather than reusing it"
            )
        # The weight vector is `builtin_vector` as it was when the search ran, and a store
        # searched before a vocabulary bump holds one of the old length (Q142, Q430). The
        # procedure digest does not cover the vocabulary, so such a store sits at the
        # current procedure's path and nothing else stops it loading. Refused here, and
        # accepted or not, as `FittedObjective` refuses a fit: read as this vocabulary it
        # is a different player under an unchanged name, and its verdict was measured on
        # the old one. Before this the loader let it through and the core refused it on
        # the first move, which is where a game against it or a rating run found out.
        weights = tuple(float(w) for w in data["weights"])
        if len(weights) != len(VOCABULARY):
            raise ValuationError(
                f"this valuation carries {len(weights)} weights where this build's "
                f"feature vocabulary has {len(VOCABULARY)} terms; search again rather "
                "than reusing it"
            )
        return cls(
            ruleset_hash=str(data["ruleset_hash"]),
            procedure=str(data["procedure"]),
            piece_ids=tuple(str(pid) for pid in data.get("piece_ids") or ()),
            vector=tuple(float(v) for v in data["values"]),
            derived=tuple(float(v) for v in data["derived"]),
            weights=weights,
            accepted=bool(data["accepted"]),
            accept_score=float(data.get("accept_score", 0.0)),
            accept_error=float(data.get("accept_error", 0.0)),
            accept_games=int(data.get("accept_games", 0)),
            ladder_score=float(data.get("ladder_score", 0.5)),
            ladder_error=float(data.get("ladder_error", 0.0)),
            comparisons=int(data.get("comparisons", 0)),
            accepted_steps=int(data.get("accepted_steps", 0)),
            exhausted=bool(data.get("exhausted", False)),
            plies={str(k): float(v) for k, v in (data.get("plies") or {}).items()},
            build=str(data.get("build", "")),
            procedure_version=version,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.as_dict(), handle, indent=2)
            handle.write("\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Valuation:
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


# Keyed by `(procedure, ruleset_hash)`, mirroring `fitted._FITTED`, and for the same
# reason: the procedure is the half the agent's name carries and the ruleset is the half
# it deliberately does not.
_VALUATIONS: dict[tuple[str, str], Valuation] = {}


def register_valuation(valuation: Valuation, replace: bool = False) -> None:
    """Make a valuation resolvable.

    ``replace`` exists for the searcher, which plays candidates *by their real name*
    before the verdict is known and then registers the verdict over the provisional
    entry. Testing a different spelling of the same table would leave the resolution path
    itself untested, which is where this is most likely to be quietly wrong.
    """
    key = (valuation.procedure, valuation.ruleset_hash)
    if replace:
        _VALUATIONS[key] = valuation
    else:
        _VALUATIONS.setdefault(key, valuation)


def registered_valuations() -> dict[tuple[str, str], Valuation]:
    return dict(_VALUATIONS)


def normalise_procedure(digest: Any) -> str:
    """A procedure digest as it came back out of an agent string.

    ``parse_agent`` coerces an option value to a number when the text survives the trip,
    so an all-digit digest arrives with its leading zeros gone. Same repair as
    ``heuristics.normalise_digest``, and it cannot be skipped: one digest in sixteen
    starts with a zero.
    """
    return str(digest).zfill(HASH_LENGTH)


def store_path(
    ruleset_hash: str, procedure: str, root: str | Path = STORE_DIR
) -> Path:
    """Where one ruleset's valuation under one search lives.

    Both halves are in the filename because a procedure bump must not overwrite the
    evidence of what the previous one produced. A search costs thousands of games; the
    file it leaves is the only record of them.
    """
    return Path(root) / f"{ruleset_hash}-{procedure}.json"


def load_valuation(
    ruleset_hash: str, procedure: str | None = None, root: str | Path = STORE_DIR
) -> Valuation | None:
    """The stored valuation for a ruleset, registered and ready, or ``None``.

    Absent is reported as absent. A missing store means the search has not been run for
    this ruleset; it is neither an error nor a verdict, and it becomes an error only when
    an agent that needs it is asked to play.
    """
    procedure = procedure or SEARCH.digest
    path = store_path(ruleset_hash, procedure, root)
    if not path.exists():
        return None
    valuation = Valuation.load(path)
    if valuation.procedure != procedure or valuation.ruleset_hash != ruleset_hash:
        raise ValuationError(
            f"{path} holds a valuation for {valuation.ruleset_hash}/"
            f"{valuation.procedure} and is filed under {ruleset_hash}/{procedure}; the "
            "file has been renamed"
        )
    register_valuation(valuation)
    return valuation


def load_all(root: str | Path = STORE_DIR) -> int:
    """Register every valuation on disk this procedure version can read. Returns how many.

    A file written by an older version, or carrying another vocabulary's weight vector,
    is skipped rather than refused: an old store sitting beside a new one is the normal
    state of a bumped procedure, and it is what keeps the previous campaign's numbers
    readable.
    """
    directory = Path(root)
    if not directory.is_dir():
        return 0
    loaded = 0
    for path in sorted(directory.glob("*.json")):
        try:
            valuation = Valuation.load(path)
        except (ValuationError, KeyError):
            continue
        register_valuation(valuation)
        loaded += 1
    return loaded


def resolve_valuation(procedure: Any, ruleset_hash: str) -> Valuation:
    """The valuation a ``(procedure, ruleset)`` pair names, or an error naming what is missing."""
    procedure = normalise_procedure(procedure)
    if not ruleset_hash:
        raise ValuationError(
            f"a searched valuation needs to know which ruleset it is playing; procedure "
            f"{procedure!r} was resolved without one. That is a caller which has not "
            "passed its ruleset hash through, not a missing file."
        )
    valuation = _VALUATIONS.get((procedure, ruleset_hash))
    if valuation is None:
        # Read the committed store before giving up. Not a default -- a missing file
        # still raises below -- but the store is the valuation's home, and the analysis
        # pass spawns fresh processes that inherit no registry, so a game played by one
        # of these would otherwise resolve in the parent and fail in the child.
        valuation = load_valuation(ruleset_hash, procedure)
    if valuation is None:
        raise ValuationError(
            f"no valuation for procedure {procedure!r} on ruleset {ruleset_hash!r} is "
            f"loaded; run `chess-variant-sandbox values` for that ruleset. A game played by one is "
            "reproducible only with its store in hand, exactly as a ruleset hash needs "
            "rulesets/"
        )
    return valuation


def stale_verdict(valuation: Valuation) -> str | None:
    """Why this valuation's scores describe a different engine, if they do.

    Reported, never enforced. A stored table is still a legal player and still the one the
    search found; what a mismatch invalidates is the *verdict* -- the score that decided
    whether it was accepted -- so the honest response is to say so and let a caller decide
    whether to re-play the match, which costs 96 games rather than a whole search.
    """
    from chess_variant_sandbox.measurement import built_fingerprint

    if not valuation.build:
        return (
            "this valuation records no core build, so which engine measured its verdict "
            "is unknown; re-run the acceptance match to find out"
        )
    current = built_fingerprint()
    if valuation.build == current:
        return None
    return (
        f"this valuation's verdict was measured on core {valuation.build} and the loaded "
        f"one is {current}; the engine decides the games, so the score may no longer hold"
    )


def searched_values(options: dict[str, Any], ruleset_hash: str) -> tuple[float, ...]:
    """The table an agent's ``values=`` option names, or ``()`` for the derived one.

    An empty tuple is what keeps every committed number where it is, and it is also what
    a *refused* search resolves to -- so an agent carrying ``values=`` on a ruleset the
    search failed on is the ordinary agent, by name and in the record.
    """
    procedure = options.get("values")
    if procedure is None:
        return ()
    return resolve_valuation(procedure, ruleset_hash).values


def searched_weights(options: dict[str, Any], ruleset_hash: str) -> tuple[float, ...]:
    """The weight vector a ``values=`` player carries, or ``()`` for a refused search.

    Not a choice this stage makes and not something the agent string says: it is
    ``features.builtin_vector`` for this ruleset, written down by the search because
    resolution holds a hash and cannot recompute it. See ``Valuation.weights``.
    """
    procedure = options.get("values")
    if procedure is None:
        return ()
    return resolve_valuation(procedure, ruleset_hash).playing_weights
