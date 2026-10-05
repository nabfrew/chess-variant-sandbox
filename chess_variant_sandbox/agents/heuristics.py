"""Heuristic players: a named plan, and a weight vector over ``features.VOCABULARY``.

A heuristic is *data*, for the same reason a ruleset is. The project's first decision --
rules are declarative data, never generated code -- applies unchanged one level up: a
proposal that could carry arbitrary code would be unhashable, unmeasurable and unsafe,
and it is the vocabulary being fixed that makes a heuristic diffable, content-addressable
and cheaply compared.

**Identity is the weight vector and nothing else.** The digest covers the weights, not
the name and not the English plan -- the exact mirror of ``canonical.MECHANIC_KEYS``
excluding ``name`` and ``description``, with the same payoff: two proposals that differ
only in what they call themselves collide, and the second is skipped before any games are
spent on it.

**Scale is normalised away before hashing.** A one-ply argmax and an alpha-beta search
are both invariant under multiplying every weight by a positive constant, so ``[1, 0.05]``
and ``[2, 0.1]`` are the *same player*. Without normalising first, a set of six proposals
routinely contains three spellings of one heuristic and ``k`` silently becomes four. The
core's floats are not exactly scale-invariant, though, so once a digest is registered it
names the raw numbers it was registered with, and a second spelling is refused (Q433).

**Resolution never defaults.** ``resolve`` on an unknown digest raises. A zero vector or
a silent fallback would produce a differently-playing agent under an unchanged name,
which is the one thing the determinism contract cannot survive -- and it is the
``LLMUnavailable`` posture applied here: failing loudly is the whole point.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_variant_sandbox.agents.features import (
    GLOSSES,
    MAX_ABS_WEIGHT,
    VOCABULARY,
    VOCABULARY_VERSION,
)
from chess_variant_sandbox.spec.canonical import HASH_LENGTH
from chess_variant_sandbox.spec.schema import MAX_PIECE_TYPES

# Weights are rounded to this many decimals before hashing. Coarse enough that two
# vectors differing only in float noise are one heuristic, fine enough that two
# genuinely different balances are two.
DIGEST_PRECISION = 6

STORE_DIR = "heuristics"


class HeuristicError(ValueError):
    """A heuristic, or a store of them, that cannot be trusted to name one player."""


@dataclass(frozen=True)
class Heuristic:
    """One player's objective: what it is called, what it is trying to do, and the numbers.

    ``plan`` is English and the engine never reads it. It exists for the report, for a
    person reading a store, and for the interface -- a number nobody can state in a
    sentence is not a heuristic, it is a fit.
    """

    name: str
    plan: str
    weights: tuple[float, ...]
    # What this player thinks the pieces are worth, or empty for the table the rules
    # imply. Ruleset-shaped rather than vocabulary-shaped, which is why it is a second
    # field and not one more entry per piece in `VOCABULARY`: the vector's length would
    # then depend on the ruleset, and a fixed length is what makes a digest comparable
    # and `heuristic_headroom` on chess and on a territory game the same quantity.
    #
    # Bounded only by `MAX_PIECE_TYPES` here. The exact length is a fact about a board
    # and this class holds none, so it is checked in `PieceValues.of`, where a spec is
    # in hand.
    values: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.weights) != len(VOCABULARY):
            raise HeuristicError(
                f"a heuristic under vocabulary {VOCABULARY_VERSION} has "
                f"{len(VOCABULARY)} weights, got {len(self.weights)}"
            )
        for index, weight in enumerate(self.weights):
            if not math.isfinite(weight):
                raise HeuristicError(f"{VOCABULARY[index]} is not a finite weight")
            if abs(weight) > MAX_ABS_WEIGHT:
                raise HeuristicError(
                    f"{VOCABULARY[index]} is {weight}, beyond +/-{MAX_ABS_WEIGHT}; "
                    "scale is meaningless here, so a large weight is a mistake rather "
                    "than an emphasis"
                )
        if not any(self.weights):
            raise HeuristicError(
                "every weight is zero, which is a player that moves at random; that is "
                "a legitimate control but never a proposal"
            )
        self._check_values()

    def _check_values(self) -> None:
        """What a piece-value override may not be, without a board to check it against.

        Royal slots are pinned at zero and a non-zero one is refused rather than
        quietly zeroed. `ROYAL_VALUE = 0.0` is load-bearing: `_reach_is_priceable`
        records what happens when a royal acquires a term nothing opposes -- king of the
        hill collapses from 54.7 mean plies to 20 and `quick_kill_rate` goes past the
        gate -- and a settable royal value reintroduces exactly that from the other
        direction. Which slots are royal is a fact about the ruleset, so this can only
        catch the case where the file says so; `PieceValues.of` catches the rest.
        """
        if not self.values:
            return
        if len(self.values) > MAX_PIECE_TYPES:
            raise HeuristicError(
                f"a piece-value override has at most {MAX_PIECE_TYPES} entries, which "
                f"is the validator's own cap on piece types, got {len(self.values)}"
            )
        for index, value in enumerate(self.values):
            if not math.isfinite(value):
                raise HeuristicError(f"piece {index} is not worth a finite amount")
        if not any(self.values):
            raise HeuristicError(
                "every piece is worth zero, which is not an opinion about the pieces "
                "but the absence of one; leave the override out instead"
            )

    @property
    def normalised(self) -> tuple[float, ...]:
        """The vector scaled to unit length and rounded. What identity is taken over.

        ``math.fsum`` because a digest must not depend on the Python version: float
        ``sum`` is compensated from CPython 3.12 and naive before it (Q448).
        """
        norm = math.sqrt(math.fsum(w * w for w in self.weights))
        if norm == 0.0:  # pragma: no cover - __post_init__ rejects this
            return self.weights
        return tuple(round(w / norm, DIGEST_PRECISION) + 0.0 for w in self.weights)

    @property
    def digest(self) -> str:
        """Identity: the unit-length weights, and the values when there are any.

        **The values enter the hashed blob only when the vector is non-empty**, so an
        opponent on the derived table serialises and hashes exactly as it did before
        this field existed. That is what keeps every committed set in `heuristics/`,
        every `model/` set, any saved `custom/` store and every stored fit valid rather
        than failing `from_dict`'s edited-file check. Empty means derived, the same
        convention the weights already use for the built-in objective.

        Scale invariance is one-sided on purpose. The weights are unit-scaled because
        argmax is invariant under a positive constant; the values are *not*, because
        with two vectors the invariance is joint -- scaling weights by `c` and values by
        `d` scales material, prey and hand by `c*d` and everything else by `c` -- so
        each vector needs its own convention. The values' convention is the derived
        table's scale, fixed by `evaluation.rescaled` before they are stored.
        """
        payload: dict[str, Any] = {
            "vocabulary_version": VOCABULARY_VERSION,
            "weights": list(self.normalised),
        }
        if self.values:
            payload["values"] = [round(v, DIGEST_PRECISION) + 0.0 for v in self.values]
        blob = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:HASH_LENGTH]

    def as_dict(self) -> dict[str, Any]:
        stated: dict[str, Any] = {
            "name": self.name,
            "plan": self.plan,
            "weights": list(self.weights),
            "digest": self.digest,
        }
        # Written only when there is one, for the reason the digest covers it only when
        # there is one: a file that never mentions piece values is a file this change
        # did not touch.
        if self.values:
            stated["values"] = list(self.values)
        return stated

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Heuristic:
        heuristic = cls(
            name=str(data["name"]),
            plan=str(data.get("plan", "")),
            weights=tuple(float(w) for w in data["weights"]),
            values=tuple(float(v) for v in data.get("values") or ()),
        )
        stated = data.get("digest")
        # A store is a cache with a verified key, not an input. A hand-edited file must
        # fail here rather than produce a different player under an unchanged name --
        # that is the whole reason the determinism contract survives content addressing.
        if stated is not None and stated != heuristic.digest:
            raise HeuristicError(
                f"heuristic {heuristic.name!r} is stored under {stated} but its weights "
                f"hash to {heuristic.digest}; the file has been edited"
            )
        return heuristic


# Content-addressed, module level, mirroring `tournament/play._EVAL_VALUES`. A
# `NativeResolver` sees an options dict and nothing else -- no engine, no spec -- so
# `h=a3f19c` has to resolve through a lookup, and that lookup has to be pure.
_HEURISTICS: dict[str, Heuristic] = {}


def register_heuristic(heuristic: Heuristic) -> str:
    """Make a heuristic resolvable by digest. Returns the digest.

    Registering the same numbers again, under any name or plan, is a no-op that keeps the
    first name: identity is the vectors. Registering *other* numbers under a held digest
    raises -- see ``_register``.
    """
    _register((heuristic,))
    return heuristic.digest


def _register(heuristics: tuple[Heuristic, ...]) -> None:
    """Register every heuristic, or none of them if one would name a second player.

    The digest is taken over the unit-length, rounded weights, and the core plays the
    raw ones. Argmax is scale-invariant in exact arithmetic and the core's arithmetic is
    not exact, so ``[1, 0.05]`` and ``[2, 0.1]`` share a digest and can play different
    games (Q431: gardner's built-in at 0.707 of unit length against the same vector at
    unit length, 0.4167 where one player scores exactly 0.5). Keeping the first silently
    made an agent string mean whichever spelling a process met first. So a digest names
    one raw weight vector and one raw value vector, and a second is refused naming both.

    All or nothing, so a store that clashes is refused whole rather than half-offered.
    """
    staged: dict[str, Heuristic] = {}
    for heuristic in heuristics:
        held = other_spelling(heuristic, staged.values())
        if held is not None:
            raise HeuristicError(
                f"{heuristic.digest} already names {held.name!r}, and {heuristic.name!r} "
                "spells it with other raw numbers; the digest is taken at unit length but "
                "the core plays the numbers as written, so they would be two players "
                "under one name -- register the vector exactly as it is already held"
            )
        if heuristic.digest not in _HEURISTICS:
            staged.setdefault(heuristic.digest, heuristic)
    _HEURISTICS.update(staged)


def other_spelling(
    heuristic: Heuristic, among: Iterable[Heuristic] = ()
) -> Heuristic | None:
    """What already names this heuristic's digest with other raw numbers, or ``None``.

    Looked for in ``among``, then in this process's registry. The same numbers under
    another name or plan are the same player and are not an answer. This is the question
    registration refuses on, asked without registering, so a caller that writes a store
    can refuse before it writes rather than after.
    """
    digest = heuristic.digest
    numbers = (heuristic.weights, heuristic.values)
    candidates = list(among)
    if digest in _HEURISTICS:
        candidates.append(_HEURISTICS[digest])
    for held in candidates:
        if held.digest == digest and (held.weights, held.values) != numbers:
            return held
    return None


def normalise_digest(digest: Any) -> str:
    """A digest as it came back out of an agent string.

    ``parse_agent`` coerces an option value to an ``int`` when it looks like one, so an
    all-digit digest arrives as a number *with its leading zeros gone* -- ``096717939346``
    comes back as ``96717939346``. Stringifying is not enough; it has to be re-padded to
    the fixed width every digest has. Found by a heuristic whose hash happened to start
    with a zero, which is one set in sixteen.
    """
    return str(digest).zfill(HASH_LENGTH)


def resolve(digest: Any) -> Heuristic:
    """The heuristic a digest names, or an error naming what is missing."""
    digest = normalise_digest(digest)
    heuristic = _HEURISTICS.get(digest)
    if heuristic is None:
        raise HeuristicError(
            f"no heuristic {digest!r} is loaded; a game played by one is reproducible "
            f"only with its store in hand, exactly as a ruleset hash needs rulesets/"
        )
    return heuristic


def registered() -> dict[str, Heuristic]:
    return dict(_HEURISTICS)


def weights_for(options: dict[str, Any]) -> tuple[float, ...]:
    """The weight vector an agent's options name, or ``()`` for the built-in objective.

    ``parse_agent`` coerces an all-digit option value to an ``int``, so the digest is
    stringified rather than compared as it arrives. An empty tuple is the signal that no
    heuristic is in play and the built-in evaluator should run untouched -- which is what
    keeps every committed number where it is.
    """
    digest = options.get("h")
    if digest is None:
        return ()
    return resolve(digest).weights


def values_for(options: dict[str, Any]) -> tuple[float, ...]:
    """The piece-value override these options name, or ``()`` for the derived table.

    Resolved through the same digest as the weights, because they are one player: the
    digest covers both, so an agent string names one objective and not a weight vector
    plus whatever values happen to be lying around.
    """
    digest = options.get("h")
    if digest is None:
        return ()
    return resolve(digest).values


# ------------------------------------------------------------------ the store


@dataclass(frozen=True)
class HeuristicSet:
    """Every heuristic for one ruleset, and what produced them.

    Frozen at generation and never re-sampled at score time. The fuzziness the whole
    idea trades on lives in *writing* this file; once written it is as fixed as the
    ruleset it names.
    """

    ruleset_hash: str
    heuristics: tuple[Heuristic, ...]
    model: str = ""
    vocabulary_version: int = VOCABULARY_VERSION
    # The ruleset's piece ids, in order, echoed beside a values array the way the
    # vocabulary is echoed beside the weights. A piece-value vector is positional and
    # says nothing about which piece each slot is, so a set that outlived the ruleset it
    # was written for would otherwise mean something else in silence rather than
    # diagnose itself. Empty where no heuristic in the set states any values.
    piece_ids: tuple[str, ...] = ()

    @property
    def k(self) -> int:
        return len(self.heuristics)

    @property
    def set_hash(self) -> str:
        """Identity of the whole set, so a diagnostic traces back to what produced it.

        Covers the ordered digests and the vocabulary version. Order matters because
        index 0 is by convention the built-in anchor, and a set that lost it is a
        different experiment.
        """
        blob = json.dumps(
            {
                "vocabulary_version": self.vocabulary_version,
                "digests": [h.digest for h in self.heuristics],
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:HASH_LENGTH]

    def register(self) -> None:
        _register(self.heuristics)

    def as_dict(self) -> dict[str, Any]:
        stated: dict[str, Any] = {
            "ruleset_hash": self.ruleset_hash,
            "vocabulary_version": self.vocabulary_version,
            # Written out redundantly so a stale file diagnoses itself rather than
            # silently meaning something else.
            "vocabulary": list(VOCABULARY),
            "model": self.model,
            "k": self.k,
            "set_hash": self.set_hash,
            "heuristics": [h.as_dict() for h in self.heuristics],
        }
        if self.piece_ids and any(h.values for h in self.heuristics):
            stated["piece_ids"] = list(self.piece_ids)
        return stated

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.as_dict(), handle, indent=2)
            handle.write("\n")
        return path

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], piece_ids: tuple[str, ...] = ()
    ) -> HeuristicSet:
        """Load a set, refusing one whose stated pieces are not this ruleset's.

        ``piece_ids`` is what the caller's ruleset actually declares, and it is optional
        for the same reason the check is: a caller holding only a hash cannot supply it,
        and a file written before values existed states none. The refusal fires **only
        when both are present and they disagree**, exactly as ``stored_vocabulary``
        already does -- so every existing file still loads.
        """
        version = int(data.get("vocabulary_version", -1))
        # A weight vector only means anything against the vocabulary it was drawn for,
        # so a mismatch is refused rather than reinterpreted. Same rule as a weights
        # version: two versions are different quantities and never share a table.
        if version != VOCABULARY_VERSION:
            raise HeuristicError(
                f"heuristics were written against vocabulary {version} and this is "
                f"{VOCABULARY_VERSION}; regenerate the set rather than reusing it"
            )
        stored_vocabulary = data.get("vocabulary")
        if stored_vocabulary is not None and tuple(stored_vocabulary) != VOCABULARY:
            raise HeuristicError(
                "the stored vocabulary does not match this one, though the versions "
                "agree; one of them was edited without a bump"
            )
        stored_pieces = data.get("piece_ids")
        if (
            stored_pieces is not None
            and piece_ids
            and tuple(stored_pieces) != tuple(piece_ids)
        ):
            raise HeuristicError(
                f"these heuristics price the pieces {list(stored_pieces)} and this "
                f"ruleset has {list(piece_ids)}; a value vector is positional, so the "
                "set was written for a different game"
            )
        return cls(
            ruleset_hash=str(data["ruleset_hash"]),
            heuristics=tuple(Heuristic.from_dict(h) for h in data["heuristics"]),
            model=str(data.get("model", "")),
            vocabulary_version=version,
            piece_ids=tuple(str(pid) for pid in stored_pieces or ()),
        )

    @classmethod
    def load(cls, path: str | Path, piece_ids: tuple[str, ...] = ()) -> HeuristicSet:
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle), piece_ids)


def store_path(ruleset_hash: str, root: str | Path = STORE_DIR) -> Path:
    """Where one ruleset's heuristics live.

    Keyed by hash rather than by name for the reason `spec_hash` excludes `name`: a
    renamed ruleset is the same game and must reuse its heuristics, and a design-loop
    candidate has no name worth a filename.
    """
    return Path(root) / f"{ruleset_hash}.json"


def load_set(
    ruleset_hash: str,
    root: str | Path = STORE_DIR,
    piece_ids: tuple[str, ...] = (),
) -> HeuristicSet | None:
    """The stored set for a ruleset, registered and ready, or ``None`` if there is none.

    Absent is reported as absent. A missing store is neither an error nor a verdict --
    it means the heuristic pass has not been run for this ruleset, and every number that
    depends on it is simply not produced.
    """
    path = store_path(ruleset_hash, root)
    if not path.exists():
        return None
    heuristic_set = HeuristicSet.load(path, piece_ids)
    heuristic_set.register()
    return heuristic_set


def vocabulary_prose() -> str:
    """The vocabulary as the proposing model sees it.

    A pure function of the vocabulary -- no ruleset, no score, no measurement -- which
    is what makes it constant across candidates and therefore free of any gradient.
    """
    lines = [f"{index}. {name} - {GLOSSES[name]}" for index, name in enumerate(VOCABULARY)]
    return "\n".join(lines)
