"""How alike the games of a library are, weighed by the library itself.

`relate.py` says *what differs* between two games, one rule at a time. This says *how
alike* every pair of a set of games is, as one number between 0 and 1, and it is the DSL's
one meaning of "alike" for the map (`docs/plan-map.md` §4): nothing downstream recomputes
it, and it reads only the specs -- no runs, no engine.

**A game is a set of words** read from its normal form (`relate.normal_form`):

- ``rule:`` every rule it involves (`relate.rules_involved`), and ``set:`` every leaf of every
  live rule's setting, a piece id at a schema position read as ``<piece>`` so that a
  renamed piece moves nothing;
- ``mechanism:`` every movement mechanism its pieces have (`relate.mechanisms`);
- ``piece:`` every piece identity: `relate.piece_identity` (movement, royalty, facing,
  region) together with the piece's other mechanic fields, and *whom it may take* --
  `captures`, `beats` and `promoted_from` name other pieces, so each named piece is read as
  that piece's own identity without these links (``<self>`` for itself), which is what
  tells a rock-paper-scissors army from three pieces that move alike;
- the board: ``lattice:``, ``size:``, ``wrap:`` (the shape of its joined edges, when any are)
  and ``army:``, how many pieces stand at the start.

**A word weighs its inverse document frequency**, ``log(N / df)`` over the set of games
given: what every game has weighs nothing, what few have weighs most. Two games' likeness is
the weighted Jaccard of their word sets -- the weight they share over the weight either
has -- and their distance is one minus it. There is no tier, ratio, scale or cut: the only
input besides the specs is which specs are in the set, so the weights move (smoothly) when
the library does, and by design.

**Deterministic and independent of order.** Every sum is `math.fsum`, which rounds once and
so does not depend on the order a set is walked in; every word is a canonical string, with
sets sorted, so it does not depend on `PYTHONHASHSEED`; and a tie in `nearest` is broken by
the games' normal-form keys before their positions. Shuffling the specs permutes the answer
and changes no number.

**Nothing here knows a game.** No ruleset or piece name appears in the module, and its only
parameters are the word set itself.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from chess_variant_sandbox.spec.relate import (
    _ANY,
    _ID,
    _IDS,
    _KEYED,
    _PIECE_FIELDS,
    _PIECE_POSITIONS,
    NormalForm,
    mechanisms,
    normal_form,
    piece_identity,
    rules_involved,
)

# Bumped when a change here would move a word, and so a weight or a distance, for a spec
# that did not change. A stored reading carries it.
LIKENESS_VERSION = "k1"

# What a piece id at a schema position reads as, in a setting and in an identity.
PIECE = "<piece>"
SELF = "<self>"

# The prefixes a word carries, which say what kind of thing it is. A reader that names a
# family by its rarest shared word (`docs/plan-map.md` §4, M2) reads the kind from these.
RULE, SETTING, MECHANISM, IDENTITY = "rule:", "set:", "mechanism:", "piece:"
LATTICE, SIZE, WRAP, ARMY = "lattice:", "size:", "wrap:", "army:"

# A piece's fields that `relate.piece_identity` already reads, or that are not the piece's
# behaviour: its id, its name and its price. Every other field is part of what it is, read
# as written, so a field a later schema adds joins the identity without a change here.
_READ_ELSEWHERE = frozenset({"id", "name", "value", "moves", "royal", "facing", "region"})


# ---------------------------------------------------------------------------- the words


def words(form: NormalForm) -> frozenset[str]:
    """What one game is made of, as words. Reads the normal form and nothing else."""
    found = {RULE + key for key in rules_involved(form)}
    found |= {MECHANISM + _canon(mechanism) for mechanism in mechanisms(form)}
    found |= {IDENTITY + _canon(identity) for identity in _identities(form).values()}
    for key, value in form.rules.items():
        _settings(key, value, _PIECE_POSITIONS.get(key), set(form.pieces), found)
    found.add(LATTICE + form.lattice)
    found.add(f"{SIZE}{form.size[0]}x{form.size[1]}")
    if form.wrap is not None:
        found.add(WRAP + _canon(form.wrap))
    found.add(f"{ARMY}{len(form.cells)}")
    return frozenset(found)


def words_of(spec: dict[str, Any]) -> frozenset[str]:
    """`words` of a spec, through its (memoised) normal form."""
    return words(normal_form(spec))


def _identities(form: NormalForm) -> dict[str, tuple[Any, ...]]:
    """Each piece's identity, with the pieces it names read as what they are.

    A piece that may take only one other is a different piece from one that may take
    anything, and the one it may take is known by what *that* piece is rather than by its
    id, so a renamed army reads the same. The named piece is read without its own links,
    which keeps a cycle (rock takes scissors takes paper takes rock) finite.
    """
    base = {pid: piece_identity(form, pid) for pid in form.pieces}

    def named(pid: str, other: Any) -> str:
        if other == pid:
            return SELF
        return _canon(base[other]) if other in base else _canon(other)

    found: dict[str, tuple[Any, ...]] = {}
    for pid, piece in form.pieces.items():
        extra = []
        for key in sorted(piece):
            value = piece[key]
            # Absent, null and a written `false` say the same: the piece does not do it.
            if key in _READ_ELSEWHERE or value is None or value is False:
                continue
            if key in _PIECE_FIELDS:
                if isinstance(value, list):
                    value = tuple(sorted(named(pid, other) for other in value))
                else:
                    value = named(pid, value)
            extra.append((key, _canon(value)))
        found[pid] = (*base[pid], tuple(extra))
    return found


def _settings(path: str, value: Any, shape: Any, ids: set[str], out: set[str]) -> None:
    """Every leaf of one rule's value as ``set:<path>=<leaf>``, into `out`.

    `shape` is the rule's row of `relate._PIECE_POSITIONS`, which says where a piece id may
    stand: only there is an id read as ``<piece>``, so a square or an ending that happens to
    spell an id is left as written. A list's items share one path (``[]``), and a bare
    object where the schema allows a list of them is read as a list of one.
    """
    if shape is _ID:
        out.add(f"{SETTING}{path}={_leaf(value, ids)}")
        return
    if shape is _IDS and isinstance(value, list):
        _leaves(f"{path}[]", [_leaf(item, ids) for item in value], out)
        return
    if shape is _KEYED and isinstance(value, dict):
        if not value:
            out.add(f"{SETTING}{path}={{}}")
        for key, inner in value.items():
            _settings(f"{path}.{PIECE if key in ids else key}", inner, None, ids, out)
        return
    if isinstance(shape, list):
        items = value if isinstance(value, list) else [value]
        if not items:
            out.add(f"{SETTING}{path}=[]")
        for item in items:
            _settings(f"{path}[]", item, shape[0], ids, out)
        return
    if isinstance(value, dict):
        if not value:
            out.add(f"{SETTING}{path}={{}}")
        rows = shape if isinstance(shape, dict) else {}
        for key, inner in value.items():
            _settings(f"{path}.{key}", inner, rows.get(key, rows.get(_ANY)), ids, out)
        return
    if isinstance(value, list):
        if not value:
            out.add(f"{SETTING}{path}=[]")
        for item in value:
            _settings(f"{path}[]", item, None, ids, out)
        return
    out.add(f"{SETTING}{path}={value!r}")


def _leaves(path: str, leaves: list[str], out: set[str]) -> None:
    if not leaves:
        out.add(f"{SETTING}{path}=[]")
    out.update(f"{SETTING}{path}={leaf}" for leaf in leaves)


def _leaf(value: Any, ids: set[str]) -> str:
    return PIECE if isinstance(value, str) and value in ids else repr(value)


def _canon(value: Any) -> str:
    """One canonical string for a nested value: sets and maps sorted, sequences in order.

    `repr` of a frozenset follows its hash order, which for strings moves with
    `PYTHONHASHSEED`, so a word built from `repr` would differ between two processes.
    """
    if isinstance(value, (frozenset, set)):
        return "{" + ",".join(sorted(_canon(item) for item in value)) + "}"
    if isinstance(value, dict):
        rows = sorted(f"{_canon(key)}:{_canon(inner)}" for key, inner in value.items())
        return "<" + ",".join(rows) + ">"
    if isinstance(value, (tuple, list)):
        return "(" + ",".join(_canon(item) for item in value) + ")"
    return repr(value)


# ------------------------------------------------------------------------- the relation


@dataclass(frozen=True, eq=False)
class Likeness:
    """The relation over one set of games, in the order they were given.

    `words[i]` is game *i*'s word set, `keys[i]` its normal-form key, and `weights` every
    word's IDF over the set. Built by `likeness` (specs) or `likeness_of_forms`.
    """

    words: tuple[frozenset[str], ...]
    keys: tuple[str, ...]
    weights: dict[str, float]
    totals: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.words)

    def similarity(self, i: int, j: int) -> float:
        """The weighted Jaccard of games *i* and *j*: 1 for one word set, 0 for none shared."""
        one, other = self.words[i], self.words[j]
        if one == other:
            return 1.0
        shared = math.fsum(self.weights[word] for word in one & other)
        union = self.totals[i] + self.totals[j] - shared
        # A union that weighs nothing is two sets of words every game has, which are then
        # one set; the equality above has caught it, and this is the guard for rounding.
        return shared / union if union > 0 else 1.0

    def distance(self, i: int, j: int) -> float:
        """One minus `similarity`: 0 for the same words, 1 for nothing rare in common."""
        return 1.0 - self.similarity(i, j)

    def matrix(self) -> list[list[float]]:
        """Every pair's distance, symmetric with a zero diagonal."""
        n = len(self.words)
        rows = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                rows[i][j] = rows[j][i] = self.distance(i, j)
        return rows

    def nearest(self, i: int, k: int) -> list[int]:
        """Game *i*'s `k` nearest others, nearest first.

        A tie is broken by the other game's normal-form key, then its position, so the
        answer does not depend on the order the library was listed in.
        """
        others = (j for j in range(len(self.words)) if j != i)
        return sorted(others, key=lambda j: (self.distance(i, j), self.keys[j], j))[:k]

    def shared(self, indices: Iterable[int]) -> list[str]:
        """The words every one of these games has, the rarest first (ties by the word)."""
        sets = [self.words[i] for i in indices]
        common = frozenset.intersection(*sets) if sets else frozenset()
        return sorted(common, key=lambda word: (-self.weights[word], word))


def likeness_of_forms(forms: Sequence[NormalForm]) -> Likeness:
    """The relation over these normal forms, each word weighed by its IDF over them."""
    bags = tuple(words(form) for form in forms)
    n = len(bags)
    counts = Counter(word for bag in bags for word in bag)
    weights = {word: math.log(n / count) for word, count in counts.items()}
    totals = tuple(math.fsum(weights[word] for word in bag) for bag in bags)
    return Likeness(
        words=bags, keys=tuple(form.key for form in forms), weights=weights, totals=totals
    )


def likeness(specs: Sequence[dict[str, Any]]) -> Likeness:
    """The relation over these specs. Reads the specs and asks the engine nothing."""
    return likeness_of_forms([normal_form(spec) for spec in specs])
