"""Betza notation, as Fairy-Stockfish actually parses it.

This module is a *port of a parser, written backwards*. Fairy-Stockfish turns a Betza
string into a set of directions in ``src/piece.cpp::from_betza``; here we turn a set of
directions back into a Betza string. Everything in it -- the atom table, the eight
images an atom expands into, which direction modifier enables which image, and the rule
that splits a mixed modifier on an orthogonal atom -- is read off that function rather
than off Betza's own documentation, because the two disagree in places and only one of
them is going to be playing the game.

**The tables below are a claim about another program, so they are checked against it.**
``tests/test_fairy_betza.py`` asks Fairy-Stockfish for the moves of a lone piece on an
empty board and compares them with the directions this module was asked to encode. A
Betza string that is merely plausible produces a legal-looking game under slightly
different rules, which is the one failure mode this whole integration has to not have.

The direction convention here is the project's own: a vector is ``(dx, dy)`` with ``+y``
forward for the first player. Fairy-Stockfish stores a direction as
``drank * FILE_NB + dfile`` and negates it for the second player, which is a 180-degree
rotation -- see ``variant._orientation_matches`` for why that is a constraint on which
rulesets translate rather than something this module can paper over.
"""

from __future__ import annotations

from itertools import combinations

# The leaper atoms, keyed by the sorted absolute components ``(larger, smaller)``.
#
# Taken from ``leaperAtoms`` in ``piece.cpp``. ``L``/``C`` and ``J``/``Z`` are aliases
# there; one of each pair is enough and the more familiar spelling is kept. ``K`` and
# ``Q`` are deliberately absent: they are two atoms under one letter, which would make
# the direction bookkeeping below a special case for no gain, since ``WF`` and ``RB``
# say the same thing in one more character.
LEAPERS: dict[tuple[int, int], str] = {
    (1, 0): "W",
    (1, 1): "F",
    (2, 0): "D",
    (2, 1): "N",
    (2, 2): "A",
    (3, 0): "H",
    (3, 1): "C",
    (3, 2): "Z",
    (3, 3): "G",
}

# The two atoms that have a dedicated unlimited-rider letter. Every other atom rides by
# doubling its own letter (``NN`` is a nightrider), which ``piece.cpp`` accepts through
# the same branch as a digit.
RIDERS: dict[tuple[int, int], str] = {(1, 0): "R", (1, 1): "B"}

# Which atoms Fairy-Stockfish's *movegen* -- as opposed to its parser -- implements for
# the awkward move kinds. The parser will happily build directions it cannot then
# generate moves along, so these are the sets named in its own documentation:
# "lame leapers (n) for N, A, Z, and D directions", "hoppers and grasshoppers for W/R
# and F/B directions".
LAME_ATOMS = frozenset({(2, 1), (2, 2), (3, 2), (2, 0)})
HOP_ATOMS = frozenset({(1, 0), (1, 1)})

# The longest ride a single digit can spell. A range beyond this is refused rather than
# clamped: a rook that stops after nine squares is a different piece from a rook.
MAX_RIDE = 9


def images(a: int, b: int) -> list[tuple[int, int]]:
    """The eight directions the atom ``(a, b)`` expands into, in the core's own order.

    Order matters and is not cosmetic: ``DIRECTION_TOKENS`` below indexes into this
    list, and both were read off the same eight ``if`` statements in ``from_betza``.
    Duplicates are kept rather than deduplicated -- an atom with ``b == 0`` or ``a == b``
    has only four distinct images, and the core simply writes the same map entry twice.
    """
    return [
        (b, a),
        (-b, -a),
        (a, -b),
        (-a, b),
        (a, b),
        (-a, -b),
        (b, -a),
        (-b, a),
    ]


# Which of the eight images each direction modifier enables.
#
# Inverted from the eight ``has_dir(...)`` disjunctions in ``from_betza``. A modifier is
# always two characters by the time the core has read it: a lone ``f`` before an atom
# becomes ``"ff"``, and ``f`` before a horizontal becomes the pair. Writing both
# characters out explicitly is what keeps two modifiers from silently merging into one
# -- ``"fs"`` is a single token and ``"ffss"`` is two, and they do not mean the same
# thing.
DIRECTION_TOKENS: dict[str, frozenset[int]] = {
    "ff": frozenset({0, 7}),
    "bb": frozenset({1, 6}),
    "vv": frozenset({0, 1, 6, 7}),
    "rr": frozenset({2, 4}),
    "ll": frozenset({3, 5}),
    "ss": frozenset({2, 3, 4, 5}),
    "rf": frozenset({0}),
    "lb": frozenset({1}),
    "br": frozenset({2}),
    "fl": frozenset({3}),
    "fr": frozenset({4}),
    "bl": frozenset({5}),
    "rb": frozenset({6}),
    "lf": frozenset({7}),
    "rv": frozenset({0, 6}),
    "lv": frozenset({1, 7}),
    "bs": frozenset({2, 5}),
    "fs": frozenset({3, 4}),
    "fh": frozenset({0, 3, 4, 7}),
    "bh": frozenset({1, 2, 5, 6}),
    "rh": frozenset({0, 2, 4, 6}),
    "lh": frozenset({1, 3, 5, 7}),
    "hr": frozenset({0, 1, 2, 3}),
    "hl": frozenset({4, 5, 6, 7}),
}

def token_images(token: str, a: int, b: int) -> frozenset[int]:
    """Which images ``token`` enables on the atom ``(a, b)``.

    The special case is the core's, not ours. For a *single-atom orthogonal* piece --
    ``b == 0``, so ``W``, ``D``, ``H`` and their riders -- a mixed modifier such as
    ``fs`` is split into ``ff`` and ``ss`` before it is applied, with a comment in
    ``from_betza`` saying this is what makes ``fsW`` mean what the xiangqi soldier
    needs. Without the split, ``fs`` would select the two images that do not exist on
    an orthogonal atom and the piece would have no moves at all.
    """
    if b == 0 and token[0] != token[1]:
        first = DIRECTION_TOKENS.get(token[0] * 2, frozenset())
        second = DIRECTION_TOKENS.get(token[1] * 2, frozenset())
        return first | second
    return DIRECTION_TOKENS.get(token, frozenset())


def atom_for(vector: tuple[int, int]) -> tuple[int, int] | None:
    """The atom a vector belongs to, or ``None`` if Betza has no letter for it."""
    a, b = sorted((abs(vector[0]), abs(vector[1])), reverse=True)
    return (a, b) if (a, b) in LEAPERS else None


def group_by_atom(
    directions: set[tuple[int, int]],
) -> dict[tuple[int, int], set[tuple[int, int]]] | None:
    """Split a direction set by atom, or ``None`` if any direction has no atom."""
    grouped: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for vector in directions:
        atom = atom_for(vector)
        if atom is None:
            return None
        grouped.setdefault(atom, set()).add(vector)
    return grouped


def _modifier(a: int, b: int, wanted: set[tuple[int, int]]) -> str | None:
    """The shortest modifier prefix selecting exactly ``wanted`` from atom ``(a, b)``.

    Returns ``""`` when every image is wanted, which is Betza's bare atom. Returns
    ``None`` when no combination of at most three modifiers hits the set exactly --
    that is a refusal, not a nearest match, because a piece that moves in one direction
    it should not is a different piece.
    """
    every = images(a, b)
    target = frozenset(index for index, vector in enumerate(every) if vector in wanted)
    # An atom with a repeated or zero component names the same square from two images,
    # so "wanted" can only ever be satisfied as a whole orbit of those duplicates.
    reachable = frozenset(range(8))
    if target == reachable or {every[index] for index in target} == set(every):
        return ""

    usable = {
        token: token_images(token, a, b)
        for token in DIRECTION_TOKENS
        if token_images(token, a, b)
    }
    # Two tokens that select the same squares are interchangeable; ordering by the text
    # keeps the choice stable so a spec always produces the same variant file.
    for size in (1, 2, 3):
        for combination in combinations(sorted(usable), size):
            selected: set[int] = set()
            for token in combination:
                selected |= usable[token]
            if {every[index] for index in selected} == wanted:
                return _render(combination)
    return None


def _render(tokens: tuple[str, ...]) -> str:
    """Write a chosen set of direction modifiers so the core reads back the same set.

    A modifier is two characters by the time the core has it, and it builds the pair
    either by combining a vertical with a horizontal or by doubling a lone character.
    So ``"ff"`` may be written ``f`` -- the atom letter that follows is uppercase and
    cannot combine with it -- and that is the spelling Fairy-Stockfish uses for its own
    pieces. Writing ``ffW`` instead is *correct* and reads back identically, and it
    still cost something: `variant.BUILTIN_PIECES` matches on the text, so a gold
    general written ``WffF`` did not match the ``gold`` it is, and translated shogi
    played eight custom pieces with no piece values instead of the engine's own.

    Two or more tokens are always written in full, because ``ff`` beside ``ss`` written
    short is ``fs``, which is one token meaning something else.
    """
    if len(tokens) == 1 and tokens[0][0] == tokens[0][1]:
        return tokens[0][0]
    return "".join(tokens)


def term(
    directions: set[tuple[int, int]],
    *,
    distance: int = 1,
    mode: str = "both",
    lame: bool = False,
    hop: str = "",
    initial: bool = False,
    en_passant: bool = False,
) -> str | None:
    """One Betza term per atom, concatenated, or ``None`` if it cannot be written.

    ``distance`` is in squares: ``1`` is a leap, ``0`` is an unlimited ride, and
    anything between is a limited one. ``hop`` is ``""``, ``"p"`` (a cannon: any
    distance past the screen) or ``"g"`` (a grasshopper: the square immediately past
    it). ``en_passant`` appends the ``e`` flag the core scans a custom piece's notation
    for, which is the only way to say that a piece captures in passing.
    """
    if not directions:
        return ""
    grouped = group_by_atom(directions)
    if grouped is None:
        return None

    prefix = {"move": "m", "capture": "c", "both": ""}.get(mode)
    if prefix is None:
        return None
    if initial:
        # The core refuses initial captures outright, so an initial term that could
        # capture would silently lose half of itself.
        if mode != "move":
            return None
        prefix += "i"
    if lame:
        prefix += "n"
    prefix += hop

    parts: list[str] = []
    for atom, wanted in sorted(grouped.items()):
        a, b = atom
        if lame and atom not in LAME_ATOMS:
            return None
        if hop and atom not in HOP_ATOMS:
            return None
        modifier = _modifier(a, b, wanted)
        if modifier is None:
            return None
        letter = _atom_text(atom, distance, lame=lame, hop=bool(hop))
        if letter is None:
            return None
        parts.append(prefix + modifier + letter)
    return "".join(parts) + ("e" if en_passant else "")


def _atom_text(atom: tuple[int, int], distance: int, *, lame: bool, hop: bool) -> str | None:
    """The atom letter plus whatever spells its range."""
    leaper = LEAPERS[atom]
    if hop:
        # A hopper always rides to find its screen; ``p`` then runs to the end of the
        # ray and ``g`` stops one square past, and the distance is carried by that
        # letter rather than by a digit.
        rider = RIDERS.get(atom)
        return rider if rider is not None else leaper * 2
    if lame:
        # A lame leap is a leap. The core sets its distance to -1 itself and a digit
        # here would turn it into a rider, which is a different piece.
        return leaper if distance == 1 else None
    if distance == 1:
        return leaper
    if distance == 0:
        rider = RIDERS.get(atom)
        return rider if rider is not None else leaper * 2
    if 2 <= distance <= MAX_RIDE:
        # The leaper letter with a digit, rather than the rider letter: the core
        # reads both as the same atom at the same limit, and `W2` is the spelling
        # `variant._is_chess_pawn` matches against.
        return f"{leaper}{distance}"
    return None
