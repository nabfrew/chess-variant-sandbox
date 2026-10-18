"""A committed spec does what its own name and description say it does (Q427, from Q132).

The user decided on 2026-09-08 that **the description is the specification**: a sentence
the rules contradict is a defect in the rules, not in the prose (`docs/archive/decisions.md`).
The case this file exists for is `archive/stones` and `archive/three-stones`. From the commit
that wrote them (`c064a475`, Q101, 2026-09-07) until Q131 repaired their rules a day later,
both said a stone *"is never captured"* while any piece could take one: 0.9668 of `stones`'
captures were stones. A person found it by reading. This test would have failed on
`c064a475`, and `test_a_stone_showcase_as_committed_before_its_repair_fails` shows it.

Every spec in `rulesets/` and `archive/` is read for five kinds of claim, and the compiled
spec is asked whether each is true. The families, their triggers and their readings are the
ones `docs/reference/description-mechanics-check.md` fixed before its prototype ran. On the
71 specs of the day, these five read 16 claims on 12 specs, and every one was true:

======================  ===================================  ====================================
family                  the prose trigger                    what the rules must do
======================  ===================================  ====================================
A capture immunity      *X is never / cannot be captured*,   no type that can capture at all
                        *uncapturable*, *immune to capture*, (`spec.describe.capturing_types`)
                        *nothing may capture X*              may take X (`capture_allowed`)
B never moves           *X never moves*, *cannot move*       X's move tables are empty, every
                                                             frame
C supply                *unlimited* beside *supply / hand*;  `drops.unlimited`; `drops.stock`
                        *has three X* for a stocked X        is the number said
D repetition            *no draw by repetition*;             `repetition_draw` is never; or is
                        *threefold repetition*               a threshold, the one said
E en passant            *no en passant*; *en passant*        `en_passant` and a live
                                                             `ep_passes`, or neither
======================  ===================================  ====================================

The prose side is keyword matching, so it reads only claims made in words it knows. The
record found 11 more claims in these families, worded in ways it misses, and all 11 are
true. It leaves out territory (its words missed `rps-v5`'s *"permanent claiming"*), rules
the prose never names (those checks catch only the cases they were written for), and
*"only X may take Y"*, which needs a wild-piece exclusion before it can be trusted. None of
this is in `validate`, a gate or the designer's save. Keyword matching has been measured
only on the library, and the design page's hint already reads a visitor's sentences.

The mechanics side reads only the compiled tables and the validated spec. What a rule
*means* stays in `engine/compile.py` (architecture rule 4). No piece name appears in the
check: a piece is found by the name its own spec gives it, never by its id, because an id
like *A* is a word of prose (rule 5).

A spec is a case only when its prose makes a claim, 17 of the 73 today (`CLAIMING`, read
at collection). A spec that claims nothing has nothing to check, and until Q745 it was a
case that returned before its assert: 56 passes that said nothing (`docs/archive/plan-tests.md`
T3). A spec whose prose gains a claim becomes a case by itself; one that loses a claim
the record read fails `test_the_claims_the_record_read_are_still_read`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import CompiledSpec, compile_spec
from chess_variant_sandbox.engine.compile import NO_REPETITION_DRAW
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.spec.describe import capturing_types

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = sorted((ROOT / "rulesets").glob("*.json")) + sorted((ROOT / "archive").glob("*.json"))

# The day the user ruled that the prose is the specification (`docs/archive/decisions.md`).
DECIDED = "2026-09-08"


def label_of(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def load(path: Path) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return spec


# ------------------------------------------------------------------ the prose side

# A sentence ends at a full stop, and a claim also ends at a semicolon, a colon or " -- ".
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\s+--\s+|;\s+|:\s+")

# A. "X is never captured"; the subject is the last piece named before the phrase.
CAPTURE_PASSIVE = re.compile(
    r"\b(?:never|not|cannot|can't|can\s+never|may\s+never|may\s+not|must\s+not)\s+"
    r"(?:ever\s+)?(?:be\s+)?(?:captured|taken)\b(?!\s+back)"
    r"|\buncapturable\b|\bimmune\s+to\s+capture\b",
    re.IGNORECASE,
)
# A. "nothing may capture X"; the victim is the first piece named just after the phrase.
CAPTURE_ACTIVE = re.compile(
    r"\b(?:nothing|no\s+piece|no\s+other\s+piece|no\s+one|nobody|none)\s+"
    r"(?:may|can|could|will)\s+(?:ever\s+)?(?:capture|take)\s+",
    re.IGNORECASE,
)
# B. "X never moves", or "cannot move" at the end of a clause.
NEVER_MOVES = re.compile(
    r"\bnever\s+moves\b"
    r"|\b(?:cannot|can't|can\s+never|does\s+not|doesn't|may\s+not|may\s+never)\s+move"
    r"(?=\s*(?:[.,;:]|$|\band\b))",
    re.IGNORECASE,
)
# C. an unlimited supply, or a number of a piece the players hold.
UNLIMITED = re.compile(
    r"\b(?:unlimited|endless|inexhaustible|infinite|never\s+runs?\s+out)\b", re.IGNORECASE
)
SUPPLY_WORD = re.compile(r"\b(?:supply|stock|reserve|hand)\b", re.IGNORECASE)
HOLDING_WORD = re.compile(
    r"\b(?:has|have|holds?|starts?\s+with|in\s+hand|stock|supply|reserve)\b", re.IGNORECASE
)
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}  # fmt: skip
# D. repetition, denied or stated.
REPETITION_NEGATIVE = re.compile(
    r"\bno\s+(?:draw\s+(?:by|on)\s+|threefold\s+|three-fold\s+)?repetition"
    r"|\brepetition\s+(?:is\s+not|isn't|never|does\s+not|doesn't|cannot|can't)\b"
    r"|\bnot\s+(?:a\s+)?drawn?\s+by\s+repetition"
    r"|\bwithout\s+(?:a\s+)?(?:draw\s+by\s+)?repetition"
    r"|\bnever\s+(?:a\s+)?drawn?\s+by\s+repetition",
    re.IGNORECASE,
)
REPETITION_POSITIVE = re.compile(
    r"\b(\w+)-?fold\s+repetition"
    r"|\bdrawn?\s+(?:by|on|after)\s+(?:\w+\s+)?repetitions?"
    r"|\brepetition\s+draws\b",
    re.IGNORECASE,
)
FOLD = {"two": 2, "three": 3, "four": 4, "five": 5, "twice": 2, "thrice": 3}
# E. en passant, denied or named.
EP_NEGATIVE = re.compile(r"\b(?:no|without)\s+en\s+passant\b", re.IGNORECASE)
EP_POSITIVE = re.compile(r"\ben\s+passant\b", re.IGNORECASE)


@dataclass(frozen=True)
class Claim:
    """One thing a spec's prose says its rules do."""

    family: str  # "A" to "E"
    where: str  # "name" or "description"
    sentence: str
    says: str  # which claim of the family: see `reading`
    piece: str = ""  # A, B and C by count: the id of the piece the claim is about
    number: int = 0  # C by count: the count said. D: the fold said, 0 when none is


def sentences(spec: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """``(where, sentence)``: the name, then the description cut into claims."""
    name = " ".join(str(spec.get("name") or "").split())
    if name:
        yield "name", name
    text = " ".join(str(spec.get("description") or "").split())
    for sentence in _SENTENCE_END.split(text):
        if sentence:
            yield "description", sentence


def piece_words(spec: dict[str, Any]) -> dict[str, re.Pattern[str]]:
    """Each named piece's name as a word, singular or plural, keyed by the piece's id.

    A piece with no name is never found: its id is a letter, and *a* is a word of prose.
    """
    words = {}
    for piece in spec.get("pieces") or []:
        name = str(piece.get("name") or "").strip().lower()
        if not name:
            continue
        forms = {name, name + "s", name + "es"}
        if name.endswith("f"):
            forms.add(name[:-1] + "ves")
        if name.endswith("y"):
            forms.add(name[:-1] + "ies")
        alternation = "|".join(sorted((re.escape(f) for f in forms), key=len, reverse=True))
        words[piece["id"]] = re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)
    return words


def claims(spec: dict[str, Any]) -> list[Claim]:
    """What the prose claims, read without compiling anything."""
    words = piece_words(spec)
    found: list[Claim] = []
    for where, sentence in sentences(spec):
        found.extend(_claims_in(where, sentence, words))
    return found


def _claims_in(where: str, sentence: str, words: dict[str, re.Pattern[str]]) -> list[Claim]:
    """The claims of one sentence, family by family, as the record's table fixes them."""
    named = sorted(
        (m.start(), piece) for piece, word in words.items() for m in word.finditer(sentence)
    )
    found: list[Claim] = []

    def claim(family: str, says: str, piece: str = "", number: int = 0) -> None:
        found.append(Claim(family, where, sentence, says, piece, number))

    victims = []
    for m in CAPTURE_PASSIVE.finditer(sentence):
        before = [piece for start, piece in named if start < m.start()]
        if before:
            victims.append(before[-1])
    for m in CAPTURE_ACTIVE.finditer(sentence):
        after = [piece for start, piece in named if m.end() <= start <= m.end() + 12]
        if after:
            victims.append(after[0])
    for victim in dict.fromkeys(victims):
        claim("A", "immune", victim)

    for m in NEVER_MOVES.finditer(sentence):
        before = [piece for start, piece in named if start < m.start()]
        if before:
            claim("B", "still", before[-1])

    if UNLIMITED.search(sentence) and SUPPLY_WORD.search(sentence):
        claim("C", "unlimited")
    if HOLDING_WORD.search(sentence):
        for start, piece in named:
            count = re.search(r"\b(\w+)\s+$", sentence[:start])
            said = count.group(1).lower() if count else ""
            number = NUMBER_WORDS.get(said, int(said) if said.isdigit() else 0)
            if number:
                claim("C", "count", piece, number)

    if REPETITION_NEGATIVE.search(sentence):
        claim("D", "never")
    else:
        for m in REPETITION_POSITIVE.finditer(sentence):
            fold = (m.group(1) or "").lower().removesuffix("fold").strip("-")
            claim("D", "draws", number=FOLD.get(fold, 0))

    if EP_NEGATIVE.search(sentence):
        claim("E", "none")
    elif EP_POSITIVE.search(sentence):
        claim("E", "has")
    return found


# ------------------------------------------------------------------ the mechanics side


@dataclass(frozen=True)
class Reading:
    """What the compiled spec does about a claim, in plain words."""

    holds: bool
    rules: str  # what the rules do, when they contradict the claim
    remedy: str  # the change to the rules that would make the claim true


def _a(name: str) -> str:
    return f"{'an' if name[:1].lower() in 'aeiou' else 'a'} {name}"


def _listed(names: list[str]) -> str:
    """*a*, *a and b*, *a, b and c*; *nothing* for none."""
    if len(names) < 2:
        return "".join(names) or "nothing"
    return ", ".join(names[:-1]) + " and " + names[-1]


def _has_moves(compiled: CompiledSpec, kind: int) -> bool:
    """Any entry in the type's leap, slide, hop or sweep tables, in any frame, anywhere."""
    return any(
        table[frame][kind][square]
        for frame in range(compiled.n_frames)
        for table in (compiled.leaps, compiled.slides, compiled.hops, compiled.sweeps)
        for square in range(compiled.n_squares)
    )


def _ep_live(compiled: CompiledSpec) -> bool:
    return any(passes for side in compiled.ep_passes for passes in side)


def reading(claim: Claim, compiled: CompiledSpec) -> Reading | None:
    """Whether the compiled spec does what ``claim`` says. ``None``: not a claim after all.

    The one ``None`` is a count of a piece nobody holds in hand: *"each side has two
    knights"* is about the army, not the supply, and C reads only stocked pieces.
    """
    names = compiled.piece_names
    kind = compiled.piece_ids.index(claim.piece) if claim.piece else -1
    name = names[kind] if claim.piece else ""

    if claim.family == "A":
        capturing = capturing_types(compiled)
        takers = [
            names[t]
            for t in range(compiled.n_types)
            if capturing[t] and compiled.capture_allowed[t][kind]
        ]
        return Reading(
            not takers,
            f"{_listed(takers)} may capture {_a(name)} "
            "(none of them has a `captures` list that leaves it out)",
            f"give each piece that can capture a `captures` list without the {name}",
        )

    if claim.family == "B":
        return Reading(
            not _has_moves(compiled, kind),
            f"the rules give {_a(name)} moves (its move tables are not empty)",
            f"give the {name} no `moves`",
        )

    drops = compiled.drops
    stock = drops.stock if drops is not None else ()
    unlimited = drops.unlimited if drops is not None else ()
    if claim.family == "C" and claim.says == "unlimited":
        held = ", ".join(f"{names[k]}: {n}" for k, n in enumerate(stock) if n)
        return Reading(
            any(unlimited),
            "no piece's supply is unlimited: "
            + (f"each player starts with {held}" if held else "nothing is held at first"),
            'make the stock of the piece it means "unlimited" in `rules.drops.stock`',
        )
    if claim.family == "C":
        endless = bool(unlimited) and unlimited[kind]
        if not endless and not (stock and stock[kind]):
            return None
        return Reading(
            not endless and stock[kind] == claim.number,
            f"each player's supply of the {name} is "
            + ("unlimited" if endless else f"{stock[kind]}"),
            f"set the {name}'s entry in `rules.drops.stock` to {claim.number}",
        )

    threshold = compiled.repetition_draw
    never = threshold == NO_REPETITION_DRAW
    if claim.family == "D" and claim.says == "never":
        return Reading(
            never,
            f"a position that occurs {threshold} times is a draw "
            f"(`rules.repetition_draw` is {threshold})",
            'set `rules.repetition_draw` to "never"',
        )
    if claim.family == "D":
        fold = claim.number
        return Reading(
            not never and (not fold or threshold == fold),
            'the rules never draw by repetition (`rules.repetition_draw` is "never")'
            if never
            else f"a position that occurs {threshold} times is a draw, not {fold}",
            f"set `rules.repetition_draw` to {fold or 'the number of occurrences it means'}",
        )

    live = compiled.en_passant and _ep_live(compiled)
    if claim.says == "none":
        return Reading(
            not live,
            "the rules have en passant: `rules.en_passant` is true and a double step "
            "opens a window for it",
            "set `rules.en_passant` to false",
        )
    return Reading(
        live,
        "the rules have no en passant: "
        + (
            "`rules.en_passant` is true, but no move opens a window for it"
            if compiled.en_passant
            else "`rules.en_passant` is false"
        ),
        "set `rules.en_passant` to true, with a double step for it to answer",
    )


def message(label: str, claim: Claim, found: Reading) -> str:
    """What a person reads when the rules contradict the prose (the record's draft)."""
    return (
        f'{label}: the {claim.where} says "{claim.sentence}", but {found.rules}. '
        f"By the decision of {DECIDED} the rules are the defect: {found.remedy}. "
        "This changes the game's mechanics hash, so it needs re-measuring and its row in "
        "tests/test_schema_nine.py::FROZEN re-freezing. If the sentence does not claim "
        "this, the keyword reading in tests/test_description_claims.py has misread it: "
        "fix the reading, not the rules."
    )


def contradictions(
    label: str, spec: dict[str, Any], compiled: CompiledSpec
) -> list[tuple[Claim, str]]:
    """Every claim the prose makes and the rules do not keep, with what to tell a person."""
    wrong = []
    for claim in claims(spec):
        found = reading(claim, compiled)
        if found is not None and not found.holds:
            wrong.append((claim, message(label, claim, found)))
    return wrong


# ------------------------------------------------------------------ the library


@pytest.fixture(scope="module")
def compiled_of() -> Iterator[Callable[[Path], CompiledSpec]]:
    """Each library spec compiled at most once in this module, and only when asked."""
    kept: dict[Path, CompiledSpec] = {}

    def compiled(path: Path) -> CompiledSpec:
        if path not in kept:
            kept[path] = compile_spec(load(path))
        return kept[path]

    yield compiled
    kept.clear()


# The library specs whose prose claims something: keyword reading only, so it is cheap at
# collection. The rest have nothing for the rules to keep.
CLAIMING = [path for path in LIBRARY if claims(load(path))]


@pytest.mark.parametrize("path", CLAIMING, ids=label_of)
def test_what_the_prose_claims_the_rules_do(
    path: Path, compiled_of: Callable[[Path], CompiledSpec]
) -> None:
    spec = load(path)
    wrong = contradictions(label_of(path), spec, compiled_of(path))
    assert not wrong, "\n\n".join(text for _claim, text in wrong)


# What the record read in these five families on the 71 specs of 2026-09-24: 16 claims on
# 12 specs, all true.
# Claims may be added, but none of these may be lost. If one is lost, either its prose was
# reworded, so take it off this list and say when, or the reading above broke.
# Added since: intransitive's "There is no draw by repetition" (Q428, 2026-09-25) under D.
RECORD_READ = {
    "A": {"rulesets/rps-hierarchy-chess.json", "archive/stones.json", "archive/three-stones.json"},
    "B": {"archive/stones.json", "archive/three-stones.json"},
    "C": {"archive/stones.json", "archive/three-stones.json"},
    "D": {"archive/perpetual.json", "rulesets/intransitive.json"},
    "E": {
        "rulesets/almost.json", "rulesets/berolina.json", "rulesets/cylinder.json",
        "rulesets/gardner.json", "rulesets/losalamos.json", "rulesets/makruk.json",
        "rulesets/shatranj.json", "archive/hex-glinski-hexagon.json",
    },
}  # fmt: skip


def test_the_claims_the_record_read_are_still_read() -> None:
    read: dict[str, set[str]] = {family: set() for family in RECORD_READ}
    for path in LIBRARY:
        for claim in claims(load(path)):
            read[claim.family].add(label_of(path))
    lost = {family: sorted(RECORD_READ[family] - read[family]) for family in RECORD_READ}
    assert not any(lost.values()), f"claims the record read and this check no longer does: {lost}"


# ------------------------------------------------------------------ the case it is for

# The two stone showcases as Q101 committed them (`c064a475`, 2026-09-07): the name and the
# description word for word, and the mechanics rebuilt from today's file by taking off the
# `captures` lists that Q131 added (`9f8eb517`), which was the whole repair. The hashes are
# history's, so this fixture is the defect as committed, not a copy of it.
BEFORE_REPAIR = {
    "archive/stones.json": (
        "d9a411e36af4",
        "stones",
        "Chess in which either player may, instead of moving, place a stone on any empty "
        "square. A stone belongs to whoever placed it, never moves, is never captured and "
        "never leaves: it is a square taken out of the game. The supply is unlimited, so the "
        "choice is always available and always costs a tempo, and a board that fills with "
        "stones is a board with fewer squares to mate on.",
    ),
    "archive/three-stones.json": (
        "707d1f6f1dbc",
        "three stones",
        "Chess in which either player has three stones and may, instead of moving, place one "
        "on any empty square. A stone never moves, is never captured and never leaves, so a "
        "placed stone is a square taken out of the game for good -- and there are only three "
        "of them, so the question is not whether to place one but when.",
    ),
}

# The failure `archive/stones.json` would have raised on `c064a475`, as the record drafts it.
STONES_BEFORE_REPAIR = (
    'archive/stones.json: the description says "A stone belongs to whoever placed it, never '
    'moves, is never captured and never leaves", but pawn, knight, bishop, rook, queen and '
    "king may capture a stone (none of them has a `captures` list that leaves it out). By the "
    "decision of 2026-09-08 the rules are the defect: give each piece that can capture a "
    "`captures` list without the stone. This changes the game's mechanics hash, so it needs "
    "re-measuring and its row in tests/test_schema_nine.py::FROZEN re-freezing. If the "
    "sentence does not claim this, the keyword reading in tests/test_description_claims.py "
    "has misread it: fix the reading, not the rules."
)


def as_committed_before_repair(label: str) -> dict[str, Any]:
    history, name, description = BEFORE_REPAIR[label]
    spec = load(ROOT / label)
    for piece in spec["pieces"]:
        piece.pop("captures", None)
    spec["name"], spec["description"] = name, description
    assert spec_hash(spec)[:12] == history, (
        f"{label} no longer rebuilds to its pre-repair hash {history}: its mechanics moved "
        "after Q131. Rebuild this fixture from the file at `c064a475` instead."
    )
    return spec


@pytest.mark.parametrize("label", sorted(BEFORE_REPAIR))
def test_a_stone_showcase_as_committed_before_its_repair_fails(
    label: str, compiled_of: Callable[[Path], CompiledSpec]
) -> None:
    before = as_committed_before_repair(label)
    wrong = contradictions(label, before, compile_spec(before))
    assert [claim.family for claim, _text in wrong] == ["A"], wrong
    claim, text = wrong[0]
    assert "is never captured" in claim.sentence
    assert "pawn, knight, bishop, rook, queen and king may capture a stone" in text
    if label == "archive/stones.json":
        assert text == STONES_BEFORE_REPAIR

    # The repair clears it, and every other claim held before and holds after.
    after = load(ROOT / label)
    assert contradictions(label, after, compiled_of(ROOT / label)) == []
    assert {c.family for c in claims(after)} == {"A", "B", "C"}


# ------------------------------------------------------------------ every family can fail


def _prose(path: Path, sentence: str) -> dict[str, Any]:
    """The spec at ``path`` with ``sentence`` for its description: its mechanics unchanged."""
    spec = load(path)
    spec["description"] = sentence
    return spec


def _mover(path: Path) -> str:
    return str(next(piece["name"] for piece in load(path)["pieces"] if piece.get("moves")))


def _stocked(path: Path) -> str:
    return str(next(iter(load(path)["rules"]["drops"]["stock"])))


def _name_of(path: Path, piece: str) -> str:
    return str(next(p["name"] for p in load(path)["pieces"] if p["id"] == piece))


STONES = ROOT / "archive" / "stones.json"
THREE_STONES = ROOT / "archive" / "three-stones.json"
PERPETUAL = ROOT / "archive" / "perpetual.json"
SHATRANJ = ROOT / "rulesets" / "shatranj.json"

# ``(spec, a sentence its rules contradict, the family, what the message must say)``. Each
# is the family's fail-before: the library test would fail if a spec said this.
FALSE_SENTENCES: list[tuple[Path, Callable[[], str], str, str]] = [
    (STONES, lambda: f"A {_mover(STONES)} never moves.", "B", "moves (its move tables"),
    (THREE_STONES, lambda: "The supply is unlimited.", "C", "no piece's supply is unlimited"),
    (
        STONES,
        lambda: f"Either player has three {_name_of(STONES, _stocked(STONES))}s.",
        "C",
        "is unlimited",
    ),
    (STONES, lambda: "There is no draw by repetition.", "D", "occurs 3 times is a draw"),
    (PERPETUAL, lambda: "Threefold repetition is a draw.", "D", "never draw by repetition"),
    (STONES, lambda: "There is no en passant.", "E", "the rules have en passant"),
    (SHATRANJ, lambda: "A double step may be taken en passant.", "E", "no en passant"),
]


@pytest.mark.parametrize(
    ("path", "sentence", "family", "said"),
    FALSE_SENTENCES,
    ids=[f"{family}-{path.stem}" for path, _s, family, _t in FALSE_SENTENCES],
)
def test_every_family_fails_on_a_sentence_its_rules_contradict(
    path: Path,
    sentence: Callable[[], str],
    family: str,
    said: str,
    compiled_of: Callable[[Path], CompiledSpec],
) -> None:
    text = sentence()
    wrong = contradictions(label_of(path), _prose(path, text), compiled_of(path))
    assert [claim.family for claim, _text in wrong] == [family], wrong
    assert wrong[0][1].startswith(f'{label_of(path)}: the description says "{text}", but ')
    assert said in wrong[0][1]
