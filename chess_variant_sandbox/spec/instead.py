"""What a refused draft could be instead: the nearest game the validator allows (R-L3, B2).

`docs/reference/refusal-offers.md` §4. A refusal says *no*; this module finds the nearest
thing that is allowed by **search**, never by reading the refusal: a small closed set of edits
to the refused draft (leave a rule out, leave a part of one out, give a listed value another of
its listed values), every candidate judged by `validate_spec` and the survivors ranked by
`relate.units`, the project's own count of how many rules two games differ in. Nothing here is
written per refusal, and no sentence is read: the validator's problems are compared as sets of
opaque strings (§2 rule 3), so a reworded sentence keeps its offers.

Pure and spec-level: it compiles nothing and plays nothing, and imports nothing of `web`.

The edits (§4.1), each generated from the draft and the schema's own data:

* **V** a stated value with a listed set of values (an ``enum``, a ``const``, a boolean, in
  `jsonschema.rule_spec_schema`) is given another of them: inside ``rules``, and
  ``board.lattice``, ``board.shape``, ``setup.orientation``. A key's off value
  (`schema.*_OFF_VALUES`) is its absence written out, which the spelling's tidy drops, so it is
  reached by the S and P edits that leave the key out.
* **S** one stated key inside a rule is left out (not one the schema requires), or one entry of
  a list of objects inside a rule.
* **P** one stated key of a piece or of a move is left out, on one piece (one move) and on
  every one that states it at once; one entry of a piece's ``sight``.
* **R** one stated rule is left out whole (not ``win``, which the schema requires).
* **D** an edit a problem's `refusals.toml` entry declares (`Entry.instead`, §4.4).

No edit adds a rule, changes a number, moves or removes a piece, changes a vector or resizes
the board: a move's geometry (`_GEOMETRY`) and a piece's identity (`_IDENTITY`) are never
edited. Every candidate is tidied by `spelling.current_spelling` before it is judged.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from chess_variant_sandbox.spec.describe import describe_changes
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.refusals import entry_of
from chess_variant_sandbox.spec.relate import identity_key, units
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.spec.spelling import current_spelling

# §4.2's constants, named and pinned (`tests/test_instead.py`).
LEVELS = 3  # at most three edits deep
BEAM = 4  # a level with no offer goes on from its four best candidates
VALIDATIONS = 2_000  # the walk stops after this many judgements of the validator
OFFERS = 3  # at most three offers are returned

# §4.3: a changed value keeps both rules, a part left out keeps most of both, a rule left out
# keeps one. A declared edit ranks by the same numbers as any, and last among equals.
CLASSES = ("V", "S", "P", "R", "D")

# A move's geometry: never edited, inside a piece's moves or a portal group's.
_GEOMETRY = frozenset({
    "kind", "vectors", "symmetry", "range", "min_range", "legs", "sweep_range", "screen_range",
})
# What names or prices a piece rather than giving it a rule.
_IDENTITY = frozenset({"id", "name", "value", "moves"})
# The places outside `rules` where V gives a listed value another.
_SETTINGS = (("board", "lattice"), ("board", "shape"), ("setup", "orientation"))

Path = tuple[Any, ...]  # keys and list indices; ``EVERY`` fans out over a list
EVERY = "[]"
Edit = tuple[Any, ...]  # ("drop", path) | ("set", path, value), path written as a string


@dataclass(frozen=True)
class Offer:
    """One allowed draft near the refused one: the draft, the edits that make it from the
    trial, how many rules it differs in (`relate.units`) and the page's sentences for it."""

    draft: dict[str, Any]
    edits: tuple[Edit, ...]
    units: int
    changes: tuple[str, ...]


@dataclass(frozen=True)
class Search:
    """What a walk found, and what it cost."""

    offers: tuple[Offer, ...]
    validations: int
    levels: int


# ------------------------------------------------------------------------------ paths


def path_text(path: Path) -> str:
    """``("rules", "portals", 0, "kind")`` as ``rules.portals[0].kind``."""
    out = ""
    for part in path:
        if part == EVERY:
            out += "[]"
        elif isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f".{part}" if out else str(part)
    return out


def parse_path(text: str) -> Path:
    """The inverse of `path_text`, read character by character (no pattern over words)."""
    parts: list[Any] = []
    word = ""
    index = 0
    while index < len(text):
        char = text[index]
        if char == ".":
            if word:
                parts.append(word)
            word = ""
        elif char == "[":
            if word:
                parts.append(word)
            word = ""
            close = text.index("]", index)
            inside = text[index + 1:close]
            parts.append(EVERY if inside == "" else int(inside))
            index = close
        else:
            word += char
        index += 1
    if word:
        parts.append(word)
    return tuple(parts)


def edit_text(edit: Edit) -> str:
    """An edit as the command line prints it: ``drop rules.en_passant``,
    ``set rules.win = "royal_capture"``."""
    if edit[0] == "drop":
        return f"drop {edit[1]}"
    return f"set {edit[1]} = {json.dumps(edit[2])}"


def _apply(node: Any, path: Path, edit: Edit) -> bool:
    """Apply ``edit`` at ``path`` under ``node`` in place; whether anything changed."""
    head, rest = path[0], path[1:]
    if head == EVERY:
        if not isinstance(node, list):
            return False
        return any([_apply(node, (index, *rest), edit) for index in range(len(node))])
    if isinstance(head, int):
        if not isinstance(node, list) or not 0 <= head < len(node):
            return False
    elif not isinstance(node, dict) or head not in node:
        return False
    if rest:
        return _apply(node[head], rest, edit)
    if edit[0] == "drop":
        del node[head]
        return True
    if node[head] == edit[2] and type(node[head]) is type(edit[2]):
        return False
    node[head] = edit[2]
    return True


def apply_edit(draft: dict[str, Any], edit: Edit) -> dict[str, Any] | None:
    """``draft`` with one edit applied and tidied (`spelling.current_spelling`), or None where
    the edit changes nothing in it."""
    out = json.loads(json.dumps(draft))
    if not _apply(out, parse_path(edit[1]), edit):
        return None
    return current_spelling(out)


def apply_edits(trial: dict[str, Any], edits: tuple[Edit, ...]) -> dict[str, Any] | None:
    """The trial with each edit applied in turn, tidied after each, as the walk makes it."""
    draft: dict[str, Any] | None = current_spelling(trial)
    for edit in edits:
        draft = apply_edit(draft, edit) if draft is not None else None
    return draft


def states(draft: dict[str, Any], wanted: str | None) -> bool:
    """Whether ``draft`` states the key path ``wanted`` (always, where none is wanted)."""
    if wanted is None:
        return True
    node: Any = draft
    for part in parse_path(wanted):
        if isinstance(part, int):
            if not isinstance(node, list) or not 0 <= part < len(node):
                return False
        elif not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


# ------------------------------------------------------------------------- the schema


@functools.cache
def _schema() -> dict[str, Any]:
    return rule_spec_schema()


def _branch(node: Any, value: Any) -> dict[str, Any] | None:
    """The branch of ``node`` that describes ``value``'s JSON shape (an ``anyOf`` is read by
    its branches' types)."""
    if not isinstance(node, dict):
        return None
    if "anyOf" not in node:
        return node
    want = "object" if isinstance(value, dict) else "array" if isinstance(value, list) else None
    for branch in node["anyOf"]:
        if want is not None and branch.get("type") == want:
            return dict(branch)
    return None if want is not None else node


def _scalar_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    return type(value).__name__


def _listed(node: Any) -> list[Any]:
    """Every value the schema lists for a scalar at ``node``: its enums, consts and, for a
    boolean, both. In the schema's order, without repeats (``0`` is not ``false``)."""
    out: list[Any] = []
    for branch in (node.get("anyOf") or [node]) if isinstance(node, dict) else []:
        found = list(branch.get("enum", ()))
        if "const" in branch:
            found.append(branch["const"])
        kinds = branch.get("type")
        if kinds == "boolean" or (isinstance(kinds, list) and "boolean" in kinds):
            found += [True, False]
        for value in found:
            if not any(value == seen and type(value) is type(seen) for seen in out):
                out.append(value)
    return out


def _alternatives(node: Any, value: Any) -> list[Any]:
    """The other listed values of a stated scalar, or none where it is not one of them."""
    listed = _listed(node)
    if not any(value == seen and _scalar_type(value) == _scalar_type(seen) for seen in listed):
        return []
    return [
        other for other in listed
        if not (other == value and _scalar_type(other) == _scalar_type(value))
    ]


def _in_moves(path: Path) -> bool:
    return "moves" in path


def _keys_in_order(value: dict[str, Any], node: dict[str, Any]) -> list[str]:
    """The keys ``value`` states that ``node`` names, in the schema's order."""
    return [key for key in node.get("properties", {}) if key in value]


# -------------------------------------------------------------------------- the edits


def _values(value: Any, node: Any, path: Path) -> Iterator[Edit]:
    """V: every stated scalar under ``path`` given each other listed value."""
    node = _branch(node, value)
    if node is None:
        return
    if isinstance(value, dict):
        for key in _keys_in_order(value, node):
            if key == "symmetry" or (key in _GEOMETRY and _in_moves(path)):
                continue
            yield from _values(value[key], node["properties"][key], (*path, key))
    elif isinstance(value, list):
        items = node.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                yield from _values(item, items, (*path, index))
    else:
        for other in _alternatives(node, value):
            yield ("set", path_text(path), other)


def _parts(value: Any, node: Any, path: Path) -> Iterator[Edit]:
    """S: under a rule, each stated key the schema names and does not require, and each entry
    of a list of objects, left out; then the same inside each."""
    node = _branch(node, value)
    if node is None:
        return
    if isinstance(value, dict):
        required = set(node.get("required", ()))
        keys = [
            key for key in _keys_in_order(value, node)
            if not (key in _GEOMETRY and _in_moves(path))
        ]
        for key in keys:
            if key not in required:
                yield ("drop", path_text((*path, key)))
        for key in keys:
            yield from _parts(value[key], node["properties"][key], (*path, key))
    elif isinstance(value, list):
        items = node.get("items")
        if not isinstance(items, dict) or not all(isinstance(item, dict) for item in value):
            return
        if len(value) > max(1, int(node.get("minItems", 0))):
            for index in range(len(value)):
                yield ("drop", path_text((*path, index)))
        # A key every entry states that states it, left out of all at once (as P does across
        # pieces): two portal groups each barred straight back are one bar to leave out.
        required = set(items.get("required", ())) if isinstance(items, dict) else set()
        for key in items.get("properties", {}):
            stating = sum(1 for item in value if key in item)
            if stating > 1 and key not in required and not (key in _GEOMETRY and _in_moves(path)):
                yield ("drop", path_text((*path, EVERY, key)))
        for index, item in enumerate(value):
            yield from _parts(item, items, (*path, index))


def _piece_keys(pieces: list[Any], node: dict[str, Any]) -> Iterator[Edit]:
    """P: each stated key of a piece left out, on one piece, then on every one at once."""
    required = set(node.get("required", ()))
    for key in node.get("properties", {}):
        if key in required or key in _IDENTITY:
            continue
        stating = [index for index, piece in enumerate(pieces) if isinstance(piece, dict) and key in piece]
        for index in stating:
            yield ("drop", path_text(("pieces", index, key)))
        if len(stating) > 1:
            yield ("drop", path_text(("pieces", EVERY, key)))
    for index, piece in enumerate(pieces):
        sight = piece.get("sight") if isinstance(piece, dict) else None
        if isinstance(sight, list) and len(sight) > 1:
            for entry in range(len(sight)):
                yield ("drop", path_text(("pieces", index, "sight", entry)))


def _move_keys(pieces: list[Any], node: dict[str, Any]) -> Iterator[Edit]:
    """P: each stated key of a move left out, on one move, then on every move at once."""
    move_node = node.get("properties", {}).get("moves", {}).get("items", {})
    required = set(move_node.get("required", ()))
    for key in move_node.get("properties", {}):
        if key in required or key in _GEOMETRY:
            continue
        stating = [
            (index, slot)
            for index, piece in enumerate(pieces) if isinstance(piece, dict)
            for slot, move in enumerate(piece.get("moves") or ())
            if isinstance(move, dict) and key in move
        ]
        for index, slot in stating:
            yield ("drop", path_text(("pieces", index, "moves", slot, key)))
        if len(stating) > 1:
            yield ("drop", path_text(("pieces", EVERY, "moves", EVERY, key)))


def edits(
    draft: dict[str, Any], problems: frozenset[str] = frozenset(), wanted: str | None = None,
) -> list[tuple[str, Edit]]:
    """Every single edit of §4.1 of ``draft``, with its class, in generation order: by class,
    then the schema's key order and the draft's list order. Never one that leaves ``wanted``
    out. ``problems`` are the draft's, for the declared edits of their entries."""
    schema = _schema()
    stated_rules, stated_pieces = draft.get("rules"), draft.get("pieces")
    rules: dict[str, Any] = stated_rules if isinstance(stated_rules, dict) else {}
    rules_node = schema["properties"]["rules"]
    pieces: list[Any] = stated_pieces if isinstance(stated_pieces, list) else []
    piece_node = schema["properties"]["pieces"]["items"]
    stated = _keys_in_order(rules, rules_node)
    found: list[tuple[str, Edit]] = []
    for part, key in _SETTINGS:
        holder = draft.get(part)
        if isinstance(holder, dict) and key in holder:
            node = schema["properties"][part]["properties"][key]
            found += [("V", edit) for edit in _values(holder[key], node, (part, key))]
    for key in stated:
        found += [("V", edit) for edit in _values(rules[key], rules_node["properties"][key], ("rules", key))]
    for key in stated:
        found += [("S", edit) for edit in _parts(rules[key], rules_node["properties"][key], ("rules", key))]
    found += [("P", edit) for edit in _piece_keys(pieces, piece_node)]
    found += [("P", edit) for edit in _move_keys(pieces, piece_node)]
    required = set(rules_node.get("required", ()))
    found += [("R", ("drop", f"rules.{key}")) for key in stated if key not in required]
    for problem in sorted(problems):
        entry = entry_of(problem)
        if entry is not None:
            found += [("D", edit) for edit in entry.instead]
    if wanted is not None:
        found = [(cls, edit) for cls, edit in found if not _leaves_out(edit, wanted)]
    return found


def _class_of_drop(path: Path) -> str:
    """The class a left-out key is by where it stands: a rule (R), a part of one (S), a
    piece's or a move's (P)."""
    if path[0] == "rules":
        return "R" if len(path) == 2 else "S"
    return "P"


def _as_done(cls: str, edit: Edit, draft: dict[str, Any]) -> tuple[str, Edit]:
    """An edit named by what it did: a value set to its key's absence written out (``false``,
    an off value), which the tidy left out, is that key left out; and a part left out that
    leaves its rule stating nothing, which the tidy left out too, is the rule left out."""
    path = parse_path(edit[1])
    if EVERY in path:
        return cls, edit
    for length in range(2, len(path) + (edit[0] == "set")):
        if not states(draft, path_text(path[:length])):
            return _class_of_drop(path[:length]), ("drop", path_text(path[:length]))
    return cls, edit


def _leaves_out(edit: Edit, wanted: str) -> bool:
    """Whether ``edit`` drops ``wanted`` or a path holding it."""
    if edit[0] != "drop":
        return False
    dropped, kept = parse_path(edit[1]), parse_path(wanted)
    return kept[:len(dropped)] == dropped


# --------------------------------------------------------------------------- the walk


@dataclass(frozen=True)
class _Candidate:
    draft: dict[str, Any]
    edits: tuple[Edit, ...]
    classes: tuple[int, ...]
    problems: frozenset[str]
    order: int


def _blob(draft: dict[str, Any]) -> str:
    return json.dumps(draft, sort_keys=True)


def _goes_on(problems: frozenset[str], before: frozenset[str]) -> bool:
    """Whether a refused candidate is a step on the way (§4.2 step 3, as built): one of the
    problems of the draft it was made from is gone. Leaving out a rule can strand keys the
    rule gave a use (a price and a bounty with no purse), which trades one problem for two;
    the next levels leave the stranded keys out, which is what §4.2 keeps its third level
    for, and a count that must fall would never reach them. The beam still prefers the
    fewest problems."""
    return not before <= problems


def search(
    trial: dict[str, Any], *, wanted: str | None = None, base: dict[str, Any] | None = None,
) -> Search:
    """The walk of §4.2, and its count of validations. `nearest` is its offers."""
    problems = frozenset(validate_spec(trial))
    if not problems:
        return Search((), 1, 0)
    start = current_spelling(trial)
    base_key = identity_key(base) if base is not None else None
    seen = {_blob(start)}
    frontier = [_Candidate(start, (), (), problems, 0)]
    validations, order = 1, 0
    level = 0
    offers: list[_Candidate] = []
    if start != trial:
        # A draft refused only in its spelling (a rule stated at its off value beside what
        # refuses it): the tidy is the nearest allowed draft, with no edit.
        validations += 1
        tidied = frozenset(validate_spec(start))
        if not tidied:
            frontier = []
            if states(start, wanted) and (base_key is None or identity_key(start) != base_key):
                offers.append(_Candidate(start, (), (), tidied, 0))
        else:
            frontier = [_Candidate(start, (), (), tidied, 0)]
    unit_cache: dict[str, int] = {}

    def units_of(candidate: _Candidate) -> int:
        blob = _blob(candidate.draft)
        if blob not in unit_cache:
            unit_cache[blob] = units(trial, candidate.draft)
        return unit_cache[blob]

    while level < LEVELS and validations < VALIDATIONS and frontier and not offers:
        level += 1
        found: list[_Candidate] = []
        for parent in frontier:
            for cls, edit in edits(parent.draft, parent.problems, wanted):
                if validations >= VALIDATIONS:
                    break
                draft = apply_edit(parent.draft, edit)
                if draft is None or not states(draft, wanted):
                    continue
                cls, edit = _as_done(cls, edit, draft)
                blob = _blob(draft)
                if blob in seen:
                    continue
                seen.add(blob)
                validations += 1
                order += 1
                candidate = _Candidate(
                    draft, (*parent.edits, edit), (*parent.classes, CLASSES.index(cls)),
                    frozenset(validate_spec(draft)), order,
                )
                if candidate.problems:
                    if _goes_on(candidate.problems, parent.problems):
                        found.append(candidate)
                elif base_key is None or identity_key(draft) != base_key:
                    offers.append(candidate)
        frontier = sorted(
            found,
            key=lambda c: (len(c.problems), units_of(c), sorted(c.classes, reverse=True), c.order),
        )[:BEAM]
    ranked = sorted(
        offers,
        key=lambda c: (units_of(c), len(c.edits), sorted(c.classes, reverse=True), c.order),
    )
    chosen: list[Offer] = []
    games: set[str] = set()
    for candidate in ranked:
        game = identity_key(candidate.draft)
        if game in games:
            continue
        games.add(game)
        chosen.append(Offer(
            candidate.draft, candidate.edits, units_of(candidate),
            tuple(describe_changes(trial, candidate.draft)),
        ))
        if len(chosen) == OFFERS:
            break
    return Search(tuple(chosen), validations, level)


def nearest(
    trial: dict[str, Any], *, wanted: str | None = None, base: dict[str, Any] | None = None,
) -> list[Offer]:
    """The nearest drafts the validator allows to the refused ``trial`` (§4), best first, at
    most `OFFERS`: each still states ``wanted`` (the key path just asked for, where one was)
    and is not ``base`` (the valid draft the trial was made from). None where the trial is
    allowed, or where no small change makes it so."""
    return list(search(trial, wanted=wanted, base=base).offers)
