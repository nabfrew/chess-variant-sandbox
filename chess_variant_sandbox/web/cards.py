"""The designer's rule cards, generated from the schema the validator enforces.

`spec.jsonschema.rule_spec_schema()` is the source. This module turns it into a *field
tree*: plain JSON that says what each control is, what bounds it, and what the schema
says about it. It says nothing about HTML. The page's `cards.py` renders the tree, and a
different renderer could consume the same tree unchanged.

The mapping from schema shape to control:

==========================================  ==========================================
schema                                      field
==========================================  ==========================================
enum, or an anyOf of finite scalars         ``menu``: typed options, a sentence each
boolean                                     ``menu`` of yes / no, blank when optional
integer / number                            ``int`` / ``number`` with the schema bounds
string naming a piece                       ``piece``: a menu of the draft's piece ids
string that is not a piece                  ``text``
``["string", "integer"]``                   ``rank``: a word or a number
array of algebraic squares                  ``squares``: picked on the board
array of piece ids / integers               ``pieces`` / ``ints``
object with properties                      ``card``: a sub-card of fields
object with additionalProperties            ``map``: one row per piece id
array of objects                            ``list``: one sub-card per item
any other anyOf                             ``either``: one field per branch
==========================================  ==========================================

A string field is a piece id or free text, and the schema does not say which. The two
tables below say it, and `fields` raises on a string neither names, so a schema bump that
adds one fails a test here rather than rendering a text box where a menu belongs.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.plain import KIND_WORDS, field_words
from chess_variant_sandbox.spec.schema import SHAPES, WRAP_PAIRS, wrap_axes

# Paths are dotted from the spec's top, with `[]` for "every item of".
PIECE_PATHS = frozenset({
    "pieces[].promoted_from",
    "pieces[].captures",
    "setup.placements[].piece",
    "rules.promotion.piece",
    "rules.promotion.to",
    "rules.promotion.to_by_player.0",
    "rules.promotion.to_by_player.1",
    "rules.promotion[].piece",
    "rules.promotion[].to",
    "rules.promotion[].to_by_player.0",
    "rules.promotion[].to_by_player.1",
    "rules.castling.king",
    "rules.castling.rook",
    "rules.strike.chain[].from",
    "rules.strike.chain[].to",
    "rules.strike.immune",
    "rules.strike.switches_sides",
    "rules.contacts.links[].mover",
    "rules.contacts.links[].onto",
    "rules.contacts.links[].mover_to",
    "rules.contacts.links[].onto_to",
    "rules.contacts.immune",
    "rules.npcs[].piece",
    "rules.npcs[].immune",
    "rules.portals[].piece",
    # Schema 21: which pieces may go in by an end.
    "rules.portals[].carries.orange",
    "rules.portals[].carries.blue",
    "rules.portals[].ends[].carries",
    "rules.reach.piece",
    "rules.extinction.pieces",
    "rules.drops.no_doubled",
    "rules.drops.no_mate",
    "rules.explode_on_capture.immune",
    "rules.explode_on_capture.by",
    "rules.explode_on_capture.of",
})

TEXT_PATHS = frozenset({
    "rules.castling.king_file.kingside",
    "rules.castling.king_file.queenside",
    "rules.castling.rook_file.kingside",
    "rules.castling.rook_file.queenside",
    # Schema 19: one square each, typed as written until P3's portal tool places them
    # on the board (`docs/plan-portals.md` D2).
    "rules.portals[].orange",
    "rules.portals[].blue",
    # Schema 21: a map's end, typed as written until Q843b's designer places it.
    "rules.portals[].ends[].at",
})

# What the hand-written form already edits. Every other key gets a generated field.
OWNED = {
    "piece": frozenset({"id", "name", "value", "royal", "moves"}),
    "move": frozenset({"kind", "vectors", "symmetry", "range", "min_range", "mode",
                       "condition.from_rank"}),
    "placement": frozenset({"piece", "player", "squares"}),
}

_SQUARES_SUFFIX = "Algebraic squares such as a1 or e4."
_MENU_LIMIT = 40
_PLAYER_LABELS = {"0": "first player", "1": "second player"}


def card_model(corpus: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
    """The field trees the designer renders, and the order *add a rule* offers them in."""
    schema = rule_spec_schema(strict=True)
    rules = schema["properties"]["rules"]
    piece = schema["properties"]["pieces"]["items"]
    move = piece["properties"]["moves"]["items"]
    placement = schema["properties"]["setup"]["properties"]["placements"]["items"]
    counts = rule_counts(corpus)
    order = sorted(rules["properties"], key=lambda key: (-counts.get(key, 0), key))
    rule_fields = fields(rules, "rules", owned=frozenset(), required_all=True)
    _word_the_sides(rule_fields)
    return {
        "rules": rule_fields,
        "shapeKeys": shape_keys(),
        "piece": fields(piece, "pieces[]", owned=OWNED["piece"]),
        "move": fields(move, "pieces[].moves[]", owned=OWNED["move"]),
        "placement": fields(placement, "setup.placements[]", owned=OWNED["placement"]),
        "ruleOrder": order,
        "ruleCounts": {key: counts.get(key, 0) for key in order},
        "rare": [key for key in order if not counts.get(key, 0)],
        "fixed": list(rules["required"]),
    }


def shape_keys() -> dict[str, dict[str, list[str]]]:
    """Which keys of a card belong to which board shape, keyed by the card's dotted path.

    `rules.wrap` states a grid's seams under `files` and `ranks` and a hexagon's under its
    three pairs of sides (schema 16), and the validator refuses each shape's keys on the
    other. The card shows a board only its own shape's rows, three on a hexagon, and the
    page reads which those are here rather than keeping a list of its own. Each key comes
    with its `_by_player` twin, from `schema.wrap_axes`, the one list of them.
    """
    return {
        "rules.wrap": {
            shape: [
                spelled
                for key, _ in wrap_axes(shape)
                for spelled in (key, f"{key}_by_player")
            ]
            for shape in SHAPES
        }
    }


# A hexagon's sides in words, as the board draws them: the first player at the bottom,
# flat-topped. `schema.WRAP_PAIRS` names them as a hex map does.
_SIDE_WORDS = {
    "n": "top", "s": "bottom", "ne": "upper right", "sw": "lower left",
    "nw": "upper left", "se": "lower right",
}


def _word_the_sides(rule_fields: list[dict[str, Any]]) -> None:
    """The wrap card's hexagon rows, in the words the board is drawn in.

    The schema's own labels are its keys, ``n s`` and a menu of ``none``, ``s``, ``n``,
    ``both``. The value written is unchanged: this relabels the three pairs and their
    per-player twins as *top and bottom*, and each value as *not joined*, *both ways* or
    *one way*, out by the side it names. A grid's rows keep the labels they had.
    """
    wrap = next((field for field in rule_fields if field["key"] == "wrap"), None)
    if wrap is None:
        return
    for field in wrap["fields"]:
        pair = WRAP_PAIRS.get(field["key"].removesuffix("_by_player"))
        if pair is None:
            continue
        low, high = _SIDE_WORDS[pair.low], _SIDE_WORDS[pair.high]
        field["label"] = f"{high} and {low}" + (
            ", by player" if field["key"].endswith("_by_player") else ""
        )
        menus = field["fields"] if field["control"] == "card" else [field]
        for menu in menus:
            for option in menu["options"]:
                value = option["value"]
                option["label"] = (
                    "not joined" if value == "none"
                    else "both ways" if value == "both"
                    else f"one way, out by the {_SIDE_WORDS[value]}"
                )


def rule_counts(corpus: Iterable[dict[str, Any]]) -> Counter[str]:
    """How many rulesets state each rule key."""
    counts: Counter[str] = Counter()
    for spec in corpus:
        counts.update((spec.get("rules") or {}).keys())
    return counts


def corpus_specs(*dirs: Path) -> list[dict[str, Any]]:
    """Every ruleset under `dirs`, read as JSON. What cannot be read is skipped."""
    found = []
    for directory in dirs:
        for path in sorted(directory.glob("*.json")):
            try:
                spec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(spec, dict):
                found.append(spec)
    return found


def fields(
    node: dict[str, Any],
    path: str,
    owned: frozenset[str],
    required_all: bool = False,
    prefix: str = "",
) -> list[dict[str, Any]]:
    """One field per property of an object schema, skipping the hand-owned ones."""
    required = set(node.get("required", []))
    out = []
    for key, sub in node["properties"].items():
        dotted = prefix + key
        if dotted in owned:
            continue
        partial = sorted(o[len(dotted) + 1:] for o in owned if o.startswith(dotted + "."))
        field = field_for(sub, f"{path}.{key}", key, required_all or key in required)
        if partial and field["control"] == "card":
            field["fields"] = [f for f in field["fields"] if f["key"] not in partial]
            field["partial"] = True
        out.append(field)
    return out


def field_for(node: dict[str, Any], path: str, key: str, required: bool) -> dict[str, Any]:
    """The field for one schema node."""
    # Each field by the designer's name for it, and the schema's description (written for
    # whoever writes the text) replaced where a visitor would read a key in it (D40).
    words = field_words(path)
    base: dict[str, Any] = {
        "key": key,
        "label": _PLAYER_LABELS.get(key) or (words[0] if words else key.replace("_", " ")),
        "required": required,
    }
    if words and words[1]:
        base["says"] = words[1]
    elif node.get("description"):
        base["says"] = node["description"]

    if "anyOf" in node:
        options = _finite(node["anyOf"])
        if options is not None:
            return {**base, "control": "menu", "options": options}
        return {**base, "control": "either", "branches": [
            {"label": _branch_label(branch), "field": field_for(branch, path, key, True)}
            for branch in node["anyOf"]
        ]}

    kind = node.get("type")
    if "enum" in node:
        return {**base, "control": "menu",
                "options": _options(node["enum"], node.get("description", ""))}
    if kind == "boolean":
        return {**base, "control": "menu", "options": _options([True, False], "")}
    if kind == ["string", "integer"]:
        return {**base, "control": "rank"}
    if kind in ("integer", "number"):
        return {**base, "control": "int" if kind == "integer" else "number",
                **_bounds(node)}
    if kind == "string":
        if path in PIECE_PATHS:
            return {**base, "control": "piece"}
        if path in TEXT_PATHS:
            return {**base, "control": "text"}
        raise KeyError(f"{path}: a string field neither PIECE_PATHS nor TEXT_PATHS names")
    if kind == "array":
        items = node["items"]
        # A list of objects, or of integer pairs: `strike.landing.vectors` (schema 12) is
        # the one rule field whose items are arrays, and each pair is an `ints` field.
        if items.get("type") in ("object", "array"):
            return {**base, "control": "list",
                    "item": field_for(items, path + "[]", key, True)}
        if items.get("type") == "integer":
            return {**base, "control": "ints", **_bounds(items)}
        if items.get("type") == "string":
            if node.get("description", "").endswith(_SQUARES_SUFFIX):
                return {**base, "control": "squares"}
            if path in PIECE_PATHS:
                return {**base, "control": "pieces"}
        raise KeyError(f"{path}: an array the card model has no control for")
    if kind == "object":
        if "properties" in node:
            return {**base, "control": "card",
                    "fields": fields(node, path, owned=frozenset())}
        value = node["additionalProperties"]
        return {**base, "control": "map",
                "value": field_for(value, path + ".*", "value", True)}
    raise KeyError(f"{path}: a schema node the card model has no control for")


def option_sentences(description: str, values: list[Any]) -> dict[str, str]:
    """Split "a: says this. b: says that." into one sentence per enum value.

    Only where every value is introduced that way. Anything else stays the field's own
    description, shown once under the menu.
    """
    names = [str(v) for v in values]
    if not names or not all(isinstance(v, str) for v in values):
        return {}
    pattern = r"(?:^|(?<=\.\s))(" + "|".join(map(re.escape, names)) + r"):\s"
    marks = list(re.finditer(pattern, description))
    if sorted(m.group(1) for m in marks) != sorted(names):
        return {}
    out = {}
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(description)
        out[mark.group(1)] = description[mark.end():end].strip()
    return out


def _options(values: list[Any], description: str) -> list[dict[str, Any]]:
    said = option_sentences(description, values)
    return [{"value": v, "label": _value_label(v), **({"says": said[v]} if v in said else {})}
            for v in values]


# Values whose spelling names a thing the page never shows: a portal pair's ends are drawn
# alike, in the pair's colour, so a menu says which end by the order they were put down
# in, never by the file's keys (Q838d, `docs/plan-portal-look.md` §5).
_VALUE_WORDS: dict[str, str] = {
    "orange_to_blue": "one way, first end to second",
    "blue_to_orange": "one way, second end to first",
}


def _value_label(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if isinstance(value, str) and value in _VALUE_WORDS:
        return _VALUE_WORDS[value]
    return str(value).replace("_", " ")


def _finite(branches: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """An anyOf whose branches are all small closed sets, as one list of options."""
    values: list[Any] = []
    for branch in branches:
        if "const" in branch:
            values.append(branch["const"])
        elif "enum" in branch:
            values.extend(branch["enum"])
        elif branch.get("type") == "boolean":
            values.extend([True, False])
        elif branch.get("type") == "integer" and "minimum" in branch and "maximum" in branch:
            span = range(branch["minimum"], branch["maximum"] + 1)
            if len(span) > _MENU_LIMIT:
                return None
            values.extend(span)
        else:
            return None
    return _options(values, "") if len(values) <= _MENU_LIMIT else None


def _branch_label(branch: dict[str, Any]) -> str:
    if "const" in branch:
        return str(branch["const"])
    if "enum" in branch:
        return " or ".join(str(v) for v in branch["enum"])
    if branch.get("type") == "object":
        return "one"
    if branch.get("type") == "array":
        return "several" if branch["items"].get("type") == "object" else "a list"
    kind = branch.get("type", "other")
    return KIND_WORDS.get(kind, str(kind)) if isinstance(kind, str) else "other"


def _bounds(node: dict[str, Any]) -> dict[str, Any]:
    return {k: node[k] for k in ("minimum", "maximum") if k in node}
