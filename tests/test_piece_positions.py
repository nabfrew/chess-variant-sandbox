"""`schema.PIECE_POSITIONS`: the one map of where a rule names a piece (CP2).

Two readers walk it -- `relate` renames ids at its positions (and `likeness` reads them),
`transform.drop_piece` takes a removed piece out of them -- so a position the map lacks is
missed by both at once, and nothing else would notice until a design run drew a child the
validator refused. These tests hold the map against the JSON Schema the model is handed,
which `tests/test_spec.py` holds against the validator's own key sets: every string-typed
field there is either a position in the map or named below as not a piece.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from chess_variant_sandbox.spec import load_spec, validate_spec
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.schema import (
    ANY_KEY,
    PIECE_FIELDS,
    PIECE_KEYED,
    PIECE_POSITIONS,
    PORTAL_OLD_PIECE_KEY,
    Each,
    Fields,
    Names,
)
from chess_variant_sandbox.spec.transform import drop_piece

ROOT = Path(__file__).resolve().parent.parent
KEY = "<KEY>"  # a map's keys, in a path below

# Every string-typed field of the JSON Schema that is not a piece id, by path (list items
# and the bare-object spelling of a list both read without ``[]``). A new field lands
# here or in the map, never in neither.
NOT_A_PIECE = frozenset({
    "name", "description", "pieces.id", "pieces.name", "setup.placements.squares",
    "pieces.region", "pieces.moves.condition.region",
    "rules.promotion.rank", "rules.promotion.squares",
    "rules.castling.king_file.kingside", "rules.castling.king_file.queenside",
    "rules.castling.rook_file.kingside", "rules.castling.rook_file.queenside",
    "rules.obstacles.squares",
    "rules.npcs.route", "rules.npcs.also_at",
    "rules.portals.moves.condition.region", "rules.portals.orange", "rules.portals.blue",
    "rules.portals.ends.at",
    "rules.purse.region", "rules.duck.start", "rules.scoring.region", "rules.reach.squares",
    "rules.drops.region.*",  # a dropped piece's squares, under the id that keys them
})
# Piece ids outside `rules` and the piece fields, which `drop_piece` handles itself: a
# placement goes with its piece.
OUTSIDE_THE_MAP = frozenset({"setup.placements.piece"})


def _string_fields(node: Any, path: tuple[str, ...], out: set[tuple[str, ...]]) -> None:
    """Every path in the JSON Schema to a string that is not an enum or a constant."""
    if not isinstance(node, dict):
        return
    for alternative in node.get("anyOf", []) + node.get("oneOf", []):
        _string_fields(alternative, path, out)
    kinds = node.get("type") if isinstance(node.get("type"), list) else [node.get("type")]
    if "string" in kinds and "enum" not in node and "const" not in node:
        out.add(path)
    for key, inner in node.get("properties", {}).items():
        _string_fields(inner, (*path, key), out)
    extra = node.get("additionalProperties")
    if isinstance(extra, dict):
        out.add((*path, KEY))
        _string_fields(extra, (*path, "*"), out)
    if isinstance(node.get("items"), dict):
        _string_fields(node["items"], path, out)


def _schema_fields() -> set[tuple[str, ...]]:
    out: set[tuple[str, ...]] = set()
    for strict in (True, False):
        _string_fields(rule_spec_schema(strict=strict), (), out)
    return out


def _position(path: tuple[str, ...]) -> Names | None:
    """The map's leaf at a JSON Schema path, or ``None`` where the map has none."""
    roots = {"rules": PIECE_POSITIONS, "pieces": PIECE_FIELDS}
    if not path or path[0] not in roots:
        return None
    node: Any = roots[path[0]]
    for index, segment in enumerate(path[1:], start=1):
        while isinstance(node, Each):
            node = node.item
        if isinstance(node, Names):
            last = index == len(path) - 1
            return node if node.shape == PIECE_KEYED and segment == KEY and last else None
        keys = node.keys if isinstance(node, Fields) else node
        node = keys.get(segment, keys.get(ANY_KEY))
        if node is None:
            return None
    while isinstance(node, Each):
        node = node.item
    return node if isinstance(node, Names) and node.shape != PIECE_KEYED else None


def _map_leaves(node: Any, path: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Every leaf of the map as a path, ``*`` standing for any key."""
    if isinstance(node, Each):
        return _map_leaves(node.item, path)
    if isinstance(node, Names):
        return [(*path, KEY) if node.shape == PIECE_KEYED else path]
    keys = node.keys if isinstance(node, Fields) else node
    return [leaf for key, inner in keys.items() for leaf in _map_leaves(inner, (*path, key))]


def _matches(pattern: tuple[str, ...], path: tuple[str, ...]) -> bool:
    return len(pattern) == len(path) and all(
        want in (ANY_KEY, got) for want, got in zip(pattern, path, strict=True)
    )


def test_every_string_field_of_the_json_schema_is_a_position_or_named_as_none() -> None:
    """A rule that names a piece cannot be added to the spec without a map row."""
    unclassified = sorted(
        ".".join(path) for path in _schema_fields()
        if _position(path) is None and ".".join(path) not in NOT_A_PIECE | OUTSIDE_THE_MAP
    )
    assert not unclassified, (
        f"string fields neither in schema.PIECE_POSITIONS/PIECE_FIELDS nor in NOT_A_PIECE: "
        f"{unclassified}"
    )


def test_nothing_is_named_twice_and_nothing_named_is_stale() -> None:
    fields = {".".join(path): path for path in _schema_fields()}
    stale = sorted((NOT_A_PIECE | OUTSIDE_THE_MAP) - fields.keys())
    assert not stale, f"named here but no longer in the JSON Schema: {stale}"
    both = sorted(name for name in NOT_A_PIECE if _position(fields[name]) is not None)
    assert not both, f"in the map and named as no piece: {both}"


def test_every_row_of_the_map_is_a_field_of_the_json_schema() -> None:
    """The other direction: a row for a field the spec no longer has is a dead row. The
    portal stand-in is the one row the validator refuses on purpose: `relate` reads a
    portal game in spelling 30's shape (`schema.portal_stand_in_shape`)."""
    fields = _schema_fields()
    leaves = [
        *_map_leaves(PIECE_POSITIONS, ("rules",)), *_map_leaves(PIECE_FIELDS, ("pieces",)),
    ]
    dead = sorted(
        ".".join(leaf) for leaf in leaves
        if leaf != ("rules", "portals", PORTAL_OLD_PIECE_KEY)
        and not any(_matches(leaf, path) for path in fields)
    )
    assert not dead, f"map rows with no field in the JSON Schema: {dead}"


def test_a_list_written_empty_where_empty_says_something_stays_as_written() -> None:
    """`only_named`: `explode_on_capture.by: []` is valid (no capture explodes), and
    removing a piece it never named leaves it as written rather than taking the rule."""
    spec = load_spec(ROOT / "archive" / "kamikaze.json")
    spec["rules"]["explode_on_capture"]["by"] = []
    assert validate_spec(spec) == []
    spec = drop_piece(spec, "N")
    assert spec["rules"]["explode_on_capture"]["by"] == []


def test_a_dropped_piece_leaves_no_beats_naming_it() -> None:
    spec = load_spec(ROOT / "archive" / "breather.json")
    queen = next(piece for piece in spec["pieces"] if piece["id"] == "Q")
    queen["beats"] = ["P", "N"]
    assert validate_spec(spec) == []
    child = drop_piece(copy.deepcopy(spec), "N")
    assert not [error for error in validate_spec(child) if "'N'" in error]
