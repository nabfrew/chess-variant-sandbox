"""Other spellings of one game, for `tests/test_normal_form.py` (Q1008a).

Each function takes a spec and returns a copy written another way that the compiler reads
identically: a default written out, a rule stated at its "no rule" value, the second spelling
of a promotion. `normal.normal_form` must take each back to the same normal form, so the copy
has the original's identity, and where the validator admits the copy it must compile to the
original's tables.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from chess_variant_sandbox.spec.schema import DEFAULT_RANGE, SCHEMA_VERSION

Respelling = Callable[[dict[str, Any]], dict[str, Any]]


def _copy(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    out = copy.deepcopy(spec)
    return out, out.setdefault("rules", {})


def newest_version(spec: dict[str, Any]) -> dict[str, Any]:
    out, _ = _copy(spec)
    out["schema_version"] = SCHEMA_VERSION
    return out


def board_defaults(spec: dict[str, Any]) -> dict[str, Any]:
    out, _ = _copy(spec)
    out["board"].setdefault("lattice", "square")
    out["board"].setdefault("shape", "grid")
    out["setup"].setdefault("mirror", True)
    if out["board"]["lattice"] == "square":
        out["setup"].setdefault("orientation", "mirror")
    return out


def piece_defaults(spec: dict[str, Any]) -> dict[str, Any]:
    out, _ = _copy(spec)
    for piece in out["pieces"]:
        piece.setdefault("royal", False)
        for move in piece.get("moves", []):
            move.setdefault("symmetry", "none")
            if move["kind"] not in ("leap", "lame"):  # a range on a jump is refused by name
                move.setdefault("range", DEFAULT_RANGE[move["kind"]])
            move.setdefault("mode", "both")
            if move["kind"] == "slide" and move["range"] != 1:
                move.setdefault("min_range", 1)
            if "condition" in move:
                move["condition"].setdefault("first_move", False)
    for placement in out["setup"].get("placements", []):
        placement.setdefault("player", 0)
    return out


def rule_defaults(spec: dict[str, Any]) -> dict[str, Any]:
    out, rules = _copy(spec)
    for key, value in (
        ("win", "checkmate"), ("stalemate", "draw"), ("max_plies", 400),
        ("repetition_draw", 3), ("no_progress_draw", 100),
    ):
        rules.setdefault(key, value)
    if "simultaneous" not in rules:  # no checkmate in a game in rounds, refused by name
        rules.setdefault("checkmate", "loss")
    if any(piece.get("royal") for piece in out["pieces"]):
        rules.setdefault("royal_facing", "allowed")
    for key in ("en_passant", "must_capture", "forbid_immobility", "promote_on_entry"):
        rules.setdefault(key, False)
    return out


def empty_rules(spec: dict[str, Any]) -> dict[str, Any]:
    """Rules stated at the value that means no rule."""
    out, rules = _copy(spec)
    rules.setdefault("turn", {"moves": 1})
    if out["board"].get("shape") != "hexagon":
        rules.setdefault("wrap", {"files": "none", "ranks": "none"})
    if "contacts" not in rules and "portals" not in rules:
        rules.setdefault("strike", {"effect": "capture"})
    return out


def promotion_spelling(spec: dict[str, Any]) -> dict[str, Any]:
    """A one-entry promotion list as an object and back, its rank zone as `rank`."""
    out, rules = _copy(spec)
    promotion = rules.get("promotion")
    entries = promotion if isinstance(promotion, list) else [promotion] if promotion else []
    height = out["board"]["height"]
    for entry in entries:
        entry.setdefault("mandatory", True)
        if "squares" in entry:
            continue
        if ("rank" not in entry and "ranks" not in entry) or entry.get("rank") == "last":
            entry["rank"] = height - 1
        elif entry.get("ranks") == [height - 1]:
            del entry["ranks"]
            entry["rank"] = "last"
    if isinstance(promotion, list) and len(promotion) == 1:
        rules["promotion"] = promotion[0]
    elif isinstance(promotion, dict):
        rules["promotion"] = [promotion]
    return out


RESPELLINGS: dict[str, Respelling] = {
    "newest_version": newest_version,
    "board_defaults": board_defaults,
    "piece_defaults": piece_defaults,
    "rule_defaults": rule_defaults,
    "empty_rules": empty_rules,
    "promotion_spelling": promotion_spelling,
}


def every_respelling(spec: dict[str, Any]) -> dict[str, Any]:
    """All of `RESPELLINGS` at once."""
    for respell in RESPELLINGS.values():
        spec = respell(spec)
    return spec
