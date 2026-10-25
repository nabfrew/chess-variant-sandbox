"""A rule written at its off value is the game without it (Q1008c).

Until Q1008c the validator refused each of these as "the default; leave it out", because the
name was the hash of the file and the two spellings would have been two games. The name is
the normal form's now (`spec/normal.py`), so they are accepted, named as the game without
them, compiled to its tables, and tidied away by the converter (`spec/spelling.py`) that the
designer, the proposer and the library's writer read a draft through.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.schema import (
    CONTACT_LINK_OFF_VALUES,
    MOVE_OFF_VALUES,
    NPC_OFF_VALUES,
    PIECE_OFF_VALUES,
    PORTAL_OFF_VALUES,
    SIGHT_OFF_VALUES,
    SIMULTANEOUS_OFF_VALUES,
)
from chess_variant_sandbox.spec.spelling import current_spelling

ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return loaded


def _piece(key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def write(spec: dict[str, Any]) -> None:
        for piece in spec["pieces"]:
            if key not in piece:
                piece[key] = copy.deepcopy(value)
    return write


def _through(spec: dict[str, Any]) -> None:
    for piece in spec["pieces"]:
        for entry in piece.get("sight", []):
            entry.setdefault("through", False)


def _captures_own(spec: dict[str, Any]) -> None:
    for piece in spec["pieces"]:
        for entry in piece.get("sight", []):
            entry.setdefault("captures_own", False)


def _straight(spec: dict[str, Any]) -> None:
    for piece in spec["pieces"]:
        for move in piece.get("moves", []):
            move.setdefault("path", "straight")
    for group in spec.get("rules", {}).get("portals", []):
        for move in group.get("moves", []):
            move.setdefault("path", "straight")


def _simultaneous(key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def write(spec: dict[str, Any]) -> None:
        spec["rules"]["simultaneous"][key] = value
    return write


def _kind(spec: dict[str, Any]) -> None:
    for group in spec["rules"]["portals"]:
        group.setdefault("kind", "momentum")


def _respawn(spec: dict[str, Any]) -> None:
    for npc in spec["rules"]["npcs"]:
        npc.setdefault("respawn", 0)


def _own(spec: dict[str, Any]) -> None:
    for link in spec["rules"]["contacts"]["links"]:
        link.setdefault("own", False)


# (where it is written, the game it is written on, the edit). Every off value of every place.
CASES = [
    ("pieces[].rest", "tests/_simultaneous_gardner.json", _piece("rest", 0)),
    ("pieces[].speed", "tests/_simultaneous_gardner.json", _piece("speed", 0)),
    ("pieces[].spy", "tests/_simultaneous_gardner.json", _piece("spy", False)),
    ("pieces[].sight", "rulesets/chess.json", _piece("sight", [])),
    ("pieces[].sight[].through", "archive/madrasi.json", _through),
    ("pieces[].sight[].captures_own", "tests/_command_k1.json", _captures_own),
    ("rules.simultaneous.tie", "tests/_simultaneous_gardner.json", _simultaneous("tie", "both")),
    ("rules.simultaneous.swap", "tests/_simultaneous_gardner.json",
     _simultaneous("swap", "chase")),
    ("rules.simultaneous.both", "tests/_simultaneous_gardner.json",
     _simultaneous("both", "draw")),
    ("rules.portals[].kind", "archive/wormhole.json", _kind),
    ("rules.contacts.links[].own", "archive/einstein.json", _own),
    ("pieces[].moves[].path", "rulesets/chess.json", _straight),
    ("rules.npcs[].respawn", "archive/wolves.json", _respawn),
]


def test_every_off_value_has_a_case() -> None:
    named = {
        **{f"pieces[].{key}": 1 for key in PIECE_OFF_VALUES},
        **{f"pieces[].sight[].{key}": 1 for key in SIGHT_OFF_VALUES},
        **{f"pieces[].moves[].{key}": 1 for key in MOVE_OFF_VALUES},
        **{f"rules.simultaneous.{key}": 1 for key in SIMULTANEOUS_OFF_VALUES},
        **{f"rules.portals[].{key}": 1 for key in PORTAL_OFF_VALUES},
        **{f"rules.contacts.links[].{key}": 1 for key in CONTACT_LINK_OFF_VALUES},
        **{f"rules.npcs[].{key}": 1 for key in NPC_OFF_VALUES},
    }
    assert set(named) == {where for where, _, _ in CASES}


@pytest.mark.parametrize(("where", "path", "edit"), CASES, ids=[case[0] for case in CASES])
def test_an_off_value_is_the_game_without_it(
    where: str, path: str, edit: Callable[[dict[str, Any]], None]
) -> None:
    base = _load(path)
    assert validate_spec(base) == [], validate_spec(base)
    written = copy.deepcopy(base)
    edit(written)
    assert written != base, f"{where}: the case wrote nothing"
    assert validate_spec(written) == [], validate_spec(written)
    assert spec_hash(written) == spec_hash(base)
    assert canonical_tables(compile_spec(written)) == canonical_tables(compile_spec(base))
    assert current_spelling(written) == current_spelling(base)
