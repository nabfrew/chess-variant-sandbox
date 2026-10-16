"""A summed ``rules.extinction`` no side starts run out of (Q937).

`docs/reference/extinction-any-kind.md` §3.4: summed over kinds one side never starts with,
the rule validated, and that side counted zero of them at ply zero, so the game was over
before anyone moved (`Engine.terminal(initial_state)` gave `extinction`). The per-type rule
was refused for this since schema 18; the summed rule now is too, with the same sentence.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from retired_specs import REBUILT

from chess_variant_sandbox.spec import validate_spec

ROOT = Path(__file__).resolve().parents[1]
STORED = sorted((ROOT / "rulesets").glob("*.json")) + sorted((ROOT / "archive").glob("*.json"))


def _gap(extinction: dict[str, Any]) -> dict[str, Any]:
    """The record's reproduction spec: two armies of different ids on a 6x6 board."""

    def leap(vector: list[int], symmetry: str) -> list[dict[str, Any]]:
        return [{"kind": "leap", "vectors": [vector], "symmetry": symmetry, "mode": "both"}]

    return {
        "schema_version": 18, "name": "gap", "board": {"width": 6, "height": 6},
        "pieces": [
            {"id": "A", "value": 3, "moves": leap([1, 0], "rotate4")},
            {"id": "B", "value": 3, "moves": leap([1, 1], "rotate4")},
            {"id": "C", "value": 3, "moves": leap([1, 2], "dihedral8")},
            {"id": "E", "value": 3, "moves": leap([2, 0], "rotate4")},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "A", "player": 0, "squares": ["b1", "e1"]},
            {"piece": "B", "player": 0, "squares": ["c1", "d1"]},
            {"piece": "C", "player": 1, "squares": ["b6", "e6"]},
            {"piece": "E", "player": 1, "squares": ["c6", "d6"]},
        ]},
        "rules": {"win": "none", "stalemate": "draw", "max_plies": 200, "repetition_draw": 3,
                  "extinction": extinction},
    }


def test_summed_over_one_side_s_kinds_is_refused() -> None:
    errors = validate_spec(_gap({"pieces": ["A", "B"]}))
    assert errors == [
        "rules.extinction: player 1 starts with 0 of 'A', 'B', fewer than min_count 1, "
        "so the game would be over before the first move"
    ]


def test_summed_over_both_sides_kinds_still_validates() -> None:
    """The horde's reading: each side starts with four of the listed kinds."""
    assert validate_spec(_gap({"pieces": ["A", "B", "C", "E"]})) == []


def test_the_count_is_summed_not_per_kind() -> None:
    assert validate_spec(_gap({"pieces": ["A", "B", "C", "E"], "min_count": 4})) == []
    errors = validate_spec(_gap({"pieces": ["A", "B", "C", "E"], "min_count": 5}))
    assert [e.split(" starts")[0] for e in errors] == ["rules.extinction: player 0", "rules.extinction: player 1"]
    assert "starts with 4 of 'A', 'B', 'C', 'E', fewer than min_count 5" in errors[0]


def test_any_counts_the_whole_army() -> None:
    assert validate_spec(_gap({"pieces": "any", "min_count": 4})) == []
    errors = validate_spec(_gap({"pieces": "any", "min_count": 5}))
    assert "rules.extinction: player 0 starts with 4 pieces, fewer than min_count 5" in errors[0]


def test_per_type_keeps_its_own_sentence() -> None:
    errors = validate_spec(_gap({"pieces": ["A", "B"], "per_type": True}))
    assert len(errors) == 2 and all(e.startswith("rules.extinction.per_type: player 1") for e in errors)


@pytest.mark.parametrize("path", STORED, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_stored_spec_still_validates(path: Path) -> None:
    assert validate_spec(json.loads(path.read_text(encoding="utf-8"))) == []


@pytest.mark.parametrize("name", sorted(REBUILT))
def test_every_retired_rebuild_still_validates(name: str) -> None:
    assert validate_spec(copy.deepcopy(REBUILT[name]())) == []
