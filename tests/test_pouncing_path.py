"""SR2 (`docs/plan-schema-next.md` §2.9, row N20): a turning path on a wild piece that pounces.

A turning path (`circle`, `zigzag`) was refused on every wild type, *"its moves are never
read"*; but a pouncer on its prey reads its own capture moves (what it attacks, and so whose
royal is in check). The probe fed the core a pouncing hound and lurker with turning paths past
the validator and held it to `tests/pounce_oracle.py` (whose stand-in, a player's piece of the
pouncer's type, is asked of the core's ordinary generator, so the turning path is the one the
player's pieces already play): legal lists, check, boards and phase logs agree at every ply,
under checkmate and royal capture. The lift is the validator's, for `pounce: "prey"` only:
an `"attackers"` pouncer reads its attackers' moves, never its own, so a path on it is still
never read and still refused.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.spec.schema import validate_spec
from tests import test_pounce as pounce_tests

HOUNDS = Path(__file__).resolve().parents[1] / "tests" / "_pounce_hounds.json"
PATHS = [("circle", [1, 2], "dihedral8"), ("zigzag", [1, 1], "rotate4")]


def _hounds(path: str, vector: list[int], symmetry: str, win: str = "checkmate",
            pounce: str = "prey") -> dict[str, Any]:
    """The hounds' fixture, its two pouncing types (`H`, `L`) turned: a rose's octagons or a
    crooked bishop's zigzag."""
    spec = copy.deepcopy(json.loads(HOUNDS.read_text(encoding="utf-8")))
    spec["rules"]["win"] = win
    for piece in spec["pieces"]:
        if piece["id"] in ("H", "L"):
            piece["moves"] = [{"kind": "slide", "mode": "both", "symmetry": symmetry,
                               "vectors": [vector], "path": path}]
    for npc in spec["rules"]["npcs"]:
        npc["pounce"] = pounce
    return spec


@pytest.mark.parametrize("path, vector, symmetry", PATHS)
def test_a_turning_path_on_a_prey_pouncer_is_valid(path: str, vector: list[int],
                                                   symmetry: str) -> None:
    assert validate_spec(_hounds(path, vector, symmetry)) == []


@pytest.mark.parametrize("path, vector, symmetry", PATHS)
def test_on_an_attackers_pouncer_it_is_still_never_read(path: str, vector: list[int],
                                                        symmetry: str) -> None:
    errors = validate_spec(_hounds(path, vector, symmetry, pounce="attackers"))
    assert any("a wild piece walks its route, and its moves are never read" in e for e in errors)


def test_on_a_wild_piece_that_does_not_pounce_it_is_still_never_read() -> None:
    spec = _hounds("circle", [1, 2], "dihedral8")
    for npc in spec["rules"]["npcs"]:
        npc.pop("pounce")
    assert any("its moves are never read" in e for e in validate_spec(spec))


@pytest.mark.parametrize("win", ["checkmate", "royal_capture"])
@pytest.mark.parametrize("path, vector, symmetry", PATHS)
def test_seeded_walks_agree_with_the_oracle(path: str, vector: list[int], symmetry: str,
                                            win: str) -> None:
    spec = _hounds(path, vector, symmetry, win)
    totals = [pounce_tests._walk(spec, seed, 80) for seed in range(3)]
    assert sum(p for p, _ in totals) > 0


@pytest.mark.slow
@pytest.mark.parametrize("path, vector, symmetry", PATHS)
def test_twenty_walks_see_pounces_and_checks(path: str, vector: list[int], symmetry: str) -> None:
    totals = [pounce_tests._walk(_hounds(path, vector, symmetry), seed, 160) for seed in range(20)]
    assert sum(p for p, _ in totals) > 100 and sum(c for _, c in totals) > 10
