"""A one-square `lame` leap is refused on admission (Q884).

A lame leap is stopped by a piece on the square it passes over; a one-square step passes
over none, so the compiler drops it and a spec that wrote one got a piece silently lacking
the move. The refusal runs where a spec is admitted (`validate_spec`'s ``new``), so a stored
spec still builds its engine, as with Q434's and Q879's.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.plain import plain_problem
from chess_variant_sandbox.web import designing

ROOT = Path(__file__).resolve().parents[1]


def _with_lame(vectors: list[list[int]], symmetry: str = "none") -> dict[str, Any]:
    """Xiangqi's horse, its vectors and symmetry replaced."""
    spec = copy.deepcopy(load_spec(ROOT / "rulesets" / "xiangqi.json"))
    horse = next(p for p in spec["pieces"] if p["id"] == "H")
    horse["moves"] = [{"kind": "lame", "vectors": vectors, "symmetry": symmetry, "mode": "both"}]
    spec["name"] = "lame draft"
    return spec


@pytest.mark.parametrize("vector", [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [1, -1]])
def test_a_one_square_lame_leap_is_refused_with_a_plain_sentence(vector: list[int]) -> None:
    spec = _with_lame([vector])
    errors = validate_spec(spec)
    assert len(errors) == 1, errors
    assert f"{vector} is a one-square 'lame' leap, which has no square to be blocked on" in (
        errors[0])
    assert "use an ordinary 'leap' or a 'slide' of range 1 instead" in errors[0]
    # A stored spec carrying one still builds its engine: it was dropped, never played.
    assert validate_spec(spec, new=False) == []


def test_a_one_square_vector_among_longer_ones_is_named_by_its_place() -> None:
    errors = validate_spec(_with_lame([[1, 2], [1, 1]], "dihedral8"))
    assert len(errors) == 1
    assert ".moves[0].vectors[1] [1, 1] is a one-square" in errors[0]


@pytest.mark.parametrize(
    "vectors, symmetry", [([[1, 2]], "dihedral8"), ([[2, 2]], "rotate4"), ([[2, 0]], "none"),
                          ([[3, 3]], "none")]
)
def test_a_longer_lame_leap_is_still_admitted(vectors: list[list[int]], symmetry: str) -> None:
    assert validate_spec(_with_lame(vectors, symmetry)) == []


def test_the_refusal_is_not_part_of_the_game_s_identity() -> None:
    """Validation only: the one-square draft hashes as it did, and xiangqi is unmoved."""
    spec = _with_lame([[1, 0]])
    assert spec_hash(spec) == spec_hash(copy.deepcopy(spec))
    assert validate_spec(load_spec(ROOT / "rulesets" / "xiangqi.json")) == []


def test_the_designer_says_it_in_words() -> None:
    spec = _with_lame([[1, 1]])
    answer = designing.check(spec)
    assert answer["valid"] is False
    assert any("a one-square blockable leap has no square to be blocked on" in e
               for e in answer["errors"]), answer["errors"]
    assert plain_problem(validate_spec(spec)[0], spec).startswith("Piece ")
