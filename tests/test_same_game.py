"""E2's own test: the two targets a person rebuilt in the designer share their originals' key.

Q503c (`docs/plan-game-pipeline.md` §5) refuses to publish a draft whose
`relate.identity_key` matches a game already on the site. Option E2 of
`docs/archive/plan-designer-ux.md` §3 names what would show the key is fit for that: the
two games rebuilt field by field in §1.1 and §1.2 of that plan match their originals. Each
rebuild is written here from that plan's table of differences, every one of them spelling.

These use `rulesets/chess.json` and `rulesets/intransitive.json` because those are the two
games the plan's person rebuilt, and the names say so. Perft to depth 3 holds each pair to
*the same game in play*, which is the premise; the key is then asked whether it agrees.

**Two spellings the key once missed** (Q503c2) are held here too:

- chess: the rook written `[0,1]` four ways and `[1,0]` four ways is one set of directions,
  and `_key_of` now sorts each set before writing it.
- intransitive: a setup stating both armies (`mirror: false`) and one stating one army
  turned round (`orientation: rotate`) put the same pieces on the same squares. The key
  reads the orientation as played: the compiler builds the same tables under both, because
  these pieces read left and right alike.

And the other direction, which a same-game refusal needs as much: a rule that does change
play -- one piece's movement, a rotated army of pieces that are not left-right symmetric
-- still moves the key.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.relate import identity_key

ROOT = Path(__file__).resolve().parents[1]


def _ruleset(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "rulesets" / f"{name}.json").read_text(encoding="utf-8"))


def rebuilt_chess() -> dict[str, Any]:
    """§1.1's draft `cf3e08cd35af`, from its table: every row of it and nothing else."""
    spec = copy.deepcopy(_ruleset("chess"))
    spec["name"] = "a chess rebuilt in the designer"
    by = {piece["id"]: piece for piece in spec["pieces"]}
    step, double, take = by["P"]["moves"]
    by["P"]["moves"] = [step, take, double]
    by["R"]["moves"] = [{"kind": "slide", "vectors": [[0, 1]], "symmetry": "rotate4"}]
    by["K"]["moves"] = [
        {"kind": "leap", "vectors": [[0, 1], [1, 1]], "symmetry": "rotate4"}
    ]
    for pid in ("N", "Q"):
        del by[pid]["moves"][0]["mode"]
    spec["pieces"] = [by[pid] for pid in "PRKNBQ"]
    spec["rules"]["promotion"]["to"] = ["R", "Q", "B", "N"]
    rows = {row["piece"]: row for row in spec["setup"]["placements"]}
    spec["setup"]["placements"] = [rows[pid] for pid in "RPNBQK"]
    return spec


def rebuilt_intransitive() -> dict[str, Any]:
    """§1.2's draft `61f3a4200947`: one row of moves a piece, one army turned round."""
    spec = copy.deepcopy(_ruleset("intransitive"))
    spec["name"] = "an intransitive rebuilt in the designer"
    for piece in spec["pieces"]:
        piece["moves"] = [
            {"kind": "leap", "vectors": [[0, 1], [1, 1]], "symmetry": "rotate4"}
        ]
    spec["setup"] = {
        "orientation": "rotate",
        "placements": [
            {key: value for key, value in row.items() if key != "player"}
            for row in spec["setup"]["placements"]
            if row["player"] == 0
        ],
    }
    return spec


TARGETS = [("chess", rebuilt_chess), ("intransitive", rebuilt_intransitive)]


@pytest.mark.parametrize("name", ["chess", "intransitive"])
def test_each_rebuilt_target_is_a_valid_game_under_another_hash(name: str) -> None:
    made = {"chess": rebuilt_chess, "intransitive": rebuilt_intransitive}[name]()
    assert validate_spec(made) == []
    assert spec_hash(made) != spec_hash(_ruleset(name))


@pytest.mark.parametrize("name", ["chess", "intransitive"])
def test_each_rebuilt_target_plays_the_same_as_its_original_to_perft_3(name: str) -> None:
    """The premise, as the plan measured it: the two are one game in play."""
    made = {"chess": rebuilt_chess, "intransitive": rebuilt_intransitive}[name]()
    original, rebuilt = Engine(_ruleset(name)), Engine(made)
    assert [original.perft(original.initial_state(), depth) for depth in (1, 2, 3)] == [
        rebuilt.perft(rebuilt.initial_state(), depth) for depth in (1, 2, 3)
    ]


@pytest.mark.parametrize(("name", "build"), TARGETS)
def test_each_rebuilt_target_shares_its_originals_identity_key(name: str, build: Any) -> None:
    """E2's test. What `drafts.twin` refuses by."""
    assert identity_key(build()) == identity_key(_ruleset(name))


def test_chess_rebuilt_but_for_the_rook_shares_chess_identity_key() -> None:
    """Every other spelling in §1.1's table already folds: the pieces' order, the pawn's
    rows, a dropped `mode`, the king's first vector, the promotion list, the placements."""
    made = rebuilt_chess()
    rook = next(piece for piece in made["pieces"] if piece["id"] == "R")
    rook["moves"] = copy.deepcopy(
        next(piece for piece in _ruleset("chess")["pieces"] if piece["id"] == "R")["moves"]
    )
    assert spec_hash(made) != spec_hash(_ruleset("chess"))
    assert identity_key(made) == identity_key(_ruleset("chess"))


def test_a_rook_written_from_either_first_vector_keys_alike() -> None:
    """The direction set inside a piece's movement is sorted, not written in build order."""
    first, second = _ruleset("chess"), _ruleset("chess")
    for spec, vector in ((first, [0, 1]), (second, [1, 0])):
        rook = next(piece for piece in spec["pieces"] if piece["id"] == "R")
        rook["moves"] = [{"kind": "slide", "vectors": [vector], "symmetry": "rotate4"}]
    assert spec_hash(first) != spec_hash(second)
    assert identity_key(first) == identity_key(second)


def test_a_changed_movement_moves_the_key() -> None:
    """A rook that also leaps one square diagonally is another game."""
    made = rebuilt_chess()
    rook = next(piece for piece in made["pieces"] if piece["id"] == "R")
    rook["moves"].append({"kind": "leap", "vectors": [[1, 1]], "symmetry": "rotate4"})
    assert validate_spec(made) == []
    assert identity_key(made) != identity_key(_ruleset("chess"))


def _leaning(spec: dict[str, Any]) -> dict[str, Any]:
    """`spec` with every piece stepping forward and forward-right only. Not left-right
    symmetric, so a mirror and a rotation give the second player different moves."""
    for piece in spec["pieces"]:
        piece["moves"] = [{"kind": "leap", "vectors": [[0, 1], [1, 1]]}]
    return spec


def test_a_rotated_army_of_leaning_pieces_is_another_game() -> None:
    """Rotation is a rule where the pieces tell left from right.

    Both specs stand the same pieces on the same squares; the second reads its vectors
    turned half round rather than reflected in the rank, so its second army leans the
    other way. Perft says they play apart, and the key must keep them apart.
    """
    stated = _leaning(_ruleset("intransitive"))
    rotated = _leaning(rebuilt_intransitive())
    assert validate_spec(stated) == [] and validate_spec(rotated) == []
    first, second = Engine(stated), Engine(rotated)
    assert first.perft(first.initial_state(), 2) != second.perft(second.initial_state(), 2)
    assert identity_key(stated) != identity_key(rotated)


def test_a_rotated_army_of_symmetric_pieces_keys_as_its_written_out_twin() -> None:
    """The same rotation over the original pieces, which read left and right alike, is
    spelling: only `setup` differs from the file."""
    stated = _ruleset("intransitive")
    rotated = copy.deepcopy(stated)
    rotated["setup"] = rebuilt_intransitive()["setup"]
    assert spec_hash(stated) != spec_hash(rotated)
    assert identity_key(stated) == identity_key(rotated)
