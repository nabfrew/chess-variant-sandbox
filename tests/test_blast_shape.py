"""SR1b: a blast's shape, stated (`docs/plan-schema-next.md` §1 rank 2, §2 X11).

``explode_on_capture.blast`` is ``{vectors, symmetry}``, read from the capture square, in
place of ``radius``'s box. The blast table is one per square, not one per player, so the
shape must read alike for both players (`schema.reads_alike_for_both`). Absent, the box is
built exactly as before. The hand positions below are counted square by square.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.describe import describe_rules
from chess_variant_sandbox.spec.distance import features

ROOT = Path(__file__).resolve().parent.parent

CROSS = {"vectors": [[1, 0]], "symmetry": "rotate4"}
FILE = {"vectors": [[0, 1], [0, 2], [0, -1], [0, -2]]}
KING_STEP = {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}


def atomic(blast: dict[str, Any] | None = None, **keys: Any) -> dict[str, Any]:
    spec = json.loads((ROOT / "rulesets" / "atomic.json").read_text("utf-8"))
    explosion = spec["rules"]["explode_on_capture"]
    if blast is not None:
        explosion.pop("radius", None)
        explosion["blast"] = copy.deepcopy(blast)
    explosion.update(keys)
    return spec


def errors_of(spec: dict[str, Any], words: str) -> list[str]:
    return [e for e in validate_spec(spec) if words in e]


def blast_at(spec: dict[str, Any], square: str) -> set[str]:
    width = spec["board"]["width"]
    index = (int(square[1:]) - 1) * width + ord(square[0]) - ord("a")
    table = compile_spec(spec).explosion
    assert table is not None
    return {f"{chr(ord('a') + s % width)}{s // width + 1}" for s in table.blast[index]}


# ------------------------------------------------------------------- the shape


def test_a_cross_and_a_file_blast_validate() -> None:
    assert validate_spec(atomic(CROSS)) == []
    assert validate_spec(atomic(FILE)) == []


def test_the_shape_compiles_from_the_capture_square() -> None:
    assert blast_at(atomic(CROSS), "e4") == {"e5", "e3", "d4", "f4"}
    assert blast_at(atomic(FILE), "e4") == {"e2", "e3", "e5", "e6"}
    # The board's edge cuts it, as it cuts the box.
    assert blast_at(atomic(FILE), "a1") == {"a2", "a3"}


def test_the_kings_step_stated_is_the_radius_ones_table() -> None:
    """A shape that *is* the box compiles to the box's table, in the box's order."""
    box, stated = atomic(), atomic(KING_STEP)
    assert spec_hash(box) != spec_hash(stated)
    assert compile_spec(box).explosion.blast == compile_spec(stated).explosion.blast


def test_atomic_is_unmoved() -> None:
    """Absent, the radius's box is today's table byte for byte (the proof over every stored
    spec is in the record; this pins the corpus's one blast)."""
    spec = atomic()
    tables = canonical_tables(compile_spec(spec))
    assert tables == canonical_tables(compile_spec(copy.deepcopy(spec)))
    assert spec_hash(spec) == spec_hash(json.loads(
        (ROOT / "rulesets" / "atomic.json").read_text("utf-8")))


# ------------------------------------------------------------------- refusals


def test_radius_and_blast_together_are_refused() -> None:
    assert errors_of(atomic(CROSS, radius=1), "radius and .blast both")
    assert errors_of(atomic(CROSS, radius=0), "radius and .blast both")


@pytest.mark.parametrize("blast, words", [
    ({"symmetry": "rotate4"}, "blast is missing required key 'vectors'"),
    ({"vectors": []}, "blast.vectors must be a non-empty array"),
    ({"vectors": [[0, 0]]}, "null vector"),
    ({"vectors": [[9, 0]], "symmetry": "rotate4"}, "within +/-8"),
    ({"vectors": [[1, 0]], "symmetry": "spin"}, "blast.symmetry must be one of"),
    ({"vectors": [[1, 0]], "symmetry": "rotate4", "range": 2}, "range"),
    ("cross", "blast must be an object"),
])
def test_a_malformed_shape_is_refused_by_name(blast: Any, words: str) -> None:
    spec = atomic()
    spec["rules"]["explode_on_capture"].pop("radius")
    spec["rules"]["explode_on_capture"]["blast"] = blast
    errors = validate_spec(spec)
    assert any(words in e for e in errors), errors


@pytest.mark.parametrize("blast", [
    {"vectors": [[0, 1]]},  # forward only: the second player's forward is the other way
    {"vectors": [[1, 1]], "symmetry": "lateral"},  # the two forward diagonals
    {"vectors": [[0, 1], [0, 2], [0, -1]]},  # the file, unevenly
])
def test_a_shape_the_two_players_read_differently_is_refused(blast: dict[str, Any]) -> None:
    assert errors_of(atomic(blast), "reads differently for the two players")


def test_a_rotated_army_reads_the_half_turn() -> None:
    """Under `orientation: "rotate"` the second player's image is the half turn, so a
    shape is refused unless the half turn keeps it; `[1, 1]` and `[-1, -1]` is one."""
    spec = atomic({"vectors": [[1, 1], [-1, -1]]})
    spec["setup"]["orientation"] = "rotate"
    assert errors_of(spec, "reads differently") == []
    spec = atomic({"vectors": [[1, 1], [-1, 1]]})
    spec["setup"]["orientation"] = "rotate"
    assert errors_of(spec, "reads differently")


def test_a_lattice_takes_its_own_symmetries() -> None:
    assert errors_of(atomic({"vectors": [[1, 0]], "symmetry": "rotate6"}), "only a hex lattice")
    assert errors_of(hex_game({"vectors": [[1, 0]], "symmetry": "rotate4"}),
                     "carry some of them off the hex lattice")


def test_immune_beside_a_shape_is_allowed() -> None:
    """Radius 0 refuses `immune` because it reaches nobody; a stated shape always reaches
    someone (a vector is never null)."""
    assert validate_spec(atomic(CROSS, immune=["P"])) == []


# ------------------------------------------------------------------- the hexagon's seam (X11)


def hex_game(blast: dict[str, Any], wrap: dict[str, Any] | None = None) -> dict[str, Any]:
    """A radius-3 hexagon (R + 1 = 4), kings and a stepping piece, with a blast."""
    step = [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate6"}]
    rules: dict[str, Any] = {"win": "checkmate", "explode_on_capture": {"blast": blast,
                                                                        "immune": ["K"]}}
    if wrap:
        rules["wrap"] = wrap
    return {
        "name": "hex blast probe",
        "board": {"width": 7, "height": 7, "lattice": "hex", "shape": "hexagon"},
        "pieces": [
            {"id": "K", "name": "king", "value": 0, "royal": True, "moves": step},
            {"id": "S", "name": "stepper", "value": 1, "moves": step},
        ],
        "setup": {"mirror": False, "orientation": "rotate", "placements": [
            {"piece": "K", "player": 0, "squares": ["d2"]},
            {"piece": "S", "player": 0, "squares": ["c3", "e3"]},
            {"piece": "K", "player": 1, "squares": ["d6"]},
            {"piece": "S", "player": 1, "squares": ["c5", "e5"]},
        ]},
        "rules": rules,
    }


RING_TWO = {"vectors": [[2, 0], [1, 1]], "symmetry": "rotate6"}  # across 4: R + 1
RING_THREE = {"vectors": [[3, 0]], "symmetry": "rotate6"}  # across 6
LINE_FAR = {"vectors": [[2, 0], [-3, 0]]}  # across 5, and not its own half turn


def test_the_seam_bound_admits_a_blast_across_r_plus_one() -> None:
    assert validate_spec(hex_game(RING_TWO, {"n_s": "both"})) == []


def test_the_seam_bound_refuses_a_blast_across_more() -> None:
    assert errors_of(hex_game(RING_THREE, {"n_s": "both"}), "blast reaches across 6 cells")
    # The bound is the shape's diameter, not its longest vector: (2, 0) and (-2, 0) are
    # each two steps out and four apart, which is R + 1, and admitted.
    assert validate_spec(hex_game({"vectors": [[2, 0], [-2, 0]]}, {"n_s": "both"})) == []
    assert errors_of(hex_game({"vectors": [[3, 0], [-2, 0], [2, 0], [-3, 0]]},
                              {"n_s": "both"}), "blast reaches across 6 cells")


def test_with_no_side_joined_any_width_is_fine() -> None:
    assert validate_spec(hex_game(RING_THREE)) == []


def test_a_hex_blast_plays() -> None:
    spec = hex_game(RING_TWO, {"n_s": "both"})
    engine = Engine(spec)
    assert engine.perft(engine.initial_state(), 2) > 0


# ------------------------------------------------------------------- play


def _taking(spec: dict[str, Any], placed: dict[str, tuple[str, int]], move: str) -> set[str]:
    """What stands after the first player plays ``move`` from ``placed``."""
    engine = Engine(spec)
    state = engine.make_state(placed)
    after = engine.apply(state, engine.find_move(state, move))
    width = spec["board"]["width"]
    return {f"{chr(ord('a') + s % width)}{s // width + 1}" for s, v in enumerate(after.board) if v}


def test_a_cross_blast_spares_the_diagonals() -> None:
    placed = {"a1": ("K", 0), "h8": ("K", 1), "d1": ("Q", 0), "d5": ("N", 1),
              "d6": ("P", 1), "c5": ("N", 1), "e6": ("N", 1), "c4": ("N", 1)}
    # Qd1xd5: the capturer and d5 go, with d6 and c5 beside it; e6 and c4 are diagonal.
    standing = _taking(atomic(CROSS), placed, "Qd1xd5")
    # A pawn is immune in atomic, so d6 stands; c5 is a knight and goes.
    assert standing == {"a1", "h8", "d6", "e6", "c4"}
    # Atomic's own box takes the diagonals too.
    assert _taking(atomic(), placed, "Qd1xd5") == {"a1", "h8", "d6"}


def test_a_file_blast_reaches_two_squares_along_the_file() -> None:
    placed = {"a1": ("K", 0), "h8": ("K", 1), "h4": ("R", 0), "d4": ("N", 1),
              "d6": ("N", 1), "d2": ("B", 1), "c4": ("N", 1), "e5": ("N", 1)}
    standing = _taking(atomic(FILE), placed, "Rh4xd4")
    assert standing == {"a1", "h8", "c4", "e5"}


def test_the_blast_specs_play_a_short_game() -> None:
    import random

    for name in ("_blast_cross.json", "_blast_file.json"):
        spec = json.loads((ROOT / "tests" / name).read_text("utf-8"))
        assert validate_spec(spec) == []
        engine = Engine(spec)
        rng = random.Random(3)
        state = engine.initial_state()
        for _ in range(40):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            state = engine.apply(state, rng.choice(moves))
            assert state.key == engine._compute_hash(state)


# ------------------------------------------------------------------- words and readers


def test_the_shape_is_said() -> None:
    said = " ".join(describe_rules(atomic(CROSS)))
    assert "one square forward, back, left and right from the square it took on" in said
    said = " ".join(describe_rules(atomic(FILE)))
    assert "up to two squares forward and back from the square it took on" in said


def test_distance_tells_a_shape_from_the_box() -> None:
    assert features(atomic())["explodes_shaped"] == 0.0
    assert features(atomic(CROSS))["explodes_shaped"] == 1.0


def test_fairy_translates_the_kings_step_and_refuses_every_other_shape() -> None:
    assert "blastOnCapture = true" in translate(atomic(KING_STEP)).config
    for shape in (CROSS, FILE):
        with pytest.raises(Untranslatable) as refused:
            translate(atomic(shape))
        assert any("blast other than the king's step" in r for r in refused.value.reasons)
