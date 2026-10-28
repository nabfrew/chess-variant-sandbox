"""SR1c: a turning path on a hex lattice (`docs/plan-schema-next.md` §1 rank 5, lifting PA2).

`schema.turning_order` lays a hex lattice's axial directions round a cell by the same integer
comparator as the square lattice's (the map into the plane keeps every cross product's sign),
so the refusal that said the hexagon's order was not built is gone, and the compiler builds a
hex path as it builds a square one (`compile._turning_rides`). Held here: the order against the
directions' angles in the plane, the hex rose (the hex knight's twelve, turned one step round
each jump) validating, compiling and playing, its moves against an oracle on a hexagon for both
players over a seeded game, the refusals restated in the plane, and every stored hex spec
unmoved.

The oracle below is a declared test oracle (CLAUDE.md rule 3): it lays the directions out by
floating angle in the plane, never `schema.turning_order`, and walks cells by axial arithmetic
and the hexagon's own bound, never a compiled table.
"""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.spec import load_spec, validate_spec
from chess_variant_sandbox.spec.complexity import expand_symmetry
from chess_variant_sandbox.spec.describe import move_sentence
from chess_variant_sandbox.spec.schema import hex_distance, turning_order

ROOT = Path(__file__).resolve().parent.parent
GLINSKI = ROOT / "archive" / "hex-glinski-hexagon.json"

# The hex rose: the hex knight's twelve, each jump turned one step round from the last.
HEX_ROSE = {"kind": "slide", "vectors": [[1, 2]], "symmetry": "dihedral12", "path": "circle"}
# The hex ring: the six steps, a circle of single steps round the cell.
HEX_RING = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate6", "path": "circle"}
# The hex bolt: the six steps, zigzagging between neighbours 60 degrees apart.
HEX_BOLT = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate6", "path": "zigzag"}

STEPS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
DIAGONALS = [(1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2), (2, -1)]


def _angle(vector: tuple[int, int]) -> float:
    """A hex direction's angle in the plane, counter-clockwise from east, in [0, 2 pi)."""
    q, r = vector
    return math.atan2(r * math.sqrt(3) / 2, q + r / 2) % (2 * math.pi)


def _by_angle(directions: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(directions, key=_angle)


def _directions(move: dict[str, Any]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for x, y in move["vectors"]:
        out |= expand_symmetry((x, y), move.get("symmetry", "none"))
    return out


def with_rider(move: dict[str, Any], piece_id: str = "N") -> dict[str, Any]:
    """Glinski's hexagon with ``piece_id``'s moves replaced by ``move`` alone."""
    spec = load_spec(GLINSKI)
    spec["name"] = "glinski with a turning rider"
    for piece in spec["pieces"]:
        if piece["id"] == piece_id:
            piece["moves"] = [copy.deepcopy(move)]
    return spec


# ------------------------------------------------------------------- the order


@pytest.mark.parametrize("directions", [
    STEPS, DIAGONALS, STEPS + DIAGONALS,
    sorted(_directions(HEX_ROSE)),
    [(1, 0), (2, 0), (0, 3), (-2, 1)],
])
def test_the_order_is_the_angle_in_the_plane(directions: list[tuple[int, int]]) -> None:
    """Shorter first along one line, as on the square lattice; otherwise by angle."""
    expected = sorted(set(directions), key=lambda v: (_angle(v), v[0] ** 2 + v[1] ** 2))
    assert turning_order(directions) == expected


def test_the_six_steps_go_round_in_sixty_degree_turns() -> None:
    assert turning_order(reversed(STEPS)) == STEPS


def test_the_hex_knights_twelve_lie_between_the_steps_and_diagonals() -> None:
    ring = turning_order(_directions(HEX_ROSE))
    assert len(ring) == 12
    assert ring == _by_angle(_directions(HEX_ROSE))
    # Each of the twelve sits in its own 30-degree sector between a step and a diagonal.
    sectors = sorted(int(math.degrees(_angle(v)) // 30) for v in ring)
    assert sectors == list(range(12))


# ------------------------------------------------------------------- the spelling


@pytest.mark.parametrize("move", [HEX_ROSE, HEX_RING, HEX_BOLT])
def test_a_hex_path_validates(move: dict[str, Any]) -> None:
    assert validate_spec(with_rider(move)) == []


def test_a_hex_rook_may_circle() -> None:
    spec = load_spec(GLINSKI)
    for piece in spec["pieces"]:
        if piece["id"] == "R":
            piece["moves"][0]["path"] = "circle"
    assert validate_spec(spec) == []


def test_a_gap_is_measured_in_the_plane() -> None:
    """Two steps 60 degrees apart leave a gap of 300 in the plane (the axial frame's 270
    would be a skewed number)."""
    move = {"kind": "slide", "vectors": [[1, 0], [0, 1]], "symmetry": "none", "path": "circle"}
    problems = validate_spec(with_rider(move))
    assert any("leave a gap of 300 degrees" in p for p in problems), problems


def test_a_zigzag_needs_a_near_pair_on_a_hex_lattice_too() -> None:
    move = {"kind": "slide", "vectors": [[1, 0], [-1, 0]], "symmetry": "none",
            "path": "zigzag"}
    problems = validate_spec(with_rider(move))
    assert any("needs two directions less than half a turn apart" in p for p in problems)


def test_a_square_symmetry_stays_refused_on_a_hex_lattice() -> None:
    move = {**HEX_ROSE, "symmetry": "dihedral8"}
    assert validate_spec(with_rider(move))


def test_two_directions_one_way_are_refused_on_a_hex_lattice() -> None:
    move = {"kind": "slide", "vectors": [[1, 0], [2, 0], [0, 1], [-1, 0], [0, -1]],
            "symmetry": "none", "path": "circle"}
    problems = validate_spec(with_rider(move))
    assert any("point the same way" in p for p in problems), problems


def test_the_hex_paths_are_said() -> None:
    assert "turning to the next direction round" in move_sentence(HEX_ROSE, hexy=True)
    assert "circles" in move_sentence(HEX_RING, hexy=True)
    assert "zigzags" in move_sentence(HEX_BOLT, hexy=True)


# ------------------------------------------------------------------- the oracle


def _cell(file: int, rank: int, radius: int) -> bool:
    return (0 <= file <= 2 * radius and 0 <= rank <= 2 * radius
            and hex_distance(file - radius, rank - radius) <= radius)


def _sequences(move: dict[str, Any], side: int, length: int) -> list[list[tuple[int, int]]]:
    """Every run of ``length`` steps the path may take; the second player's directions are
    the first's turned half a turn (`orientation: "rotate"`)."""
    steps = _directions(move)
    if side == 1:
        steps = {(-x, -y) for x, y in steps}
    ring = _by_angle(steps)
    count = len(ring)
    out = []
    for start in range(count):
        if move["path"] == "circle":
            out.append([ring[(start + k) % count] for k in range(length)])
            out.append([ring[(start - k) % count] for k in range(length)])
            continue
        for neighbour in (ring[(start + 1) % count], ring[(start - 1) % count]):
            turn = math.degrees(_angle(neighbour) - _angle(ring[start])) % 360
            if 0 < min(turn, 360 - turn) < 180:
                out.append([ring[start] if k % 2 == 0 else neighbour for k in range(length)])
    return out


def oracle_targets(spec: dict[str, Any], board: bytes, side: int) -> set[tuple[int, int]]:
    """Every (origin, target) ``side``'s turning paths may make on a hexagon ``board``."""
    width = spec["board"]["width"]
    radius = width // 2
    n_types = len(spec["pieces"])
    found: set[tuple[int, int]] = set()
    for origin, value in enumerate(board):
        if value == 0 or (value - 1) // n_types != side:
            continue
        for move in spec["pieces"][(value - 1) % n_types]["moves"]:
            if move.get("path") not in ("circle", "zigzag"):
                continue
            for sequence in _sequences(move, side, width * width):
                file, rank = origin % width, origin // width
                seen = {origin}
                for dq, dr in sequence:
                    file, rank = file + dq, rank + dr
                    square = rank * width + file
                    if not _cell(file, rank, radius) or square in seen:
                        break
                    seen.add(square)
                    held = board[square]
                    if held == 0 or (held - 1) // n_types != side:
                        found.add((origin, square))
                    if held:
                        break
    return found


def _core_targets(engine: Engine, state: Any, spec: dict[str, Any]) -> set[tuple[int, int]]:
    n_types = len(spec["pieces"])
    turning = {i for i, piece in enumerate(spec["pieces"])
               if any(move.get("path") for move in piece["moves"])}
    return {
        (origin, target)
        for origin, target, _, _ in map(decode_move, engine.pseudo_legal_moves(state))
        if (state.board[origin] - 1) % n_types in turning
    }


def _walk(move: dict[str, Any], seed: int, plies: int = 80) -> int:
    """A seeded random game: after every move, the core's turning moves equal the oracle's."""
    spec = with_rider(move)
    engine = Engine(spec)
    chooser = random.Random(seed)
    state = engine.initial_state()
    checked = 0
    for _ in range(plies):
        assert _core_targets(engine, state, spec) == oracle_targets(
            spec, state.board, state.side_to_move)
        checked += 1
        legal = engine.legal_moves(state)
        if not legal or engine.terminal(state, legal) is not None:
            break
        state = engine.apply(state, chooser.choice(legal))
    return checked


@pytest.mark.parametrize("move", [HEX_ROSE, HEX_RING, HEX_BOLT], ids=["rose", "ring", "bolt"])
def test_the_core_and_the_oracle_agree_over_a_seeded_game(move: dict[str, Any]) -> None:
    """SR1c's acceptance: the hex rose validates and plays, and its moves are the oracle's
    for both players, on the hexagon's own edge."""
    assert _walk(move, 0) >= 10


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(1, 11))
@pytest.mark.parametrize("move", [HEX_ROSE, HEX_RING, HEX_BOLT], ids=["rose", "ring", "bolt"])
def test_the_core_and_the_oracle_agree_over_ten_seeded_games(
    move: dict[str, Any], seed: int
) -> None:
    assert _walk(move, seed) >= 2


def test_the_rose_reaches_further_than_the_knight() -> None:
    """From the start the rose has the knight's jumps and more: it turns on past them."""
    spec = with_rider(HEX_ROSE)
    engine = Engine(spec)
    state = engine.initial_state()
    rose = _core_targets(engine, state, spec)
    knight = Engine(load_spec(GLINSKI))
    knights = {
        (origin, target)
        for origin, target, _, _ in map(decode_move, knight.pseudo_legal_moves(state))
        if state.board[origin] and spec["pieces"][(state.board[origin] - 1) % 6]["id"] == "N"
    }
    assert knights < rose


def test_the_rose_plays_to_an_end() -> None:
    spec = with_rider(HEX_ROSE)
    engine = Engine(spec)
    chooser = random.Random(7)
    state = engine.initial_state()
    for _ in range(spec["rules"].get("max_plies", 400) + 1):
        legal = engine.legal_moves(state)
        if not legal or engine.terminal(state, legal) is not None:
            return
        state = engine.apply(state, chooser.choice(legal))
    pytest.fail("the game never ended")


# ------------------------------------------------------------------- the readers


def test_fairy_refuses_a_hex_path_by_name() -> None:
    from chess_variant_sandbox.fairy.variant import Untranslatable, translate

    with pytest.raises(Untranslatable) as refused:
        translate(with_rider(HEX_ROSE))
    assert any("circular and crooked modifiers" in reason
               for reason in refused.value.reasons), refused.value.reasons


def test_a_hex_rose_is_not_at_distance_zero_from_the_hex_knight() -> None:
    from chess_variant_sandbox.spec.distance import distance, features

    glinski = load_spec(GLINSKI)
    rose = with_rider(HEX_ROSE)
    assert features(glinski)["turning_share"] == 0.0
    assert features(rose)["turning_share"] > 0.0
    assert distance(glinski, rose) > 0.0


def test_the_inventory_names_a_hex_path_and_only_a_hex_path() -> None:
    from tests.feature_inventory import _schema_forty_nine_uses

    assert _schema_forty_nine_uses(with_rider(HEX_ROSE)) == {"pieces.moves.slide.path.hex"}
    assert _schema_forty_nine_uses(load_spec(GLINSKI)) == set()
    assert _schema_forty_nine_uses(load_spec(ROOT / "tests" / "_path_rose.json")) == set()
