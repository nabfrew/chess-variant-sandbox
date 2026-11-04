"""R14: `advance` and `centrality` read the board's shape from compiled tables, not the index.

Until R14 both features read a square's rank and file as `square // width` and `square %
width`, in the core (`features.rs`) and in the oracle (`agents/features.py`) alike. That is
the rhombus the board is stored in. On a hexagon it measured distance from the middle in the
stored frame and missed one of the board's three pairs of sides, so a cell two steps from
the centre read as one; on a wrapped board it measured distance to a seam, which is no edge.
`compile._feature_geometry` now resolves both per square and player (rule 4), and these tests
hold three things: the old reading was wrong on a hexagon and a wrapped board, the new one is
right there, and on a board with plain edges nothing moved. The last arms hold the core to
the oracle on the boards whose answers changed.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import CompiledSpec, compile_spec
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.play import play_game
from tests.test_feature_terms import _assert_valued_alike, _heuristic, _vector

ROOT = Path(__file__).resolve().parents[1]
HEXAGON = ROOT / "archive" / "hex-glinski-hexagon.json"
PINWHEEL = ROOT / "archive" / "pinwheel.json"  # a hexagon with a one-way nw_se seam
CYLINDER = ROOT / "rulesets" / "cylinder.json"
TORUS = ROOT / "rulesets" / "torus.json"

# The six axial king steps of a hex lattice, (file, rank).
HEX_STEPS = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))


def _compiled(path: Path) -> CompiledSpec:
    return compile_spec(load_spec(path))


def _old_centre(spec: CompiledSpec, square: int) -> int:
    """The pre-R14 reading, doubled: Chebyshev distance from the frame's middle."""
    rank, file = divmod(square, spec.width)
    return max(abs(2 * file - (spec.width - 1)), abs(2 * rank - (spec.height - 1)))


def _old_rank(spec: CompiledSpec, player: int, square: int) -> int:
    rank = square // spec.width
    return rank if player == 0 else spec.height - 1 - rank


def _steps_from_centre(spec: CompiledSpec) -> dict[int, int]:
    """King steps from a hexagon's centre cell to every cell on it, walked cell by cell.

    An independent answer: the hexagon's cells are the squares that are not absent, and a
    step is one of the six axial neighbours. Nothing here reads a feature table.
    """
    width = spec.width
    on_board = set(range(spec.n_squares)) - set(spec.absent)
    centre = (spec.n_squares - 1) // 2
    steps = {centre: 0}
    queue = deque([centre])
    while queue:
        square = queue.popleft()
        rank, file = divmod(square, width)
        for d_file, d_rank in HEX_STEPS:
            f, r = file + d_file, rank + d_rank
            if not (0 <= f < width and 0 <= r < spec.height):
                continue
            neighbour = r * width + f
            if neighbour in on_board and neighbour not in steps:
                steps[neighbour] = steps[square] + 1
                queue.append(neighbour)
    return steps


def test_on_a_hexagon_the_centre_distance_is_the_steps_from_the_centre() -> None:
    spec = _compiled(HEXAGON)
    assert spec.wrap is None
    steps = _steps_from_centre(spec)
    assert len(steps) == spec.n_squares - len(spec.absent)
    for player in (0, 1):
        for square, count in steps.items():
            assert spec.feature_centre[player][square] == 2 * count, square
    # And the old reading was wrong: a cell two steps out read as one step out.
    wrong = [square for square, count in steps.items() if _old_centre(spec, square) != 2 * count]
    assert wrong
    centre = (spec.n_squares - 1) // 2
    two_out = centre + spec.width + 1  # one file right and one rank up: (1, 1) in axial
    assert steps[two_out] == 2 and _old_centre(spec, two_out) == 2


def test_a_hexagon_seam_is_no_edge_for_the_player_who_crosses_it() -> None:
    """Pinwheel joins the nw_se pair one way for each player: the file axis's sides are a
    seam for both, so the distance reads only the other two pairs."""
    spec = _compiled(PINWHEEL)
    width = spec.width
    for player in (0, 1):
        for square in set(range(spec.n_squares)) - set(spec.absent):
            rank, file = divmod(square, width)
            along_file, along_rank = 2 * file - (width - 1), 2 * rank - (spec.height - 1)
            assert spec.feature_centre[player][square] == max(
                abs(along_rank), abs(along_file + along_rank)
            )


def test_on_a_cylinder_every_file_is_as_central_as_every_other() -> None:
    spec = _compiled(CYLINDER)
    width = spec.width
    for player in (0, 1):
        for rank in range(spec.height):
            row = {spec.feature_centre[player][rank * width + file] for file in range(width)}
            assert len(row) == 1, rank
    # The old reading made the a- and h-files the edge of a board that has none there.
    assert _old_centre(spec, 3 * width) != _old_centre(spec, 3 * width + 3)
    # Ranks still have edges, so advance reads as it did.
    for player in (0, 1):
        assert spec.feature_rank[player] == tuple(
            _old_rank(spec, player, square) for square in range(spec.n_squares)
        )


def test_on_a_torus_nothing_is_central_and_there_is_no_far_side() -> None:
    spec = _compiled(TORUS)
    for player in (0, 1):
        assert set(spec.feature_centre[player]) == {0}
        assert set(spec.feature_rank[player]) == {0}


@pytest.mark.parametrize("name", ["chess", "gardner", "capablanca", "xiangqi", "shogi"])
def test_a_board_with_plain_edges_reads_what_it_always_did(name: str) -> None:
    spec = _compiled(ROOT / "rulesets" / f"{name}.json")
    assert spec.feature_rank_span == spec.height - 1
    for player in (0, 1):
        assert spec.feature_rank[player] == tuple(
            _old_rank(spec, player, square) for square in range(spec.n_squares)
        )
        assert spec.feature_centre[player] == tuple(
            _old_centre(spec, square) for square in range(spec.n_squares)
        )


# The core against the oracle on the boards whose answers moved: each term alone, one ply
# deep, at every position of a short game its own player plays (`test_feature_terms.py`'s
# comparison, to the bit).
GEOMETRY_ARMS: list[tuple[str, str, Path]] = [
    ("hexagon-centrality", "centrality", HEXAGON),
    ("seamed-hexagon-centrality", "centrality", PINWHEEL),
    ("cylinder-centrality", "centrality", CYLINDER),
    ("torus-advance", "advance", TORUS),
    ("hexagon-advance", "advance", HEXAGON),
]
POSITIONS = 12


@pytest.mark.parametrize(
    "label,term,path", GEOMETRY_ARMS, ids=[arm[0] for arm in GEOMETRY_ARMS]
)
def test_the_core_reads_the_geometry_as_the_oracle_does(label: str, term: str, path: Path) -> None:
    engine = Engine(load_spec(path))
    weights = _vector(material=1.0, mobility=0.05, **{term: 2.0})
    player = f"minimax:h={_heuristic(weights, f'material-and-{term}')},nodes=150"
    record = play_game(engine, (player, "greedy"), seed=3)
    states: list[Any] = []
    state = engine.initial_state()
    for move in record.moves[:POSITIONS]:
        states.append(state)
        state = engine.apply(state, move)
    alone = f"minimax:h={_heuristic(_vector(**{term: 1.0}), term)},nodes=1000000,max_depth=1"
    _assert_valued_alike(engine, alone, states)
