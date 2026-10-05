"""`archive/hex-glinski-promoting.json` is Glinski's board, read off the spec (Q50, Q280).

The hex-ness is the author's choice of vectors on an 11x11 axial rhombus, with thirty
walls cutting it to Glinski's 91-cell hexagon, and `board.lattice: "hex"` declares it so
the validator can police the spellings that would compile a plausible wrong hex game.
This file reads the spec rather than the engine's idea of chess: the walls, the vector
sets, the spellings the declaration forbids, and the hand-counted opening move list.

Axial coordinates: file ``q``, rank ``r``. The six neighbours are ``(+-1, 0)``,
``(0, +-1)``, ``(1, -1)`` and ``(-1, 1)``. A cell is on the board when
``5 <= q + r <= 15``.

`docs/reference/hex-convention.md` holds the reasoning and the measurement.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import move_from
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION, obstacle_squares, validate_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "archive" / "hex-glinski-promoting.json"
SIZE = 11

NEIGHBOURS = {(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)}
DIAGONALS = {(1, 1), (-1, -1), (2, -1), (-2, 1), (1, -2), (-1, 2)}


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def counts(spec: dict[str, Any]) -> collections.Counter[int]:
    """How many legal first moves leave each cell."""
    engine = Engine(spec)
    return collections.Counter(move_from(m) for m in engine.legal_moves(engine.initial_state()))


def _cell(name: str) -> int:
    return (int(name[1:]) - 1) * SIZE + ord(name[0]) - ord("a")


def _hex_distance(dq: int, dr: int) -> int:
    return max(abs(dq), abs(dr), abs(dq + dr))


def _vectors(spec: dict[str, Any], piece_id: str) -> set[tuple[int, int]]:
    piece = next(p for p in spec["pieces"] if p["id"] == piece_id)
    return {tuple(v) for move in piece["moves"] for v in move["vectors"]}  # type: ignore[misc]


class TestTheSpellingIsAHexBoard:
    def test_it_validates_at_the_current_schema(self, spec: dict[str, Any]) -> None:
        assert spec["schema_version"] == 14 <= SCHEMA_VERSION
        assert validate_spec(spec) == []

    def test_the_board_declares_the_lattice(self, spec: dict[str, Any]) -> None:
        assert spec["board"]["lattice"] == "hex"

    def test_the_walls_cut_the_rhombus_to_glinskis_hexagon(self, spec: dict[str, Any]) -> None:
        walls = obstacle_squares(spec["rules"], SIZE, SIZE, spec["setup"]["orientation"])
        expected = {r * SIZE + q for q in range(SIZE) for r in range(SIZE) if not 5 <= q + r <= 15}
        assert walls == expected
        assert SIZE * SIZE - len(walls) == 91

    def test_the_second_army_is_rotated_not_mirrored(self, spec: dict[str, Any]) -> None:
        """Wrong spelling 1: `mirror` sends neighbour (1, -1) to non-neighbour (1, 1)."""
        assert spec["setup"]["orientation"] == "rotate"
        assert "reach" not in spec["rules"]

    def test_no_move_uses_a_square_lattice_symmetry(self, spec: dict[str, Any]) -> None:
        """Wrong spelling 2: `rotate4` and `dihedral8` expand off the hex lattice."""
        for piece in spec["pieces"]:
            for move in piece["moves"]:
                assert move.get("symmetry", "none") == "none", piece["id"]

    def test_no_piece_faces(self, spec: dict[str, Any]) -> None:
        """Wrong spelling 3: a quarter turn, which the hex lattice lacks."""
        assert all("facing" not in piece for piece in spec["pieces"])

    def test_the_promotion_zone_is_squares_and_not_a_rank(self, spec: dict[str, Any]) -> None:
        """Wrong spelling 4: the far edge of a hexagon is a zigzag, not a rank."""
        promotion = spec["rules"]["promotion"]
        assert "rank" not in promotion and "ranks" not in promotion
        assert len(promotion["squares"]) == 11

    def test_the_vector_sets_are_the_hex_pieces(self, spec: dict[str, Any]) -> None:
        assert _vectors(spec, "R") == NEIGHBOURS
        assert _vectors(spec, "B") == DIAGONALS
        assert _vectors(spec, "Q") == NEIGHBOURS | DIAGONALS
        assert _vectors(spec, "K") == NEIGHBOURS | DIAGONALS
        knight = _vectors(spec, "N")
        assert len(knight) == 12
        for dq, dr in knight:
            assert _hex_distance(dq, dr) == 3
            assert (dq, dr) not in {(3 * a, 3 * b) for a, b in NEIGHBOURS}


class TestHandCountedMovesFromTheStart:
    """Counts worked by hand on the axial board, walls and blockers included.

    The total, 51, is also the number of opening moves usually quoted for Glinski's
    chess. The second army is rotated rather than reflected, so the deeper counts are
    this engine's own; they are pinned as a regression in
    `test_schema_fourteen.py::TestItCompiles`.
    """

    @pytest.mark.parametrize(
        ("origin", "piece", "expected"),
        [
            ("c4", "rook", 3),
            ("i1", "rook", 3),
            ("f1", "bishop", 2),
            ("f2", "bishop", 8),
            ("f3", "bishop", 2),
            ("d3", "knight", 4),
            ("h1", "knight", 4),
            ("e2", "queen", 6),
            ("g1", "king", 2),
            ("f5", "pawn", 1),
            ("g4", "pawn", 2),
        ],
    )
    def test_one_piece(
        self, counts: collections.Counter[int], origin: str, piece: str, expected: int
    ) -> None:
        assert counts[_cell(origin)] == expected, f"{piece} on {origin}"

    def test_the_whole_first_move(self, counts: collections.Counter[int]) -> None:
        assert sum(counts.values()) == 51
