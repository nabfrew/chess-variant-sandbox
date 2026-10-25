"""The hexagon's resolver in `spec/schema.py`, and the readers moved onto it (Q281a1).

`WRAP_PAIRS`, `hex_distance`, `absent_squares` and `wrap_stated` are reached by nothing but
these tests and the readers that already go through them, because the validator still
refuses `board.shape` and the pair keys (that is Q281a4). So this file proves two things
at the function level: the new pieces answer the spec (`docs/game-ideas.md` *Q281
specified*) on hand-built dicts, and every committed and archived spec reads exactly as it
did before the readers moved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import hex_geometry as oracle
import pytest
from retired_specs import hex_glinski_promoting

from chess_variant_sandbox.loop.grid import _edges
from chess_variant_sandbox.spec.distance import _wrap_features, features
from chess_variant_sandbox.spec.relate import _wrap_form
from chess_variant_sandbox.spec.schema import (
    MAX_BOARD_DIM,
    WRAP_FILE_EDGES,
    WRAP_PAIRS,
    WRAP_RANK_EDGES,
    absent_squares,
    hex_distance,
    obstacle_squares,
    validate_spec,
    wrap_axes,
    wrap_stated,
)

ROOT = Path(__file__).resolve().parents[1]
# The library as it stood before the hexagon: every committed and archived spec that
# declares no `board.shape`. Schema 16's showcases (Q281a4) are the ones that do, and
# `tests/test_schema_sixteen.py` holds them to what they claim instead.
LIBRARY = [
    path
    for path in sorted((ROOT / "rulesets").glob("*.json")) + sorted((ROOT / "archive").glob("*.json"))
    if "shape" not in json.loads(path.read_text(encoding="utf-8"))["board"]
]
RADII = range(1, 8)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _key(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


def _hexagon(radius: int, wrap: dict[str, Any] | None = None) -> dict[str, Any]:
    """A hand-built hexagon: only the keys the readers here look at."""
    size = 2 * radius + 1
    spec: dict[str, Any] = {
        "board": {"width": size, "height": size, "lattice": "hex", "shape": "hexagon"},
        "rules": {},
    }
    if wrap is not None:
        spec["rules"]["wrap"] = wrap
    return spec


def _old_two_axis_reading(wrap: Any) -> dict[str, tuple[Any, Any]]:
    """How `grid._edges`, `distance._wrap_features` and `describe._describe_wrap` read
    `rules.wrap` before the resolver: `X_by_player`, else `X` for both, else `"none"`."""
    out: dict[str, tuple[Any, Any]] = {}
    for axis in ("files", "ranks"):
        if not isinstance(wrap, dict):
            out[axis] = ("none", "none")
            continue
        stated = wrap.get(f"{axis}_by_player") or {
            "0": wrap.get(axis, "none"), "1": wrap.get(axis, "none")
        }
        out[axis] = (stated["0"], stated["1"])
    return out


class TestWrapPairs:
    def test_the_copies_at_r5_are_the_spec_table(self) -> None:
        assert {key: (pair.low, pair.high, pair.copy(5)) for key, pair in WRAP_PAIRS.items()} == {
            "n_s": ("s", "n", (-6, 11)),
            "ne_sw": ("sw", "ne", (5, 6)),
            "nw_se": ("nw", "se", (11, -5)),
        }

    @pytest.mark.parametrize("radius", RADII)
    def test_each_high_copy_is_the_oracles_copy_across_that_side(self, radius: int) -> None:
        """The oracle names the copy across each side from lattice units, not from here."""
        a, b = oracle.basis(radius)
        for pair in WRAP_PAIRS.values():
            (i, j), = [c for c, side in oracle.SIDE_OF_COPY.items() if side == pair.high]
            assert pair.copy(radius) == (i * a[0] + j * b[0], i * a[1] + j * b[1])
            (i, j), = [c for c, side in oracle.SIDE_OF_COPY.items() if side == pair.low]
            assert tuple(-x for x in pair.copy(radius)) == (
                i * a[0] + j * b[0], i * a[1] + j * b[1]
            )

    @pytest.mark.parametrize("radius", RADII)
    def test_every_copy_is_one_board_width_away(self, radius: int) -> None:
        for pair in WRAP_PAIRS.values():
            assert hex_distance(*pair.copy(radius)) == 2 * radius + 1

    @pytest.mark.parametrize(
        ("origin", "vector", "side", "lands"),
        [((2, 10), (0, 1), "n", (8, 0)), ((10, 2), (1, 0), "se", (0, 7))],
        ids=["c11 up to i1", "k3 right to a8"],
    )
    def test_the_worked_crossings(
        self, origin: tuple[int, int], vector: tuple[int, int], side: str,
        lands: tuple[int, int],
    ) -> None:
        radius = 5
        (copy,) = [p.copy(radius) for p in WRAP_PAIRS.values() if p.high == side]
        raw = (origin[0] + vector[0], origin[1] + vector[1])
        assert hex_distance(raw[0] - radius, raw[1] - radius) > radius
        assert (raw[0] - copy[0], raw[1] - copy[1]) == lands

    def test_wrap_axes_on_a_grid_are_todays_two(self) -> None:
        assert wrap_axes() == (("files", WRAP_FILE_EDGES), ("ranks", WRAP_RANK_EDGES))
        assert wrap_axes("grid") == wrap_axes()

    def test_wrap_axes_on_a_hexagon_are_the_three_pairs(self) -> None:
        assert wrap_axes("hexagon") == (
            ("n_s", ("none", "s", "n", "both")),
            ("ne_sw", ("none", "sw", "ne", "both")),
            ("nw_se", ("none", "nw", "se", "both")),
        )


class TestHexDistance:
    @pytest.mark.parametrize(
        ("q", "r", "expected"),
        [(0, 0, 0), (1, 0, 1), (1, -1, 1), (1, 1, 2), (2, -1, 2), (3, -1, 3), (-6, 11, 11)],
    )
    def test_hand_counted(self, q: int, r: int, expected: int) -> None:
        assert hex_distance(q, r) == expected


class TestAbsentSquares:
    @pytest.mark.parametrize("radius", RADII)
    def test_the_count(self, radius: int) -> None:
        size = 2 * radius + 1
        absent = absent_squares(_hexagon(radius)["board"])
        assert len(absent) == size * size - (3 * radius * radius + 3 * radius + 1)

    @pytest.mark.parametrize("radius", RADII)
    def test_the_cells_are_the_oracles(self, radius: int) -> None:
        size = 2 * radius + 1
        absent = absent_squares(_hexagon(radius)["board"])
        cells = {oracle.square(cell, size) for cell in oracle.cells(radius)}
        assert cells == set(range(size * size)) - absent

    def test_at_r5_they_are_glinskis_thirty_walls(self) -> None:
        spec = hex_glinski_promoting()
        walls = obstacle_squares(spec["rules"], 11, 11, spec["setup"]["orientation"])
        assert len(walls) == 30
        assert absent_squares(_hexagon(5)["board"]) == walls

    @pytest.mark.parametrize(
        "board",
        [
            None,
            "hexagon",
            {"width": 11, "height": 11, "lattice": "hex"},
            {"width": 11, "height": 11, "lattice": "hex", "shape": "grid"},
            {"width": 10, "height": 10, "shape": "hexagon"},
            {"width": 11, "height": 9, "shape": "hexagon"},
            {"width": True, "height": True, "shape": "hexagon"},
            {"width": "11", "height": "11", "shape": "hexagon"},
            {"height": 11, "shape": "hexagon"},
            {"width": MAX_BOARD_DIM + 1, "height": MAX_BOARD_DIM + 1, "shape": "hexagon"},
        ],
    )
    def test_empty_on_anything_else(self, board: Any) -> None:
        assert absent_squares(board) == frozenset()

    def test_every_committed_board_has_none(self) -> None:
        for path in LIBRARY:
            assert absent_squares(_load(path)["board"]) == frozenset(), path.name


class TestWrapStated:
    @pytest.mark.parametrize("path", LIBRARY, ids=_key)
    def test_every_committed_spec_reads_as_before(self, path: Path) -> None:
        spec = _load(path)
        wrap = spec.get("rules", {}).get("wrap")
        assert "shape" not in spec["board"]
        assert wrap_stated(wrap, spec["board"].get("shape")) == _old_two_axis_reading(wrap)

    def test_the_library_has_wraps_to_read(self) -> None:
        """Guard on the test above: it must meet shared, per-player and one-way seams."""
        values = [
            _load(path).get("rules", {}).get("wrap") for path in LIBRARY
        ]
        wraps = [w for w in values if w]
        assert len(wraps) >= 5
        assert any("files_by_player" in w or "ranks_by_player" in w for w in wraps)

    @pytest.mark.parametrize(
        ("wrap", "expected"),
        [
            (None, {"files": ("none", "none"), "ranks": ("none", "none")}),
            ("both", {"files": ("none", "none"), "ranks": ("none", "none")}),
            ({}, {"files": ("none", "none"), "ranks": ("none", "none")}),
            ({"files": "both"}, {"files": ("both", "both"), "ranks": ("none", "none")}),
            (
                {"ranks_by_player": {"0": "top", "1": "bottom"}},
                {"files": ("none", "none"), "ranks": ("top", "bottom")},
            ),
        ],
    )
    def test_a_grid(self, wrap: Any, expected: dict[str, Any]) -> None:
        assert wrap_stated(wrap) == expected
        assert wrap_stated(wrap, "grid") == expected

    def test_a_hexagon_reads_the_three_pairs(self) -> None:
        wrap = {"nw_se": "both", "n_s_by_player": {"0": "n", "1": "s"}, "files": "both"}
        assert wrap_stated(wrap, "hexagon") == {
            "n_s": ("n", "s"), "ne_sw": ("none", "none"), "nw_se": ("both", "both"),
        }

    def test_a_player_the_object_leaves_out_falls_back_to_the_shared_value(self) -> None:
        """Only malformed input reaches this; it is `relate`'s reading, kept."""
        assert wrap_stated({"files": "left", "files_by_player": {"0": "right"}})["files"] == (
            "right", "left"
        )


class TestTheReadersOnAHexagon:
    """What each moved reader answers on a hand-built hexagon, before the gate opens."""

    def test_a_seam_is_the_joined_bin(self) -> None:
        assert _edges(_hexagon(5, {"nw_se": "both"})) == 1
        assert _edges(_hexagon(5, {"n_s_by_player": {"0": "n", "1": "none"}})) == 1

    def test_no_seam_is_the_edged_bin(self) -> None:
        assert _edges(_hexagon(5)) == 0
        assert _edges(_hexagon(5, {"n_s": "none"})) == 0
        # The rectangle's keys mean nothing on a hexagon (the validator will refuse them).
        assert _edges(_hexagon(5, {"files": "both"})) == 0

    def test_distance_sees_a_one_sided_pair(self) -> None:
        wrap = {"ne_sw_by_player": {"0": "ne", "1": "sw"}}
        assert _wrap_features(wrap, "hexagon")["wraps_asymmetrically"] == 1.0
        assert _wrap_features({"ne_sw": "both"}, "hexagon")["wraps_asymmetrically"] == 0.0

    def test_relate_keeps_the_pairs_apart(self) -> None:
        assert _wrap_form(_hexagon(5)) is None
        assert _wrap_form(_hexagon(5, {"nw_se": "both"})) == (
            ("none", "none"), ("none", "none"), ("both", "both"),
        )


class TestTheValidatorStillRefusesIt:
    """Inert: nothing outside these tests reaches a hexagon until Q281a4."""

    @staticmethod
    def _glinski() -> dict[str, Any]:
        return hex_glinski_promoting()

    @pytest.mark.parametrize("key", ["n_s", "ne_sw", "nw_se", "n_s_by_player"])
    def test_a_pair_key(self, key: str) -> None:
        spec = self._glinski()
        spec["rules"]["wrap"] = {key: "none"}
        assert any(key in error for error in validate_spec(spec))


# `distance.features` of every committed and archived spec, computed by the code before
# the readers moved (`4c5cfbe`): sha256 of `json.dumps(features, sort_keys=True)`, first
# sixteen hex digits. A feature added later goes in `_ADDED_SINCE`, which keeps it out of
# the digest, so the ones that were here must still read the same.
_ADDED_SINCE: frozenset[str] = frozenset({
    "wraps_n_s", "wraps_ne_sw", "wraps_nw_se",  # Q281a3
    "strike_stays_put", "strike_rebirths", "capturer_switches_sides", "explodes_alone",  # Q540
    "contacts_keep_the_mover",  # Q781
    "extinction_per_type", "promotes_only_lost",  # Q783
    "drops_placement",  # Q784
    "drops_either_face",  # Q785
    "simultaneous", "simultaneous_initiative", "simultaneous_impossible", "beats_share",  # Q918a
    "wins_by_leaving",  # Q786
    "portal_pairs", "portals_move", "portals_one_way",  # Q771
    "portals_no_back",  # Q807
    "portals_ring", "portals_funnel", "portals_other_map", "portals_filtered",  # Q843a
    "portals_fold",  # Q843c
    "npcs_share_a_walk",  # Q850
    "facing_fixed",  # Q869
    "simultaneous_rest", "simultaneous_swap_collides", "simultaneous_both_score",  # Q942a
    "spy_types",  # Q935a
    "speed_ranks", "speed_royal", "speed_by_value", "speed_by_reach",  # Q943a
    "rest_mean", "rest_max",  # Q983a
    "sight_freeze_share", "sight_disarm_share", "sight_shield_share", "sight_aura_share",  # Q999a
    "sight_command_share", "sight_turncoat_share",  # Q1007c
    "lame_rider_share",  # Q1007a
    "turning_share",  # T1
    "strikers_share",  # A1
})
_FEATURES_BEFORE_THE_RESOLVER: dict[str, str] = {
    "rulesets/almost": "9035530ccf64d8cd",
    "rulesets/antichess": "2fe16943c39e9212",
    "rulesets/atomic": "da0551bdcad0ddad",
    "rulesets/bastion": "04500d3db1737a81",
    "rulesets/berolina": "488735d94bc4a1ee",
    "rulesets/capablanca": "8cfb5815c674f10c",
    "archive/chess-oriented-pawns": "f1530cae8408d901",
    "rulesets/chess": "c638843d0cbb3fe4",
    "rulesets/courier": "425cdfa92363dc89",
    "archive/crazyhouse-torus": "efcaa301e7874eff",  # Q869: the rear pawn respelled as a fixed facing (was a0afec7cf0db257c)
    "rulesets/crazyhouse": "89bef2937fd67dd5",
    "rulesets/cylinder": "fe858992498b3166",
    "rulesets/dobutsu": "291093f5f614ed54",
    "rulesets/dull": "05d801c86261eabd",
    "rulesets/gardner": "3f1a0eb8f663987a",
    "rulesets/horde": "4df671b02e01c2fc",
    "rulesets/intransitive": "d89b77587de03c8c",
    "rulesets/kingofthehill": "fc8d59e276b65a6d",
    "rulesets/knightmate": "f510d76d13cadc00",
    "rulesets/losalamos": "563f6210d832dcb9",
    "rulesets/makruk": "e91699390ef310ea",
    "archive/onewaycylinder": "ff6c7bc9512dedfa",
    "archive/onewaytorus": "d97610a080d2797a",
    "rulesets/racingkings": "e36e3c227d55b6cc",
    "rulesets/rps-hierarchy-chess": "34e88e16719159cf",
    "rulesets/rps-v1": "9d62600de20d2821",
    "rulesets/rps-v2": "e30c9ecc74798fca",
    "rulesets/rps-v3": "34ee8433fa237eea",
    "rulesets/rps-v4": "c272f0494f483dcf",
    "rulesets/rps-v5": "b91364abfa8fc2c9",
    "archive/sardines": "de0716da8f51261b",
    "rulesets/shatranj": "291e3c37e50edd01",
    "rulesets/shogi": "9399bd9c1e2cc449",
    "rulesets/threecheck": "d193b87292055a74",
    "archive/torus-less-extended": "1dcc63ed58bc931e",  # Q869: the rear pawn respelled as a fixed facing (was 0870ea5726ae3adf)
    "rulesets/torus": "a9b670d4a057ad89",
    "rulesets/xiangqi": "51cec6a3a851bc63",
    "archive/andernach": "352d33d9f6b29010",  # Q540, schema 17
    "archive/circe": "352d33d9f6b29010",  # Q540, schema 17
    "archive/kamikaze": "da0551bdcad0ddad",  # Q540, schema 17
    "archive/rifle": "352d33d9f6b29010",  # Q540, schema 17
    "archive/shogi-drop-promoted": "e533c3319ca37e67",  # Q787
    "archive/shogi-backward-drops": "05d80d6d640c72bc",  # Q787
    "archive/pawn-knockdown": "e4b0982ee874a806",  # Q787
    "archive/divide": "24d51a6b605d2aa5",  # Q772a, schema 19
    "archive/ferry": "62007817ab0c2bfb",  # Q772a, schema 19
    "archive/ferry-no-straight-back": "3cd0008b2552d704",  # Q807, schema 20
    "archive/wolves-patrol": "ad7a6f1e406f28f3",  # Q850, schema 23
    "archive/carousel": "cce67fce00c2207f",  # Q772a, schema 19
    "archive/wormhole": "a82a1c3ee0f4049b",  # Q772a, schema 19
    "archive/shut-wormhole": "8a7ee8f749713bb5",  # Q772a, schema 19
    "archive/cylinder-wormhole": "af2b3931c3b0dda1",  # Q772a, schema 19
    "archive/einstein": "5b12c7cca128a3a7",  # Q781, schema 18
    # Q781, schema 18. Q923, 2026-10-15: `promotion_zone_share` leaves its walled zone squares
    # out, 0.1154 to 0.0962 (the one library spec it moves; was 5e421c4ae0d238e9).
    "archive/omega": "a8d75e2022231e51",
    "archive/extinction-chess": "8f42396c44179f33",  # Q783, schema 18
    "archive/grand": "227f27e884fc2efc",  # Q783, schema 18
    "archive/placement-chess": "46bff0e6d06b6fa7",  # Q784, schema 18
    "archive/sittuyin": "f3a43bd577292f85",  # Q784, schema 18
    "archive/kyoto": "2aed5b07a6185d43",  # Q785, schema 18
    "archive/jeson-mor": "2a87f67280ed3538",  # Q786, schema 18
    # Q870, 2026-10-13: `archive/hex-glinski-turned` and `-turned-mirrored` retired for `pinwheel`,
    # which has a board shape and so is not read here (rows were cb415ec30ffc72bd and eaa3e61fd8e5cd3a).
    "archive/cannonade": "9e78a35b5d909262",
    "archive/chess-sweeping-bishops": "980b6362973afb82",
    "archive/hierarchy-extended": "157ab1f4fa7f0800",
    "archive/hierarchy-fall": "352d33d9f6b29010",
    "archive/hierarchy-rise-stay": "f3506f940053ff23",
    "archive/hierarchy-rise-throw": "9a8baa7002b61992",
    "archive/hierarchy-rise": "9e376f4d864a4c7e",
    "archive/marchfield": "40770b3fd3a78af9",
    "archive/merger": "4ffa63d4d9d8c4e7",
    "archive/perpetual": "08607e54f46c8204",
    "archive/quarry": "9a46cc0fe3d33493",
    "archive/queen-bomb": "96448cc509033d5a",
    "archive/queens-wake": "96448cc509033d5a",
    "archive/reckoning": "2df32b7c7e8739f7",
    "archive/stones": "c35f4dc8fb3b5ddd",
    "archive/succession": "2b280a64cd4dcfbf",
    "archive/three-stones": "4b5ae063d0a8a24d",
    "archive/weathervane-half-turn": "2446af4757bc9152",
    "archive/weathervane": "48a037bdc8b167d9",
    "archive/wolves-bounty": "d4069a814edcdb9e",
    # Q876, 2026-10-13: `archive/wolves-immune` retired for `pitchfork` (the row was 63b9005f18b28cdf).
    "archive/pitchfork": "63b9005f18b28cdf",  # Q876, schema 25
    "archive/sluice": "29e224df70ed859d",  # Q844b, schema 21
    "archive/crease": "f4debbefb2770d57",  # Q844b, schema 24
    "archive/pawnproof-wormhole": "8a7ee8f749713bb5",  # Q844b, schema 21
    # Q962b, 2026-10-18: `archive/gardner-in-rounds` retired from the previews (`docs/reference/previews-2026-10-18.md`); the row was ef35aeadf82e3d4e.
    # Q962b, 2026-10-18: `archive/gardner-in-rounds-with-meek-pawns` retired from the previews (`docs/reference/previews-2026-10-18.md`); the row was e30dc7de2f1226d9.
    # The same features as gardner in rounds: `distance.features` has none for the tie rule.
    # Q962b, 2026-10-18: `archive/gardner-in-rounds-with-the-initiative` retired from the previews (`docs/reference/previews-2026-10-18.md`); the row was ef35aeadf82e3d4e.
    "archive/watering-hole": "449300665f7cd831",  # Q918e, schema 26
    # Q962c, 2026-10-19: `archive/gardner-in-rounds-resting` retired from the previews as a near copy of breather (`docs/reference/previews-2026-10-18.md`); the row was e058e13b9d1bc774.
    # The same features as resting: each chase switch is one feature of `distance` that
    # `_ADDED_SINCE` leaves out, and both add one special rule.
    # Q962b, 2026-10-18: `archive/gardner-in-rounds-where-swaps-collide` retired from the previews (`docs/reference/previews-2026-10-18.md`); the row was e058e13b9d1bc774.
    # Q962b, 2026-10-18: `archive/gardner-in-rounds-scored` retired from the previews (`docs/reference/previews-2026-10-18.md`); the row was 7ecccaf8caa59060.
    "archive/breather": "cdf0efe85ea762c5",  # Q942c, schema 27; Q962: renamed from gardner-in-rounds-against-the-chase
    # Q962b, 2026-10-18: `archive/gardner-with-spies` retired from the previews (`docs/reference/previews-2026-10-18.md`); the row was d488370eab00cfbd.
    # Q1005, 2026-10-23: `archive/gardner-with-spies-resting` retired from the previews (`docs/reference/preview-call-2026-10.md`); the row was a12a32ec9aa7c11a.
    "archive/riders-and-leapers": "6917cf0a1e6e5bbd",  # Q936s, schema 27
    # Q1005, 2026-10-23: `archive/gardner-in-rounds-with-a-swift-king` retired from the previews (`docs/reference/preview-call-2026-10.md`); the row was 6fdc011869f7824a.
    # Q1005, 2026-10-23: `archive/gardner-in-rounds-with-swift-knights` retired from the previews (`docs/reference/preview-call-2026-10.md`); the row was 6fdc011869f7824a.
    # Q1005, 2026-10-23: `archive/watering-hole-by-speed` retired from the previews (`docs/reference/preview-call-2026-10.md`); the row was b0e8b91febef654d.
    "archive/circe-mortal-diagonals": "352d33d9f6b29010",  # Q798, schema 17; circe's features: none reads `immune`
    "archive/the-golem-well": "6d8fd9917d699345",  # Q994p, schema 28, a model's design (Q994's blind trial)
    "archive/kraken-ford": "a9cd50a19f198066",  # Q994p, schema 28, a model's design (Q994's blind trial)
    # Q1005, 2026-10-23: `archive/gardner-with-weary-heavies` retired from the previews (`docs/reference/preview-call-2026-10.md`); the row was b7bba61ec683778a.
    # Q1005, 2026-10-23: `archive/breather-by-weight` retired from the previews (`docs/reference/preview-call-2026-10.md`); the row was 95ab3bf4fa8a527d.
    "archive/chess-with-a-weary-queen": "cca02b4d69cf1272",  # Q983d, schema 29
    "archive/opposite-colours": "d36bc9991015602c",  # Q999b, schema 30; Q1007e, 2026-10-24: replaced under its name by R1, riders as rook and queen (was a0e70d8f14e2936a)
    "archive/chess-with-nightriders": "57d71a18e3db3ea9",  # Q1007e, schema 31
    "archive/crooked-bishops": "cca02b4d69cf1272",  # T2, generation 34; chess's own, `turning_share` added since
    "archive/circling-queen": "cca02b4d69cf1272",  # T3, generation 34; chess's own, `turning_share` added since
    "archive/horse-archers": "352d33d9f6b29010",  # A2, generation 35; `strikers_share` 1/6 (the knights of six capturing types)
    "archive/anchors": "b57b92f43654cb71",  # I3, generation 33
    "archive/madrasi": "b6e19675623090e2",  # Q999d, schema 30
    "archive/chess-with-a-sheltering-king": "e7321c09e075fea8",  # Q999d, schema 30
    "archive/chess-with-daunting-knights": "cca02b4d69cf1272",  # Q999d, schema 30
    "archive/wolves": "ad7a6f1e406f28f3",
}


class TestFeaturesAreUnchanged:
    def test_the_freeze_covers_the_library(self) -> None:
        assert set(_FEATURES_BEFORE_THE_RESOLVER) == {_key(path) for path in LIBRARY}

    @pytest.mark.parametrize("path", LIBRARY, ids=_key)
    def test_every_committed_spec(self, path: Path) -> None:
        measured = {
            name: value for name, value in features(_load(path)).items()
            if name not in _ADDED_SINCE
        }
        digest = hashlib.sha256(json.dumps(measured, sort_keys=True).encode()).hexdigest()
        assert digest[:16] == _FEATURES_BEFORE_THE_RESOLVER[_key(path)]
