"""Schema 14: `board.lattice: "hex"` and `promotion.squares` (Q229).

`board.lattice: "hex"` declares a convention a spec could only keep by hand, so the
validator can refuse the spellings that compile a plausible wrong hex game and the prose
and the boards can say the board is hexagonal. `promotion.squares` states a promotion zone
as named squares, which a hexagon's far edge needs and which closes legan's shaped zone on
a square board. Specified in `docs/game-ideas.md`, *Q229 specified*.

The inertness proof is `test_schema_nine.py::TestTheBumpMovedNothing`, which pins every
committed and archived ruleset's mechanics hash and tables fingerprint. This file holds
the feature: gated, refused where it would read wrong, compiled, and read by every surface
`new-move-primitive` names.

Axial coordinates: file `q`, rank `r`. A cell is on Glinski's board when `5 <= q + r <= 15`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from retired_specs import hex_glinski_promoting

from chess_variant_sandbox.agents.features import feature_context
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import MOVE_PROMOTION, move_from, move_to, move_type
from chess_variant_sandbox.fairy import supports
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.llm.propose import SCHEMA_PREAMBLE
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.distance import distance, features
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.schema import (
    LATTICES,
    SCHEMA_VERSION,
    obstacle_squares,
    reflect_square,
)
from chess_variant_sandbox.viz.wrapped import zone_payload
from tests.feature_inventory import _schema_fourteen_uses

ROOT = Path(__file__).resolve().parents[1]
SIZE = 11

# The first promotion possible is the mover's fifth move, ply 9, so through depth 8 the
# ladder is the one this board gives with no promotion rule at all. Depths 1 to 3 are
# fast; depth 4 is `slow`.
LADDER = {1: 51, 2: 2_541, 3: 132_581}
LADDER_DEEP = {4: 6_873_785}

FAR_EDGE = ["a11", "b11", "c11", "d11", "e11", "f11", "g10", "h9", "i8", "j7", "k6"]


@pytest.fixture(scope="module")
def showcase() -> dict[str, Any]:
    return hex_glinski_promoting()


@pytest.fixture(scope="module")
def square_grid(showcase: dict[str, Any]) -> dict[str, Any]:
    """The showcase with both schema-14 keys off — the hexagon drawn on a square grid.

    Q280 retired `archive/hex-glinski.json`, which spelled this by hand, so the arm the
    two keys are read against is built here.
    """
    spec = copy.deepcopy(showcase)
    spec["schema_version"] = 12
    spec["board"].pop("lattice")
    spec["rules"].pop("promotion")
    return spec


@pytest.fixture(scope="module")
def chess() -> dict[str, Any]:
    return json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))


def _cell(name: str) -> int:
    return (int(name[1:]) - 1) * SIZE + ord(name[0]) - ord("a")


def _name(square: int) -> str:
    return f"{chr(ord('a') + square % SIZE)}{square // SIZE + 1}"


def _errors(spec: dict[str, Any]) -> list[str]:
    return validate_spec(spec)


def _with(spec: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """A deep copy with dotted paths set, `None` deleting the key."""
    out = copy.deepcopy(spec)
    for dotted, value in changes.items():
        *path, last = dotted.split("__")
        node = out
        for key in path:
            node = node[int(key)] if isinstance(node, list) else node.setdefault(key, {})
        if value is None:
            node.pop(last, None)
        else:
            node[last] = value
    return out


def _shaped_chess(chess: dict[str, Any], squares: list[str]) -> dict[str, Any]:
    spec = copy.deepcopy(chess)
    spec["schema_version"] = 14
    spec["rules"]["promotion"].pop("rank", None)
    spec["rules"]["promotion"]["squares"] = squares
    return spec


# ------------------------------------------------------------------------ the version


class TestTheVersionIsBumped:
    def test_the_schema_version_is_fourteen(self) -> None:
        assert SCHEMA_VERSION >= 14

    def test_the_collector_names_both_features(
        self, chess: dict[str, Any], showcase: dict[str, Any]
    ) -> None:
        assert _schema_fourteen_uses(chess) == set()
        assert _schema_fourteen_uses(showcase) == {"board.lattice=hex", "rules.promotion.squares"}

    def test_square_written_out_is_a_schema_14_key_and_the_same_game(
        self, chess: dict[str, Any]
    ) -> None:
        """A spelling, so one name since Q1008a (`spec/normal.py`)."""
        # chess's stamp before Q1008b re-spelt it at the current one
        spelled = _with(dict(chess, schema_version=1), board__lattice="square")
        assert _errors(spelled) == []
        assert spec_hash(spelled) == spec_hash(chess)

    def test_a_lattice_outside_the_table_is_refused(self, showcase: dict[str, Any]) -> None:
        assert LATTICES == ("square", "hex")
        errors = _errors(_with(showcase, board__lattice="triangle"))
        assert any("board.lattice must be one of" in e for e in errors), errors


# ------------------------------------------------------------------------ the far edge


class TestTheFarEdgeIsTheCellsWithNoForwardStep:
    """Pinned against the wall rule rather than against the author's list."""

    def test_the_showcase_is_a_square_grid_hexagon_plus_exactly_two_keys(
        self, showcase: dict[str, Any], square_grid: dict[str, Any]
    ) -> None:
        """Both keys are mechanics, and neither makes the rest of the spec legal."""
        zone = showcase["rules"]["promotion"]
        assert _errors(showcase) == []
        assert _errors(square_grid) == []

        lattice_only = _with(square_grid, schema_version=14, board__lattice="hex")
        zone_only = _with(square_grid, schema_version=14, rules__promotion=zone)
        assert _errors(lattice_only) == []
        assert _errors(zone_only) == []
        for arm in (square_grid, lattice_only, zone_only):
            assert spec_hash(arm) != spec_hash(showcase)
        assert spec_hash(_with(lattice_only, rules__promotion=zone)) == spec_hash(showcase)

    def test_every_zone_cell_has_no_forward_step_and_every_such_cell_is_in_the_zone(
        self, showcase: dict[str, Any]
    ) -> None:
        walls = obstacle_squares(showcase["rules"], SIZE, SIZE, "rotate")

        def on_board(q: int, r: int) -> bool:
            return 0 <= q < SIZE and 0 <= r < SIZE and r * SIZE + q not in walls

        stuck = {
            r * SIZE + q
            for q in range(SIZE)
            for r in range(SIZE)
            if on_board(q, r) and not on_board(q, r + 1)
        }
        compiled = compile_spec(showcase)
        assert compiled.promotions[0].squares[0] == stuck
        assert sorted(_name(s) for s in stuck) == sorted(FAR_EDGE)
        assert sorted(stuck) == [65, 75, 85, 95, 105, 110, 111, 112, 113, 114, 115]

    def test_the_second_player_promotes_on_the_first_players_home_edge(
        self, showcase: dict[str, Any]
    ) -> None:
        walls = obstacle_squares(showcase["rules"], SIZE, SIZE, "rotate")

        def on_board(q: int, r: int) -> bool:
            return 0 <= q < SIZE and 0 <= r < SIZE and r * SIZE + q not in walls

        home = {
            r * SIZE + q
            for q in range(SIZE)
            for r in range(SIZE)
            if on_board(q, r) and not on_board(q, r - 1)
        }
        assert compile_spec(showcase).promotions[0].squares[1] == home

    def test_the_rank_reflection_of_the_far_edge_is_five_walls(
        self, showcase: dict[str, Any]
    ) -> None:
        """Wrong spelling 4, concretely: a mirrored zone would sit partly off the board."""
        walls = obstacle_squares(showcase["rules"], SIZE, SIZE, "rotate")
        mirrored = {
            rank * SIZE + file
            for name in FAR_EDGE
            for file, rank in [reflect_square(ord(name[0]) - 97, int(name[1:]) - 1, SIZE, SIZE, "mirror")]
        }
        assert len(mirrored & walls) == 5


# ------------------------------------------------------------------- what is refused


class TestTheHexRefusals:
    """Each spelling Q7 found compiles a plausible wrong game on a hex lattice, by name."""

    @pytest.mark.parametrize("orientation", [None, "mirror"])
    def test_1_the_army_must_be_rotated(
        self, showcase: dict[str, Any], orientation: str | None
    ) -> None:
        spec = _with(showcase, setup__orientation=orientation)
        assert any('setup.orientation must be "rotate"' in e for e in _errors(spec))

    @pytest.mark.parametrize(
        ("mirror", "refused"), [(None, True), (True, True), ("rotate", False), (False, False)]
    )
    def test_2_a_goal_must_not_be_reflected_by_rank(
        self, showcase: dict[str, Any], mirror: Any, refused: bool
    ) -> None:
        reach: dict[str, Any] = {"piece": "K", "squares": ["f6"]}
        if mirror is not None:
            reach["mirror"] = mirror
        errors = _errors(_with(showcase, rules__reach=reach))
        assert any("rules.reach.mirror must be" in e for e in errors) is refused, errors

    @pytest.mark.parametrize("symmetry", ["lateral", "rotate4", "dihedral8"])
    def test_3_a_descriptor_must_list_its_vectors(
        self, showcase: dict[str, Any], symmetry: str
    ) -> None:
        spec = copy.deepcopy(showcase)
        spec["pieces"][3]["moves"][0]["symmetry"] = symmetry
        assert any(
            "pieces[3].moves[0].symmetry" in e and "hex" in e for e in _errors(spec)
        )

    def test_3_a_landing_must_list_its_vectors(self, showcase: dict[str, Any]) -> None:
        strike = {
            "effect": "demote", "position": "choose", "immune": ["K"],
            "chain": [{"from": "Q", "to": "R"}],
            "landing": {"vectors": [[1, 0]], "symmetry": "rotate4"},
        }
        assert any(
            "rules.strike.landing.symmetry" in e for e in _errors(_with(showcase, rules__strike=strike))
        )
        strike["landing"]["symmetry"] = "none"
        assert not any(
            "landing.symmetry" in e for e in _errors(_with(showcase, rules__strike=strike))
        )

    def test_4_a_facing_may_not_turn_by_quarters(self, showcase: dict[str, Any]) -> None:
        spec = copy.deepcopy(showcase)
        spec["pieces"][0]["facing"] = {"axes": 4}
        assert any("facing.axes 4" in e for e in _errors(spec))
        spec["pieces"][0]["facing"] = {"axes": 2}
        assert not any("facing.axes" in e for e in _errors(spec))

    @pytest.mark.parametrize(
        ("zone", "named"),
        [
            ({"rank": "last"}, "'rank'"),
            ({"ranks": [9, 10]}, "'ranks'"),
            ({}, "the default 'rank': 'last'"),
        ],
    )
    def test_5_a_zone_is_squares(
        self, showcase: dict[str, Any], zone: dict[str, Any], named: str
    ) -> None:
        promotion = {"piece": "P", "to": ["Q"], **zone}
        errors = _errors(_with(showcase, rules__promotion=promotion))
        assert any(named in e and "far edge is not a rank" in e for e in errors), errors

    def test_6_a_lame_leap_has_no_leg(self, showcase: dict[str, Any]) -> None:
        spec = copy.deepcopy(showcase)
        spec["pieces"][1]["moves"][0]["kind"] = "lame"
        assert any("'lame' is refused on a hex lattice" in e for e in _errors(spec))

    def test_the_same_spellings_pass_on_a_square_lattice(self, chess: dict[str, Any]) -> None:
        """The refusals are the lattice's, not the keys': chess writes three of them."""
        assert chess["setup"].get("orientation", "mirror") == "mirror"
        assert chess["rules"]["promotion"]["rank"] == "last"
        assert _errors(_with(chess, schema_version=14)) == []

    def test_en_passant_becomes_writable_on_a_hex_board(self, showcase: dict[str, Any]) -> None:
        assert _errors(_with(showcase, rules__en_passant=True)) == []


class TestTheZoneOnEitherLattice:
    def test_squares_beside_a_rank_is_refused(self, chess: dict[str, Any]) -> None:
        spec = _shaped_chess(chess, ["a8"])
        spec["rules"]["promotion"]["rank"] = "last"
        assert any("'squares' beside 'rank'" in e for e in _errors(spec))

    @pytest.mark.parametrize("squares", [[], None])
    def test_an_empty_zone_is_refused(self, chess: dict[str, Any], squares: Any) -> None:
        spec = _shaped_chess(chess, ["a8"])
        spec["rules"]["promotion"]["squares"] = squares
        assert any("promotion.squares must be a non-empty array" in e for e in _errors(spec))

    def test_a_square_off_the_board_is_refused(self, chess: dict[str, Any]) -> None:
        errors = _errors(_shaped_chess(chess, ["a8", "i8"]))
        assert any("promotion.squares entry 'i8'" in e for e in errors), errors

    def test_a_wall_in_the_zone_is_refused(self, showcase: dict[str, Any]) -> None:
        spec = copy.deepcopy(showcase)
        spec["rules"]["promotion"]["squares"] = [*FAR_EDGE, "k11"]
        errors = _errors(spec)
        assert any("rules.promotion.squares names an obstacle at k11" in e for e in errors), errors


# --------------------------------------------------------------- what it compiles to


class TestItCompiles:
    def test_a_rank_restated_as_squares_is_the_same_table(self, chess: dict[str, Any]) -> None:
        restated = _shaped_chess(chess, [f"{f}8" for f in "abcdefgh"])
        assert compile_spec(restated).promotions[0].squares == compile_spec(chess).promotions[0].squares
        assert build_tables(compile_spec(restated))["promo_squares"] == (
            build_tables(compile_spec(chess))["promo_squares"]
        )
        assert features(restated)["promotion_zone_share"] == features(chess)["promotion_zone_share"]

    def test_a_legan_shaped_zone_on_a_square_board_fills_the_entry_table(self) -> None:
        """The record's *Still out of reach* row, closed: a zone that is a corner."""
        spec = json.loads((ROOT / "rulesets" / "crazyhouse.json").read_text(encoding="utf-8"))
        spec["schema_version"] = 14
        corner = ["a5", "a6", "a7", "a8", "b8", "c8", "d8"]
        spec["rules"]["promotion"] = {"piece": "P", "squares": corner, "to": ["Q", "N"]}
        spec["rules"]["drops"].pop("region")
        spec["rules"]["promote_on_entry"] = True
        assert _errors(spec) == []
        compiled = compile_spec(spec)
        pawn, queen = compiled.piece_ids.index("P"), compiled.piece_ids.index("Q")
        first = {(int(n[1]) - 1) * 8 + ord(n[0]) - 97 for n in corner}
        second = {(7 - s // 8) * 8 + s % 8 for s in first}
        assert compiled.promotions[0].squares == (first, second)
        assert compiled.entry_promotion is not None
        for player, zone in ((0, first), (1, second)):
            row = compiled.entry_promotion.becomes[player][pawn]
            assert {s for s, t in enumerate(row) if t != -1} == zone
            assert {row[s] for s in zone} == {queen}

    def test_a_pawn_promotes_on_the_north_east_side(self, showcase: dict[str, Any]) -> None:
        """g9 to g10 promotes; under a rank-11 zone the pawn would stall against the wall."""
        engine = Engine(showcase)
        state = engine.make_state({"f1": ("K", 0), "f11": ("K", 1), "g9": ("P", 0)})
        promotions = [
            m for m in engine.legal_moves(state)
            if move_from(m) == _cell("g9") and move_type(m) == MOVE_PROMOTION
        ]
        assert {move_to(m) for m in promotions} == {_cell("g10")}
        assert len(promotions) == 4

    def test_the_second_player_promotes_on_the_first_players_home_edge(
        self, showcase: dict[str, Any]
    ) -> None:
        engine = Engine(showcase)
        state = engine.make_state(
            {"f1": ("K", 0), "f11": ("K", 1), "d4": ("P", 1)}, side_to_move=1
        )
        targets = {
            move_to(m) for m in engine.legal_moves(state)
            if move_from(m) == _cell("d4") and move_type(m) == MOVE_PROMOTION
        }
        assert targets == {_cell("d3")}

    @pytest.mark.parametrize("depth", sorted(LADDER))
    def test_the_ladder(self, showcase: dict[str, Any], depth: int) -> None:
        engine = Engine(showcase)
        assert engine.perft(engine.initial_state(), depth) == LADDER[depth]

    @pytest.mark.slow
    @pytest.mark.parametrize("depth", sorted(LADDER_DEEP))
    def test_the_deep_ladder(self, showcase: dict[str, Any], depth: int) -> None:
        engine = Engine(showcase)
        assert engine.perft(engine.initial_state(), depth) == LADDER_DEEP[depth]


# ------------------------------------------------------------- every surface reads it


class TestEverySurfaceReadsIt:
    def test_the_compiled_spec_carries_the_lattice(
        self, showcase: dict[str, Any], chess: dict[str, Any]
    ) -> None:
        assert compile_spec(showcase).lattice == "hex"
        assert compile_spec(chess).lattice == "square"

    def test_the_zone_crosses_to_the_boards_as_eleven_squares_a_side(
        self, showcase: dict[str, Any]
    ) -> None:
        compiled = compile_spec(showcase)
        first, second = zone_payload(compiled.promotions)  # type: ignore[misc]
        assert first == sorted(compiled.promotions[0].squares[0])
        assert len(first) == len(second) == 11

    def test_the_evaluator_reads_the_compiled_zone(self, showcase: dict[str, Any]) -> None:
        compiled = compile_spec(showcase)
        assert feature_context(compiled).promotion_squares == compiled.promotions[0].squares

    def test_the_prose_says_hexagonal_and_names_the_edge(self, showcase: dict[str, Any]) -> None:
        text = describe_spec(showcase)
        assert text.splitlines()[0] == (
            "hex glinski promoting - a two-player game on a hexagonal board of 91 cells, "
            "an 11 by 11 hex grid with 30 cut off; every cell touches six others."
        )
        assert f"P reaching {', '.join(FAR_EDGE)} becomes one of Q, R, B, N" in text

    def test_the_model_can_propose_both(self) -> None:
        schema = rule_spec_schema()
        board = schema["properties"]["board"]["properties"]
        promotion = schema["properties"]["rules"]["properties"]["promotion"]["anyOf"][0]
        assert board["lattice"]["enum"] == list(LATTICES)
        assert "squares" in promotion["properties"]

    def test_the_proposer_is_told_the_point_of_both(self) -> None:
        assert 'board.lattice: "hex"' in SCHEMA_PREAMBLE
        assert "promotion.squares" in SCHEMA_PREAMBLE

    def test_distance_separates_the_showcase_from_the_same_board_on_a_square_grid(
        self, showcase: dict[str, Any], square_grid: dict[str, Any]
    ) -> None:
        ours, theirs = features(showcase), features(square_grid)
        assert ours["hex_lattice"] == 1.0 and theirs["hex_lattice"] == 0.0
        assert ours["promotion_zone_share"] == pytest.approx(11 / 91)
        assert distance(showcase, square_grid) > 0

    def test_fairy_refuses_the_showcase_on_its_walls(self, showcase: dict[str, Any]) -> None:
        assert supports(showcase) is False

    def test_fairy_refuses_a_shaped_zone_by_name(self, chess: dict[str, Any]) -> None:
        """Verified against the installed engine on 2026-09-15: written as square names in
        `promotionRegion*`, a zone adding d6 and e6 to the far rank played `e4e3` there
        where this engine promotes, so the translation is refused rather than written."""
        shaped = _shaped_chess(chess, ["a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8", "d6", "e6"])
        with pytest.raises(Untranslatable, match="not whole ranks"):
            translate(shaped)
        assert supports(shaped) is False

    def test_fairy_writes_a_rank_restated_as_squares_as_chess_does(
        self, chess: dict[str, Any]
    ) -> None:
        restated = _shaped_chess(chess, [f"{f}8" for f in "abcdefgh"])

        def promotion_lines(spec: dict[str, Any]) -> list[str]:
            return [line for line in translate(spec).config.splitlines() if "promotion" in line]

        assert promotion_lines(restated) == promotion_lines(chess)


# ------------------------------------------------ Q922: the zone's edges and its share


class TestAPromotionRankIsOnTheBoardOrNothing:
    @pytest.mark.parametrize(
        ("stated", "ranks_hit"),
        [
            ([0], {0}),
            ([3], {3}),
            ([4], set()),
            ([-1], set()),
            ([0, 4, -1, 3], {0, 3}),
            ([True], set()),
            (["1"], set()),
        ],
        ids=str,
    )
    def test_ranks(self, stated: list, ranks_hit: set[int]) -> None:
        from chess_variant_sandbox.spec.schema import promotion_squares

        first, _ = promotion_squares({"ranks": stated}, 3, 4)
        assert first == {rank * 3 + file for rank in ranks_hit for file in range(3)}


class TestTheZoneShareIsOverTheCellsThatAreLeft:
    @staticmethod
    def _share(walls: list[str], zone: list[str], orientation: str | None = None) -> float:
        from chess_variant_sandbox.spec.distance import _promotion_zone_share

        spec: dict[str, Any] = {
            "board": {"width": 2, "height": 2},
            "rules": {
                "promotion": {"squares": zone, "to": ["N"]},
                "obstacles": {"squares": walls},
            },
        }
        if orientation:
            spec["setup"] = {"orientation": orientation}
        return _promotion_zone_share(spec)

    def test_one_cell_left_is_a_share_of_the_whole(self) -> None:
        assert self._share(["a1", "b1", "a2"], ["b2"]) == 1.0

    def test_no_cell_left_is_nothing_not_an_error(self) -> None:
        assert self._share(["a1", "b1", "a2", "b2"], ["b2"]) == 0.0

    def test_two_cells_left_halve_it(self) -> None:
        assert self._share(["a1", "b1"], ["b2"]) == 0.5

    def test_a_walled_zone_square_is_not_in_the_zone(self) -> None:
        """Q923: the zone is counted over the cells left, so a wall in it is out of both
        sides of the share. Three cells, one of the two zone squares walled: a third."""
        assert self._share(["a2"], ["a2", "b2"]) == pytest.approx(1 / 3)

    def test_the_share_never_passes_one(self) -> None:
        """Q923: one free cell and a two-square zone that is all wall read 2.0 before."""
        assert self._share(["a2", "b1", "b2"], ["a2", "b2"]) == 0.0
        for walls in ([], ["a1"], ["a2"], ["a1", "b2"], ["a2", "b2"], ["a1", "b1", "a2"]):
            assert 0.0 <= self._share(walls, ["a2", "b2"]) <= 1.0, walls

    def test_a_mirrored_wall_lands_where_the_setup_orientation_puts_it(self) -> None:
        """A wall on a 3x3 board's left edge, centre rank, is its own image under a flip and
        has a partner on the far edge under a half turn: one wall or two."""
        from chess_variant_sandbox.spec.distance import _promotion_zone_share

        spec = {
            "board": {"width": 3, "height": 3},
            "rules": {
                "promotion": {"squares": ["a3"], "to": ["N"]},
                "obstacles": {"squares": ["a2"], "mirror": True},
            },
        }
        assert _promotion_zone_share(spec) == 1 / 8
        spec["setup"] = {"orientation": "rotate"}
        assert _promotion_zone_share(spec) == 1 / 7


class TestARefusedBoardReportsNoSize:
    """`_validate_board` answers `(width, height, lattice)`, with zeros wherever it refused."""

    @staticmethod
    def _board(board: Any) -> tuple[tuple[int, int, str], list[str]]:
        from chess_variant_sandbox.spec.schema import _validate_board

        errors: list[str] = []
        return _validate_board(board, errors), errors

    def test_a_board_that_is_not_an_object(self) -> None:
        for board in (None, [8, 8], "8x8"):
            answer, errors = self._board(board)
            assert answer == (0, 0, "square") and len(errors) == 1

    @pytest.mark.parametrize(
        "board",
        [{"width": "8", "height": 8}, {"width": 8, "height": True}, {"width": 1, "height": 8},
         {"width": 8, "height": 17}, {"height": 8}],
        ids=str,
    )
    def test_a_size_that_is_refused_is_no_size(self, board: dict) -> None:
        answer, errors = self._board(board)
        assert answer == (0, 0, "square") and errors
        answer, _ = self._board({**board, "lattice": "hex"})
        assert answer == (0, 0, "hex")

    def test_a_good_size_is_returned_with_its_lattice(self) -> None:
        assert self._board({"width": 8, "height": 5}) == ((8, 5, "square"), [])
        assert self._board({"width": 2, "height": 16, "lattice": "hex"}) == ((2, 16, "hex"), [])

    def test_an_unknown_lattice_is_the_square_one(self) -> None:
        answer, errors = self._board({"width": 8, "height": 8, "lattice": "triangle"})
        assert answer == (8, 8, "square") and len(errors) == 1
