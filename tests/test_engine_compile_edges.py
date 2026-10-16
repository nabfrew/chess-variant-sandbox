"""Engine mutation report, F-engine-1: the compiler's defaults, edges and overlaps.

The `compile.py` mutants whose killing test belongs beside `tests/test_engine.py`
(`docs/reference/mutation/engine.md`): a default left out of a spec, a hop that cannot
capture, a condition that does not apply here, two moves to one square, a seam only one player
may cross. Each test names the input the mutant could not survive. Every spec is a small
purpose-made dictionary or a committed ruleset with one key changed; none asserts a number
that only chess has.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import (
    MODE_CAPTURE,
    MODE_MOVE,
    PromotionRule,
    _needs_runtime_dedup,
    compile_spec,
)
from chess_variant_sandbox.spec import validate_spec

ROOT = Path(__file__).resolve().parents[1]
KING = {"id": "K", "name": "king", "value": 0, "royal": True,
        "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]}


def _load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _probe(
    pieces: list[dict[str, Any]],
    rules: dict[str, Any] | None = None,
    placements: list[dict[str, Any]] | None = None,
    width: int = 5,
    height: int = 5,
) -> dict[str, Any]:
    """Two kings and ``pieces`` on a bare board; ``placements`` replaces the kings' squares."""
    spec = {
        "schema_version": 26,
        "name": "probe",
        "board": {"width": width, "height": height},
        "pieces": [KING, *pieces],
        "setup": {"mirror": False, "placements": placements or [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": [f"{chr(96 + width)}{height}"]},
        ]},
        "rules": {"win": "royal_capture", **(rules or {})},
    }
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def _texts(engine: Engine, state: Any, piece: str) -> list[str]:
    return sorted(t for t in engine.move_texts(state) if t.startswith(piece))


# ------------------------------------------------------------------ the compiled record


class TestTheCompiledRecordReadsTheSpecAsGiven:
    def test_the_single_promotion_rule_is_the_promotion_and_two_rules_are_none(self) -> None:
        chess = compile_spec(_load("rulesets/chess.json"))
        assert isinstance(chess.promotion, PromotionRule)
        assert chess.promotion is chess.promotions[0]
        assert compile_spec(_load("rulesets/dull.json")).promotion is None

    @pytest.mark.fresh_compile  # the session's memo is keyed on sorted keys and would hide it
    def test_the_source_does_not_depend_on_the_order_the_keys_were_written_in(self) -> None:
        spec = _load("rulesets/chess.json")
        reordered = dict(reversed(list(spec.items())))
        reordered["rules"] = dict(reversed(list(spec["rules"].items())))
        assert list(reordered) != list(spec)
        assert compile_spec(reordered).source == compile_spec(spec).source

    def test_a_spec_that_states_no_clocks_gets_the_documented_ones(self) -> None:
        spec = _load("rulesets/chess.json")
        del spec["rules"]["max_plies"]
        del spec["rules"]["no_progress_draw"]
        compiled = compile_spec(spec)
        assert (compiled.max_plies, compiled.no_progress_draw) == (400, 100)

    def test_a_castling_that_omits_king_steps_moves_the_king_two_squares(self) -> None:
        spec = _load("rulesets/chess.json")
        del spec["rules"]["castling"]["king_steps"]
        castling = compile_spec(spec).castling
        assert castling is not None and castling.king_steps == 2
        engine = Engine(spec)
        state = engine.make_state({"e1": ("K", 0), "h1": ("R", 0), "e8": ("K", 1)})
        assert "Ke1-g1 (castle)" in engine.move_texts(state)

    def test_an_explosion_that_omits_its_radius_blasts_the_eight_neighbours(self) -> None:
        spec = _load("rulesets/atomic.json")
        del spec["rules"]["explode_on_capture"]["radius"]
        stated = compile_spec(_load("rulesets/atomic.json"))
        omitted = compile_spec(spec)
        assert omitted.explosion is not None and stated.explosion is not None
        d4 = 3 * 8 + 3
        assert set(omitted.explosion.blast[d4]) == {
            d4 + dx + 8 * dy for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        } - {d4}
        assert omitted.explosion.blast == stated.explosion.blast


# ---------------------------------------------------------------- what attacks a square


class TestWhatAttacksASquare:
    HOP_MOVE = {"id": "H", "name": "hopper", "value": 1,
                "moves": [{"kind": "hop", "vectors": [[1, 0]], "mode": "move"}]}
    HOP_FAR = {"id": "H", "name": "hopper", "value": 1,
               "moves": [{"kind": "hop", "vectors": [[1, 0]], "mode": "capture",
                          "min_range": 2}]}
    STEPPER = {"id": "S", "name": "stepper", "value": 1,
               "moves": [{"kind": "leap", "vectors": [[0, 1]]}]}

    def test_a_hop_that_only_moves_attacks_nothing(self) -> None:
        engine = Engine(_probe([self.HOP_MOVE, self.STEPPER]))
        assert engine.spec.has_hop_attacks is False
        # Black's king stands where the hop would land, past the screen on b3; black's
        # stepper can still move, which it could not if the king were in check.
        state = engine.make_state(
            {"a1": ("K", 0), "c3": ("K", 1), "a3": ("H", 0), "b3": ("S", 0), "e4": ("S", 1)},
            side_to_move=1,
        )
        assert not engine.in_check(state, 1)
        assert _texts(engine, state, "S") == ["Se4-e3"]

    def test_a_capturing_hop_whose_near_squares_cannot_capture_still_attacks_the_far_ones(
        self,
    ) -> None:
        engine = Engine(_probe([self.HOP_FAR, self.STEPPER]))
        assert engine.spec.has_hop_attacks is True
        state = engine.make_state(
            {"a1": ("K", 0), "c3": ("K", 1), "a3": ("H", 0), "b3": ("S", 0)}, side_to_move=1
        )
        assert engine.in_check(state, 1)

    def test_a_rook_gives_check_in_an_explosion_game_whose_king_is_declared_first(self) -> None:
        spec = _load("rulesets/atomic.json")
        by_id = {piece["id"]: piece for piece in spec["pieces"]}
        spec["pieces"] = [by_id["K"], *(by_id[i] for i in by_id if i != "K")]
        engine = Engine(spec)
        state = engine.make_state(
            {"a1": ("K", 0), "e8": ("K", 1), "e2": ("R", 0)}, side_to_move=1
        )
        assert engine.in_check(state, 1)


# --------------------------------------------------------------------- movement tables


class TestMovementTables:
    def test_a_move_after_a_region_condition_is_kept_outside_that_region(self) -> None:
        piece = {"id": "M", "name": "mover", "value": 1, "moves": [
            {"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4",
             "condition": {"region": ["a1", "b1"]}},
            {"kind": "leap", "vectors": [[1, 1]], "symmetry": "rotate4"},
        ]}
        engine = Engine(_probe([piece]))
        outside = engine.make_state({"a1": ("K", 0), "e5": ("K", 1), "c3": ("M", 0)})
        assert _texts(engine, outside, "M") == ["Mc3-b2", "Mc3-b4", "Mc3-d2", "Mc3-d4"]
        inside = engine.make_state({"a1": ("K", 0), "e5": ("K", 1), "b1": ("M", 0)})
        assert _texts(engine, inside, "M") == ["Mb1-a2", "Mb1-b2", "Mb1-c1", "Mb1-c2"]

    def test_a_slide_keeps_its_quiet_move_where_a_capture_only_leap_lands(self) -> None:
        piece: dict[str, Any] = {"id": "X", "name": "both", "value": 1, "moves": [
            {"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "capture"},
            {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both"},
        ]}
        engine = Engine(_probe([piece, {**piece, "id": "Y", "moves": piece["moves"][:1]}]))
        empty = engine.make_state({"a1": ("K", 0), "e5": ("K", 1), "c3": ("X", 0)})
        assert "Xc3-d3" in engine.move_texts(empty)
        taken = engine.make_state(
            {"a1": ("K", 0), "e5": ("K", 1), "c3": ("X", 0), "d3": ("Y", 1)}
        )
        # One capture of d3, not two, and no quiet move onto it.
        assert [t for t in engine.move_texts(taken) if t.endswith("d3")] == ["Xc3xd3"]

    def test_a_capture_only_group_and_a_later_move_only_group_share_no_move(self) -> None:
        leaps = (((5,), (MODE_CAPTURE,), False),)
        slides = (((5,), (MODE_MOVE,), False),)
        assert _needs_runtime_dedup(leaps, slides, ()) is False
        assert _needs_runtime_dedup(leaps, (((5,), (MODE_CAPTURE,), False),), ()) is True

    def test_the_double_step_window_is_found_when_the_first_square_of_the_ray_is_skipped(
        self,
    ) -> None:
        # range 4 from min_range 2: the first square is on the ray but is not a landing.
        engine = Engine(self._pawn_game([
            {"kind": "slide", "vectors": [[0, 1]], "range": 4, "min_range": 2, "mode": "move",
             "condition": {"from_rank": 1}},
            {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
        ]))
        assert engine.spec.ep_passes[0][8] == ((24, 16),)
        assert engine.spec.ep_passes[1][48] == ((32, 40),)

    def test_a_capture_only_diagonal_ray_opens_no_window(self) -> None:
        engine = Engine(self._pawn_game([
            {"kind": "slide", "vectors": [[0, 1]], "range": 2, "mode": "move"},
            {"kind": "slide", "vectors": [[1, 1]], "range": 2, "symmetry": "lateral",
             "mode": "capture"},
        ]))
        # b2: the forward ray only. The diagonals' second squares, d4 and... are not windows.
        assert engine.spec.ep_passes[0][8] == ((24, 16),)
        assert engine.spec.ep_passes[0][12] == ((28, 20),)

    @staticmethod
    def _pawn_game(moves: list[dict[str, Any]]) -> dict[str, Any]:
        spec = _load("rulesets/chess.json")
        spec["schema_version"] = 26
        del spec["rules"]["castling"]
        next(p for p in spec["pieces"] if p["id"] == "P")["moves"] = moves
        assert validate_spec(spec) == [], validate_spec(spec)
        return spec


# ----------------------------------------------------------------------- one-way pieces


def _one_way(spec: dict[str, Any]) -> set[str]:
    compiled = compile_spec(spec)
    return {compiled.piece_ids[index] for index in compiled.one_way_types}


class TestWhichPiecesCannotComeBack:
    DIAGONAL = {"id": "D", "name": "diagonal", "value": 1,
                "moves": [{"kind": "leap", "vectors": [[1, 1]]}]}
    PROMOTED = {"id": "Q", "name": "promoted", "value": 9,
                "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]}
    GRUNT = {"id": "G", "name": "grunt", "value": 1,
             "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1}]}

    @staticmethod
    def _wrapped_3x3(zone: list[str]) -> dict[str, Any]:
        """A diagonal leaper on a torus whose every diagonal cycle enters ``zone``."""
        return _probe(
            [TestWhichPiecesCannotComeBack.DIAGONAL, TestWhichPiecesCannotComeBack.PROMOTED],
            {"wrap": {"files": "both", "ranks": "both"},
             "promotion": {"piece": "D", "to": ["Q"], "squares": zone}},
            [
                {"piece": "K", "player": 0, "squares": ["c1"]},
                {"piece": "K", "player": 1, "squares": ["a3"]},
                {"piece": "D", "player": 0, "squares": ["c2"]},
                {"piece": "D", "player": 1, "squares": ["b2"]},
            ],
            3, 3,
        )

    @pytest.mark.parametrize("zone", [["a1", "b1", "a2"], ["a1", "c2", "b3"]])
    def test_each_player_is_judged_by_the_promotion_squares_that_are_hers(
        self, zone: list[str]
    ) -> None:
        # Every diagonal cycle of either army passes through that army's own zone, and
        # through the other's only by chance: reading one army's zone for the other loses it.
        assert _one_way(self._wrapped_3x3(zone)) == {"D"}

    @pytest.mark.parametrize("crossing", [{"0": "both", "1": "none"}, {"0": "none", "1": "both"}])
    def test_a_rank_seam_only_one_player_may_cross_still_takes_the_grunt_home(
        self, crossing: dict[str, str]
    ) -> None:
        spec = _probe(
            [self.GRUNT],
            {"wrap": {"ranks_by_player": crossing}},
            [
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 1, "squares": ["e5"]},
                {"piece": "G", "player": 0, "squares": ["c2"]},
                {"piece": "G", "player": 1, "squares": ["c4"]},
            ],
        )
        assert _one_way(spec) == set()
        # And with the seam shut for both, the same grunt cannot come back.
        spec["rules"]["wrap"] = {"ranks_by_player": {"0": "none", "1": "none"}}
        assert _one_way(spec) == {"G"}

    def test_a_sideways_sweep_beside_a_forward_step_can_come_back(self) -> None:
        sweeper = {"id": "W", "name": "sweeper", "value": 1, "moves": [
            {"kind": "slide", "vectors": [[0, 1]], "range": 1},
            {"kind": "sweep", "vectors": [[1, 0]], "symmetry": "lateral"},
        ]}
        placements = [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["e5"]},
            {"piece": "W", "player": 0, "squares": ["c2"]},
            {"piece": "W", "player": 1, "squares": ["c4"]},
        ]
        assert _one_way(_probe([sweeper], placements=placements)) == set()
        forward = {**sweeper, "moves": [{"kind": "sweep", "vectors": [[0, 1]]}]}
        assert _one_way(_probe([forward], placements=placements)) == {"W"}
