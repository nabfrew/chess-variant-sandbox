"""Engine mutation report, F-engine-1: move text, landing and the edges of the board.

Each test here was written to fail on a mutant of ``engine/engine.py`` or ``engine/state.py``
that the suite let survive (``docs/reference/mutation/engine.md``, *Proposed fix tasks*), and
``engine.md`` names the test beside each mutant it kills. Every spec is a chess-derived
dictionary with one rule changed, because the rule is the subject: the strike families
(stay, choose, home), a fold or a portal standing on a1, a long turn, a simultaneous round.
No test here is about chess, and none asserts a number from it.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.agents.base import native_spec
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import (
    MOVE_PORTAL,
    decode_move,
    move_from,
    move_to,
)
from chess_variant_sandbox.engine.state import GameState, Outcome, PhaseStep
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION
from chess_variant_sandbox.tournament.play import eval_values

ROOT = Path(__file__).resolve().parents[1]
STEP = {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
KINGS = {"e1": ("K", 0), "e8": ("K", 1)}


def _load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _chess_with(
    rules: dict[str, Any] | None = None,
    *,
    order: list[str] | None = None,
    drop: tuple[str, ...] = (),
    unplaced: tuple[str, ...] = (),
) -> Engine:
    """The calibration ruleset's pieces and setup with ``rules`` laid over its rules.

    ``drop`` removes rules it carries, ``order`` redeclares the pieces (the type index is
    the declaration order, so ``order`` moves which piece is type 0 and which is last) and
    ``unplaced`` leaves a piece out of the setup while keeping it declared.
    """
    spec = _load("rulesets/chess.json")
    spec["schema_version"] = SCHEMA_VERSION
    for key in drop:
        spec["rules"].pop(key, None)
    spec["rules"].update(rules or {})
    if order is not None:
        by_id = {piece["id"]: piece for piece in spec["pieces"]}
        assert sorted(order) == sorted(by_id)
        spec["pieces"] = [by_id[piece_id] for piece_id in order]
    if unplaced:
        spec["setup"]["placements"] = [
            p for p in spec["setup"]["placements"] if p["piece"] not in unplaced
        ]
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _strike(effect: str, position: str, chain: list[tuple[str, str]], **more: Any) -> dict:
    return {
        "strike": {
            "effect": effect,
            "position": position,
            "chain": [{"from": a, "to": b} for a, b in chain],
            "immune": ["K"],
            **more,
        }
    }


def _circe(immune: list[str] | None = None) -> dict[str, Any]:
    return {"strike": {"effect": "capture", "position": "home", "immune": immune or ["K"]}}


def _square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + ord(name[0]) - ord("a")


def _name(square: int, width: int = 8) -> str:
    return f"{chr(ord('a') + square % width)}{square // width + 1}"


def _move(engine: Engine, state: GameState, origin: str, target: str) -> int:
    width = engine.spec.width
    (move,) = [
        m
        for m in engine.legal_moves(state)
        if (move_from(m), move_to(m)) == (_square(origin, width), _square(target, width))
    ]
    return move


def _texts_of_captures(engine: Engine, state: GameState) -> list[str]:
    return [text for text in engine.move_texts(state) if "x" in text]


def _placement_of(engine: Engine, state: GameState) -> dict[str, tuple[str, int]]:
    """The board of ``state`` as ``make_state`` takes it."""
    spec = engine.spec
    out: dict[str, tuple[str, int]] = {}
    for square, value in enumerate(state.board):
        if value:
            out[_name(square, spec.width)] = (
                spec.piece_ids[spec.type_of(value)],
                spec.owner_of(value),
            )
    return out


# ------------------------------------------------------------------ a strike, in words


class TestAStrikeIsWrittenAsWhatItDid:
    """``describe_move``'s strike branch, read through exact texts."""

    FALL = [("N", "P")]  # a knight struck is demoted to P, the first declared piece (type 0)

    def test_a_victim_demoted_to_the_first_declared_type_is_named_and_stands_fast(self) -> None:
        engine = _chess_with(_strike("demote", "stay", self.FALL), drop=("castling",))
        state = engine.make_state({**KINGS, "d4": ("R", 0), "d7": ("N", 1)})
        assert engine.describe_move(state, _move(engine, state, "d4", "d7")) == (
            "Rd4xd7->P (stands fast)"
        )

    def test_a_victim_the_chain_leaves_alone_is_a_plain_capture_under_stay(self) -> None:
        engine = _chess_with(_strike("demote", "stay", self.FALL), drop=("castling",))
        state = engine.make_state({**KINGS, "d4": ("R", 0), "d7": ("P", 1)})
        assert engine.describe_move(state, _move(engine, state, "d4", "d7")) == "Rd4xd7"

    def test_a_contact_that_turns_the_attacker_into_the_first_declared_type_says_so(self) -> None:
        # No `onto_to`: the victim is simply taken, the knight becomes a pawn (type 0).
        engine = _chess_with(
            {
                "contacts": {
                    "position": "advance",
                    "immune": ["K"],
                    "links": [{"mover": "N", "onto": "P", "mover_to": "P"}],
                }
            },
            drop=("castling",),
        )
        state = engine.make_state({**KINGS, "c3": ("N", 0), "d5": ("P", 1)})
        assert engine.describe_move(state, _move(engine, state, "c3", "d5")) == "Nc3xd5/N->P"

    def test_a_contact_that_changes_the_attacker_overrides_the_promotion_word(self) -> None:
        engine = _chess_with(
            {
                "contacts": {
                    "position": "advance",
                    "immune": ["K"],
                    "links": [{"mover": "P", "onto": "N", "mover_to": "R"}],
                }
            },
            drop=("castling", "en_passant"),
        )
        state = engine.make_state({"e1": ("K", 0), "a8": ("K", 1), "e7": ("P", 0), "d8": ("N", 1)})
        # One move per promotion target, all written alike: the "=Q" would be a false word.
        assert set(_texts_of_captures(engine, state)) == {"Pe7xd8/P->R"}

    def test_a_rook_taken_under_stay_names_the_unchanged_attacker_only_by_standing_fast(
        self,
    ) -> None:
        # The attacker's own column is empty, so nothing may be appended after the victim.
        engine = _chess_with(_strike("demote", "stay", [("R", "B")]), drop=("castling",))
        state = engine.make_state({**KINGS, "d4": ("Q", 0), "d7": ("R", 1)})
        assert engine.describe_move(state, _move(engine, state, "d4", "d7")) == (
            "Qd4xd7->B (stands fast)"
        )

    def test_a_king_castling_onto_its_own_rook_square_is_written_as_a_castle_only(self) -> None:
        # A strike ruleset has no friendly links, so `own` reads empty tables.
        engine = _chess_with(
            {
                "castling": {
                    "king": "K",
                    "rook": "R",
                    "king_file": {"kingside": "g", "queenside": "c"},
                    "rook_file": {"kingside": "f", "queenside": "d"},
                },
                "strike": {"effect": "capture", "position": "stay", "immune": ["K"]},
            },
        )
        state = engine.make_state({"f1": ("K", 0), "g1": ("R", 0), "e8": ("K", 1)})
        assert [t for t in engine.move_texts(state) if "castle" in t] == ["Kf1-g1 (castle)"]


# ----------------------------------------------------------- where a struck piece lands


class TestAChosenLandingIsNamedAndFound:
    LAND = {"landing": STEP}

    def _demote_knight_to_pawn(self) -> Engine:
        return _chess_with(
            _strike("demote", "choose", [("N", "P")], **self.LAND),
            drop=("castling", "en_passant", "promotion"),
        )

    def test_a_victim_demoted_to_the_first_declared_type_is_placed_and_named(self) -> None:
        engine = self._demote_knight_to_pawn()
        state = engine.make_state({**KINGS, "d4": ("R", 0), "d7": ("N", 1)})
        takes = [m for m in engine.legal_moves(state) if move_from(m) == _square("d4")
                 and move_to(m) == _square("d7")]
        assert len(takes) == 7  # one move per free square beside d7
        for move in takes:
            square = engine.landing_square(state, move)
            assert square is not None
            assert engine.describe_move(state, move) == f"Rd4xd7->P (placed on {_name(square)})"

    def test_a_victim_the_chain_does_not_keep_alive_has_no_landing_and_no_placement_word(
        self,
    ) -> None:
        engine = self._demote_knight_to_pawn()
        state = engine.make_state({**KINGS, "d4": ("R", 0), "d7": ("Q", 1)})
        move = _move(engine, state, "d4", "d7")
        assert engine.landing_square(state, move) is None
        assert engine.describe_move(state, move) == "Rd4xd7"

    def test_a_quiet_move_has_no_landing_even_when_the_last_declared_piece_is_linked(
        self,
    ) -> None:
        # An empty square reads as type -1, which wraps to the last declared piece: here Q,
        # which the chain demotes, so a landing would be found if the emptiness were ignored.
        engine = _chess_with(
            _strike("demote", "choose", [("Q", "R")], **self.LAND),
            order=["P", "N", "B", "R", "K", "Q"],
            drop=("castling", "en_passant", "promotion"),
        )
        state = engine.make_state({**KINGS, "d4": ("R", 0)})
        quiet = _move(engine, state, "d4", "d5")
        assert engine.landing_square(state, quiet) is None
        assert engine.describe_move(state, quiet) == "Rd4-d5"

    def test_a_choose_en_passant_names_a_pawn_that_becomes_the_first_declared_type(
        self,
    ) -> None:
        # P is declared second, so its row is not the first: the victim's and the mover's
        # types are different numbers, and the text is built from the table cell of the pair.
        engine = _chess_with(
            _strike("promote", "choose", [("P", "N")], **self.LAND),
            order=["N", "P", "B", "R", "Q", "K"],
            drop=("castling",),
        )
        state = engine.make_state(
            {**KINGS, "e5": ("P", 0), "d5": ("P", 1)}, ep_target="d6", ep_victim="d5"
        )
        passing = [t for t in engine.move_texts(state) if t.endswith(" e.p.")]
        assert passing, engine.move_texts(state)
        for text in passing:
            assert text.startswith("Pe5xd6->N (placed on ") and text.endswith(") e.p."), text

    def test_a_choose_en_passant_that_the_chain_does_not_keep_is_a_plain_capture(self) -> None:
        engine = _chess_with(
            _strike("demote", "choose", [("N", "B")], **self.LAND), drop=("castling",)
        )
        state = engine.make_state(
            {**KINGS, "e5": ("P", 0), "d5": ("P", 1)}, ep_target="d6", ep_victim="d5"
        )
        assert [t for t in engine.move_texts(state) if "e.p." in t] == ["Pe5xd6 e.p."]


# ----------------------------------------------------------------------------- circe


class TestARebirthIsWrittenWhereItHappened:
    def test_a_knight_is_reborn_on_one_of_its_two_homes(self) -> None:
        engine = _chess_with(_circe(), drop=("castling",))
        state = engine.make_state({**KINGS, "b5": ("B", 0), "c6": ("N", 1)})
        # Black's knights start on b8 and g8; the nearer free one is chosen by the compiler.
        (text,) = _texts_of_captures(engine, state)
        assert text in {"Bb5xc6 (reborn b8)", "Bb5xc6 (reborn g8)"}
        assert text == "Bb5xc6 (reborn g8)"

    def test_a_piece_the_rule_leaves_dead_is_a_plain_capture(self) -> None:
        engine = _chess_with(_circe(["K", "N"]), drop=("castling",))
        state = engine.make_state({**KINGS, "b5": ("B", 0), "c6": ("N", 1)})
        assert _texts_of_captures(engine, state) == ["Bb5xc6"]

    def test_a_rook_taken_on_its_home_file_is_reborn_by_the_squares_colour(self) -> None:
        # Q963, orthodox Circe: a4 is light, so the rook goes to h1 (light), not to a1 on
        # its file; until then the file came first and this read "reborn a1".
        engine = _chess_with(_circe(), drop=("castling",))
        state = engine.make_state(
            {**KINGS, "a4": ("R", 0), "a8": ("R", 1)}, side_to_move=1
        )
        assert "Ra8xa4 (reborn h1)" in engine.move_texts(state)
        state = engine.make_state(
            {**KINGS, "a5": ("R", 0), "a8": ("R", 1)}, side_to_move=1
        )
        assert "Ra8xa5 (reborn a1)" in engine.move_texts(state)

    def test_a_piece_with_no_home_is_removed_whatever_stands_on_the_last_square(self) -> None:
        # No bishop is placed by the setup, so none has a home; another black bishop stands
        # on h8, the last square, which is where a missing home would be read from.
        engine = _chess_with(_circe(), drop=("castling",), unplaced=("B",))
        state = engine.make_state(
            {**KINGS, "c1": ("R", 0), "c6": ("B", 1), "h8": ("B", 1)}
        )
        assert _texts_of_captures(engine, state) == ["Rc1xc6 (removed: no free home)"]

    def test_a_home_held_by_an_enemy_piece_of_the_same_type_is_not_free(self) -> None:
        engine = _chess_with(_circe(), drop=("castling",))
        state = engine.make_state(
            {**KINGS, "e4": ("P", 0), "d5": ("P", 1), "d7": ("P", 0)}
        )
        assert "Pe4xd5 (removed: no free home)" in engine.move_texts(state)

    def test_a_capture_with_no_en_passant_square_still_finds_the_victims_home(self) -> None:
        engine = _chess_with(_circe(), drop=("castling",))
        state = engine.make_state({**KINGS, "e4": ("P", 0), "d5": ("P", 1)})
        assert "Pe4xd5 (reborn d7)" in engine.move_texts(state)

    def test_an_en_passant_capture_is_reborn_from_the_square_of_the_passed_pawn(self) -> None:
        engine = _chess_with(_circe(), drop=("castling",))
        state = engine.make_state(
            {**KINGS, "e5": ("P", 0), "d5": ("P", 1)}, ep_target="d6", ep_victim="d5"
        )
        assert _texts_of_captures(engine, state) == ["Pe5xd6 (reborn d7) e.p."]


# ------------------------------------------------------------- folds and portals on a1


def _stripped(spec: dict[str, Any], placements: list[tuple[str, int, str]]) -> dict[str, Any]:
    spec = copy.deepcopy(spec)
    for key in ("promotion", "castling", "en_passant"):
        spec["rules"].pop(key, None)
    spec["setup"] = {
        "mirror": False,
        "placements": [{"piece": p, "player": who, "squares": [at]} for p, who, at in placements],
    }
    return spec


def _fold_on_a1() -> Engine:
    """``archive/crease.json`` with its first fold end moved onto a1 (square 0).

    Ends in order: orange a1, blue i5, orange i4, blue b5; the fold squares are
    ``(0, 48, 38, 41)`` and the fold ``a1`` leads to ``i5``.
    """
    spec = _stripped(
        _load("archive/crease.json"),
        [("K", 0, "f1"), ("K", 1, "f8"), ("R", 0, "a4")],
    )
    spec["rules"]["portals"][0]["orange"] = "a1"
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _portal_on_a1() -> tuple[Engine, GameState]:
    """``archive/wormhole.json`` with the orange end on a1 and a bishop that can enter it."""
    spec = _stripped(
        _load("archive/wormhole.json"),
        [("K", 0, "e1"), ("K", 1, "e5"), ("B", 0, "b2")],
    )
    spec["rules"]["portals"][0]["orange"] = "a1"
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    return engine, engine.initial_state()


class TestAFoldOnTheFirstSquare:
    def test_the_folds_stand_where_they_start_in_a_hand_built_position(self) -> None:
        engine = _fold_on_a1()
        built = engine.make_state({"f1": ("K", 0), "f8": ("K", 1), "a4": ("R", 0)})
        assert built.folds == (0, 48, 38, 41)
        assert built.folds == engine.initial_state().folds

    def test_fold_square_answers_each_end_with_its_own_start(self) -> None:
        engine = _fold_on_a1()
        for state in (engine.initial_state(),
                      engine.make_state({"f1": ("K", 0), "f8": ("K", 1), "a4": ("R", 0)})):
            assert [engine.fold_square(state, end) for end in (0, 1)] == [0, 48]

    def test_a_move_through_the_fold_on_a1_is_routed_in_at_a1_and_out_at_i5(self) -> None:
        engine = _fold_on_a1()
        state = engine.initial_state()
        through = _move(engine, state, "a4", "i5")
        assert engine.fold_route(state, through) == (0, 48)
        assert engine.describe_move(state, through) == "Ra4-a1~i5"

    def test_a_game_without_folds_has_no_fold_square(self) -> None:
        chess = _chess_with()
        assert chess.fold_square(chess.initial_state(), 0) is None

    def test_a_portal_game_without_folds_has_no_fold_square(self) -> None:
        engine, state = _portal_on_a1()
        assert engine.fold_square(state, 0) is None


class TestAPortalOnTheFirstSquare:
    def test_the_entry_of_a_transit_is_a1_and_the_text_names_it(self) -> None:
        engine, state = _portal_on_a1()
        (transit,) = [
            m for m in engine.legal_moves(state) if decode_move(m)[2] == MOVE_PORTAL
        ]
        assert engine.portal_entry(state, transit) == 0
        assert engine.describe_move(state, transit) == "Bb2-a1>c2"

    def test_a_position_that_has_no_such_end_gives_no_entry_and_a_question_mark(self) -> None:
        engine, state = _portal_on_a1()
        (transit,) = [
            m for m in engine.legal_moves(state) if decode_move(m)[2] == MOVE_PORTAL
        ]
        # `make_state` places only what it is told: no portal end stands anywhere on it.
        bare = engine.make_state({"e1": ("K", 0), "e5": ("K", 1), "b2": ("B", 0)})
        assert engine.portal_entry(bare, transit) is None
        assert engine.describe_move(bare, transit) == "Bb2-?>c2"


# ------------------------------------------------------------------- make_state itself


def _probe_16() -> Engine:
    """A 16x16 board: two kings and a rook, the black king on the last square (p16)."""
    spec = {
        "schema_version": 1,
        "name": "wide",
        "board": {"width": 16, "height": 16},
        "pieces": [
            {"id": "K", "name": "king", "value": 0, "royal": True,
             "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]],
                        "symmetry": "rotate4", "mode": "both"}]},
            {"id": "R", "name": "rook", "value": 5,
             "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4",
                        "mode": "both"}]},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "R", "player": 0, "squares": ["b1"]},
            {"piece": "K", "player": 1, "squares": ["p16"]},
        ]},
        "rules": {"win": "royal_capture"},
    }
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def test_a_piece_on_the_last_square_of_the_largest_board_crosses_the_boundary() -> None:
    engine = _probe_16()
    state = engine.initial_state()
    assert state.unmoved >> 255 & 1
    # 256 bits cross as 32 bytes; 31 hold only 248 and the packing overflows.
    assert engine.legal_moves(state)
    assert engine.apply(state, engine.legal_moves(state)[0]).ply == 1


OPENINGS = [
    "rulesets/chess.json",
    "rulesets/racingkings.json",
    "archive/jeson-mor.json",
    "tests/_simultaneous_gardner.json",
    "tests/_simultaneous_merge.json",
]


class TestAHandBuiltOpeningIsTheCoresOwn:
    @pytest.mark.parametrize("path", OPENINGS)
    def test_make_state_of_the_opening_placement_equals_initial_state(self, path: str) -> None:
        engine = Engine(_load(path))
        opening = engine.initial_state()
        built = engine.make_state(_placement_of(engine, opening))
        assert built == opening  # halfmove, ply, initiative, goal_left_by, void_run, key...

    def test_the_simultaneous_tie_rule_that_keeps_the_initiative_starts_it_where_the_core_does(
        self,
    ) -> None:
        engine = Engine(_load("tests/_simultaneous_merge.json"))
        built = engine.make_state(_placement_of(engine, engine.initial_state()))
        assert built.initiative == engine.initial_state().initiative
        assert built.initiative in (0, 1)
        assert built.key == engine.initial_state().key

    def test_a_round_game_without_a_void_run_has_the_key_the_core_gives_it(self) -> None:
        engine = Engine(_load("tests/_simultaneous_gardner.json"))
        built = engine.make_state(_placement_of(engine, engine.initial_state()))
        assert built.void_run == (-1, -1)
        assert built.key == engine.initial_state().key

    def test_a_goal_left_by_nobody_is_not_a_win_for_anyone_in_a_leave_game(self) -> None:
        engine = Engine(_load("archive/jeson-mor.json"))
        built = engine.make_state(_placement_of(engine, engine.initial_state()))
        assert built.goal_left_by == -1
        assert engine.terminal(built) is None

    def test_unmoved_holds_exactly_the_placed_squares_when_a1_is_empty(self) -> None:
        engine = _chess_with()
        built = engine.make_state({"e1": ("K", 0), "h1": ("R", 0), "e8": ("K", 1)})
        assert built.unmoved == (1 << _square("e1")) | (1 << _square("h1")) | (1 << _square("e8"))

    def test_a_hand_built_position_has_one_move_left_in_its_turn(self) -> None:
        # `remaining` is not set from the turn rule: whoever moves first moves once.
        engine = _chess_with({"turn": {"moves": 2}}, drop=("castling", "en_passant"))
        built = engine.make_state({**KINGS, "a1": ("R", 0)})
        assert built.remaining == 1
        after = engine.apply(built, _move(engine, built, "a1", "a2"))
        assert after.side_to_move == 1

    def test_a_hand_built_position_has_no_turns_taken_so_the_ramp_starts_at_its_first(
        self,
    ) -> None:
        engine = _chess_with(
            {"turn": {"moves": 3, "ramp": [1, 2]}}, drop=("castling", "en_passant")
        )
        built = engine.make_state({**KINGS, "a1": ("R", 0)})
        after = engine.apply(built, _move(engine, built, "a1", "a2"))
        assert (after.side_to_move, after.remaining, after.turns_taken) == (1, 2, 1)


class TestASquareNameIsCheckedAgainstTheBoard:
    @pytest.mark.parametrize("name", ["`1", "i1", "a9", "a0", "a-1"])
    def test_a_name_off_the_board_is_refused_as_a_placement(self, name: str) -> None:
        engine = _chess_with()
        with pytest.raises(ValueError, match="is not a square on a 8x8 board"):
            engine.make_state({name: ("K", 0)})

    @pytest.mark.parametrize("name", ["`1", "i1", "a9", "a0"])
    def test_a_name_off_the_board_is_refused_as_an_en_passant_square(self, name: str) -> None:
        engine = _chess_with()
        with pytest.raises(ValueError, match="is not a square on a 8x8 board"):
            engine.make_state(KINGS, ep_target=name)

    @pytest.mark.parametrize(("name", "index"), [("a1", 0), ("h1", 7), ("a8", 56), ("h8", 63)])
    def test_the_four_corners_are_squares(self, name: str, index: int) -> None:
        engine = _chess_with()
        state = engine.make_state({**KINGS, name: ("R", 0)} if name not in KINGS else KINGS)
        assert state.board[index] != 0


# ----------------------------------------------------- the defaults of the analysis calls


def _imbalance(engine: Engine, drop: str, side: int) -> GameState:
    opening = engine.initial_state()
    kept = {
        square: held
        for square, held in _placement_of(engine, opening).items()
        if not (held[1] == side and held[0] == drop)
    }
    return engine.make_state(kept)


class TestTheAnalysisDefaultsAreTheOffSwitches:
    def test_root_scores_without_a_goal_weight_is_the_scores_with_zero(self) -> None:
        engine = Engine(_load("rulesets/intransitive.json"))
        values = eval_values(engine.spec)
        state = _imbalance(engine, "R", 0)
        default = engine.root_scores(state, values, 100, 2)
        assert default == engine.root_scores(state, values, 100, 2, 0.0, 0.0)
        # The test could not tell the default from another weight were the term inert here.
        assert default != engine.root_scores(state, values, 100, 2, 1.0, 0.0)

    def test_root_scores_without_an_immunity_weight_is_the_scores_with_zero(self) -> None:
        engine = Engine(_load("rulesets/rps-v1.json"))
        values = eval_values(engine.spec)
        state = _imbalance(engine, "P", 1)
        default = engine.root_scores(state, values, 100, 2)
        assert default == engine.root_scores(state, values, 100, 2, 0.0, 0.0)
        assert default != engine.root_scores(state, values, 100, 2, 0.0, 1.0)

    def test_play_from_without_a_series_returns_none_of_it(self) -> None:
        engine = _chess_with()
        natives: Any = (native_spec("greedy"), native_spec("greedy"))
        values = eval_values(engine.spec)
        plain = engine.play_from(engine.initial_state(), natives, values, 3)
        assert plain[3] == [] and plain[4] == []
        series = engine.play_from(engine.initial_state(), natives, values, 3, True)
        assert series[3] and series[4]
        assert series[0] == plain[0]  # the moves played do not depend on recording them

    def test_a_perft_at_a_decided_root_is_the_same_on_every_thread_count(self) -> None:
        engine = Engine(_load("rulesets/kingofthehill.json"))
        decided = engine.make_state(
            {"e4": ("K", 0), "a8": ("K", 1), "h1": ("R", 0)}, side_to_move=1
        )
        assert engine.terminal(decided) is not None
        counts = {
            engine.perft(decided, 2, threads=threads, stop_at_terminal=True)
            for threads in (1, 2, 4)
        }
        assert counts == {0}
        assert engine.perft(decided, 2, stop_at_terminal=True) == 0


# ---------------------------------------------------------------------- the dataclasses


class TestTheStateRecordsAreFrozenHashableAndSlotted:
    @staticmethod
    def _instances() -> list[Any]:
        engine = _chess_with()
        return [engine.initial_state(), Outcome(0, "checkmate"), PhaseStep(1, 2, 0)]

    def test_no_field_can_be_set(self) -> None:
        for record in self._instances():
            field = dataclasses.fields(record)[0].name
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(record, field, getattr(record, field))

    def test_each_is_hashable_by_value(self) -> None:
        for record in self._instances():
            assert hash(record) == hash(copy.copy(record))

    def test_none_carries_a_dict(self) -> None:
        for record in self._instances():
            with pytest.raises(TypeError):
                vars(record)

    def test_the_state_class_is_the_one_the_engine_builds(self) -> None:
        assert isinstance(_chess_with().initial_state(), GameState)
