"""Schema 17: what a capture leaves behind (Q540).

Four features, all rulings on a capture that movegen already generates, so none moves a
perft of any ruleset that does not use it:

* rifle -- ``strike: {"effect": "capture", "position": "stay"}``: the victim is removed
  and the attacker stays where it stood;
* circe -- ``strike: {"effect": "capture", "position": "home"}``: the victim is reborn on
  its home square, or removed when that square is taken;
* Andernach -- ``strike.switches_sides``: the capturer changes sides where it lands;
* kamikaze -- ``explode_on_capture.radius: 0``: the capturer and its victim both go.

The inertness proof is `test_schema_nine.py::TestTheBumpMovedNothing`, which pins the
mechanics hash and tables fingerprint of every committed and archived ruleset. This file
holds the features: gated, refused where they mean nothing, played by the core, and read
by every surface `new-move-primitive` names.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables, canonical_tables
from chess_variant_sandbox.engine.compile import REMOVED_IN_PLACE, compile_spec
from chess_variant_sandbox.engine.moves import move_from, move_to
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.complexity import measure_complexity
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.distance import distance
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.schema import (
    PLAYED_AS_CAPTURE,
    SCHEMA_VERSION,
    _schema_seventeen_uses,
)

ROOT = Path(__file__).resolve().parents[1]
SHOWCASES = ("rifle", "circe", "andernach", "kamikaze")


def _load(name: str) -> dict[str, Any]:
    folder = "rulesets" if name == "chess" else "archive"
    return json.loads((ROOT / folder / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def games() -> dict[str, dict[str, Any]]:
    return {name: _load(name) for name in (*SHOWCASES, "chess")}


def _square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + ord(name[0]) - ord("a")


def _moves_between(engine: Engine, state: Any, origin: str, target: str) -> list[int]:
    wanted = (_square(origin), _square(target))
    return [m for m in engine.legal_moves(state) if (move_from(m), move_to(m)) == wanted]


def _play(engine: Engine, state: Any, line: list[tuple[str, str]]) -> Any:
    for origin, target in line:
        moves = _moves_between(engine, state, origin, target)
        assert len(moves) == 1, (origin, target, moves)
        state = engine.apply(state, moves[0])
        assert state.key == engine._compute_hash(state), "hash drifted"
    return state


def _occupant(engine: Engine, state: Any, square: str) -> tuple[int, str] | None:
    value = state.board[_square(square)]
    if value == 0:
        return None
    return engine.spec.owner_of(value), engine.spec.piece_ids[engine.spec.type_of(value)]


def _position(spec: dict[str, Any], placements: list[dict[str, Any]]) -> dict[str, Any]:
    """The showcase's rules on a hand-built position."""
    built = copy.deepcopy(spec)
    built["rules"].pop("castling")
    built["setup"] = {"mirror": False, "placements": placements}
    assert validate_spec(built) == [], validate_spec(built)
    return built


# 1. e4 d5 2. exd5: the first capture of each showcase.
OPENING = [("e2", "e4"), ("d7", "d5"), ("e4", "d5")]


class TestTheVersionIsBumped:
    def test_the_schema_version_is_seventeen(self) -> None:
        assert SCHEMA_VERSION >= 17

    def test_the_collector_names_each_showcase_once(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        assert _schema_seventeen_uses(games["chess"]) == set()
        assert _schema_seventeen_uses(games["rifle"]) == {"rules.strike.position=stay(capture)"}
        assert _schema_seventeen_uses(games["circe"]) == {"rules.strike.position=home"}
        assert _schema_seventeen_uses(games["andernach"]) == {"rules.strike.switches_sides"}
        assert _schema_seventeen_uses(games["kamikaze"]) == {"rules.explode_on_capture.radius=0"}

    @pytest.mark.parametrize("name", SHOWCASES)
    def test_an_older_stamp_is_refused_by_name(
        self, games: dict[str, dict[str, Any]], name: str
    ) -> None:
        spec = copy.deepcopy(games[name])
        spec["schema_version"] = 16
        errors = validate_spec(spec)
        assert any("schema 17" in e for e in errors), errors

    def test_a_stay_strike_under_a_chain_is_schema_six_still(self) -> None:
        stay_chain = {
            "rules": {
                "strike": {
                    "effect": "demote",
                    "position": "stay",
                    "chain": [{"from": "Q", "to": "R"}],
                }
            }
        }
        assert _schema_seventeen_uses(stay_chain) == set()


class TestTheValidator:
    @pytest.mark.parametrize("name", SHOWCASES)
    def test_the_showcase_validates(self, games: dict[str, dict[str, Any]], name: str) -> None:
        assert validate_spec(games[name]) == []

    def test_a_royal_never_switches_sides(self, games: dict[str, dict[str, Any]]) -> None:
        spec = copy.deepcopy(games["andernach"])
        spec["rules"]["strike"]["switches_sides"].append("K")
        assert any("royal piece 'K'" in e for e in validate_spec(spec))

    def test_an_empty_switch_list_is_refused(self, games: dict[str, dict[str, Any]]) -> None:
        spec = copy.deepcopy(games["andernach"])
        spec["rules"]["strike"]["switches_sides"] = []
        assert any("switches_sides must be a non-empty" in e for e in validate_spec(spec))

    def test_a_royal_must_be_immune_under_rifle_and_circe(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        for name in ("rifle", "circe"):
            spec = copy.deepcopy(games[name])
            spec["rules"]["strike"].pop("immune")
            assert any("not in rules.strike.immune" in e for e in validate_spec(spec)), name

    def test_home_is_a_strike_position_only(self, games: dict[str, dict[str, Any]]) -> None:
        spec = copy.deepcopy(games["chess"])
        spec["schema_version"] = 17
        spec["rules"]["contacts"] = {
            "position": "home",
            "immune": ["K"],
            "links": [{"mover": "P", "onto": "N", "mover_to": "B"}],
        }
        assert any("rules.contacts.position must be one of" in e for e in validate_spec(spec))

    def test_a_capture_that_swaps_still_states_nothing(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        spec = copy.deepcopy(games["rifle"])
        spec["rules"]["strike"]["position"] = "swap"
        assert any(PLAYED_AS_CAPTURE in e for e in validate_spec(spec))

    def test_kamikaze_refuses_an_immune_list(self, games: dict[str, dict[str, Any]]) -> None:
        spec = copy.deepcopy(games["kamikaze"])
        spec["rules"]["explode_on_capture"]["immune"] = ["P"]
        assert any("radius 0 reaches nobody" in e for e in validate_spec(spec))

    @pytest.mark.parametrize("other", ["drops", "explode_on_capture"])
    @pytest.mark.parametrize("name", ["rifle", "circe", "andernach"])
    def test_one_rule_says_what_a_capture_leaves(
        self, games: dict[str, dict[str, Any]], name: str, other: str
    ) -> None:
        spec = copy.deepcopy(games[name])
        spec["rules"][other] = (
            {"from_captures": True} if other == "drops" else {"radius": 1}
        )
        assert any(f"rules.strike and rules.{other}" in e for e in validate_spec(spec))


class TestTheCompiledTables:
    def test_rifle_removes_in_place(self, games: dict[str, dict[str, Any]]) -> None:
        strike = compile_spec(games["rifle"]).strike
        assert strike is not None and strike.removes_in_place
        king = games["rifle"]["pieces"].index(next(p for p in games["rifle"]["pieces"] if p["id"] == "K"))
        assert strike.compact_to is not None
        assert strike.compact_to[king] == -1
        assert all(
            entry == REMOVED_IN_PLACE for index, entry in enumerate(strike.compact_to) if index != king
        )

    def test_circe_reborn_squares_are_orthodox_on_chess_setup(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        spec = compile_spec(games["circe"])
        strike = spec.strike
        assert strike is not None and strike.position == "home"
        ids = spec.piece_ids

        def home(owner: int, piece: str, taken: str) -> str | None:
            square = strike.rebirth[
                (owner * spec.n_types + ids.index(piece)) * spec.n_squares + _square(taken)
            ]
            return None if square < 0 else f"{chr(97 + square % 8)}{square // 8 + 1}"

        assert home(0, "P", "e5") == "e2"
        assert home(1, "P", "b3") == "b7"
        assert home(0, "R", "d4") == "a1"  # d4 is dark, as a1 is
        assert home(0, "R", "e4") == "h1"
        assert home(0, "N", "d4") == "g1"
        assert home(0, "B", "d4") == "c1"
        assert home(1, "B", "d4") == "f8"
        assert home(1, "Q", "a1") == "d8"
        # Q963: on a home file the colour still decides for a type with one home of each
        # colour. Until then the file came first, and a knight taken on b4 went to b1.
        assert home(0, "N", "b4") == "g1"  # b4 is dark, as g1 is
        assert home(0, "N", "g5") == "g1"
        assert home(1, "N", "g5") == "b8"
        assert home(0, "B", "c4") == "f1"  # a light-squared bishop stays one
        assert home(1, "B", "f3") == "c8"
        assert home(0, "R", "h4") == "a1"
        assert home(0, "Q", "e5") == "d1"
        assert home(0, "P", "c5") == "c2"  # a pawn's file still decides

    def test_every_rebirth_keeps_a_one_per_colour_piece_on_its_colour(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        spec = compile_spec(games["circe"])
        strike = spec.strike
        assert strike is not None
        width = spec.width
        for piece in ("R", "N", "B"):
            for owner in (0, 1):
                for taken in range(spec.n_squares):
                    reborn = strike.rebirth[
                        (owner * spec.n_types + spec.piece_ids.index(piece)) * spec.n_squares
                        + taken
                    ]
                    shade = (reborn % width + reborn // width) % 2
                    assert shade == (taken % width + taken // width) % 2, (piece, owner, taken)

    def test_the_file_still_decides_for_a_type_with_two_homes_of_one_colour(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        # Two rooks on one colour (a1, c1): colour cannot tell them apart, so the file
        # does, then the nearest -- the reading the generic rule keeps for pawns.
        spec = copy.deepcopy(games["circe"])
        for placement in spec["setup"]["placements"]:
            if placement["piece"] == "R":
                placement["squares"] = ["a1", "c1"]
            elif placement["piece"] == "B":
                placement["squares"] = ["f1", "h1"]
        del spec["rules"]["castling"]
        assert validate_spec(spec) == []
        compiled = compile_spec(spec)
        strike = compiled.strike
        assert strike is not None
        rook = compiled.piece_ids.index("R")
        lookup = {
            taken: strike.rebirth[rook * compiled.n_squares + _square(taken)]
            for taken in ("a5", "c6", "b4", "h8")
        }
        assert lookup == {
            "a5": _square("a1"),
            "c6": _square("c1"),
            "b4": _square("a1"),  # dark, as a1 and c1 are: nearest by file, then lower
            "h8": _square("c1"),  # dark too: c1 is nearer by file
        }

    def test_andernach_crosses_as_the_contact_tables(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        compiled = compile_spec(games["andernach"])
        tables = build_tables(compiled)
        n = compiled.n_types
        assert tables["strike_to"] == []
        assert len(tables["contact_mover_flips"]) == n * n
        king = compiled.piece_ids.index("K")
        assert not any(tables["contact_mover_flips"][king * n : (king + 1) * n])

    def test_kamikaze_blast_reaches_nobody(self, games: dict[str, dict[str, Any]]) -> None:
        explosion = compile_spec(games["kamikaze"]).explosion
        assert explosion is not None
        assert all(squares == () for squares in explosion.blast)

    def test_the_new_fields_stay_out_of_every_older_rulesets_tables(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        for name in ("chess", "kamikaze"):
            tables = canonical_tables(compile_spec(games[name]))
            assert "strike_rebirth" not in tables and "contact_mover_flips" not in tables


class TestTheCorePlaysIt:
    def test_rifle_leaves_the_capturer_where_it_stood(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        engine = Engine(games["rifle"])
        after = _play(engine, engine.initial_state(), OPENING)
        assert _occupant(engine, after, "e4") == (0, "P")
        assert _occupant(engine, after, "d5") is None

    def test_circe_puts_the_victim_back_and_it_may_move_as_new(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        engine = Engine(games["circe"])
        after = _play(engine, engine.initial_state(), OPENING)
        assert _occupant(engine, after, "d5") == (0, "P")
        assert _occupant(engine, after, "d7") == (1, "P")
        assert after.unmoved >> _square("d7") & 1
        # 2... c5 3. dxc6 e.p. puts the c-pawn back on c7, and clears d5: the pawn reborn
        # on d7 stands as if it had never moved, so its double step is back.
        after = _play(engine, after, [("c7", "c5"), ("d5", "c6"), ("d7", "d5")])
        assert _occupant(engine, after, "c7") == (1, "P")
        assert _occupant(engine, after, "d5") == (1, "P")

    def test_circe_removes_a_piece_whose_home_is_taken(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        engine = Engine(games["circe"])
        # 2... Nf6 3. Nc3 Nxd5: the white pawn's home is d2, where white's d-pawn stands.
        after = _play(
            engine, engine.initial_state(), OPENING + [("g8", "f6"), ("b1", "c3"), ("f6", "d5")]
        )
        assert _occupant(engine, after, "d5") == (1, "N")
        white_pawns = sum(
            1 for value in after.board
            if value and engine.spec.owner_of(value) == 0
            and engine.spec.piece_ids[engine.spec.type_of(value)] == "P"
        )
        assert white_pawns == 7

    def test_circe_sends_a_knight_home_by_its_colour_in_play(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        # Q963. 1. Nc3 a6 2. Nf3 a5 3. Nh4 a4 4. Ng6 hxg6: the king's knight falls on g6,
        # a light square, so it is reborn on b1 (light, emptied by 1. Nc3), not on g1.
        engine = Engine(games["circe"])
        line = [
            ("b1", "c3"), ("a7", "a6"), ("g1", "f3"), ("a6", "a5"),
            ("f3", "h4"), ("a5", "a4"), ("h4", "g6"), ("h7", "g6"),
        ]
        after = _play(engine, engine.initial_state(), line)
        assert _occupant(engine, after, "g6") == (1, "P")
        assert _occupant(engine, after, "b1") == (0, "N")
        assert _occupant(engine, after, "g1") is None

    def test_andernach_sends_the_capturer_over(self, games: dict[str, dict[str, Any]]) -> None:
        engine = Engine(games["andernach"])
        after = _play(engine, engine.initial_state(), OPENING)
        assert _occupant(engine, after, "d5") == (1, "P")
        # It is black's pawn now, and walks black's way.
        after = _play(engine, after, [("d5", "d4")])
        assert _occupant(engine, after, "d4") == (1, "P")

    def test_andernach_refuses_a_capture_that_turns_on_its_own_king(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        engine = Engine(
            _position(
                games["andernach"],
                [
                    {"piece": "K", "squares": ["e1"]},
                    {"piece": "R", "squares": ["e4"]},
                    {"piece": "P", "squares": ["h2"]},
                    {"piece": "K", "player": 1, "squares": ["a8"]},
                    {"piece": "P", "player": 1, "squares": ["a7"]},
                    {"piece": "N", "player": 1, "squares": ["e3"]},
                    {"piece": "N", "player": 1, "squares": ["h4"]},
                ],
            )
        )
        state = engine.initial_state()
        # Rxe3 would leave a black rook on e3 facing the white king on e1.
        assert _moves_between(engine, state, "e4", "e3") == []
        assert len(_moves_between(engine, state, "e4", "h4")) == 1

    def test_kamikaze_takes_both_pieces_and_the_king_cannot_take(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        engine = Engine(games["kamikaze"])
        after = _play(engine, engine.initial_state(), OPENING)
        assert _occupant(engine, after, "d5") is None
        assert _occupant(engine, after, "e4") is None
        lone = Engine(
            _position(
                games["kamikaze"],
                [
                    {"piece": "K", "squares": ["e1"]},
                    {"piece": "K", "player": 1, "squares": ["e8"]},
                    {"piece": "P", "player": 1, "squares": ["e2"]},
                ],
            )
        )
        assert _moves_between(lone, lone.initial_state(), "e1", "e2") == []

    @pytest.mark.parametrize("name", SHOWCASES)
    def test_make_and_unmake_agree_with_a_walk_that_never_unmakes(
        self, games: dict[str, dict[str, Any]], name: str
    ) -> None:
        """Perft plays every move with `make` and takes it back with `unmake`. `apply`
        builds each position afresh. A reborn piece `unmake` failed to lift, or a side
        switch it failed to undo, would show as a different count one ply later."""
        engine = Engine(games[name])
        state = _play(
            engine, engine.initial_state(), [("e2", "e4"), ("d7", "d5"), ("g1", "f3"), ("c8", "g4")]
        )
        walked = 0
        for first in engine.legal_moves(state):
            one = engine.apply(state, first)
            for second in engine.legal_moves(one):
                walked += len(engine.legal_moves(engine.apply(one, second)))
        assert engine.perft(state, 3) == walked

    @pytest.mark.parametrize("name", SHOWCASES)
    def test_a_short_game_keeps_its_incremental_hash(
        self, games: dict[str, dict[str, Any]], name: str
    ) -> None:
        engine = Engine(games[name])
        rng = random.Random(17)
        state = engine.initial_state()
        captures = 0
        for _ in range(160):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            taking = [m for m in moves if state.board[move_to(m)]]
            captures += bool(taking)
            move = rng.choice(taking) if taking and rng.random() < 0.5 else rng.choice(moves)
            state = engine.apply(state, move)
            assert state.key == engine._compute_hash(state), "hash drifted"
        assert captures > 5

    @pytest.mark.parametrize("name", SHOWCASES)
    def test_perft_matches_chess_until_a_capture(
        self, games: dict[str, dict[str, Any]], name: str
    ) -> None:
        chess, variant = Engine(games["chess"]), Engine(games[name])
        for depth in (1, 2, 3):
            assert variant.perft(variant.initial_state(), depth) == chess.perft(
                chess.initial_state(), depth
            )


class TestTheOtherSurfaces:
    @pytest.mark.parametrize(
        ("name", "phrase"),
        [
            ("rifle", "stays where it stood"),
            ("circe", "reborn on its home square"),
            ("andernach", "changes sides"),
            ("kamikaze", "only the capturing piece and the piece it took"),
        ],
    )
    def test_the_description_says_it(
        self, games: dict[str, dict[str, Any]], name: str, phrase: str
    ) -> None:
        assert phrase in describe_spec(games[name])

    @pytest.mark.parametrize("name", ["rifle", "circe", "andernach"])
    def test_a_capture_is_named_by_what_it_left(
        self, games: dict[str, dict[str, Any]], name: str
    ) -> None:
        engine = Engine(games[name])
        state = _play(engine, engine.initial_state(), OPENING[:2])
        (move,) = _moves_between(engine, state, "e4", "d5")
        text = engine.describe_move(state, move)
        expected = {"rifle": "(stays)", "circe": "(reborn d7)", "andernach": "(changes sides)"}
        assert text == f"Pe4xd5 {expected[name]}", text

    @pytest.mark.parametrize("name", SHOWCASES)
    def test_it_is_charged_and_has_a_distance_from_chess(
        self, games: dict[str, dict[str, Any]], name: str
    ) -> None:
        assert measure_complexity(games[name]).rule_mass > measure_complexity(
            games["chess"]
        ).rule_mass
        assert distance(games[name], games["chess"]) > 0.0

    def test_the_four_are_apart_from_each_other(self, games: dict[str, dict[str, Any]]) -> None:
        for index, first in enumerate(SHOWCASES):
            for second in SHOWCASES[index + 1 :]:
                assert distance(games[first], games[second]) > 0.0, (first, second)

    @pytest.mark.parametrize("name", SHOWCASES)
    def test_fairy_refuses_it_by_name(self, games: dict[str, dict[str, Any]], name: str) -> None:
        with pytest.raises(Untranslatable) as refusal:
            translate(compile_spec(games[name]))
        key = "explode_on_capture" if name == "kamikaze" else "rules.strike"
        assert any(key in reason for reason in refusal.value.reasons)

    def test_the_json_schema_offers_it_to_a_proposer(self) -> None:
        rules = rule_spec_schema()["properties"]["rules"]["properties"]
        assert "home" in rules["strike"]["properties"]["position"]["enum"]
        assert "home" not in rules["contacts"]["properties"]["position"]["enum"]
        assert "switches_sides" in rules["strike"]["properties"]
        assert rules["explode_on_capture"]["properties"]["radius"]["minimum"] == 0


class TestAContactOverridesAPromotion:
    """Q539, assessed in `docs/archive/plan-schema-17-18.md` §8: when a contact link and a
    promotion fire on one capture, the contact wins (`strike_writes` reads the promoted piece and
    replaces it), so the move text must not claim the promotion."""

    def test_the_text_says_what_stands_there(self, games: dict[str, dict[str, Any]]) -> None:
        spec = copy.deepcopy(games["chess"])
        spec["schema_version"] = 17
        spec["rules"].pop("castling")
        spec["rules"].pop("en_passant")
        spec["rules"]["contacts"] = {
            "position": "advance",
            "immune": ["K"],
            "links": [{"mover": "P", "onto": "R", "mover_to": "N"}],
        }
        spec["setup"] = {
            "mirror": False,
            "placements": [
                {"piece": "K", "squares": ["e1"]},
                {"piece": "P", "squares": ["b7"]},
                {"piece": "K", "player": 1, "squares": ["h8"]},
                {"piece": "R", "player": 1, "squares": ["a8"]},
                {"piece": "P", "player": 1, "squares": ["h7"]},
            ],
        }
        assert validate_spec(spec) == [], validate_spec(spec)
        engine = Engine(spec)
        state = engine.initial_state()
        takes = _moves_between(engine, state, "b7", "a8")
        assert len(takes) == 4  # one word per promotion target
        for move in takes:
            text = engine.describe_move(state, move)
            assert "/P->N" in text and "=" not in text, text
            assert _occupant(engine, engine.apply(state, move), "a8") == (0, "N")
