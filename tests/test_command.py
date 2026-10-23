"""Q1007c: the `command` effect, one test per decision of `docs/plan-schema-31.md` §3.

A fourth sight effect: the affected piece's **opponent** may also move it, to empty squares,
and with ``captures_own`` make it take its own side's non-royal pieces. Royals are never
commanded, a commanded move never takes the commander's piece, never castles, takes or opens en
passant, drops or promotes, and every one is verified for the commander's royal. The positions
are hand-built on chess (K1: commanding knights; K2: a spellbinding king, `captures_own`) and on
the long-turn test spec; `tests/test_command_walks.py` holds the core to the oracle over games.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import SIGHT_COMMAND, SIGHT_COMMAND_OWN, compile_spec
from chess_variant_sandbox.engine.moves import MOVE_COMMANDED, MOVE_ROTATE, decode_move
from chess_variant_sandbox.engine.state import NO_SQUARE, GameState
from chess_variant_sandbox.spec import spec_hash, validate_spec

HERE = Path(__file__).parent
ROOT = HERE.parent

KNIGHT_COMMAND = {"vectors": [[1, 2]], "symmetry": "dihedral8", "range": 1,
                  "effect": "command", "whose": "enemy"}
KING_COMMAND = {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 1,
                "effect": "command", "whose": "enemy", "captures_own": True}


def chess() -> dict[str, Any]:
    spec: dict[str, Any] = json.loads((ROOT / "rulesets" / "chess.json").read_text("utf-8"))
    spec.pop("provenance", None)
    return spec


def with_sight(spec: dict[str, Any], pid: str, sight: dict[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(spec)
    for piece in spec["pieces"]:
        if piece["id"] == pid:
            piece.setdefault("sight", []).append(copy.deepcopy(sight))
    return spec


def k1() -> dict[str, Any]:
    return with_sight(chess(), "N", KNIGHT_COMMAND)


def k2() -> dict[str, Any]:
    return with_sight(chess(), "K", KING_COMMAND)


@cache
def engine_of(text: str) -> Engine:
    return Engine(json.loads(text))


def engine(spec: dict[str, Any]) -> Engine:
    return engine_of(json.dumps(spec, sort_keys=True))


def position(spec: dict[str, Any], placement: dict[str, tuple[str, int]], side: int = 0,
             moved: tuple[str, ...] = ()) -> tuple[Engine, GameState]:
    built = engine(spec)
    return built, built.make_state(placement, side_to_move=side, moved=moved)


def commanded_texts(built: Engine, state: GameState) -> list[str]:
    return [text for text in built.move_texts(state) if text.startswith("~")]


def square(built: Engine, name: str) -> int:
    return built._square_index(name)


# --------------------------------------------------------------------------- the spelling


def errors_of(spec: dict[str, Any]) -> list[str]:
    return validate_spec(spec)


def test_k1_and_k2_validate() -> None:
    assert errors_of(k1()) == []
    assert errors_of(k2()) == []


def test_a_royal_named_by_a_command_is_refused() -> None:
    spec = with_sight(chess(), "N", {**KNIGHT_COMMAND, "affects": ["K", "Q"]})
    assert any("a royal cannot be commanded" in error for error in errors_of(spec))


def test_a_command_without_affects_skips_the_royal_types() -> None:
    sight = compile_spec(k1()).sight
    assert sight is not None
    king = 5
    assert sight.has_command
    assert all(not found for side in sight.sources for found in side[king])


def test_captures_own_means_something_only_beside_command() -> None:
    spec = with_sight(chess(), "N", {**KNIGHT_COMMAND, "effect": "freeze", "captures_own": True})
    assert any("only beside effect 'command'" in error for error in errors_of(spec))
    spec = with_sight(chess(), "N", {**KNIGHT_COMMAND, "captures_own": "yes"})
    assert any("captures_own must be true or false" in error for error in errors_of(spec))


def test_captures_own_false_is_the_off_value() -> None:
    off = with_sight(chess(), "N", {**KNIGHT_COMMAND, "captures_own": False})
    assert errors_of(off) == []
    assert spec_hash(off) == spec_hash(k1())


@pytest.mark.parametrize(
    ("key", "value", "words"),
    [
        ("simultaneous", {}, "rules.simultaneous"),
        ("reach", {"squares": {"white": ["e8"], "black": ["e1"]}}, "rules.reach"),
        ("territory", {"win_at": 40}, "rules.territory"),
        ("strike", {"effect": "promote", "chain": [{"from": "P", "to": "N"}], "immune": ["K"]},
         "rules.strike"),
    ],
)
def test_the_rules_refused_beside_a_command(key: str, value: Any, words: str) -> None:
    spec = k1()
    if key == "simultaneous":
        # A round game has no check, castling or en passant; leave them out so only the
        # command is left to refuse.
        for rule in ("castling", "en_passant"):
            spec["rules"].pop(rule, None)
        spec["rules"]["win"] = "royal_capture"
    spec["rules"][key] = value
    assert any(
        "beside " + words in error and "command" in error for error in errors_of(spec)
    ), errors_of(spec)


def test_a_rest_beside_a_command_is_refused() -> None:
    spec = k1()
    spec["pieces"][1]["rest"] = 1
    assert any("beside pieces[].rest" in error for error in errors_of(spec))


def test_a_sweep_beside_a_command_is_refused() -> None:
    spec = k1()
    spec["pieces"][3]["moves"] = [{"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}]
    assert any("command" in error and "sweep" in error for error in errors_of(spec))


def test_captures_own_beside_a_capture_hand_is_refused() -> None:
    spec = k2()
    spec["rules"]["drops"] = {}
    assert any("from_captures" in error for error in errors_of(spec))
    spec = k1()
    spec["rules"]["drops"] = {}
    assert not any("from_captures" in error for error in errors_of(spec))


def test_a_command_that_can_never_bind_is_refused() -> None:
    spec = with_sight(chess(), "N", {**KNIGHT_COMMAND, "affects": ["P"]})
    for piece in spec["pieces"]:
        if piece["id"] == "P":
            piece["captures"] = ["K"]
    spec2 = with_sight(chess(), "N", {**KNIGHT_COMMAND, "affects": ["P"], "captures_own": True})
    for piece in spec2["pieces"]:
        if piece["id"] == "P":
            piece["captures"] = ["K"]
    assert errors_of(spec) == []
    assert any(
        "no type it commands may take a piece of its own side" in error
        for error in errors_of(spec2)
    )


def test_every_existing_sight_game_compiles_to_the_tables_it_had() -> None:
    """Inert: the new flag sits at its inert default, so no sight game's tables move."""
    for path in sorted((ROOT / "archive").glob("*.json")):
        spec = json.loads(path.read_text("utf-8"))
        if not any(piece.get("sight") for piece in spec.get("pieces", [])):
            continue
        tables = canonical_tables(compile_spec(spec))
        assert "sight_command" not in tables, path.name


# ------------------------------------------------------------------------- the positions

# K2's position: the queen beside the king, protected from h5 (§6.3: the classic mate).
K2_MATE = {
    "e1": ("K", 0), "e2": ("Q", 1), "h5": ("B", 1), "a8": ("K", 1), "b5": ("N", 1),
}


def test_a_check_is_answered_by_commanding_the_checker_away() -> None:
    built, state = position(k2(), K2_MATE)
    assert built.in_check(state, 0)
    moves = built.legal_moves(state)
    assert moves and all(move & MOVE_COMMANDED for move in moves)
    texts = commanded_texts(built, state)
    assert "~Qe2-c4" in texts
    # Or made to take its own protector, or another of its side.
    assert "~Qe2xh5" in texts and "~Qe2xb5" in texts
    # Not along the king's file or beside it, where it would still give check.
    assert "~Qe2-e5" not in texts and "~Qe2-d2" not in texts
    assert built.terminal(state) is None
    plain, plain_state = position(chess(), K2_MATE)
    assert plain.legal_moves(plain_state) == []


def test_a_commanded_move_is_found_by_its_text() -> None:
    built, state = position(k2(), K2_MATE)
    move = built.find_move(state, "~Qe2-c4")
    assert move & MOVE_COMMANDED
    after = built.apply(state, move)
    assert after.board[square(built, "c4")] == state.board[square(built, "e2")]
    assert after.side_to_move == 1


def test_the_effects_report_the_command_and_count_the_commanded_moves() -> None:
    built, state = position(k2(), K2_MATE)
    effects = built.effects(state)
    bits = effects.bits[square(built, "e2")]
    assert bits & SIGHT_COMMAND and bits & SIGHT_COMMAND_OWN
    assert effects.commanded == len(built.legal_moves(state))
    plain, plain_state = position(chess(), K2_MATE)
    assert plain.effects(plain_state).commanded == 0


def test_a_commanded_move_can_discover_check() -> None:
    built, state = position(k1(), {
        "e1": ("R", 0), "h1": ("K", 0), "c4": ("N", 0), "e5": ("N", 1), "e8": ("K", 1),
    })
    after = built.apply(state, built.find_move(state, "~Ne5-g6"))
    assert built.in_check(after, 1)


def test_a_commanded_move_that_unblocks_its_own_side_is_not_offered() -> None:
    """A knight shielding the king from its own side's rook may not be commanded away."""
    built, state = position(k2(), {
        "e1": ("K", 0), "e2": ("N", 1), "e7": ("R", 1), "a8": ("K", 1),
    })
    origin = square(built, "e2")
    assert any(
        move & MOVE_COMMANDED and move & 0xFF == origin
        for move in built.pseudo_legal_moves(state)
    )
    assert commanded_texts(built, state) == []
    assert built.legal_moves(state)


def test_a_commanded_move_that_would_expose_the_commanders_royal_is_not_offered() -> None:
    built, state = position(k1(), {
        "e1": ("K", 0), "f2": ("N", 0), "e4": ("B", 1), "e8": ("R", 1), "a8": ("K", 1),
    })
    origin = square(built, "e4")
    assert any(
        move & MOVE_COMMANDED and move & 0xFF == origin
        for move in built.pseudo_legal_moves(state)
    )
    assert not any(text.startswith("~Be4") for text in built.move_texts(state))


def test_a_frozen_piece_is_not_commanded() -> None:
    freezing = with_sight(k1(), "R", {"vectors": [[1, 0]], "symmetry": "rotate4",
                                      "effect": "freeze", "whose": "enemy"})
    placement = {"a1": ("K", 0), "c3": ("N", 0), "d5": ("B", 1), "h8": ("K", 1)}
    built, state = position(freezing, {**placement, "d1": ("R", 0)})
    assert not any(text.startswith("~Bd5") for text in built.move_texts(state))
    built, state = position(freezing, {**placement, "h1": ("R", 0)})
    assert any(text.startswith("~Bd5") for text in built.move_texts(state))


def test_no_promotion_and_no_window_by_command() -> None:
    built, state = position(k1(), {
        "h1": ("K", 0), "a4": ("N", 0), "b2": ("P", 1), "c5": ("N", 0), "d7": ("P", 1),
        "d3": ("N", 0), "e5": ("P", 1), "h8": ("K", 1),
    }, moved=("e5",))
    texts = commanded_texts(built, state)
    assert not any(text.startswith("~Pb2") for text in texts)  # every arrival must promote
    assert "~Pe5-e4" in texts  # a commanded pawn steps toward the commander
    assert "~Pd7-d5" in texts  # and keeps its first move
    after = built.apply(state, built.find_move(state, "~Pd7-d5"))
    assert after.ep_target == NO_SQUARE and after.ep_victim == NO_SQUARE


def test_under_optional_promotion_a_commanded_piece_arrives_as_it_is() -> None:
    spec = k1()
    spec["rules"]["promotion"]["mandatory"] = False
    assert errors_of(spec) == []
    built, state = position(spec, {
        "h1": ("K", 0), "a4": ("N", 0), "b2": ("P", 1), "h8": ("K", 1),
    }, moved=("b2",))
    assert commanded_texts(built, state) == ["~Pb2-b1"]


def test_a_commanded_rook_loses_its_castling_right() -> None:
    built, state = position(k1(), {
        "e1": ("K", 0), "g6": ("N", 0), "h8": ("R", 1), "e8": ("K", 1),
    })
    after = built.apply(state, built.find_move(state, "~Rh8-h7"))
    assert not after.unmoved >> square(built, "h8") & 1
    assert not after.unmoved >> square(built, "h7") & 1


def test_a_commanded_move_never_castles() -> None:
    built, state = position(k1(), {
        "e1": ("K", 0), "g6": ("N", 0), "h8": ("R", 1), "e8": ("K", 1),
    })
    assert all(not text.startswith("~K") for text in built.move_texts(state))


def test_a_disarmed_piece_is_commanded_without_its_turncoat_captures() -> None:
    disarming = with_sight(k2(), "B", {"vectors": [[1, 1]], "symmetry": "rotate4",
                                       "effect": "disarm", "whose": "enemy"})
    built, state = position(disarming, {**K2_MATE, "c4": ("B", 0)})
    texts = commanded_texts(built, state)
    assert "~Qe2-f3" in texts
    assert not any("x" in text for text in texts)


def test_a_shielded_friend_is_not_taken_by_a_turncoat() -> None:
    shielding = with_sight(k2(), "B", {"vectors": [[1, 1]], "symmetry": "rotate4", "range": 1,
                                       "effect": "shield", "whose": "own"})
    built, state = position(shielding, {**K2_MATE, "c6": ("B", 1)})
    assert "~Qe2xb5" not in commanded_texts(built, state)
    built, state = position(shielding, {**K2_MATE, "h6": ("B", 1)})
    assert "~Qe2xb5" in commanded_texts(built, state)


def test_must_capture_counts_a_turncoat_capture() -> None:
    spec = k2()
    spec["rules"]["must_capture"] = True
    built, state = position(spec, K2_MATE)
    assert sorted(built.move_texts(state)) == ["~Qe2xb5", "~Qe2xh5"]


def test_stalemate_is_not_declared_while_a_commanded_move_exists() -> None:
    placement = {"a1": ("K", 0), "b2": ("R", 1), "c2": ("K", 1)}
    plain, plain_state = position(chess(), placement)
    assert plain.legal_moves(plain_state) == []
    built, state = position(k2(), placement)
    moves = built.legal_moves(state)
    assert moves and all(move & MOVE_COMMANDED for move in moves)
    assert built.terminal(state) is None


def test_a_turncoat_capture_can_make_its_owner_extinct() -> None:
    spec = k2()
    spec["rules"]["extinction"] = {"pieces": ["R"], "min_count": 1, "result": "loss"}
    assert errors_of(spec) == []
    built, state = position(spec, {
        "e1": ("K", 0), "h1": ("R", 0), "e2": ("Q", 1), "h5": ("B", 1), "b5": ("R", 1),
        "a8": ("K", 1),
    })
    after = built.apply(state, built.find_move(state, "~Qe2xb5"))
    outcome = built.terminal(after)
    assert outcome is not None and outcome.winner == 0


def test_a_long_turn_may_command_then_use_the_move_left() -> None:
    spec = json.loads((HERE / "_command_turns.json").read_text("utf-8"))
    built = engine(spec)
    state = built.make_state({
        "e1": ("K", 0), "c2": ("N", 0), "d4": ("P", 1), "a5": ("K", 1),
    }, moved=("d4",))
    state = dataclasses.replace(state, remaining=2)
    state = dataclasses.replace(state, key=built._compute_hash(state))
    after = built.apply(state, built.find_move(state, "~Pd4-d3"))
    assert after.side_to_move == 0 and after.remaining == 1
    texts = built.move_texts(after)
    assert any(text.startswith("Nc2") for text in texts)


# ------------------------------------------------------------------------- the surfaces


def test_describe_says_the_command_and_its_turncoat_captures() -> None:
    from chess_variant_sandbox.spec.describe import describe_spec

    said = describe_spec(k2())
    assert "commands enemy non-royal pieces it sees" in said
    assert "take a piece of their own side" in said
    assert "A royal is never commanded" in said
    assert "commanding the checker away" in said
    assert "take a piece of their own side" not in describe_spec(k1())


def test_complexity_charges_the_effect_and_half_a_rule_for_captures_own() -> None:
    from chess_variant_sandbox.spec import measure_complexity

    plain = measure_complexity(chess()).special_rules
    assert measure_complexity(k1()).special_rules == plain + 1.0
    turncoat = with_sight(chess(), "N", {**KNIGHT_COMMAND, "captures_own": True})
    assert measure_complexity(turncoat).special_rules == plain + 1.5


def test_a_command_and_its_captures_own_are_at_a_distance() -> None:
    from chess_variant_sandbox.spec.distance import distance

    turncoat = with_sight(chess(), "N", {**KNIGHT_COMMAND, "captures_own": True})
    assert distance(chess(), k1()) > 0
    assert distance(k1(), turncoat) > 0


def test_the_inventory_names_the_command_and_captures_own() -> None:
    import feature_inventory

    assert feature_inventory.uses(31, k1()) == {"pieces.sight.command"}
    assert feature_inventory.uses(31, k2()) == {"pieces.sight.command", "pieces.sight.captures_own"}
    assert feature_inventory.uses(30, k2()) == set()


def test_a_commanded_turn_keeps_the_piece_its_owners() -> None:
    """A commanded piece turns as its owner may turn it (§3.2), and stays its owner's."""
    spec = json.loads((HERE / "_command_facing.json").read_text("utf-8"))
    built, state = position(spec, {
        "e1": ("K", 0), "c3": ("N", 0), "d5": ("P", 1), "e8": ("K", 1),
    }, moved=("d5",))
    turns = [
        move for move in built.legal_moves(state)
        if move & MOVE_COMMANDED and decode_move(move)[2] == MOVE_ROTATE
    ]
    assert len(turns) == 3
    after = built.apply(state, turns[0])
    piece = after.board[square(built, "d5")]
    assert built.spec.owner_of(piece) == 1 and built.spec.type_of(piece) == 0
    assert after.side_to_move == 1
    assert after.key == built._compute_hash(after)
