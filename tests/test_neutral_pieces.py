"""TC3 (`docs/plan-turncoat.md` §4): neutral pieces and majority control.

`setup.placements[].player: "nobody"` puts pieces in nobody's block: they block, and either
player may take them. `pieces[].control: "majority"` lets the player with strictly more of
their own pieces next to one move it, as one of their moves, in frame 0; a tie (none against
none included) means nobody moves it. A held neutral takes only the mover's opponent's
non-royal pieces, attacks nothing, and stays nobody's.

Hand-built specs, not chess: a 6x6 board, two kings, a rook, a guard, and a condottiere that
steps like a king and belongs to nobody. `tests/neutral_oracle.py` is the declared oracle.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables, canonical_tables
from chess_variant_sandbox.engine.compile import CONTROL_MAJORITY, CONTROL_NEUTRAL, compile_spec
from chess_variant_sandbox.engine.moves import (
    MOVE_FOREIGN,
    MOVE_NORMAL,
    encode_move,
    move_foreign,
    move_from,
    move_to,
)
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.schema import validate_spec
from tests.neutral_oracle import NeutralOracle

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}


def _sq(name: str, width: int = WIDTH) -> int:
    return (int(name[1:]) - 1) * width + ord(name[0]) - ord("a")


def _spec(**rules: Any) -> dict[str, Any]:
    """6x6: kings on a1/a6, rooks on f1/f6, guards on b1/b6, condottieri nobody's on c3/c4."""
    return {
        "schema_version": 31,
        "name": "the scrum",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "G", "value": 2, "moves": [KING]},
            {"id": "C", "name": "condottiere", "value": 3, "moves": [KING], "control": "majority"},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "R", "squares": ["f1"]},
            {"piece": "G", "squares": ["b1"]},
            {"piece": "C", "player": "nobody", "squares": ["c3"]},
        ]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 120, **rules},
    }


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


def _position(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0):
    state = engine.make_state(placement, side_to_move=side, moved=list(placement))
    return dataclasses.replace(state, hands=engine.initial_state().hands)


def _foreign(engine: Engine, state) -> list[int]:
    return sorted(m for m in engine.legal_moves(state) if move_foreign(m))


def _word(origin: str, target: str) -> int:
    return encode_move(_sq(origin), _sq(target), MOVE_NORMAL, 0) | MOVE_FOREIGN


# ------------------------------------------------------------------ who holds it


class TestWhoHoldsIt:
    def test_none_against_none_is_nobodys(self):
        engine = _engine(_spec())
        state = engine.initial_state()
        assert engine.controller_of(state, _sq("c3")) == -1
        assert _foreign(engine, state) == []

    def test_a_tie_is_nobodys(self):
        engine = _engine(_spec())
        state = _position(engine, {
            "a1": ("K", 0), "f6": ("K", 1), "c3": ("C", 2), "b2": ("G", 0), "d4": ("G", 1),
        })
        assert engine.controller_of(state, _sq("c3")) == -1
        assert _foreign(engine, state) == []

    def test_strictly_more_holds_it_and_only_on_its_turn(self):
        engine = _engine(_spec())
        placement = {
            "a1": ("K", 0), "f6": ("K", 1), "c3": ("C", 2),
            "b2": ("G", 0), "d2": ("R", 0), "d4": ("G", 1),
        }
        white = _position(engine, placement, side=0)
        black = _position(engine, placement, side=1)
        assert engine.controller_of(white, _sq("c3")) == 0
        assert _foreign(engine, white) and all(move_from(m) == _sq("c3") for m in _foreign(engine, white))
        assert _foreign(engine, black) == []

    def test_a_royal_counts(self):
        engine = _engine(_spec())
        state = _position(engine, {"b2": ("K", 0), "f6": ("K", 1), "c3": ("C", 2)})
        assert engine.controller_of(state, _sq("c3")) == 0
        assert _word("c3", "c4") in _foreign(engine, state)

    def test_mirrored_positions_give_mirrored_holders(self):
        engine = _engine(_spec())
        rng = random.Random(7)
        for _ in range(30):
            squares = rng.sample(range(WIDTH * WIDTH), 9)
            kinds = ["G", "R", "G", "G", "R", "G"]
            placement = {_name(squares[0]): ("K", 0), _name(squares[1]): ("K", 1)}
            placement[_name(squares[2])] = ("C", 2)
            for index, square in enumerate(squares[3:]):
                placement[_name(square)] = (kinds[index], index % 2)
            mirrored = {
                _name(_mirror(_sq(name))): (pid, player if player == 2 else 1 - player)
                for name, (pid, player) in placement.items()
            }
            here = engine.controller_of(_position(engine, placement), _sq(_name(squares[2])))
            there = engine.controller_of(_position(engine, mirrored), _mirror(squares[2]))
            assert there == (-1 if here == -1 else 1 - here)


def _name(square: int) -> str:
    return f"{chr(ord('a') + square % WIDTH)}{square // WIDTH + 1}"


def _mirror(square: int) -> int:
    return (WIDTH - 1 - square // WIDTH) * WIDTH + square % WIDTH


# ------------------------------------------------------------------ what it may do


class TestWhatAHeldNeutralMayDo:
    def _held(self, engine: Engine, extra: dict[str, tuple[str, int]]):
        placement = {"a1": ("K", 0), "c3": ("C", 2), "b2": ("G", 0), **extra}
        return _position(engine, placement)

    def test_it_takes_the_opponents_piece_but_never_a_royal_its_holders_or_nobodys(self):
        engine = _engine(_spec())
        state = self._held(
            engine, {"d4": ("K", 1), "c4": ("R", 1), "d3": ("G", 0), "c2": ("G", 0), "b4": ("C", 2)},
        )
        moves = _foreign(engine, state)
        held_from_c3 = {move_to(m) for m in moves if move_from(m) == _sq("c3")}
        assert _sq("c4") in held_from_c3  # the opponent's rook
        assert _sq("d4") not in held_from_c3  # the opponent's royal
        assert _sq("d3") not in held_from_c3 and _sq("b2") not in held_from_c3  # the holder's
        assert _sq("b4") not in held_from_c3  # nobody's

    def test_moving_it_leaves_it_nobodys(self):
        engine = _engine(_spec())
        state = self._held(engine, {"f6": ("K", 1)})
        after = engine.apply(state, _word("c3", "c4"))
        value = after.board[_sq("c4")]
        assert engine.spec.owner_of(value) == 2 and after.board[_sq("c3")] == 0
        assert engine._compute_hash(after) == after.key

    def test_a_discovered_check_is_illegal_for_the_holders_royal(self):
        engine = _engine(_spec())
        state = _position(engine, {
            "a1": ("K", 0), "a3": ("C", 2), "b2": ("G", 0), "a6": ("R", 1), "f6": ("K", 1),
        })
        assert engine.controller_of(state, _sq("a3")) == 0
        moves = _foreign(engine, state)
        assert moves and all(move_to(m) in (_sq("a2"), _sq("a4")) for m in moves)

    def test_a_discovered_check_on_the_other_royal_is_check(self):
        engine = _engine(_spec())
        state = _position(engine, {
            "a6": ("K", 1), "a3": ("C", 2), "b2": ("G", 0), "a1": ("R", 0), "f1": ("K", 0),
        })
        after = engine.apply(state, _word("a3", "b3"))
        assert engine.in_check(after, 1)

    def test_a_held_neutral_gives_no_check(self):
        engine = _engine(_spec())
        state = _position(engine, {
            "a1": ("K", 0), "c4": ("C", 2), "b3": ("G", 0), "d5": ("K", 1),
        })
        assert not engine.in_check(state, 1)


# ------------------------------------------------------------------ taken


class TestTaken:
    @pytest.mark.parametrize("side", [0, 1])
    def test_either_side_takes_it(self, side):
        engine = _engine(_spec())
        rook = "c1" if side == 0 else "c6"
        state = _position(engine, {
            "a1": ("K", 0), "f6": ("K", 1), "c3": ("C", 2), rook: ("R", side),
        }, side=side)
        take = next(
            m for m in engine.legal_moves(state)
            if move_from(m) == _sq(rook) and move_to(m) == _sq("c3")
        )
        after = engine.apply(state, take)
        assert engine.spec.owner_of(after.board[_sq("c3")]) == side

    def test_taken_it_enters_the_takers_hand(self):
        engine = _engine(_spec(drops={"from_captures": True}))
        state = _position(engine, {
            "a1": ("K", 0), "f6": ("K", 1), "c3": ("C", 2), "c1": ("R", 0),
        })
        take = next(m for m in engine.legal_moves(state) if move_to(m) == _sq("c3"))
        after = engine.apply(state, take)
        assert after.hands[0][3] == 1


# ------------------------------------------------------------------ geometry


class TestNextTo:
    def test_a_wall_is_never_next_to_it(self):
        spec = _spec(obstacles={"squares": ["d3"]})
        compiled = compile_spec(spec)
        assert _sq("d3") not in compiled.neutrals.neighbours[3][_sq("c3")]
        assert _sq("d4") in compiled.neutrals.neighbours[3][_sq("c3")]

    def test_a_named_geometry_counts_across_a_wall(self):
        spec = _spec(obstacles={"squares": ["d3"]})
        spec["pieces"][3]["control"] = {"majority": {"vectors": [[2, 0]], "symmetry": "rotate4"}}
        engine = _engine(spec)
        state = _position(engine, {"a1": ("K", 0), "f6": ("K", 1), "c3": ("C", 2), "e3": ("G", 0)})
        assert engine.controller_of(state, _sq("c3")) == 0
        state = _position(engine, {"a1": ("K", 0), "f6": ("K", 1), "c3": ("C", 2), "d4": ("G", 0)})
        assert engine.controller_of(state, _sq("c3")) == -1

    def test_a_wrap_counts_across_the_seam(self):
        spec = _spec(wrap={"files": "both"})
        spec["setup"]["placements"][3]["squares"] = ["a3"]
        spec["setup"]["placements"][0]["squares"] = ["b1"]
        spec["setup"]["placements"][2]["squares"] = ["c1"]
        engine = _engine(spec)
        state = _position(engine, {"b1": ("K", 0), "d6": ("K", 1), "a3": ("C", 2), "f3": ("G", 0)})
        assert engine.controller_of(state, _sq("a3")) == 0
        assert _word("a3", "f2") in _foreign(engine, state)


# ------------------------------------------------------------------ the oracle


def _walk(spec: dict[str, Any], seed: int, plies: int) -> int:
    engine = _engine(spec)
    oracle = NeutralOracle(spec, engine)
    rng = random.Random(seed)
    state = engine.initial_state()
    held = 0
    for _ in range(plies):
        if engine.terminal(state) is not None:
            break
        core = sorted(engine.legal_moves(state))
        assert core == oracle.legal_moves(state), (seed, state.ply)
        for square in range(WIDTH * WIDTH):
            assert engine.controller_of(state, square) == oracle.holder(state, square)
        assert engine._compute_hash(state) == state.key
        foreign = [m for m in core if move_foreign(m)]
        held += bool(foreign)
        # Lean toward the neutrals so the walk spends time in the scrum.
        move = rng.choice(foreign) if foreign and rng.random() < 0.5 else rng.choice(core)
        state = engine.apply(state, move)
    return held


def _walk_spec() -> dict[str, Any]:
    spec = _spec()
    spec["setup"]["placements"] = [
        {"piece": "K", "squares": ["c1"]},
        {"piece": "R", "squares": ["a1", "f1"]},
        {"piece": "G", "squares": ["a2", "b2", "e2", "f2"]},
        {"piece": "C", "player": "nobody", "squares": ["b3", "e3"]},
    ]
    return spec


class TestTheOracle:
    def test_three_walks_agree(self):
        assert sum(_walk(_walk_spec(), seed, 60) for seed in range(3)) > 0

    @pytest.mark.slow
    def test_twenty_walks_agree(self):
        assert sum(_walk(_walk_spec(), seed, 120) for seed in range(20)) > 20


# ------------------------------------------------------------------ the spec


class TestTheSpec:
    def test_the_tables_cross_only_for_a_neutral_game(self):
        tables = build_tables(compile_spec(_spec()))
        assert tables["neutral_control"] == [0, 0, 0, CONTROL_MAJORITY]
        assert len(tables["neighbour_index"]) == 4 * WIDTH * WIDTH + 1
        assert "neutral_control" in canonical_tables(compile_spec(_spec()))
        standing = _spec()
        del standing["pieces"][3]["control"]
        tables = build_tables(compile_spec(standing))
        assert tables["neutral_control"] == [0, 0, 0, CONTROL_NEUTRAL]
        assert tables["neighbour_index"] == []
        none = _spec()
        none["setup"]["placements"].pop()
        del none["pieces"][3]["control"]
        tables = build_tables(compile_spec(none))
        assert tables["neutral_control"] == [] and tables["control_kind"] == []

    def test_stands_is_the_off_value(self):
        plain = _spec()
        del plain["pieces"][3]["control"]
        written = copy.deepcopy(plain)
        written["pieces"][3]["control"] = "stands"
        assert validate_spec(written) == []
        assert spec_hash(written) == spec_hash(plain)

    def test_a_standing_neutral_is_never_moved(self):
        spec = _spec()
        del spec["pieces"][3]["control"]
        engine = _engine(spec)
        state = _position(engine, {"a1": ("K", 0), "f6": ("K", 1), "c3": ("C", 2), "b2": ("G", 0)})
        assert engine.controller_of(state, _sq("c3")) == -1 and _foreign(engine, state) == []

    def test_the_perft_is_frozen(self):
        engine = _engine(_walk_spec())
        assert [engine.perft(engine.initial_state(), depth) for depth in (1, 2, 3)] == PERFT

    def test_fairy_stockfish_refuses_it_by_name(self):
        held = _spec()
        held["rules"]["win"] = "checkmate"
        with pytest.raises(Untranslatable) as refused:
            translate(held)
        reasons = " ".join(refused.value.reasons)
        assert "setup.placements.player" in reasons and "pieces.control" in reasons
        standing = _spec()
        del standing["pieces"][3]["control"]
        with pytest.raises(Untranslatable) as refused:
            translate(standing)
        reasons = " ".join(refused.value.reasons)
        assert "setup.placements.player" in reasons and "pieces.control" not in reasons

    def test_the_words_say_it(self):
        text = describe_spec(_spec())
        assert "belong to nobody" in text and "strictly more of their own pieces" in text
        assert "on a tie, none against none included, nobody may move it" in text


# Frozen at TC3 (2026-10-27), when the oracle walks first agreed with the core.
PERFT = [34, 1176, 40377]


def _refused(spec: dict[str, Any], words: str) -> None:
    errors = validate_spec(spec)
    assert any(words in error for error in errors), errors


class TestRefusals:
    def test_control_on_a_royal(self):
        spec = _spec()
        spec["pieces"][0]["control"] = "majority"
        _refused(spec, "on a royal")

    def test_majority_on_a_type_nobody_holds(self):
        spec = _spec()
        spec["pieces"][2]["control"] = "majority"
        _refused(spec, "no placement puts in nobody's block")

    def test_a_royal_for_nobody(self):
        spec = _spec()
        spec["setup"]["placements"].append({"piece": "K", "player": "nobody", "squares": ["e3"]})
        _refused(spec, "a royal always belongs to a player")

    def test_a_spy_for_nobody(self):
        spec = _spec()
        spec["pieces"][2]["spy"] = True
        spec["setup"]["placements"].append({"piece": "G", "player": "nobody", "squares": ["e3"]})
        _refused(spec, "a spy is never gained")

    @pytest.mark.parametrize("key, value", [
        ("simultaneous", {}),
        ("strike", {"effect": "capture"}),
        ("npcs", [{"piece": "G", "route": ["e3", "e4"]}]),
    ])
    def test_beside_a_rule_that_shares_the_block_or_the_capture(self, key, value):
        spec = _spec(**{key: value})
        _refused(spec, f"a placement for nobody is refused beside rules.{key}")

    @pytest.mark.parametrize("key, value", [
        ("territory", {"repaint": True}),
        ("reach", {"piece": "K", "squares": ["f6"]}),
        ("explode_on_capture", True),
    ])
    def test_majority_beside(self, key, value):
        _refused(_spec(**{key: value}), f"majority control is refused beside rules.{key}")

    def test_a_long_turn(self):
        _refused(_spec(turn={"moves": 2}), "a turn longer than one move")

    def test_a_majority_type_that_reads_unalike(self):
        spec = _spec()
        spec["pieces"][3]["moves"] = [{"kind": "leap", "vectors": [[0, 1]]}]
        _refused(spec, "reads differently for the two players")

    def test_a_majority_type_that_faces_or_promotes(self):
        spec = _spec(promotion={"piece": "C", "rank": 5, "to": ["R"]})
        _refused(spec, "never promotes")

    def test_a_nobodys_square_under_a_players_piece_or_its_image(self):
        spec = _spec()
        spec["setup"]["placements"][3]["squares"] = ["b6"]
        _refused(spec, "lands on a square a player's piece starts on")

    def test_the_second_player_may_not_be_named_but_nobody_may(self):
        spec = _spec()
        assert validate_spec(spec) == []
        spec["setup"]["placements"].append({"piece": "G", "player": "somebody", "squares": ["e3"]})
        _refused(spec, "must be 0, 1 or 'nobody'")


def test_the_mirror_puts_nobodys_image_in_nobodys_block():
    engine = _engine(_spec())
    state = engine.initial_state()
    for name in ("c3", "c4"):
        assert engine.spec.owner_of(state.board[_sq(name)]) == 2
        assert not state.unmoved >> _sq(name) & 1
