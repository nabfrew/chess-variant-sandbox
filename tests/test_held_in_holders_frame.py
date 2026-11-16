"""SR8i (`docs/plan-schema-next.md` §2.11, *the mover's reading*; O9, O13, O14, O17, O18,
O19): a held piece moves in its holder's frame.

A piece nobody owns has no side, so whatever about it names a side is read, on the move a
player makes with it, as if that player owned it: its forward, ranks, region and screens are
its holder's, a facing is its arrow on the board (written in nobody's frame, the first
player's), and a first move is its own mark. It stays nobody's after the move and still gives
no check. The core builds the holder's frame per player (`compile.CompiledSpec.held_frames`,
read by `foreign.rs::push_held`); `tests/neutral_oracle.py` reads the same moves off the
holder's own twin and is held to the core here, list by list and by perft.

Hand-built 6x6 specs, not chess.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from pathlib import Path
from typing import Any

import pytest
import tomllib

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import (
    MOVE_NORMAL,
    MOVE_ROTATE,
    move_foreign,
    move_from,
    move_promotion,
    move_to,
    move_type,
)
from chess_variant_sandbox.spec import refusals
from chess_variant_sandbox.spec.schema import validate_spec
from tests.neutral_oracle import NeutralOracle

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
PAWN = [
    {"kind": "leap", "vectors": [[0, 1]], "mode": "move"},
    {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
]


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _spec(held: dict[str, Any], **rules: Any) -> dict[str, Any]:
    """Kings a1/a6, guards b1/b6, rooks f1/f6, and the held type ``H`` nobody's on c3/c4."""
    return {
        "schema_version": 31,
        "name": "the holder's frame",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "G", "value": 2, "moves": [KING]},
            {"id": "H", "value": 3, "control": "majority", **held},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "R", "squares": ["f1"]},
            {"piece": "G", "squares": ["b1"]},
            {"piece": "H", "player": "nobody", "squares": ["c3"]},
        ]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 80, **rules},
    }


def _pawn() -> dict[str, Any]:
    return _spec({"moves": copy.deepcopy(PAWN)})


def _arrow() -> dict[str, Any]:
    """A held piece that faces: one step along its arrow, turned by its holder."""
    spec = _spec({
        "moves": [{"kind": "leap", "vectors": [[0, 1]]}],
        "facing": {"axes": 4, "initial": "forward"},
    })
    spec["setup"]["placements"][3]["facing"] = "right"
    return spec


def _regioned() -> dict[str, Any]:
    return _spec({"moves": [copy.deepcopy(ROOK)], "region": [
        f"{file}{rank}" for file in "abcdef" for rank in (1, 2, 3, 4)
    ]})


def _double_step() -> dict[str, Any]:
    """A pawn that steps two from its holder's second rank, and two on its first move."""
    moves = copy.deepcopy(PAWN) + [
        {"kind": "slide", "vectors": [[0, 1]], "range": 2, "mode": "move",
         "condition": {"from_rank": 1}},
        {"kind": "leap", "vectors": [[1, 2]], "symmetry": "lateral", "mode": "move",
         "condition": {"first_move": True}},
    ]
    spec = _spec({"moves": moves})
    spec["setup"]["placements"][3]["squares"] = ["c2"]
    return spec


def _hopper(screen: str) -> dict[str, Any]:
    return _spec({"moves": [{
        "kind": "hop", "vectors": [[1, 0]], "symmetry": "rotate4", "screen": screen,
    }]})


def _taker() -> dict[str, Any]:
    """A held pawn that takes royals: its attack is its holder's frame's too."""
    spec = _pawn()
    spec["pieces"][3]["takes_royal"] = True
    return spec


ALL_SPECS = {
    "pawn": _pawn, "arrow": _arrow, "region": _regioned, "double step": _double_step,
    "own screen": lambda: _hopper("own"), "enemy screen": lambda: _hopper("enemy"),
    "takes royal": _taker,
}


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


def _position(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0, moved=None):
    state = engine.make_state(
        placement, side_to_move=side, moved=list(placement) if moved is None else moved
    )
    return dataclasses.replace(state, hands=engine.initial_state().hands)


def _held_from(engine: Engine, state, square: str) -> list[int]:
    return sorted(
        m for m in engine.legal_moves(state) if move_foreign(m) and move_from(m) == _sq(square)
    )


def _targets(engine: Engine, state, square: str) -> set[int]:
    return {move_to(m) for m in _held_from(engine, state, square) if move_type(m) == MOVE_NORMAL}


# A holder for each side: a guard of theirs next to c3, the kings far away.
HOLDERS = {
    0: {"a1": ("K", 0), "f6": ("K", 1), "b2": ("G", 0)},
    1: {"a1": ("K", 0), "f6": ("K", 1), "b4": ("G", 1)},
}


class TestTheRefusalsGo:
    @pytest.mark.parametrize("name", sorted(ALL_SPECS))
    def test_each_spec_validates(self, name):
        assert validate_spec(ALL_SPECS[name]()) == []

    def test_the_six_rows_are_lifted(self):
        table = tomllib.loads(
            (Path(refusals.__file__).parent / "refusals.toml").read_text(encoding="utf-8")
        )
        six = {"O9", "O13", "O14", "O17", "O18", "O19"}
        assert not six & {row["id"] for row in table["refusal"]}
        assert six <= {row["id"] for row in table["lifted"] if row.get("by") == "SR8i"}

    def test_a_standing_nobodys_piece_placed_with_a_facing_validates(self):
        """O9 alone: a piece nobody owns, standing (no control), placed pointing right."""
        spec = _arrow()
        del spec["pieces"][3]["control"]
        assert validate_spec(spec) == []

    def test_a_sweep_and_a_promotion_are_lifted(self):
        """O15 was lifted by SR8j (`tests/test_held_promotes.py`), O16 by SR8k
        (`tests/test_held_sweep.py`)."""
        spec = _pawn()
        spec["rules"]["promotion"] = {"piece": "H", "rank": 5, "to": ["R"]}
        assert validate_spec(spec) == []
        spec["pieces"][3]["moves"] = [{"kind": "sweep", "vectors": [[0, 1]], "mode": "both"}]
        assert validate_spec(spec) == []


class TestAlikeGamesKeepFrameZero:
    def test_a_type_that_reads_alike_crosses_no_held_frames(self):
        spec = _spec({"moves": [copy.deepcopy(KING)]})
        assert build_tables(compile_spec(spec))["neutral_held_frames"] == []

    def test_a_directional_type_crosses_them(self):
        assert build_tables(compile_spec(_pawn()))["neutral_held_frames"] == [0, 1]


class TestForward:
    @pytest.mark.parametrize("holder, ahead, behind", [(0, "c4", "c2"), (1, "c2", "c4")])
    def test_a_held_pawn_advances_toward_its_holders_opponent(self, holder, ahead, behind):
        engine = _engine(_pawn())
        state = _position(engine, {**HOLDERS[holder], "c3": ("H", 2)}, side=holder)
        assert engine.controller_of(state, _sq("c3")) == holder
        targets = _targets(engine, state, "c3")
        assert _sq(ahead) in targets and _sq(behind) not in targets

    @pytest.mark.parametrize("holder, prey", [(0, "d4"), (1, "d2")])
    def test_it_captures_forward_for_its_holder(self, holder, prey):
        engine = _engine(_pawn())
        # The prey stands next to it too, so a second guard keeps the count the holder's.
        state = _position(engine, {
            **HOLDERS[holder], "c3": ("H", 2), prey: ("R", 1 - holder), "b3": ("G", holder),
        }, side=holder)
        assert _sq(prey) in _targets(engine, state, "c3")

    def test_it_stays_nobodys_and_gives_no_check(self):
        engine = _engine(_pawn())
        # Its capture diagonal from c4 reaches d5, where the second player's king stands.
        state = _position(engine, {
            "a1": ("K", 0), "d5": ("K", 1), "b3": ("G", 0), "b4": ("G", 0), "c4": ("H", 2),
        })
        assert engine.controller_of(state, _sq("c4")) == 0
        assert not engine.in_check(state, 1)
        step = _held_from(engine, state, "c4")[0]
        after = engine.apply(state, step)
        assert engine.spec.owner_of(after.board[move_to(step)]) == 2


class TestFacing:
    def test_a_placement_for_nobody_stands_at_its_facing_and_its_image_reflected(self):
        engine = _engine(_arrow())
        spec = engine.spec
        board = engine.initial_state().board
        assert spec.owner_of(board[_sq("c3")]) == 2 and spec.facing_of(board[_sq("c3")]) == 1
        # Under the mirror, right stays right on the board: nobody's index 1 again.
        assert spec.facing_of(board[_sq("c4")]) == 1

    @pytest.mark.parametrize("holder", [0, 1])
    def test_it_moves_by_its_arrow_for_either_holder(self, holder):
        engine = _engine(_arrow())
        state = _position(engine, {**HOLDERS[holder], "c3": ("H", 2)}, side=holder)
        state = _facing(engine, state, "c3", 1)  # right, in nobody's frame
        assert _targets(engine, state, "c3") == {_sq("d3")}

    @pytest.mark.parametrize("holder", [0, 1])
    def test_its_holder_may_turn_it_and_it_stays_nobodys(self, holder):
        engine = _engine(_arrow())
        state = _position(engine, {**HOLDERS[holder], "c3": ("H", 2)}, side=holder)
        turns = {
            move_promotion(m): m for m in _held_from(engine, state, "c3")
            if move_type(m) == MOVE_ROTATE
        }
        assert set(turns) == {1, 2, 3}  # every facing but forward, named in nobody's frame
        after = engine.apply(state, turns[2])
        value = after.board[_sq("c3")]
        assert engine.spec.owner_of(value) == 2 and engine.spec.facing_of(value) == 2
        assert engine._compute_hash(after) == after.key
        # Pointing backward (towards the first player), it steps down for either holder.
        for other in (0, 1):
            again = _position(engine, {**HOLDERS[other], "c3": ("H", 2)}, side=other)
            again = _facing(engine, again, "c3", 2)
            assert _targets(engine, again, "c3") == {_sq("c2")}


def _facing(engine: Engine, state, square: str, facing: int):
    """``state`` with the nobody's piece on ``square`` turned to ``facing`` (nobody's frame)."""
    spec = engine.spec
    board = bytearray(state.board)
    value = board[_sq(square)]
    board[_sq(square)] = (2 * spec.n_facings + facing) * spec.n_types + spec.type_of(value) + 1
    state = dataclasses.replace(state, board=bytes(board))
    return dataclasses.replace(state, key=engine._compute_hash(state))


class TestRegion:
    @pytest.mark.parametrize("holder, inside, outside", [(0, "c4", "c5"), (1, "c5", "c2")])
    def test_it_lands_only_in_its_holders_region(self, holder, inside, outside):
        engine = _engine(_regioned())
        state = _position(engine, {**HOLDERS[holder], "c3": ("H", 2)}, side=holder)
        # From c3 up the file: the first player's region is ranks 1-4, the second's 3-6.
        targets = _targets(engine, state, "c3")
        assert _sq(inside) in targets and _sq(outside) not in targets


class TestConditions:
    def test_from_rank_counts_from_the_holders_back_rank(self):
        engine = _engine(_double_step())
        first = _position(engine, {"a1": ("K", 0), "f6": ("K", 1), "b1": ("G", 0), "c2": ("H", 2)})
        assert _sq("c4") in _targets(engine, first, "c2")
        second = _position(
            engine, {"a1": ("K", 0), "f6": ("K", 1), "b6": ("G", 1), "c5": ("H", 2)}, side=1,
        )
        assert _sq("c3") in _targets(engine, second, "c5")
        # On the first player's second rank, the second player is not on theirs.
        wrong = _position(
            engine, {"a1": ("K", 0), "f6": ("K", 1), "b1": ("G", 1), "c2": ("H", 2)}, side=1,
        )
        assert _sq("c4") not in _targets(engine, wrong, "c2")

    def test_first_move_reads_the_pieces_own_mark(self):
        engine = _engine(_double_step())
        state = engine.initial_state()
        assert state.unmoved >> _sq("c2") & 1 and state.unmoved >> _sq("c5") & 1
        placement = {"a1": ("K", 0), "f6": ("K", 1), "b1": ("G", 0), "c2": ("H", 2)}
        fresh = _position(engine, placement, moved=["a1", "f6", "b1"])
        moved = _position(engine, placement)
        assert _sq("d4") in _targets(engine, fresh, "c2")
        assert _sq("d4") not in _targets(engine, moved, "c2")


class TestScreens:
    @pytest.mark.parametrize("screen, holder, over, hops", [
        ("own", 0, ("G", 0), True), ("own", 0, ("G", 1), False), ("own", 0, ("H", 2), False),
        ("enemy", 0, ("G", 1), True), ("enemy", 0, ("G", 0), False),
        ("enemy", 0, ("H", 2), False), ("any", 0, ("H", 2), True),
        ("own", 1, ("G", 1), True), ("enemy", 1, ("G", 0), True), ("enemy", 1, ("H", 2), False),
    ])
    def test_own_is_the_holders_and_a_nobodys_piece_screens_only_under_any(
        self, screen, holder, over, hops,
    ):
        engine = _engine(_hopper(screen))
        state = _position(engine, {**HOLDERS[holder], "c3": ("H", 2), "e3": over}, side=holder)
        assert (_sq("f3") in _targets(engine, state, "c3")) is hops


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
        for player in (0, 1):
            assert engine.in_check(state, player) == oracle.attacked(state, player)
        assert engine._compute_hash(state) == state.key
        foreign = [m for m in core if move_foreign(m)]
        held += bool(foreign)
        move = rng.choice(foreign) if foreign and rng.random() < 0.6 else rng.choice(core)
        state = engine.apply(state, move)
    return held


def _walk_spec(make) -> dict[str, Any]:
    spec = make()
    nobodys = spec["setup"]["placements"][3]
    spec["setup"]["placements"] = [
        {"piece": "K", "squares": ["c1"]},
        {"piece": "R", "squares": ["a1", "f1"]},
        {"piece": "G", "squares": ["a2", "b2", "e2", "f2"]},
        {**nobodys, "squares": ["b3", "e3"]},
    ]
    return spec


def _oracle_perft(engine: Engine, oracle: NeutralOracle, state, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(
        _oracle_perft(engine, oracle, engine.apply(state, m), depth - 1)
        for m in oracle.legal_moves(state)
    )


class TestTheOracle:
    @pytest.mark.parametrize("name", sorted(ALL_SPECS))
    def test_walks_agree(self, name):
        spec = _walk_spec(ALL_SPECS[name])
        assert sum(_walk(spec, seed, 40) for seed in range(2)) > 0

    @pytest.mark.parametrize("name", sorted(ALL_SPECS))
    def test_a_perft_agrees(self, name):
        """The core's perft(3) from a position where both sides hold a piece, against the
        oracle's legal lists walked through `apply`."""
        spec = _walk_spec(ALL_SPECS[name])
        engine = _engine(spec)
        oracle = NeutralOracle(spec, engine)
        state = engine.initial_state()
        assert engine.perft(state, 3) == _oracle_perft(engine, oracle, state, 3)

    @pytest.mark.slow
    @pytest.mark.parametrize("name", sorted(ALL_SPECS))
    def test_long_walks_agree(self, name):
        spec = _walk_spec(ALL_SPECS[name])
        assert sum(_walk(spec, seed, 100) for seed in range(10)) > 10
