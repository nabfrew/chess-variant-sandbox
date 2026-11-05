"""SR6d (`docs/plan-schema-next.md` §2.6 row W6): a sweep on a wrapped board, with a declared path.

The ruling (2026-10-27, the user: *"Have the player declare the path if ambiguous"*): where two
of a piece's sweep rays from one origin reach one target, each is its own move. The move word
carries the ray (its index among the slot's sweep entries, in the tag, `moves.move_ray`), and
`Tables::sweep_path` reads it instead of scanning for the first ray; the text tells the two
apart by its `(sweeps ...)` list. Two rays that would remove the same pieces are one move, the
lower ray's. Each ray is legal, blocked or check on its own.

The word names its ray only in a game where two rays of some slot reach one square along
different paths (`CompiledSpec.sweep_rays`, which the core derives from the same table): every
other sweep game, and every record stored before, keeps its words. `tests/sweep_oracle.py`
states the cylinder again by its own means. Hand-built 6x6 specs, not chess.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import sweep_oracle

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import rust_supports
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import (
    MAX_SWEEP_RAYS,
    MOVE_SWEEP,
    decode_move,
    move_from,
    move_ray,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import load_spec, validate_spec
from chess_variant_sandbox.spec.refusals import refused_ids

ROOT = Path(__file__).resolve().parent.parent
WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
ROOK = {"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}
IDS = ("X", "K", "P", "B")


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _spec(wrap: bool = True, sweep: dict[str, Any] = ROOK, **rules: Any) -> dict[str, Any]:
    """A rook-like sweeper X, kings K, a pawn-like P and a block B that X may not take."""
    spec: dict[str, Any] = {
        "name": "the sweep on a cylinder (SR6d test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "X", "name": "sweeper", "value": 5, "moves": [copy.deepcopy(sweep)],
             "captures": ["X", "K", "P"]},
            {"id": "K", "name": "king", "royal": True, "value": 0, "moves": [KING]},
            {"id": "P", "name": "pawn", "value": 1,
             "moves": [{"kind": "leap", "vectors": [[0, 1]], "mode": "move"}]},
            {"id": "B", "name": "block", "value": 1, "moves": [KING]},
        ],
        "setup": {"mirror": True, "placements": [
            {"piece": "X", "squares": ["a1", "f1"]},
            {"piece": "K", "squares": ["c1"]},
            {"piece": "P", "squares": ["b2", "c2", "e2"]},
            {"piece": "B", "squares": ["d2"]},
        ]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 80, **rules},
    }
    if wrap:
        spec["rules"]["wrap"] = {"files": "both"}
    return spec


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


# The sweeper on a3; the rank's other squares hold an enemy pawn on c3 and e3. East of a3 the ray
# is b3, c3, d3, e3, f3; west it is f3, e3, d3, c3, b3: two paths to every square of the rank.
KINGS = {"a1": ("K", 0), "d6": ("K", 1)}
ROW = {**KINGS, "a3": ("X", 0), "c3": ("P", 1), "e3": ("P", 1)}


def _state(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0) -> GameState:
    return engine.make_state(placement, side_to_move=side, moved=list(placement))


def _texts_to(engine: Engine, state: GameState, origin: str, target: str) -> list[str]:
    return [
        engine.describe_move(state, m) for m in engine.legal_moves(state)
        if (move_from(m), move_to(m)) == (_sq(origin), _sq(target))
    ]


class TestTwoRaysToOneTarget:
    def test_two_moves_where_they_sweep_different_pieces(self) -> None:
        """d3 east of a3 sweeps c3, west of it e3: two moves, told apart by their lists."""
        engine = _engine(_spec())
        state = _state(engine, ROW)
        assert sorted(_texts_to(engine, state, "a3", "d3")) == [
            "Xa3xd3 (sweeps c3)", "Xa3xd3 (sweeps e3)",
        ]

    def test_one_move_where_they_sweep_the_same(self) -> None:
        """An empty rank: both rays reach d3 over nothing, which is one plain move."""
        engine = _engine(_spec())
        state = _state(engine, {**KINGS, "a3": ("X", 0)})
        assert _texts_to(engine, state, "a3", "d3") == ["Xa3-d3"]

    def test_a_ray_over_nothing_and_a_ray_over_a_piece_are_two(self) -> None:
        """c3 east of a3 is a capture over an empty b3; west of it a sweep of e3."""
        engine = _engine(_spec())
        state = _state(engine, ROW)
        assert sorted(_texts_to(engine, state, "a3", "c3")) == ["Xa3xc3", "Xa3xc3 (sweeps e3)"]

    def test_find_move_round_trips_both_texts(self) -> None:
        engine = _engine(_spec())
        state = _state(engine, ROW)
        east = engine.find_move(state, "Xa3xd3 (sweeps c3)")
        west = engine.find_move(state, "Xa3xd3 (sweeps e3)")
        assert east != west and move_type(east) == move_type(west) == MOVE_SWEEP
        assert engine.describe_move(state, east) == "Xa3xd3 (sweeps c3)"
        assert engine.describe_move(state, west) == "Xa3xd3 (sweeps e3)"
        assert move_ray(east) != move_ray(west)

    def test_each_ray_removes_its_own_path(self) -> None:
        engine = _engine(_spec())
        state = _state(engine, ROW)
        east = engine.apply(state, engine.find_move(state, "Xa3xd3 (sweeps c3)"))
        west = engine.apply(state, engine.find_move(state, "Xa3xd3 (sweeps e3)"))
        spec = engine.spec
        assert east.board[_sq("c3")] == 0 and spec.type_of(east.board[_sq("e3")]) == 2
        assert west.board[_sq("e3")] == 0 and spec.type_of(west.board[_sq("c3")]) == 2
        for after in (east, west):
            assert spec.type_of(after.board[_sq("d3")]) == 0 and after.board[_sq("a3")] == 0
            assert after.key == engine._compute_hash(after)

    def test_a_block_on_one_ray_leaves_the_other(self) -> None:
        """A block X may not take on b3 ends the east ray; d3 is still reached going west."""
        engine = _engine(_spec())
        state = _state(engine, {**ROW, "b3": ("B", 1)})
        assert _texts_to(engine, state, "a3", "d3") == ["Xa3xd3 (sweeps e3)"]

    def test_two_lengths_along_one_vector_are_one_ray(self) -> None:
        """Q1063: a sweep of length 2 and one of the whole line share their squares, so on a
        flat board they reach no square two ways, and the words carry no ray."""
        sweep = {"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4", "sweep_range": 2}
        spec = _spec(wrap=False)
        spec["pieces"][0]["moves"].append(copy.deepcopy(ROOK))
        spec["pieces"][0]["moves"][0] = sweep
        engine = _engine(spec)
        assert not engine.spec.sweep_rays
        state = _state(engine, ROW)
        assert _texts_to(engine, state, "a3", "d3") == ["Xa3xd3 (sweeps c3)"]


class TestCheck:
    @pytest.mark.parametrize(("blocked", "checked"), [
        (None, True), ("b3", True), ("f3", True), (("b3", "f3"), False),
    ])
    def test_check_along_either_ray(self, blocked: Any, checked: bool) -> None:
        """The enemy king on d3, pawns on c3 and e3: the sweeper on a3 gives check along either
        ray, and a block on one ray leaves the other's check."""
        engine = _engine(_spec())
        placement = {"a1": ("K", 0), "d3": ("K", 1), "a3": ("X", 0),
                     "c3": ("P", 1), "e3": ("P", 1)}
        for square in (blocked,) if isinstance(blocked, str) else blocked or ():
            placement[square] = ("B", 1)
        state = _state(engine, placement, side=1)
        assert engine.in_check(state) is checked

    def test_a_ray_that_sweeps_the_own_king_is_not_offered(self) -> None:
        """The mover's own king on e3: the west ray to d3 would sweep it, the east one not."""
        engine = _engine(_spec())
        state = _state(engine, {"e3": ("K", 0), "d6": ("K", 1), "a3": ("X", 0), "c3": ("P", 1)})
        assert _texts_to(engine, state, "a3", "d3") == ["Xa3xd3 (sweeps c3)"]


class TestBesideAHand:
    def test_each_ray_sends_its_own_pieces_to_the_hand(self) -> None:
        """The mover's own pawn on e3: west of a3 it is swept to no hand, east the enemy's goes."""
        engine = _engine(_spec(drops={"from_captures": True}))
        state = _state(engine, {**KINGS, "a3": ("X", 0), "c3": ("P", 1), "e3": ("P", 0)})
        # `make_state` leaves the hands empty; a hand game's position holds one slot per type.
        state = dataclasses.replace(state, hands=engine.initial_state().hands)
        east = engine.apply(state, engine.find_move(state, "Xa3xd3 (sweeps c3)"))
        west = engine.apply(state, engine.find_move(state, "Xa3xd3 (sweeps e3)"))
        assert sum(east.hands[0]) == 1 and sum(west.hands[0]) == 0


class TestTheWord:
    def test_the_flag_reads_the_tables(self) -> None:
        assert compile_spec(_spec()).sweep_rays
        assert not compile_spec(_spec(wrap=False)).sweep_rays

    @pytest.mark.parametrize("path", [
        "archive/chess-sweeping-bishops.json", "archive/short-sweeping-bishop-chess.json",
    ])
    def test_a_stored_sweep_game_names_no_ray(self, path: str) -> None:
        """Every record stored before SR6d holds sweep words with no ray, and still reads."""
        engine = Engine(load_spec(ROOT / path))
        assert not engine.spec.sweep_rays
        state = engine.initial_state()
        rng = random.Random(7)
        for _ in range(60):
            moves = engine.legal_moves(state)
            if not moves or engine.terminal(state, moves) is not None:
                break
            assert all(move_ray(m) == 0 for m in moves if move_type(m) == MOVE_SWEEP)
            state = engine.apply(state, rng.choice(moves))

    def test_too_many_rays_to_name_are_refused(self) -> None:
        """A torus sweeper with forty rays from a square: the word's five bits name 32."""
        sweep = {"kind": "sweep", "vectors": [[1, 0], [1, 1], [2, 1], [3, 1], [3, 2], [4, 1]],
                 "symmetry": "dihedral8"}
        spec = _spec(sweep=sweep)
        spec["board"] = {"width": 12, "height": 12}
        spec["rules"]["wrap"] = {"files": "both", "ranks": "both"}
        spec["setup"]["placements"] = [
            {"piece": "X", "squares": ["a1"]}, {"piece": "K", "squares": ["f1"]},
        ]
        assert validate_spec(spec) == []
        why = rust_supports(compile_spec(spec))
        assert why is not None and str(MAX_SWEEP_RAYS) in why

    def test_wrap_beside_a_sweep_is_no_longer_refused(self) -> None:
        assert "W6" not in refused_ids(validate_spec(_spec()))
        assert validate_spec(_spec()) == []


def _count(engine: Engine, position: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(_count(engine, engine.apply(position, m), depth - 1)
               for m in engine.legal_moves(position))


def _taken(engine: Engine, state: GameState, move: int) -> frozenset[int]:
    swept = {square for square, _ in engine.spec.sweep_victims(state.board, move)}
    target = move_to(move)
    return frozenset(swept | ({target} if state.board[target] else set()))


class TestAgainstTheOracle:
    @pytest.mark.parametrize("seed", range(4))
    def test_the_core_agrees_with_the_oracle_over_random_games(self, seed: int) -> None:
        spec = _spec()
        spec["pieces"][0].pop("captures")
        spec["pieces"].pop()
        spec["setup"]["placements"].pop()
        engine = _engine(spec)
        oracle = sweep_oracle.Oracle(spec)
        n_types = len(spec["pieces"])
        rng = random.Random(seed)
        state = engine.initial_state()
        for _ in range(50):
            moves = engine.legal_moves(state)
            pos = sweep_oracle.position_of(state.board, state.side_to_move, state.unmoved, n_types)
            mine = Counter((*decode_move(m)[:2], _taken(engine, state, m)) for m in moves)
            theirs = Counter((o, t, taken) for (o, t, _, _), taken in oracle.legal_taken(pos))
            assert mine == theirs, (seed, state.ply)
            if not moves or engine.terminal(state, moves) is not None:
                break
            sweeps = [m for m in moves if move_type(m) == MOVE_SWEEP]
            state = engine.apply(state, rng.choice(sweeps if sweeps and rng.random() < 0.7 else moves))

    def test_perft_against_the_oracle_and_apply(self) -> None:
        spec = _spec()
        spec["pieces"][0].pop("captures")
        spec["pieces"].pop()
        spec["setup"]["placements"].pop()
        engine = _engine(spec)
        state = _state(engine, ROW)
        pos = sweep_oracle.position_of(state.board, 0, state.unmoved, len(spec["pieces"]))
        oracle = sweep_oracle.Oracle(spec)
        assert engine.perft(state, 2) == oracle.perft(pos, 2) == _count(engine, state, 2)
        assert engine.perft(state, 3) == _count(engine, state, 3)

    def test_walks_keep_the_key(self) -> None:
        engine = _engine(_spec())
        for seed in range(3):
            rng = random.Random(seed)
            state = engine.initial_state()
            for _ in range(40):
                moves = engine.legal_moves(state)
                if not moves or engine.terminal(state, moves) is not None:
                    break
                sweeps = [m for m in moves if move_type(m) == MOVE_SWEEP]
                move = rng.choice(sweeps if sweeps and rng.random() < 0.8 else moves)
                board = bytes(state.board)
                after = engine.apply(state, move)
                assert bytes(state.board) == board
                assert after.key == engine._compute_hash(after)
                state = after
