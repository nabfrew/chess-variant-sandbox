"""SR6e (`docs/plan-schema-next.md` §2.6 row W8): a sweep beside portals.

The ruling (2026-10-27, the user: *"a fold type portal should be able to propagate a sweep"*):
a sweep whose landing square is an empty fold that lets the sweeper in may stop there or go
through and land on the end it leads to, as any go-through (onto an empty square, or onto an
enemy it may take where the landing could capture, never its own piece); the pieces it passed
before the fold are swept as usual, nothing beyond the exit is swept, one fold per move. A
sweep passing over a fold is unchanged by it: a piece standing in a fold is swept as anywhere.
A momentum end is a piece nobody may take to a sweep: it ends the ray, and no sweep transits.

The word of a sweep through a fold is a `MOVE_SWEEP` to the far square naming its ray in the
tag and its step on the ray in the promotion field (`moves.move_through`), so it reads without
the position, and the pair of a declared ray (SR6d, a cylinder) and a fold works.
`tests/sweep_oracle.py` states the folds again by its own means. Hand-built 6x6 specs, not
chess.
"""

from __future__ import annotations

import copy
import random
from collections import Counter
from typing import Any

import pytest
import sweep_oracle

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import rust_supports
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import (
    MAX_SWEEP_THROUGH,
    MOVE_NORMAL,
    MOVE_PORTAL,
    MOVE_SWEEP,
    decode_move,
    move_fold,
    move_from,
    move_ray,
    move_through,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.refusals import refused_ids

WIDTH = 6
WILD = 2
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
ROOK = {"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}
# A fold pair on c3 and e5, or a momentum pair on c3 and e4 (e5 holds a piece at the start).
FOLDS = [{"orange": "c3", "blue": "e5", "kind": "fold"}]
MOMENTUM = [{"orange": "c3", "blue": "e4"}]
MOMENTUM_ENDS = {"c3": ("1o", WILD), "e4": ("1b", WILD)}


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _spec(portals: list[dict[str, Any]] = FOLDS, wrap: bool = False,
          block: bool = True, **rules: Any) -> dict[str, Any]:
    """A rook-like sweeper X, kings K, a pawn-like P, and (``block``) a block B X may not take."""
    pieces: list[dict[str, Any]] = [
        {"id": "X", "name": "sweeper", "value": 5, "moves": [copy.deepcopy(ROOK)]},
        {"id": "K", "name": "king", "royal": True, "value": 0, "moves": [KING]},
        {"id": "P", "name": "pawn", "value": 1,
         "moves": [{"kind": "leap", "vectors": [[0, 1]], "mode": "move"}]},
    ]
    placements = [
        {"piece": "X", "squares": ["a1", "f1"]},
        {"piece": "K", "squares": ["c1"]},
        {"piece": "P", "squares": ["b2", "d2", "e2"]},
    ]
    if block:
        pieces[0]["captures"] = ["X", "K", "P"]
        pieces.append({"id": "B", "name": "block", "value": 1, "moves": [KING]})
        placements.append({"piece": "B", "squares": ["f2"]})
    spec: dict[str, Any] = {
        "name": "the sweep through a fold (SR6e test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": pieces,
        "setup": {"mirror": True, "placements": placements},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 80,
                  "portals": copy.deepcopy(portals), **rules},
    }
    if wrap:
        spec["rules"]["wrap"] = {"files": "both"}
    return spec


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


KINGS = {"a1": ("K", 0), "a6": ("K", 1)}


def _state(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0) -> GameState:
    return engine.make_state(placement, side_to_move=side, moved=list(placement))


def _texts_from(engine: Engine, state: GameState, origin: str) -> list[str]:
    return sorted(
        engine.describe_move(state, m) for m in engine.legal_moves(state)
        if move_from(m) == _sq(origin)
    )


def _at(engine: Engine, state: GameState, square: str) -> str | None:
    value = state.board[_sq(square)]
    return None if not value else engine.spec.piece_ids[engine.spec.type_of(value)]


class TestThroughAFold:
    def test_a_landing_on_an_empty_fold_may_stop_or_go_through(self) -> None:
        """X on a3, an enemy pawn on b3, the fold on c3 leading to e5: the sweep may stop in
        the fold or come out on e5, sweeping b3 either way; the ray runs on past the fold."""
        engine = _engine(_spec())
        state = _state(engine, {**KINGS, "a3": ("X", 0), "b3": ("P", 1)})
        texts = _texts_from(engine, state, "a3")
        assert "Xa3xc3 (sweeps b3)" in texts
        assert "Xa3-c3~e5 (sweeps b3)" in texts
        assert {"Xa3xd3 (sweeps b3)", "Xa3xe3 (sweeps b3)", "Xa3xf3 (sweeps b3)"} <= set(texts)

    def test_the_go_through_sweeps_only_before_the_fold(self) -> None:
        """b3 goes, the fold stays on c3, and d3's pawn past the fold, on the ray, stays; e6
        beside the exit stays: nothing beyond the exit is swept."""
        engine = _engine(_spec())
        state = _state(engine, {**KINGS, "a3": ("X", 0), "b3": ("P", 1), "d3": ("P", 1),
                                "e6": ("P", 1)})
        move = engine.find_move(state, "Xa3-c3~e5 (sweeps b3)")
        assert move_type(move) == MOVE_SWEEP and move_to(move) == _sq("e5")
        assert move_through(move) == 2  # c3 is the ray's second square
        after = engine.apply(state, move)
        assert _at(engine, after, "e5") == "X" and _at(engine, after, "a3") is None
        assert _at(engine, after, "b3") is None
        assert _at(engine, after, "d3") == "P" and _at(engine, after, "e6") == "P"
        assert engine.fold_square(after, 0) == _sq("c3")
        assert after.key == engine._compute_hash(after)

    def test_a_go_through_may_take_on_the_far_end(self) -> None:
        engine = _engine(_spec())
        state = _state(engine, {**KINGS, "a3": ("X", 0), "b3": ("P", 1), "e5": ("P", 1)})
        after = engine.apply(state, engine.find_move(state, "Xa3-c3~xe5 (sweeps b3)"))
        assert _at(engine, after, "e5") == "X" and _at(engine, after, "b3") is None

    def test_never_onto_its_own_piece(self) -> None:
        engine = _engine(_spec())
        state = _state(engine, {**KINGS, "a3": ("X", 0), "b3": ("P", 1), "e5": ("P", 0)})
        assert not any("~" in text for text in _texts_from(engine, state, "a3"))

    def test_over_an_empty_path_it_is_a_plain_go_through(self) -> None:
        engine = _engine(_spec())
        state = _state(engine, {**KINGS, "a3": ("X", 0)})
        move = engine.find_move(state, "Xa3-c3~e5")
        assert move_type(move) == MOVE_NORMAL and move_fold(move) == 1
        assert engine.fold_route(state, move) == (_sq("c3"), _sq("e5"))

    def test_a_sweep_passing_over_a_fold_sweeps_the_piece_in_it(self) -> None:
        """An enemy pawn standing in the fold on c3: the sweep to e3 takes it, and the fold
        stays where it stood."""
        engine = _engine(_spec())
        state = _state(engine, {**KINGS, "a3": ("X", 0), "b3": ("P", 1), "c3": ("P", 1)})
        after = engine.apply(state, engine.find_move(state, "Xa3xe3 (sweeps b3, c3)"))
        assert _at(engine, after, "c3") is None and _at(engine, after, "b3") is None
        assert engine.fold_square(after, 0) == _sq("c3")

    def test_never_onto_a_square_it_sweeps(self) -> None:
        """A fold pair c3 and b3, an enemy pawn standing in the b3 end: the sweep reaching c3
        sweeps b3, so it does not come out there; taking on b3 directly stays."""
        engine = _engine(_spec([{"orange": "c3", "blue": "b3", "kind": "fold"}]))
        state = _state(engine, {**KINGS, "a3": ("X", 0), "b3": ("P", 1)})
        texts = _texts_from(engine, state, "a3")
        assert "Xa3xb3" in texts and not any("~" in text for text in texts)

    def test_a_go_through_that_sweeps_what_the_direct_sweep_does_is_that_move(self) -> None:
        """A fold pair c3 and e3 on the sweeper's own rank: through c3 to e3 and straight to
        e3 both sweep b3 alone, so they are one move, written as the direct one."""
        engine = _engine(_spec([{"orange": "c3", "blue": "e3", "kind": "fold"}]))
        state = _state(engine, {**KINGS, "a3": ("X", 0), "b3": ("P", 1)})
        to_e3 = [t for t in _texts_from(engine, state, "a3") if t.endswith("e3 (sweeps b3)")]
        assert to_e3 == ["Xa3xe3 (sweeps b3)"]
        assert move_through(engine.find_move(state, "Xa3xe3 (sweeps b3)")) == 0

    def test_a_block_ends_the_ray_before_the_fold(self) -> None:
        engine = _engine(_spec())
        state = _state(engine, {**KINGS, "a3": ("X", 0), "b3": ("B", 1)})
        assert not any("c3" in text for text in _texts_from(engine, state, "a3"))

    def test_python_reads_the_path_off_the_word(self) -> None:
        engine = _engine(_spec())
        state = _state(engine, {**KINGS, "a3": ("X", 0), "b3": ("P", 1)})
        move = engine.find_move(state, "Xa3-c3~e5 (sweeps b3)")
        assert engine.spec.sweep_path(state.board, move) == (_sq("b3"),)
        assert engine.sweep_fold_route(state, move) == (_sq("c3"), _sq("e5"))


class TestCheckThroughAFold:
    @pytest.mark.parametrize(("extra", "checked"), [
        ({}, True), ({"b3": ("P", 1)}, True), ({"b3": ("B", 1)}, False),
    ])
    def test_a_sweep_gives_check_through_a_fold(
        self, extra: dict[str, tuple[str, int]], checked: bool,
    ) -> None:
        """The enemy king on e5, the far end; X on a3 reaches it through c3, over b3's pawn
        too, but not past a block."""
        engine = _engine(_spec())
        state = _state(engine, {"a1": ("K", 0), "e5": ("K", 1), "a3": ("X", 0), **extra},
                       side=1)
        assert engine.in_check(state) is checked

    def test_the_mover_may_not_sweep_into_a_check_through_a_fold(self) -> None:
        """The mover's own king on b3: the ray through c3 would sweep it, so no sweep east."""
        engine = _engine(_spec())
        state = _state(engine, {"b3": ("K", 0), "a6": ("K", 1), "a3": ("X", 0)})
        assert not any(
            text.startswith("Xa3") and ("c3" in text or "~" in text)
            for text in _texts_from(engine, state, "a3")
        )


class TestAMomentumEnd:
    def test_it_ends_the_ray_and_no_sweep_transits(self) -> None:
        engine = _engine(_spec(MOMENTUM))
        state = _state(engine, {**KINGS, **MOMENTUM_ENDS, "a3": ("X", 0), "b3": ("P", 1)})
        moves = [m for m in engine.legal_moves(state) if move_from(m) == _sq("a3")]
        assert not any(move_type(m) == MOVE_PORTAL for m in moves)
        reached = {move_to(m) for m in moves}
        assert _sq("b3") in reached and not reached & {_sq(s) for s in ("c3", "d3", "e3", "f3")}

    @pytest.mark.parametrize("king", ["d3", "f4"])
    def test_no_check_past_it_or_through_it(self, king: str) -> None:
        """A king on d3, past the end on the ray, or on f4, where a slide through c3 comes
        out of e4: the sweeper on a3 attacks neither."""
        engine = _engine(_spec(MOMENTUM))
        state = _state(engine, {"a1": ("K", 0), king: ("K", 1), **MOMENTUM_ENDS,
                                "a3": ("X", 0)}, side=1)
        assert not engine.in_check(state)


class TestWithADeclaredRay:
    """SR6d's ray and SR6e's fold on one word: on a cylinder two rays reach the fold."""

    def test_two_rays_into_one_fold_are_two_moves(self) -> None:
        """X on a3, the fold on d3 leading to e5, enemy pawns on c3 and e3: east of a3 the ray
        reaches d3 over c3, west over e3."""
        engine = _engine(_spec([{"orange": "d3", "blue": "e5", "kind": "fold"}], wrap=True))
        state = _state(engine, {**KINGS, "a3": ("X", 0), "c3": ("P", 1), "e3": ("P", 1)})
        texts = [t for t in _texts_from(engine, state, "a3") if "~" in t]
        assert texts == ["Xa3-d3~e5 (sweeps c3)", "Xa3-d3~e5 (sweeps e3)"]
        east = engine.find_move(state, "Xa3-d3~e5 (sweeps c3)")
        west = engine.find_move(state, "Xa3-d3~e5 (sweeps e3)")
        assert move_ray(east) != move_ray(west)
        for move, gone, stays in ((east, "c3", "e3"), (west, "e3", "c3")):
            after = engine.apply(state, move)
            assert _at(engine, after, gone) is None and _at(engine, after, stays) == "P"
            assert after.key == engine._compute_hash(after)

    def test_a_ray_longer_than_a_step_can_name_is_refused(self) -> None:
        """A diagonal sweeper on an 8 by 7 torus: its ray runs 55 squares before it meets
        itself, past what the promotion field's five bits name."""
        spec = _spec(block=False)
        spec["pieces"][0]["moves"] = [{"kind": "sweep", "vectors": [[1, 1]], "symmetry": "rotate4"}]
        spec["board"] = {"width": 8, "height": 7}
        spec["rules"]["wrap"] = {"files": "both", "ranks": "both"}
        spec["setup"]["placements"] = [
            {"piece": "X", "squares": ["a1"]}, {"piece": "K", "squares": ["f1"]},
        ]
        assert validate_spec(spec) == []
        why = rust_supports(compile_spec(spec))
        assert why is not None and str(MAX_SWEEP_THROUGH) in why


class TestTheValidator:
    @pytest.mark.parametrize("portals", [FOLDS, MOMENTUM])
    def test_portals_beside_a_sweep_are_no_longer_refused(self, portals: list[Any]) -> None:
        errors = validate_spec(_spec(portals))
        assert "W8" not in refused_ids(errors) and errors == []
        assert rust_supports(compile_spec(_spec(portals))) is None


def _taken(engine: Engine, state: GameState, move: int) -> frozenset[int]:
    swept = {square for square, _ in engine.spec.sweep_victims(state.board, move)}
    target = move_to(move)
    return frozenset(swept | ({target} if state.board[target] else set()))


# The fold squares as the oracle reads them: each end's square to its partner's.
ORACLE_FOLDS = {_sq("c3"): _sq("e5"), _sq("e5"): _sq("c3")}


class TestAgainstTheOracle:
    @pytest.mark.parametrize(("seed", "wrap"), [(0, False), (1, False), (2, True), (3, True)])
    def test_the_core_agrees_with_the_oracle_over_random_games(self, seed: int, wrap: bool) -> None:
        spec = _spec(block=False, wrap=wrap)
        engine = _engine(spec)
        oracle = sweep_oracle.Oracle(spec, ORACLE_FOLDS)
        n_types = engine.spec.n_types
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
            through = [m for m in moves if move_through(m) or move_fold(m)]
            sweeps = through or [m for m in moves if move_type(m) == MOVE_SWEEP]
            state = engine.apply(state, rng.choice(sweeps if sweeps and rng.random() < 0.7 else moves))
            assert state.key == engine._compute_hash(state)

    @pytest.mark.parametrize("wrap", [False, True])
    def test_perft_against_the_oracle_and_apply(self, wrap: bool) -> None:
        spec = _spec(block=False, wrap=wrap)
        self._perft(spec)

    def test_perft_with_a_ride_past_the_sweep(self) -> None:
        """Q1063's n < m arm: a sweep that clears one step and rides on, so the fold on c3
        stands on its ride; b3 is swept, a piece on the ride ends the ray."""
        spec = _spec(block=False)
        spec["pieces"][0]["moves"][0]["sweep_range"] = 1
        self._perft(spec)

    def _perft(self, spec: dict[str, Any]) -> None:
        engine = _engine(spec)
        state = _state(engine, {"a1": ("K", 0), "f6": ("K", 1), "a3": ("X", 0),
                                "b3": ("P", 1), "d3": ("P", 1), "e3": ("X", 1)})
        pos = sweep_oracle.position_of(state.board, 0, state.unmoved, engine.spec.n_types)
        oracle = sweep_oracle.Oracle(spec, ORACLE_FOLDS)

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(count(engine.apply(position, m), depth - 1)
                       for m in engine.legal_moves(position))

        assert engine.perft(state, 2) == oracle.perft(pos, 2) == count(state, 2)
        assert engine.perft(state, 3) == count(state, 3)
