"""A turning-back wild piece beside check (generation 125, SR8h, N5, `docs/plan-schema-next.md`
§2.8, ruled 2026-10-27: *its attack is the square ahead after the turn, a second attack source*).

A wild piece on a line threatens the square its next step lands on, read on the board as it
stands: the square ahead by its heading, or, where that holds a piece it may not take, the one
behind after it turns round; on a line two pieces walk, in the phase's own order (a piece ahead
that walks away first is no block). Checkmate, `forbid_check` and `reach.safe` read it
(`turning.rs::line_threatens`, asked from `threatened`); the legality filter reads the headings
the phase leaves.

Hand positions on a 6x6 board, the lift, a perft and seeded walks against the oracle
(`tests/turning_oracle.py::legal_moves`), and the words.
"""

from __future__ import annotations

import random
from typing import Any

import pytest
from turning_oracle import TurningOracle

from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.describe import describe_spec

KING = [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]
ROUTE = ["a3", "b3", "c3", "d3", "e3"]


def _spec(also_at: list[str] | None = None, **rules: Any) -> dict[str, Any]:
    """Kings on a1 and f6, a rook each, and a wolf pacing a3 to e3, won by checkmate. The
    wolf eats a king or a rook, never the stone ``S``."""
    npc: dict[str, Any] = {"piece": "W", "route": ROUTE, "turns_back": True}
    if also_at:
        npc["also_at"] = also_at
    return {
        "schema_version": 31,
        "name": "turning beside check",
        "board": {"width": 6, "height": 6},
        "pieces": [
            {"id": "K", "name": "king", "royal": True, "value": 0, "moves": KING},
            {"id": "R", "name": "rook", "value": 5,
             "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
            {"id": "S", "name": "stone", "value": 1, "moves": KING},
            {"id": "W", "name": "wolf", "value": 3, "moves": KING, "captures": ["K", "R"]},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["f6"]},
            {"piece": "R", "player": 0, "squares": ["f1"]},
            {"piece": "R", "player": 1, "squares": ["a6"]},
        ]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 200,
                  "npcs": [npc], **rules},
    }


def _engine(also_at: list[str] | None = None, **rules: Any) -> Engine:
    spec = _spec(also_at, **rules)
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _state(engine: Engine, placement: dict[str, tuple[str, int]], back: tuple[str, ...] = (),
           side: int = 0) -> GameState:
    return engine.make_state(placement, side_to_move=side, heading_back=list(back))


def _texts(engine: Engine, state: GameState) -> set[str]:
    return {engine.describe_move(state, move) for move in engine.legal_moves(state)}


class TestTheThreat:
    def test_the_square_ahead(self) -> None:
        engine = _engine()
        assert engine.in_check(_state(engine, {"c3": ("K", 0), "f6": ("K", 1), "b3": ("W", 2)}), 0)
        assert not engine.in_check(_state(engine, {"a3": ("K", 0), "f6": ("K", 1), "b3": ("W", 2)}), 0)

    def test_walking_back_the_square_behind_it(self) -> None:
        engine = _engine()
        state = _state(engine, {"a3": ("K", 0), "f6": ("K", 1), "b3": ("W", 2)}, back=("b3",))
        assert engine.in_check(state, 0)

    def test_after_the_turn_at_a_block(self) -> None:
        """A stone it may not take on c3: it turns and steps back onto the king on a3."""
        engine = _engine()
        state = _state(engine, {"a3": ("K", 0), "f6": ("K", 1), "b3": ("W", 2), "c3": ("S", 1)})
        assert engine.in_check(state, 0)

    def test_a_piece_it_may_take_is_no_block(self) -> None:
        engine = _engine()
        state = _state(engine, {"a3": ("K", 0), "f6": ("K", 1), "b3": ("W", 2), "c3": ("R", 1)})
        assert not engine.in_check(state, 0)

    def test_on_an_end_it_faces_in(self) -> None:
        engine = _engine()
        state = _state(engine, {"d3": ("K", 0), "f6": ("K", 1), "e3": ("W", 2)})
        assert engine.in_check(state, 0)

    def test_a_queue_walking_away_is_no_block(self) -> None:
        """Two wolves walking forward on b3 and c3: c3 steps first, so b3 follows it and the
        king behind on a3 is not threatened."""
        engine = _engine(also_at=["c3"])
        state = _state(engine, {"a3": ("K", 0), "f6": ("K", 1), "b3": ("W", 2), "c3": ("W", 2)})
        assert not engine.in_check(state, 0)

    def test_head_on_the_forward_one_turns(self) -> None:
        """Head-on (b3 forward, c3 back) the forward one steps first, finds c3 held, turns and
        steps onto a3."""
        engine = _engine(also_at=["c3"])
        state = _state(engine, {"a3": ("K", 0), "f6": ("K", 1), "b3": ("W", 2), "c3": ("W", 2)},
                       back=("c3",))
        assert engine.in_check(state, 0)


class TestLegality:
    def test_the_king_may_not_step_into_its_way(self) -> None:
        """The wolf on a3 faces b3: b3 is eaten in the mover's own phase, and c3 is where it
        steps next after that; c2 is free."""
        engine = _engine()
        state = _state(engine, {"b2": ("K", 0), "f6": ("K", 1), "a3": ("W", 2)})
        texts = _texts(engine, state)
        assert not any(t.startswith("Kb2-b3") or t.startswith("Kb2-c3") for t in texts), texts
        assert any(t.startswith("Kb2-c2") for t in texts), texts

    def test_the_filter_reads_the_headings_the_phase_leaves(self) -> None:
        """The wolf on c3 walks back to b3 in the mover's phase, still walking back, so a3 is
        where it steps next: Ka2-a3 is not offered (a stale heading would read c3)."""
        engine = _engine()
        state = _state(engine, {"a2": ("K", 0), "f6": ("K", 1), "c3": ("W", 2)}, back=("c3",))
        texts = _texts(engine, state)
        assert not any(t.startswith("Ka2-a3") for t in texts), texts
        assert any(t.startswith("Ka2-a1") for t in texts), texts

    def test_mate_by_the_wolf(self) -> None:
        """The king on a3 with the wolf on b3 walking back to it, every escape (and taking the
        wolf) held by the rooks on a6, b6 and f2: checkmate by the wolf. (The a-file rook alone
        would not hold a2: the wolf stepping onto a3 in the mover's phase blocks its file.)"""
        engine = _engine()
        state = _state(engine, {"a3": ("K", 0), "f6": ("K", 1), "b3": ("W", 2),
                                "a6": ("R", 1), "b6": ("R", 1), "f2": ("R", 1)}, back=("b3",))
        assert engine.in_check(state, 0)
        assert engine.legal_moves(state) == []
        outcome = engine.terminal(state)
        assert outcome is not None and outcome.winner == 1 and outcome.reason == "checkmate"

    def test_forbid_check_bars_a_block_that_turns_it_onto_the_other_king(self) -> None:
        """A stone set on d3 turns the wolf on c3 back to b3, facing the king on a3: giving
        check, so not offered under forbid_check; the stone's other steps are."""
        engine = _engine(forbid_check=True)
        state = _state(engine, {"a1": ("K", 0), "a3": ("K", 1), "c3": ("W", 2), "d4": ("S", 0)})
        texts = _texts(engine, state)
        assert not any(t.startswith("Sd4-d3") for t in texts), texts
        assert any(t.startswith("Sd4-e4") for t in texts), texts

    def test_a_safe_goal_where_the_wolf_steps_decides_nothing(self) -> None:
        """A rook arriving on c3 wins only where the wolf's next step is not c3."""
        engine = _engine(reach={"squares": ["c3"], "piece": "R", "safe": True})
        ahead = _state(engine, {"a1": ("K", 0), "f6": ("K", 1), "a3": ("W", 2), "c5": ("R", 0)})
        after = engine.apply(ahead, engine.find_move(ahead, "Rc5-c3"))
        assert engine.terminal(after) is None
        away = _state(engine, {"a1": ("K", 0), "f6": ("K", 1), "b3": ("W", 2), "c5": ("R", 0)},
                      back=("b3",))
        after = engine.apply(away, engine.find_move(away, "Rc5-c3"))
        outcome = engine.terminal(after)
        assert outcome is not None and outcome.winner == 0 and outcome.reason == "reached"


def _oracle_perft(oracle: TurningOracle, state: GameState, depth: int) -> int:
    """The oracle's count: its legal list at every node, every move's phase, board and
    headings held to the core's."""
    if depth == 0:
        return 1
    engine = oracle.engine
    if engine.terminal(state) is not None:
        return 0
    moves = engine.legal_moves(state)
    assert sorted(moves) == sorted(oracle.legal_moves(state)), state
    total = 0
    for move in moves:
        after = oracle.check(state, move)
        assert after is not None
        total += _oracle_perft(oracle, after, depth - 1)
    return total


class TestTheOracle:
    @pytest.mark.parametrize(
        ("rules", "also_at"),
        [({}, None), ({"forbid_check": True}, None), ({}, ["c3"])],
        ids=["checkmate", "forbid_check", "two-wolves"],
    )
    def test_a_perft(self, rules: dict[str, Any], also_at: list[str] | None) -> None:
        spec = _spec(also_at, **rules)
        oracle = TurningOracle(spec)
        engine = oracle.engine
        placement = {"b2": ("K", 0), "f5": ("K", 1), "a3": ("W", 2), "e1": ("R", 0),
                     "c6": ("R", 1), "d4": ("S", 0), "c5": ("S", 1)}
        if also_at:
            placement["d3"] = ("W", 2)
        state = _state(engine, placement)
        assert engine.perft(state, 3) == _oracle_perft(oracle, state, 3)

    @pytest.mark.parametrize("seed", range(6))
    def test_a_seeded_walk(self, seed: int) -> None:
        """Forty plies from the opening of the two-wolf game with stones, every legal list the
        oracle's."""
        spec = _spec(["e3"])
        spec["setup"]["placements"] += [
            {"piece": "S", "player": 0, "squares": ["b2", "d2"]},
            {"piece": "S", "player": 1, "squares": ["b5", "d5"]},
        ]
        oracle = TurningOracle(spec)
        engine = oracle.engine
        rng = random.Random(seed)
        state = engine.initial_state()
        for _ in range(40):
            moves = engine.legal_moves(state)
            if not moves or engine.terminal(state, moves) is not None:
                break
            assert sorted(moves) == sorted(oracle.legal_moves(state))
            after = oracle.check(state, rng.choice(moves))
            assert after is not None
            state = after


class TestTheWords:
    def test_describe_says_the_threat(self) -> None:
        text = describe_spec(_spec())
        assert "standing on the square it would step to next is in check from it" in text
        assert "the one behind once it has turned round" in text
