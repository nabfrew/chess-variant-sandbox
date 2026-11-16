"""Head-on routes (generation 123, SR8f, N4: `docs/plan-schema-next.md` §2.8, ruled 2026-10-27).

`turns_back` beside `also_at`: several wild pieces pace one line. They step front first by
their headings (a piece whose square ahead holds a piece still to step goes after it), and
two that meet head-on step in the order the plan ruled: **the one nearer the route's first
square moves first**. Until this generation the pair was refused (*an order the queue rule
does not give*).

Hand positions on a 6x6 board, each worked by hand and laid out so that the ascending walk of
the squares, the order the phase had for one piece a line, would give another answer; then
the declared oracle (`tests/turning_oracle.py`, which states the order on its own terms) over
seeded games of the turning wolves with two wolves on each line, the key at every ply, a perft
against `apply`, the coming phases, the words and Fairy-Stockfish's refusal.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
from turning_oracle import TurningOracle

from chess_variant_sandbox import fuzz
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.bridge import rust_supports
from chess_variant_sandbox.engine.coming import coming_phases
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.describe import describe_spec

ROOT = Path(__file__).resolve().parents[1]
KING = [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]
WOLVES = ROOT / "tests" / "_turning_wolves.json"
FORWARD = ["a3", "b3", "c3", "d3", "e3", "f3"]
REVERSED = FORWARD[::-1]


def _spec(route: list[str], also_at: list[str]) -> dict[str, Any]:
    """Kings on a1 and f6 and wolves pacing ``route``, one starting on its first square and one
    on each of ``also_at``. A wolf eats a king, a rook or a knight, never the stone ``S``."""
    return {
        "schema_version": 31,
        "name": "head-on hand positions",
        "board": {"width": 6, "height": 6},
        "pieces": [
            {"id": "K", "name": "king", "royal": True, "value": 0, "moves": KING},
            {"id": "R", "name": "rook", "value": 5,
             "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
            {"id": "N", "name": "knight", "value": 3,
             "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}]},
            {"id": "S", "name": "stone", "value": 1, "moves": KING},
            {"id": "W", "name": "wolf", "value": 3, "moves": KING, "captures": ["K", "R", "N"]},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["f6"]},
        ]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 200,
                  "npcs": [{"piece": "W", "route": route, "also_at": also_at,
                            "turns_back": True}]},
    }


def _engine(route: list[str] = FORWARD, also_at: list[str] | None = None) -> Engine:
    spec = _spec(route, also_at or [route[3]])
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _name(square: int) -> str:
    return "abcdef"[square % 6] + str(square // 6 + 1)


def _phase(engine: Engine, state: GameState, text: str) -> tuple[GameState, list[tuple]]:
    after, phase = engine.apply_with_phase(state, engine.find_move(state, text))
    assert after.key == engine._compute_hash(after), text
    return after, [(_name(s.origin), _name(s.target), s.eaten, s.kind) for s in phase]


def _back(state: GameState) -> list[str]:
    return [_name(square) for square in state.heading_back]


def _position(engine: Engine, wolves: list[str], back: list[str], **more: str) -> GameState:
    board: dict[str, tuple[str, int]] = {"a1": ("K", 0), "f6": ("K", 1)}
    board.update({square: ("W", 2) for square in wolves})
    board.update({square: ("S", 0) for square in more.get("stones", "").split()})
    return engine.make_state(board, heading_back=back)


class TestThePair:
    def test_it_validates_compiles_and_plays(self) -> None:
        spec = _spec(FORWARD, ["d3"])
        assert validate_spec(spec) == []
        compiled = compile_spec(spec)
        assert rust_supports(compiled) is None
        engine = Engine(spec)
        state = engine.initial_state()
        wolf = engine.spec.encode_piece(2, engine.spec.piece_ids.index("W"))
        assert [_name(sq) for sq, value in enumerate(state.board) if value == wolf] == ["a3", "d3"]
        state, phase = _phase(engine, state, "Ka1-b1")
        assert phase == [("a3", "b3", 0, "step"), ("d3", "e3", 0, "step")]

    def test_fairy_stockfish_refuses_it_with_every_wild_piece(self) -> None:
        with pytest.raises(Untranslatable) as caught:
            translate(_spec(FORWARD, ["d3"]))
        assert any("rules.npcs" in reason for reason in caught.value.reasons)


class TestHeadOn:
    def test_the_one_nearer_the_first_square_moves_first(self) -> None:
        """Route f3 to a3: the wolf on d3 walks forward (to c3), the one on c3 walks back (to
        d3). d3 is nearer f3, so it moves first: blocked, it turns round onto e3; then c3 steps
        into d3. Ascending, c3 would have gone first and turned onto b3."""
        engine = _engine(REVERSED)
        state = _position(engine, ["d3", "c3"], ["c3"])
        state, phase = _phase(engine, state, "Ka1-b1")
        assert phase == [("d3", "e3", 0, "turn"), ("c3", "d3", 0, "step")]
        assert _back(state) == ["d3", "e3"]

    def test_on_a_line_walked_the_other_way(self) -> None:
        """Route a3 to f3, the same meeting: c3 walks forward and is nearer a3, so it turns onto
        b3 first, and d3 steps into c3."""
        engine = _engine(FORWARD)
        state = _position(engine, ["c3", "d3"], ["d3"])
        state, phase = _phase(engine, state, "Ka1-b1")
        assert phase == [("c3", "b3", 0, "turn"), ("d3", "c3", 0, "step")]
        assert _back(state) == ["b3", "c3"]

    def test_the_first_blocked_both_ways_waits_turned(self) -> None:
        """Route f3 to a3, a stone on e3 behind the forward wolf on d3: it turns and waits; the
        wolf on c3, walking back into it, then turns too and steps to b3."""
        engine = _engine(REVERSED)
        state = _position(engine, ["d3", "c3"], ["c3"], stones="e3")
        state, phase = _phase(engine, state, "Ka1-b1")
        assert phase == [("d3", "d3", 0, "turn"), ("c3", "b3", 0, "turn")]
        assert _back(state) == ["d3"]


class TestFrontFirst:
    def test_a_queue_walking_back_moves_as_one(self) -> None:
        """Route f3 to a3, both wolves walking back (toward f3): the front one, d3, steps first,
        and c3 follows into d3. Ascending, c3 would have met d3 and turned onto b3."""
        engine = _engine(REVERSED)
        state = _position(engine, ["c3", "d3"], ["c3", "d3"])
        state, phase = _phase(engine, state, "Ka1-b1")
        assert phase == [("d3", "e3", 0, "step"), ("c3", "d3", 0, "step")]
        assert _back(state) == ["d3", "e3"]

    def test_a_queue_walking_forward_moves_as_one(self) -> None:
        """Route f3 to a3, both walking forward (toward a3): the front one, c3, first."""
        engine = _engine(REVERSED)
        state = _position(engine, ["c3", "d3"], [])
        state, phase = _phase(engine, state, "Ka1-b1")
        assert phase == [("c3", "b3", 0, "step"), ("d3", "c3", 0, "step")]

    def test_the_front_one_turning_turns_the_queue(self) -> None:
        """Route a3 to f3, wolves on c3 and d3 walking forward, a stone on e3: d3 goes first,
        turns, finds c3 behind it and waits turned; c3 then meets d3 and turns onto b3."""
        engine = _engine(FORWARD)
        state = _position(engine, ["c3", "d3"], [], stones="e3")
        state, phase = _phase(engine, state, "Ka1-b1")
        assert phase == [("d3", "d3", 0, "turn"), ("c3", "b3", 0, "turn")]
        assert _back(state) == ["b3", "d3"]


class TestTheState:
    def test_coming_shows_the_meeting(self) -> None:
        engine = _engine(REVERSED)
        state = _position(engine, ["d3", "c3"], ["c3"])
        first, _ = coming_phases(engine, state, 2)
        assert [(_name(s.origin), _name(s.target), s.kind) for s in first] == [
            ("d3", "e3", "turn"), ("c3", "d3", "step")]

    def test_unmake_puts_both_headings_back(self) -> None:
        """perft(3) from a meeting against the count made by applying every move afresh."""
        engine = _engine(REVERSED)
        state = _position(engine, ["d3", "c3"], ["c3"])

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            if engine.terminal(position) is not None:
                return 0
            return sum(count(engine.apply(position, move), depth - 1)
                       for move in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)

    def test_a_king_in_the_turning_wolfs_way_is_eaten(self) -> None:
        """A king stepping onto e3 beside the meeting is eaten by the wolf turning there; in
        ascending order c3 would have turned away first and d3 stepped on to c3, sparing it."""
        engine = _engine(REVERSED)
        state = engine.make_state(
            {"e2": ("K", 0), "f6": ("K", 1), "d3": ("W", 2), "c3": ("W", 2)},
            heading_back=["c3"],
        )
        after, phase = _phase(engine, state, "Ke2-e3")
        assert phase[0][:2] == ("d3", "e3") and phase[0][2] and phase[0][3] == "turn"
        outcome = engine.terminal(after)
        assert outcome is not None and outcome.winner == 1


class TestTheWords:
    def test_describe_says_the_head_on_order(self) -> None:
        text = describe_spec(_spec(FORWARD, ["d3"]))
        assert "paces the same line" in text
        assert "meet head-on move nearer a3 first" in text


def _wolves_in_pairs() -> dict[str, Any]:
    """The turning wolves with a second wolf on each pacing line (and its mirror)."""
    spec = json.loads(WOLVES.read_text(encoding="utf-8"))
    spec["name"] = "turning wolves in pairs"
    npcs = spec["rules"]["npcs"]
    npcs[0]["also_at"] = ["c4"]
    npcs[1]["also_at"] = ["g4"]
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def _walk(spec: dict[str, Any], seed: int, plies: int) -> int:
    """One seeded random game held to the oracle at every ply; the head-on turns it saw."""
    oracle = TurningOracle(spec)
    engine = oracle.engine
    rng = random.Random(seed)
    state = engine.initial_state()
    turns = 0
    for _ in range(plies):
        if engine.terminal(state) is not None:
            break
        move = rng.choice(engine.legal_moves(state))
        after = oracle.check(state, move)
        if after is None:
            after = engine.apply(state, move)
        assert after.key == engine._compute_hash(after)
        fuzz.every_npc_stands_on_its_own_route(engine, after)
        turns += sum(1 for s in engine.apply_with_phase(state, move)[1] if s.kind == "turn")
        state = after
    return turns


class TestTheOracle:
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_a_seeded_walk(self, seed: int) -> None:
        _walk(_wolves_in_pairs(), seed, 120)

    def test_the_walks_see_the_wolves_meet(self) -> None:
        spec = _wolves_in_pairs()
        assert sum(_walk(spec, seed, 60) for seed in range(10, 14)) > 0

    @pytest.mark.slow
    def test_twenty_walks(self) -> None:
        spec = _wolves_in_pairs()
        assert sum(_walk(spec, seed, 300) for seed in range(100, 120)) > 0
