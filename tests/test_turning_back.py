"""Wild pieces that turn back (generation 38, N5, `docs/plan-npc.md` §4.7).

An entry with ``turns_back: true`` walks its route as a line, first to last and back. A piece
faces forward at the start; landing on an end turns it into the line; a step blocked ahead turns
it round and it steps the other way, and blocked both ways it waits, turned. The heading is
state, in the key.

Hand positions on a 6x6 board (the end turn, the block turn, the double block, a block on an
end, the heading in the key, a taken piece's heading dropped, a return on a line), the inert
spellings, the refusals and the words; then the oracle (`tests/turning_oracle.py`) walked over
seeded games of `tests/_turning_wolves.json`, and the key over a round game.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest
from turning_oracle import TurningOracle

from chess_variant_sandbox import fuzz
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.coming import coming_phases
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.describe import describe_spec

ROOT = Path(__file__).resolve().parents[1]
KING = [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]
WOLVES = ROOT / "tests" / "_turning_wolves.json"
ROUNDS = ROOT / "tests" / "_turning_rounds.json"


def _spec(turns_back: Any = True, **npc_extra: Any) -> dict:
    """Kings on a1 and f6, and one wolf pacing a3, b3, c3, d3. The wolf eats a king, a rook
    or a knight, never the stone ``S``."""
    npc: dict[str, Any] = {"piece": "W", "route": ["a3", "b3", "c3", "d3"], **npc_extra}
    if turns_back is not None:
        npc["turns_back"] = turns_back
    return {
        "schema_version": 31,
        "name": "turning hand positions",
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
                  "npcs": [npc]},
    }


def _engine(**kwargs: Any) -> Engine:
    spec = _spec(**kwargs)
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _sq(engine: Engine, name: str) -> int:
    return engine._square_index(name)


def _wolf_at(engine: Engine, state: GameState) -> str:
    wolf = engine.spec.encode_piece(2, engine.spec.piece_ids.index("W"))
    [square] = [sq for sq, value in enumerate(state.board) if value == wolf]
    return "abcdef"[square % 6] + str(square // 6 + 1)


def _play(engine: Engine, state: GameState, text: str) -> tuple[GameState, tuple]:
    after, phase = engine.apply_with_phase(state, engine.find_move(state, text))
    assert after.key == engine._compute_hash(after), text
    return after, phase


def _back(engine: Engine, state: GameState) -> list[str]:
    return ["abcdef"[sq % 6] + str(sq // 6 + 1) for sq in state.heading_back]


# The kings shuffle, so the wolf's walk is the only thing the phase does.
_SHUFFLE = ["Ka1-b1", "Kf6-f5", "Kb1-a1", "Kf5-f6"] * 4


class TestTheWalk:
    def test_the_end_turn(self) -> None:
        """Forward to d3, where it turns; back to a3, where it turns again."""
        engine = _engine()
        state = engine.initial_state()
        seen = []
        for text in _SHUFFLE[:7]:
            state, phase = _play(engine, state, text)
            assert all(step.kind == "step" for step in phase)
            seen.append((_wolf_at(engine, state), _back(engine, state)))
        assert seen == [
            ("b3", []), ("c3", []), ("d3", ["d3"]), ("c3", ["c3"]), ("b3", ["b3"]),
            ("a3", []), ("b3", []),
        ]

    def test_the_block_turn(self) -> None:
        """Walking forward from c3 into a stone on d3, it turns and steps back to b3."""
        engine = _engine()
        state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "c3": ("W", 2),
                                   "d3": ("S", 0)})
        state, phase = _play(engine, state, "Ka1-b1")
        [step] = phase
        assert (step.origin, step.target, step.eaten, step.kind) == (
            _sq(engine, "c3"), _sq(engine, "b3"), 0, "turn")
        assert _back(engine, state) == ["b3"]
        state, phase = _play(engine, state, "Kf6-f5")
        assert _wolf_at(engine, state) == "a3" and _back(engine, state) == []

    def test_the_double_block(self) -> None:
        """Stones on a3 and c3: it waits on b3, turned, and turns again the next phase."""
        engine = _engine()
        state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "b3": ("W", 2),
                                   "a3": ("S", 0), "c3": ("S", 0)})
        state, phase = _play(engine, state, "Ka1-b1")
        assert [(s.origin, s.target, s.eaten, s.kind) for s in phase] == [
            (_sq(engine, "b3"), _sq(engine, "b3"), 0, "turn")]
        assert _wolf_at(engine, state) == "b3" and _back(engine, state) == ["b3"]
        state, phase = _play(engine, state, "Kf6-f5")
        assert [s.kind for s in phase] == ["turn"] and _back(engine, state) == []

    def test_blocked_on_an_end_it_waits_facing_in(self) -> None:
        engine = _engine()
        state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "a3": ("W", 2),
                                   "b3": ("S", 0)})
        state, phase = _play(engine, state, "Ka1-b1")
        assert phase == () and _back(engine, state) == []

    def test_a_turned_step_eats(self) -> None:
        """Blocked ahead by a stone, it turns onto the knight behind it and takes it."""
        engine = _engine()
        state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "c3": ("W", 2),
                                   "d3": ("S", 0), "b3": ("N", 1)})
        state, phase = _play(engine, state, "Ka1-a2")
        [step] = phase
        assert step.kind == "turn" and step.eaten and _wolf_at(engine, state) == "b3"
        assert state.halfmove == 0

    def test_it_eats_a_king_in_its_way(self) -> None:
        engine = _engine()
        state = engine.make_state({"d2": ("K", 0), "f6": ("K", 1), "b3": ("W", 2)})
        state, _ = _play(engine, state, "Kd2-c3")
        assert _wolf_at(engine, state) == "c3"
        outcome = engine.terminal(state)
        assert outcome is not None and outcome.winner == 1

    def test_a_walking_wolf_puts_nothing_in_check(self) -> None:
        """Its line carries no attack source: the rule is refused where check is asked."""
        engine = _engine()
        state = engine.make_state({"c3": ("K", 0), "f6": ("K", 1), "b3": ("W", 2)})
        assert not engine.in_check(state, 0)


class TestTheHeading:
    def test_the_heading_is_in_the_key(self) -> None:
        engine = _engine()
        board = {"a1": ("K", 0), "f6": ("K", 1), "c3": ("W", 2)}
        forward = engine.make_state(board)
        back = engine.make_state(board, heading_back=["c3"])
        assert back.heading_back == (_sq(engine, "c3"),)
        assert forward.key != back.key
        assert engine._compute_hash(back) == back.key
        state, _ = _play(engine, back, "Ka1-b1")
        assert _wolf_at(engine, state) == "b3"

    def test_a_heading_where_no_wolf_stands_is_dropped(self) -> None:
        engine = _engine()
        state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "c3": ("W", 2)},
                                  heading_back=["b3"])
        assert state.heading_back == ()
        after, _ = _play(engine, state, "Ka1-b1")
        assert _back(engine, after) == ["d3"]

    def test_a_taken_wolfs_heading_goes_with_it(self) -> None:
        engine = _engine()
        state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "c3": ("W", 2),
                                   "c6": ("R", 0)}, heading_back=["c3"])
        state, _ = _play(engine, state, "Rc6xc3")
        assert state.heading_back == ()

    def test_a_wolf_that_comes_back_faces_forward(self) -> None:
        engine = _engine(respawn=1)
        state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "c3": ("W", 2),
                                   "c6": ("R", 0)}, heading_back=["c3"])
        state, phase = _play(engine, state, "Rc6xc3")
        assert [s.kind for s in phase] == ["return"]
        assert _wolf_at(engine, state) == "a3" and state.heading_back == ()
        state, _ = _play(engine, state, "Kf6-f5")
        assert _wolf_at(engine, state) == "b3"

    def test_unmake_puts_the_heading_back(self) -> None:
        """perft(3) from a position with a wolf walking back, against the same count made
        by applying every move from a fresh position (each `apply` builds a new game)."""
        engine = _engine()
        state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "c3": ("W", 2),
                                   "d3": ("S", 0), "e1": ("R", 0)}, heading_back=["c3"])

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            if engine.terminal(position) is not None:
                return 0
            return sum(count(engine.apply(position, move), depth - 1)
                       for move in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)

    def test_coming_shows_the_turn(self) -> None:
        engine = _engine()
        state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "c3": ("W", 2),
                                   "d3": ("S", 0)})
        first, then = coming_phases(engine, state, 2)
        assert [(s.target, s.kind) for s in first] == [(_sq(engine, "b3"), "turn")]
        assert [(s.target, s.kind) for s in then] == [(_sq(engine, "a3"), "step")]


class TestInert:
    def test_false_is_the_game_without_it(self) -> None:
        base, written = _spec(turns_back=None), _spec(turns_back=False)
        assert validate_spec(written) == []
        assert spec_hash(written) == spec_hash(base)
        assert canonical_tables(compile_spec(written)) == canonical_tables(compile_spec(base))

    def test_a_game_without_it_crosses_no_line(self) -> None:
        tables = canonical_tables(compile_spec(_spec(turns_back=None)))
        assert "npc_prev" not in tables and "zobrist_heading" not in tables
        assert Engine(_spec(turns_back=None)).initial_state().heading_back == ()

    def test_the_off_value_is_dropped_in_the_normal_form(self) -> None:
        spec = copy.deepcopy(_spec(turns_back=False))
        assert spec_hash(spec) == spec_hash(_spec(turns_back=None))


class TestRefusals:
    def _errors(self, spec: dict) -> list[str]:
        return [e for e in validate_spec(spec) if "turns_back" in e]

    def test_not_a_boolean(self) -> None:
        assert any("must be a boolean" in e for e in self._errors(_spec(turns_back=1)))

    @pytest.mark.parametrize(
        ("change", "said"),
        [
            (lambda r: r.update(win="checkmate"), '"win": "checkmate"'),
            (lambda r: r.update(forbid_check=True), "forbid_check"),
            (lambda r: r.update(reach={"squares": ["f1"], "safe": True}), "rules.reach.safe"),
        ],
        ids=["checkmate", "forbid_check", "reach-safe"],
    )
    def test_beside_check(self, change: Any, said: str) -> None:
        spec = _spec()
        change(spec["rules"])
        errors = self._errors(spec)
        assert any(said in e and "depends on what blocks it" in e for e in errors), errors

    def test_beside_also_at(self) -> None:
        errors = self._errors(_spec(also_at=["c3"]))
        assert any("beside also_at" in e for e in errors), errors

    def test_on_one_square(self) -> None:
        spec = _spec()
        spec["rules"]["npcs"][0]["route"] = ["c3"]
        assert any("nothing to turn on" in e for e in self._errors(spec))

    def test_fairy_stockfish_refuses_it_with_every_wild_piece(self) -> None:
        with pytest.raises(Untranslatable) as caught:
            translate(_spec())
        assert any("rules.npcs" in reason for reason in caught.value.reasons)


class TestTheWords:
    def test_describe_says_the_line_and_the_turn(self) -> None:
        text = describe_spec(_spec())
        assert "paces the line a3, b3, c3 and d3" in text or "paces the line" in text
        assert "turns round and steps the other way" in text
        assert "in check from it" not in text

    def test_a_game_without_it_says_nothing_of_it(self) -> None:
        assert "turns round" not in describe_spec(_spec(turns_back=None))


def _walk(spec: dict, seed: int, plies: int) -> int:
    """One seeded random game checked by the oracle at every ply; the turns it saw."""
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
        _walk(json.loads(WOLVES.read_text(encoding="utf-8")), seed, 120)

    @pytest.mark.slow
    def test_twenty_walks_see_turns(self) -> None:
        spec = json.loads(WOLVES.read_text(encoding="utf-8"))
        assert sum(_walk(spec, seed, 300) for seed in range(100, 120)) > 0

    def test_the_round_game_keys_alike(self) -> None:
        engine = Engine(json.loads(ROUNDS.read_text(encoding="utf-8")))
        rng = random.Random(5)
        state = engine.initial_state()
        turned = 0
        for _ in range(60):
            if engine.terminal(state) is not None:
                break
            first = rng.choice(engine.legal_moves_for(state, 0))
            second = rng.choice(engine.legal_moves_for(state, 1))
            state, phase, _ = engine.apply_round_with_phase(state, first, second)
            assert state.key == engine._compute_hash(state)
            turned += sum(1 for s in phase if s.kind == "turn")
