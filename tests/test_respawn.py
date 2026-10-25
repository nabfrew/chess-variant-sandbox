"""Wild pieces that come back (generation 36, N1, `docs/plan-npc.md` §4.1).

A taken wild piece of an entry with ``respawn: n`` returns to the first of its entry's
starts that is empty or holds a player's non-royal piece its type may take, ``n`` phases
after it was taken; a royal, a piece it may not take and another wild piece each hold a
start, and a held return is tried every phase. The clock is state, in the key.

Hand positions on a 6x6 board, one per holding rule, then the clock's timing and restart,
the key walked over seeded games against `_compute_hash`, make/unmake against a fresh
position per ply (perft), the inert spellings, the refusals and the words.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.describe import describe_spec

ROOT = Path(__file__).resolve().parents[1]
KING = [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]


def _spec(respawn: Any = 2, also_at: list[str] | None = None, route: list[str] | None = None) -> dict:
    """Kings on a1 and f6, and one wolf walking a3, b3, c3 (and back to a3). The wolf eats a
    king, a rook or a knight, never the stone ``S``."""
    npc: dict[str, Any] = {"piece": "W", "route": route or ["a3", "b3", "c3"]}
    if respawn is not None:
        npc["respawn"] = respawn
    if also_at:
        npc["also_at"] = also_at
    return {
        "schema_version": 31,
        "name": "respawn hand positions",
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


def _at(engine: Engine, state: GameState, square: str) -> tuple[str, int] | None:
    value = state.board[engine._square_index(square)]
    if not value:
        return None
    spec = engine.spec
    owner, rest = divmod(value - 1, spec.n_facings * spec.n_types)
    return spec.piece_ids[rest % spec.n_types], owner


def _play(engine: Engine, state: GameState, text: str) -> tuple[GameState, tuple]:
    after, phase = engine.apply_with_phase(state, engine.find_move(state, text))
    assert after.key == engine._compute_hash(after), text
    return after, phase


def _returns(phase: tuple) -> list:
    return [step for step in phase if step.kind == "return"]


class TestTheClock:
    def test_a_taken_wolf_comes_back_two_phases_later(self) -> None:
        """Taken by white's move, counted down by that move's phase, back in the next."""
        engine = _engine(respawn=2)
        state = engine.make_state({
            "a1": ("K", 0), "f6": ("K", 1), "c1": ("R", 0), "c3": ("W", 2),
        })
        state, phase = _play(engine, state, "Rc1xc3")
        assert _at(engine, state, "c3") == ("R", 0)
        assert _returns(phase) == [] and state.respawn == (1,)
        state, phase = _play(engine, state, "Kf6-f5")
        [back] = _returns(phase)
        assert (back.origin, back.target, back.eaten) == (engine._square_index("a3"),) * 2 + (0,)
        assert _at(engine, state, "a3") == ("W", 2)
        assert state.respawn == (0,)

    def test_one_phase_is_the_phase_of_the_capture(self) -> None:
        engine = _engine(respawn=1)
        state = engine.make_state({
            "a1": ("K", 0), "f6": ("K", 1), "c1": ("R", 0), "c3": ("W", 2),
        })
        state, phase = _play(engine, state, "Rc1xc3")
        assert len(_returns(phase)) == 1 and _at(engine, state, "a3") == ("W", 2)

    def test_the_clock_is_in_the_key(self) -> None:
        engine = _engine(respawn=4)
        board = {"a1": ("K", 0), "f6": ("K", 1)}
        keys = {engine.make_state(board, respawn=(clock,)).key for clock in range(5)}
        assert len(keys) == 5

    def test_it_returns_eating_a_piece_it_may_take(self) -> None:
        """White's knight goes onto a3 on the move the wolf is due back on, and is eaten."""
        engine = _engine(respawn=2)
        state = engine.make_state({
            "a1": ("K", 0), "f6": ("K", 1), "c4": ("R", 0), "b5": ("N", 0),
        }, respawn=(1,))
        state, phase = _play(engine, state, "Nb5-a3")
        [back] = _returns(phase)
        assert back.eaten == engine.spec.encode_piece(0, engine.spec.piece_ids.index("N"))
        assert _at(engine, state, "a3") == ("W", 2)
        assert state.halfmove == 0  # a return that eats is a capture


class TestWhatHoldsAReturnBack:
    @pytest.mark.parametrize(
        ("holder", "why"),
        [(("K", 0), "royal"), (("S", 0), "uncapturable")],
        ids=["royal", "uncapturable"],
    )
    def test_a_held_start_waits_and_tries_every_phase(self, holder: tuple[str, int], why: str) -> None:
        engine = _engine(respawn=2)
        board = {"f6": ("K", 1), "a3": holder, "e1": ("R", 0)}
        if holder[0] != "K":
            board["a1"] = ("K", 0)
        state = engine.make_state(board, respawn=(1,))
        state, phase = _play(engine, state, "Re1-e2")
        assert _returns(phase) == [], why
        assert state.respawn == (1,) and _at(engine, state, "a3") == holder
        state, phase = _play(engine, state, "Kf6-f5")
        assert _returns(phase) == [] and state.respawn == (1,)
        # The holder steps off, and the return comes in that move's phase.
        state, phase = _play(engine, state, f"{holder[0]}a3-a4")
        assert len(_returns(phase)) == 1 and state.respawn == (0,)
        assert _at(engine, state, "a3") == ("W", 2)

    def test_a_wild_piece_on_the_first_start_sends_it_to_the_next(self) -> None:
        """Two wolves on one walk (``also_at``); one taken, the other standing on a3 when
        the return is due, so it comes back on c3, the second start."""
        engine = _engine(respawn=1, also_at=["c3"])
        state = engine.make_state({
            "a1": ("K", 0), "f6": ("K", 1), "a5": ("N", 0), "c3": ("W", 2), "b3": ("W", 2),
        })
        state, phase = _play(engine, state, "Na5xb3")
        assert [(s.origin, s.target, s.kind) for s in phase] == [
            (engine._square_index("c3"), engine._square_index("a3"), "step"),
            (engine._square_index("c3"), engine._square_index("c3"), "return"),
        ]
        assert _at(engine, state, "a3") == ("W", 2) and _at(engine, state, "c3") == ("W", 2)

    def test_every_start_held_holds_it(self) -> None:
        engine = _engine(respawn=1, also_at=["c3"])
        state = engine.make_state({
            "a1": ("K", 0), "f6": ("K", 1), "a3": ("S", 0), "c3": ("S", 0), "b3": ("W", 2),
            "e1": ("R", 0),
        })
        state, phase = _play(engine, state, "Re1-e2")
        assert _returns(phase) == [] and state.respawn == (1,)


class TestTheClockStartsAgain:
    def test_two_missing_come_back_one_clock_apart(self) -> None:
        """Both wolves of a walk gone: one back after two phases, the clock starts again, the
        second back two phases after that, and the clock goes idle."""
        engine = _engine(respawn=2, also_at=["c3"])
        state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1)})
        returned_at = []
        moves = ["Ka1-a2", "Kf6-f5", "Ka2-a1", "Kf5-f6", "Ka1-b1", "Kf6-e6"]
        for index, text in enumerate(moves, start=1):
            state, phase = _play(engine, state, text)
            if _returns(phase):
                returned_at.append(index)
        assert returned_at == [2, 4]
        assert state.respawn == (0,)
        wolves = [sq for sq in range(engine.spec.n_squares) if state.board[sq] and
                  _at(engine, state, f"{'abcdef'[sq % 6]}{sq // 6 + 1}") == ("W", 2)]
        assert len(wolves) == 2


class TestAMirroredEntryHasTwoClocks:
    def test_each_half_counts_alone(self) -> None:
        spec = json.loads((ROOT / "tests" / "_respawn_wolf_hunt.json").read_text(encoding="utf-8"))
        compiled = compile_spec(spec)
        assert compiled.npcs is not None and len(compiled.npcs.respawn) == 4
        assert Engine(spec).initial_state().respawn == (0, 0, 0, 0)


def _walk(engine: Engine, seed: int, plies: int) -> int:
    rng = random.Random(seed)
    state = engine.initial_state()
    returns = 0
    for _ in range(plies):
        moves = engine.legal_moves(state)
        if engine.terminal(state, moves) is not None:
            break
        state, phase = engine.apply_with_phase(state, rng.choice(moves))
        assert state.key == engine._compute_hash(state)
        returns += len(_returns(phase))
    return returns


class TestTheKeyWalks:
    def test_the_hunt_returns_and_the_key_holds(self) -> None:
        engine = Engine(json.loads((ROOT / "tests" / "_respawn_wolf_hunt.json").read_text(encoding="utf-8")))
        assert sum(_walk(engine, seed, 200) for seed in range(3)) > 0

    def test_perft_agrees_with_a_fresh_position_per_ply(self) -> None:
        """make/unmake (the core's perft) against `apply` from a fresh snapshot each ply: a
        clock not put back by `unmake` moves the count."""
        engine = _engine(respawn=1)
        state = engine.make_state({
            "a1": ("K", 0), "f6": ("K", 1), "c1": ("R", 0), "c3": ("W", 2), "c5": ("N", 0),
        })

        def count(at: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(count(engine.apply(at, m), depth - 1) for m in engine.legal_moves(at))

        assert engine.perft(state, 3) == count(state, 3)


class TestRounds:
    def test_a_round_game_returns_and_the_key_holds(self) -> None:
        engine = Engine(json.loads((ROOT / "tests" / "_respawn_rounds.json").read_text(encoding="utf-8")))
        returns = 0
        for seed in range(6):
            rng = random.Random(seed)
            state = engine.initial_state()
            for _ in range(60):
                if engine.terminal(state) is not None:
                    break
                first = rng.choice(engine.legal_moves_for(state, 0))
                second = rng.choice(engine.legal_moves_for(state, 1))
                state, phase, _ = engine.apply_round_with_phase(state, first, second)
                assert state.key == engine._compute_hash(state)
                returns += len(_returns(phase))
        assert returns > 0


class TestInert:
    def test_zero_is_the_game_without_it(self) -> None:
        base = _spec(respawn=None)
        written = _spec(respawn=0)
        assert validate_spec(written) == []
        assert spec_hash(written) == spec_hash(base)
        assert canonical_tables(compile_spec(written)) == canonical_tables(compile_spec(base))

    def test_a_game_without_it_crosses_no_respawn_table(self) -> None:
        tables = canonical_tables(compile_spec(_spec(respawn=None)))
        assert "npc_respawn" not in tables and "zobrist_respawn" not in tables
        assert Engine(_spec(respawn=None)).initial_state().respawn == ()


class TestRefusals:
    @pytest.mark.parametrize("value", [17, -1, True, "8", 2.5])
    def test_out_of_range_or_not_a_count(self, value: Any) -> None:
        errors = validate_spec(_spec(respawn=value))
        assert any("respawn must be a whole number of phases from 1 to 16" in e for e in errors), errors

    def test_fairy_stockfish_refuses_it_with_every_wild_piece(self) -> None:
        with pytest.raises(Untranslatable) as caught:
            translate(_spec(respawn=4))
        assert any("rules.npcs" in reason for reason in caught.value.reasons)


class TestTheWords:
    def test_describe_says_when_and_where(self) -> None:
        text = describe_spec(_spec(respawn=4))
        assert "comes back on a3, 4 moves after it is taken" in text
        text = describe_spec(_spec(respawn=4, also_at=["c3"]))
        assert "on the first of a3, c3 that it may come back onto" in text

    def test_a_game_without_it_says_nothing_of_it(self) -> None:
        assert "comes back" not in describe_spec(_spec(respawn=None))

    def test_the_off_value_is_dropped_in_the_normal_form(self) -> None:
        spec = copy.deepcopy(_spec(respawn=0))
        assert spec_hash(spec) == spec_hash(_spec(respawn=None))
