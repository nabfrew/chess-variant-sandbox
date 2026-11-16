"""A pouncer that comes back (generation 124, SR8g, N7, `docs/plan-schema-next.md` §2.8).

Ruled 2026-10-27: *a pouncer respawns on its entry's first free start*. A pouncer leaves its
route and may pounce back onto another route of its type, so the board does not say which walk
it belongs to. Settled here: a pouncing entry is one count, its mirrored starts beside its own,
and its pieces are the wild pieces of its type anywhere on the board, the earlier pouncing
entries of the type claiming theirs first; one comes back on the entry's free start first in the
mover's reading order (as a pounce reads its prey), so mirrored positions return mirrored
pieces. A non-pouncing entry in a pounce game keeps its route's count and its first free start.

Hand positions on an 8x8 board, the lift and what stays refused, the words, a perft against the
oracle (`tests/pounce_oracle.py`, which plays step 3 since this generation) and seeded walks of
`tests/_pounce_hounds.json` with every entry coming back.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest
from pounce_oracle import PounceOracle

from chess_variant_sandbox import fuzz
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.coming import coming_phases
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.describe import describe_spec

ROOT = Path(__file__).resolve().parents[1]
HOUNDS = ROOT / "tests" / "_pounce_hounds.json"
KING = [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}]
FILES = "abcdefgh"


def _spec(win: str = "royal_capture", every: int = 2) -> dict[str, Any]:
    """Kings, a knight and a rook for the players; a hound ``H`` (king's moves) that pounces on
    its prey, on c3-d3 mirrored onto c6-d6, coming back after ``every`` phases; a wolf ``W`` that
    does not pounce on f4-f5, coming back after 2."""
    return copy.deepcopy({
        "schema_version": 31,
        "name": "respawning pouncers",
        "board": {"width": 8, "height": 8},
        "pieces": [
            {"id": "K", "name": "king", "royal": True, "value": 0, "moves": KING},
            {"id": "N", "name": "knight", "value": 3,
             "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}]},
            {"id": "R", "name": "rook", "value": 5,
             "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "dihedral8"}]},
            {"id": "H", "name": "hound", "value": 2, "moves": KING},
            {"id": "W", "name": "wolf", "value": 3, "moves": KING},
        ],
        "setup": {"mirror": True, "placements": [
            {"piece": "K", "player": 0, "squares": ["e1"]},
            {"piece": "N", "player": 0, "squares": ["b1"]},
        ]},
        "rules": {"win": win, "stalemate": "draw", "max_plies": 200, "npcs": [
            {"piece": "H", "route": ["c3", "d3"], "mirror": True, "pounce": "prey", "respawn": every},
            {"piece": "W", "route": ["f4", "f5"], "respawn": 2},
        ]},
    })


def _engine(**kwargs: Any) -> Engine:
    spec = _spec(**kwargs)
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _name(square: int) -> str:
    return FILES[square % 8] + str(square // 8 + 1)


def _log(phase: Any) -> list[tuple[str, str, str]]:
    return [(_name(s.origin), _name(s.target), s.kind) for s in phase]


def _play(engine: Engine, state: GameState, text: str) -> tuple[GameState, list[tuple[str, str, str]]]:
    after, phase = engine.apply_with_phase(state, engine.find_move(state, text))
    assert after.key == engine._compute_hash(after), text
    return after, _log(phase)


KINGS = {"a1": ("K", 0), "h8": ("K", 1)}
WOLF = {"f4": ("W", 2)}


class TestTheCount:
    def test_one_count_per_pouncing_entry_and_one_per_wolf_half(self) -> None:
        engine = _engine()
        assert engine.initial_state().respawn == (0, 0)
        assert [len(starts) for _, starts in engine.spec.npcs.respawn] == [2, 1]

    def test_a_pouncer_off_its_route_still_counts(self) -> None:
        engine = _engine()
        state = engine.make_state({**KINGS, **WOLF, "e5": ("H", 2), "c6": ("H", 2)})
        state, log = _play(engine, state, "Ka1-a2")
        assert state.respawn[0] == 0
        assert all(kind != "return" for _, _, kind in log)

    def test_one_short_starts_the_clock(self) -> None:
        engine = _engine()
        state = engine.make_state({**KINGS, **WOLF, "e5": ("H", 2)})
        state, _ = _play(engine, state, "Ka1-a2")
        assert state.respawn[0] == 1
        state, log = _play(engine, state, "Kh8-h7")
        assert ("c6", "c6", "return") in log
        assert state.respawn[0] == 0


class TestTheStart:
    @pytest.mark.parametrize(("side", "move", "start"), [(0, "Ka1-a2", "c3"), (1, "Kh8-h7", "c6")])
    def test_the_free_start_nearest_the_mover(self, side: int, move: str, start: str) -> None:
        engine = _engine()
        state = engine.make_state({**KINGS, **WOLF, "e5": ("H", 2)}, side_to_move=side, respawn=(1, 0))
        state, log = _play(engine, state, move)
        assert (start, start, "return") in log
        square = FILES.index(start[0]) + 8 * (int(start[1]) - 1)
        assert state.board[square] == engine.spec.encode_piece(2, engine.spec.piece_ids.index("H"))

    def test_a_royal_holds_the_nearest_start(self) -> None:
        engine = _engine()
        state = engine.make_state(
            {"c3": ("K", 0), "h8": ("K", 1), **WOLF, "e5": ("H", 2)}, respawn=(1, 0)
        )
        _, log = _play(engine, state, "Kc3-b2")
        # The king left c3 in its move, so c3 is free again when the hound comes back.
        assert ("c3", "c3", "return") in log
        state = engine.make_state(
            {"c3": ("K", 0), "h8": ("K", 1), "a1": ("R", 0), **WOLF, "e5": ("H", 2)}, respawn=(1, 0)
        )
        _, log = _play(engine, state, "Ra1-a2")
        assert ("c6", "c6", "return") in log

    def test_it_eats_a_piece_on_the_start(self) -> None:
        engine = _engine()
        state = engine.make_state({**KINGS, **WOLF, "e5": ("H", 2), "b1": ("N", 0)}, respawn=(1, 0))
        after, phase = engine.apply_with_phase(state, engine.find_move(state, "Nb1-c3"))
        returns = [step for step in phase if step.kind == "return"]
        assert [(_name(s.target), s.eaten) for s in returns] == [
            ("c3", engine.spec.encode_piece(0, engine.spec.piece_ids.index("N")))
        ]

    def test_the_wolf_keeps_its_routes_count_and_its_first_start(self) -> None:
        engine = _engine()
        state = engine.make_state({**KINGS, "c6": ("H", 2), "c3": ("H", 2)}, side_to_move=1, respawn=(0, 1))
        _, log = _play(engine, state, "Kh8-h7")
        assert ("f4", "f4", "return") in log


class TestTheOtherReaders:
    def test_coming_shows_the_return(self) -> None:
        engine = _engine()
        state = engine.make_state({**KINGS, **WOLF, "e5": ("H", 2)}, respawn=(1, 0))
        first = _log(coming_phases(engine, state, 1)[0])
        assert ("c3", "c3", "return") in first

    def test_unmake_puts_the_clocks_back(self) -> None:
        engine = _engine(win="checkmate")
        state = engine.make_state(
            {"e1": ("K", 0), "e8": ("K", 1), "e5": ("H", 2), "b1": ("N", 0), "g6": ("R", 1), **WOLF},
            respawn=(2, 0),
        )
        oracle = PounceOracle(_spec(win="checkmate"))
        assert engine.perft(state, 3) == _oracle_perft(oracle, state, 3)


def _oracle_perft(oracle: PounceOracle, state: GameState, depth: int) -> int:
    """The oracle's count: its legal list at every node, every move's phase, board and clocks
    held to the core's."""
    if depth == 0:
        return 1
    engine = oracle.engine
    if engine.terminal(state) is not None:
        return 0
    moves = engine.legal_moves(state)
    assert moves == oracle.legal_moves(state)
    return sum(_oracle_perft(oracle, oracle.check(state, move), depth - 1) for move in moves)


class TestTheLift:
    def test_respawn_beside_a_pounce_validates(self) -> None:
        assert validate_spec(_spec()) == []

    def test_split_beside_respawn_stays_refused(self) -> None:
        spec = _spec()
        spec["rules"]["npcs"][0].update(pounce="attackers", split=4)
        assert any("respawn beside split" in e for e in validate_spec(spec))

    def test_the_inventory_names_it(self) -> None:
        from tests.feature_inventory import _schema_one_hundred_twenty_four_uses

        assert _schema_one_hundred_twenty_four_uses(_spec()) == {"rules.npcs.respawn.pounce"}
        spec = _spec()
        spec["rules"]["npcs"][0].pop("respawn")
        spec["rules"]["npcs"][1].pop("respawn")
        assert _schema_one_hundred_twenty_four_uses(spec) == set()

    def test_describe_says_it_counts_by_kind(self) -> None:
        text = describe_spec(_spec())
        assert "counted by its kind" in text
        assert "nearest the first rank of the player who has just moved" in text
        assert "both walks share one count" in text


def _hounds_respawning() -> dict[str, Any]:
    spec = json.loads(HOUNDS.read_text(encoding="utf-8"))
    for every, npc in zip((3, 5, 4), spec["rules"]["npcs"], strict=True):
        npc["respawn"] = every
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def _walk(spec: dict[str, Any], seed: int, plies: int) -> int:
    """One seeded random game held to the oracle at every ply; the returns it saw."""
    oracle = PounceOracle(spec)
    engine = oracle.engine
    rng = random.Random(seed)
    state = engine.initial_state()
    returns = 0
    for _ in range(plies):
        if engine.terminal(state) is not None:
            break
        moves = engine.legal_moves(state)
        assert moves == oracle.legal_moves(state)
        assert engine.in_check(state) == oracle.in_check(state)
        move = rng.choice(moves)
        after = oracle.check(state, move)
        assert after.key == engine._compute_hash(after)
        fuzz.every_npc_stands_on_its_own_route(engine, after)
        returns += sum(1 for s in engine.apply_with_phase(state, move)[1] if s.kind == "return")
        state = after
    return returns


class TestTheOracle:
    @pytest.mark.parametrize("seed", [1, 2])
    def test_a_seeded_walk(self, seed: int) -> None:
        _walk(_hounds_respawning(), seed, 80)

    @pytest.mark.slow
    def test_twenty_walks_see_returns(self) -> None:
        spec = _hounds_respawning()
        assert sum(_walk(spec, seed, 200) for seed in range(200, 220)) > 20

    def test_a_perft_from_the_opening(self) -> None:
        spec = _hounds_respawning()
        oracle = PounceOracle(spec)
        state = oracle.engine.initial_state()
        assert oracle.engine.perft(state, 2) == _oracle_perft(oracle, state, 2)
