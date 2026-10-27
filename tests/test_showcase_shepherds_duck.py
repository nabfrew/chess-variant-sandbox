"""N6: *the shepherd's duck*, the showcase of the duck, its shield, turning back and respawn
(`docs/archive/plan-npc.md` §6.1).

Chess won by taking the king, with the classic duck shielding the eight squares round it and a
mirrored wolf pair pacing ranks 4 and 5 end to end, turning back at what it may not eat and
returning to a4 and a5 eight phases after it is taken, for a point. This is V1 on the file itself:

- after every ply of seeded random games (3 fast, 20 slow) the core's legal list equals the
  sight oracle's (`tests/sight_oracle.py`, which works the shield out from the forward lines and
  the sight-stripped twin's lists); no capture a player makes lands on a shielded piece, and no
  wolf eats a piece shielded at its phase's start; no wolf steps onto the duck; and every
  return lands on a4 or a5 exactly eight phases after the capture (the capture's own phase the
  first, so seven phases later) or, where the start was held then, at the first phase after
  that found it free;
- hand positions: a wolf turned by the duck, a wolf turned by a pawn the duck shields (and,
  with the duck elsewhere, the same pawn eaten), and a return held back by a king and by the
  duck standing on the start.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.compile import SIGHT_SHIELD
from chess_variant_sandbox.engine.moves import MOVE_DUCK, decode_move
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from tests.sight_oracle import SightOracle, normalise

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "archive" / "the-shepherds-duck.json"
RESPAWN = 8


def _file() -> dict[str, Any]:
    return copy.deepcopy(load_spec(FILE))


def _engine() -> Engine:
    return Engine(_file())


def _sq(engine: Engine, name: str) -> int:
    return engine._square_index(name)


def _at(engine: Engine, state: GameState, name: str) -> tuple[str, int] | None:
    value = state.board[_sq(engine, name)]
    if not value:
        return None
    spec = engine.spec
    return spec.piece_ids[spec.type_of(value)], spec.owner_of(value)


def test_the_file_is_the_registered_draft() -> None:
    spec = _file()
    assert validate_spec(spec) == []
    assert spec_hash(spec) == "e7d2d7406586"
    rules = spec["rules"]
    assert rules["win"] == "royal_capture" and rules["max_plies"] == 600
    assert rules["no_progress_draw"] == 100 and rules["scoring"] == {"count": "points"}
    # `goes: "anywhere"`, as registered, is the default the current spelling leaves out.
    assert rules["duck"] == {"piece": "D"}
    [wolves] = rules["npcs"]
    assert wolves == {
        "piece": "W", "route": [f"{f}4" for f in "abcdefgh"], "mirror": True,
        "turns_back": True, "respawn": RESPAWN, "bounty": 1,
    }
    duck = next(p for p in spec["pieces"] if p["id"] == "D")
    assert duck["moves"] == [] and duck["sight"][0]["effect"] == "shield"
    assert duck["sight"][0]["whose"] == "any" and duck["sight"][0]["through"] is True


def _before_phase(after: GameState, phase: Any) -> bytes:
    """The board the phase started on: the core's phase log taken back, newest first."""
    board = bytearray(after.board)
    for step in reversed(phase):
        if step.kind == "return":
            board[step.target] = step.eaten
        elif step.origin != step.target:
            board[step.origin] = board[step.target]
            board[step.target] = step.eaten
    return bytes(board)


def _held(engine: Engine, board: bytes, square: int, effects: tuple[int, ...]) -> bool:
    """Whether a return onto ``square`` is held back there: a royal, the duck, a wild piece
    or a shielded piece stands on it (a wolf eats every type, so nothing else holds it)."""
    value = board[square]
    if not value:
        return False
    spec = engine.spec
    return (
        spec.owner_of(value) == 2
        or spec.type_of(value) in spec.royal_types
        or bool(effects[square] & SIGHT_SHIELD)
    )


def _walk(seed: int, plies: int = 240) -> None:
    engine = _engine()
    oracle = SightOracle(_file())
    spec = engine.spec
    rng = random.Random(seed)
    starts = {_sq(engine, "a4"): 4, _sq(engine, "a5"): 5}
    # Per half (by rank): the phase the wolf was taken in, while it is away.
    taken: dict[int, int] = {}
    phases = 0
    state = engine.initial_state()
    for _ in range(plies):
        if engine.terminal(state) is not None:
            break
        moves = engine.legal_moves(state)
        assert normalise(moves) == normalise(oracle.legal_moves(state))
        move = rng.choice(moves)
        origin, target, kind, _ = decode_move(move)
        effects = engine.effects(state).bits
        victim = state.board[target]
        if kind != MOVE_DUCK and victim and spec.owner_of(victim) != state.side_to_move:
            assert not effects[target] & SIGHT_SHIELD, "a player took a shielded piece"
        wolf_taken = (
            kind != MOVE_DUCK and victim and spec.owner_of(victim) == 2
            and spec.type_of(victim) != spec.duck.piece_type
        )
        after, phase = engine.apply_with_phase(state, move)
        if wolf_taken:
            # Taken on the piece move; that turn's phase, after the duck half, is the first.
            taken[target // spec.width + 1] = phases + 1
        if phase or engine.duck_owed(state):
            phases += 1
            before = _before_phase(after, phase)
            shields = engine.effects(dataclasses.replace(after, board=before)).bits
            duck = engine.duck_square(after)
            for step in phase:
                assert step.target != duck, "a wolf stepped onto the duck"
                if step.eaten:
                    assert not shields[step.target] & SIGHT_SHIELD, "a wolf ate a shielded piece"
                if step.kind == "return":
                    rank = starts[step.target]
                    assert rank in taken, "a return with no wolf away"
                    assert phases - taken.pop(rank) >= RESPAWN - 1
            for square, rank in starts.items():
                if rank in taken and phases - taken[rank] >= RESPAWN - 1:
                    # Due: it came back, or its start was held at the phase's start.
                    assert _held(engine, before, square, shields), (seed, phases, rank)
        state = after


@pytest.mark.parametrize("seed", range(3))
def test_a_seeded_random_walk(seed: int) -> None:
    _walk(seed)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(3, 23))
def test_twenty_seeded_random_games(seed: int) -> None:
    _walk(seed, plies=600)


def _phase(engine: Engine, state: GameState, text: str) -> tuple[GameState, list[Any]]:
    after, phase = engine.apply_with_phase(state, engine.find_move(state, text))
    assert after.key == engine._compute_hash(after), text
    return after, list(phase)


class TestTheHandPositions:
    def test_a_wolf_turned_by_the_duck(self) -> None:
        engine = _engine()
        state = engine.make_state(
            {"e1": ("K", 0), "e8": ("K", 1), "d4": ("W", 2), "h1": ("R", 0), "b7": ("D", 2)},
            duck_owed=True,
        )
        after, phase = _phase(engine, state, "the duck to e4")
        [step] = phase
        assert step.kind == "turn"
        assert (step.origin, step.target) == (_sq(engine, "d4"), _sq(engine, "c4"))
        assert _at(engine, after, "c4") == ("W", 2) and _at(engine, after, "e4") == ("D", 2)

    def test_a_wolf_turned_by_a_shielded_pawn(self) -> None:
        engine = _engine()
        board = {"e1": ("K", 0), "e8": ("K", 1), "d4": ("W", 2), "e4": ("P", 0), "b7": ("D", 2)}
        state = engine.make_state(board, duck_owed=True)
        after, phase = _phase(engine, state, "the duck to f3")
        [step] = phase
        assert step.kind == "turn" and step.eaten == 0
        assert _at(engine, after, "c4") == ("W", 2) and _at(engine, after, "e4") == ("P", 0)
        # The duck out of reach: the same pawn is eaten.
        after, phase = _phase(engine, state, "the duck to h8")
        [step] = phase
        assert step.kind == "step" and step.eaten
        assert _at(engine, after, "e4") == ("W", 2)

    def test_a_return_held_by_a_king(self) -> None:
        engine = _engine()
        board = {"a4": ("K", 0), "e8": ("K", 1), "h1": ("R", 0), "f5": ("W", 2), "c7": ("D", 2)}
        # The rank-4 wolf is away, due in this turn's phase.
        state = engine.make_state(board, respawn=(1, 0), duck_owed=True)
        after, phase = _phase(engine, state, "the duck to c6")
        assert not [s for s in phase if s.kind == "return"]
        assert _at(engine, after, "a4") == ("K", 0) and after.respawn[0] == 1
        after, _ = _phase(engine, after, "Ke8-d8")
        after, phase = _phase(engine, after, "the duck to c5")
        assert not [s for s in phase if s.kind == "return"], "the king still holds a4"
        after, _ = _phase(engine, after, "Ka4-b3")
        after, phase = _phase(engine, after, "the duck to c6")
        [back] = [s for s in phase if s.kind == "return"]
        assert back.target == _sq(engine, "a4") and _at(engine, after, "a4") == ("W", 2)

    def test_a_return_held_by_the_duck(self) -> None:
        engine = _engine()
        board = {"e1": ("K", 0), "e8": ("K", 1), "h1": ("R", 0), "f5": ("W", 2), "c7": ("D", 2)}
        state = engine.make_state(board, respawn=(1, 0), duck_owed=True)
        after, phase = _phase(engine, state, "the duck to a4")
        assert not [s for s in phase if s.kind == "return"]
        assert _at(engine, after, "a4") == ("D", 2)
        after, _ = _phase(engine, after, "Ke8-d8")
        # The duck may not stay, so the start is free at the end of every later turn.
        after, phase = _phase(engine, after, "the duck to b1")
        [back] = [s for s in phase if s.kind == "return"]
        assert back.target == _sq(engine, "a4") and _at(engine, after, "a4") == ("W", 2)
