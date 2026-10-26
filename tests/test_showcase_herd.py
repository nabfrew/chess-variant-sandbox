"""N9: *the herd*, the second showcase of a wild piece's sight (freeze, disarm, shield) and of the duck
that walks by its moves from a start square (`docs/plan-npc.md` §6.4).

Chess on 8 by 9 won by taking the king, stalemate a loss, a duck that starts on e5 and walks a king's
step, and four wild pieces that eat nothing, touch only the four squares beside them and pace their
lines: a gorgon freezing on a5-c5 and a siren disarming on h5-f5, alone on the middle rank, and two
dryads shielding on c4-f4 and c6-f6, all turning back at a block and returning eight phases after
they are taken. This is V1 on the file itself:

- after every ply of seeded random games (3 fast, 20 slow) the core's legal list equals the sight
  oracle's (`tests/sight_oracle.py`); the duck stands on e5 at the start, and every duck move is one
  king's step onto an empty square, or the pass when it has none; no player's move starts on a
  frozen piece, no disarmed piece captures, and no capture lands on a shielded piece; no wild piece
  steps onto the duck or eats anything; every square a wild sight touches is an orthogonal
  neighbour of a wild piece of that effect;
- hand positions: the gorgon turned by the duck; a piece the gorgon freezes held, and free once it
  walks on, while its diagonal neighbour is never touched; a capture barred by the siren and one
  barred by a dryad (a diagonal neighbour of the dryad still taken); a side frozen out of every move
  losing; the boxed duck's pass.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.compile import SIGHT_DISARM, SIGHT_FREEZE, SIGHT_SHIELD
from chess_variant_sandbox.engine.moves import DUCK_PASS, MOVE_DUCK, decode_move
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from tests.sight_oracle import SightOracle, normalise

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "archive" / "the-herd.json"
WILD = 2
EFFECT_OF = {"G": SIGHT_FREEZE, "S": SIGHT_DISARM, "Y": SIGHT_SHIELD}


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
    assert spec_hash(spec) == "2135109e8191"
    assert spec["board"] == {"width": 8, "height": 9}
    rules = spec["rules"]
    assert rules["win"] == "royal_capture" and rules["stalemate"] == "loss"
    assert rules["max_plies"] == 600 and rules["no_progress_draw"] == 100
    assert rules["scoring"] == {"count": "material"}
    assert rules["duck"] == {"piece": "D", "start": "e5", "goes": "by_moves"}
    lines = {
        "G": (["a5", "b5", "c5"], False),
        "S": (["h5", "g5", "f5"], False),
        "Y": (["c4", "d4", "e4", "f4"], True),
    }
    assert sorted(entry["piece"] for entry in rules["npcs"]) == ["G", "S", "Y"]
    for entry in rules["npcs"]:
        route, mirrored = lines[entry["piece"]]
        assert entry["route"] == route and entry.get("mirror", False) is mirrored
        assert entry["turns_back"] is True and entry["respawn"] == 8
    effects = {"G": "freeze", "S": "disarm", "Y": "shield"}
    for piece in spec["pieces"]:
        if piece["id"] in effects:
            [sight] = piece["sight"]
            assert sight["effect"] == effects[piece["id"]] and sight["whose"] == "any"
            assert sight["vectors"] == [[1, 0]] and sight["symmetry"] == "rotate4"
            assert sight["range"] == 1 and sight["through"] is True
            assert "affects" not in sight  # the kings included, on purpose (§6.4)
            assert piece["captures"] == []
    duck = next(p for p in spec["pieces"] if p["id"] == "D")
    assert "sight" not in duck and [m["mode"] for m in duck["moves"]] == ["move"]


def _king_step(engine: Engine, origin: int, target: int) -> bool:
    width = engine.spec.width
    df, dr = abs(origin % width - target % width), abs(origin // width - target // width)
    return max(df, dr) == 1


def _orthogonal(width: int, a: int, b: int) -> bool:
    return abs(a % width - b % width) + abs(a // width - b // width) == 1


def _touched_only_beside(engine: Engine, state: GameState) -> None:
    spec = engine.spec
    width = spec.width
    wild = engine.effects(state).wild
    sources = {
        square: EFFECT_OF[spec.piece_ids[spec.type_of(v)]]
        for square, v in enumerate(state.board)
        if v and spec.owner_of(v) == WILD and spec.piece_ids[spec.type_of(v)] in EFFECT_OF
    }
    for square, bits in enumerate(wild):
        for effect in (SIGHT_FREEZE, SIGHT_DISARM, SIGHT_SHIELD):
            if bits & effect:
                assert any(
                    kind == effect and _orthogonal(width, source, square)
                    for source, kind in sources.items()
                ), f"an effect on square {square} with no creature beside it"


def _walk(seed: int, plies: int = 240) -> None:
    engine = _engine()
    oracle = SightOracle(_file())
    spec = engine.spec
    rng = random.Random(seed)
    state = engine.initial_state()
    assert engine.duck_square(state) == _sq(engine, "e5")
    for _ in range(plies):
        if engine.terminal(state) is not None:
            break
        moves = engine.legal_moves(state)
        assert normalise(moves) == normalise(oracle.legal_moves(state))
        _touched_only_beside(engine, state)
        bits = engine.effects(state).bits
        duck = engine.duck_square(state)
        for move in moves:
            origin, target, kind, how = decode_move(move)
            if kind == MOVE_DUCK:
                if how == DUCK_PASS:
                    assert moves == [move], "a pass beside another duck move"
                else:
                    assert origin == duck and not state.board[target]
                    assert _king_step(engine, origin, target)
                continue
            assert not bits[origin] & SIGHT_FREEZE, "a frozen piece moved"
            victim = state.board[target]
            if victim and spec.owner_of(victim) != state.side_to_move:
                assert not bits[origin] & SIGHT_DISARM, "a disarmed piece captured"
                assert not bits[target] & SIGHT_SHIELD, "a shielded piece was taken"
        move = rng.choice(moves)
        after, phase = engine.apply_with_phase(state, move)
        duck = engine.duck_square(after)
        for step in phase:
            assert step.target != duck, "a wild piece stepped onto the duck"
            assert step.eaten == 0, "a wild piece ate"
        state = after


@pytest.mark.parametrize("seed", range(3))
def test_a_seeded_random_walk(seed: int) -> None:
    _walk(seed)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(3, 23))
def test_twenty_seeded_random_games(seed: int) -> None:
    _walk(seed, plies=600)


def _play(engine: Engine, state: GameState, text: str) -> tuple[GameState, list[Any]]:
    after, phase = engine.apply_with_phase(state, engine.find_move(state, text))
    assert after.key == engine._compute_hash(after), text
    return after, list(phase)


# The wild pieces every hand position leaves out are away; their clocks start in the first phase
# and bring them back eight phases on, after every position here is done.
KINGS = {"e1": ("K", 0), "e9": ("K", 1)}


class TestTheHandPositions:
    def test_the_gorgon_turned_by_the_duck(self) -> None:
        engine = _engine()
        state = engine.make_state(
            {**KINGS, "b5": ("G", 2), "h1": ("R", 0), "d5": ("D", 2)}, duck_owed=True)
        after, phase = _play(engine, state, "the duck to c5")
        [step] = [s for s in phase if s.origin == _sq(engine, "b5")]
        assert step.kind == "turn"
        assert _at(engine, after, "a5") == ("G", 2) and _at(engine, after, "c5") == ("D", 2)

    def test_a_frozen_piece_held_and_freed_and_the_diagonal_untouched(self) -> None:
        engine = _engine()
        board = {**KINGS, "b5": ("G", 2), "b4": ("N", 0), "c4": ("B", 0), "h8": ("D", 2)}
        state = engine.make_state(board)
        effects = engine.effects(state)
        assert effects.bits[_sq(engine, "b4")] & SIGHT_FREEZE and effects.frozen > 0
        assert not effects.bits[_sq(engine, "c4")] & SIGHT_FREEZE  # a diagonal neighbour
        moving = {decode_move(m)[0] for m in engine.legal_moves(state)}
        assert _sq(engine, "b4") not in moving and _sq(engine, "c4") in moving
        # A turn on: the gorgon walks to c5, beside the bishop and diagonal to the knight.
        after, _ = _play(engine, state, "Ke1-f1")
        after, _ = _play(engine, after, "the duck to h7")
        assert _at(engine, after, "c5") == ("G", 2)
        bits = engine.effects(after).bits
        assert not bits[_sq(engine, "b4")] & SIGHT_FREEZE
        assert bits[_sq(engine, "c4")] & SIGHT_FREEZE
        # Black to move here; the same board with White to move shows who is free.
        white = engine.make_state({
            "f1": ("K", 0), "e9": ("K", 1), "c5": ("G", 2), "b4": ("N", 0), "c4": ("B", 0),
            "h7": ("D", 2)}, heading_back=["c5"])
        moving = {decode_move(m)[0] for m in engine.legal_moves(white)}
        assert _sq(engine, "b4") in moving and _sq(engine, "c4") not in moving

    def test_a_capture_barred_by_the_siren(self) -> None:
        engine = _engine()
        board = {**KINGS, "g5": ("S", 2), "f5": ("R", 0), "f8": ("Q", 1), "h8": ("D", 2)}
        state = engine.make_state(board)
        assert engine.effects(state).bits[_sq(engine, "f5")] & SIGHT_DISARM
        with pytest.raises(LookupError):
            engine.find_move(state, "Rf5xf8")
        assert engine.find_move(state, "Rf5-f6")  # disarmed, it still moves

    def test_a_capture_barred_by_a_dryad_and_a_diagonal_taken(self) -> None:
        engine = _engine()
        board = {
            **KINGS, "d4": ("Y", 2), "e4": ("N", 1), "c3": ("N", 1),
            "e2": ("R", 0), "c1": ("R", 0), "h8": ("D", 2),
        }
        state = engine.make_state(board)
        bits = engine.effects(state).bits
        assert bits[_sq(engine, "e4")] & SIGHT_SHIELD and not bits[_sq(engine, "c3")] & SIGHT_SHIELD
        assert engine.effects(state).shielded > 0
        with pytest.raises(LookupError):
            engine.find_move(state, "Re2xe4")
        assert engine.find_move(state, "Rc1xc3")

    def test_a_side_frozen_out_of_every_move_loses(self) -> None:
        engine = _engine()
        board = {"b4": ("K", 0), "e9": ("K", 1), "b5": ("G", 2), "h8": ("D", 2)}
        state = engine.make_state(board)
        assert engine.effects(state).bits[_sq(engine, "b4")] & SIGHT_FREEZE
        assert engine.legal_moves(state) == []
        outcome = engine.terminal(state)
        assert outcome is not None and outcome.winner == 1

    def test_the_boxed_duck_passes(self) -> None:
        engine = _engine()
        board = {**KINGS, "a9": ("D", 2), "a8": ("P", 1), "b8": ("P", 1), "b9": ("N", 1)}
        state = engine.make_state(board, duck_owed=True)
        [move] = engine.legal_moves(state)
        assert decode_move(move)[2:] == (MOVE_DUCK, DUCK_PASS)
        after = engine.apply(state, move)
        assert engine.duck_square(after) == _sq(engine, "a9") and after.side_to_move == 1
