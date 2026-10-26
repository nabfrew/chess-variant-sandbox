"""N8: *the menagerie*, the showcase of a wild piece's sight (freeze, disarm, shield) and of the duck
that walks by its moves from a start square (`docs/plan-npc.md` §6.3).

Chess on 8 by 9 won by taking the king, a duck that starts on e5 and walks a king's step, and six
wild pieces that eat nothing and pace their lines: gorgons freezing the eight squares round them,
sirens disarming them and dryads shielding them, all turning back at a block and returning eight
phases after they are taken. Measured and retired (0.816 against V3's 0.840), it is kept as
`tests/_retired_the_menagerie.json`. This is V1 on the file itself:

- after every ply of seeded random games (3 fast, 20 slow) the core's legal list equals the sight
  oracle's (`tests/sight_oracle.py`, which works the effects out from the forward lines and the
  sight-stripped twin's lists); the duck stands on e5 at the start, and every duck move is one
  king's step onto an empty square, or the pass when it has none; no player's move starts on a
  frozen piece, no disarmed piece captures, and no capture lands on a shielded piece; no wild
  piece steps onto the duck or eats anything;
- hand positions: a gorgon turned by the duck; a piece a gorgon freezes held, and free once the
  gorgon walks on; a capture barred by a siren and one barred by a dryad; the boxed duck's pass.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.compile import SIGHT_DISARM, SIGHT_FREEZE, SIGHT_SHIELD
from chess_variant_sandbox.engine.moves import DUCK_PASS, MOVE_DUCK, decode_move
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from tests.retired_specs import retired_preview_path
from tests.sight_oracle import SightOracle, normalise

# Built, measured and retired by the call registered before its first measured game (the index
# missed V3's 0.840, §6.3), so the file is kept in `tests/` (`retired_specs.RETIRED_PREVIEWS`);
# the mechanism it shows still holds and is held here.
FILE = retired_preview_path("the-menagerie")


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
    assert spec_hash(spec) == "3ab2ce41d4c9"
    assert spec["board"] == {"width": 8, "height": 9}
    rules = spec["rules"]
    assert rules["win"] == "royal_capture" and rules["stalemate"] == "draw"
    assert rules["max_plies"] == 600 and rules["no_progress_draw"] == 100
    assert rules["scoring"] == {"count": "material"}
    assert rules["duck"] == {"piece": "D", "start": "e5", "goes": "by_moves"}
    lines = {
        "G": [["a4", "b4", "c4", "d4"]],
        "S": [["h4", "g4", "f4", "e4"]],
        "Y": [["a5", "b5", "c5"], ["h5", "g5", "f5"]],
    }
    for entry in rules["npcs"]:
        assert entry["route"] in lines[entry["piece"]]
        assert entry["turns_back"] is True and entry["respawn"] == 8
        assert entry.get("mirror", False) is (entry["piece"] != "Y")
    effects = {"G": "freeze", "S": "disarm", "Y": "shield"}
    for piece in spec["pieces"]:
        if piece["id"] in effects:
            [sight] = piece["sight"]
            assert sight["effect"] == effects[piece["id"]] and sight["whose"] == "any"
            assert sight["range"] == 1 and sight["through"] is True
            assert piece["captures"] == []
    duck = next(p for p in spec["pieces"] if p["id"] == "D")
    assert "sight" not in duck and [m["mode"] for m in duck["moves"]] == ["move"]


def _king_step(engine: Engine, origin: int, target: int) -> bool:
    width = engine.spec.width
    df, dr = abs(origin % width - target % width), abs(origin // width - target // width)
    return max(df, dr) == 1


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
    def test_a_gorgon_turned_by_the_duck(self) -> None:
        engine = _engine()
        state = engine.make_state(
            {**KINGS, "b4": ("G", 2), "h1": ("R", 0), "d5": ("D", 2)}, duck_owed=True)
        after, phase = _play(engine, state, "the duck to c4")
        [step] = [s for s in phase if s.origin == _sq(engine, "b4")]
        assert step.kind == "turn"
        assert _at(engine, after, "a4") == ("G", 2) and _at(engine, after, "c4") == ("D", 2)

    def test_a_frozen_piece_held_and_freed(self) -> None:
        engine = _engine()
        board = {**KINGS, "c4": ("G", 2), "c3": ("N", 0), "h8": ("D", 2)}
        state = engine.make_state(board)
        frozen = engine.effects(state)
        assert frozen.bits[_sq(engine, "c3")] & SIGHT_FREEZE and frozen.frozen > 0
        assert not [m for m in engine.legal_moves(state) if decode_move(m)[0] == _sq(engine, "c3")]
        # A turn on: the gorgon walks to d4, and the knight on c3 is still beside it.
        after, _ = _play(engine, state, "Ke1-f1")
        after, _ = _play(engine, after, "the duck to h7")
        assert _at(engine, after, "d4") == ("G", 2)
        assert engine.effects(after).bits[_sq(engine, "c3")] & SIGHT_FREEZE
        after, _ = _play(engine, after, "Ke9-f9")
        after, _ = _play(engine, after, "the duck to h8")
        # d4 is the line's end: the gorgon turned back to c4 and holds the knight again.
        assert _at(engine, after, "c4") == ("G", 2)
        assert engine.effects(after).bits[_sq(engine, "c3")] & SIGHT_FREEZE
        after, _ = _play(engine, after, "Kf1-e1")
        after, _ = _play(engine, after, "the duck to h7")
        after, _ = _play(engine, after, "Kf9-e9")
        after, _ = _play(engine, after, "the duck to h8")
        # Two phases on it walked to a4 by b4; the knight on c3 is out of reach and moves.
        assert _at(engine, after, "a4") == ("G", 2)
        assert not engine.effects(after).bits[_sq(engine, "c3")] & SIGHT_FREEZE
        assert [m for m in engine.legal_moves(after) if decode_move(m)[0] == _sq(engine, "c3")]

    def test_a_capture_barred_by_a_siren(self) -> None:
        engine = _engine()
        board = {**KINGS, "f4": ("S", 2), "e3": ("R", 0), "e7": ("Q", 1), "h8": ("D", 2)}
        state = engine.make_state(board)
        assert engine.effects(state).bits[_sq(engine, "e3")] & SIGHT_DISARM
        with pytest.raises(LookupError):
            engine.find_move(state, "Re3xe7")
        assert engine.find_move(state, "Re3-e6")  # disarmed, it still moves

    def test_a_capture_barred_by_a_dryad(self) -> None:
        engine = _engine()
        board = {**KINGS, "b5": ("Y", 2), "c6": ("N", 1), "c1": ("R", 0), "h8": ("D", 2)}
        state = engine.make_state(board)
        assert engine.effects(state).bits[_sq(engine, "c6")] & SIGHT_SHIELD
        assert engine.effects(state).shielded > 0
        with pytest.raises(LookupError):
            engine.find_move(state, "Rc1xc6")

    def test_the_boxed_duck_passes(self) -> None:
        engine = _engine()
        board = {**KINGS, "a9": ("D", 2), "a8": ("P", 1), "b8": ("P", 1), "b9": ("N", 1)}
        state = engine.make_state(board, duck_owed=True)
        [move] = engine.legal_moves(state)
        assert decode_move(move)[2:] == (MOVE_DUCK, DUCK_PASS)
        after = engine.apply(state, move)
        assert engine.duck_square(after) == _sq(engine, "a9") and after.side_to_move == 1
