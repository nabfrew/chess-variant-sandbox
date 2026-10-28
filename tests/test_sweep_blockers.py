"""SR2 (`docs/plan-schema-next.md` §2.7, rows W9 and W10; §2.10, row K2): what a sweep may not
pass, and the duck beside portals.

* **W9, lifted.** A sweep *"jumps to any square along the line, removing every piece passed
  over; a piece it may not capture blocks"*, and nothing captures the duck: it blocks. The probe
  found the core already stops a sweep at the duck, both seats, so the lift is the validator's.
  A duck type that sweeps stays refused (the duck captures nothing).
* **W10, handed to SR6g.** As W9, a shielded piece may not be captured, so it should block a
  sweep. The probe found the core sweeps it (the sweep reads the static capture matrix, a
  shield is read from the board), and lands on it, and reads a sweep over a shielded piece as
  check: the refusal stays, and the reading is a strict xfail naming SR6g.
* **K2, handed to SR3d.** The ruled reading: the duck is an uncapturable piece to every transit
  and fold go-through. The core refuses the pair outright (`the duck and portals share the
  third owner's block`): the refusal stays, and the reading is a strict xfail naming SR3d.

The oracle for W9 is this file's `_sweep_walk`: no sweep's line crosses or ends on the duck's
square (a declared test oracle, rule 3). Hand-built specs, not chess.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine import compile as compile_module
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import MOVE_SWEEP, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
SWEEP = {"kind": "sweep", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _spec(win: str = "royal_capture", **rules: Any) -> dict[str, Any]:
    """6x6: kings, a sweeper `S` (a queen's lines), guards; `D` the duck where one is wanted."""
    return copy.deepcopy({
        "schema_version": 31, "name": "the broom cupboard", "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "S", "value": 5, "moves": [SWEEP]},
            {"id": "G", "value": 2, "moves": [KING]},
            {"id": "D", "value": 0, "moves": []},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]}, {"piece": "S", "squares": ["f1"]},
            {"piece": "G", "squares": ["b2", "c2", "d2", "e2"]},
        ]},
        "rules": {"win": win, "stalemate": "draw", "max_plies": 80, **rules},
    })


def _duck(start: str = "c4", goes: str = "anywhere") -> dict[str, Any]:
    spec = _spec(duck={"piece": "D", "start": start, "goes": goes})
    if goes == "by_moves":
        spec["pieces"][3]["moves"] = [{"kind": "slide", "vectors": [[1, 0], [1, 1]],
                                       "symmetry": "rotate4"}]
    return spec


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _position(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0) -> GameState:
    return engine.make_state(placement, side_to_move=side, moved=list(placement))


def _targets(engine: Engine, state: GameState, origin: str) -> set[int]:
    return {(m >> 8) & 0xFF for m in engine.legal_moves(state) if (m & 0xFF) == _sq(origin)}


def _line(origin: int, target: int) -> list[int]:
    """The squares a straight or diagonal line passes from ``origin`` to ``target``, both ends
    included."""
    ox, oy, tx, ty = origin % WIDTH, origin // WIDTH, target % WIDTH, target // WIDTH
    steps = max(abs(tx - ox), abs(ty - oy))
    dx, dy = (tx - ox) // steps, (ty - oy) // steps
    return [(oy + k * dy) * WIDTH + ox + k * dx for k in range(steps + 1)]


def _sweep_walk(spec: dict[str, Any], seeds: range, plies: int) -> int:
    """Seeded walks leaning toward sweeps: no sweep offered crosses or lands on the duck; the
    key is the full scan's. Returns the sweeps offered while the duck stood on a line."""
    engine = _engine(spec)
    nearby = 0
    for seed in seeds:
        rng = random.Random(seed)
        state = engine.initial_state()
        for _ in range(plies):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            duck = engine.duck_square(state)
            sweeps = [m for m in moves if move_type(m) == MOVE_SWEEP]
            for m in sweeps:
                assert duck not in _line(m & 0xFF, (m >> 8) & 0xFF), (seed, state.ply)
            if duck is not None:
                nearby += sum(1 for m in moves if move_type(m) == MOVE_SWEEP)
            move = rng.choice(sweeps) if sweeps and rng.random() < 0.5 else rng.choice(moves)
            state = engine.apply(state, move)
            assert state.key == engine._compute_hash(state)
    return nearby


class TestTheDuckBlocksASweep:
    @pytest.mark.parametrize("goes", ["anywhere", "by_moves"])
    def test_it_is_valid(self, goes: str) -> None:
        assert validate_spec(_duck(goes=goes)) == []

    def test_a_duck_that_sweeps_is_still_refused(self) -> None:
        spec = _duck(goes="by_moves")
        spec["pieces"][3]["moves"] = [{"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}]
        assert any("the duck's type 'D' sweeps" in e for e in validate_spec(spec))

    @pytest.mark.parametrize("side", [0, 1])
    def test_the_sweep_stops_short_of_the_duck(self, side: int) -> None:
        """The sweeper on c1, an enemy guard on c2, the duck on c4: up the file it sweeps the
        guard to c3 and no further; on the empty diagonal it runs to the edge."""
        engine = _engine(_duck())
        state = _position(engine, {"a1": ("K", side), "f6": ("K", 1 - side), "c1": ("S", side),
                                   "c2": ("G", 1 - side), "c4": ("D", 2)}, side)
        assert not engine.duck_owed(state) and engine.duck_square(state) == _sq("c4")
        reach = _targets(engine, state, "c1")
        assert {_sq("c2"), _sq("c3")} <= reach
        assert not reach & {_sq("c4"), _sq("c5"), _sq("c6")}
        assert _sq("f4") in reach

    @pytest.mark.parametrize("goes", ["anywhere", "by_moves"])
    def test_a_walk_never_sweeps_past_the_duck(self, goes: str) -> None:
        assert _sweep_walk(_duck(goes=goes), range(6), 80) > 0

    @pytest.mark.slow
    def test_twenty_walks(self) -> None:
        assert _sweep_walk(_duck(), range(20), 160) > 100

    def test_unmake_puts_the_sweep_and_the_duck_back(self) -> None:
        engine = _engine(_duck())
        state = _position(engine, {"a1": ("K", 0), "f6": ("K", 1), "c1": ("S", 0),
                                   "c2": ("G", 1), "c4": ("D", 2), "e3": ("S", 1)})

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(count(engine.apply(position, m), depth - 1)
                       for m in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)


# --------------------------------------------------------------------------- handed on


def _past_the_validator(monkeypatch: pytest.MonkeyPatch, spec: dict[str, Any]) -> Engine:
    monkeypatch.setattr(compile_module, "validate_or_raise", lambda spec: None)
    return Engine(compile_spec(spec))


def _shielded(win: str) -> dict[str, Any]:
    spec = _spec(win=win)
    spec["pieces"][3] = {"id": "W", "value": 2, "moves": [KING], "sight": [
        {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 1,
         "effect": "shield", "whose": "own"}]}
    spec["setup"]["placements"].append({"piece": "W", "squares": ["b1"]})
    return spec


def test_a_shield_beside_a_sweep_is_still_refused() -> None:
    assert "W10" in refused_ids(validate_spec(_shielded("checkmate")))


@pytest.mark.xfail(strict=True, reason="W10 is handed to SR6g: the sweep reads the static "
                   "capture matrix, a shield is read from the board, so a sweep passes a "
                   "shielded piece and takes it")
@pytest.mark.parametrize("side", [0, 1])
def test_a_shielded_piece_blocks_a_sweep(monkeypatch: pytest.MonkeyPatch, side: int) -> None:
    """An enemy guard on c4 shielded by its warden on d5: the sweeper on c1 may not take it or
    pass it."""
    engine = _past_the_validator(monkeypatch, _shielded("checkmate"))
    state = _position(engine, {"a1": ("K", side), "f6": ("K", 1 - side), "c1": ("S", side),
                               "c4": ("G", 1 - side), "d5": ("W", 1 - side)}, side)
    assert not _targets(engine, state, "c1") & {_sq("c4"), _sq("c5"), _sq("c6")}


@pytest.mark.xfail(strict=True, reason="W10 is handed to SR6g: a sweep over a shielded piece "
                   "reads as check on the royal behind it")
@pytest.mark.parametrize("side", [0, 1])
def test_a_shielded_piece_screens_its_royal_from_a_sweep(
    monkeypatch: pytest.MonkeyPatch, side: int,
) -> None:
    engine = _past_the_validator(monkeypatch, _shielded("checkmate"))
    state = _position(engine, {"c1": ("K", side), "f6": ("K", 1 - side), "c6": ("S", 1 - side),
                               "c3": ("G", side), "d3": ("W", side), "a6": ("G", 1 - side)}, side)
    assert not engine.in_check(state, side)


def _duck_and_portals(kind: str | None) -> dict[str, Any]:
    spec = _duck(start="a4")
    spec["pieces"][1] = {"id": "R", "value": 5, "moves": [ROOK]}
    spec["setup"]["placements"][1]["piece"] = "R"
    group: dict[str, Any] = {"orange": "b3", "blue": "e4"}
    if kind:
        group["kind"] = kind
    spec["rules"]["portals"] = [group]
    return spec


@pytest.mark.parametrize("kind", [None, "fold"])
def test_the_duck_beside_portals_is_still_refused(kind: str | None) -> None:
    assert "K2" in refused_ids(validate_spec(_duck_and_portals(kind)))


@pytest.mark.xfail(strict=True, reason="K2 is handed to SR3d: the core refuses the duck and "
                   "portals together (they share the third owner's block)")
@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("kind, exit_square", [(None, "e5"), ("fold", "e4")])
def test_the_duck_blocks_a_transit(monkeypatch: pytest.MonkeyPatch, side: int,
                                   kind: str | None, exit_square: str) -> None:
    """A rook on b1 runs up the b-file into the end on b3: with the duck on the exit (one step
    beyond the blue end for a momentum portal, the blue end itself for a fold), no transit."""
    engine = _past_the_validator(monkeypatch, _duck_and_portals(kind))
    state = _position(engine, {"a1": ("K", side), "f6": ("K", 1 - side), "b1": ("R", side),
                               exit_square: ("D", 2)}, side)
    assert _sq(exit_square) not in _targets(engine, state, "b1")
