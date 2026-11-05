"""SR2 (`docs/plan-schema-next.md` §2.7, rows W9 and W10; §2.10, row K2): what a sweep may not
pass, and the duck beside portals.

* **W9, lifted.** A sweep *"jumps to any square along the line, removing every piece passed
  over; a piece it may not capture blocks"*, and nothing captures the duck: it blocks. The probe
  found the core already stops a sweep at the duck, both seats, so the lift is the validator's.
  A duck type that sweeps stays refused (the duck captures nothing).
* **W10, handed to SR6g, lifted there.** As W9, a shielded piece may not be captured, so it
  blocks a sweep. SR2's probe found the core swept it (the sweep read the static capture
  matrix, a shield is read from the board), landed on it, and read a sweep over a shielded
  piece as check; SR6g reads the shield off the board in the sweep's walk and its attack test
  (`Game::sweep_shielded`), so the strict xfails pass on a validated spec. Beside each earlier
  batch D rule (a path strike or contacts, a blast, a declared ray on a cylinder, a fold, a
  commanded sweep) the core's list is `tests/sight_oracle.py`'s, whose twin sweeps that would
  remove a shielded piece are no moves.
* **K2, handed to SR3d, lifted there.** The ruled reading: the duck is an uncapturable piece to
  every transit and fold go-through. The core refused the pair outright (`the duck and portals
  share the third owner's block`); SR3d removed that refusal and the validator's, and the
  probe below, a strict xfail until then, now passes on a validated spec. The walks are
  `tests/test_duck_beside_portals.py`.

The oracle for W9 is this file's `_sweep_walk`: no sweep's line crosses or ends on the duck's
square (a declared test oracle, rule 3). Hand-built specs, not chess.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_SWEEP, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.schema import validate_spec
from tests.sight_oracle import SightOracle, normalise

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


# --------------------------------------------------------------------------- W10, lifted by SR6g


WARDEN = {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 1,
          "effect": "shield", "whose": "own"}


def _shielded(win: str = "checkmate", **rules: Any) -> dict[str, Any]:
    """The broom cupboard with a warden `W` (a king stepper) shielding its own side's pieces
    next to it, in place of the duck's type."""
    spec = _spec(win=win, **rules)
    spec["pieces"][3] = {"id": "W", "value": 2, "moves": [KING], "sight": [dict(WARDEN)]}
    spec["setup"]["placements"].append({"piece": "W", "squares": ["b1"]})
    return spec


def _swept(engine: Engine, state: GameState, origin: str) -> dict[int, tuple[int, ...]]:
    """The sweeps from ``origin``, by target: the squares each removes."""
    return {
        (m >> 8) & 0xFF: tuple(sq for sq, _ in engine.spec.sweep_victims(state.board, m))
        for m in engine.legal_moves(state)
        if (m & 0xFF) == _sq(origin) and move_type(m) == MOVE_SWEEP
    }


@pytest.mark.parametrize("win", ["checkmate", "royal_capture"])
def test_a_shield_beside_a_sweep_is_valid(win: str) -> None:
    assert validate_spec(_shielded(win)) == []


@pytest.mark.parametrize("side", [0, 1])
def test_a_shielded_piece_blocks_a_sweep(side: int) -> None:
    """An enemy guard on c4 shielded by its warden on d5: the sweeper on c1 may not take it or
    pass it; with the warden out of reach it sweeps the guard and runs on."""
    engine = _engine(_shielded())
    state = _position(engine, {"a1": ("K", side), "f6": ("K", 1 - side), "c1": ("S", side),
                               "c4": ("G", 1 - side), "d5": ("W", 1 - side)}, side)
    reach = _targets(engine, state, "c1")
    assert {_sq("c2"), _sq("c3")} <= reach
    assert not reach & {_sq("c4"), _sq("c5"), _sq("c6")}
    free = _position(engine, {"a1": ("K", side), "f6": ("K", 1 - side), "c1": ("S", side),
                              "c4": ("G", 1 - side), "e6": ("W", 1 - side)}, side)
    assert {_sq("c4"), _sq("c5"), _sq("c6")} <= _targets(engine, free, "c1")
    assert _swept(engine, free, "c1")[_sq("c6")] == (_sq("c4"),)


@pytest.mark.parametrize("side", [0, 1])
def test_a_shielded_piece_screens_its_royal_from_a_sweep(side: int) -> None:
    engine = _engine(_shielded())
    state = _position(engine, {"c1": ("K", side), "f6": ("K", 1 - side), "c6": ("S", 1 - side),
                               "c3": ("G", side), "d3": ("W", side), "a6": ("G", 1 - side)}, side)
    assert not engine.in_check(state, side)
    bare = _position(engine, {"c1": ("K", side), "f6": ("K", 1 - side), "c6": ("S", 1 - side),
                              "c3": ("G", side), "f3": ("W", side), "a6": ("G", 1 - side)}, side)
    assert engine.in_check(bare, side)


class TestAShieldBlocksASweep:
    """W10 (`docs/plan-schema-next.md` §2.7, ruled 2026-10-27): as W9, a shielded piece may not
    be captured, so it blocks a sweep. SR6g reads the shield off the board in the sweep's walk
    and in its attack test (`Game::sweep_shielded`)."""

    @pytest.mark.parametrize("side", [0, 1])
    def test_its_own_sides_shielded_piece_blocks_it(self, side: int) -> None:
        """A sweep clears its own side's pieces too, but not a shielded one: the guard on c3
        beside its warden on b3 ends the ray."""
        engine = _engine(_shielded())
        state = _position(engine, {"a1": ("K", side), "f6": ("K", 1 - side), "c1": ("S", side),
                                   "c3": ("G", side), "b3": ("W", side)}, side)
        reach = _targets(engine, state, "c1")
        assert _sq("c2") in reach
        assert not reach & {_sq("c3"), _sq("c4"), _sq("c5"), _sq("c6")}
        bare = _position(engine, {"a1": ("K", side), "f6": ("K", 1 - side), "c1": ("S", side),
                                  "c3": ("G", side), "a5": ("W", side)}, side)
        assert _swept(engine, bare, "c1")[_sq("c6")] == (_sq("c3"),)

    @pytest.mark.parametrize("side", [0, 1])
    def test_the_sweep_takes_what_stands_before_the_shield(self, side: int) -> None:
        """Guards on c2 (bare) and c4 (shielded from d5): the sweep removes c2 and lands on c3,
        and goes no further."""
        engine = _engine(_shielded())
        state = _position(engine, {"a1": ("K", side), "f6": ("K", 1 - side), "c1": ("S", side),
                                   "c2": ("G", 1 - side), "c4": ("G", 1 - side),
                                   "d5": ("W", 1 - side)}, side)
        assert _swept(engine, state, "c1").get(_sq("c3")) == (_sq("c2"),)
        assert not _targets(engine, state, "c1") & {_sq("c4"), _sq("c5"), _sq("c6")}

    @pytest.mark.parametrize("side", [0, 1])
    def test_the_warden_may_not_walk_off_the_screen(self, side: int) -> None:
        """The check filter reads the shield on the board the move leaves: the warden on d3
        steps only where it still shields the guard on c3, or onto the file itself."""
        engine = _engine(_shielded())
        state = _position(engine, {"c1": ("K", side), "f6": ("K", 1 - side), "c6": ("S", 1 - side),
                                   "c3": ("G", side), "d3": ("W", side), "a6": ("G", 1 - side)}, side)
        warden = _targets(engine, state, "d3")
        assert warden
        assert warden <= {_sq(s) for s in ("c2", "c4", "d2", "d4")}
        assert not warden & {_sq(s) for s in ("e2", "e3", "e4")}

    def test_the_text_round_trips(self) -> None:
        engine = _engine(_shielded())
        state = _position(engine, {"a1": ("K", 0), "f6": ("K", 1), "c1": ("S", 0),
                                   "c3": ("G", 1), "c5": ("G", 1), "d6": ("W", 1)})
        move = next(m for m in engine.legal_moves(state)
                    if move_type(m) == MOVE_SWEEP and (m >> 8) & 0xFF == _sq("c4"))
        assert engine.find_move(state, engine.describe_move(state, move)) == move
        assert not {_sq("c5"), _sq("c6")} & _targets(engine, state, "c1")


# ---------------------------------------------- W10 beside the earlier batch D rules, by oracle


COMMAND = {"vectors": [[1, 0]], "symmetry": "rotate4", "effect": "command", "whose": "enemy",
           "captures_own": True}


def _walking(**rules: Any) -> dict[str, Any]:
    """A busier shielded game for the walks: wardens beside the guards, a sweeper each side."""
    spec = _shielded(**rules)
    spec["setup"]["placements"] = [
        {"piece": "K", "squares": ["a1"]}, {"piece": "S", "squares": ["f1"]},
        {"piece": "W", "squares": ["c1"]}, {"piece": "G", "squares": ["b2", "c2", "e2"]},
    ]
    return spec


def _pairings() -> dict[str, dict[str, Any]]:
    """Each earlier batch D rule beside a shield: SR6a's path strike, SR6b's path contacts,
    SR6c's blast, SR6d's declared ray on a cylinder, SR6e's fold, SR6f's commanded sweep."""
    games = {
        "plain": _walking(),
        "cylinder": _walking(wrap={"files": "both"}),
        "fold": _walking(portals=[{"orange": "c4", "blue": "e3", "kind": "fold",
                                   "moves": [{"kind": "leap", "vectors": [[1, 0]],
                                              "symmetry": "rotate4"}]}]),
        "blast": _walking(explode_on_capture={"radius": 1}),
    }
    strike = _walking(strike={"effect": "demote", "position": "stay", "immune": ["K", "S", "W"],
                              "chain": [{"from": "G", "to": "W"}]})
    strike["pieces"][1]["moves"][0]["effects"] = "path"
    games["path strike"] = strike
    contacts = _walking(contacts={"position": "stay", "immune": ["K"], "links": [
        {"mover": "S", "onto": "G", "onto_to": "W"}]})
    contacts["pieces"][1]["moves"][0]["effects"] = "path"
    games["path contacts"] = contacts
    command = _walking()
    command["pieces"][3]["sight"].append(dict(COMMAND))
    games["command"] = command
    return games


class _BlastOracle(SightOracle):
    """The sight oracle for a game with a blast beside the shield. Its twin has no sight, so
    the twin's blast clears shielded pieces, which SR5b's ruling spares (a shield shields from
    a blast); the board a move makes is the engine's here, that reading being SR5b's and pinned
    by `tests/test_shield_from_blast.py` and the blast test below. What this oracle still
    decides is what W10 changes: which sweeps a shield stops, and what they attack."""

    def after(self, state: GameState, move: int) -> GameState:
        return self.engine.apply(state, move)


def _oracle(game: str) -> SightOracle:
    spec = _pairings()[game]
    return _BlastOracle(spec) if "explode_on_capture" in spec["rules"] else SightOracle(spec)


def _oracle_walk(oracle: SightOracle, seed: int, plies: int) -> int:
    """A seeded walk leaning toward sweeps, the core's legal list the oracle's at every ply and
    the key the full scan's. Returns the plies on which a shield stopped a twin's sweep."""
    engine = oracle.engine
    rng = random.Random(seed)
    state = engine.initial_state()
    stopped = 0
    for _ in range(plies):
        moves = engine.legal_moves(state)
        assert normalise(moves) == oracle.legal_moves(state), (seed, state.ply)
        assert state.key == engine._compute_hash(state)
        if engine.terminal(state, moves) is not None:
            break
        effects = oracle.effects(state.board, oracle.fold_exits(state))
        stopped += any(
            move_type(m) == MOVE_SWEEP
            and oracle._sweeps_a_shield(state.board, m, effects, None)
            for m in oracle.twin.pseudo_legal_moves(state)
        )
        sweeps = [m for m in moves if move_type(m) == MOVE_SWEEP]
        move = rng.choice(sweeps) if sweeps and rng.random() < 0.5 else rng.choice(moves)
        state = engine.apply(state, move)
    return stopped


@pytest.mark.parametrize("game", sorted(_pairings()))
def test_each_pairing_is_valid(game: str) -> None:
    assert validate_spec(_pairings()[game]) == []


@pytest.mark.parametrize("game", sorted(_pairings()))
def test_walks_agree_with_the_oracle(game: str) -> None:
    oracle = _oracle(game)
    assert sum(_oracle_walk(oracle, seed, 40) for seed in range(2)) > 0


@pytest.mark.slow
@pytest.mark.parametrize("game", sorted(_pairings()))
def test_longer_walks_agree_with_the_oracle(game: str) -> None:
    oracle = _oracle(game)
    assert sum(_oracle_walk(oracle, seed, 120) for seed in range(2, 10)) > 0


@pytest.mark.parametrize("side", [0, 1])
def test_a_sweeps_blast_spares_a_shielded_piece(side: int) -> None:
    """SR5b beside SR6g: the sweeper on c1 sweeps the guard on c3 and lands on c4 (a blast of a
    king's step there); of the two enemy guards beside it, the one its warden shields stays."""
    engine = _engine(_shielded(explode_on_capture={"radius": 1}))
    state = _position(engine, {"a1": ("K", side), "f6": ("K", 1 - side), "c1": ("S", side),
                               "c3": ("G", 1 - side), "b5": ("G", 1 - side),
                               "d5": ("G", 1 - side), "e6": ("W", 1 - side)}, side)
    move = next(m for m in engine.legal_moves(state)
                if move_type(m) == MOVE_SWEEP and (m >> 8) & 0xFF == _sq("c4"))
    after = engine.apply(state, move)
    assert not after.board[_sq("c3")] and not after.board[_sq("b5")]
    assert after.board[_sq("d5")]


@pytest.mark.parametrize("game", ["plain", "cylinder", "fold"])
def test_perft_against_an_apply_built_count(game: str) -> None:
    engine = Engine(_pairings()[game])
    state = engine.initial_state()

    def count(position: GameState, depth: int) -> int:
        if depth == 0:
            return 1
        return sum(count(engine.apply(position, m), depth - 1)
                   for m in engine.legal_moves(position))

    assert engine.perft(state, 3) == count(state, 3)


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
def test_the_duck_beside_portals_is_valid(kind: str | None) -> None:
    """K2, lifted by SR3d."""
    assert validate_spec(_duck_and_portals(kind)) == []


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("kind, exit_square", [(None, "e5"), ("fold", "e4")])
def test_the_duck_blocks_a_transit(side: int, kind: str | None, exit_square: str) -> None:
    """A rook on b1 runs up the b-file into the end on b3: with the duck on the exit (one step
    beyond the blue end for a momentum portal, the blue end itself for a fold), no transit."""
    engine = _engine(_duck_and_portals(kind))
    state = _position(engine, {"a1": ("K", side), "f6": ("K", 1 - side), "b1": ("R", side),
                               exit_square: ("D", 2)}, side)
    assert _sq(exit_square) not in _targets(engine, state, "b1")
