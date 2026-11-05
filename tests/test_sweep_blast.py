"""SR6c (`docs/plan-schema-next.md` §2.6 row W3): a sweep beside a blast, and the sweep switch
`moves[].effects`.

The ruling (2026-10-27): **the sweep switch**. Under both values the sweep removes what it
removes and lands where it lands; the switch says which removals the blast acts on. Under
`"landing"` (the default) one blast is centred on the landing square, after the sweep, when the
move removed anything (swept or taken on landing) whose capture by the sweeper sets the blast
off (`explode_on_capture.by` and `.of`). Under `"path"` one blast is centred on each swept square
whose piece sets it off, and one on the landing if its capture does; all clear together after the
sweep, and the sweeper is destroyed once. A sweep beside both a blast and a strike or contacts
stays refused (W3's narrower row).

Readings the row left open, settled on the plan's principles and written in the record: a swept
piece sets a blast off whoever owns it (the sweep removes it whoever owns it); each blast's
shields are read before anything clears; beside a capture hand the swept pieces reach the hand
and the blasts' victims do not (H6); `"path"` is refused as inert where the sweeper is outside
`by`; Q1063, a capture on a ride square is the landing's. Hand-built 6x6 specs, not chess.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import MOVE_SWEEP, decode_move, move_from, move_to
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
IDS = ("K", "S", "G", "R")


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _name(square: int) -> str:
    return f"{'abcdef'[square % WIDTH]}{square // WIDTH + 1}"


def _spec(effects: str | None = "path", sweep_range: int | None = None,
          **blast: Any) -> dict[str, Any]:
    """Kings a1/a6, a sweeper (S, a rook's sweep) b1, guards c1/d1 and an R e1, all mirrored;
    an atomic blast of radius 1 sparing the kings (and `R` under `immune` below)."""
    sweep: dict[str, Any] = {"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}
    if effects is not None:
        sweep["effects"] = effects
    if sweep_range is not None:
        sweep["sweep_range"] = sweep_range
    return copy.deepcopy({
        "name": "the sweep that explodes (SR6c test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "S", "value": 5, "moves": [sweep]},
            {"id": "G", "value": 2, "moves": [KING]},
            {"id": "R", "value": 3, "moves": [KING]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "S", "squares": ["b1"]},
            {"piece": "G", "squares": ["c1", "d1"]},
            {"piece": "R", "squares": ["e1"]},
        ]},
        "rules": {
            "win": "royal_capture", "stalemate": "draw", "max_plies": 60,
            "explode_on_capture": {"radius": 1, "immune": ["K"], **blast},
        },
    })


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


def _move(engine: Engine, state: GameState, origin: str, target: str) -> int:
    for move in engine.legal_moves(state):
        if (move_from(move), move_to(move)) == (_sq(origin), _sq(target)):
            return move
    raise AssertionError(f"no move {origin}-{target}")


def _board(engine: Engine, state: GameState) -> dict[str, tuple[str, int]]:
    spec = engine.spec
    return {
        _name(sq): (IDS[spec.type_of(v)], spec.owner_of(v)) for sq, v in enumerate(state.board) if v
    }


KINGS = {"a1": ("K", 0), "f6": ("K", 1)}
# The sweeper on a3 sweeps b3 and c3 (enemy guards) to f3: an enemy guard beside b3 (b4), the
# mover's own guard beside c3 (d2), an enemy guard beside the landing (f4), and one far off (d6).
ROW = {
    **KINGS, "a3": ("S", 0), "b3": ("G", 1), "c3": ("G", 1),
    "b4": ("G", 1), "d2": ("G", 0), "f4": ("G", 1), "d6": ("G", 1),
}


def _play(spec: dict[str, Any], placement: dict[str, tuple[str, int]] = ROW,
          origin: str = "a3", target: str = "f3") -> tuple[Engine, GameState, GameState, int]:
    engine = _engine(spec)
    state = engine.make_state(placement, side_to_move=0, moved=list(placement))
    move = _move(engine, state, origin, target)
    assert decode_move(move)[2] == MOVE_SWEEP
    after = engine.apply(state, move)
    assert after.key == engine._compute_hash(after)
    return engine, state, after, move


def _without(placement: dict[str, tuple[str, int]], *squares: str) -> dict[str, tuple[str, int]]:
    return {sq: piece for sq, piece in placement.items() if sq not in squares}


class TestLanding:
    def test_one_blast_on_the_landing(self) -> None:
        """`"landing"`: the swept guards go, then one blast on f3 takes the sweeper and the guard
        on f4; the pieces beside the swept squares stand."""
        engine, _, after, _ = _play(_spec(effects=None))
        assert _board(engine, after) == _without(ROW, "a3", "b3", "c3", "f4")

    def test_a_capture_on_the_landing_blasts_there(self) -> None:
        placement = {**ROW, "f3": ("G", 1)}
        engine, _, after, _ = _play(_spec(effects=None), placement)
        assert _board(engine, after) == _without(placement, "a3", "b3", "c3", "f3", "f4")

    def test_of_names_what_sets_it_off(self) -> None:
        """`of: ["R"]`: sweeping guards sets off nothing; sweeping an R does, on the landing."""
        engine, _, after, _ = _play(_spec(effects=None, of=["R"]))
        assert _board(engine, after) == {**_without(ROW, "a3", "b3", "c3"), "f3": ("S", 0)}
        placement = {**ROW, "c3": ("R", 1)}
        engine, _, after, _ = _play(_spec(effects=None, of=["R"]), placement)
        assert _board(engine, after) == _without(placement, "a3", "b3", "c3", "f4")

    def test_the_mover_s_own_swept_piece_sets_it_off(self) -> None:
        placement = {**KINGS, "a3": ("S", 0), "c3": ("G", 0), "f4": ("G", 1)}
        engine, _, after, _ = _play(_spec(effects=None), placement)
        assert _board(engine, after) == KINGS

    def test_landing_written_is_the_game_without_it(self) -> None:
        assert spec_hash(_spec(effects="landing")) == spec_hash(_spec(effects=None))
        explosion = compile_spec(_spec(effects="landing")).explosion
        assert explosion is not None and explosion.sweep_path == ()


class TestPath:
    def test_a_blast_on_each_swept_square(self) -> None:
        """`"path"`: blasts on b3 and c3 take b4 and d2 (own pieces too); the sweeper goes up
        once on f3, which captured nothing, so f4 stands."""
        engine, _, after, _ = _play(_spec())
        assert _board(engine, after) == _without(ROW, "a3", "b3", "c3", "b4", "d2")

    def test_the_landing_blasts_when_its_capture_does(self) -> None:
        placement = {**ROW, "f3": ("G", 1)}
        engine, _, after, _ = _play(_spec(), placement)
        assert _board(engine, after) == _without(placement, "a3", "b3", "c3", "b4", "d2", "f3", "f4")

    def test_of_picks_the_squares(self) -> None:
        """`of: ["R"]`: only the swept R's square blasts (c3, which reaches b4 and d2); the
        swept guard's b3 does not, so a2's guard stands."""
        placement = {**ROW, "c3": ("R", 1), "a2": ("G", 1)}
        engine, _, after, _ = _play(_spec(of=["R"]), placement)
        assert _board(engine, after) == _without(placement, "a3", "b3", "c3", "b4", "d2")

    def test_immune_survives_every_blast(self) -> None:
        placement = {**ROW, "b2": ("R", 1), "c4": ("R", 0)}
        engine, _, after, _ = _play(_spec(immune=["K", "R"]), placement)
        assert _board(engine, after) == _without(placement, "a3", "b3", "c3", "b4", "d2")

    def test_a_short_sweep_blasts_only_what_it_clears(self) -> None:
        """Q1063: a sweep clearing two squares rides on to take e3 as its landing: blasts on b3
        and c3, and one on e3, whose capture explodes."""
        placement = {**KINGS, "a3": ("S", 0), "b3": ("G", 1), "c3": ("G", 1), "e3": ("G", 1),
                     "f2": ("G", 1), "a5": ("G", 1)}
        engine, _, after, _ = _play(_spec(sweep_range=2), placement, target="e3")
        assert _board(engine, after) == {**KINGS, "a5": ("G", 1)}

    def test_by_must_name_the_sweeper(self) -> None:
        engine, _, after, _ = _play(_spec(by=["S"]))
        assert _board(engine, after) == _without(ROW, "a3", "b3", "c3", "b4", "d2")
        assert any("is set off by no capture of it" in e for e in validate_spec(_spec(by=["G"])))
        # Outside `by` under "landing", nothing explodes.
        engine, _, after, _ = _play(_spec(effects=None, by=["G"]))
        assert _board(engine, after) == {**_without(ROW, "a3", "b3", "c3"), "f3": ("S", 0)}

    def test_the_compiled_rule_carries_the_switch(self) -> None:
        explosion = compile_spec(_spec()).explosion
        assert explosion is not None
        assert explosion.sweep_path == tuple(t == IDS.index("S") for t in range(len(IDS)))


class TestTheFilter:
    @pytest.mark.parametrize(("effects", "offered"), [("path", False), (None, True)])
    def test_a_blast_that_takes_the_own_king_is_not_offered(
        self, effects: str | None, offered: bool,
    ) -> None:
        """Kings not immune: a blast on b3 reaches the mover's king on a2, so under `"path"` the
        sweep to f3 loses it and is not offered; under `"landing"` the blast is on f3."""
        spec = _spec(effects=effects)
        spec["rules"]["explode_on_capture"]["immune"] = []
        spec["rules"]["win"] = "checkmate"
        engine = _engine(spec)
        placement = {"a2": ("K", 0), "f6": ("K", 1), "a3": ("S", 0), "b3": ("G", 1)}
        state = engine.make_state(placement, side_to_move=0, moved=list(placement))
        pairs = {(move_from(m), move_to(m)) for m in engine.legal_moves(state)}
        assert ((_sq("a3"), _sq("f3")) in pairs) is offered


def _count(engine: Engine, position: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(_count(engine, engine.apply(position, m), depth - 1)
               for m in engine.legal_moves(position))


class TestUndo:
    @pytest.mark.parametrize("effects", ["path", None])
    @pytest.mark.parametrize("win", ["royal_capture", "checkmate"])
    def test_perft_counts_what_apply_counts(self, effects: str | None, win: str) -> None:
        spec = _spec(effects=effects)
        spec["rules"]["win"] = win
        engine = _engine(spec)
        state = engine.make_state(ROW, side_to_move=0, moved=list(ROW))
        assert engine.perft(state, 3) == _count(engine, state, 3)

    def test_walks_keep_the_key(self) -> None:
        """Seeded walks from the opening, sweeps preferred: the key is the full scan's every
        ply, and `apply` leaves its argument as it was."""
        for effects in ("path", None):
            engine = _engine(_spec(effects=effects))
            for seed in range(3):
                rng = random.Random(seed)
                state = engine.initial_state()
                for _ in range(40):
                    moves = engine.legal_moves(state)
                    if not moves or engine.terminal(state, moves) is not None:
                        break
                    loud = [m for m in moves if decode_move(m)[2] == MOVE_SWEEP]
                    move = rng.choice(loud if loud and rng.random() < 0.8 else moves)
                    board = bytes(state.board)
                    after = engine.apply(state, move)
                    assert bytes(state.board) == board
                    assert after.key == engine._compute_hash(after)
                    state = after


class TestBesideAHand:
    def _spec(self, effects: str | None) -> dict[str, Any]:
        spec = _spec(effects=effects)
        spec["rules"]["drops"] = {"from_captures": True}
        return spec

    @pytest.mark.parametrize("effects", ["path", None])
    def test_the_swept_reach_the_hand_and_the_blasted_do_not(self, effects: str | None) -> None:
        """H6: a blast's victims reach no hand; the swept guards (two) and the landing's (one)
        do, whatever the switch blasts."""
        placement = {**ROW, "f3": ("G", 1)}
        engine = _engine(self._spec(effects))
        state = engine.make_state(placement, side_to_move=0, moved=list(placement))
        state = dataclasses.replace(state, hands=engine.initial_state().hands)
        after = engine.apply(state, _move(engine, state, "a3", "f3"))
        guard = IDS.index("G")
        assert after.hands[0][guard] == 3
        assert sum(after.hands[0]) == 3 and sum(after.hands[1]) == 0
        assert after.key == engine._compute_hash(after)
        assert engine.perft(state, 2) == _count(engine, state, 2)


class TestTheRefusals:
    def test_a_sweep_beside_a_blast_validates(self) -> None:
        for effects in (None, "path"):
            assert validate_spec(_spec(effects=effects)) == []

    def test_a_blast_beside_a_strike_is_refused(self) -> None:
        spec = _spec(effects=None)
        spec["rules"]["strike"] = {
            "effect": "demote", "position": "stay", "immune": ["K", "S"],
            "chain": [{"from": "G", "to": "R"}],
        }
        assert "W3" in refused_ids(validate_spec(spec))

    def test_a_blast_beside_contacts_is_refused(self) -> None:
        spec = _spec(effects=None)
        spec["rules"]["contacts"] = {
            "position": "stay", "links": [{"mover": "G", "onto": "R", "onto_to": "G"}],
        }
        assert "W3" in refused_ids(validate_spec(spec))

    def test_path_without_a_blast_is_refused(self) -> None:
        spec = _spec()
        del spec["rules"]["explode_on_capture"]
        assert any("leave effects out" in e for e in validate_spec(spec))


def test_the_switch_names_another_game() -> None:
    assert spec_hash(_spec()) != spec_hash(_spec(effects=None))
