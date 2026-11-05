"""SR6a (`docs/plan-schema-next.md` §2.6 row W1): a sweep beside a strike, and the sweep switch
`moves[].effects`.

The ruling (2026-10-27): **the sweep switch**, `effects: "landing"` (the default and the off value)
or `"path"`. Under both the sweep removes what it removes today and lands where it lands today;
the switch says only which removals the strike acts on. Under `"landing"` the strike acts on the
landing's capture alone, as on any capture; the swept pieces are removed. Under `"path"` every
swept piece is struck as the landing's target is: each not in `strike.immune` follows its chain
link where it stands (sent home under `"home"`; with no link it is removed); the strike's
`position` governs only the landing's target, and `strike.by` must name the sweeper for any of it.
`effects` is refused as inert without a strike, contacts or a blast (and, settled here, beside a
strike that strikes nothing the sweeper takes).

Readings the row left open, settled on the plan's principles and written in the record: the
landing's capture is struck first (its position may write the sweeper's origin), then the swept
pieces in path order, each onto a square empty by then (two homes colliding: the first is reborn,
the second removed); a swept piece of the mover's own side is removed (a strike never meets its
own); the mover's side switch (`switches_sides`) is the landing's alone; a strike placed
`"choose"` stays refused beside a sweep (W1's narrower row: a sweep's word names no landing
square); a piece's sweeps share one switch. Q1063: a capture on a ride square is a landing
capture, and only the cleared part holds pieces in a legal sweep. Hand-built 6x6 specs, not chess.
"""

from __future__ import annotations

import copy
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
IDS = ("K", "S", "G", "P", "R")
DEMOTE = [{"from": "G", "to": "P"}]


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _name(square: int) -> str:
    return f"{'abcdef'[square % WIDTH]}{square // WIDTH + 1}"


def _spec(position: str = "stay", effect: str = "demote", effects: str | None = "path",
          sweep_range: int | None = None) -> dict[str, Any]:
    """Kings a1/a6, a sweeper (S, a rook's sweep) b1, guards c1/d1 and a rook-like R e1, all
    mirrored; a strike demoting a guard to a pawn (kings, sweepers and R immune), or under
    effect `"capture"` the position alone (circe under `"home"`)."""
    sweep: dict[str, Any] = {"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}
    if effects is not None:
        sweep["effects"] = effects
    if sweep_range is not None:
        sweep["sweep_range"] = sweep_range
    spec: dict[str, Any] = {
        "name": "the sweep that strikes (SR6a test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "S", "value": 5, "moves": [sweep]},
            {"id": "G", "value": 2, "moves": [KING]},
            {"id": "P", "value": 1, "moves": [KING]},
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
            "strike": {"effect": effect, "position": position, "immune": ["K", "S", "R"]},
        },
    }
    if effect != "capture":
        spec["rules"]["strike"]["chain"] = DEMOTE
    return spec


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


# The sweeper on a3 sweeps along the third rank: enemy guards on b3 and c3, an enemy R (immune to
# the strike) on d3, the mover's own guard on e3; f3 is empty.
ROW = {
    "a1": ("K", 0), "f6": ("K", 1), "a3": ("S", 0),
    "b3": ("G", 1), "c3": ("G", 1), "d3": ("R", 1), "e3": ("G", 0),
}
KINGS = {"a1": ("K", 0), "f6": ("K", 1)}


def _play(spec: dict[str, Any], placement: dict[str, tuple[str, int]] = ROW,
          origin: str = "a3", target: str = "f3") -> tuple[Engine, GameState, GameState, int]:
    engine = _engine(spec)
    state = engine.make_state(placement, side_to_move=0, moved=list(placement))
    move = _move(engine, state, origin, target)
    after = engine.apply(state, move)
    assert after.key == engine._compute_hash(after)
    return engine, state, after, move


class TestLanding:
    def test_landing_removes_the_swept_and_strikes_the_landing(self) -> None:
        """`"landing"` (left out): every swept piece goes; a capture on the landing is struck as
        any capture (rifle with a chain: the guard demoted where it stood, the sweeper back on
        its origin)."""
        placement = {**KINGS, "a3": ("S", 0), "b3": ("G", 1), "c3": ("G", 1), "d3": ("G", 1)}
        engine, _, after, _ = _play(_spec(effects=None), placement, target="d3")
        assert _board(engine, after) == {**KINGS, "a3": ("S", 0), "d3": ("P", 1)}

    def test_landing_written_is_the_game_without_it(self) -> None:
        assert spec_hash(_spec(effects="landing")) == spec_hash(_spec(effects=None))
        assert compile_spec(_spec(effects="landing")).strike.sweep_path == ()

    def test_landing_onto_an_empty_square_strikes_nothing(self) -> None:
        engine, _, after, _ = _play(_spec(effects=None))
        assert _board(engine, after) == {**KINGS, "f3": ("S", 0)}


class TestPath:
    def test_each_swept_piece_follows_its_link_where_it_stands(self) -> None:
        """`"path"`: the guards on b3 and c3 are demoted where they stood; the immune R and
        the mover's own guard are removed, as a sweep removes them."""
        engine, state, after, move = _play(_spec())
        assert _board(engine, after) == {
            **KINGS, "f3": ("S", 0), "b3": ("P", 1), "c3": ("P", 1),
        }
        assert engine.describe_move(state, move) == (
            "Sa3xf3 (sweeps b3, c3, d3, e3) (struck: P@b3, P@c3)"
        )

    def test_a_missing_link_removes_the_piece(self) -> None:
        """A swept pawn has no link (the chain demotes guards only), so it is removed."""
        placement = {**KINGS, "a3": ("S", 0), "b3": ("P", 1), "c3": ("G", 1)}
        engine, _, after, _ = _play(_spec(), placement, target="d3")
        assert _board(engine, after) == {**KINGS, "d3": ("S", 0), "c3": ("P", 1)}

    def test_the_landing_capture_reads_the_position(self) -> None:
        """Rifle's `stay` governs the landing alone: the sweeper stays on a3, the landing's guard
        is demoted on d3, and the swept guards on b3 and c3 are demoted where they stood."""
        placement = {**KINGS, "a3": ("S", 0), "b3": ("G", 1), "c3": ("G", 1), "d3": ("G", 1)}
        engine, _, after, _ = _play(_spec(), placement, target="d3")
        assert _board(engine, after) == {
            **KINGS, "a3": ("S", 0), "b3": ("P", 1), "c3": ("P", 1), "d3": ("P", 1),
        }

    def test_a_home_strike_sends_each_swept_piece_home(self) -> None:
        """Circe: each swept guard is reborn on its home (the compiled table's), the first of
        two whose homes collide reborn and the second removed."""
        spec = _spec("home", "capture")
        engine = _engine(spec)
        strike = engine.spec.strike
        guard = IDS.index("G")

        def home(square: str) -> str:
            n = engine.spec.n_types
            return _name(strike.rebirth[(1 * n + guard) * engine.spec.n_squares + _sq(square)])

        placement = {**KINGS, "a3": ("S", 0), "b3": ("G", 1), "c3": ("G", 1)}
        _, _, after, _ = _play(spec, placement, target="d3")
        homes = [home("b3"), home("c3")]
        expected = {**KINGS, "d3": ("S", 0)}
        for square in homes:
            expected.setdefault(square, ("G", 1))
        assert _board(engine, after) == expected
        # b3 and d3 share a colour, so their guards share a home.
        assert home("b3") == home("d3")
        placement = {**KINGS, "a3": ("S", 0), "b3": ("G", 1), "d3": ("G", 1)}
        _, _, after, _ = _play(spec, placement, target="e3")
        assert _board(engine, after) == {**KINGS, "e3": ("S", 0), home("b3"): ("G", 1)}

    def test_the_landing_is_struck_before_the_path(self) -> None:
        """A landing's circe rebirth that takes a swept square's home: the landing's guard is
        reborn first, so the swept guard whose home it is, is removed; undone exactly."""
        spec = _spec("home", "capture")
        engine = _engine(spec)
        strike = engine.spec.strike
        n = engine.spec.n_types
        guard = IDS.index("G")

        def home(square: str) -> str:
            return _name(strike.rebirth[(1 * n + guard) * engine.spec.n_squares + _sq(square)])

        # Two guards on squares of one colour: the swept one first in path order, the landing's
        # second; the landing's is reborn, the swept one finds its home taken.
        placement = {**KINGS, "a3": ("S", 0), "b3": ("G", 1), "d3": ("G", 1)}
        assert home("b3") == home("d3")
        _, state, after, move = _play(spec, placement, target="d3")
        assert _board(engine, after) == {**KINGS, "d3": ("S", 0), home("d3"): ("G", 1)}
        assert engine.perft(state, 2) == sum(
            len(engine.legal_moves(engine.apply(state, m))) for m in engine.legal_moves(state)
        )
        assert decode_move(move)[2] == MOVE_SWEEP

    def test_a_home_on_a_swept_square_is_undone(self) -> None:
        """The landing's guard reborn on a square the sweep emptied (its home), the guard swept
        from there removed: `unmake` takes the rebirth back before it restores the swept piece."""
        spec = _spec("home", "capture")
        engine = _engine(spec)
        strike = engine.spec.strike
        n = engine.spec.n_types
        guard = IDS.index("G")
        # The enemy guards' homes are on the sixth rank (c6, d6): a sweep along it.
        placement = {"a1": ("K", 0), "f4": ("K", 1), "a6": ("S", 0)}
        for landing in ("e6", "f6"):
            home = strike.rebirth[(1 * n + guard) * engine.spec.n_squares + _sq(landing)]
            if _name(home) in ("c6", "d6"):
                break
        swept = _name(home)
        placement = {**placement, swept: ("G", 1), landing: ("G", 1)}
        _, state, after, _ = _play(spec, placement, origin="a6", target=landing)
        assert _board(engine, after)[swept] == ("G", 1)
        assert landing in _board(engine, after) and _board(engine, after)[landing] == ("S", 0)

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(count(engine.apply(position, m), depth - 1)
                       for m in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)

    def test_by_names_the_sweeper(self) -> None:
        """`strike.by` naming a type that is not the sweeper: the sweeper strikes nothing, so
        `"path"` is refused as inert; naming it, the path is struck."""
        spec = _spec()
        spec["rules"]["strike"]["by"] = ["S"]
        engine, _, after, _ = _play(spec)
        assert _board(engine, after)["b3"] == ("P", 1)
        spec["rules"]["strike"]["by"] = ["G"]
        assert any("strikes nothing it sweeps" in e for e in validate_spec(spec))

    def test_a_short_sweep_strikes_only_what_it_clears(self) -> None:
        """Q1063: a sweep that clears two squares and rides on: the two cleared guards are
        struck, and the ride ends at the first piece past them, which it takes as a landing."""
        placement = {**KINGS, "a3": ("S", 0), "b3": ("G", 1), "c3": ("G", 1), "e3": ("G", 1)}
        engine, _, after, _ = _play(_spec(sweep_range=2), placement, target="e3")
        assert _board(engine, after) == {
            **KINGS, "a3": ("S", 0), "b3": ("P", 1), "c3": ("P", 1), "e3": ("P", 1),
        }


class TestTheFilter:
    def test_a_struck_piece_that_gives_check_makes_the_sweep_illegal(self) -> None:
        """Under checkmate a swept guard promoted to an R on b2 attacks the king on b1: the
        filter places it as `make` does, so the sweep is not offered under `"path"`; under
        `"landing"` the guard goes and the sweep is legal."""
        placement = {"b1": ("K", 0), "f6": ("K", 1), "a2": ("S", 0), "b2": ("G", 1)}
        for effects, offered in (("path", False), (None, True)):
            spec = _spec(effect="promote", effects=effects)
            spec["rules"]["win"] = "checkmate"
            spec["rules"]["strike"]["chain"] = [{"from": "G", "to": "R"}]
            spec["rules"]["strike"]["immune"] = ["K", "S"]
            engine = _engine(spec)
            state = engine.make_state(placement, side_to_move=0, moved=list(placement))
            pairs = {(move_from(m), move_to(m)) for m in engine.legal_moves(state)}
            assert ((_sq("a2"), _sq("c2")) in pairs) is offered, effects


class TestUndo:
    @pytest.mark.parametrize(("position", "effect"), [
        ("stay", "demote"), ("swap", "demote"), ("home", "capture"), ("home", "demote"),
    ])
    @pytest.mark.parametrize("effects", ["path", None])
    def test_perft_counts_what_apply_counts(
        self, position: str, effect: str, effects: str | None,
    ) -> None:
        engine = _engine(_spec(position, effect, effects))
        state = engine.make_state(ROW, side_to_move=0, moved=list(ROW))

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(count(engine.apply(position, m), depth - 1)
                       for m in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)

    def test_walks_keep_the_key(self) -> None:
        """Seeded walks from the opening, captures preferred: the key is the full scan's every
        ply, and `apply` leaves its argument as it was."""
        for position, effect in (("stay", "demote"), ("home", "capture")):
            engine = _engine(_spec(position, effect))
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


class TestTheRefusals:
    def test_a_sweep_beside_a_strike_validates(self) -> None:
        for effects in (None, "path"):
            for position, effect in (("stay", "demote"), ("swap", "demote"), ("home", "capture")):
                assert validate_spec(_spec(position, effect, effects)) == []

    def test_choose_is_refused_beside_a_sweep(self) -> None:
        spec = _spec("choose", effects=None)
        spec["rules"]["strike"]["landing"] = {"vectors": [[1, 0]], "symmetry": "rotate4"}
        assert "W1" in refused_ids(validate_spec(spec))

    def test_path_without_an_on_capture_rule_is_refused(self) -> None:
        spec = _spec()
        del spec["rules"]["strike"]
        assert any("leave effects out" in e for e in validate_spec(spec))

    def test_path_beside_a_shot_is_refused_as_inert(self) -> None:
        """Rifle's shot (effect `capture` placed `stay`) keeps no victim: every swept piece is
        removed under either value."""
        assert any("strikes nothing it sweeps" in e for e in validate_spec(_spec("stay", "capture")))

    def test_a_piece_whose_sweeps_disagree_is_refused(self) -> None:
        spec = _spec()
        spec["pieces"][1]["moves"].append(
            {"kind": "sweep", "vectors": [[1, 1]], "symmetry": "dihedral8"}
        )
        assert any("disagree on effects" in e for e in validate_spec(spec))

    def test_effects_is_a_sweep_s(self) -> None:
        spec = _spec()
        spec["pieces"][2]["moves"] = [{**KING, "effects": "path"}]
        assert any("only applies to a sweep" in e for e in validate_spec(spec))
        spec = _spec()
        spec["pieces"][1]["moves"][0]["effects"] = "everything"
        assert any("effects must be one of" in e for e in validate_spec(spec))

    def test_a_blast_beside_the_strike_is_refused(self) -> None:
        """W3's narrower row (SR6c lifted the rest: `tests/test_sweep_blast.py`): a blast
        beside a sweep and a strike, whose order nobody has written."""
        spec = _spec()
        spec["rules"]["explode_on_capture"] = {"radius": 1, "immune": ["K"]}
        assert "W3" in refused_ids(validate_spec(spec))

    def test_the_compiled_rule_carries_the_switch(self) -> None:
        compiled = compile_spec(_spec())
        assert compiled.strike is not None
        assert compiled.strike.sweep_path == tuple(t == IDS.index("S") for t in range(len(IDS)))
        assert compile_spec(_spec(effects=None)).strike.sweep_path == ()


def test_the_switch_names_another_game() -> None:
    spec = _spec()
    other = copy.deepcopy(spec)
    del other["pieces"][1]["moves"][0]["effects"]
    assert spec_hash(spec) != spec_hash(other)
