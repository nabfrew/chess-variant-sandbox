"""SR8k (`docs/plan-schema-next.md` §2.11 O16, *the mover's reading*, ruled 2026-10-28): a held
sweep.

A held sweep removes what its **holder** could capture with it: the other player's non-royal
pieces its capture list allows. The holder's pieces, a royal, another nobody's piece and a
shielded piece end its ray (never swept, never a landing), and a royal on its line is not in
check (a held piece gives none). It sweeps in its holder's frame (SR8i), promotes in its holder's
zone and stays nobody's (SR8j), and under ``effects: "path"`` beside a strike each swept piece is
struck as its holder's piece's sweep would strike it. The core walks it
(`foreign.rs::push_foreign_sweeps`, `Game::commanded_sweeps` with the holder's opponent as the
side it may take) and reads its path in the holder's frame (`Tables::sweep_path`);
`tests/neutral_oracle.py` reads the same moves off the holder's twin and is held to the core
here, list by list and by perft.

Hand-built 6x6 specs, not chess.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest
import tomllib

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import (
    MOVE_NORMAL,
    MOVE_SWEEP,
    move_foreign,
    move_from,
    move_promotion,
    move_to,
    move_type,
)
from chess_variant_sandbox.spec import refusals
from chess_variant_sandbox.spec.schema import validate_spec
from tests.neutral_oracle import NeutralOracle
from tests.test_held_in_holders_frame import (
    KING,
    _engine,
    _oracle_perft,
    _position,
    _spec,
    _sq,
    _walk,
)

HERE = Path(__file__).parent
ROOK_SWEEP = {"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}
FORWARD_SWEEP = {"kind": "sweep", "vectors": [[0, 1]]}


def _rook_sweeper() -> dict[str, Any]:
    """The held type ``H`` sweeps as a rook: it reads alike for both players (frame 0)."""
    return _spec({"moves": [copy.deepcopy(ROOK_SWEEP)]})


def _forward_sweeper() -> dict[str, Any]:
    """``H`` sweeps forward only: it reads differently for the two players, so it sweeps in
    its holder's frame (SR8i), and a played sweep's path is read there too."""
    return _spec({"moves": [copy.deepcopy(FORWARD_SWEEP)]})


def _promoting_sweeper() -> dict[str, Any]:
    """The forward sweeper promotes on its holder's last rank, to the rook, the guard or the
    king (a royal, never offered to a held piece, SR8j)."""
    spec = _forward_sweeper()
    spec["rules"]["promotion"] = [{"piece": "H", "rank": "last", "to": ["R", "G", "K"]}]
    return spec


def _shared_sweeper() -> dict[str, Any]:
    spec = _rook_sweeper()
    spec["pieces"][3]["control"] = "shared"
    return spec


def _short_sweeper() -> dict[str, Any]:
    """A rook sweep that clears its first two steps and rides on past them (Q1063)."""
    spec = _rook_sweeper()
    spec["pieces"][3]["moves"][0].update(range=5, sweep_range=2)
    return spec


ALL_SPECS = {
    "rook": _rook_sweeper, "forward": _forward_sweeper, "promoting": _promoting_sweeper,
    "shared": _shared_sweeper, "short": _short_sweeper,
}


def _flip(placement: dict[str, tuple[str, int]], side: int) -> dict[str, tuple[str, int]]:
    """``placement`` with the two players swapped where ``side`` is 1 (nobody's stays)."""
    if side == 0:
        return placement
    return {sq: (pid, owner if owner == 2 else 1 - owner) for sq, (pid, owner) in placement.items()}


def _held_from(engine: Engine, state, square: str) -> list[int]:
    return sorted(
        m for m in engine.legal_moves(state) if move_foreign(m) and move_from(m) == _sq(square)
    )


def _name(square: int) -> str:
    return f"{'abcdef'[square % 6]}{square // 6 + 1}"


# The held rook sweeper on c3, held by the side under test (its guards on b2 and b3, the other
# side's guard on d3). On its rank: the other side's guard d3 and rook e3 (both sweepable), f3
# empty; the holder's own guard b3 to its left. Up its file: c4 empty, the other side's king
# on c5. Down: another nobody's sweeper on c2.
ROW = {
    "a1": ("K", 0), "c5": ("K", 1), "b2": ("G", 0), "b3": ("G", 0),
    "c3": ("H", 2), "c2": ("H", 2), "d3": ("G", 1), "e3": ("R", 1),
}


class TestTheRefusalGoes:
    @pytest.mark.parametrize("name", sorted(ALL_SPECS))
    def test_each_spec_validates(self, name):
        assert validate_spec(ALL_SPECS[name]()) == []

    def test_the_row_is_lifted(self):
        table = tomllib.loads(
            (Path(refusals.__file__).parent / "refusals.toml").read_text(encoding="utf-8")
        )
        assert "O16" not in {row["id"] for row in table["refusal"]}
        assert "O16" in {row["id"] for row in table["lifted"] if row.get("by") == "SR8k"}

    def test_a_held_sweeper_that_takes_royals_validates(self):
        spec = _rook_sweeper()
        spec["pieces"][3]["takes_royal"] = True
        assert validate_spec(spec) == []


class TestWhatItRemoves:
    @pytest.mark.parametrize("side", [0, 1])
    def test_only_what_its_holder_could_capture(self, side):
        """Right: the other side's guard and rook are swept or taken, f3 reached over both.
        Left: the holder's guard ends the ray. Up: c4, and the king on c5 ends the ray. Down:
        the other nobody's piece ends it."""
        engine = _engine(_rook_sweeper())
        state = _position(engine, _flip(ROW, side), side=side)
        assert engine.controller_of(state, _sq("c3")) == side
        moves = _held_from(engine, state, "c3")
        assert {_name(move_to(m)) for m in moves} == {"d3", "e3", "f3", "c4"}
        kinds = {_name(move_to(m)): move_type(m) for m in moves}
        assert kinds == {"d3": MOVE_NORMAL, "e3": MOVE_SWEEP, "f3": MOVE_SWEEP, "c4": MOVE_NORMAL}

    @pytest.mark.parametrize("side", [0, 1])
    def test_the_swept_go_and_the_sweeper_stays_nobodys(self, side):
        engine = _engine(_rook_sweeper())
        state = _position(engine, _flip(ROW, side), side=side)
        move = next(m for m in _held_from(engine, state, "c3") if move_to(m) == _sq("f3"))
        assert {sq for sq, _ in engine.spec.sweep_victims(state.board, move, side)} == {
            _sq("d3"), _sq("e3"),
        }
        after = engine.apply(state, move)
        assert not after.board[_sq("d3")] and not after.board[_sq("e3")]
        assert engine.spec.owner_of(after.board[_sq("f3")]) == 2
        assert not after.board[_sq("c3")]
        assert engine._compute_hash(after) == after.key
        assert "sweeps" in engine.describe_move(state, move)

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_royal_on_its_line_is_not_in_check(self, side):
        engine = _engine(_rook_sweeper())
        state = _position(engine, _flip(ROW, side), side=side)
        assert not engine.in_check(state, 1 - side)

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_held_sweeper_that_takes_royals_never_sweeps_onto_one(self, side):
        """Q1077 beside O16: a held piece of a `takes_royal` type takes a royal by its other
        moves, but a royal ends a held sweep's ray, so the sweep gives no check."""
        spec = _rook_sweeper()
        spec["pieces"][3]["takes_royal"] = True
        engine = _engine(spec)
        state = _position(engine, _flip(ROW, side), side=side)
        assert "c5" not in {_name(move_to(m)) for m in _held_from(engine, state, "c3")}
        assert not engine.in_check(state, 1 - side)

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_shielded_piece_ends_the_ray(self, side):
        """SR6g's shield beside O16: the other side's warden on f4 shields its rook on e3, which
        ends the held ray and is never a landing."""
        spec = _rook_sweeper()
        spec["pieces"].append({"id": "W", "value": 2, "moves": [KING], "sight": [
            {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 1,
             "effect": "shield", "whose": "own"}]})
        spec["setup"]["placements"].append({"piece": "W", "squares": ["e1"]})
        engine = _engine(spec)
        state = _position(engine, _flip({**ROW, "f4": ("W", 1)}, side), side=side)
        targets = {_name(move_to(m)) for m in _held_from(engine, state, "c3")}
        assert "d3" in targets and "e3" not in targets and "f3" not in targets

    @pytest.mark.parametrize("side", [0, 1])
    def test_the_forward_sweeper_sweeps_toward_its_holders_opponent(self, side):
        """SR8i's frame: held by the first player it sweeps up the file, by the second down."""
        engine = _engine(_forward_sweeper())
        placement = {
            "a1": ("K", 0), "f6": ("K", 1), "b3": ("G", side), "c3": ("H", 2),
            "c4": ("G", 1), "c2": ("G", 0),
        }
        state = _position(engine, placement, side=side)
        moves = _held_from(engine, state, "c3")
        want = {"c4", "c5", "c6"} if side == 0 else {"c2", "c1"}
        assert {_name(move_to(m)) for m in moves} == want
        far = next(m for m in moves if _name(move_to(m)) == ("c6" if side == 0 else "c1"))
        assert move_type(far) == MOVE_SWEEP
        after = engine.apply(state, far)
        assert not after.board[_sq("c4" if side == 0 else "c2")]
        assert engine._compute_hash(after) == after.key
        # The path is read in the holder's frame: the first holder's frame has no ray down.
        assert engine.spec.sweep_victims(state.board, far, side)

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_held_sweep_promotes_in_its_holders_zone(self, side):
        """SR8j beside O16: over the other side's guard onto the holder's last rank, promoting
        to the rook or the guard (never the king), and stays nobody's."""
        engine = _engine(_promoting_sweeper())
        start, over, last = ("c4", "c5", "c6") if side == 0 else ("c3", "c2", "c1")
        placement = {"a1": ("K", 0), "f6": ("K", 1), "b3": ("G", side), "b4": ("G", side),
                     start: ("H", 2), over: ("G", 1 - side)}
        state = _position(engine, placement, side=side)
        promoting = {
            engine.spec.piece_ids[move_promotion(m) - 1]: m
            for m in _held_from(engine, state, start)
            if move_type(m) == MOVE_SWEEP and move_to(m) == _sq(last)
        }
        assert set(promoting) == {"R", "G"}
        after = engine.apply(state, promoting["R"])
        value = after.board[_sq(last)]
        assert engine.spec.owner_of(value) == 2 and engine.spec.piece_ids[engine.spec.type_of(value)] == "R"
        assert not after.board[_sq(over)]
        assert engine._compute_hash(after) == after.key


class TestBesideAStrike:
    @staticmethod
    def _striking() -> dict[str, Any]:
        """The rook sweeper under `effects: "path"` beside a strike that demotes a guard to a
        pawn where it stands (kings, the sweeper and the rook immune)."""
        spec = _rook_sweeper()
        spec["pieces"][3]["moves"][0]["effects"] = "path"
        spec["pieces"].append({"id": "P", "value": 1, "moves": [KING]})
        spec["rules"]["win"] = "royal_capture"
        spec["rules"]["strike"] = {
            "effect": "demote", "position": "stay", "immune": ["K", "H", "R"],
            "chain": [{"from": "G", "to": "P"}],
        }
        return spec

    @pytest.mark.parametrize("side", [0, 1])
    def test_each_swept_piece_is_struck(self, side):
        engine = _engine(self._striking())
        placement = {
            "a1": ("K", 0), "a6": ("K", 1), "b2": ("G", side), "b4": ("G", side),
            "c3": ("H", 2), "d3": ("G", 1 - side), "e3": ("G", 1 - side),
        }
        state = _position(engine, placement, side=side)
        move = next(m for m in _held_from(engine, state, "c3") if move_to(m) == _sq("f3"))
        after = engine.apply(state, move)
        pawn = engine.spec.piece_ids.index("P")
        for square in ("d3", "e3"):
            value = after.board[_sq(square)]
            assert engine.spec.type_of(value) == pawn and engine.spec.owner_of(value) == 1 - side
        assert engine.spec.owner_of(after.board[_sq("f3")]) == 2
        assert engine._compute_hash(after) == after.key


# ------------------------------------------------------------------ the oracle


def _walk_spec(make) -> dict[str, Any]:
    """Kings c1/c6, rooks a1/f1 and their images, guards on the second ranks, and two nobody's
    sweepers on b3/e3 (and images b4/e4), one held by each side from the opening."""
    spec = make()
    nobodys = spec["setup"]["placements"][3]
    spec["setup"]["placements"] = [
        {"piece": "K", "squares": ["c1"]},
        {"piece": "R", "squares": ["a1", "f1"]},
        {"piece": "G", "squares": ["a2", "b2", "e2", "f2"]},
        {**nobodys, "squares": ["b3", "e3"]},
    ]
    return spec


class TestTheOracle:
    @pytest.mark.parametrize("name", sorted(ALL_SPECS))
    def test_walks_agree(self, name):
        spec = _walk_spec(ALL_SPECS[name])
        assert sum(_walk(spec, seed, 40) for seed in range(2)) > 0

    @pytest.mark.parametrize("name", sorted(ALL_SPECS))
    def test_a_perft_agrees(self, name):
        """The core's perft(3) from the opening, where both sides hold a sweeper, against the
        oracle's legal lists walked through `apply`."""
        spec = _walk_spec(ALL_SPECS[name])
        engine = _engine(spec)
        oracle = NeutralOracle(spec, engine)
        state = engine.initial_state()
        # A forward sweeper's file is closed by its mirrored image at the opening.
        held = any(move_foreign(m) for m in oracle.legal_moves(state))
        assert held is (name not in ("forward", "promoting"))
        # Either player moves a shared sweeper, so its tree is the widest: two plies keep it fast.
        depth = 2 if name == "shared" else 3
        assert engine.perft(state, depth) == _oracle_perft(engine, oracle, state, depth)

    @pytest.mark.parametrize("name", sorted(ALL_SPECS))
    def test_held_sweeps_are_played(self, name):
        """The walks reach held sweeps over pieces, each one's victims the other side's."""
        spec = _walk_spec(ALL_SPECS[name])
        engine = _engine(spec)
        swept = 0
        for seed in range(4):
            rng = random.Random(seed)
            state = engine.initial_state()
            for _ in range(60):
                if engine.terminal(state) is not None:
                    break
                moves = sorted(engine.legal_moves(state))
                sweeps = [m for m in moves if move_foreign(m) and move_type(m) == MOVE_SWEEP]
                for m in sweeps:
                    for _, value in engine.spec.sweep_victims(state.board, m, state.side_to_move):
                        assert engine.spec.owner_of(value) == 1 - state.side_to_move
                    swept += 1
                state = engine.apply(state, rng.choice(sweeps or moves))
        assert swept > 0

    @pytest.mark.slow
    @pytest.mark.parametrize("name", sorted(ALL_SPECS))
    def test_long_walks_agree(self, name):
        spec = _walk_spec(ALL_SPECS[name])
        assert sum(_walk(spec, seed, 100) for seed in range(10)) > 10


class TestTheLoopReachesIt:
    """The second of the two edits by which the loop reaches the pair (no stored game is one
    edit from it: `tests/test_operators.py::_CANNOT_ENTER`): a held type that slides, its slide
    turned into a sweep by `sweep-a-slider`."""

    def test_sweep_a_slider_turns_a_held_slider_into_a_held_sweeper(self):
        from chess_variant_sandbox.loop.operators import OPERATORS
        from tests.feature_inventory import _schema_one_hundred_twenty_eight_uses as uses

        apply = next(op for op in OPERATORS if op.name == "sweep-a-slider")
        crowd = HERE.parent / "archive" / "the-crowd.json"
        parent = json.loads(crowd.read_text(encoding="utf-8"))
        condottiere = next(p for p in parent["pieces"] if p["id"] == "C")
        condottiere["moves"] = [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]
        del condottiere["takes_royal"]
        assert validate_spec(parent) == [], validate_spec(parent)
        assert not uses(parent)
        for seed in range(60):
            child = copy.deepcopy(parent)
            if apply.apply(child, random.Random(seed)) is None:
                continue
            assert validate_spec(child) == [], validate_spec(child)
            if uses(child):
                return
        pytest.fail("sweep-a-slider never entered the pair in 60 seeds")
