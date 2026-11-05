"""SR6b (`docs/plan-schema-next.md` §2.6 row W2): a sweep beside a contact table, and the sweep
switch `moves[].effects` read there.

The ruling (2026-10-27): **the sweep switch**, as W1. Under `"landing"` (the default) the swept
pieces are removed and the contact acts on the landing square's capture alone, as on any
capture. Under `"path"` each swept piece meets the mover by its link read with the mover's type
at the move's start: its `onto_to` where it stands; no link, removed; a friendly one by an `own`
link. The mover's `mover_to` applies once, from the landing's link if it names one, else from
the last swept meeting that names one.

Readings settled here (written in the record, generation 91): the landing's meeting is written
first, so a landing link that keeps the mover on its origin (`position: "stay"`) is where a
path's `mover_to` lands it; the contact's `position` governs only the landing; a friendly swept
piece with no own link (or no own table) is removed, as a sweep removes it; wild and nobody's
pieces are removed (SR6a's reading); a contact placed `"choose"` stays refused beside a sweep
(W2's narrower row); `"path"` beside a table with no link whose mover is the sweeper is refused
as inert. Q1063: a capture on a ride square is the landing's. Hand-built 6x6 specs, not chess.
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
IDS = ("K", "S", "G", "P", "R", "Q", "T", "X")
# The table: an enemy guard is demoted where it stands; meeting an enemy R turns the sweeper into
# a Q (the R removed); the mover's own guard becomes an R; meeting its own pawn turns the sweeper
# into a T (the pawn removed); an enemy X (which never moves) becomes a pawn.
LINKS = [
    {"mover": "S", "onto": "G", "onto_to": "P"},
    {"mover": "S", "onto": "R", "mover_to": "Q"},
    {"mover": "S", "onto": "G", "own": True, "onto_to": "R"},
    {"mover": "S", "onto": "P", "own": True, "mover_to": "T"},
    {"mover": "S", "onto": "X", "onto_to": "P"},
]


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _name(square: int) -> str:
    return f"{'abcdef'[square % WIDTH]}{square // WIDTH + 1}"


def _spec(position: str = "stay", effects: str | None = "path",
          sweep_range: int | None = None, win: str = "royal_capture") -> dict[str, Any]:
    sweep: dict[str, Any] = {"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}
    if effects is not None:
        sweep["effects"] = effects
    if sweep_range is not None:
        sweep["sweep_range"] = sweep_range
    return {
        "name": "the sweep that meets its path (SR6b test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "S", "value": 5, "moves": [sweep]},
            {"id": "G", "value": 2, "moves": [KING]},
            {"id": "P", "value": 1, "moves": [KING]},
            {"id": "R", "value": 3, "moves": [KING]},
            {"id": "Q", "value": 6, "moves": [KING]},
            {"id": "T", "value": 4, "moves": [KING]},
            {"id": "X", "value": 1, "moves": []},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "S", "squares": ["b1"]},
            {"piece": "G", "squares": ["c1", "d1"]},
            {"piece": "R", "squares": ["e1"]},
            {"piece": "P", "squares": ["b2", "c2"]},
        ]},
        "rules": {
            "win": win, "stalemate": "draw", "max_plies": 60,
            "contacts": {"position": position, "immune": ["K"], "links": copy.deepcopy(LINKS)},
        },
    }


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
# The sweeper on a3: an enemy guard b3, an enemy R c3, its own guard d3 and own pawn e3; f3 empty.
ROW = {
    **KINGS, "a3": ("S", 0),
    "b3": ("G", 1), "c3": ("R", 1), "d3": ("G", 0), "e3": ("P", 0),
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


def _count(engine: Engine, position: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(_count(engine, engine.apply(position, m), depth - 1)
               for m in engine.legal_moves(position))


class TestLanding:
    def test_landing_removes_the_swept_and_meets_the_landing(self) -> None:
        """`"landing"`: every swept piece goes, friendly ones too; the landing's guard meets the
        sweeper by its link (demoted where it stood, the sweeper back on its origin)."""
        placement = {**KINGS, "a3": ("S", 0), "b3": ("R", 1), "c3": ("P", 0), "d3": ("G", 1)}
        engine, _, after, _ = _play(_spec(effects=None), placement, target="d3")
        assert _board(engine, after) == {**KINGS, "a3": ("S", 0), "d3": ("P", 1)}

    def test_landing_onto_an_empty_square_meets_nothing(self) -> None:
        engine, _, after, _ = _play(_spec(effects=None))
        assert _board(engine, after) == {**KINGS, "f3": ("S", 0)}

    def test_landing_written_is_the_game_without_it(self) -> None:
        assert spec_hash(_spec(effects="landing")) == spec_hash(_spec(effects=None))
        assert compile_spec(_spec(effects="landing")).strike.sweep_path == ()


class TestPath:
    def test_each_swept_piece_meets_the_mover_by_its_link(self) -> None:
        """`"path"`: the enemy guard demoted on b3, the enemy R removed, the own guard an R on
        d3 by its own link, the own pawn removed; the mover takes the last `mover_to` named in
        path order (the own pawn's T, not the R's Q)."""
        engine, state, after, move = _play(_spec())
        assert _board(engine, after) == {
            **KINGS, "b3": ("P", 1), "d3": ("R", 0), "f3": ("T", 0),
        }
        assert engine.describe_move(state, move) == (
            "Sa3xf3 (sweeps b3, c3, d3, e3) (met: P@b3, R@d3) /S->T"
        )

    def test_a_missing_link_removes_the_piece(self) -> None:
        """An enemy pawn and an own R have no link: both removed, the mover unchanged."""
        placement = {**KINGS, "a3": ("S", 0), "b3": ("P", 1), "c3": ("R", 0)}
        engine, _, after, _ = _play(_spec(), placement, target="d3")
        assert _board(engine, after) == {**KINGS, "d3": ("S", 0)}

    def test_the_landing_s_mover_to_wins(self) -> None:
        """The landing's link names a `mover_to` (an R taken on c3: Q), so the swept own pawn's T
        is not applied; under `stay` the changed mover stands on its origin."""
        placement = {**KINGS, "a3": ("S", 0), "b3": ("P", 0), "c3": ("R", 1)}
        engine, state, after, move = _play(_spec(), placement, target="c3")
        assert _board(engine, after) == {**KINGS, "a3": ("Q", 0)}
        assert engine.describe_move(state, move) == "Sa3xc3 (sweeps b3)/S->Q (stands fast)"

    def test_a_path_mover_to_lands_where_the_landing_kept_the_mover(self) -> None:
        """The landing's guard is demoted under `stay`, which keeps the sweeper on a3; the swept
        R's `mover_to` then turns it into a Q there."""
        placement = {**KINGS, "a3": ("S", 0), "b3": ("R", 1), "c3": ("G", 1)}
        engine, state, after, move = _play(_spec(), placement, target="c3")
        assert _board(engine, after) == {**KINGS, "a3": ("Q", 0), "c3": ("P", 1)}
        assert engine.describe_move(state, move).endswith(" /S->Q")

    def test_swap_places_the_landing_s_survivor_on_the_origin(self) -> None:
        placement = {**KINGS, "a3": ("S", 0), "b3": ("R", 1), "c3": ("G", 1)}
        engine, _, after, _ = _play(_spec("swap"), placement, target="c3")
        assert _board(engine, after) == {**KINGS, "a3": ("P", 1), "c3": ("Q", 0)}

    def test_a_short_sweep_meets_only_what_it_clears(self) -> None:
        """Q1063: a sweep clearing two squares and riding on: the cleared guard and R meet the
        mover, the ride's capture on e3 is the landing's (demoted, the mover kept on a3, where
        the swept R's Q applies)."""
        placement = {**KINGS, "a3": ("S", 0), "b3": ("G", 1), "c3": ("R", 1), "e3": ("G", 1)}
        engine, _, after, _ = _play(_spec(sweep_range=2), placement, target="e3")
        assert _board(engine, after) == {
            **KINGS, "a3": ("Q", 0), "b3": ("P", 1), "e3": ("P", 1),
        }


class TestTheFilter:
    def test_a_met_piece_that_gives_check_makes_the_sweep_illegal(self) -> None:
        """Under checkmate an enemy X (which attacks nothing) on b2 becomes a pawn beside the king
        on b1: the filter places it as `make` does, so the sweep is not offered under `"path"`;
        under `"landing"` the X goes and the sweep is legal."""
        placement = {"b1": ("K", 0), "f6": ("K", 1), "a2": ("S", 0), "b2": ("X", 1)}
        for effects, offered in (("path", False), (None, True)):
            engine = _engine(_spec(effects=effects, win="checkmate"))
            state = engine.make_state(placement, side_to_move=0, moved=list(placement))
            pairs = {(move_from(m), move_to(m)) for m in engine.legal_moves(state)}
            assert ((_sq("a2"), _sq("c2")) in pairs) is offered, effects


class TestUndo:
    @pytest.mark.parametrize("position", ["stay", "swap"])
    @pytest.mark.parametrize("effects", ["path", None])
    def test_perft_counts_what_apply_counts(self, position: str, effects: str | None) -> None:
        engine = _engine(_spec(position, effects))
        for placement in (ROW, {**KINGS, "a3": ("S", 0), "b3": ("R", 1), "c3": ("G", 1)}):
            state = engine.make_state(placement, side_to_move=0, moved=list(placement))
            assert engine.perft(state, 3) == _count(engine, state, 3)

    def test_walks_keep_the_key(self) -> None:
        """Seeded walks from the opening, sweeps preferred: the key is the full scan's every
        ply, and `apply` leaves its argument as it was."""
        for position in ("stay", "swap"):
            engine = _engine(_spec(position))
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
    @pytest.mark.parametrize("position", ["stay", "swap"])
    @pytest.mark.parametrize("effects", ["path", "landing", None])
    def test_a_sweep_beside_contacts_validates(self, position: str, effects: str | None) -> None:
        assert validate_spec(_spec(position, effects)) == []

    def test_choose_is_refused_beside_a_sweep(self) -> None:
        spec = _spec("choose", effects=None)
        spec["rules"]["contacts"]["landing"] = {"vectors": [[1, 0]], "symmetry": "rotate4"}
        assert "W2" in refused_ids(validate_spec(spec))

    def test_path_beside_a_table_that_never_names_the_sweeper_is_refused(self) -> None:
        spec = _spec()
        spec["rules"]["contacts"]["links"] = [{"mover": "G", "onto": "G", "onto_to": "P"}]
        assert any("names it as the mover" in e for e in validate_spec(spec))
        spec["rules"]["contacts"]["links"] = [{"mover": "S", "onto": "K", "onto_to": "P"},
                                              {"mover": "G", "onto": "G", "onto_to": "P"}]
        assert any("names it as the mover" in e for e in validate_spec(spec))

    def test_the_compiled_rule_carries_the_switch(self) -> None:
        compiled = compile_spec(_spec())
        assert compiled.strike is not None
        assert compiled.strike.sweep_path == tuple(t == IDS.index("S") for t in range(len(IDS)))
        assert compile_spec(_spec(effects=None)).strike.sweep_path == ()

    def test_a_mover_to_alone_sets_the_switch(self) -> None:
        """A table whose only sweeper link changes the mover still meets the path."""
        spec = _spec()
        spec["rules"]["contacts"]["position"] = "advance"
        spec["rules"]["contacts"]["links"] = [{"mover": "S", "onto": "R", "mover_to": "Q"}]
        placement = {**KINGS, "a3": ("S", 0), "b3": ("R", 1)}
        engine, _, after, _ = _play(spec, placement, target="d3")
        assert _board(engine, after) == {**KINGS, "d3": ("Q", 0)}


def test_the_switch_names_another_game() -> None:
    spec = _spec()
    other = copy.deepcopy(spec)
    del other["pieces"][1]["moves"][0]["effects"]
    assert spec_hash(spec) != spec_hash(other)
