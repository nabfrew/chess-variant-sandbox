"""SR3f (`docs/plan-schema-next.md` §2.11 rows O1 and O2): pieces nobody owns beside wild
pieces and beside the duck.

The rulings (2026-10-28): the third owner's block is shared by kind. A wild piece eats a
nobody's piece its type may capture as it eats a player's -- only another wild piece, the duck
and a portal end stop it -- and the eat pays nobody (O1). The duck lands on empty squares only,
so a nobody's piece is to it as any piece (O2).

The probe (SR2's method, run first): fed past the validator, the core refused the pair at its
boundary (`tables.rs::third_owner`); with that refusal gone, the compiler crossed the neutral
kinds alone, so the wild type was no route walker and never moved, and the duck's type was not
the duck's kind. With the kinds merged (`compile._compile_control`) the duck played as ruled,
but the wild piece waited in front of every nobody's piece (`owner_of == WILD`), its type able
to take it or not, and a player taking a nobody's piece on a route square was paid the
route's wild bounty besides its type's (`Tables::bounty_paid` read the block as wild). So O2
was the boundary and the compiler, and O1 also the phase's eat (`Tables::halts_wild`) and the
bounty (`is_wild_piece`).

Hand-built 6x6 specs, not chess: two kings and a rook each, a condottiere (majority control)
and a stone (stands) placed for nobody on a wild piece's loop, and a wild wolf. Every walk
checks the incremental key against `_compute_hash` each ply and holds each phase to the
oracle below (`_phase_oracle`), both seats.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables
from chess_variant_sandbox.engine.compile import (
    CONTROL_DUCK,
    CONTROL_MAJORITY,
    CONTROL_NEUTRAL,
    CONTROL_ROUTE,
    compile_spec,
)
from chess_variant_sandbox.engine.moves import MOVE_DUCK, move_foreign, move_to, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from tests.feature_inventory import _schema_sixty_three_uses

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
LOOP = ["b3", "c3", "d3", "e3", "e4", "d4", "c4", "b4"]
WILD = 2
WOLF = 4  # the wolf's type (and the duck's), declared last


def _kind(engine: Engine, value: int) -> int:
    control = engine.spec.control
    assert control is not None
    return control.kind[engine.spec.type_of(value)]


def _move(engine: Engine, state: GameState, origin: str, target: str) -> int:
    wanted = (_sq(origin), _sq(target))
    (found,) = [m for m in engine.legal_moves(state) if (m & 0xFF, (m >> 8) & 0xFF) == wanted]
    return found


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _base() -> dict[str, Any]:
    return {
        "schema_version": 31,
        "name": "the wolf and the crowd",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "C", "name": "condottiere", "value": 3, "moves": [KING], "control": "majority"},
            {"id": "S", "name": "stone", "value": 2, "moves": [KING]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "R", "squares": ["f1"]},
            {"piece": "C", "player": "nobody", "squares": ["c3"]},
            {"piece": "S", "player": "nobody", "squares": ["e3"]},
        ]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 80},
    }


def _wild(captures: list[str] | None = None, bounty: bool = False) -> dict[str, Any]:
    """The crowd with a wolf on the loop b3..b4, which passes c3 and e3 (both nobody's) and
    their mirrors c4 and e4. ``captures`` narrows what the wolf may take."""
    spec = _base()
    wolf: dict[str, Any] = {"id": "W", "name": "wolf", "value": 3, "moves": [KING]}
    if captures is not None:
        wolf["captures"] = captures
    spec["pieces"].append(wolf)
    spec["rules"]["npcs"] = [{"piece": "W", "route": list(LOOP)}]
    if bounty:
        # A type's bounty on the pieces nobody owns: taking one pays the taker, unless the
        # taker is the wolf, which is nobody.
        for piece in spec["pieces"]:
            if piece["id"] in ("C", "S"):
                piece["bounty"] = 4
        spec["rules"]["scoring"] = {"count": "points"}
    return spec


def _duck(goes: str = "anywhere") -> dict[str, Any]:
    spec = _base()
    duck: dict[str, Any] = {"id": "D", "name": "duck", "value": 0, "moves": []}
    spec["rules"]["duck"] = {"piece": "D"}
    if goes == "by_moves":
        duck["moves"] = [copy.deepcopy(KING) | {"symmetry": "dihedral8"}]
        spec["rules"]["duck"] = {"piece": "D", "goes": "by_moves", "start": "d2"}
    spec["pieces"].append(duck)
    return spec


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


def _next(square: int) -> int:
    names = [_sq(name) for name in LOOP]
    return names[(names.index(square) + 1) % len(names)]


def _phase_oracle(engine: Engine, before: GameState, after: GameState, steps: tuple) -> None:
    """One wolf on the loop. Where it stood at the move's start (and was not taken by it), the
    phase stepped it onto its next square exactly when that square, as the move left it, was
    empty or held a piece its type may capture other than a wild piece or the duck; and an eat
    paid nobody (the points the move's own capture paid aside, which the wolf's eat cannot
    share a ply with here: one wolf, and it either was taken or stepped)."""
    spec = engine.spec
    wolf_type = WOLF
    here = [sq for sq, v in enumerate(before.board) if v and spec.owner_of(v) == WILD
            and spec.type_of(v) == wolf_type]
    if not here:
        assert steps == ()
        return
    (origin,) = here
    target = _next(origin)
    wolf_after = [sq for sq, v in enumerate(after.board) if v and spec.owner_of(v) == WILD
                  and spec.type_of(v) == wolf_type]
    if not wolf_after:
        assert steps == ()  # the move took the wolf
        return
    if steps:
        (step,) = steps
        assert (step.origin, step.target) == (origin, target), steps
        assert wolf_after == [target]
        eaten = step.eaten
        if eaten:
            eaten_type = spec.type_of(eaten)
            assert spec.capture_allowed[wolf_type][eaten_type], (eaten_type, steps)
            assert spec.owner_of(eaten) != WILD or _kind(engine, eaten) in (
                CONTROL_NEUTRAL, CONTROL_MAJORITY,
            )
    else:
        # It waited: its next square holds what it may not eat (the move may have put it
        # there; the phase only moves the wolf, so the square reads the same after).
        assert wolf_after == [origin], (origin, wolf_after)
        standing = after.board[target]
        assert standing != 0, (origin, target)
        assert not spec.capture_allowed[wolf_type][spec.type_of(standing)], (origin, target)


def _walk(engine: Engine, seed: int, plies: int, check) -> dict[str, int]:
    rng = random.Random(seed)
    state = engine.initial_state()
    seen = {"held": 0, "eaten_nobodys": 0, "waits_at_nobodys": 0, "duck": 0}
    for _ in range(plies):
        if engine.terminal(state) is not None:
            break
        moves = engine.legal_moves(state)
        if not moves:
            break
        move = rng.choice(moves)
        seen["held"] += int(move_foreign(move))
        after, steps = engine.apply_with_phase(state, move)
        assert engine._compute_hash(after) == after.key
        check(engine, state, move, after, steps, seen)
        state = after
    return seen


def _wild_check(engine, state, move, after, steps, seen) -> None:
    _phase_oracle(engine, state, after, steps)
    for step in steps:
        if step.eaten and engine.spec.owner_of(step.eaten) == WILD:
            seen["eaten_nobodys"] += 1
            assert after.points == state.points  # the eat pays nobody
    spec = engine.spec
    for sq, value in enumerate(after.board):
        if value and spec.owner_of(value) == WILD and spec.type_of(value) == WOLF and not steps:
            standing = after.board[_next(sq)]
            if standing and spec.owner_of(standing) == WILD:
                seen["waits_at_nobodys"] += 1


def _duck_check(engine, state, move, after, steps, seen) -> None:
    if move_type(move) == MOVE_DUCK:
        seen["duck"] += 1
        landing = move_to(move)
        assert state.board[landing] == 0 or _kind(engine, state.board[landing]) == CONTROL_DUCK
    # No nobody's piece is ever under the duck: every neutral that stood stays or moved.
    for value in after.board:
        if value and engine.spec.owner_of(value) == WILD:
            assert _kind(engine, value) in (CONTROL_DUCK, CONTROL_NEUTRAL, CONTROL_MAJORITY)


# ------------------------------------------------------------------------- the validator


class TestTheValidator:
    def test_wild_pieces_and_the_duck_beside_nobodys_pieces_validate(self):
        for spec in (_wild(), _wild(["K", "R"]), _wild(bounty=True), _duck(), _duck("by_moves")):
            assert validate_spec(spec) == [], validate_spec(spec)

    def test_nobodys_pieces_beside_portals_are_no_longer_refused(self):
        """SR3g lifted P10, the portal arm this file pinned (`tests/test_nobody_beside_portals.py`)."""
        spec = _base()
        spec["rules"]["portals"] = [{"orange": "a3", "blue": "f4"}]
        assert "P10" not in refused_ids(validate_spec(spec)), validate_spec(spec)

    def test_a_wild_type_or_the_duck_is_never_placed_for_nobody(self):
        wild = _wild()
        wild["setup"]["placements"].append({"piece": "W", "player": "nobody", "squares": ["a5"]})
        assert any("walks its route" in error for error in validate_spec(wild))
        duck = _duck()
        duck["setup"]["placements"].append({"piece": "D", "player": "nobody", "squares": ["a5"]})
        assert validate_spec(duck) != []

    def test_the_duck_never_starts_on_a_nobodys_piece(self):
        spec = _duck("by_moves")
        spec["rules"]["duck"]["start"] = "c3"
        assert any("does not leave empty" in error for error in validate_spec(spec))

    def test_the_inventory_names_both(self):
        assert _schema_sixty_three_uses(_wild()) == {"rules.npcs.nobodys"}
        assert _schema_sixty_three_uses(_duck()) == {"rules.duck.nobodys"}
        assert _schema_sixty_three_uses(_base()) == set()

    def test_describe_says_it(self):
        assert "eats it if its type may capture" in describe_spec(_wild())
        assert "never lands on a piece nobody owns" in describe_spec(_duck())


# ------------------------------------------------------------------------- the kinds


class TestTheKinds:
    def test_each_type_keeps_its_own_kind(self):
        wild = compile_spec(_wild())
        assert wild.control is not None and wild.control.kind == (
            CONTROL_ROUTE, CONTROL_ROUTE, CONTROL_MAJORITY, CONTROL_NEUTRAL, CONTROL_ROUTE,
        )
        duck = compile_spec(_duck())
        assert duck.control is not None and duck.control.kind == (
            0, 0, CONTROL_MAJORITY, CONTROL_NEUTRAL, CONTROL_DUCK,
        )

    def test_a_neutral_game_alone_keeps_its_kinds(self):
        tables = build_tables(compile_spec(_base()))
        assert tables["control_kind"] == tables["neutral_control"]


# ------------------------------------------------------------------------- O1, move by move


def _position(engine: Engine, placement: dict[str, tuple[str, int]], side: int) -> GameState:
    return engine.make_state(placement, side_to_move=side, moved=list(placement))


class TestTheWolfEatsWhatItMay:
    """The wolf on b3, c3 holding a nobody's piece: after any quiet move, by either seat."""

    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("piece", ["C", "S"])
    def test_it_eats_a_nobodys_piece_its_type_may_capture(self, side, piece):
        engine = _engine(_wild())
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0), "f6": ("R", 1),
            "b3": ("W", 2), "c3": (piece, 2),
        }, side)
        move = _move(engine, state, *(("f1", "f2") if side == 0 else ("f6", "f5")))
        after, steps = engine.apply_with_phase(state, move)
        assert [(s.origin, s.target) for s in steps] == [(_sq("b3"), _sq("c3"))]
        assert steps[0].eaten == state.board[_sq("c3")]
        assert engine.spec.type_of(after.board[_sq("c3")]) == WOLF
        assert after.points == state.points
        assert engine._compute_hash(after) == after.key

    @pytest.mark.parametrize("side", [0, 1])
    def test_it_waits_in_front_of_one_it_may_not(self, side):
        engine = _engine(_wild(["K", "R", "S"]))
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0), "f6": ("R", 1),
            "b3": ("W", 2), "c3": ("C", 2), "e3": ("S", 2),
        }, side)
        for move in engine.legal_moves(state):
            after, steps = engine.apply_with_phase(state, move)
            if after.board[_sq("b3")] == 0:
                continue  # the move took the wolf
            assert steps == (), engine.describe_move(state, move)
            assert after.board[_sq("c3")] == state.board[_sq("c3")] or move_foreign(move)
            assert engine._compute_hash(after) == after.key

    def test_the_eat_pays_nobody_a_bounty(self):
        engine = _engine(_wild(bounty=True))
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0), "f6": ("R", 1),
            "b3": ("W", 2), "c3": ("S", 2), "d5": ("C", 2),
        }, 0)
        after, steps = engine.apply_with_phase(state, _move(engine, state, "f1", "f2"))
        assert steps and steps[0].eaten
        assert after.points == (0.0, 0.0)
        # A player's capture of a nobody's piece is paid its type's bounty, as TC4 says.
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "f3": ("R", 0), "f6": ("R", 1), "e3": ("S", 2),
            "b5": ("W", 2),
        }, 0)
        assert engine.apply(state, _move(engine, state, "f3", "e3")).points[0] == 4.0


# ------------------------------------------------------------------------- the walks


class TestWalks:
    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize("captures", [None, ["K", "R", "S"], ["K", "R"]])
    def test_a_wolf_beside_the_crowd(self, seed, captures):
        engine = _engine(_wild(captures))
        _walk(engine, seed, 60, _wild_check)

    def test_the_walks_meet_the_cases(self):
        """The walks above eat a nobody's piece and wait in front of one, and a held
        condottiere moves: the oracle was asked something."""
        totals = {"held": 0, "eaten_nobodys": 0, "waits_at_nobodys": 0}
        for seed in range(8):
            for captures in (None, ["K", "R", "S"]):
                seen = _walk(_engine(_wild(captures)), seed, 60, _wild_check)
                for key in totals:
                    totals[key] += seen[key]
        assert all(totals.values()), totals

    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize("goes", ["anywhere", "by_moves"])
    def test_the_duck_beside_the_crowd(self, seed, goes):
        engine = _engine(_duck(goes))
        seen = _walk(engine, seed, 60, _duck_check)
        assert seen["duck"] > 0

    @pytest.mark.parametrize("goes", ["anywhere", "by_moves"])
    def test_no_duck_half_offers_a_nobodys_square(self, goes):
        engine = _engine(_duck(goes))
        rng = random.Random(3)
        state = engine.initial_state()
        held = 0
        for _ in range(60):
            if engine.terminal(state) is not None:
                break
            moves = engine.legal_moves(state)
            if not moves:
                break
            for move in moves:
                if move_type(move) == MOVE_DUCK:
                    target = state.board[move_to(move)]
                    assert target == 0 or _kind(engine, target) == CONTROL_DUCK
            move = rng.choice(moves)
            held += int(move_foreign(move))
            state = engine.apply(state, move)
            assert engine._compute_hash(state) == state.key
        assert held > 0


class TestPerft:
    @pytest.mark.parametrize("build", [_wild, _duck])
    def test_perft_matches_an_apply_count(self, build):
        engine = _engine(build())
        state = engine.initial_state()

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(count(engine.apply(position, m), depth - 1)
                       for m in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)
