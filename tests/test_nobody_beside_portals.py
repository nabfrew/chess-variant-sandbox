"""SR3g (`docs/plan-schema-next.md` §2.10 rows P9 and P10): the purse and pieces nobody owns
beside portals.

The rulings (2026-10-28). P9: a portal end's square is not empty for a buy; an empty fold's
square is, and a bought piece never goes through it (a drop never does). P10: nobody's pieces
and portal ends share the third owner's block by kind; a held piece transits as its holder's
would, a fold may land on a nobody's piece, and an exit onto one takes it where the mover may.

The probe (SR2's method, run first, both seats, past the validator): P9 the core already played
-- `generate_buys` offers only empty board squares, so never a momentum end's, and a fold is
no board byte, so an empty fold is offered and the bought piece stands in it -- so P9 is the
validator's lift alone. P10 the core refused at its boundary (`tables.rs::third_owner`); with
that arm gone the fold copy of movegen (`generate_folded`) listed no held move at all, and the
held walk (`foreign.rs::push_foreign_piece`) knew neither a transit nor a go-through. A fold's
own move onto a nobody's piece and a player's exit onto one played as ruled already.

What stays refused: a held type that takes a royal (`takes_royal`) beside portals, by the
validator and by the core (`Tables::majority_rule`), since `held_attacks` reads no portal.

The oracle for a held piece is its **twin**: the same position with the held piece made the
holder's own piece of the same type, whose transits and go-throughs the core has played since
schemas 19 and 24 (held to `tests/portal_oracle.py` and `tests/fold_oracle.py` there). A held
piece's portal moves are exactly the twin's that land on an empty square or on a non-royal
piece of the opponent (a held piece never takes a royal, its holder's piece or a nobody's).
Hand-built 6x6 specs, not chess; every walk checks the incremental key against
`_compute_hash` each ply.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import _core
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables
from chess_variant_sandbox.engine.compile import WILD, compile_spec
from chess_variant_sandbox.engine.moves import (
    MOVE_BUY,
    MOVE_FOLD,
    MOVE_FOREIGN,
    MOVE_NORMAL,
    MOVE_PORTAL,
    move_fold,
    move_foreign,
    move_from,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from tests.feature_inventory import _schema_sixty_four_uses

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
STEP = {"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}
BOTH = {"orange": [0, 1], "blue": [0, 1]}
MOMENTUM = {"orange": "b4", "blue": "e5", "moves": [STEP], "moved_by": BOTH}
FOLD = {"orange": "d4", "blue": "a5", "kind": "fold", "moves": [STEP], "moved_by": BOTH}
GAMES = {"momentum": [MOMENTUM], "fold": [FOLD], "both": [MOMENTUM, FOLD]}
ENDS = {"b4": ("1o", WILD), "e5": ("1b", WILD)}  # the momentum pair, where it starts
MARKET = Path(__file__).parent / "_purse_market.json"


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _name(square: int) -> str:
    return "abcdef"[square % WIDTH] + str(square // WIDTH + 1)


def _crowd(groups: list[dict[str, Any]], captures: list[str] | None = None,
           majority: bool = True) -> dict[str, Any]:
    """Kings and rooks, a condottiere (majority control) on c3 and a stone on e3 placed for
    nobody (mirrored to c4, e4), and ``groups`` of portals. ``captures`` narrows the rook's."""
    rook: dict[str, Any] = {"id": "R", "value": 5, "moves": [ROOK]}
    if captures is not None:
        rook["captures"] = captures
    condottiere: dict[str, Any] = {"id": "C", "name": "condottiere", "value": 3, "moves": [KING]}
    if majority:
        condottiere["control"] = "majority"
    return copy.deepcopy({
        "name": "the crowd at the portals (SR3g test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            rook,
            condottiere,
            {"id": "S", "name": "stone", "value": 2, "moves": [KING]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "R", "squares": ["f1"]},
            {"piece": "C", "player": "nobody", "squares": ["c3"]},
            {"piece": "S", "player": "nobody", "squares": ["e3"]},
        ]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 80,
                  "portals": groups},
    })


def _market() -> dict[str, Any]:
    """The purse market (`tests/_purse_market.json`) with a momentum pair on e1/e6 and a fold
    pair on f1/f6, both on a back rank a bought piece goes to."""
    spec = json.loads(MARKET.read_text(encoding="utf-8"))
    spec["name"] = "the purse market at the portals (SR3g test spec)"
    spec["setup"]["placements"] = [
        {"piece": "K", "squares": ["c1"]}, {"piece": "R", "squares": ["a1"]},
        {"piece": "N", "squares": ["b1"]}, {"piece": "P", "squares": ["a2", "b2", "c2"]},
    ]
    spec["rules"]["portals"] = [
        {"orange": "e1", "blue": "e6", "moves": [STEP], "moved_by": BOTH},
        {"orange": "f1", "blue": "f6", "kind": "fold", "moves": [STEP], "moved_by": BOTH},
    ]
    return spec


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


# ------------------------------------------------------------------------- the validator


class TestTheValidator:
    def test_both_pairs_validate(self):
        for groups in GAMES.values():
            assert validate_spec(_crowd(groups)) == []
        assert validate_spec(_market()) == []

    def test_neither_row_is_refused(self):
        errors = validate_spec(_market()) + validate_spec(_crowd([MOMENTUM, FOLD]))
        assert not {"P9", "P10"} & refused_ids(errors)

    def test_a_held_type_that_takes_a_royal_stays_refused_beside_portals(self):
        spec = _crowd([MOMENTUM])
        spec["pieces"][2]["takes_royal"] = True
        assert any("takes_royal beside rules.portals" in e for e in validate_spec(spec))
        del spec["rules"]["portals"]
        assert validate_spec(spec) == []

    def test_the_inventory_names_both(self):
        assert _schema_sixty_four_uses(_market()) == {"rules.purse.portals"}
        assert _schema_sixty_four_uses(_crowd([FOLD])) == {"rules.portals.nobodys"}
        bare = _crowd([FOLD])
        del bare["rules"]["portals"]
        assert _schema_sixty_four_uses(bare) == set()

    def test_describe_says_it(self):
        assert "never bought onto a portal's square" in describe_spec(_market())
        text = describe_spec(_crowd([MOMENTUM, FOLD]))
        assert "as its holder's piece would" in text
        assert "a fold may move under a piece nobody owns" in text


# ------------------------------------------------------------------------- the core's boundary


class TestTheBoundary:
    def test_the_core_takes_nobodys_pieces_beside_portals(self):
        """The last arm of `third_owner`'s neutral refusal went (SR3g)."""
        for groups in GAMES.values():
            _core.Core(build_tables(compile_spec(_crowd(groups))))

    def test_the_core_refuses_a_held_royal_taker_beside_portals(self):
        spec = _crowd([MOMENTUM])
        del spec["rules"]["portals"]
        spec["pieces"][2]["takes_royal"] = True
        tables = dict(build_tables(compile_spec(spec)))
        tables["has_portals"] = True
        with pytest.raises(ValueError, match="^a held neutral that takes a royal beside portals"):
            _core.Core(tables)


# ------------------------------------------------------------------------- P9, the purse


def _buy_squares(engine: Engine, state: GameState) -> set[int]:
    return {move_to(m) for m in engine.legal_moves(state) if move_type(m) == MOVE_BUY}


class TestBuying:
    @pytest.mark.parametrize("side", [0, 1])
    def test_never_onto_a_momentum_end_and_onto_an_empty_fold(self, side):
        engine = _engine(_market())
        state = engine.make_state({
            "c1": ("K", 0), "c6": ("K", 1), "e1": ("1o", WILD), "e6": ("1b", WILD),
        }, side_to_move=side, moved=["c1", "c6", "e1", "e6"])
        rank = 1 if side == 0 else 6
        assert _buy_squares(engine, state) == {
            _sq(f"{f}{rank}") for f in "abdf"
        }, sorted(_name(s) for s in _buy_squares(engine, state))

    @pytest.mark.parametrize("side", [0, 1])
    def test_the_bought_piece_stands_in_the_fold(self, side):
        engine = _engine(_market())
        state = engine.make_state({
            "c1": ("K", 0), "c6": ("K", 1), "e1": ("1o", WILD), "e6": ("1b", WILD),
        }, side_to_move=side, moved=["c1", "c6", "e1", "e6"])
        fold = _sq("f1" if side == 0 else "f6")
        buys = [m for m in engine.legal_moves(state) if move_type(m) == MOVE_BUY
                and move_to(m) == fold]
        assert buys
        for move in buys:
            after = engine.apply(state, move)
            assert after.board[fold] != 0 and engine.spec.owner_of(after.board[fold]) == side
            assert after.folds == state.folds  # nothing went through, no fold moved
            assert engine._compute_hash(after) == after.key

    def test_not_into_a_fold_a_piece_stands_in(self):
        engine = _engine(_market())
        state = engine.make_state({
            "c1": ("K", 0), "c6": ("K", 1), "e1": ("1o", WILD), "e6": ("1b", WILD),
            "f1": ("P", 0),
        }, side_to_move=0, moved=["c1", "c6", "e1", "e6", "f1"])
        assert _sq("f1") not in _buy_squares(engine, state)

    def test_a_momentum_end_that_moves_off_frees_its_square(self):
        engine = _engine(_market())
        state = engine.make_state({
            "c1": ("K", 0), "c6": ("K", 1), "e1": ("1o", WILD), "e6": ("1b", WILD),
        }, side_to_move=0, moved=["c1", "c6", "e1", "e6"])
        (step,) = [m for m in engine.legal_moves(state)
                   if (move_from(m), move_to(m)) == (_sq("e1"), _sq("e2"))]
        state = engine.apply(state, step)
        (back,) = [m for m in engine.legal_moves(state)
                   if (move_from(m), move_to(m)) == (_sq("c6"), _sq("b6"))]
        state = engine.apply(state, back)
        assert _sq("e1") in _buy_squares(engine, state)

    @pytest.mark.parametrize("seed", range(6))
    def test_walks_keep_the_key_and_never_buy_onto_an_end(self, seed):
        engine = _engine(_market())
        rng = random.Random(seed)
        state = engine.initial_state()
        bought = 0
        for _ in range(80):
            if engine.terminal(state) is not None:
                break
            moves = engine.legal_moves(state)
            if not moves:
                break
            for move in moves:
                if move_type(move) == MOVE_BUY:
                    assert state.board[move_to(move)] == 0
            move = rng.choice(moves)
            bought += int(move_type(move) == MOVE_BUY)
            state = engine.apply(state, move)
            assert engine._compute_hash(state) == state.key
        assert bought > 0


# ------------------------------------------------------------------------- P10, the held piece


def _placement(engine: Engine, state: GameState) -> dict[str, tuple[str, int]]:
    spec = engine.spec
    return {
        _name(square): (spec.piece_ids[spec.type_of(value)], spec.owner_of(value))
        for square, value in enumerate(state.board) if value
    }


def _portal_moves(moves: list[int], origin: int) -> set[int]:
    """The moves from ``origin`` a portal makes: transits and go-throughs, less the foreign bit."""
    return {
        m & ~MOVE_FOREIGN for m in moves
        if move_from(m) == origin and (
            move_type(m) == MOVE_PORTAL or (move_type(m) == MOVE_NORMAL and move_fold(m))
        )
    }


def _held_oracle(engine: Engine, twin: Engine, state: GameState) -> int:
    """Every held piece's portal moves are its twin's (the holder's own piece of its type on
    the same square) that land on an empty square or an opponent's non-royal piece. Returns how
    many such moves the position offered."""
    spec = engine.spec
    side = state.side_to_move
    moves = engine.pseudo_legal_moves(state)
    held = {move_from(m) for m in moves if move_foreign(m)}
    offered = 0
    for origin in held:
        placement = _placement(engine, state)
        pid, _ = placement[_name(origin)]
        placement[_name(origin)] = (pid, side)
        folds = [_name(square) for square in state.folds if square >= 0]
        mirror = twin.make_state(placement, side_to_move=side, moved=list(placement), folds=folds)
        expected = set()
        for m in _portal_moves(twin.pseudo_legal_moves(mirror), origin):
            standing = state.board[move_to(m)]
            if standing == 0 or (
                spec.owner_of(standing) == 1 - side
                and spec.type_of(standing) not in spec.royal_types
            ):
                expected.add(m)
        got = _portal_moves([m for m in moves if move_foreign(m)], origin)
        assert got == expected, (
            _name(origin), sorted(engine.describe_move(state, m | MOVE_FOREIGN) for m in got),
            sorted(twin.describe_move(mirror, m) for m in expected),
        )
        offered += len(got)
    return offered


class TestTheHeldPiece:
    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("game", sorted(GAMES))
    def test_it_transits_as_its_holders_piece_would(self, side, game):
        engine = _engine(_crowd(GAMES[game]))
        twin = _engine(_crowd(GAMES[game], majority=False))
        holders = ("b2", "c2") if side == 0 else ("b3", "d2")
        placement = {"a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0), "f6": ("R", 1),
                     "c3": ("C", WILD), "c4": ("S", WILD)}
        if game != "fold":
            placement |= ENDS
        for square in holders:
            placement[square] = ("R", side)
        state = engine.make_state(placement, side_to_move=side, moved=list(placement))
        assert _held_oracle(engine, twin, state) > 0
        kinds = {
            "transit" if move_type(m) == MOVE_PORTAL else "through"
            for m in engine.legal_moves(state)
            if move_foreign(m) and move_from(m) == _sq("c3")
            and (move_type(m) == MOVE_PORTAL or move_fold(m))
        }
        assert kinds == {"momentum": {"transit"}, "fold": {"through"},
                         "both": {"transit", "through"}}[game]

    @pytest.mark.parametrize("side", [0, 1])
    def test_its_moves_play_and_keep_the_key(self, side):
        engine = _engine(_crowd([MOMENTUM, FOLD]))
        holders = ("b2", "c2") if side == 0 else ("b3", "d2")
        placement = {"a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0), "f6": ("R", 1),
                     "c3": ("C", WILD), "c4": ("S", WILD)} | ENDS
        for square in holders:
            placement[square] = ("R", side)
        state = engine.make_state(placement, side_to_move=side, moved=list(placement))
        for move in engine.legal_moves(state):
            if not move_foreign(move):
                continue
            after = engine.apply(state, move)
            assert engine._compute_hash(after) == after.key
            moved = after.board[move_to(move)]
            assert engine.spec.owner_of(moved) == WILD  # still nobody's
            assert engine.spec.piece_ids[engine.spec.type_of(moved)] == "C"
            assert after.board[_sq("c3")] == 0


class TestFoldsAndExits:
    def test_a_fold_moved_onto_a_nobodys_piece_holds_it(self):
        engine = _engine(_crowd([FOLD]))
        state = engine.make_state({"a1": ("K", 0), "a6": ("K", 1), "d3": ("S", WILD)},
                                  side_to_move=0, moved=["a1", "a6", "d3"])
        (fold,) = [m for m in engine.legal_moves(state)
                   if move_type(m) == MOVE_FOLD and move_to(m) == _sq("d3")]
        after = engine.apply(state, fold)
        assert after.board[_sq("d3")] == state.board[_sq("d3")]
        assert _sq("d3") in after.folds
        assert engine._compute_hash(after) == after.key

    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("may", [True, False])
    def test_an_exit_onto_a_nobodys_piece_takes_it_where_the_mover_may(self, side, may):
        engine = _engine(_crowd([MOMENTUM], None if may else ["K", "R", "C"]))
        rook, exit_ = ("b1", "e6") if side == 0 else ("b6", "e4")
        placement = {"a1": ("K", 0), "a6": ("K", 1), rook: ("R", side), exit_: ("S", WILD)} | ENDS
        state = engine.make_state(placement, side_to_move=side, moved=list(placement))
        transits = [m for m in engine.legal_moves(state)
                    if move_type(m) == MOVE_PORTAL and move_to(m) == _sq(exit_)]
        assert bool(transits) == may
        for move in transits:
            after = engine.apply(state, move)
            assert engine.spec.owner_of(after.board[_sq(exit_)]) == side
            assert engine._compute_hash(after) == after.key


# ------------------------------------------------------------------------- the walks


class TestWalks:
    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("game", sorted(GAMES))
    def test_a_crowd_at_the_portals(self, seed, game):
        engine = _engine(_crowd(GAMES[game]))
        twin = _engine(_crowd(GAMES[game], majority=False))
        rng = random.Random(seed)
        state = engine.initial_state()
        for _ in range(60):
            if engine.terminal(state) is not None:
                break
            _held_oracle(engine, twin, state)
            moves = engine.legal_moves(state)
            if not moves:
                break
            state = engine.apply(state, rng.choice(moves))
            assert engine._compute_hash(state) == state.key

    def test_the_walks_meet_the_cases(self):
        """The walks above offer a held piece a portal move, and play one."""
        offered = played = 0
        for game in sorted(GAMES):
            engine = _engine(_crowd(GAMES[game]))
            twin = _engine(_crowd(GAMES[game], majority=False))
            for seed in range(6):
                rng = random.Random(seed)
                state = engine.initial_state()
                for _ in range(60):
                    if engine.terminal(state) is not None:
                        break
                    offered += _held_oracle(engine, twin, state)
                    moves = engine.legal_moves(state)
                    if not moves:
                        break
                    held = [m for m in moves if move_foreign(m) and (
                        move_type(m) == MOVE_PORTAL or move_fold(m))]
                    move = rng.choice(held) if held and rng.random() < 0.5 else rng.choice(moves)
                    played += int(move in held)
                    state = engine.apply(state, move)
                    assert engine._compute_hash(state) == state.key
        assert offered > 0 and played > 0, (offered, played)


class TestPerft:
    @pytest.mark.parametrize("build", [lambda: _crowd([MOMENTUM, FOLD]), _market])
    def test_perft_matches_an_apply_count(self, build):
        engine = _engine(build())
        state = engine.initial_state()

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(count(engine.apply(position, m), depth - 1)
                       for m in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)
