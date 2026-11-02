"""SR3d (`docs/plan-schema-next.md` §2.9, row K2): the duck beside momentum portals and folds.

The ruled reading: the duck is an uncapturable piece to every transit and fold go-through (an
exit or a fold end it stands on is blocked); the duck never transits, its own move never lands
on a portal end, and a fold is floor to it (it may stop in an empty fold, which never moves
it). Not in the ruling, decided here: a straight-back bar (``no_straight_back``) set on a piece
half waits through the mover's own duck half for the other player's piece half, the move it
binds (`Game::bar_reader`).

The oracle is `tests/fold_oracle.py` (which holds `tests/portal_oracle.py`'s momentum rules),
read off a twin of the spec in which every player's piece lists every type but the duck in
``captures``, and the duck is a third owner: a piece that blocks, that nothing may take, and
that a fold may move under, which is the ruling word for word. The core's moves of each kind
are held to it at every piece half of seeded walks, the incremental key to the full scan at
every ply, and every duck half to the squares the ruling leaves it. Each walk counts the
positions where the duck changed what the portals offer (the oracle asked again with the duck
lifted), so a walk where the duck never met a portal fails. Hand-built specs, not chess.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest
from fold_oracle import FoldOracle, Folds
from portal_oracle import Position
from test_schema_nineteen import _core_transits, _unmoved
from test_schema_twenty_four import _core_fold_moves, _core_ordinary, _folds, _last_fold

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import rust_supports
from chess_variant_sandbox.engine.compile import WILD, compile_spec
from chess_variant_sandbox.engine.moves import (
    DUCK_PASS,
    MOVE_DUCK,
    MOVE_FOLD,
    MOVE_NORMAL,
    move_from,
    move_promotion,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import validate_spec

KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
BISHOP = {"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4"}
KNIGHT = {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}
STEP = {"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}
BOTH = {"orange": [0, 1], "blue": [0, 1]}
MOMENTUM = {"orange": "b3", "blue": "e4", "moved_by": BOTH, "moves": [STEP],
            "no_straight_back": True}
FOLD = {"orange": "c4", "blue": "d3", "kind": "fold", "moved_by": BOTH, "moves": [STEP],
        "no_straight_back": True}


def _game(groups: list[dict[str, Any]], goes: str = "anywhere") -> dict[str, Any]:
    """6x6, won by taking the king: kings, a rook, a bishop and a knight a side, the duck."""
    duck: dict[str, Any] = {"piece": "D"}
    if goes != "anywhere":
        duck["goes"] = goes
    return copy.deepcopy({
        "name": "the duck pond (SR3d test spec)",
        "board": {"width": 6, "height": 6},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "B", "value": 3, "moves": [BISHOP]},
            {"id": "N", "value": 3, "moves": [KNIGHT]},
            {"id": "D", "value": 0, "moves": [KING] if goes == "by_moves" else []},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "R", "player": 0, "squares": ["b1"]},
            {"piece": "B", "player": 0, "squares": ["c1"]},
            {"piece": "N", "player": 0, "squares": ["e1"]},
            {"piece": "K", "player": 1, "squares": ["f6"]},
            {"piece": "R", "player": 1, "squares": ["e6"]},
            {"piece": "B", "player": 1, "squares": ["d6"]},
            {"piece": "N", "player": 1, "squares": ["b6"]},
        ]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 120,
                  "duck": duck, "portals": copy.deepcopy(groups)},
    })


METERS = ("duck mattered", "duck in a fold", "through a portal", "end moves")
GAMES = {
    "momentum": [MOMENTUM],
    "fold": [FOLD],
    "both": [MOMENTUM, FOLD],
}


def _oracle_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """The twin the oracle reads: no player's piece may take the duck."""
    twin = copy.deepcopy(spec)
    ids = [piece["id"] for piece in twin["pieces"] if piece["id"] != "D"]
    for piece in twin["pieces"]:
        if piece["id"] != "D":
            piece["captures"] = ids
    return twin


def _position(engine: Engine, state: GameState, lift_duck: bool = False) -> Position:
    """The oracle's position: a portal end is ``"portal"``, the duck ``"duck"`` (or gone)."""
    spec = engine.spec
    found: Position = {}
    for square, value in enumerate(state.board):
        if not value:
            continue
        pid = spec.piece_ids[spec.type_of(value)]
        owner = spec.owner_of(value)
        if owner != WILD:
            found[square] = (pid, owner)
        elif pid == "D":
            if not lift_duck:
                found[square] = (pid, "duck")
        else:
            found[square] = (pid, "portal")
    return found


def _core_end_moves(engine: Engine, state: GameState) -> set[tuple[int, int]]:
    """A momentum end's own moves: `test_schema_nineteen._core_portal_moves` less a fold's
    move from under the duck (whose origin holds the duck's nobody's byte)."""
    spec = engine.spec
    return {
        (move_from(move), move_to(move))
        for move in engine.pseudo_legal_moves(state)
        if move_type(move) == MOVE_NORMAL and state.board[move_from(move)]
        and spec.owner_of(state.board[move_from(move)]) == WILD
    }


def _offered(oracle: FoldOracle, position: Position, folds: Folds, player: int,
             unmoved: set[int]) -> tuple[Any, ...]:
    return (
        oracle.ordinary_moves(position, folds, player, unmoved),
        oracle.momentum_transits(position, folds, player, unmoved),
    )


def _piece_half(engine: Engine, oracle: FoldOracle, state: GameState,
                last_end: tuple[int, int] | None,
                last_fold: tuple[str, int, int] | None) -> tuple[bool, int]:
    """The core and the oracle agree on every kind of move of a piece half. Whether the
    duck changed what the portals offer, and how many moves go through one."""
    position = _position(engine, state)
    folds = _folds(engine, state)
    player = state.side_to_move
    unmoved = _unmoved(state)
    ordinary, transits = _offered(oracle, position, folds, player, unmoved)
    assert _core_ordinary(engine, state) == ordinary
    assert _core_transits(engine, state) == transits
    assert _core_fold_moves(engine, state) == oracle.fold_moves(position, folds, player, last_fold)
    assert _core_end_moves(engine, state) == oracle.momentum_moves(
        position, folds, player, last_end, unmoved,
    )
    pseudo = engine.pseudo_legal_moves(state)
    assert sorted(engine.legal_moves(state)) == sorted(pseudo)  # no check rule beside the duck
    duck = engine.duck_square(state)
    assert all(move_to(m) != duck for m in pseudo if move_type(m) != MOVE_FOLD)
    gone_through = len(transits) + sum(1 for item in ordinary if item[3] is not None)
    if duck is None:
        return False, gone_through
    lifted = _offered(oracle, _position(engine, state, lift_duck=True), folds, player, unmoved)
    # The duck's own square is a move once it is lifted; only the portals' answers count.
    through = {item for item in lifted[0] if item[3] is not None}
    mattered = (
        through != {item for item in ordinary if item[3] is not None} or lifted[1] != transits
    )
    return mattered, gone_through


def _duck_half(engine: Engine, state: GameState, goes: str) -> int:
    """Every duck move ends on a square the ruling leaves it; how many may stop in a fold."""
    moves = engine.legal_moves(state)
    assert moves and all(move_type(m) == MOVE_DUCK for m in moves)
    folds = set(_folds(engine, state).values())
    if any(move_promotion(m) == DUCK_PASS for m in moves):
        assert len(moves) == 1
        return 0
    targets = {move_to(m) for m in moves}
    assert all(state.board[target] == 0 for target in targets)  # never a portal end
    if goes == "anywhere":
        assert targets == {sq for sq, value in enumerate(state.board) if value == 0}
    return len(targets & folds)


def _walk(name: str, goes: str, seed: int, plies: int = 60) -> dict[str, int]:
    spec = _game(GAMES[name], goes)
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    oracle = FoldOracle(_oracle_spec(spec))
    seen = dict.fromkeys(METERS, 0)
    rng = random.Random(seed)
    state = engine.initial_state()
    last_end: tuple[int, int] | None = None
    last_fold: tuple[str, int, int] | None = None
    for _ in range(plies):
        if engine.terminal(state) is not None:
            break
        if engine.duck_owed(state):
            seen["duck in a fold"] += _duck_half(engine, state, goes)
            move = rng.choice(engine.legal_moves(state))
            folds_before = _folds(engine, state)
            state = engine.apply(state, move)
            assert _folds(engine, state) == folds_before  # a duck move never moves a fold
        else:
            mattered, gone_through = _piece_half(engine, oracle, state, last_end, last_fold)
            seen["duck mattered"] += mattered
            seen["through a portal"] += gone_through
            moves = engine.legal_moves(state)
            portal = [m for m in moves if move_type(m) == MOVE_FOLD
                      or _moves_an_end(engine, state, m)]
            seen["end moves"] += len(portal)
            move = rng.choice(portal) if portal and rng.random() < 0.4 else rng.choice(moves)
            value = state.board[move_from(move)]
            # The bar a piece half sets waits through the duck half (`Game::bar_reader`).
            last_end = (
                (move_from(move), move_to(move))
                if move_type(move) == MOVE_NORMAL and value
                and engine.spec.owner_of(value) == WILD else None
            )
            last_fold = _last_fold(engine, state, move)
            state = engine.apply(state, move)
        assert state.key == engine._compute_hash(state)
    return seen


def _moves_an_end(engine: Engine, state: GameState, move: int) -> bool:
    """A momentum end's own move (a piece half has no duck move)."""
    value = state.board[move_from(move)]
    return move_type(move) == MOVE_NORMAL and bool(value) and engine.spec.owner_of(value) == WILD


class TestTheSpecs:
    @pytest.mark.parametrize("name", sorted(GAMES))
    @pytest.mark.parametrize("goes", ["anywhere", "by_moves"])
    def test_they_validate(self, name: str, goes: str) -> None:
        """The validator and `rust_supports` agree: both take the pair since SR3d."""
        spec = _game(GAMES[name], goes)
        assert validate_spec(spec) == []
        assert rust_supports(compile_spec(spec)) is None

    def test_the_duck_may_start_in_a_fold_and_not_on_a_momentum_end(self) -> None:
        spec = _game(GAMES["both"])
        spec["rules"]["duck"]["start"] = "c4"
        assert validate_spec(spec) == []
        spec["rules"]["duck"]["start"] = "b3"
        assert any("rules.duck.start b3" in error for error in validate_spec(spec))


class TestWalks:
    """Seeded walks, the key checked every ply and every piece half held to the oracle."""

    @pytest.mark.parametrize("name", sorted(GAMES))
    @pytest.mark.parametrize("goes", ["anywhere", "by_moves"])
    def test_a_walk_agrees_with_the_oracle(self, name: str, goes: str) -> None:
        totals = dict.fromkeys(METERS, 0)
        for seed in range(4):
            for key, count in _walk(name, goes, seed).items():
                totals[key] += count
        assert totals["duck mattered"] > 0, totals
        assert totals["end moves"] > 0 and totals["through a portal"] > 0, totals
        if name != "momentum" and goes == "anywhere":
            assert totals["duck in a fold"] > 0, totals


class TestPositions:
    """The ruling square by square, on both seats."""

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_fold_moves_under_the_duck_and_the_duck_blocks_its_far_end(self, side: int) -> None:
        engine = Engine(_game(GAMES["fold"]))
        state = engine.make_state(
            {"a1": ("K", side), "f6": ("K", 1 - side), "c1": ("R", side), "d2": ("D", 2)},
            side_to_move=side, moved=["a1", "f6", "c1"],
        )
        names = {name: engine._square_index(name) for name in ("c1", "c4", "d2", "d3")}
        # The fold on d3 steps onto the duck on d2: the duck stands in it.
        fold_moves = _core_fold_moves(engine, state)
        assert ("1b", names["d3"], names["d2"]) in fold_moves
        move = next(m for m in engine.legal_moves(state) if move_type(m) == MOVE_FOLD
                    and move_from(m) == names["d3"] and move_to(m) == names["d2"])
        after = engine.apply(state, move)
        assert engine.duck_square(after) == names["d2"]
        assert _folds(engine, after)["1b"] == names["d2"]
        # The rook runs up the c-file into the fold on c4: its far end holds the duck, so
        # the rook stops in c4 (and passes it) and never comes out on d2.
        state = engine.make_state(
            {"a1": ("K", side), "f6": ("K", 1 - side), "c1": ("R", side), "d2": ("D", 2)},
            side_to_move=side, moved=["a1", "f6", "c1"], folds=["c4", "d2"],
        )
        rook = {move_to(m) for m in engine.legal_moves(state) if move_from(m) == names["c1"]}
        assert names["c4"] in rook and names["d2"] not in rook
        # Lift the duck, and the same run goes through the fold onto d2.
        state = engine.make_state(
            {"a1": ("K", side), "f6": ("K", 1 - side), "c1": ("R", side)},
            side_to_move=side, moved=["a1", "f6", "c1"], folds=["c4", "d2"],
        )
        rook = {move_to(m) for m in engine.legal_moves(state) if move_from(m) == names["c1"]}
        assert names["d2"] in rook

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_bar_waits_through_the_duck_half(self, side: int) -> None:
        """The mover steps the momentum end b3 to b2; after its duck half, the other player
        may move that end anywhere but straight back to b3."""
        engine = Engine(_game(GAMES["momentum"]))
        state = engine.make_state(
            {"a1": ("K", side), "f6": ("K", 1 - side), "b3": ("1o", WILD), "e4": ("1b", WILD),
             "d5": ("D", 2)},
            side_to_move=side, moved=["a1", "f6"],
        )
        b2, b3 = engine._square_index("b2"), engine._square_index("b3")
        step = next(m for m in engine.legal_moves(state)
                    if move_from(m) == b3 and move_to(m) == b2)
        state = engine.apply(state, step)
        assert engine.duck_owed(state)
        state = engine.apply(state, engine.legal_moves(state)[0])
        assert state.side_to_move == 1 - side and state.key == engine._compute_hash(state)
        back = {(move_from(m), move_to(m)) for m in engine.legal_moves(state)}
        assert (b2, b3) not in back
        assert (b2, engine._square_index("c2")) in back

    def test_no_bar_waits_for_a_player_who_may_not_move_the_end(self) -> None:
        """Q883 beside the duck: the first player steps the blue end, which only it may move;
        the bar would bind the second player's piece half, which cannot move it, so none is
        set (and none keyed), duck half or not."""
        group = dict(MOMENTUM, moved_by={"orange": [0, 1], "blue": [0]})
        spec = _game([group])
        assert validate_spec(spec) == []
        engine = Engine(spec)
        state = engine.make_state(
            {"a1": ("K", 0), "f6": ("K", 1), "b3": ("1o", WILD), "e4": ("1b", WILD),
             "d5": ("D", 2)},
            moved=["a1", "f6"],
        )
        e4, e3 = engine._square_index("e4"), engine._square_index("e3")
        step = next(m for m in engine.legal_moves(state) if move_from(m) == e4 and move_to(m) == e3)
        state = engine.apply(state, step)
        assert state.portal_bar == (-1, -1)
        state = engine.apply(state, engine.legal_moves(state)[0])
        assert state.portal_bar == (-1, -1) and state.key == engine._compute_hash(state)

    @pytest.mark.parametrize("name", sorted(GAMES))
    def test_unmake_puts_the_duck_and_the_portals_back(self, name: str) -> None:
        """The core's perft (make and unmake) against a count built from `apply` alone, from
        a position where the duck stands beside both kinds of end."""
        engine = Engine(_game(GAMES[name]))
        placement = {"a1": ("K", 0), "f6": ("K", 1), "b1": ("R", 0), "c6": ("R", 1),
                     "e5": ("D", 2)}
        if name != "fold":
            placement.update({"b3": ("1o", WILD), "e4": ("1b", WILD)})
        state = engine.make_state(placement, moved=list(placement))

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(count(engine.apply(position, m), depth - 1)
                       for m in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)
