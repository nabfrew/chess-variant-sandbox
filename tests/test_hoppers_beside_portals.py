"""F10b group 1 (Q752b): a hopping piece in the rarer generators, held to the declared oracles.

`docs/plan-tests.md` F10b: the core's lines no test reached because no test spec put a hop
beside a portal, a fold or a commanding sight. A cannon (a hop that captures, a slide that
moves), a grasshopper (a hop landing just past its screen, one of whose hops needs an enemy
screen and an unmoved piece) and pawns with first-move leaps and slides play on a 6 x 6
board with a momentum pair and a fold pair that either player may move, each group moving by
a step, a slide and a hop; a promotion that is optional, and one that fires on leaving the
zone. The core's transits, a portal end's own moves, a fold's own moves, the ordinary moves
with the fold rules and the legal list are held to `tests/portal_oracle.py` and
`tests/fold_oracle.py` at every ply of seeded random walks (`test_schema_twenty_four._agrees`),
and the same army under a commanding sight is held to `tests/sight_oracle.py`
(`test_command_walks.walk`'s method). Each walk also counts the moves of the kind it is for,
so a walk that never met a hop through a portal fails rather than passing vacuously.
"""

from __future__ import annotations

import random
from functools import cache
from typing import Any

import pytest
from fold_oracle import FoldOracle
from sight_oracle import SightOracle, normalise
from test_schema_nineteen import _core_portal_moves, _core_transits, _position, _unmoved
from test_schema_twenty_four import _agrees, _core_fold_moves, _folds, _last_fold

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import WILD
from chess_variant_sandbox.engine.moves import MOVE_FOREIGN, decode_move
from chess_variant_sandbox.spec import validate_spec

SLIDE_STEP = {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"}


def _hopper_army() -> list[dict[str, Any]]:
    """Pawns with first-move leaps and slides, a cannon, a grasshopper and a king."""
    return [
        {"id": "P", "name": "pawn", "value": 1, "moves": [
            SLIDE_STEP,
            {"kind": "slide", "vectors": [[0, 1]], "range": 2, "mode": "move",
             "condition": {"first_move": True}},
            {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
            {"kind": "leap", "vectors": [[1, 2]], "symmetry": "lateral", "mode": "move",
             "condition": {"first_move": True}},
        ]},
        {"id": "C", "name": "cannon", "value": 4, "moves": [
            {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "move"},
            {"kind": "hop", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "capture"},
        ]},
        {"id": "G", "name": "grasshopper", "value": 3, "moves": [
            {"kind": "hop", "vectors": [[1, 1]], "symmetry": "rotate4", "mode": "both",
             "screen_range": 1},
            {"kind": "hop", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both",
             "screen": "enemy", "condition": {"first_move": True}},
            {"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both"},
        ]},
        {"id": "K", "name": "king", "value": 0, "royal": True, "moves": [
            {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"},
        ]},
    ]


def _hopper_game(name: str, **rules: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "schema_version": 31,
        "name": name,
        "description": "Test spec for F10b (Q752b): hoppers beside a rarer generator.",
        "board": {"width": 6, "height": 6},
        "pieces": _hopper_army(),
        "setup": {"mirror": True, "placements": [
            {"piece": "P", "squares": ["a2", "b2", "c2", "d2", "e2", "f2"]},
            {"piece": "C", "squares": ["a1", "f1"]},
            {"piece": "G", "squares": ["b1", "e1"]},
            {"piece": "K", "squares": ["d1"]},
        ]},
        "rules": {
            "win": "checkmate",
            "stalemate": "draw",
            "promotion": [
                {"piece": "P", "rank": "last", "to": ["C", "G"], "mandatory": False},
                {"piece": "G", "ranks": [4, 5], "to": ["C"], "mandatory": False,
                 "on_leaving": True},
            ],
            "max_plies": 200,
            **rules,
        },
    }
    return spec


# A portal group's moves: a step, a two-square slide, a diagonal hop over one piece onto the
# empty squares up to two past it, and a straight one onto any past it (a portal only ever
# moves onto an empty square).
GROUP_MOVES = [
    {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"},
    {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "range": 2},
    {"kind": "hop", "vectors": [[1, 1]], "symmetry": "rotate4", "screen_range": 2},
    {"kind": "hop", "vectors": [[1, 0]], "symmetry": "rotate4"},
]
BOTH = {"orange": [0, 1], "blue": [0, 1]}
# A momentum end's moves while it has not moved (a fold has no first move): a two-square
# leap, a three-square slide and a hop on its file or rank.
FIRST_MOVES = [
    {"kind": "leap", "vectors": [[2, 0]], "symmetry": "rotate4",
     "condition": {"first_move": True}},
    {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "range": 3,
     "condition": {"first_move": True}},
    {"kind": "hop", "vectors": [[1, 0]], "symmetry": "rotate4",
     "condition": {"first_move": True}},
]


def portals_and_folds() -> dict[str, Any]:
    """The hopper army with a momentum pair on b3 and e4 and a fold pair on c4 and d3."""
    return _hopper_game(
        "hoppers beside portals and folds (F10b test spec)",
        portals=[
            {"orange": "b3", "blue": "e4", "moved_by": BOTH,
             "moves": [*GROUP_MOVES, *FIRST_MOVES]},
            {"orange": "c4", "blue": "d3", "kind": "fold", "moved_by": BOTH,
             "moves": GROUP_MOVES},
        ],
    )


def commanded_hoppers() -> dict[str, Any]:
    """The hopper army where the grasshopper commands the enemy pieces two squares along
    its lines, turncoat captures included, so cannons, grasshoppers and moved pawns are
    moved by the other side."""
    spec = _hopper_game("hoppers under command (F10b test spec)")
    for piece in spec["pieces"]:
        if piece["id"] == "G":
            piece["sight"] = [{
                "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 2,
                "effect": "command", "whose": "enemy", "captures_own": True,
            }]
    return spec


@pytest.mark.parametrize("make", [portals_and_folds, commanded_hoppers])
def test_the_specs_validate(make: Any) -> None:
    assert validate_spec(make()) == []


def _diagonal_jump(origin: int, target: int, width: int = 6) -> bool:
    """A move two or more squares along a diagonal: only a group's hop makes one."""
    dx = target % width - origin % width
    dy = target // width - origin // width
    return abs(dx) == abs(dy) >= 2


@pytest.mark.parametrize("seed", range(3))
def test_hoppers_beside_portals_and_folds_agree_with_the_oracle(seed: int) -> None:
    """Every ply of seeded walks: transits, a momentum end's moves (a hop, and a first-move
    leap once the end has moved), a fold's moves (a hop), the ordinary moves through a fold
    and the legal list. Each kind is counted, so a walk that never met one fails."""
    spec = portals_and_folds()
    engine = Engine(spec)
    oracle = FoldOracle(spec)
    seen = dict.fromkeys(
        ("hopper transit", "end hop", "moved end", "fold hop", "hop through fold"), 0,
    )
    rng = random.Random(seed)
    state = engine.initial_state()
    last = None
    for _ in range(20):
        if engine.terminal(state):
            break
        _agrees(engine, oracle, state, last)
        position = _position(engine, state)
        unmoved = _unmoved(state)
        seen["hopper transit"] += sum(
            position[origin][0] in ("C", "G") for origin, _, _, _ in _core_transits(engine, state)
        )
        portal_moves = _core_portal_moves(engine, state)
        seen["end hop"] += sum(_diagonal_jump(o, t) for o, t in portal_moves)
        seen["moved end"] += any(o not in unmoved for o, _ in portal_moves)
        seen["fold hop"] += sum(
            _diagonal_jump(o, t) for _, o, t in _core_fold_moves(engine, state)
        )
        seen["hop through fold"] += sum(
            name is not None and position[o][0] in ("C", "G")
            for o, _, _, name in oracle.ordinary_moves(
                position, _folds(engine, state), state.side_to_move, unmoved,
            )
        )
        move = rng.choice(engine.legal_moves(state))
        last = _last_fold(engine, state, move)
        state = engine.apply(state, move)
        assert state.key == engine._compute_hash(state)
    assert all(seen.values()), seen


def test_a_cannon_hops_into_a_portal_and_takes_where_it_comes_out() -> None:
    """The cannon on b1 hops its own pawn on b2 into the end on b3 (the first piece past the
    screen) and comes out of e4 going up, onto e5, taking the black pawn there: a capture
    only its hop can make, since its slide only moves."""
    spec = portals_and_folds()
    engine = Engine(spec)
    state = engine.make_state({
        "b1": ("C", 0), "b2": ("P", 0), "b3": ("1o", WILD), "e4": ("1b", WILD),
        "e5": ("P", 1), "d1": ("K", 0), "a6": ("K", 1),
    })
    _agrees(engine, FoldOracle(spec), state)
    b1, b3, e5 = (engine._square_index(name) for name in ("b1", "b3", "e5"))
    assert (b1, b3, e5, None) in _core_transits(engine, state)


@cache
def _command_oracle() -> SightOracle:
    return SightOracle(commanded_hoppers())


@pytest.mark.parametrize("seed", range(4))
def test_commanded_hoppers_agree_with_the_oracle(seed: int) -> None:
    """`test_command_walks.walk`'s method on the hopper army: the core's legal list, commanded
    moves sorted, equals the oracle's at every ply, and the walk plays a commanded move one
    ply in two where one is offered. Counted: a commanded grasshopper's hop (two squares on a
    diagonal, which only its hop makes) and a commanded piece that has moved (its first-move
    lines barred)."""
    judge = _command_oracle()
    engine = judge.engine
    spec = engine.spec
    seen = dict.fromkeys(("hop", "moved"), 0)
    rng = random.Random(seed)
    state = engine.initial_state()
    for _ in range(60):
        if engine.terminal(state) is not None:
            break
        moves = engine.legal_moves(state)
        assert normalise(moves) == judge.legal_moves(state), (seed, state.ply)
        if not moves:
            break
        commanded = [m for m in moves if m & MOVE_FOREIGN]
        unmoved = _unmoved(state)
        for move in commanded:
            origin, target, _, _ = decode_move(move)
            pid = spec.piece_ids[spec.type_of(state.board[origin])]
            seen["hop"] += pid == "G" and _diagonal_jump(origin, target)
            seen["moved"] += origin not in unmoved
        pick = commanded if commanded and rng.random() < 1 / 2 else moves
        state = engine.apply(state, rng.choice(pick))
        assert state.key == engine._compute_hash(state)
    assert all(seen.values()), seen


def test_a_commanded_grasshopper_in_its_zone_hops_and_arrives_unpromoted() -> None:
    """Black's grasshopper on c2 stands in its own zone of the promotion that fires on
    leaving (black's ranks 4 and 5 are the board's first two), two squares down White's
    grasshopper's c-file line, so White commands it: its hop over the black pawn on b3 to a4
    leaves the zone, and a commanded piece never promotes (optional here, so it arrives as
    it is). The list is the oracle's, and the commanded hop is in it, unpromoted."""
    judge = _command_oracle()
    engine = judge.engine
    state = engine.make_state(
        {"f1": ("K", 0), "c4": ("G", 0), "f6": ("K", 1), "c2": ("G", 1), "b3": ("P", 1)},
        side_to_move=0, moved=("c2", "b3"),
    )
    moves = engine.legal_moves(state)
    assert normalise(moves) == judge.legal_moves(state)
    c2, a4 = engine._square_index("c2"), engine._square_index("a4")
    hops = [m for m in moves if m & MOVE_FOREIGN and decode_move(m)[:2] == (c2, a4)]
    assert len(hops) == 1 and decode_move(hops[0])[2] == 0
