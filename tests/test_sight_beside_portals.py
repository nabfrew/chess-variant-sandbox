"""SR5c (`docs/plan-schema-next.md` §2.7 row V1): a line of sight beside portals.

The ruling (2026-10-27, the user's words): *fold-type portal propagates line of sight. Momentum
does not.* A momentum end stops a line, ray or aura, as a wall does: it sees nothing at or past
the end. A fold is floor to a line, and a line that reaches an empty fold that lets the sight's
piece in also sees the square of the end that fold leads to and the piece there, as a move along
the line could land there; one fold per line, and the line does not continue from the exit. An
attack through either kind reads the effects as any attack does: a frozen or disarmed attacker
attacks nothing through it, and a shielded target is attacked by nothing.

The probe (fed past the validator, with the core's boundary refusal gone and nothing else, build
`09d21f5c02cc09ad` on SR5b's `65261ae93ab8942b`): a ray already stopped at a momentum end (an
end is a board byte), but an aura saw past it; a ray reaching an empty fold did not see the far
end's square; in a portal game the attack test read no effect at all, so a frozen rook gave
check through a momentum portal and through a fold, a shielded king was checked through one, and
(the fold copy of movegen ran no sight filter) a frozen rook's moves were offered.

Readings the row left open, settled on the plan's principles and written in the record:

* **An aura and a momentum end.** The row says a line, ray or aura; an aura sees past pieces but
  not past a wall, so beside momentum portals an aura's path is its squares before the seen one,
  each marked `SIGHT_PATH_AURA`, closed only by a portal end.
* **What lets the sight's piece in** is the near end's `enter` and `carries` for the seeing
  piece's owner and type (`admits`), as for the piece's own move. A wild piece or the duck goes
  through no fold, so its sight goes through none.
* **A piece never sees itself** through a fold, as a wrapped line never does.
* **A fold's own move** moves no piece and takes nothing, so no effect bars it (a fold may move
  under a frozen piece standing in it).
* **A command beside portals stayed refused** here (the narrowed V1); SR5c2 lifted it: a
  commanded piece passes a portal as its owner's would (`tests/test_command_beside_portals.py`).

The oracle is `tests/sight_oracle.py`, widened to portals (CLAUDE.md rule 3: a declared test
oracle). Hand-built 6x6 specs, not chess.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import (
    SIGHT_DISARM,
    SIGHT_FREEZE,
    SIGHT_PATH_AURA,
    SIGHT_SHIELD,
    WILD,
    compile_spec,
)
from chess_variant_sandbox.engine.moves import MOVE_FOLD, MOVE_PORTAL, decode_move
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from tests.feature_inventory import _schema_eighty_uses
from tests.sight_oracle import SightOracle

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
BISHOP = {"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4"}
STEP = {"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}
LINE = {"vectors": [[1, 0]], "symmetry": "rotate4"}
AROUND = {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 1}
MOMENTUM = {"orange": "c3", "blue": "e5"}
FOLD = {"orange": "c3", "blue": "e5", "kind": "fold"}
BOTH = {"orange": [0, 1], "blue": [0, 1]}


def _spec(*groups: dict[str, Any]) -> dict[str, Any]:
    """Kings, rooks and bishops; a freezer `F` (a rook-line ray on the enemy), an aura `A` (a
    rook-line aura that disarms the enemy) and a warden `W` (shields its own side a king's step
    away). ``groups`` are the portals."""
    return copy.deepcopy({
        "name": "the sight at the portals (SR5c test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "B", "value": 3, "moves": [BISHOP]},
            {"id": "F", "value": 2, "moves": [KING],
             "sight": [{**LINE, "effect": "freeze", "whose": "enemy"}]},
            {"id": "A", "value": 2, "moves": [KING],
             "sight": [{**LINE, "effect": "disarm", "whose": "enemy", "through": True}]},
            {"id": "W", "value": 2, "moves": [KING],
             "sight": [{**AROUND, "effect": "shield", "whose": "own"}]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["d1"]}, {"piece": "R", "squares": ["a1"]},
            {"piece": "B", "squares": ["f1"]}, {"piece": "F", "squares": ["b1"]},
            {"piece": "A", "squares": ["e1"]}, {"piece": "W", "squares": ["c1"]},
        ]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 80,
                  **({"portals": list(groups)} if groups else {})},
    })


def _moving(group: dict[str, Any]) -> dict[str, Any]:
    return {**group, "moves": [STEP], "moved_by": BOTH}


def _engine(*groups: dict[str, Any]) -> Engine:
    spec = _spec(*groups)
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


def _sq(engine: Engine, name: str) -> int:
    return engine._square_index(name)


def _position(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0) -> Any:
    """``placement`` with the kings out of the way (a1 and f6 unless placed) and, in a momentum
    game, its ends where they start."""
    full = dict(placement)
    portals = engine.spec.portals
    if portals is not None and portals.folds is None:
        full.update({"c3": ("1o", WILD), "e5": ("1b", WILD)})
    if not any(piece == "K" and player == 0 for piece, player in placement.values()):
        full["a1"] = ("K", 0)
    if not any(piece == "K" and player == 1 for piece, player in placement.values()):
        full["f6"] = ("K", 1)
    return engine.make_state(full, side_to_move=side)


def _bits(engine: Engine, state: Any, name: str) -> int:
    return engine.effects(state).bits[_sq(engine, name)]


# ------------------------------------------------------------------------- the validator


class TestTheValidator:
    def test_a_sight_beside_either_kind_validates(self) -> None:
        for groups in ((MOMENTUM,), (FOLD,), (_moving(MOMENTUM),), (_moving(FOLD),)):
            assert validate_spec(_spec(*groups)) == []

    def test_a_command_beside_portals_is_lifted_too(self) -> None:
        """SR5c2: the command arm went (`tests/test_command_beside_portals.py` plays it)."""
        spec = _spec(FOLD)
        spec["pieces"][3]["sight"][0]["effect"] = "command"
        assert "V1" not in refused_ids(validate_spec(spec))

    def test_the_inventory_names_it(self) -> None:
        assert _schema_eighty_uses(_spec(FOLD)) == {"pieces.sight.portals"}
        assert _schema_eighty_uses(_spec()) == set()

    def test_describe_says_both_readings(self) -> None:
        text = describe_spec(_spec(MOMENTUM, {"orange": "a4", "blue": "f4",
                                                 "kind": "fold"}))
        assert "stops a line of sight" in text
        assert "the square of the end it leads to" in text
        assert "attacks nothing through one" in text

    def test_fairy_stockfish_refuses_it(self) -> None:
        with pytest.raises(Untranslatable):
            translate(_spec(FOLD))


# ------------------------------------------------------------------------- the compiler


class TestTheTables:
    def test_an_aura_beside_momentum_ends_carries_its_squares_marked(self) -> None:
        """An aura's path beside momentum portals is its squares before the seen one, marked;
        a ray's is unmarked; beside folds alone an aura's path stays empty."""
        def paths(*groups: dict[str, Any]) -> set[tuple[int, ...]]:
            sight = compile_spec(_spec(*groups)).sight
            assert sight is not None
            return {
                source[3] for side in sight.sources for row in side for found in row
                for source in found if source[3]
            }
        momentum = paths(MOMENTUM)
        assert any(all(s & SIGHT_PATH_AURA for s in path) for path in momentum)
        assert any(not any(s & SIGHT_PATH_AURA for s in path) for path in momentum)
        assert not any(s & SIGHT_PATH_AURA for path in paths(FOLD) for s in path)
        assert not any(s & SIGHT_PATH_AURA for path in paths() for s in path)


# ------------------------------------------------------------------------- the readings


class TestAMomentumEndStopsALine:
    def test_a_ray_stops_at_the_end(self) -> None:
        engine = _engine(MOMENTUM)
        state = _position(engine, {"a3": ("F", 0), "d3": ("R", 1)})
        assert not _bits(engine, state, "d3") & SIGHT_FREEZE
        before = _position(engine, {"a3": ("F", 0), "b3": ("R", 1)})
        assert _bits(engine, before, "b3") & SIGHT_FREEZE

    def test_an_aura_stops_at_the_end(self) -> None:
        engine = _engine(MOMENTUM)
        state = _position(engine, {"a3": ("A", 0), "b3": ("B", 0), "d3": ("R", 1)})
        assert not _bits(engine, state, "d3") & SIGHT_DISARM
        # Past a piece it still sees: the end is the wall, not the bishop.
        past = _position(engine, {"a4": ("A", 0), "b4": ("B", 0), "d4": ("R", 1)})
        assert _bits(engine, past, "d4") & SIGHT_DISARM

    def test_an_aura_sees_past_a_moved_end(self) -> None:
        """The end is a wall where it stands, not where it started."""
        engine = _engine(_moving(MOMENTUM))
        state = engine.make_state({
            "a1": ("K", 0), "f6": ("K", 1), "a3": ("A", 0), "d3": ("R", 1),
            "c2": ("1o", WILD), "e5": ("1b", WILD),
        })
        assert _bits(engine, state, "d3") & SIGHT_DISARM

    def test_a_frozen_attacker_gives_no_check_through_it(self) -> None:
        """The rook on c6 goes down into c3 and comes out of e5 onto e4, the king's square."""
        engine = _engine(MOMENTUM)
        free = _position(engine, {"c6": ("R", 1), "e4": ("K", 0)})
        assert engine.in_check(free, 0)
        frozen = _position(engine, {"a6": ("F", 0), "c6": ("R", 1), "e4": ("K", 0)})
        assert not engine.in_check(frozen, 0)
        c6 = _sq(engine, "c6")
        assert not [
            m for m in engine.legal_moves(frozen.with_(side_to_move=1))
            if decode_move(m)[0] == c6
        ]
        assert [
            m for m in engine.legal_moves(free.with_(side_to_move=1))
            if decode_move(m)[0] == c6 and decode_move(m)[2] == MOVE_PORTAL
        ]

    def test_a_shielded_royal_is_not_checked_through_it(self) -> None:
        engine = _engine(MOMENTUM)
        state = _position(engine, {"c6": ("R", 1), "e4": ("K", 0), "e3": ("W", 0)})
        assert not engine.in_check(state, 0)
        assert engine.effects(state).suppressed


class TestAFoldPropagatesALine:
    def test_a_ray_sees_the_exit_and_nothing_past_it(self) -> None:
        engine = _engine(FOLD)
        state = _position(engine, {"a3": ("F", 0), "e5": ("R", 1), "f5": ("B", 1)})
        assert _bits(engine, state, "e5") & SIGHT_FREEZE
        assert not _bits(engine, state, "f5") & SIGHT_FREEZE
        # The exit's empty: the line sees the square and nothing past it.
        empty = _position(engine, {"a3": ("F", 0), "f5": ("R", 1)})
        assert not _bits(engine, empty, "f5") & SIGHT_FREEZE

    def test_the_fold_is_floor_and_the_line_goes_on(self) -> None:
        engine = _engine(FOLD)
        state = _position(engine, {"a3": ("F", 0), "d3": ("R", 1), "e5": ("B", 1)})
        assert _bits(engine, state, "d3") & SIGHT_FREEZE
        assert _bits(engine, state, "e5") & SIGHT_FREEZE

    def test_an_occupied_fold_leads_nowhere(self) -> None:
        engine = _engine(FOLD)
        state = _position(engine, {"a3": ("F", 0), "c3": ("B", 1), "e5": ("R", 1)})
        assert _bits(engine, state, "c3") & SIGHT_FREEZE
        assert not _bits(engine, state, "e5") & SIGHT_FREEZE

    def test_an_aura_goes_through_too(self) -> None:
        engine = _engine(FOLD)
        state = _position(engine, {"a3": ("A", 0), "b3": ("B", 1), "e5": ("R", 1)})
        assert _bits(engine, state, "e5") & SIGHT_DISARM

    def test_a_fold_that_lets_the_piece_in_and_one_that_does_not(self) -> None:
        engine = _engine({**FOLD, "carries": {"orange": ["K", "R", "B"], "blue": ["K", "R", "B"]}})
        state = _position(engine, {"a3": ("F", 0), "e5": ("R", 1)})
        assert not _bits(engine, state, "e5") & SIGHT_FREEZE

    def test_sight_of_shows_the_exit(self) -> None:
        engine = _engine(FOLD)
        state = _position(engine, {"a3": ("F", 0), "e5": ("R", 1)})
        seen = dict(engine.sight_of(state, _sq(engine, "a3")))
        assert seen.get(_sq(engine, "e5"), 0) & SIGHT_FREEZE

    def test_a_frozen_attacker_gives_no_check_through_it(self) -> None:
        """The rook on c6 goes down into the empty fold on c3 and out of e5, the king's."""
        engine = _engine(FOLD)
        free = _position(engine, {"c6": ("R", 1), "e5": ("K", 0), "f1": ("K", 1)})
        assert engine.in_check(free, 0)
        frozen = _position(engine, {"a6": ("F", 0), "c6": ("R", 1), "e5": ("K", 0),
                                    "f1": ("K", 1)})
        assert not engine.in_check(frozen, 0)

    def test_a_frozen_piece_is_offered_nothing_and_a_fold_still_moves_under_it(self) -> None:
        engine = _engine(_moving(FOLD))
        state = _position(engine, {"a3": ("F", 0), "e5": ("R", 1)}, side=1)
        moves = engine.legal_moves(state)
        e5 = _sq(engine, "e5")
        assert not [m for m in moves if decode_move(m)[0] == e5 and decode_move(m)[2] != MOVE_FOLD]
        assert [m for m in moves if decode_move(m)[2] == MOVE_FOLD and decode_move(m)[0] == e5]

    def test_a_shielded_piece_on_the_exit_is_not_taken_through_it(self) -> None:
        engine = _engine(FOLD)
        state = _position(engine, {"c6": ("R", 1), "e5": ("B", 0), "e4": ("W", 0)}, side=1)
        moves = engine.legal_moves(state)
        assert not [m for m in moves if decode_move(m)[1] == _sq(engine, "e5")]
        assert engine.effects(state).bits[_sq(engine, "e5")] & SIGHT_SHIELD


# ------------------------------------------------------------------------- the oracle


def _games() -> dict[str, dict[str, Any]]:
    return {
        "momentum": _spec(_moving(MOMENTUM)),
        "fold": _spec(_moving(FOLD)),
        "both": _spec(_moving(MOMENTUM), _moving({"orange": "a4", "blue": "f3", "kind": "fold"})),
    }


@pytest.mark.parametrize("game", sorted(_games()))
@pytest.mark.parametrize("seed", range(3))
def test_walks_agree_with_the_oracle(game: str, seed: int) -> None:
    """Every ply of seeded walks: the core's effects, check and legal list equal the oracle's,
    and the incremental key the full scan's. Counted: a board whose effects a fold changed (in a game
    with one) and a check the effects suppressed, so a walk that met neither fails."""
    spec = _games()[game]
    oracle = SightOracle(spec)
    engine = oracle.engine
    rng = random.Random(seed)
    state = engine.initial_state()
    seen = {"through a fold": 0, "suppressed": 0}
    for _ in range(70):
        exits = oracle.fold_exits(state)
        effects = engine.effects(state)
        assert list(effects.bits) == oracle.effects(state.board, exits)
        seen["through a fold"] += list(effects.bits) != oracle.effects(state.board, {})
        seen["suppressed"] += effects.suppressed
        for player in (0, 1):
            assert engine.in_check(state, player) == oracle.royal_attacked(state, player)
        moves = engine.legal_moves(state)
        assert moves == oracle.legal_moves(state)
        assert state.key == engine._compute_hash(state)
        if engine.terminal(state, moves) is not None:
            break
        state = engine.apply(state, rng.choice(moves))
    assert seen["suppressed"], seen
    assert seen["through a fold"] or game == "momentum", seen
