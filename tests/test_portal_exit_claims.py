"""SR5i (`docs/plan-schema-next.md` §2.10 rows P6, P7): a portal's exit beside territory and a
region.

The rulings (2026-10-28): a transit paints the exit square only, never a portal end's (P6); an
exit outside the type's region is a blocked exit, no move, as a wall is (P7).

The probe (both pairs fed past the validator and the bridge's hold on SR5h's build
`619bf98c974c66a0`, a 6x6 board, a momentum pair and a fold pair on b3/e4, a rook on b1): a
transit already claimed its exit and a go-through the fold's far square, since `make`'s
`write_territory` reads the move's target, which both words name; a fold's own move claimed
nothing (`make_fold` claims nothing). Two things were wrong. A momentum end a player moved
claimed the square it stepped to for the mover (an end is a nobody's byte, which
`arrival_owner` reads as the mover's). And an exit outside the rook's region was offered,
through a momentum end and through a fold, since the exit table and the far square are
per-square, not per type; and the attack tests saw through it too. So `write_territory` skips a
portal end's own move, and `PortalTables::stand` (`compile.PortalRule.stand`, each type's
region per player) blocks an exit, a go-through and an attack outside it.

Readings the rows left open, settled on the plan's principles: a go-through's far square is its
exit (P3's reading), so it is claimed although a fold stands on it, the piece standing there;
the near end never. An entry outside the region was already no move (a step onto a square
outside the region is no landing, and an end is entered by a landing), and stays so: the region
binds every square the piece would stand on or step onto, the exit included.

Hand-built 6x6 specs, not chess. The walks check the incremental key against `_compute_hash`
each ply, and `perft` against an `apply` count puts every colour back.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine, bridge
from chess_variant_sandbox.engine.compile import WILD, compile_spec
from chess_variant_sandbox.engine.moves import MOVE_FOLD, MOVE_PORTAL, move_from, move_to, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from tests.feature_inventory import _schema_eighty_eight_uses

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
KINDS = ("momentum", "fold")
EVERY = [f"{file}{rank}" for rank in range(1, 7) for file in "abcdef"]


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _name(square: int) -> str:
    return "abcdef"[square % WIDTH] + str(square // WIDTH + 1)


def _spec(
    kind: str, *, territory: str | None = None, region: list[str] | None = None,
    moved: bool = False, win: str = "royal_capture",
) -> dict[str, Any]:
    pair: dict[str, Any] = {"orange": "b3", "blue": "e4"}
    if kind == "fold":
        pair["kind"] = "fold"
    if moved:
        pair["moves"] = [copy.deepcopy(KING)]
        pair["moved_by"] = {"orange": [0, 1], "blue": [0, 1]}
    rook: dict[str, Any] = {"id": "R", "value": 5, "moves": [copy.deepcopy(ROOK)]}
    if region is not None:
        rook["region"] = list(region)
    rules: dict[str, Any] = {"win": win, "stalemate": "draw", "max_plies": 80, "portals": [pair]}
    if territory:
        rules["territory"] = {"mode": territory}
    return {
        "name": "the claimed door (SR5i test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [copy.deepcopy(KING)]},
            rook,
            {"id": "S", "name": "stone", "value": 2, "moves": [copy.deepcopy(KING)]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "R", "squares": ["b1"]},
        ]},
        "rules": rules,
    }


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


def _origin(side: int) -> str:
    return "b1" if side == 0 else "b6"


def _exit(kind: str, side: int) -> str:
    """Where a rook coming in at b3 comes out: past e4 the way it travelled through the
    momentum pair (up for the first player, down for the second), on e4 through the fold."""
    if kind == "fold":
        return "e4"
    return "e5" if side == 0 else "e3"


def _reflected(name: str) -> str:
    """A square written for the first player, as the second player's region reads it."""
    return name[0] + str(WIDTH + 1 - int(name[1:]))


def _without(square: str, side: int) -> list[str]:
    """Every square but one, written for the first player so that ``side`` reads it without
    ``square``."""
    written = square if side == 0 else _reflected(square)
    return [name for name in EVERY if name != written]


def _ends(kind: str) -> dict[str, tuple[str, int]]:
    return {"b3": ("1o", WILD), "e4": ("1b", WILD)} if kind == "momentum" else {}


def _position(engine: Engine, kind: str, side: int, **extra: tuple[str, int]) -> GameState:
    placement = {"a1": ("K", 0), "f6": ("K", 1), _origin(side): ("R", side), **_ends(kind)}
    placement.update(extra)
    return engine.make_state(placement, side_to_move=side, moved=list(placement))


def _from(engine: Engine, state: GameState, origin: str) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for move in engine.legal_moves(state):
        if move_from(move) == _sq(origin):
            found.setdefault(_name(move_to(move)), []).append(move)
    return found


def _claimed(state: GameState) -> dict[str, int]:
    return {_name(square): colour for square, colour in enumerate(state.colour) if colour}


# --------------------------------------------------------------------------- P6: the claim


class TestATransitClaimsItsExit:
    @pytest.mark.parametrize("mode", ["permanent", "repaint"])
    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("kind", KINDS)
    def test_the_exit_and_nothing_else(self, kind: str, side: int, mode: str) -> None:
        engine = _engine(_spec(kind, territory=mode))
        state = _position(engine, kind, side)
        through = _from(engine, state, _origin(side))[_exit(kind, side)]
        if kind == "momentum":
            (move,) = through
            assert move_type(move) == MOVE_PORTAL
        else:
            # The fold's far square is reached only by going through it.
            (move,) = through
        after = engine.apply(state, move)
        assert _claimed(after) == {_exit(kind, side): side + 1}
        assert after.key == engine._compute_hash(after)

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_capture_through_the_door_claims_the_exit(self, side: int) -> None:
        engine = _engine(_spec("momentum", territory="repaint"))
        state = _position(engine, "momentum", side, **{_exit("momentum", side): ("S", 1 - side)})
        (move,) = _from(engine, state, _origin(side))[_exit("momentum", side)]
        after = engine.apply(state, move)
        assert _claimed(after) == {_exit("momentum", side): side + 1}

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_fold_stay_claims_the_square_it_ends_on(self, side: int) -> None:
        """Staying on an empty fold is a move that ends there: the square is claimed, the far
        end's is not."""
        engine = _engine(_spec("fold", territory="permanent"))
        state = _position(engine, "fold", side)
        stays = [m for m in _from(engine, state, _origin(side)).get("b3", [])]
        assert len(stays) == 1
        after = engine.apply(state, stays[0])
        assert _claimed(after) == {"b3": side + 1}

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_transit_claim_counts_toward_the_win(self, side: int) -> None:
        spec = _spec("momentum", territory="permanent")
        spec["rules"]["territory"]["win_at"] = 1
        engine = _engine(spec)
        state = _position(engine, "momentum", side)
        (move,) = _from(engine, state, _origin(side))[_exit("momentum", side)]
        outcome = engine.terminal(engine.apply(state, move))
        assert outcome is not None and outcome.winner == side

    @pytest.mark.parametrize("mode", ["permanent", "repaint"])
    @pytest.mark.parametrize("kind", KINDS)
    def test_moving_a_portal_claims_nothing(self, kind: str, mode: str) -> None:
        engine = _engine(_spec(kind, territory=mode, moved=True))
        state = _position(engine, kind, 0)
        own = [
            m for m in engine.legal_moves(state)
            if move_from(m) == _sq("b3") and (kind == "momentum" or move_type(m) == MOVE_FOLD)
        ]
        assert own
        for move in own:
            after = engine.apply(state, move)
            assert _claimed(after) == {}, engine.describe_move(state, move)
            assert after.key == engine._compute_hash(after)


# --------------------------------------------------------------------------- P7: the region


class TestAnExitOutsideTheRegion:
    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("kind", KINDS)
    def test_is_no_move(self, kind: str, side: int) -> None:
        engine = _engine(_spec(kind, region=_without(_exit(kind, side), side)))
        state = _position(engine, kind, side)
        found = _from(engine, state, _origin(side))
        assert _exit(kind, side) not in found
        if kind == "fold":
            # The stay on the near fold is a landing inside the region, and stands.
            assert "b3" in found

    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("kind", KINDS)
    def test_inside_it_the_door_is_open(self, kind: str, side: int) -> None:
        engine = _engine(_spec(kind, region=_without("a4", side)))
        state = _position(engine, kind, side)
        assert len(_from(engine, state, _origin(side))[_exit(kind, side)]) == 1

    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("kind", KINDS)
    def test_an_entry_outside_it_is_no_move(self, kind: str, side: int) -> None:
        """The reading the rows left open: an end is entered by a landing on its square, and a
        square outside the region is no landing."""
        engine = _engine(_spec(kind, region=_without("b3", side)))
        state = _position(engine, kind, side)
        found = _from(engine, state, _origin(side))
        assert _exit(kind, side) not in found and "b3" not in found

    @pytest.mark.parametrize("kind", KINDS)
    def test_a_piece_without_a_region_still_transits(self, kind: str) -> None:
        """The stone has no region; only the rook is kept to one."""
        engine = _engine(_spec(kind, region=_without(_exit(kind, 0), 0)))
        state = _position(engine, kind, 0, b2=("S", 0))
        far = "e5" if kind == "momentum" else "e4"
        assert far in _from(engine, state, "b2")

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("excluded", [True, False])
    def test_no_check_through_a_blocked_exit(self, kind: str, excluded: bool) -> None:
        """The second player's king on the exit: the confined rook checks it through the door
        only where the exit is in its region."""
        region = _without(_exit(kind, 0), 0) if excluded else _without("a4", 0)
        engine = _engine(_spec(kind, region=region, win="checkmate"))
        state = engine.make_state(
            {"a1": ("K", 0), "b1": ("R", 0), _exit(kind, 0): ("K", 1), "f6": ("S", 1),
             **_ends(kind)},
            side_to_move=1, moved=["a1", "b1", _exit(kind, 0), "f6"],
        )
        assert engine.in_check(state, 1) is (not excluded)


# --------------------------------------------------------------------------- the lift


class TestTheLift:
    @pytest.mark.parametrize("kind", KINDS)
    def test_both_pairs_are_valid(self, kind: str) -> None:
        spec = _spec(kind, territory="repaint", region=_without("e5", 0))
        errors = validate_spec(spec)
        assert errors == []
        assert not {"P6", "P7"} & set(refused_ids(errors))

    def test_the_bridge_holds_nothing(self) -> None:
        assert all(ids != ("P6",) for ids, _ in bridge.PORTAL_CORE_HOLDS)
        compiled = compile_spec(_spec("momentum", territory="repaint"))
        assert bridge.rust_supports(compiled) is None

    def test_the_stand_table_is_empty_without_a_region(self) -> None:
        compiled = compile_spec(_spec("momentum", territory="repaint"))
        assert compiled.portals is not None and compiled.portals.stand == ()
        assert bridge.build_tables(compiled)["portal_stand"] == []

    def test_the_stand_table_reflects_the_region(self) -> None:
        compiled = compile_spec(_spec("momentum", region=_without("e5", 0)))
        portals = compiled.portals
        assert portals is not None
        rook = compiled.piece_ids.index("R")
        assert not portals.may_stand(0, rook, _sq("e5"))
        assert portals.may_stand(1, rook, _sq("e5"))
        assert not portals.may_stand(1, rook, _sq("e2"))
        assert portals.may_stand(0, compiled.piece_ids.index("S"), _sq("e5"))

    def test_the_inventory_names_them(self) -> None:
        both = _spec("momentum", territory="repaint", region=_without("e5", 0))
        assert _schema_eighty_eight_uses(both) == {"rules.portals.territory", "rules.portals.region"}
        assert _schema_eighty_eight_uses(_spec("fold", territory="permanent")) == {
            "rules.portals.territory"
        }
        del both["rules"]["portals"]
        assert _schema_eighty_eight_uses(both) == set()

    @pytest.mark.parametrize("kind", KINDS)
    def test_describe_says_it(self, kind: str) -> None:
        text = describe_spec(_spec(kind, territory="repaint", region=_without("e5", 0)))
        assert "claims the square it comes out on, never the portal's own square" in text
        assert "only on a square of its region" in text
        assert ("or a fold" in text) == (kind == "fold")
        plain = describe_spec(_spec(kind))
        assert "claims the square it comes out on" not in plain
        assert "only on a square of its region" not in plain

    def test_fairy_stockfish_refuses_it(self) -> None:
        with pytest.raises(Untranslatable):
            translate(_spec("momentum", territory="repaint"))


# --------------------------------------------------------------------------- the walks


def _busy(engine: Engine, kind: str) -> GameState:
    placement = {
        "a1": ("K", 0), "a6": ("K", 1), "b1": ("R", 0), "b6": ("R", 1),
        "d2": ("S", 0), "d5": ("S", 1), "e5": ("S", 1), "e2": ("S", 0),
    }
    return engine.make_state({**placement, **_ends(kind)}, moved=list(placement))


def _walk_spec(kind: str) -> dict[str, Any]:
    region = [name for name in EVERY if name not in ("e5", "c4")]
    return _spec(kind, territory="repaint", region=region, moved=True)


class TestTheWalks:
    @pytest.mark.parametrize("kind", KINDS)
    def test_walks_keep_the_key_and_the_region(self, kind: str) -> None:
        """Seeded walks with doors that move: the key is the full scan's each ply, no portal's
        square is ever claimed by its own move, and no rook ever stands outside its region."""
        spec = _walk_spec(kind)
        engine = _engine(spec)
        compiled = engine.spec
        rook = compiled.piece_ids.index("R")
        through = 0
        for seed in range(8):
            rng = random.Random(seed)
            state = _busy(engine, kind)
            for _ in range(60):
                moves = engine.legal_moves(state)
                if not moves or engine.terminal(state, moves) is not None:
                    break
                doors = [m for m in moves if move_type(m) == MOVE_PORTAL
                         or engine.fold_route(state, m) is not None]
                move = rng.choice(doors if doors and rng.random() < 0.7 else moves)
                through += move in doors
                after = engine.apply(state, move)
                assert after.key == engine._compute_hash(after), (kind, seed)
                for square, value in enumerate(after.board):
                    if value and compiled.type_of(value) == rook:
                        owner = compiled.owner_of(value)
                        assert compiled.portals is not None
                        assert compiled.portals.may_stand(owner, rook, square), (seed, square)
                state = after
        assert through > 0, kind

    @pytest.mark.parametrize("kind", KINDS)
    def test_perft_counts_what_apply_counts(self, kind: str) -> None:
        engine = _engine(_walk_spec(kind))
        state = _busy(engine, kind)

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(
                count(engine.apply(position, m), depth - 1) for m in engine.legal_moves(position)
            )

        assert engine.perft(state, 3) == count(state, 3)
