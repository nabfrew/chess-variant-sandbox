"""SR5h (`docs/plan-schema-next.md` §2.10 row P5): a blast at a portal's exit.

The ruling (2026-10-28): the blast centres on the exit square, where the capture happened; a
portal end in the blast stands (a portal is never captured); a fold's square in the blast loses
the piece standing in it, never the fold.

The probe (the pair fed past the validator and the bridge's hold on SR5g's build
`7f77c9b55096ffc8`, a momentum pair and a fold pair on a 6x6 board, a king's-step blast): the
blast already centred on the exit, since `make`'s `write_blast` and the filter's blast arm read
the move's target, which a transit's word and a go-through's word both name; and a fold stands
on no board byte, so the piece in a fold in the blast went and the fold stayed. But the momentum
end beside the exit was cleared: an end is a board byte of its own internal type, and no blast
spared it. So the compiler puts every end's type in the blast's immune set (as it does the
duck's), and `Tables::portal_tables` refuses tables that say otherwise.

Hand-built 6x6 specs, not chess. Every walk checks the incremental key against `_compute_hash`
each ply, and `perft` against an `apply` count puts every cleared piece back.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine, bridge
from chess_variant_sandbox.engine.compile import WILD, compile_spec
from chess_variant_sandbox.engine.moves import MOVE_PORTAL, move_from, move_to, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from tests.feature_inventory import _schema_eighty_six_uses

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
# A pair on b3 and e4, a momentum pair or a fold pair; beside the fold pair a second fold
# pair, f5 and a5, whose f5 end stands in a blast at e4.
DOORS: dict[str, list[dict[str, Any]]] = {
    "momentum": [{"orange": "b3", "blue": "e4"}],
    "fold": [{"orange": "b3", "blue": "e4", "kind": "fold"},
             {"orange": "f5", "blue": "a5", "kind": "fold"}],
}
KINDS = sorted(DOORS)
# A momentum end is a board byte, nobody's; a fold is not.
ENDS: dict[str, dict[str, tuple[str, int]]] = {
    "momentum": {"b3": ("1o", WILD), "e4": ("1b", WILD)}, "fold": {},
}


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _spec(kind: str, **blast: Any) -> dict[str, Any]:
    return copy.deepcopy({
        "name": "the blasted door (SR5h test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "S", "name": "stone", "value": 2, "moves": [KING]},
            {"id": "N", "name": "guard", "value": 3, "moves": [KING]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "R", "squares": ["b1"]},
            {"piece": "S", "squares": ["f2"]},
            {"piece": "N", "squares": ["d2"]},
        ]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 80,
                  "portals": DOORS[kind], "explode_on_capture": {"radius": 1, **blast}},
    })


def _checked(kind: str) -> dict[str, Any]:
    spec = _spec(kind)
    spec["rules"]["win"] = "checkmate"
    return spec


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


def _at(engine: Engine, state: GameState, square: str) -> tuple[str, int] | None:
    value = state.board[_sq(square)]
    if not value:
        return None
    spec = engine.spec
    return spec.piece_ids[spec.type_of(value)], spec.owner_of(value)


def _exit(kind: str, side: int) -> str:
    """Where the rook coming through b3 comes out: past e4 the way it travelled (up for the
    first player, down for the second) through the momentum pair, on e4 itself through the
    fold."""
    if kind == "fold":
        return "e4"
    return "e5" if side == 0 else "e3"


def _origin(side: int) -> str:
    return "b1" if side == 0 else "b6"


def _through(engine: Engine, state: GameState, origin: str, target: str) -> list[int]:
    return [
        m for m in engine.legal_moves(state)
        if (move_from(m), move_to(m)) == (_sq(origin), _sq(target))
    ]


# The squares round each exit: the other player's guard and the mover's own guard, both in
# the blast, by kind and seat.
_AROUND = {
    ("momentum", 0): ("d6", "f4"), ("momentum", 1): ("d2", "f4"),
    ("fold", 0): ("d5", "f3"), ("fold", 1): ("d5", "f3"),
}


def _position(engine: Engine, kind: str, side: int, **extra: tuple[str, int]) -> GameState:
    """The rook of ``side`` on its b-file corner, a stone of the other side on the exit, an
    enemy and an own guard beside the exit, an enemy guard on c3 beside the entry (b3); for the
    fold, the other player's guard standing in the second pair's fold on f5, beside e4."""
    foe = 1 - side
    enemy, own = _AROUND[(kind, side)]
    placement = {
        "a1": ("K", 0), "a6": ("K", 1), _origin(side): ("R", side),
        _exit(kind, side): ("S", foe), enemy: ("N", foe), own: ("N", side), "c3": ("N", foe),
        **ENDS[kind],
    }
    if kind == "fold":
        placement["f5"] = ("N", foe)
    placement.update(extra)
    return engine.make_state(placement, side_to_move=side, moved=list(placement))


# --------------------------------------------------------------------------- the acceptance


class TestTheBlastCentresOnTheExit:
    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("kind", KINDS)
    def test_a_capture_through_a_door_clears_round_the_exit(self, kind: str, side: int) -> None:
        engine = _engine(_spec(kind))
        state = _position(engine, kind, side)
        origin, exit_square = _origin(side), _exit(kind, side)
        moves = _through(engine, state, origin, exit_square)
        assert len(moves) == 1, [engine.describe_move(state, m) for m in moves]
        if kind == "momentum":
            assert move_type(moves[0]) == MOVE_PORTAL
        after = engine.apply(state, moves[0])
        enemy, own = _AROUND[(kind, side)]
        # The capturer, the stone it took and both guards beside the exit are gone.
        for square in (origin, exit_square, enemy, own):
            assert _at(engine, after, square) is None, (kind, side, square)
        # Nothing round the entry: the guard beside b3 stands.
        assert _at(engine, after, "c3") == ("N", 1 - side)
        assert after.key == engine._compute_hash(after)
        # Undo exact: the position played from is the position it was.
        assert engine.apply(state, moves[0]).board == after.board

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_momentum_end_in_the_blast_stands(self, side: int) -> None:
        """e4, the end the rook came out of, is beside its exit; b3 is where it went in."""
        engine = _engine(_spec("momentum"))
        state = _position(engine, "momentum", side)
        (move,) = _through(engine, state, _origin(side), _exit("momentum", side))
        after = engine.apply(state, move)
        assert _at(engine, after, "e4") == ("1b", WILD)
        assert _at(engine, after, "b3") == ("1o", WILD)
        # And the doors still work: the other player's rook goes through after the blast.
        assert engine.legal_moves(after)

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_fold_in_the_blast_loses_its_piece_and_stays(self, side: int) -> None:
        engine = _engine(_spec("fold"))
        state = _position(engine, "fold", side)
        (move,) = _through(engine, state, _origin(side), "e4")
        after = engine.apply(state, move)
        # The guard standing in the f5 fold is gone; every fold is where it was.
        assert _at(engine, after, "f5") is None
        assert after.folds == state.folds
        assert after.key == engine._compute_hash(after)

    @pytest.mark.parametrize("kind", KINDS)
    def test_a_blast_at_the_exit_takes_the_other_royal(self, kind: str) -> None:
        """The second player's king beside the exit (f6 or f4): the capture wins, under
        checkmate as under royal capture."""
        engine = _engine(_checked(kind))
        king = "f6" if kind == "momentum" else "f4"
        state = _position(engine, kind, 0, a6=("N", 0), **{king: ("K", 1)})
        (move,) = _through(engine, state, "b1", _exit(kind, 0))
        after = engine.apply(state, move)
        assert _at(engine, after, king) is None
        outcome = engine.terminal(after)
        assert outcome is not None and outcome.winner == 0

    @pytest.mark.parametrize("kind", KINDS)
    def test_a_blast_at_the_exit_never_takes_the_movers_royal(self, kind: str) -> None:
        """The first player's own king beside the exit: the capture through the door is no move,
        and the same king beside the entry does not bar it. Under checkmate, where a move that
        loses the mover's royal is illegal (under royal capture it is a move that loses)."""
        engine = _engine(_checked(kind))
        king = "f6" if kind == "momentum" else "f4"
        barred = _position(engine, kind, 0, a1=("N", 0), **{king: ("K", 0)})
        assert _through(engine, barred, "b1", _exit(kind, 0)) == []
        beside_entry = _position(engine, kind, 0, a1=("N", 0), a2=("K", 0))
        assert len(_through(engine, beside_entry, "b1", _exit(kind, 0))) == 1


class TestBesideTheStrikeSwitch:
    """SR5a's `explode_on_capture.strikes` read at the exit: a demoting strike placed `stay`
    strikes the piece taken where it stood (the exit) and each victim where it stood (round
    the exit), never round the entry."""

    @staticmethod
    def _struck(kind: str, strikes: str) -> dict[str, Any]:
        spec = _spec(kind, strikes=strikes)
        spec["rules"]["strike"] = {"effect": "demote", "position": "stay", "immune": ["K"],
                                   "chain": [{"from": "S", "to": "N"}]}
        return spec

    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("kind", KINDS)
    def test_target(self, kind: str, side: int) -> None:
        engine = _engine(self._struck(kind, "target"))
        state = _position(engine, kind, side)
        (move,) = _through(engine, state, _origin(side), _exit(kind, side))
        after = engine.apply(state, move)
        enemy, own = _AROUND[(kind, side)]
        assert _at(engine, after, _exit(kind, side)) == ("N", 1 - side)
        assert _at(engine, after, enemy) is None and _at(engine, after, own) is None
        assert _at(engine, after, _origin(side)) is None
        assert after.key == engine._compute_hash(after)

    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("kind", KINDS)
    def test_blast(self, kind: str, side: int) -> None:
        """Stones beside the exit, of both sides, demote to guards where they stood; the stone
        taken goes (it is the target, not a victim); the guard on c3, beside the entry, stands."""
        engine = _engine(self._struck(kind, "blast"))
        enemy, own = _AROUND[(kind, side)]
        stones = {enemy: ("S", 1 - side), own: ("S", side)}
        if kind == "fold":
            stones["f5"] = ("S", 1 - side)
        state = _position(engine, kind, side, **stones)
        (move,) = _through(engine, state, _origin(side), _exit(kind, side))
        after = engine.apply(state, move)
        assert _at(engine, after, _exit(kind, side)) is None
        assert _at(engine, after, enemy) == ("N", 1 - side)
        assert _at(engine, after, own) == ("N", side)
        assert _at(engine, after, "c3") == ("N", 1 - side)
        if kind == "momentum":
            assert _at(engine, after, "e4") == ("1b", WILD)
        else:
            assert _at(engine, after, "f5") == ("N", 1 - side)
            assert after.folds == state.folds
        assert after.key == engine._compute_hash(after)


# --------------------------------------------------------------------------- the validator and the core


class TestTheLift:
    @pytest.mark.parametrize("kind", KINDS)
    def test_the_pair_is_valid(self, kind: str) -> None:
        errors = validate_spec(_spec(kind))
        assert errors == []
        assert "P5" not in refused_ids(errors)

    def test_the_bridge_holds_nothing(self) -> None:
        assert all(ids != ("P5",) for ids, _ in bridge.PORTAL_CORE_HOLDS)

    @pytest.mark.parametrize("kind", KINDS)
    def test_every_end_is_immune(self, kind: str) -> None:
        compiled = compile_spec(_spec(kind))
        assert compiled.explosion is not None
        # The ends are the internal types named by their group's number (`1o`, `1b`, ...).
        ends = {k for k, pid in enumerate(compiled.piece_ids) if pid[:1].isdigit()}
        assert len(ends) == len(DOORS[kind]) * 2
        assert ends <= compiled.explosion.immune_types

    def test_the_core_refuses_an_end_a_blast_clears(self) -> None:
        compiled = compile_spec(_spec("momentum"))
        assert compiled.explosion is not None
        declared = {compiled.piece_ids.index(pid) for pid in ("K", "R", "S", "N")}
        bare = compiled.explosion.__class__(
            **{**compiled.explosion.__dict__,
               "immune_types": frozenset(compiled.explosion.immune_types & declared)}
        )
        broken = compiled.__class__(**{**compiled.__dict__, "explosion": bare})
        with pytest.raises(ValueError, match="clears a portal end"):
            Engine(broken)

    def test_the_inventory_names_it(self) -> None:
        assert _schema_eighty_six_uses(_spec("momentum")) == {"rules.portals.explode_on_capture"}
        assert _schema_eighty_six_uses(_spec("fold")) == {"rules.portals.explode_on_capture"}
        no_blast = _spec("momentum")
        del no_blast["rules"]["explode_on_capture"]
        assert _schema_eighty_six_uses(no_blast) == set()
        no_doors = _spec("momentum")
        del no_doors["rules"]["portals"]
        assert _schema_eighty_six_uses(no_doors) == set()

    @pytest.mark.parametrize("kind", KINDS)
    def test_describe_says_it(self, kind: str) -> None:
        text = describe_spec(_spec(kind))
        assert "explodes on the square the move comes out on" in text
        assert "A portal in the blast stands" in text
        assert ("the fold stays" in text) == (kind == "fold")
        no_blast = _spec(kind)
        del no_blast["rules"]["explode_on_capture"]
        assert "explodes on the square the move comes out on" not in describe_spec(no_blast)

    def test_fairy_stockfish_refuses_it(self) -> None:
        with pytest.raises(Untranslatable):
            translate(_spec("momentum"))


# --------------------------------------------------------------------------- the walks


def _busy(engine: Engine, kind: str) -> GameState:
    """Both rooks on their b-file corners, stones and guards round the doors."""
    placement = {
        "a1": ("K", 0), "a6": ("K", 1), "b1": ("R", 0), "b6": ("R", 1),
        "e4" if kind == "fold" else "e5": ("S", 1), "e3": ("S", 0),
        "d2": ("N", 0), "d5": ("N", 1), "f4": ("N", 1), **ENDS[kind],
    }
    return engine.make_state(placement, moved=[sq for sq in placement if sq not in ENDS[kind]])


class TestTheWalks:
    @pytest.mark.parametrize("kind", KINDS)
    def test_walks_keep_the_key(self, kind: str) -> None:
        """Seeded walks, captures through a door preferred; the key is the full scan's each
        ply, and no momentum end is ever lost."""
        engine = _engine(_spec(kind))
        through = 0
        for seed in range(8):
            rng = random.Random(seed)
            state = _busy(engine, kind)
            for _ in range(60):
                moves = engine.legal_moves(state)
                if not moves or engine.terminal(state, moves) is not None:
                    break
                loud = [m for m in moves if state.board[move_to(m)]]
                doors = [m for m in loud if ">" in engine.describe_move(state, m)
                         or "~" in engine.describe_move(state, m)]
                if doors and rng.random() < 0.9:
                    move = rng.choice(doors)
                    through += 1
                else:
                    move = rng.choice(loud if loud and rng.random() < 0.5 else moves)
                after = engine.apply(state, move)
                assert after.key == engine._compute_hash(after), (kind, seed)
                if kind == "momentum":
                    assert _at(engine, after, "b3") == ("1o", WILD)
                    assert _at(engine, after, "e4") == ("1b", WILD)
                state = after
        assert through > 0, kind

    @pytest.mark.parametrize("kind", KINDS)
    def test_perft_counts_what_apply_counts(self, kind: str) -> None:
        """Make and unmake against `apply`: every cleared piece put back exactly, through the
        door both ways."""
        engine = _engine(_spec(kind))
        state = _busy(engine, kind)

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(
                count(engine.apply(position, m), depth - 1) for m in engine.legal_moves(position)
            )

        assert engine.perft(state, 3) == count(state, 3)
