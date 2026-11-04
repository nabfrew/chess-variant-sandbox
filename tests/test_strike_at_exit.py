"""SR4m (`docs/plan-schema-next.md` §2.10 rows P3 and P4): a strike or a link at a portal's exit.

The rulings (2026-10-28). P3: the exit square is the capture square, and every position reads
from it as anywhere: `stay` (rifle) leaves the mover on the square it moved from, `swap` puts the
struck piece there, `choose` and `home` advance the mover to the exit. P4, read as P3's: the exit
is where the mover meets its target. Through a fold, each the same with the fold's far square as
the exit.

The probe (SR2's method, the pair fed past the validator and the bridge's hold on SR4l's build
`05d05b248eb30a9f`, a momentum pair and a fold pair, six readings: rifle, circe, swap, choose, a
demoting chain, a contact table). `make` already played rifle, circe, swap, the chain and the
link at the exit as ruled -- `strike_writes` reads the move's target, which a transit's word and
a go-through's word both name -- but `choose` panicked: the transit word's tag holds the entered
end's type and the go-through's the fold end plus one, and `make` read either as the landing
index (`rebirth`'s table indexed past its one entry), and neither walk fanned a capture out over
its landings. So the core asks `Game::fans_out` before tagging: a capture that places a survivor
under `choose` carries its landing in the tag and names no end, through a momentum end (one move
per exit and landing, however many ends lead there) or a fold (the direct capture's word, so
`dedup_fold_segment` compares those words whole). Python reads such a word's end as none
(`portal_entry`, `fold_route`), and the text of a transit or a go-through says what the strike did
(`_strike_text`), as a capture's always has.

Hand-built 6x6 specs, not chess. Every walk checks the incremental key against `_compute_hash`
each ply, and `perft` against an `apply` count puts every struck, placed and reborn piece back.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import WILD, compile_spec
from chess_variant_sandbox.engine.moves import (
    MOVE_FOREIGN,
    MOVE_PORTAL,
    move_from,
    move_landing,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from tests.feature_inventory import _schema_seventy_seven_uses

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
JUMP = {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}
DEMOTE = [{"from": "S", "to": "N"}, {"from": "R", "to": "S"}]

RULES: dict[str, tuple[str, dict[str, Any]]] = {
    "rifle": ("strike", {"effect": "capture", "position": "stay", "immune": ["K"]}),
    "circe": ("strike", {"effect": "capture", "position": "home", "immune": ["K"]}),
    "swap": ("strike", {"effect": "demote", "position": "swap", "immune": ["K"], "chain": DEMOTE}),
    "choose": ("strike", {"effect": "demote", "position": "choose", "immune": ["K"],
                          "landing": {"vectors": [[0, 1]]}, "chain": DEMOTE}),
    "demote": ("strike", {"effect": "demote", "position": "stay", "immune": ["K"],
                          "chain": DEMOTE}),
    "andernach": ("strike", {"effect": "capture", "switches_sides": ["R", "J"],
                             "immune": ["K"]}),
    "contacts": ("contacts", {"position": "stay", "immune": ["K"], "links": [
        {"mover": "R", "onto": "S", "onto_to": "N"},
        {"mover": "J", "onto": "S", "onto_to": "N"},
    ]}),
}
READINGS = sorted(RULES)
# A pair on b3 and e4: a momentum pair, or a fold pair.
DOORS = {
    "momentum": {"orange": "b3", "blue": "e4"},
    "fold": {"orange": "b3", "blue": "e4", "kind": "fold"},
}
KINDS = sorted(DOORS)
# A momentum end is a board byte, nobody's; a fold is not (it stands where it starts).
ENDS: dict[str, dict[str, tuple[str, int]]] = {
    "momentum": {"b3": ("1o", WILD), "e4": ("1b", WILD)}, "fold": {},
}


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _base(kind: str) -> dict[str, Any]:
    return copy.deepcopy({
        "name": "the struck door (SR4m test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "J", "name": "jumper", "value": 3, "moves": [JUMP]},
            {"id": "S", "name": "stone", "value": 2, "moves": [KING]},
            {"id": "N", "name": "guard", "value": 3, "moves": [KING]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "R", "squares": ["b1"]},
            {"piece": "J", "squares": ["c1"]},
            {"piece": "S", "squares": ["f2"]},
            {"piece": "N", "squares": ["d2"]},
        ]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 80,
                  "portals": [DOORS[kind]]},
    })


def _spec(reading: str, kind: str) -> dict[str, Any]:
    spec = _base(kind)
    key, value = RULES[reading]
    spec["rules"][key] = copy.deepcopy(value)
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


# --------------------------------------------------------------------------- the acceptance


def _exit(kind: str, side: int) -> str:
    """Where the rook coming through b3 comes out: past e4 the way it travelled (up for the
    first player, down for the second) through the momentum pair, on e4 itself through the
    fold."""
    if kind == "fold":
        return "e4"
    return "e5" if side == 0 else "e3"


def _struck_position(engine: Engine, kind: str, side: int) -> tuple[GameState, str, str]:
    """The rook of ``side`` on its back rank's b-file, a stone of the other side on the exit;
    the stone's home (f2 for the first player, f5 for the second) is empty."""
    origin = "b1" if side == 0 else "b6"
    exit_square = _exit(kind, side)
    state = engine.make_state({
        "a1": ("K", 0), "a6": ("K", 1), origin: ("R", side), exit_square: ("S", 1 - side),
        **ENDS[kind],
    }, side_to_move=side, moved=["a1", "a6", origin, exit_square])
    return state, origin, exit_square


def _through(engine: Engine, state: GameState, origin: str, target: str) -> list[int]:
    return [m for m in engine.legal_moves(state) if (move_from(m), move_to(m)) == (_sq(origin), _sq(target))]


def _expected(reading: str, side: int, origin: str, exit_square: str) -> dict[str, tuple[str, int] | None]:
    foe = 1 - side
    forward = 1 if side == 0 else -1
    placed = exit_square[0] + str(int(exit_square[1:]) + forward)
    home = "f5" if foe == 1 else "f2"
    return {
        # Rifle: the stone at the exit is gone, the rook never left.
        "rifle": {origin: ("R", side), exit_square: None},
        # Circe: the rook stands on the exit, the stone is reborn at its home.
        "circe": {origin: None, exit_square: ("R", side), home: ("S", foe)},
        # Swap: the rook on the exit, the struck stone (a guard now) on the square it left.
        "swap": {origin: ("N", foe), exit_square: ("R", side)},
        # Choose: the rook on the exit, the guard one step forward of it (the mover's forward).
        "choose": {origin: None, exit_square: ("R", side), placed: ("N", foe)},
        # A demoting chain under stay: the rook never left, the stone a guard at the exit.
        "demote": {origin: ("R", side), exit_square: ("N", foe)},
        # Andernach: the rook takes at the exit and changes sides there.
        "andernach": {origin: None, exit_square: ("R", foe)},
        # A link under stay: as the chain.
        "contacts": {origin: ("R", side), exit_square: ("N", foe)},
    }[reading]


class TestTheExitIsTheCaptureSquare:
    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("reading", READINGS)
    def test_each_reading_reads_from_the_exit(self, reading: str, kind: str, side: int) -> None:
        engine = _engine(_spec(reading, kind))
        state, origin, exit_square = _struck_position(engine, kind, side)
        moves = _through(engine, state, origin, exit_square)
        assert len(moves) == 1, [engine.describe_move(state, m) for m in moves]
        move = moves[0]
        if kind == "momentum":
            assert move_type(move) == MOVE_PORTAL
        after = engine.apply(state, move)
        for square, expected in _expected(reading, side, origin, exit_square).items():
            assert _at(engine, after, square) == expected, (reading, kind, side, square)
        assert after.key == engine._compute_hash(after)
        # Undo exact: the position played from is the position it was.
        assert engine.apply(state, move).board == after.board

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_text_says_what_the_capture_did(self, kind: str) -> None:
        texts = {}
        for reading in ("rifle", "swap", "choose", "demote"):
            engine = _engine(_spec(reading, kind))
            state, origin, exit_square = _struck_position(engine, kind, 0)
            texts[reading] = engine.describe_move(state, _through(engine, state, origin, exit_square)[0])
        sign = ">x" if kind == "momentum" else "~x"
        assert texts["rifle"].endswith("(stays)") and sign in texts["rifle"]
        assert texts["swap"].endswith("->N (swap)")
        assert texts["demote"].endswith("->N (stands fast)")
        # A choosing capture names its landing and no end.
        placed = "e6" if kind == "momentum" else "e5"
        assert texts["choose"] == f"Rb1{'>' if kind == 'momentum' else ''}x{_exit(kind, 0)}->N (placed on {placed})"

    @pytest.mark.parametrize("kind", KINDS)
    def test_a_choosing_word_carries_its_landing_and_no_end(self, kind: str) -> None:
        engine = _engine(_spec("choose", kind))
        state, origin, exit_square = _struck_position(engine, kind, 0)
        (move,) = _through(engine, state, origin, exit_square)
        assert move_landing(move) == 0  # the landing list's one entry
        assert engine.landing_square(state, move) == _sq("e6" if kind == "momentum" else "e5")
        assert engine.portal_entry(state, move) is None
        assert engine.fold_route(state, move) is None

    @pytest.mark.parametrize("kind", KINDS)
    def test_a_choosing_capture_with_no_free_landing_is_no_move(self, kind: str) -> None:
        engine = _engine(_spec("choose", kind))
        state, origin, exit_square = _struck_position(engine, kind, 0)
        placement = {"a1": ("K", 0), "a6": ("K", 1), origin: ("R", 0), exit_square: ("S", 1),
                     **ENDS[kind]}
        placement["e6" if kind == "momentum" else "e5"] = ("N", 1)
        blocked = engine.make_state(placement, moved=list(placement))
        assert _through(engine, blocked, origin, exit_square) == []


# --------------------------------------------------------------------------- one move per landing


def _map_spec(reading: str) -> dict[str, Any]:
    """Two ends, b3 and c2, both leading to e4: a jumper on a1 reaches both with the same sign
    of travel (up and right), so both come out on f5."""
    spec = _spec(reading, "momentum")
    spec["rules"]["portals"] = [{"ends": [
        {"at": "b3", "to": 2}, {"at": "c2", "to": 2}, {"at": "e4", "to": 0},
    ]}]
    return spec


def _map_state(engine: Engine) -> GameState:
    """The jumper on a1, a stone on f5; the map's three ends where they start."""
    ends = {square: (engine.spec.piece_ids[engine.spec.type_of(value)], WILD)
            for square, value in _named(engine.initial_state()).items()
            if engine.spec.owner_of(value) == WILD}
    return engine.make_state({"a1": ("J", 0), "a6": ("K", 1), "f1": ("K", 0), "f5": ("S", 1),
                              **ends}, moved=["a1", "a6", "f1", "f5"])


def _named(state: GameState) -> dict[str, int]:
    return {"abcdef"[s % WIDTH] + str(s // WIDTH + 1): v for s, v in enumerate(state.board) if v}


class TestTwoWaysToOneExit:
    def test_without_choose_each_end_is_its_own_move(self) -> None:
        engine = _engine(_map_spec("rifle"))
        state = _map_state(engine)
        moves = _through(engine, state, "a1", "f5")
        texts = sorted(engine.describe_move(state, m) for m in moves)
        assert texts == ["Ja1-b3>xf5 (stays)", "Ja1-c2>xf5 (stays)"]

    def test_under_choose_one_move_per_landing(self) -> None:
        engine = _engine(_map_spec("choose"))
        state = _map_state(engine)
        moves = _through(engine, state, "a1", "f5")
        assert [engine.describe_move(state, m) for m in moves] == ["Ja1>xf5->N (placed on f6)"]
        after = engine.apply(state, moves[0])
        assert _at(engine, after, "f5") == ("J", 0)
        assert _at(engine, after, "f6") == ("N", 1)


class TestAGoThroughAndADirectCapture:
    def test_under_choose_they_are_one_move(self) -> None:
        """A rook on b1 reaches d1 directly along the rank and through the fold on b3, whose
        far end stands on d1: one word, the direct one (the same position)."""
        spec = _spec("choose", "fold")
        spec["rules"]["portals"] = [{"orange": "b3", "blue": "d1", "kind": "fold"}]
        engine = _engine(spec)
        state = engine.make_state({"b1": ("R", 0), "a6": ("K", 1), "a1": ("K", 0),
                                   "d1": ("S", 1)}, moved=["b1", "a6", "a1", "d1"])
        moves = _through(engine, state, "b1", "d1")
        assert len(moves) == 1, [engine.describe_move(state, m) for m in moves]
        assert engine.describe_move(state, moves[0]) == "Rb1xd1->N (placed on d2)"


class TestAHeldPieceThroughTheDoor:
    @pytest.mark.parametrize("kind", KINDS)
    def test_it_places_by_its_holders_landing(self, kind: str) -> None:
        """A condottiere nobody owns, held by the first player (two pieces beside it, none of
        the second's), steps from a2 onto b3 and comes out on f5 (momentum) or e4 (the fold's
        far square), striking the stone there under choose: one word, foreign, the guard placed
        one step forward of the exit by its holder's forward, the condottiere nobody's still."""
        spec = _spec("choose", kind)
        spec["pieces"].append({"id": "C", "name": "condottiere", "value": 3, "moves": [KING],
                               "control": "majority"})
        spec["setup"]["placements"].append({"piece": "C", "player": "nobody", "squares": ["c3"]})
        engine = _engine(spec)
        exit_square, placed = ("f5", "f6") if kind == "momentum" else ("e4", "e5")
        state = engine.make_state({
            "a1": ("K", 0), "b1": ("N", 0), "a2": ("C", WILD), "a6": ("K", 1),
            exit_square: ("S", 1), **ENDS[kind],
        }, moved=["a1", "b1", "a2", "a6", exit_square])
        moves = _through(engine, state, "a2", exit_square)
        assert len(moves) == 1, [engine.describe_move(state, m) for m in moves]
        assert moves[0] & MOVE_FOREIGN
        after = engine.apply(state, moves[0])
        assert _at(engine, after, exit_square) == ("C", WILD)
        assert _at(engine, after, placed) == ("N", 1)
        assert _at(engine, after, "a2") is None
        assert after.key == engine._compute_hash(after)


# --------------------------------------------------------------------------- the validator


class TestTheLift:
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("reading", READINGS)
    def test_the_pair_is_valid(self, reading: str, kind: str) -> None:
        errors = validate_spec(_spec(reading, kind))
        assert errors == []
        assert not {"P3", "P4"} & refused_ids(errors)

    def test_the_inventory_names_both(self) -> None:
        assert _schema_seventy_seven_uses(_spec("rifle", "momentum")) == {"rules.portals.strike"}
        assert _schema_seventy_seven_uses(_spec("contacts", "fold")) == {"rules.portals.contacts"}
        assert _schema_seventy_seven_uses(_base("momentum")) == set()
        no_doors = _spec("rifle", "momentum")
        del no_doors["rules"]["portals"]
        assert _schema_seventy_seven_uses(no_doors) == set()

    @pytest.mark.parametrize("reading", ["rifle", "contacts"])
    def test_describe_says_it(self, reading: str) -> None:
        text = describe_spec(_spec(reading, "momentum"))
        assert "A capture through a portal or a fold is made on the square the move comes out" in text
        assert "A capture through a portal" not in describe_spec(_base("momentum"))

    @pytest.mark.parametrize("reading", ["rifle", "contacts"])
    def test_fairy_stockfish_refuses_it(self, reading: str) -> None:
        with pytest.raises(Untranslatable):
            translate(_spec(reading, "momentum"))


# --------------------------------------------------------------------------- the walks


class TestTheWalks:
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("reading", READINGS)
    def test_walks_keep_the_key(self, reading: str, kind: str) -> None:
        """Seeded walks, transit and go-through captures preferred; the key is the full scan's
        each ply, and every move's text is unique in its list."""
        engine = _engine(_spec(reading, kind))
        through = 0
        for seed in range(6):
            rng = random.Random(seed)
            state = engine.initial_state()
            for _ in range(60):
                moves = engine.legal_moves(state)
                if not moves or engine.terminal(state, moves) is not None:
                    break
                texts = [engine.describe_move(state, m) for m in moves]
                assert len(set(texts)) == len(texts), texts
                loud = [m for m in moves if state.board[move_to(m)]]
                doors = [m for m in loud if ">" in engine.describe_move(state, m)
                         or "~" in engine.describe_move(state, m)
                         or engine.landing_square(state, m) is not None]
                if doors and rng.random() < 0.9:
                    move = rng.choice(doors)
                    through += 1
                else:
                    move = rng.choice(loud if loud and rng.random() < 0.5 else moves)
                after = engine.apply(state, move)
                assert after.key == engine._compute_hash(after), (reading, kind, seed)
                state = after
        assert through > 0, (reading, kind)

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("reading", READINGS)
    def test_perft_counts_what_apply_counts(self, reading: str, kind: str) -> None:
        """Make and unmake against `apply`: every struck, placed or reborn piece put back
        exactly, through the door both ways."""
        engine = _engine(_spec(reading, kind))
        state = engine.make_state({
            "a1": ("K", 0), "a6": ("K", 1), "b1": ("R", 0), "b6": ("R", 1),
            "e4" if kind == "fold" else "e5": ("S", 1), "e3": ("S", 0),
            "c1": ("J", 0), "d2": ("N", 0), "d5": ("N", 1), **ENDS[kind],
        }, moved=["a1", "a6", "b1", "b6", "c1", "d2", "d5"])

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(
                count(engine.apply(position, m), depth - 1) for m in engine.legal_moves(position)
            )

        assert engine.perft(state, 3) == count(state, 3)
