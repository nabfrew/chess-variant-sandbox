"""SR4k (`docs/plan-schema-next.md` §2.11 rows O3 and O4): pieces nobody owns beside a strike
and beside contacts.

The rulings (2026-10-28). O3: a struck nobody's piece becomes what the chain says and stays
nobody's; a held piece's capture strikes as its holder's piece would. O4, read as O3's: a
nobody's piece a link transforms stays nobody's, and a held piece's capture meets its target
by the holder's links.

The probe (SR2's method, the pair fed past the validator on SR4j's build `e598d703556543d3`,
both seats, seven readings: rifle, circe, swap, choose, Andernach, a demoting chain, a contact
table). A player's capture of a nobody's piece already played as ruled under every reading but
circe, where the core panicked reading the rebirth table for the third owner (`tables.rs`, the
table had two blocks). A held piece's capture played as its holder's under rifle, circe, swap,
a chain and contacts; under Andernach the core panicked (`1 - mover_owner` for the third
owner), and under choose a held capture neither fanned out over its landings (the foreign walk
never asked) nor could read them (`landing(mover_owner)`). So the core reads the holder for a
held mover (`strike_writes`, `unmake`, `foreign.rs`), writes no side switch for it (the mover's
reading: moving it never made it the holder's, so it stays nobody's), compiles a third rebirth
block where nobody's pieces stand, and `arriving_value` leaves a nobody's survivor at facing 0
with no promotion zone. The validator reads the types a nobody's piece can become
(`schema.nobodys_ids`), so a chain into a majority type makes a held piece, and a chain into a
type a nobody's piece may not be (royal, resting, seeing) is refused as a placement is.

Hand-built 6x6 specs, not chess: two kings, a rook and a guard each, a condottiere (majority
control) on c3 and a stone on e3 placed for nobody (mirrored to c4, e4). Every walk checks the
incremental key against `_compute_hash` each ply.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import CONTROL_MAJORITY, WILD, compile_spec
from chess_variant_sandbox.engine.moves import MOVE_FOREIGN, move_from, move_to
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import majority_ids, nobodys_ids, validate_spec
from tests.feature_inventory import _schema_seventy_five_uses

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
DEMOTE = [{"from": "S", "to": "N"}, {"from": "R", "to": "S"}]

RULES: dict[str, tuple[str, dict[str, Any]]] = {
    "rifle": ("strike", {"effect": "capture", "position": "stay", "immune": ["K"]}),
    "circe": ("strike", {"effect": "capture", "position": "home", "immune": ["K"]}),
    "swap": ("strike", {"effect": "demote", "position": "swap", "immune": ["K"], "chain": DEMOTE}),
    "choose": ("strike", {"effect": "demote", "position": "choose", "immune": ["K"],
                          "landing": {"vectors": [[0, 1]]}, "chain": DEMOTE}),
    "andernach": ("strike", {"effect": "capture", "switches_sides": ["R", "C"], "immune": ["K"]}),
    "demote": ("strike", {"effect": "demote", "position": "stay", "immune": ["K"],
                          "chain": DEMOTE}),
    "contacts": ("contacts", {"position": "stay", "immune": ["K"], "links": [
        {"mover": "R", "onto": "S", "onto_to": "N"},
        {"mover": "C", "onto": "R", "onto_to": "N", "mover_to": "S"},
    ]}),
}
READINGS = sorted(RULES)


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _base() -> dict[str, Any]:
    return copy.deepcopy({
        "name": "the struck crowd (SR4k test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "C", "name": "condottiere", "value": 3, "moves": [KING], "control": "majority"},
            {"id": "S", "name": "stone", "value": 2, "moves": [KING]},
            {"id": "N", "name": "guard", "value": 3, "moves": [KING]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "R", "squares": ["f1"]},
            {"piece": "N", "squares": ["b2", "d2"]},
            {"piece": "C", "player": "nobody", "squares": ["c3"]},
            {"piece": "S", "player": "nobody", "squares": ["e3"]},
        ]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 80},
    })


def _spec(reading: str) -> dict[str, Any]:
    spec = _base()
    key, value = RULES[reading]
    spec["rules"][key] = copy.deepcopy(value)
    return spec


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


def _move(engine: Engine, state: GameState, origin: str, target: str) -> int:
    for move in engine.legal_moves(state):
        if (move_from(move), move_to(move)) == (_sq(origin), _sq(target)):
            return move
    raise AssertionError(f"no move {origin}-{target}")


def _at(engine: Engine, state: GameState, square: str) -> tuple[str, int] | None:
    value = state.board[_sq(square)]
    if not value:
        return None
    spec = engine.spec
    return spec.piece_ids[spec.type_of(value)], spec.owner_of(value)


def _placement(engine: Engine, state: GameState) -> dict[str, tuple[str, int]]:
    spec = engine.spec
    return {
        "abcdef"[square % WIDTH] + str(square // WIDTH + 1):
            (spec.piece_ids[spec.type_of(value)], spec.owner_of(value))
        for square, value in enumerate(state.board) if value
    }


# --------------------------------------------------------------------------- the struck piece


# What a rook on e1 taking the stone on e3 leaves, square by square, under each reading.
STRUCK: dict[str, dict[str, tuple[str, int] | None]] = {
    "rifle": {"e1": ("R", 0), "e3": None},
    "swap": {"e1": ("N", WILD), "e3": ("R", 0)},
    "choose": {"e1": None, "e3": ("R", 0), "e4": ("N", WILD)},
    "andernach": {"e1": None, "e3": ("R", 1)},
    "demote": {"e1": ("R", 0), "e3": ("N", WILD)},
    "contacts": {"e1": ("R", 0), "e3": ("N", WILD)},
}


class TestAStruckNobodysPiece:
    @pytest.mark.parametrize("reading", sorted(STRUCK))
    def test_it_becomes_what_the_rule_says_and_stays_nobodys(self, reading: str) -> None:
        engine = _engine(_spec(reading))
        state = engine.make_state({
            "a1": ("K", 0), "a6": ("K", 1), "e1": ("R", 0), "e3": ("S", WILD), "f6": ("R", 1),
        }, moved=["a1", "a6", "e1", "f6"])
        after = engine.apply(state, _move(engine, state, "e1", "e3"))
        for square, expected in STRUCK[reading].items():
            assert _at(engine, after, square) == expected, (reading, square)
        assert after.key == engine._compute_hash(after)

    def test_circe_reborn_on_a_home_it_stands_on_for_nobody(self) -> None:
        """The stone's homes are e3 and e4, one of each colour: taken on d3, it goes back to
        e4, nobody's (the third rebirth block)."""
        engine = _engine(_spec("circe"))
        state = engine.make_state({
            "a1": ("K", 0), "a6": ("K", 1), "d1": ("R", 0), "d3": ("S", WILD),
        }, moved=["a1", "a6", "d1"])
        move = _move(engine, state, "d1", "d3")
        after = engine.apply(state, move)
        assert _at(engine, after, "d3") == ("R", 0)
        assert _at(engine, after, "e4") == ("S", WILD)
        assert after.key == engine._compute_hash(after)
        assert engine.describe_move(state, move).endswith("(reborn e4)")

    def test_circe_with_its_home_taken_removes_it(self) -> None:
        engine = _engine(_spec("circe"))
        state = engine.make_state({
            "a1": ("K", 0), "a6": ("K", 1), "d1": ("R", 0), "d3": ("S", WILD),
            "e4": ("N", 1),
        }, moved=["a1", "a6", "d1", "e4"])
        after = engine.apply(state, _move(engine, state, "d1", "d3"))
        assert sum(1 for v in after.board if v and engine.spec.owner_of(v) == WILD) == 0

    def test_a_chain_into_a_majority_type_makes_a_held_piece(self) -> None:
        """The stone struck into a condottiere is nobody's condottiere, held by the count
        beside it: the next move, the player with more pieces next to it may move it."""
        spec = _spec("demote")
        spec["rules"]["strike"]["chain"] = [{"from": "S", "to": "C"}]
        engine = _engine(spec)
        compiled = engine.spec
        assert compiled.neutrals is not None
        assert compiled.neutrals.kind[compiled.piece_ids.index("C")] == CONTROL_MAJORITY
        state = engine.make_state({
            "a1": ("K", 0), "a6": ("K", 1), "e1": ("R", 0), "e3": ("S", WILD),
            "d4": ("N", 1), "f4": ("N", 1),
        }, moved=["a1", "a6", "e1", "d4", "f4"])
        after = engine.apply(state, _move(engine, state, "e1", "e3"))
        assert _at(engine, after, "e3") == ("C", WILD)
        held = [m for m in engine.legal_moves(after) if move_from(m) == _sq("e3")]
        assert held and all(m & MOVE_FOREIGN for m in held)


# --------------------------------------------------------------------------- the held piece


def _holds_to_its_twin(engine: Engine, state: GameState, move: int, after: GameState) -> None:
    """The held piece's move against its holder's own piece of the type on the same square
    (the twin): every square as the twin leaves it but one, the mover's, where the held game
    has the same type in nobody's block (any change of type kept, no side switch)."""
    spec = engine.spec
    side = state.side_to_move
    origin = move_from(move)
    placement = _placement(engine, state)
    name = "abcdef"[origin % WIDTH] + str(origin // WIDTH + 1)
    placement[name] = (placement[name][0], side)
    twin = engine.make_state(placement, side_to_move=side, moved=list(placement))
    board = engine.apply(twin, move & ~MOVE_FOREIGN).board
    differ = [s for s in range(len(board)) if board[s] != after.board[s]]
    assert len(differ) == 1, differ
    square = differ[0]
    assert square in (origin, move_to(move))
    assert spec.owner_of(after.board[square]) == WILD
    assert spec.owner_of(board[square]) != WILD
    assert spec.type_of(after.board[square]) == spec.type_of(board[square])


class TestAHeldPiecesCapture:
    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("reading", READINGS)
    def test_it_strikes_as_its_holders_piece_would(self, reading: str, side: int) -> None:
        engine = _engine(_spec(reading))
        foe = 1 - side
        state = engine.make_state({
            "a1": ("K", 0), "a6": ("K", 1), "c3": ("C", WILD), "d4": ("R", foe),
            "b2": ("N", side), "b3": ("N", side), "f6": ("R", foe), "f1": ("R", side),
        }, side_to_move=side, moved=["a1", "a6", "d4", "b2", "b3", "f6", "f1"])
        moves = [m for m in engine.legal_moves(state)
                 if (move_from(m), move_to(m)) == (_sq("c3"), _sq("d4"))]
        assert moves and all(m & MOVE_FOREIGN for m in moves)
        for move in moves:
            after = engine.apply(state, move)
            assert after.key == engine._compute_hash(after)
            _holds_to_its_twin(engine, state, move, after)

    @pytest.mark.parametrize("side", [0, 1])
    def test_choose_places_by_the_holders_forward(self, side: int) -> None:
        """The landing is one step forward of the struck square, forward being the holder's:
        d5 for the first player, d3 for the second (c3, the origin, is left for it)."""
        engine = _engine(_spec("choose"))
        state = engine.make_state({
            "a1": ("K", 0), "a6": ("K", 1), "c3": ("C", WILD), "d4": ("R", 1 - side),
            "b2": ("N", side), "b3": ("N", side),
        }, side_to_move=side, moved=["a1", "a6", "d4", "b2", "b3"])
        move = _move(engine, state, "c3", "d4")
        after = engine.apply(state, move)
        placed = "d5" if side == 0 else "d3"
        assert _at(engine, after, placed) == ("S", 1 - side)
        assert _at(engine, after, "d4") == ("C", WILD)
        assert engine.landing_square(state, move) == _sq(placed)
        assert engine.describe_move(state, move).endswith(f"(placed on {placed})")

    def test_andernach_never_switches_a_held_piece(self) -> None:
        engine = _engine(_spec("andernach"))
        state = engine.make_state({
            "a1": ("K", 0), "a6": ("K", 1), "c3": ("C", WILD), "d4": ("R", 1),
            "b2": ("N", 0), "b3": ("N", 0),
        }, moved=["a1", "a6", "d4", "b2", "b3"])
        move = _move(engine, state, "c3", "d4")
        after = engine.apply(state, move)
        assert _at(engine, after, "d4") == ("C", WILD)
        assert "changes sides" not in engine.describe_move(state, move)

    def test_a_link_changes_the_held_piece_and_it_stays_nobodys(self) -> None:
        """The contact table's link from the condottiere onto a rook makes it a stone, nobody's
        still, standing where it took (`position: stay` keeps it on c3)."""
        engine = _engine(_spec("contacts"))
        state = engine.make_state({
            "a1": ("K", 0), "a6": ("K", 1), "c3": ("C", WILD), "d4": ("R", 1),
            "b2": ("N", 0), "b3": ("N", 0),
        }, moved=["a1", "a6", "d4", "b2", "b3"])
        after = engine.apply(state, _move(engine, state, "c3", "d4"))
        assert _at(engine, after, "c3") == ("S", WILD)
        assert _at(engine, after, "d4") == ("N", 1)


# --------------------------------------------------------------------------- the validator


class TestTheLift:
    @pytest.mark.parametrize("reading", READINGS)
    def test_the_pair_is_valid(self, reading: str) -> None:
        errors = validate_spec(_spec(reading))
        assert errors == []
        assert not {"O3", "O4"} & refused_ids(errors)

    def test_nobodys_ids_follow_the_tables(self) -> None:
        assert nobodys_ids(_base()) == {"C", "S"}
        assert nobodys_ids(_spec("demote")) == {"C", "S", "N"}
        assert nobodys_ids(_spec("contacts")) == {"C", "S", "N"}
        assert nobodys_ids(_spec("rifle")) == {"C", "S"}
        chained = _spec("demote")
        chained["rules"]["strike"]["chain"] = [{"from": "S", "to": "C"}]
        chained["pieces"][2]["control"] = "stands"
        chained["setup"]["placements"] = [
            p for p in chained["setup"]["placements"] if p["piece"] != "C"
        ]
        assert nobodys_ids(chained) == {"C", "S"}
        assert majority_ids(chained) == set()
        chained["pieces"][2]["control"] = "majority"
        assert majority_ids(chained) == {"C"}
        assert validate_spec(chained) == []

    def test_a_chain_into_a_type_nobody_may_own_is_refused(self) -> None:
        spec = _spec("demote")
        spec["pieces"][4]["rest"] = 1
        errors = validate_spec(spec)
        assert any("which a strike or a contact makes of a piece nobody owns" in e
                   and "rest" in e for e in errors), errors

    def test_the_inventory_names_both(self) -> None:
        assert _schema_seventy_five_uses(_spec("rifle")) == {"rules.strike.nobodys"}
        assert _schema_seventy_five_uses(_spec("contacts")) == {"rules.contacts.nobodys"}
        assert _schema_seventy_five_uses(_base()) == set()

    @pytest.mark.parametrize("reading", ["demote", "contacts"])
    def test_describe_says_it(self, reading: str) -> None:
        text = describe_spec(_spec(reading))
        assert "reaches a piece nobody owns turns it into what the" in text
        assert "may also be a N" in text
        assert "it never changes sides" in text
        assert "reaches a piece nobody owns" not in describe_spec(_base())

    @pytest.mark.parametrize("reading", ["rifle", "contacts"])
    def test_fairy_stockfish_refuses_it(self, reading: str) -> None:
        with pytest.raises(Untranslatable):
            translate(_spec(reading))


# --------------------------------------------------------------------------- the walks


class TestTheWalks:
    @pytest.mark.parametrize("reading", READINGS)
    def test_walks_keep_the_key_and_the_holders_reading(self, reading: str) -> None:
        """Seeded walks, captures preferred; every held capture holds to its twin, every
        nobody's byte stays in nobody's block, and the key is the full scan's each ply."""
        engine = _engine(_spec(reading))
        held = struck = 0
        for seed in range(8):
            rng = random.Random(seed)
            state = engine.initial_state()
            for _ in range(60):
                moves = engine.legal_moves(state)
                if not moves or engine.terminal(state, moves) is not None:
                    break
                loud = [m for m in moves if state.board[move_to(m)]]
                held_loud = [m for m in loud if m & MOVE_FOREIGN]
                if held_loud and rng.random() < 0.9:
                    move = rng.choice(held_loud)
                else:
                    move = rng.choice(loud if loud and rng.random() < 0.7 else moves)
                victim = state.board[move_to(move)]
                after = engine.apply(state, move)
                if move & MOVE_FOREIGN and victim:
                    held += 1
                    _holds_to_its_twin(engine, state, move, after)
                elif victim and engine.spec.owner_of(victim) == WILD:
                    struck += 1
                assert after.key == engine._compute_hash(after), (reading, seed)
                state = after
        assert held > 0 and struck > 0, (reading, held, struck)

    @pytest.mark.parametrize("reading", READINGS)
    def test_perft_counts_what_apply_counts(self, reading: str) -> None:
        """Make and unmake against `apply`: every struck, placed, reborn or held piece put
        back exactly."""
        engine = _engine(_spec(reading))
        state = engine.make_state({
            "a1": ("K", 0), "a6": ("K", 1), "c1": ("R", 0), "d6": ("R", 1),
            "b3": ("N", 0), "d4": ("N", 1), "c3": ("C", WILD), "e3": ("S", WILD),
            "c4": ("C", WILD), "e4": ("S", WILD),
        }, moved=["a1", "a6", "c1", "d6", "b3", "d4"])

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(
                count(engine.apply(position, m), depth - 1) for m in engine.legal_moves(position)
            )

        assert engine.perft(state, 3) == count(state, 3)
