"""SR4j (`docs/plan-schema-next.md` §2.4 row T10, box SR4j): a strike's `by` and its
`switches_sides` read apart.

The ruling (2026-10-27): ``by`` says whose captures strike and ``switches_sides`` which capturers
change sides, independently. Under rifle a switcher outside ``by`` captures as usual (advances)
and changes sides on the square it took; a striker outside the switch stays where it stood; one
in both stays and changes sides there. A ``by`` beside pure Andernach (a switch and nothing else)
names a strike that never fires, and is refused.

**The probe** (the pair fed past the validator on SR4i's build `645d9d6762c9b1e7`, the position of
`tests/test_readings_without_core.py`): the knight, a switcher outside ``by``, neither changed
sides nor stayed (`compile._compile_strike` flipped only a listed striker); with that line read
apart, the core still placed every flipping capturer by the strike's one position, so the knight
would have stayed on its origin. `strike_writes` now reads a meeting the strike does not reach
(its row of the table is plain capture) as an advance, the switch alone changing the capturer's
side there; `rounds.rs::stays_put` reads it alike.

The oracle is this file's: each capture leaves the board its twin without ``switches_sides``
leaves, with the capturer, wherever that twin left it, changed to the other side when its type
is listed and its victim is not immune (a declared test oracle, rule 3). `make` is checked
through `apply`, `unmake` through `perft` against an `apply`-built count and through searching
games replayed. Hand-built specs, not chess.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.tournament.play import play_game
from tests.feature_inventory import _schema_seventy_four_uses

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
ARCHER = {"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4"}
KNIGHT = {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}

# The acceptance: rook and archer strike, knight and archer switch, the guard does neither.
RIFLE = {"effect": "capture", "position": "stay", "immune": ["K"],
         "by": ["A", "R"], "switches_sides": ["N", "A"]}
CIRCE = {"effect": "capture", "position": "home", "immune": ["K"],
         "by": ["A"], "switches_sides": ["N", "A"]}
DEMOTE = {"effect": "demote", "position": "stay", "immune": ["K"], "by": ["R", "A"],
          "chain": [{"from": "R", "to": "N"}, {"from": "N", "to": "G"}],
          "switches_sides": ["N", "A", "G"]}
READINGS: dict[str, dict[str, Any]] = {"rifle": RIFLE, "circe": CIRCE, "demote": DEMOTE}


def _spec(strike: dict[str, Any] | None) -> dict[str, Any]:
    """6x6 under royal capture: a king, rooks, archers `A` (a diagonal slider), knights and
    guards `G` (a king's step); ``strike`` added."""
    spec: dict[str, Any] = {
        "schema_version": 31, "name": "the turncoat archers",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "A", "value": 3, "moves": [ARCHER]},
            {"id": "N", "value": 3, "moves": [KNIGHT]},
            {"id": "G", "value": 2, "moves": [KING]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["c1"]}, {"piece": "R", "squares": ["a1"]},
            {"piece": "A", "squares": ["d1"]}, {"piece": "N", "squares": ["b1", "e1"]},
            {"piece": "G", "squares": ["f1", "a2", "c2", "e2"]},
        ]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 80},
    }
    if strike is not None:
        spec["rules"]["strike"] = copy.deepcopy(strike)
    return spec


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _as(engine: Engine, state: GameState) -> GameState:
    return dataclasses.replace(state, key=engine._compute_hash(state))


def _without_switch(strike: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in strike.items() if key != "switches_sides"}


def _expected_board(both: Engine, twin: Engine, strike: dict[str, Any], state: GameState,
                    move: int) -> bytes:
    """The oracle: the twin's board, the capturer flipped where the twin left it when its
    type switches and its victim is not immune."""
    spec = both.spec
    board = bytearray(twin.apply(_as(twin, state), move).board)
    origin, target = move & 0xFF, (move >> 8) & 0xFF
    mover, victim = state.board[origin], state.board[target]
    side = state.side_to_move
    if not victim or spec.owner_of(victim) == side:
        return bytes(board)
    ids = spec.piece_ids
    if ids[spec.type_of(mover)] not in strike["switches_sides"]:
        return bytes(board)
    if ids[spec.type_of(victim)] in strike["immune"]:
        return bytes(board)
    ends = origin if board[origin] else target
    standing = board[ends]
    assert spec.owner_of(standing) == side
    board[ends] = spec.encode_piece(1 - side, spec.type_of(standing))
    return bytes(board)


# --------------------------------------------------------------------------- the validator


class TestTheLift:
    @pytest.mark.parametrize("strike", READINGS.values(), ids=READINGS.keys())
    def test_each_reading_validates(self, strike: dict[str, Any]) -> None:
        assert validate_spec(_spec(strike)) == []

    def test_by_beside_pure_andernach_never_fires(self) -> None:
        andernach = {"effect": "capture", "immune": ["K"], "by": ["A"], "switches_sides": ["N"]}
        errors = validate_spec(_spec(andernach))
        assert any("its strike never fires" in error for error in errors), errors

    def test_the_inventory_names_the_pair(self) -> None:
        assert _schema_seventy_four_uses(_spec(RIFLE)) == {"rules.strike.by.switches_sides"}
        assert _schema_seventy_four_uses(_spec(_without_switch(RIFLE))) == set()
        assert _schema_seventy_four_uses(
            _spec({k: v for k, v in RIFLE.items() if k != "by"})) == set()

    def test_describe_says_each_list(self) -> None:
        text = describe_spec(_spec(RIFLE))
        assert "shoots" in text
        assert "changes sides: it becomes the other player's piece where its capture leaves it" \
            in text

    def test_fairy_stockfish_refuses_the_strike(self) -> None:
        with pytest.raises(Untranslatable, match="strike"):
            translate(_spec(RIFLE))


# --------------------------------------------------------------------------- the acceptance


# (origin, target): where the capturer ends and whether it changed sides, by its two lists.
RIFLE_EXPECTED = {
    ("c3", "d5"): ("d5", True),    # knight: switches, does not strike: advances, changes sides
    ("c5", "d4"): ("c5", True),    # archer: both: stays, changes sides
    ("a3", "a5"): ("a3", False),   # rook: strikes, does not switch: stays
    ("e3", "e4"): ("e4", False),   # guard: neither: an ordinary capture
}


class TestTheAcceptance:
    @pytest.mark.parametrize("side", [0, 1])
    def test_each_capturer_reads_its_two_lists(self, side: int) -> None:
        engine = _engine(_spec(RIFLE))
        other = 1 - side
        placement = {
            "a1": ("K", side), "c3": ("N", side), "c5": ("A", side), "a3": ("R", side),
            "e3": ("G", side), "f6": ("K", other),
            "d5": ("G", other), "d4": ("G", other), "a5": ("G", other), "e4": ("N", other),
        }
        state = engine.make_state(placement, side_to_move=side, moved=list(placement))
        ids = engine.spec.piece_ids
        for (origin, target), (ends, flips) in RIFLE_EXPECTED.items():
            move = next(m for m in engine.legal_moves(state)
                        if (m & 0xFF, (m >> 8) & 0xFF) == (_sq(origin), _sq(target)))
            after = engine.apply(state, move)
            stood = after.board[_sq(ends)]
            assert ids[engine.spec.type_of(stood)] == placement[origin][0], (origin, target)
            assert engine.spec.owner_of(stood) == (other if flips else side), (origin, target)
            assert after.board[_sq(target if ends == origin else origin)] == 0, (origin, target)
            assert after.key == engine._compute_hash(after)


# --------------------------------------------------------------------------- walks


class TestTheWalks:
    @pytest.mark.parametrize("strike", READINGS.values(), ids=READINGS.keys())
    def test_the_pair_plays_as_the_oracle_says(self, strike: dict[str, Any]) -> None:
        """Seeded walks, both seats, leaning on captures: every list is the twin's (a switch
        changes no move), each board the oracle's and the key the full scan's."""
        both, twin = _engine(_spec(strike)), _engine(_spec(_without_switch(strike)))
        switched = {0: 0, 1: 0}
        for seed in range(16):
            rng = random.Random(seed)
            state = both.initial_state()
            for _ in range(60):
                moves = both.legal_moves(state)
                if both.terminal(state, moves) is not None:
                    break
                assert sorted(moves) == sorted(twin.legal_moves(_as(twin, state))), (seed, state.ply)
                takes = [m for m in moves if state.board[(m >> 8) & 0xFF]]
                move = rng.choice(takes) if takes and rng.random() < 0.7 else rng.choice(moves)
                after = both.apply(state, move)
                expected = _expected_board(both, twin, strike, state, move)
                assert after.board == expected, (seed, state.ply, both.describe_move(state, move))
                assert after.key == both._compute_hash(after), (seed, state.ply)
                if after.board != twin.apply(_as(twin, state), move).board:
                    switched[state.side_to_move] += 1
                state = after
        assert switched[0] and switched[1], switched


# --------------------------------------------------------------------------- unmake


def _apply_count(engine: Engine, state: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(
        _apply_count(engine, engine.apply(state, move), depth - 1)
        for move in engine.legal_moves(state)
    )


class TestTheUndo:
    @pytest.mark.parametrize("strike", READINGS.values(), ids=READINGS.keys())
    def test_perft_counts_what_apply_counts(self, strike: dict[str, Any]) -> None:
        """`perft` makes and unmakes on one position; a capturer an unmake left on the wrong
        square or side would miscount the next sibling."""
        engine = _engine(_spec(strike))
        placement = {
            "a1": ("K", 0), "c3": ("N", 0), "b2": ("A", 0), "a3": ("R", 0), "e3": ("G", 0),
            "f6": ("K", 1), "d5": ("G", 1), "d4": ("A", 1), "a5": ("R", 1), "e4": ("N", 1),
        }
        state = engine.make_state(placement, side_to_move=0, moved=list(placement))
        assert engine.perft(state, 3) == _apply_count(engine, state, 3)

    def test_a_searching_game_replays(self) -> None:
        engine = _engine(_spec(RIFLE))
        for seed in range(2):
            record = play_game(engine, ("minimax:nodes=200", "minimax:nodes=200"), seed)
            state = engine.initial_state()
            for move in record.moves:
                assert move in engine.legal_moves(state)
                state = engine.apply(state, move)
            outcome = engine.terminal(state)
            assert outcome is not None
            assert (outcome.winner, outcome.reason) == (record.winner, record.reason)
