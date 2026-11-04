"""SR4b (`docs/plan-schema-next.md` §2.6 row W4, box SR4b): swept pieces into the hand.

The ruling (2026-10-27): beside `rules.drops`, every piece a sweep removes that the mover does
not own goes to the mover's hand, as a piece taken on the landing square does (demoted by
`promoted_from`, an unlimited slot untouched); a swept friendly piece is removed, to no hand.
The core deposits from the same log the sweep's clear writes (`Game::sweep_to_hand`), and
`unmake` reads it back, so the variable-length deposit is undone as the clear is.

The oracle is this file's `_expected_hands`: a mover's hand changes by the demoted types of
the pieces it does not own that left the board on its move, less a piece it dropped, and the
other hand not at all (a declared test oracle, rule 3). `make` is checked through `apply`,
`unmake` through `perft` (which makes and unmakes on one position) against an `apply`-built
count, and through searching games replayed. Hand-built specs, not chess.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_DROP, MOVE_SWEEP, move_to, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.tournament.play import play_game

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
SWEEP = {"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}
STEP = {"kind": "leap", "vectors": [[0, 1]]}


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _spec(**drops: Any) -> dict[str, Any]:
    """6x6 under royal capture: kings, a sweeper `S` (a rook's lines), guards `G`, a stepper
    `W` that promotes to `X` (`promoted_from` `W`, so a taken `X` is held as a `W`), and a
    hand that fills from captures."""
    return copy.deepcopy({
        "schema_version": 31, "name": "the dustpan", "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "S", "value": 5, "moves": [SWEEP]},
            {"id": "G", "value": 2, "moves": [KING]},
            {"id": "W", "value": 1, "moves": [STEP]},
            {"id": "X", "value": 3, "moves": [KING], "promoted_from": "W"},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]}, {"piece": "S", "squares": ["f1"]},
            {"piece": "G", "squares": ["b1", "c1"]},
            {"piece": "W", "squares": ["b2", "c2", "d2", "e2"]},
        ]},
        "rules": {
            "win": "royal_capture", "stalemate": "draw", "max_plies": 80,
            "promotion": {"piece": "W", "rank": "last", "to": ["X"]},
            "drops": {"from_captures": True, **drops},
        },
    })


def _position(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0,
              hands: tuple[tuple[int, ...], tuple[int, ...]] | None = None) -> GameState:
    # `make_state` leaves the hands empty; a hand game's position holds one slot per type.
    state = engine.make_state(placement, side_to_move=side, moved=list(placement))
    return dataclasses.replace(state, hands=hands or engine.initial_state().hands)


def _move(engine: Engine, state: GameState, origin: str, target: str) -> int:
    for move in engine.legal_moves(state):
        if (move & 0xFF, (move >> 8) & 0xFF) == (_sq(origin), _sq(target)):
            return move
    raise AssertionError(f"no move {origin}-{target}")


def _type(engine: Engine, pid: str) -> int:
    return engine.spec.piece_ids.index(pid)


def _expected_hands(
    engine: Engine, before: GameState, move: int, after: GameState,
    unlimited: frozenset[int] = frozenset(),
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The oracle: the mover's hand gains the demoted type of every piece it does not own that
    left the board on its move, and loses the type it dropped; the other hand is unchanged."""
    spec = engine.spec
    mover = before.side_to_move
    def theirs(board: bytes) -> list[int]:
        counts = [0] * len(before.hands[mover])
        for value in board:
            if value and spec.owner_of(value) != mover:
                counts[spec.type_of(value)] += 1
        return counts

    hand = list(before.hands[mover])
    for piece_type, (was, now) in enumerate(zip(theirs(before.board), theirs(after.board), strict=True)):
        slot = spec.demotions[piece_type]
        if slot not in unlimited:
            hand[slot] += was - now
    if move_type(move) == MOVE_DROP:
        dropped = spec.type_of(after.board[move_to(move)])
        if dropped not in unlimited:
            hand[dropped] -= 1
    hands = list(before.hands)
    hands[mover] = tuple(hand)
    return (hands[0], hands[1])


# --------------------------------------------------------------------------- the validator


class TestTheLift:
    def test_a_sweep_beside_a_hand_is_valid(self) -> None:
        assert validate_spec(_spec()) == []
        assert validate_spec(_spec(stock={"G": 1})) == []
        assert validate_spec(_spec(from_captures=False, stock={"G": 2})) == []

    def test_the_other_sweep_refusals_stand(self) -> None:
        spec = _spec()
        spec["rules"]["explode_on_capture"] = {"radius": 1}
        assert "W3" in refused_ids(validate_spec(spec))
        assert "W4" not in refused_ids(validate_spec(spec))

    def test_describe_says_where_the_swept_pieces_go(self) -> None:
        text = describe_spec(_spec())
        assert "A sweep captures every piece it removes" in text
        assert "A sweep captures" not in describe_spec(_spec(from_captures=False, stock={"G": 1}))

    def test_fairy_stockfish_refuses_the_sweep(self) -> None:
        """A sweep has no Betza term, so the pair is refused with the sweep's own reason."""
        with pytest.raises(Untranslatable, match="sweep"):
            translate(_spec())


# --------------------------------------------------------------------------- make


class TestTheDeposit:
    def test_every_swept_enemy_piece_reaches_the_hand_and_no_friendly_one(self) -> None:
        engine = Engine(_spec())
        g, s, w = (_type(engine, pid) for pid in "GSW")
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "a3": ("S", 0),
            "b3": ("G", 1), "c3": ("G", 0), "d3": ("X", 1), "e3": ("S", 1), "f3": ("G", 1),
        })
        move = _move(engine, state, "a3", "f3")
        assert move_type(move) == MOVE_SWEEP
        after = engine.apply(state, move)
        for square in ("b3", "c3", "d3", "e3"):
            assert after.board[_sq(square)] == 0
        hand = [0] * len(state.hands[0])
        hand[g] = 2  # b3 swept, f3 taken on landing; c3, the mover's own, to no hand
        hand[s] = 1  # e3
        hand[w] = 1  # d3's X, held as the W it was promoted from
        assert after.hands == (tuple(hand), tuple([0] * len(hand)))
        assert after.key == engine._compute_hash(after)
        assert after.hands == _expected_hands(engine, state, move, after)

    def test_a_sweep_onto_an_empty_square_deposits_its_path(self) -> None:
        engine = Engine(_spec())
        g = _type(engine, "G")
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "f4": ("S", 1),
            "e4": ("G", 0), "d4": ("G", 0), "c4": ("G", 1),
        }, side=1)
        after = engine.apply(state, _move(engine, state, "f4", "b4"))
        assert after.hands[1][g] == 2 and after.hands[0][g] == 0
        assert after.hands[1][_type(engine, "S")] == 0

    def test_an_unlimited_slot_takes_no_deposit(self) -> None:
        engine = Engine(_spec(stock={"G": "unlimited"}))
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "a3": ("S", 0),
            "b3": ("G", 1), "c3": ("S", 1),
        })
        start = state.hands
        after = engine.apply(state, _move(engine, state, "a3", "d3"))
        s = _type(engine, "S")
        assert after.hands[0][_type(engine, "G")] == start[0][_type(engine, "G")]
        assert after.hands[0][s] == start[0][s] + 1

    def test_a_held_piece_drops_back(self) -> None:
        engine = Engine(_spec())
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "a3": ("S", 0),
            "b3": ("G", 1), "c3": ("G", 1),
        })
        after = engine.apply(state, _move(engine, state, "a3", "d3"))
        after = engine.apply(after, _move(engine, after, "a6", "b6"))
        drops = [m for m in engine.legal_moves(after) if move_type(m) == MOVE_DROP]
        assert drops and all(
            engine.spec.type_of(engine.apply(after, m).board[move_to(m)]) == _type(engine, "G")
            for m in drops
        )

    @pytest.mark.parametrize("seed", range(12))
    def test_random_walks_hold_the_hands_to_the_oracle(self, seed: int) -> None:
        """Every ply of a walk from the start: the hands are the oracle's, the key the full
        scan's. Over the twelve walks, sweeps take both enemy and friendly pieces."""
        engine = Engine(_spec())
        rng = random.Random(seed)
        state = engine.initial_state()
        for _ in range(80):
            moves = engine.legal_moves(state)
            if not moves or engine.terminal(state, moves) is not None:
                break
            sweeps = [m for m in moves if move_type(m) == MOVE_SWEEP]
            move = rng.choice(sweeps if sweeps and rng.random() < 0.5 else moves)
            after = engine.apply(state, move)
            assert after.hands == _expected_hands(engine, state, move, after)
            assert after.key == engine._compute_hash(after)
            state = after

    def test_the_walks_sweep_both_sides(self) -> None:
        """The walks above reach what they are for: sweeps over enemy and over own pieces."""
        engine = Engine(_spec())
        enemy = own = 0
        for seed in range(12):
            rng = random.Random(seed)
            state = engine.initial_state()
            for _ in range(80):
                moves = engine.legal_moves(state)
                if not moves or engine.terminal(state, moves) is not None:
                    break
                sweeps = [m for m in moves if move_type(m) == MOVE_SWEEP]
                move = rng.choice(sweeps if sweeps and rng.random() < 0.5 else moves)
                for _, value in engine.spec.sweep_victims(state.board, move):
                    if engine.spec.owner_of(value) == state.side_to_move:
                        own += 1
                    else:
                        enemy += 1
                state = engine.apply(state, move)
        assert enemy > 0 and own > 0


# --------------------------------------------------------------------------- unmake


def _apply_count(engine: Engine, state: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(
        _apply_count(engine, engine.apply(state, move), depth - 1)
        for move in engine.legal_moves(state)
    )


class TestTheUndo:
    def test_perft_counts_what_apply_counts(self) -> None:
        """`perft` makes and unmakes on one position; a hand left wrong after an unmake offers
        drops that are not there (or loses ones that are) at the next sibling."""
        engine = Engine(_spec())
        g = _type(engine, "G")
        hand = [0] * len(engine.initial_state().hands[0])
        hand[g] = 1
        state = _position(engine, {
            "a1": ("K", 0), "f6": ("K", 1), "a3": ("S", 0), "d6": ("S", 1),
            "b3": ("G", 1), "c3": ("G", 0), "d4": ("X", 1), "d5": ("G", 0), "e3": ("W", 1),
        }, hands=(tuple(hand), tuple(hand)))
        assert engine.perft(state, 3) == _apply_count(engine, state, 3)

    def test_a_searching_game_replays(self) -> None:
        """The core's searching players make and unmake thousands of times a move; a hand
        their unmake left wrong would put a drop in the record that `apply` refuses, or end
        the game elsewhere."""
        engine = Engine(_spec())
        for seed in range(3):
            record = play_game(engine, ("minimax:nodes=200", "minimax:nodes=200"), seed)
            state = engine.initial_state()
            for move in record.moves:
                assert move in engine.legal_moves(state)
                state = engine.apply(state, move)
            outcome = engine.terminal(state)
            assert outcome is not None
            assert (outcome.winner, outcome.reason) == (record.winner, record.reason)
