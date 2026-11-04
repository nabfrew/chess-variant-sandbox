"""SR4l (`docs/plan-schema-next.md` §2.11 row O22): the purse beside a blast.

The ruling (2026-10-28): who is paid for a blast's victims is the mover, for every enemy
victim's bounty; nothing is paid for the mover's own victims, and a wild victim's bounty is
T8's (SR4f).

The probe (fed past the validator on SR4k's build `76249fcd5d80f769`): a rook taking a guard
(bounty one) whose blast destroyed an enemy rook (four), an enemy guard (one) and its own guard
(one) was paid one, the taken guard's, into the points and the purse alike, with or without a
points count: the blast's log paid only wild pieces (`pay_swept`'s `wild_only`, SR4f). So the
core pays every piece on the blast's log as taking it would (`Tables::bounty_paid`: an enemy its
type's bounty, a wild piece its route's, the mover's own nothing, the capturer among them), to
`bounty_payee`, into the points and under a purse the purse, capped; `unmake` reads the points
back off the same log and the purse off its own. One payment, so a points count without a purse
is paid the same (SR4f's test of a guard in a blast paying nothing was SR4l's question), and in
rounds each captor whose blast reached a piece is paid for it.

The oracle is this file's `_lost`: the bounties of the pieces the mover's opponents lost on the
move, which is every capture and every blast victim of theirs. Hand-built 6x6 specs, not chess.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import MOVE_BUY, move_from, move_to, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import PURSE_MAX, validate_spec
from chess_variant_sandbox.tournament.play import play_game
from tests.feature_inventory import _schema_seventy_six_uses

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
PRICE = 6


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _spec(purse: bool = True, **rules: Any) -> dict[str, Any]:
    """Kings a1/a6, a rook (bounty 4) f1/f6, guards (bounty 1) c1-e1/c6-e6, a scout for sale at
    six; atomic's blast (a king's step, the kings immune); a clock of 60 plies counted in
    points."""
    spec: dict[str, Any] = {
        "name": "the blast that pays (SR4l test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK], "bounty": 4},
            {"id": "G", "value": 2, "moves": [KING], "bounty": 1},
            {"id": "S", "value": 1, "moves": [KING], "price": PRICE},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "R", "squares": ["f1"]},
            {"piece": "G", "squares": ["c1", "d1", "e1"]},
        ]},
        "rules": {
            "win": "royal_capture", "stalemate": "draw", "max_plies": 60,
            "scoring": {"count": "points"},
            "explode_on_capture": {"radius": 1, "immune": ["K"]},
            **rules,
        },
    }
    if purse:
        spec["rules"]["purse"] = {"start": 0, "market": ["S"]}
    else:
        spec["pieces"][3].pop("price")
    return spec


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


def _move(engine: Engine, state: GameState, origin: str, target: str) -> int:
    for move in engine.legal_moves(state):
        if (move_from(move), move_to(move)) == (_sq(origin), _sq(target)):
            return move
    raise AssertionError(f"no move {origin}-{target}")


def _lost(engine: Engine, before: GameState, after: GameState) -> int:
    """The oracle: the bounties of the pieces the mover's opponent lost on this move."""
    spec = engine.spec
    mover = before.side_to_move

    def held(board: bytes) -> int:
        return sum(
            spec.bounty[spec.type_of(value)] for value in board
            if value and spec.owner_of(value) != mover
        )

    return int(held(before.board) - held(after.board))


# The blast of the rook on f3 taking the guard on d3: an enemy rook on d4 and guard on c3, the
# mover's own guard on c2. Paid: one (the guard taken) + four + one; the own guard nothing.
BLAST = {
    "a1": ("K", 0), "a6": ("K", 1), "f3": ("R", 0), "d3": ("G", 1),
    "d4": ("R", 1), "c3": ("G", 1), "c2": ("G", 0),
}


def _mirror(placement: dict[str, tuple[str, int]]) -> dict[str, tuple[str, int]]:
    return {f"{sq[0]}{WIDTH + 1 - int(sq[1:])}": (pid, 1 - p) for sq, (pid, p) in placement.items()}


def _state(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0,
           purse: tuple[int, int] | None = (0, 0)) -> GameState:
    return engine.make_state(placement, side_to_move=side, moved=list(placement), purse=purse)


class TestTheProbe:
    """The first tests, as the box asks: what a blast pays, before and after."""

    @pytest.mark.parametrize("side", [0, 1])
    def test_two_priced_enemy_pieces_in_the_blast_pay_both(self, side: int) -> None:
        engine = _engine(_spec())
        placement = BLAST if side == 0 else _mirror(BLAST)
        state = _state(engine, placement, side)
        origin, target = ("f3", "d3") if side == 0 else ("f4", "d4")
        after = engine.apply(state, _move(engine, state, origin, target))
        for square in ("d4", "c3", "c2") if side == 0 else ("d3", "c4", "c5"):
            assert after.board[_sq(square)] == 0
        paid = [0.0, 0.0]
        paid[side] = 6.0
        assert after.points == tuple(paid)
        coins = [0, 0]
        coins[side] = 6
        assert after.purse == tuple(coins)
        assert after.key == engine._compute_hash(after)

    def test_without_a_purse_the_points_are_paid_alike(self) -> None:
        engine = _engine(_spec(purse=False))
        state = _state(engine, BLAST, purse=None)
        after = engine.apply(state, _move(engine, state, "f3", "d3"))
        assert after.points == (6.0, 0.0)

    def test_the_movers_own_victims_pay_nothing(self) -> None:
        engine = _engine(_spec())
        state = _state(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "f3": ("R", 0), "d3": ("G", 1),
            "d4": ("R", 0), "c2": ("G", 0),
        })
        after = engine.apply(state, _move(engine, state, "f3", "d3"))
        assert after.board[_sq("d4")] == after.board[_sq("c2")] == 0
        assert after.points == (1.0, 0.0)
        assert after.purse == (1, 0)

    def test_a_purse_at_its_cap_loses_the_rest(self) -> None:
        engine = _engine(_spec())
        state = _state(engine, BLAST, purse=(PURSE_MAX - 2, 0))
        after = engine.apply(state, _move(engine, state, "f3", "d3"))
        assert after.purse == (PURSE_MAX, 0)
        assert after.points == (6.0, 0.0)
        assert after.key == engine._compute_hash(after)

    def test_to_the_loser_the_victims_owner_is_paid(self) -> None:
        """Under `bounty_to: "loser"` a blast pays as each capture would: the player who lost
        the pieces, six; nothing for the mover's own guard."""
        spec = _spec()
        spec["rules"]["purse"]["bounty_to"] = "loser"
        engine = _engine(spec)
        state = _state(engine, BLAST)
        after = engine.apply(state, _move(engine, state, "f3", "d3"))
        assert after.points == (0.0, 6.0)
        assert after.purse == (0, 6)


class TestUndo:
    def test_perft_counts_what_apply_counts(self) -> None:
        """Five coins before, the scout at six: a blast that pays makes a buy, so a purse an
        unmake left wrong would add or take buys from the next sibling's tree."""
        engine = _engine(_spec())
        state = _state(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "f3": ("R", 0), "d3": ("G", 1), "d4": ("R", 1),
            "c3": ("G", 1), "e5": ("G", 0), "b5": ("G", 1),
        }, purse=(PRICE - 1, PRICE - 1))

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(count(engine.apply(position, m), depth - 1)
                       for m in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)

    def test_a_searching_game_replays_to_the_same_points(self) -> None:
        engine = _engine(_spec(max_plies=24))
        for seed in range(3):
            record = play_game(engine, ("minimax:nodes=300", "minimax:nodes=300"), seed)
            state = engine.initial_state()
            for move in record.moves:
                assert move in engine.legal_moves(state)
                state = engine.apply(state, move)
            outcome = engine.terminal(state)
            assert outcome is not None
            assert (outcome.winner, outcome.reason) == (record.winner, record.reason)


class TestTheWalks:
    def test_points_and_purse_hold_to_the_oracle(self) -> None:
        """Seeded walks, captures preferred; every ply the mover is paid what its opponent
        lost, into the points and (capped, less a buy) the purse, the other player nothing; the
        key is the full scan's. Over the walks some blast pays for a piece it did not take."""
        engine = _engine(_spec())
        bounty = engine.spec.bounty
        blasts = 0
        for seed in range(6):
            rng = random.Random(seed)
            state = engine.initial_state()
            for _ in range(60):
                moves = engine.legal_moves(state)
                if not moves or engine.terminal(state, moves) is not None:
                    break
                loud = [m for m in moves if state.board[move_to(m)]]
                move = rng.choice(loud if loud and rng.random() < 0.7 else moves)
                after = engine.apply(state, move)
                mover = state.side_to_move
                paid = _lost(engine, state, after)
                taken = state.board[move_to(move)]
                blasts += bool(taken) and paid > bounty[engine.spec.type_of(taken)]
                assert after.points[mover] - state.points[mover] == paid
                assert after.points[1 - mover] == state.points[1 - mover]
                spent = PRICE if move_type(move) == MOVE_BUY else 0
                assert after.purse[mover] == min(state.purse[mover] - spent + paid, PURSE_MAX)
                assert after.purse[1 - mover] == state.purse[1 - mover]
                assert after.key == engine._compute_hash(after)
                state = after
        assert blasts > 0


class TestRounds:
    def test_a_rounds_blast_pays_an_enemy_bounty_to_its_captor(self) -> None:
        """No purse in rounds; the points are paid as in a turn (`rounds.rs::blasts`)."""
        engine = _engine(_spec(purse=False, simultaneous={}))
        state = _state(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "f3": ("R", 0), "d3": ("G", 1), "c3": ("G", 1),
            "d4": ("R", 1),
        }, purse=None)
        first = next(m for m in engine.legal_moves_for(state, 0)
                     if (move_from(m), move_to(m)) == (_sq("f3"), _sq("d3")))
        second = next(m for m in engine.legal_moves_for(state, 1)
                      if (move_from(m), move_to(m)) == (_sq("a6"), _sq("b6")))
        after = engine.apply_round(state, first, second)
        assert after.board[_sq("c3")] == after.board[_sq("d4")] == 0
        assert after.points == (6.0, 0.0)
        assert after.key == engine._compute_hash(after)


class TestTheLift:
    def test_the_pair_is_valid(self) -> None:
        errors = validate_spec(_spec())
        assert errors == []
        assert "O22" not in refused_ids(errors)

    def test_the_inventory_names_it(self) -> None:
        assert _schema_seventy_six_uses(_spec()) == {"rules.purse.explode_on_capture"}
        assert _schema_seventy_six_uses(_spec(purse=False)) == set()

    def test_describe_says_it(self) -> None:
        text = describe_spec(_spec())
        assert "a blast pays the player who made the capture for every piece it destroys" in text

    def test_fairy_stockfish_refuses_the_purse(self) -> None:
        with pytest.raises(Untranslatable):
            translate(_spec())
