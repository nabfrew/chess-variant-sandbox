"""SR4c (`docs/plan-schema-next.md` §2.6 row W5, box SR4c): a bounty per swept wild piece.

The ruling (2026-10-27): beside `rules.npcs` each wild piece a sweep removes pays its bounty to
the mover, independent of the sweep switch. A sweep takes every piece on its path, so the core
pays each as a capture of it pays (`Tables::bounty_paid`, the one statement of a bounty): a wild
piece its route's bounty at the square it stood on, and -- the same statement -- any piece but
the mover's own its type's `pieces[].bounty` (TC4), which a sweep paid only on its landing
square before. `Game::pay_swept` pays off the sweep's `cleared` log in `make`, and
`refund_swept` reads the same log back in `unmake`; under a purse the payments fill it under
its log, capped, as any bounty does.

The oracle is this file's `_paid`: what the mover's move took (the sweep's victims and the
landing's piece) priced as a capture prices it, a wild piece by its route's bounty at its
square, any other piece the mover does not own by its type's bounty (a declared test oracle,
rule 3). `make` is checked through `apply` on every ply of seeded walks; `unmake` through games
the core's searching players play (make and unmake on one position, thousands of times a
move), replayed through `apply` and decided by the same points, and through `perft` against an
`apply`-built count under a purse near its cap. Hand-built 6x6 specs, not chess.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import (
    MOVE_NORMAL,
    MOVE_PROMOTION,
    MOVE_SWEEP,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import PURSE_MAX, validate_spec
from chess_variant_sandbox.tournament.play import play_game

WIDTH = 6
WILD = 2
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
SWEEP = {"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}
WOLF_BOUNTY, BOAR_BOUNTY, GUARD_BOUNTY = 2, 3, 1


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _spec(**rules: Any) -> dict[str, Any]:
    """6x6 counted in points: kings, a sweeper `S` (a rook's lines), guards `G` with a bounty,
    and two wild pieces, a wolf on b3-b4 and a boar on d3-d4, worth two and three."""
    return copy.deepcopy({
        "schema_version": 31, "name": "the beaters", "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "S", "value": 5, "moves": [SWEEP]},
            {"id": "G", "value": 2, "moves": [KING], "bounty": GUARD_BOUNTY},
            {"id": "W", "name": "wolf", "value": 3, "moves": [KING]},
            {"id": "B", "name": "boar", "value": 3, "moves": [KING]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]}, {"piece": "S", "squares": ["f1"]},
            {"piece": "G", "squares": ["b1", "c1"]},
        ]},
        "rules": {
            "win": "royal_capture", "stalemate": "draw", "max_plies": 40,
            "scoring": {"count": "points"},
            "npcs": [
                {"piece": "W", "route": ["b3", "b4"], "bounty": WOLF_BOUNTY},
                {"piece": "B", "route": ["d3", "d4"], "bounty": BOAR_BOUNTY},
            ],
            **rules,
        },
    })


def _purse_spec(start: int, **rules: Any) -> dict[str, Any]:
    """`_spec` with a purse of ``start`` coins each and a market selling the guard."""
    spec = _spec(**rules)
    spec["pieces"][2]["price"] = 5
    spec["rules"]["purse"] = {"start": start, "market": ["G"]}
    return spec


def _position(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0,
              purse: tuple[int, int] | None = None) -> GameState:
    return engine.make_state(placement, side_to_move=side, moved=list(placement), purse=purse)


def _move(engine: Engine, state: GameState, origin: str, target: str) -> int:
    for move in engine.legal_moves(state):
        if (move & 0xFF, (move >> 8) & 0xFF) == (_sq(origin), _sq(target)):
            return move
    raise AssertionError(f"no move {origin}-{target}")


def _paid(engine: Engine, state: GameState, move: int) -> float:
    """The oracle: what the pieces the mover's move takes are worth to the mover -- every
    swept piece and the landing's, a wild one by its route's bounty at its square, any other
    piece but the mover's own by its type's bounty."""
    spec = engine.spec
    mover = state.side_to_move
    kind = move_type(move)
    taken = list(spec.sweep_victims(state.board, move)) if kind == MOVE_SWEEP else []
    target = move_to(move)
    if kind in (MOVE_NORMAL, MOVE_PROMOTION, MOVE_SWEEP) and state.board[target]:
        taken.append((target, state.board[target]))
    total = 0.0
    for square, value in taken:
        owner = spec.owner_of(value)
        if owner == mover:
            continue
        if owner == WILD:
            assert spec.npcs is not None
            total += spec.npcs.bounty[square]
        total += spec.bounty[spec.type_of(value)] if spec.bounty else 0
    return total


# --------------------------------------------------------------------------- the validator


class TestTheLift:
    def test_a_sweep_beside_wild_pieces_is_valid(self) -> None:
        assert validate_spec(_spec()) == []
        assert validate_spec(_purse_spec(0)) == []

    def test_the_other_sweep_refusals_stand(self) -> None:
        spec = _spec()
        spec["rules"]["explode_on_capture"] = {"radius": 1}
        ids = refused_ids(validate_spec(spec))
        assert "W3" in ids and "W5" not in ids

    def test_describe_says_the_sweep_pays(self) -> None:
        text = describe_spec(_spec())
        assert "A sweep takes every piece it removes" in text
        plain = _spec()
        plain["pieces"][1]["moves"] = [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]
        assert "A sweep takes every piece" not in describe_spec(plain)

    def test_fairy_stockfish_refuses_the_pair(self) -> None:
        """Neither a sweep nor a wild piece has a Fairy-Stockfish term."""
        with pytest.raises(Untranslatable):
            translate(_spec())


# --------------------------------------------------------------------------- make


class TestThePayment:
    def test_a_sweep_over_two_wild_pieces_pays_both_bounties(self) -> None:
        """The acceptance: the wolf and the boar, swept on one move, both pay the mover."""
        engine = Engine(_spec())
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "a3": ("S", 0), "b3": ("W", WILD), "d3": ("B", WILD),
        })
        move = _move(engine, state, "a3", "f3")
        assert move_type(move) == MOVE_SWEEP
        after = engine.apply(state, move)
        assert after.board[_sq("b3")] == 0 and after.board[_sq("d3")] == 0
        assert after.points == (WOLF_BOUNTY + BOAR_BOUNTY, 0.0)
        assert after.key == engine._compute_hash(after)

    def test_the_second_player_is_paid_the_same(self) -> None:
        engine = Engine(_spec())
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "f4": ("S", 1), "d4": ("B", WILD), "b4": ("W", WILD),
        }, side=1)
        after = engine.apply(state, _move(engine, state, "f4", "a4"))
        assert after.points == (0.0, WOLF_BOUNTY + BOAR_BOUNTY)

    def test_every_swept_piece_pays_as_its_capture_would(self) -> None:
        """A guard of the enemy's pays its type's bounty swept as taken; the mover's own guard
        pays nothing; the wild piece landed on pays its route's, as any capture of it."""
        engine = Engine(_spec())
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "a3": ("S", 0),
            "b3": ("W", WILD), "c3": ("G", 1), "d3": ("B", WILD), "e3": ("G", 0),
        })
        move = _move(engine, state, "a3", "f3")
        after = engine.apply(state, move)
        assert after.points == (WOLF_BOUNTY + GUARD_BOUNTY + BOAR_BOUNTY, 0.0)
        assert after.points[0] == _paid(engine, state, move)
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "a3": ("S", 0), "b3": ("W", WILD), "d3": ("B", WILD),
        })
        move = _move(engine, state, "a3", "d3")
        assert engine.apply(state, move).points == (WOLF_BOUNTY + BOAR_BOUNTY, 0.0)

    def test_the_purse_is_paid_and_capped(self) -> None:
        engine = Engine(_purse_spec(0))
        placement = {"a1": ("K", 0), "a6": ("K", 1), "a3": ("S", 0),
                     "b3": ("W", WILD), "d3": ("B", WILD)}
        state = _position(engine, placement, purse=(0, 0))
        after = engine.apply(state, _move(engine, state, "a3", "f3"))
        assert after.purse == (WOLF_BOUNTY + BOAR_BOUNTY, 0)
        assert after.key == engine._compute_hash(after)
        state = _position(engine, placement, purse=(PURSE_MAX - 1, 0))
        after = engine.apply(state, _move(engine, state, "a3", "f3"))
        assert after.purse == (PURSE_MAX, 0)
        assert after.points == (WOLF_BOUNTY + BOAR_BOUNTY, 0.0)

    @pytest.mark.parametrize("seed", range(10))
    def test_random_walks_hold_the_points_to_the_oracle(self, seed: int) -> None:
        """Every ply of a walk from the start, sweeps preferred: the mover is paid what the
        oracle prices, the other player nothing; the key is the full scan's."""
        engine = Engine(_spec(max_plies=80))
        rng = random.Random(seed)
        state = engine.initial_state()
        for _ in range(80):
            moves = engine.legal_moves(state)
            if not moves or engine.terminal(state, moves) is not None:
                break
            sweeps = [m for m in moves if move_type(m) == MOVE_SWEEP]
            move = rng.choice(sweeps if sweeps and rng.random() < 0.6 else moves)
            after = engine.apply(state, move)
            mover = state.side_to_move
            assert after.points[mover] == state.points[mover] + _paid(engine, state, move)
            assert after.points[1 - mover] == state.points[1 - mover]
            assert after.key == engine._compute_hash(after)
            state = after

    def test_the_walks_sweep_wild_pieces(self) -> None:
        """The walks above reach what they are for: sweeps over wild pieces, and sweeps over
        two or more pieces at once."""
        engine = Engine(_spec(max_plies=80))
        wild = several = 0
        for seed in range(10):
            rng = random.Random(seed)
            state = engine.initial_state()
            for _ in range(80):
                moves = engine.legal_moves(state)
                if not moves or engine.terminal(state, moves) is not None:
                    break
                sweeps = [m for m in moves if move_type(m) == MOVE_SWEEP]
                move = rng.choice(sweeps if sweeps and rng.random() < 0.6 else moves)
                victims = engine.spec.sweep_victims(state.board, move)
                wild += sum(engine.spec.owner_of(value) == WILD for _, value in victims)
                several += len(victims) >= 2
                state = engine.apply(state, move)
        assert wild > 0 and several > 0


# --------------------------------------------------------------------------- unmake


class TestTheRefund:
    def test_perft_counts_what_apply_counts_under_a_purse(self) -> None:
        """`perft` makes and unmakes on one position; near the cap a purse left wrong after an
        unmake offers buys that are not there (or loses ones that are) at the next sibling."""
        engine = Engine(_purse_spec(0))
        state = _position(engine, {
            "a1": ("K", 0), "f6": ("K", 1), "a3": ("S", 0), "f4": ("S", 1),
            "b3": ("W", WILD), "d3": ("B", WILD), "c3": ("G", 1),
        }, purse=(3, 4))

        def count(position: GameState, depth: int) -> int:
            # No terminal test: perft has none, so a count that stopped at a taken royal
            # would differ from it for that reason alone.
            if depth == 0:
                return 1
            return sum(count(engine.apply(position, move), depth - 1)
                       for move in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)

    def test_each_root_move_is_scored_on_its_own_points(self) -> None:
        """The one place the points steer a search: an ending at the clock, scored by them.
        A ply before the clock, behind by three, the mover wins only by sweeping both wild
        pieces (five). `root_scores` makes and unmakes every root move on one game, the guard's
        quiet moves after the sweeps, so points a sweep's unmake left behind would score
        them as wins. Each score's sign must be the outcome `apply` reaches."""
        engine = Engine(_spec(max_plies=40))
        placement = {"a1": ("K", 0), "f6": ("K", 1), "a3": ("S", 0),
                     "b3": ("W", WILD), "d3": ("B", WILD), "c5": ("G", 0)}
        state = engine.make_state(placement, side_to_move=0, ply=39, moved=list(placement))
        state = dataclasses.replace(state, points=(0.0, 3.0))
        values = [float(piece["value"]) for piece in _spec()["pieces"]]
        moves = engine.legal_moves(state)
        scores = engine.root_scores(state, values, 2000, 1)
        outcomes = []
        for move, score in zip(moves, scores, strict=True):
            outcome = engine.terminal(engine.apply(state, move))
            assert outcome is not None and outcome.reason == "scored"
            assert (score > 0) == (outcome.winner == 0), engine.describe_move(state, move)
            outcomes.append(outcome.winner)
        assert 0 in outcomes and 1 in outcomes
        last_sweep = max(i for i, m in enumerate(moves) if move_type(m) == MOVE_SWEEP)
        assert any(winner == 1 for winner in outcomes[last_sweep + 1:])

    def test_a_searching_game_replays_to_the_same_points(self) -> None:
        """The core's searching players make and unmake thousands of times a move; points an
        unmake left wrong would steer the search off the record's points, and the game
        decided by them would end otherwise in a replay through `apply`."""
        engine = Engine(_spec(max_plies=24))
        for seed in range(3):
            record = play_game(engine, ("minimax:nodes=300", "minimax:nodes=300"), seed)
            state = engine.initial_state()
            for move in record.moves:
                assert move in engine.legal_moves(state)
                state = engine.apply(state, move)
            outcome = engine.terminal(state)
            assert outcome is not None
            assert (outcome.winner, outcome.reason) == (record.winner, record.reason)
