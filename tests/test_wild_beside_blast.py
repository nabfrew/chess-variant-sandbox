"""SR4f (`docs/plan-schema-next.md` §2.4 row T8, box SR4f): wild pieces beside a blast.

The ruling (2026-10-27): a wild piece's eat does not explode, and a blast that destroys a wild
piece pays its bounty to the mover.

The probe (fed past the validator on SR4d2's build `3fac65c5dba10136`, `TestTheProbe` below
keeps it): the wild phase's eat already cleared nothing round it -- the phase writes its own
steps and never reaches `write_blast` -- so a rook standing beside the eaten guard stood on. A
player's capture of a wild piece exploded as any capture does, and paid the wolf's bounty
through the ordinary capture. But a blast that destroyed a wolf standing beside the capture
cleared it and paid nobody. So the core change is the payment: `Game::write_payments` pays
each wild piece on the blast's `cleared` log, as taking it would (`pay_swept` with
`wild_only`, `Tables::bounty_paid`), and `unmake` reads the same log back (`refund_swept`); in
rounds `blasts` pays each captor whose blast reached the wild piece. A player's piece in the
blast paid nothing until SR4l (O22, `tests/test_blast_pays.py`), which pays every enemy victim's
type bounty; the specs here carry none but where a test says so.

The oracle is this file's `_paid` (rule 3's declared test oracle): the wild pieces on the
blast's squares, by their route's bounty at their square, and the piece the capture took, as
an ordinary capture pays. `make` is held to it through `apply` on seeded walks; `unmake` through
`perft` against an `apply`-built count, `root_scores` at the clock, and searching games replayed
through `apply`. Hand-built 6x6 specs, not chess.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from typing import Any

import pytest

import chess_variant_sandbox.engine.compile as compile_module
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import move_to
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.loop.operators import OPERATORS
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.tournament.play import play_game
from tests.feature_inventory import _schema_seventy_uses

WIDTH = 6
WILD = 2
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
LOOP = ["b3", "c3", "d3", "e3", "e4", "d4", "c4", "b4"]
WOLF_BOUNTY, BOAR_BOUNTY = 4, 3


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _spec(**rules: Any) -> dict[str, Any]:
    """6x6 counted in points under atomic's blast: kings, a rook and two guards each, a wolf
    on a loop through the middle (bounty four) and a boar on f3-f4 (bounty three)."""
    return copy.deepcopy({
        "schema_version": 31, "name": "the wolf and the bomb",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "N", "name": "guard", "value": 3, "moves": [KING]},
            {"id": "W", "name": "wolf", "value": 3, "moves": [KING]},
            {"id": "B", "name": "boar", "value": 3, "moves": [KING]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]}, {"piece": "R", "squares": ["f1"]},
            {"piece": "N", "squares": ["c2", "d2"]},
        ]},
        "rules": {
            "win": "royal_capture", "stalemate": "draw", "max_plies": 60,
            "scoring": {"count": "points"},
            "npcs": [
                {"piece": "W", "route": list(LOOP), "bounty": WOLF_BOUNTY},
                {"piece": "B", "route": ["f3", "f4"], "bounty": BOAR_BOUNTY},
            ],
            "explode_on_capture": {"radius": 1, "immune": ["K"]},
            **rules,
        },
    })


def _twin(spec: dict[str, Any]) -> dict[str, Any]:
    """The same game without the blast: what an ordinary capture does."""
    twin = copy.deepcopy(spec)
    twin["rules"].pop("explode_on_capture")
    return twin


def _position(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0) -> GameState:
    moved = [square for square, (_, player) in placement.items() if player != WILD]
    return engine.make_state(placement, side_to_move=side, moved=moved)


def _move(engine: Engine, state: GameState, origin: str, target: str) -> int:
    for move in engine.legal_moves(state):
        if (move & 0xFF, (move >> 8) & 0xFF) == (_sq(origin), _sq(target)):
            return move
    raise AssertionError(f"no move {origin}-{target}")


def _byte(engine: Engine, pid: str, player: int) -> int:
    return engine.spec.encode_piece(player, engine.spec.piece_ids.index(pid))


def _paid(engine: Engine, state: GameState, move: int) -> float:
    """The oracle: what the mover's capture pays -- the piece taken as an ordinary capture
    pays it (a wild piece its route's bounty), and, where the blast fires, every wild piece
    on the blast's squares its route's bounty at its square. Atomic's blast fires on every
    capture and spares only the royal, so its squares are the king's step round the target."""
    spec = engine.spec
    assert spec.npcs is not None
    target = move_to(move)
    taken = state.board[target]
    if not taken or spec.owner_of(taken) == state.side_to_move:
        return 0.0
    total = spec.npcs.bounty[target] if spec.owner_of(taken) == WILD else 0.0
    if spec.explosion is None:
        return total
    for square in range(len(state.board)):
        df = abs(square % WIDTH - target % WIDTH)
        dr = abs(square // WIDTH - target // WIDTH)
        value = state.board[square]
        if square != target and max(df, dr) == 1 and value and spec.owner_of(value) == WILD:
            total += spec.npcs.bounty[square]
    return total


# --------------------------------------------------------------------------- the probe


class TestTheProbe:
    """The first tests, as SR4f's box asks: what the core does with the pair."""

    def test_a_wild_eat_clears_nothing_round_it(self) -> None:
        """The wolf on b3 steps to c3 and eats the guard there, with a rook and a guard of the
        same player beside it: both stand, and the board is the twin's without a blast."""
        engine, twin = Engine(_spec()), Engine(_twin(_spec()))
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0), "d2": ("R", 1),
            "c3": ("N", 1), "b2": ("N", 1), "b3": ("W", WILD),
        })
        move = _move(engine, state, "a1", "a2")
        after, steps = engine.apply_with_phase(state, move)
        assert [(s.origin, s.target, s.eaten) for s in steps] == [
            (_sq("b3"), _sq("c3"), _byte(engine, "N", 1))
        ]
        assert after.board[_sq("d2")] == _byte(engine, "R", 1)
        assert after.board[_sq("b2")] == _byte(engine, "N", 1)
        assert after.board == twin.apply(state, move).board
        assert after.points == (0.0, 0.0)
        assert after.key == engine._compute_hash(after)

    def test_a_players_capture_of_a_wild_piece_explodes(self) -> None:
        """The rook takes the wolf: the blast clears the rook and the guard beside, and the
        wolf pays its bounty once, as the capture's."""
        engine = Engine(_spec())
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "f3": ("R", 0), "e3": ("W", WILD), "e4": ("N", 1),
        })
        after = engine.apply(state, _move(engine, state, "f3", "e3"))
        assert after.board[_sq("e3")] == after.board[_sq("e4")] == after.board[_sq("f3")] == 0
        assert after.points == (float(WOLF_BOUNTY), 0.0)
        assert after.key == engine._compute_hash(after)


# --------------------------------------------------------------------------- the payment


class TestThePayment:
    def test_a_blast_that_destroys_a_wild_piece_pays_its_bounty(self) -> None:
        """The rook takes the guard on d3 with the wolf beside it on c3: the blast takes the
        wolf, and the mover is paid its bounty."""
        engine = Engine(_spec())
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "f3": ("R", 0), "d3": ("N", 1), "c3": ("W", WILD),
        })
        after = engine.apply(state, _move(engine, state, "f3", "d3"))
        assert after.board[_sq("c3")] == 0
        assert after.points == (float(WOLF_BOUNTY), 0.0)
        assert after.key == engine._compute_hash(after)

    def test_two_wild_pieces_in_one_blast_pay_both(self) -> None:
        """A capture on e3 whose blast holds the wolf on d3 and the boar on f4: four and three,
        for the second player as for the first, and the other player is paid nothing."""
        engine = Engine(_spec())
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "e1": ("R", 1), "e3": ("N", 0),
            "d3": ("W", WILD), "f4": ("B", WILD),
        }, side=1)
        after = engine.apply(state, _move(engine, state, "e1", "e3"))
        assert after.board[_sq("d3")] == after.board[_sq("f4")] == 0
        assert after.points == (0.0, float(WOLF_BOUNTY + BOAR_BOUNTY))

    def test_a_blast_the_wild_type_is_immune_to_pays_nothing(self) -> None:
        """`explode_on_capture.immune` naming the wolf leaves it standing (it then steps on
        to d3, which the blast emptied), so nothing is paid."""
        engine = Engine(_spec(explode_on_capture={"radius": 1, "immune": ["K", "W"]}))
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "f3": ("R", 0), "d3": ("N", 1), "c3": ("W", WILD),
        })
        after = engine.apply(state, _move(engine, state, "f3", "d3"))
        assert after.board[_sq("d3")] == _byte(engine, "W", WILD)
        assert after.points == (0.0, 0.0)

    def test_a_players_piece_in_the_blast_pays_its_bounty(self) -> None:
        """SR4l (O22) answered it: an enemy guard with a bounty destroyed by the blast pays it,
        as the guard taken pays its own."""
        spec = _spec()
        spec["pieces"][2]["bounty"] = 2
        engine = Engine(spec)
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "f3": ("R", 0), "d3": ("N", 1), "d4": ("N", 1),
        })
        after = engine.apply(state, _move(engine, state, "f3", "d3"))
        assert after.board[_sq("d4")] == 0
        assert after.points == (4.0, 0.0)

    @pytest.mark.parametrize("seed", range(8))
    def test_random_walks_hold_the_points_to_the_oracle(self, seed: int) -> None:
        """Every ply of a walk from the start, captures preferred: the mover is paid what the
        oracle prices, the other player nothing; the key is the full scan's."""
        engine = Engine(_spec())
        rng = random.Random(seed)
        state = engine.initial_state()
        for _ in range(60):
            moves = engine.legal_moves(state)
            if not moves or engine.terminal(state, moves) is not None:
                break
            captures = [m for m in moves if state.board[move_to(m)]]
            move = rng.choice(captures if captures and rng.random() < 0.6 else moves)
            after = engine.apply(state, move)
            mover = state.side_to_move
            assert after.points[mover] == state.points[mover] + _paid(engine, state, move)
            assert after.points[1 - mover] == state.points[1 - mover]
            assert after.key == engine._compute_hash(after)
            state = after

    def test_the_walks_blast_wild_pieces(self) -> None:
        """The walks above reach what they are for: a blast that destroys a wild piece."""
        engine = Engine(_spec())
        reached = 0
        for seed in range(8):
            rng = random.Random(seed)
            state = engine.initial_state()
            for _ in range(60):
                moves = engine.legal_moves(state)
                if not moves or engine.terminal(state, moves) is not None:
                    break
                captures = [m for m in moves if state.board[move_to(m)]]
                move = rng.choice(captures if captures and rng.random() < 0.6 else moves)
                taken = state.board[move_to(move)]
                as_capture = engine.spec.owner_of(taken) == WILD if taken else False
                bounty = engine.spec.npcs.bounty[move_to(move)] if as_capture else 0.0
                reached += _paid(engine, state, move) > bounty
                state = engine.apply(state, move)
        assert reached > 0


# --------------------------------------------------------------------------- unmake


class TestTheRefund:
    PLACEMENT = {
        "a1": ("K", 0), "a6": ("K", 1), "f2": ("R", 0), "a4": ("R", 1),
        "d3": ("N", 1), "c3": ("W", WILD), "f4": ("B", WILD), "e5": ("N", 0),
    }

    def test_perft_counts_what_apply_counts(self) -> None:
        """`perft` makes and unmakes on one position; a blast's board left wrong by an unmake
        changes the next sibling's moves. (Points steer no move, so the next two tests hold
        them.)"""
        engine = Engine(_spec())
        state = _position(engine, self.PLACEMENT)

        def count(position: GameState, depth: int) -> int:
            # No terminal test: perft has none.
            if depth == 0:
                return 1
            return sum(count(engine.apply(position, move), depth - 1)
                       for move in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)

    def test_each_root_move_is_scored_on_its_own_points(self) -> None:
        """A ply before the clock, behind by three, the mover wins only by a blast that takes
        the wolf (four). `root_scores` makes and unmakes every root move on one game, so points
        an unmake left behind would score a later sibling as a win. Each score's sign must be
        the outcome `apply` reaches."""
        engine = Engine(_spec())
        placement = {"a1": ("K", 0), "a6": ("K", 1), "f3": ("R", 0), "d3": ("N", 1),
                     "c3": ("W", WILD), "c5": ("N", 0)}
        state = engine.make_state(placement, side_to_move=0, ply=59,
                                  moved=["a1", "a6", "f3", "d3", "c5"])
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
        blast = moves.index(_move(engine, state, "f3", "d3"))
        assert any(winner == 1 for winner in outcomes[blast + 1:])

    def test_a_searching_game_replays_to_the_same_points(self) -> None:
        """The searching players make and unmake thousands of times a move; points an unmake
        left wrong would end the replayed game otherwise."""
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


# --------------------------------------------------------------------------- rounds


class TestRounds:
    def test_a_rounds_blast_pays_the_wild_piece_to_its_captor(self) -> None:
        """In rounds (`rounds.rs::blasts`) a wild piece a captor's blast destroys pays that
        captor; the other player, moving quietly, is paid nothing."""
        engine = Engine(_spec(simultaneous={}))
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "f3": ("R", 0), "d3": ("N", 1), "c3": ("W", WILD),
        })
        first = next(m for m in engine.legal_moves_for(state, 0)
                     if (m & 0xFF, (m >> 8) & 0xFF) == (_sq("f3"), _sq("d3")))
        second = next(m for m in engine.legal_moves_for(state, 1)
                      if (m & 0xFF, (m >> 8) & 0xFF) == (_sq("a6"), _sq("b6")))
        after = engine.apply_round(state, first, second)
        assert after.board[_sq("c3")] == 0
        assert after.points == (float(WOLF_BOUNTY), 0.0)
        assert after.key == engine._compute_hash(after)


# --------------------------------------------------------------------------- the lift


class TestTheLift:
    def test_the_pair_is_valid(self) -> None:
        errors = validate_spec(_spec())
        assert errors == []
        assert "T8" not in refused_ids(errors)

    def test_the_purse_is_lifted_beside_the_blast(self) -> None:
        """SR4l (O22)."""
        spec = _spec()
        spec["pieces"][2]["price"] = 5
        spec["rules"]["purse"] = {"start": 0, "market": ["N"]}
        assert "O22" not in refused_ids(validate_spec(spec))

    def test_describe_says_it(self) -> None:
        text = describe_spec(_spec())
        assert "A wild piece's eat never explodes" in text
        assert "pays its bounty" in text
        assert "never explodes" not in describe_spec(_twin(_spec()))

    def test_fairy_stockfish_refuses_the_pair(self) -> None:
        """Fairy-Stockfish refuses `npcs` itself, so the pair is refused there by that."""
        with pytest.raises(Untranslatable, match="rules.npcs"):
            translate(compile_module.compile_spec(_spec()))

    def test_the_inventory_names_it(self) -> None:
        assert _schema_seventy_uses(_spec()) == {"rules.npcs.explode_on_capture"}
        assert _schema_seventy_uses(_twin(_spec())) == set()

    @pytest.mark.parametrize("name", ["set-the-blast", "set-a-patrol"])
    def test_the_operators_enter_the_pair(self, name: str) -> None:
        """`set-the-blast` adds a blast to a wolves game, `set-a-patrol` a wild piece to a blast
        game; each no longer declines on the other rule, and a child is valid."""
        operator = next(op for op in OPERATORS if op.name == name)
        parent = _twin(_spec()) if name == "set-the-blast" else _spec()
        if name == "set-a-patrol":
            parent["rules"].pop("npcs")
            parent["rules"].pop("scoring")
        entered = 0
        for seed in range(40):
            child = copy.deepcopy(parent)
            if operator.apply(child, random.Random(seed)) is None:
                continue
            if validate_spec(child) == [] and _schema_seventy_uses(child):
                entered += 1
        assert entered > 0
