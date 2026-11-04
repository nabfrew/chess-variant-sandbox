"""SR4i (`docs/plan-schema-next.md` §2.4 row T9, box SR4i): an own contact is no capture for
`must_capture`.

The ruling (2026-10-27): `must_capture` is about the enemy, so moving onto your own piece along
an own contact link (`contacts.links[].own`) is no capture. Beside `must_capture` a merge is
offered only where no enemy capture is, and never compelled.

**The probe** (the pair fed past the validator on SR4h's build `b1351100c699f145`, the position of
`tests/test_readings_without_core.py`): with nothing enemy to take, the merge was the only move
(`['Pb2xa3/P->Q']`), and beside a real capture it was offered with it. `restrict`'s `takes`
counted any occupied target under `MOVE_NORMAL` as a capture; it now asks that the target is not
the side to move's.

The oracle is this file's: the pair's legal list is its twin's without `must_capture`, narrowed
to the moves that take a piece the side to move does not own where there is one (a declared
test oracle, rule 3); each move leaves the board the twin leaves. `make` is checked through
`apply`, `unmake` through `perft` against an `apply`-built count and through searching games
replayed. Hand-built specs, not chess.
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
from tests.feature_inventory import _schema_seventy_three_uses

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
KNIGHT = {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}

# A knight that moves onto its own guard merges with it into a rook, and so does a guard
# that moves onto its own knight.
MERGES = {"position": "advance", "immune": ["K"], "links": [
    {"mover": "N", "onto": "G", "own": True, "mover_to": "R"},
    {"mover": "G", "onto": "N", "own": True, "mover_to": "R"}]}


def _spec(must_capture: bool = True, win: str = "royal_capture") -> dict[str, Any]:
    """6x6: kings, rooks, knights and guards `G` (a king's step), the merges, and
    ``must_capture``."""
    spec: dict[str, Any] = {
        "schema_version": 31, "name": "the mergers", "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "N", "value": 3, "moves": [KNIGHT]},
            {"id": "G", "value": 2, "moves": [KING]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["c1"]}, {"piece": "R", "squares": ["a1", "f1"]},
            {"piece": "N", "squares": ["b1", "e1"]},
            {"piece": "G", "squares": ["a2", "b2", "d2", "e2", "f2"]},
        ]},
        "rules": {"win": win, "stalemate": "draw", "max_plies": 80,
                  "contacts": copy.deepcopy(MERGES)},
    }
    if must_capture:
        spec["rules"]["must_capture"] = True
    return spec


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _as(engine: Engine, state: GameState) -> GameState:
    return dataclasses.replace(state, key=engine._compute_hash(state))


def _takes_enemy(engine: Engine, state: GameState, move: int) -> bool:
    held = state.board[(move >> 8) & 0xFF]
    return bool(held) and engine.spec.owner_of(held) != state.side_to_move


def _merges(engine: Engine, state: GameState, move: int) -> bool:
    held = state.board[(move >> 8) & 0xFF]
    return bool(held) and engine.spec.owner_of(held) == state.side_to_move


def _expected(both: Engine, alone: Engine, state: GameState) -> list[int]:
    """The oracle: the twin's list, narrowed to the enemy captures where there is one."""
    moves = alone.legal_moves(_as(alone, state))
    taking = [m for m in moves if _takes_enemy(alone, state, m)]
    return sorted(taking or moves)


# --------------------------------------------------------------------------- the validator


class TestTheLift:
    @pytest.mark.parametrize("win", ["royal_capture", "checkmate"])
    def test_the_pair_validates(self, win: str) -> None:
        assert validate_spec(_spec(win=win)) == []

    def test_the_inventory_names_the_pair(self) -> None:
        assert _schema_seventy_three_uses(_spec()) == {"rules.must_capture.contacts.own"}
        assert _schema_seventy_three_uses(_spec(must_capture=False)) == set()
        enemy_only = _spec()
        enemy_only["rules"]["contacts"]["links"] = [{"mover": "N", "onto": "G", "onto_to": "N"}]
        enemy_only["rules"]["contacts"]["position"] = "stay"
        assert _schema_seventy_three_uses(enemy_only) == set()

    def test_describe_says_a_merge_is_no_capture(self) -> None:
        assert "Moving onto your own piece is not a capture" in describe_spec(_spec())
        assert "Moving onto your own piece is not a capture" not in describe_spec(
            _spec(must_capture=False))

    def test_fairy_stockfish_refuses_the_contacts(self) -> None:
        """Fairy-Stockfish has no contact table, so the pair is refused by its contacts."""
        with pytest.raises(Untranslatable, match="contacts"):
            translate(_spec())


# --------------------------------------------------------------------------- the acceptance


class TestTheAcceptance:
    @pytest.mark.parametrize("side", [0, 1])
    def test_with_nothing_to_take_the_merge_is_offered_not_compelled(self, side: int) -> None:
        engine = _engine(_spec())
        other = 1 - side
        placement = {"a1": ("K", side), "c3": ("N", side), "d5": ("G", side),
                     "f6": ("K", other), "f4": ("G", other)}
        state = engine.make_state(placement, side_to_move=side, moved=list(placement))
        texts = engine.move_texts(state)
        assert "Nc3xd5/N->R" in texts, texts
        assert len(texts) > 1 and any("x" not in t for t in texts), texts
        after = engine.apply(state, next(
            m for m in engine.legal_moves(state)
            if (m & 0xFF, (m >> 8) & 0xFF) == (_sq("c3"), _sq("d5"))))
        merged = after.board[_sq("d5")]
        assert engine.spec.type_of(merged) == engine.spec.piece_ids.index("R")
        assert engine.spec.owner_of(merged) == side
        assert after.board[_sq("c3")] == 0
        assert after.key == engine._compute_hash(after)

    @pytest.mark.parametrize("side", [0, 1])
    def test_beside_an_enemy_capture_only_captures_are_offered(self, side: int) -> None:
        engine = _engine(_spec())
        other = 1 - side
        placement = {"a1": ("K", side), "c3": ("N", side), "d5": ("G", side),
                     "f6": ("K", other), "e4": ("G", other)}
        state = engine.make_state(placement, side_to_move=side, moved=list(placement))
        assert sorted(engine.move_texts(state)) == ["Gd5xe4", "Nc3xe4"], engine.move_texts(state)


# --------------------------------------------------------------------------- walks


class TestTheWalks:
    @pytest.mark.parametrize("win", ["royal_capture", "checkmate"])
    def test_the_pair_plays_as_the_oracle_says(self, win: str) -> None:
        """Seeded walks, both seats, leaning on merges: every legal list is the oracle's and
        each move leaves the twin's board and the full scan's key."""
        both, alone = _engine(_spec(win=win)), _engine(_spec(must_capture=False, win=win))
        offered = {0: 0, 1: 0}
        for seed in range(16):
            rng = random.Random(seed)
            state = both.initial_state()
            for _ in range(60):
                moves = both.legal_moves(state)
                if both.terminal(state, moves) is not None:
                    break
                assert sorted(moves) == _expected(both, alone, state), (seed, state.ply)
                merges = [m for m in moves if _merges(both, state, m)]
                if merges:
                    offered[state.side_to_move] += 1
                    assert not any(_takes_enemy(both, state, m) for m in moves)
                move = rng.choice(merges) if merges and rng.random() < 0.6 else rng.choice(moves)
                after = both.apply(state, move)
                assert after.board == alone.apply(_as(alone, state), move).board, (seed, state.ply)
                assert after.key == both._compute_hash(after), (seed, state.ply)
                state = after
        assert offered[0] and offered[1], offered


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
        """`perft` makes and unmakes on one position; a merge an unmake left wrong would
        miscount the next sibling, and a narrowing read off a stale board would too."""
        engine = _engine(_spec())
        assert engine.perft(engine.initial_state(), 3) == _apply_count(
            engine, engine.initial_state(), 3)

    def test_a_searching_game_replays(self) -> None:
        engine = _engine(_spec())
        for seed in range(2):
            record = play_game(engine, ("minimax:nodes=200", "minimax:nodes=200"), seed)
            state = engine.initial_state()
            for move in record.moves:
                assert move in engine.legal_moves(state)
                state = engine.apply(state, move)
            outcome = engine.terminal(state)
            assert outcome is not None
            assert (outcome.winner, outcome.reason) == (record.winner, record.reason)
