"""SR5d (`docs/plan-schema-next.md` §2.7 rows V3, V4, box SR5d): a commanded arrival wins and
claims for the piece's owner.

The rulings (2026-10-27): a command (`pieces[].sight` effect `command`) lets a player move an
opponent's piece. Beside `rules.reach` the arrival is a fact about the piece, so a commanded piece
on its owner's goal wins for its owner, and on the commander's goal decides nothing (V3). Beside
`rules.territory` a commanded move claims the square it lands on for the piece's owner (V4).
Settled here on the same principle (the rows say *arrival*): under `reach.on: "leave"` a
commanded piece taken off its owner's goal wins for its owner, and off the commander's decides
nothing; under `reach.safe` the owner's piece must stand where the commander cannot take it.

**The probe** (the pairs fed past the validator on SR5c's build `c4b23b4189582f37`). An arrival
already won for the owner: `terminal_prefix` reads the board, where the piece is its owner's.
A departure did not: `write_goal_left` read the side to move, so a commanded knight leaving the
commander's goal won for the commander, and one leaving its owner's goal decided nothing. A
territory claim did not either: `write_territory` painted the commander's colour. Both now ask
`Game::arrival_owner` (the moving byte's owner as it set out; the mover for a piece nobody owns).

The oracles are this file's: the owner is read off the board byte before the move, the colour a
claim leaves is the mode's (declared test oracles, rule 3). `unmake` is checked through `perft`
against an `apply`-built count and through searching games replayed. Hand-built specs, not chess.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_FOREIGN, move_from, move_to
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.tournament.play import play_game
from tests.feature_inventory import _schema_eighty_one_uses

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
KNIGHT = {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}
# The preacher commands the first enemy piece on each of its rook lines.
PREACH = {"vectors": [[1, 0]], "symmetry": "rotate4", "effect": "command", "whose": "enemy"}

# The knights' goals: the first player's e3 and d3, the second's (mirrored) e4 and d4.
ARRIVE = {"piece": "N", "squares": ["e3", "d3"], "mirror": True}
SAFE = {**ARRIVE, "safe": True}
# Leaving: the second player's goal is c5 (the knight's square in the position below) ...
LEAVE_OWN = {"piece": "N", "squares": ["c2"], "mirror": True, "on": "leave"}
# ... or c5 is the first player's.
LEAVE_THEIRS = {"piece": "N", "squares": ["c5"], "mirror": True, "on": "leave"}
PERMANENT = {"mode": "permanent", "win_at": 30}
REPAINT = {"mode": "repaint", "win_at": 30}

READINGS: dict[str, dict[str, Any]] = {
    "arrive": {"reach": ARRIVE},
    "safe": {"reach": SAFE},
    "leave": {"reach": LEAVE_OWN},
    "permanent": {"territory": PERMANENT},
    "repaint": {"territory": REPAINT},
}


def _spec(preach: bool = True, **rules: Any) -> dict[str, Any]:
    """6x6 under checkmate: kings, a preacher `M` (a king's step, commanding along its rook
    lines), rooks, knights and guards `G`; ``rules`` added."""
    spec: dict[str, Any] = {
        "schema_version": 31, "name": "the preachers", "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "M", "value": 3, "moves": [KING], **({"sight": [PREACH]} if preach else {})},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "N", "value": 3, "moves": [KNIGHT]},
            {"id": "G", "value": 2, "moves": [KING]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["c1"]}, {"piece": "M", "squares": ["d1"]},
            {"piece": "R", "squares": ["a1", "f1"]}, {"piece": "N", "squares": ["b1", "e1"]},
            {"piece": "G", "squares": ["b2", "d2", "e2"]},
        ]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 80},
    }
    spec["rules"].update(copy.deepcopy(rules))
    return spec


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _flip(name: str, side: int) -> str:
    """``name`` as the given side sees it: the second player's position is the first's
    mirrored by rank."""
    return name if side == 0 else f"{name[0]}{WIDTH + 1 - int(name[1:])}"


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _position(engine: Engine, side: int, **extra: tuple[str, int]) -> GameState:
    """``side`` to move, its preacher on c3 commanding the opponent's knight on c5 (as the first
    player sees it; the second player's is mirrored by rank)."""
    placement = {"a1": ("K", 0), "c3": ("M", 0), "f6": ("K", 1), "c5": ("N", 1), **extra}
    mirrored = {_flip(sq, side): (pid, owner ^ side) for sq, (pid, owner) in placement.items()}
    return engine.make_state(mirrored, side_to_move=side, moved=list(mirrored))


def _commanded(engine: Engine, state: GameState, side: int, target: str) -> int:
    origin, target = _sq(_flip("c5", side)), _sq(_flip(target, side))
    found = [m for m in engine.legal_moves(state)
             if move_from(m) == origin and move_to(m) == target]
    assert len(found) == 1, engine.move_texts(state)
    assert found[0] & MOVE_FOREIGN
    return found[0]


def _owner(engine: Engine, board: bytes, square: int) -> int:
    return engine.spec.owner_of(board[square])


# --------------------------------------------------------------------------- the validator


class TestTheLift:
    @pytest.mark.parametrize("rules", READINGS.values(), ids=READINGS.keys())
    def test_each_pair_validates(self, rules: dict[str, Any]) -> None:
        assert validate_spec(_spec(**rules)) == []

    def test_the_inventory_names_each_pair(self) -> None:
        assert _schema_eighty_one_uses(_spec(reach=ARRIVE)) == {"pieces.sight.command.reach"}
        assert _schema_eighty_one_uses(_spec(territory=PERMANENT)) == {
            "pieces.sight.command.territory"}
        assert _schema_eighty_one_uses(_spec(preach=False, reach=ARRIVE, territory=REPAINT)) == set()

    def test_describe_says_whose_arrival_it_is(self) -> None:
        reach = describe_spec(_spec(reach=ARRIVE))
        assert "A commanded piece arrives for its owner" in reach
        assert "claims the square it lands on for its owner" not in reach
        claim = describe_spec(_spec(territory=PERMANENT))
        assert "A commanded piece claims the square it lands on for its owner." in claim
        assert "arrives for its owner" not in describe_spec(_spec(preach=False, reach=ARRIVE))

    def test_fairy_stockfish_refuses_the_command(self) -> None:
        """Fairy-Stockfish moves no opponent's piece, so a command is refused as a sight."""
        for rules in READINGS.values():
            with pytest.raises(Untranslatable, match="sight"):
                translate(_spec(**rules))


# --------------------------------------------------------------------------- make


class TestTheAcceptance:
    @pytest.mark.parametrize("side", [0, 1])
    def test_a_commanded_arrival_on_its_owners_goal_wins_for_its_owner(self, side: int) -> None:
        engine = _engine(_spec(reach=ARRIVE))
        state = _position(engine, side)
        won = engine.apply(state, _commanded(engine, state, side, "e4"))
        outcome = engine.terminal(won)
        assert outcome is not None and (outcome.winner, outcome.reason) == (1 - side, "reached")
        # On the commander's own goal the piece is still the opponent's: nothing is decided.
        assert engine.terminal(engine.apply(state, _commanded(engine, state, side, "d3"))) is None

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_safe_goal_reads_the_commanders_attack(self, side: int) -> None:
        """Under `safe` the owner's piece must stand where the commander cannot take it: a
        commander's knight on f2 guards e4, so the commanded arrival there decides nothing."""
        guard = {"f2": ("N", 0)}
        engine = _engine(_spec(reach=SAFE))
        state = _position(engine, side, **guard)
        assert engine.terminal(engine.apply(state, _commanded(engine, state, side, "e4"))) is None
        engine = _engine(_spec(reach=ARRIVE))
        state = _position(engine, side, **guard)
        outcome = engine.terminal(engine.apply(state, _commanded(engine, state, side, "e4")))
        assert outcome is not None and outcome.winner == 1 - side

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_commanded_departure_decides_for_its_owner(self, side: int) -> None:
        """The probe's finding: the departure read the side to move."""
        own = _engine(_spec(reach=LEAVE_OWN))
        state = _position(own, side)
        for target in ("e4", "d3", "a4"):
            outcome = own.terminal(own.apply(state, _commanded(own, state, side, target)))
            assert outcome is not None and (outcome.winner, outcome.reason) == (1 - side, "reached")
        theirs = _engine(_spec(reach=LEAVE_THEIRS))
        state = _position(theirs, side)
        for target in ("e4", "d3", "a4"):
            assert theirs.terminal(theirs.apply(state, _commanded(theirs, state, side, target))) is None

    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("territory", [PERMANENT, REPAINT], ids=["permanent", "repaint"])
    def test_a_commanded_move_claims_for_its_owner(
        self, territory: dict[str, Any], side: int
    ) -> None:
        engine = _engine(_spec(territory=territory))
        state = _position(engine, side)
        for target in ("e4", "d3", "a4"):
            after = engine.apply(state, _commanded(engine, state, side, target))
            square = _sq(_flip(target, side))
            assert after.colour[square] == (1 - side) + 1
            assert sum(1 for c in after.colour if c) == 1
            assert after.key == engine._compute_hash(after)

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_repainting_command_takes_the_commanders_square(self, side: int) -> None:
        engine = _engine(_spec(territory=REPAINT))
        state = _position(engine, side)
        colour = bytearray(state.colour)
        colour[_sq(_flip("e4", side))] = side + 1
        state = dataclasses.replace(state, colour=bytes(colour))
        state = dataclasses.replace(state, key=engine._compute_hash(state))
        after = engine.apply(state, _commanded(engine, state, side, "e4"))
        assert after.colour[_sq(_flip("e4", side))] == (1 - side) + 1
        assert after.key == engine._compute_hash(after)

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_commanded_claim_can_win_for_its_owner(self, side: int) -> None:
        engine = _engine(_spec(territory={"mode": "permanent", "win_at": 3}))
        state = _position(engine, side)
        colour = bytearray(state.colour)
        for name in ("a6", "b6"):
            colour[_sq(_flip(name, side))] = (1 - side) + 1
        state = dataclasses.replace(state, colour=bytes(colour))
        state = dataclasses.replace(state, key=engine._compute_hash(state))
        outcome = engine.terminal(engine.apply(state, _commanded(engine, state, side, "e4")))
        assert outcome is not None and (outcome.winner, outcome.reason) == (1 - side, "coloured")


# --------------------------------------------------------------------------- walks


class TestTheWalks:
    @pytest.mark.parametrize("territory", [PERMANENT, REPAINT], ids=["permanent", "repaint"])
    def test_every_claim_is_its_owners(self, territory: dict[str, Any]) -> None:
        """Seeded walks leaning on commanded moves: each move colours its landing for the piece
        that moved (the board before the move), as the mode allows; the key is the full scan's."""
        engine = _engine(_spec(territory=territory))
        commanded = {0: 0, 1: 0}
        for seed in range(10):
            rng = random.Random(seed)
            state = engine.initial_state()
            for _ in range(60):
                moves = engine.legal_moves(state)
                if engine.terminal(state, moves) is not None:
                    break
                foreign = [m for m in moves if m & MOVE_FOREIGN]
                move = rng.choice(foreign) if foreign and rng.random() < 0.7 else rng.choice(moves)
                owner = _owner(engine, state.board, move_from(move))
                after = engine.apply(state, move)
                expected = bytearray(state.colour)
                target = move_to(move)
                if territory["mode"] == "repaint" or expected[target] == 0:
                    expected[target] = owner + 1
                assert after.colour == bytes(expected), (seed, state.ply)
                assert after.key == engine._compute_hash(after), (seed, state.ply)
                if move & MOVE_FOREIGN:
                    assert owner == 1 - state.side_to_move
                    commanded[state.side_to_move] += 1
                state = after
        assert commanded[0] and commanded[1], commanded


# --------------------------------------------------------------------------- unmake


def _apply_count(engine: Engine, state: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(
        _apply_count(engine, engine.apply(state, move), depth - 1)
        for move in engine.legal_moves(state)
    )


class TestTheUndo:
    @pytest.mark.parametrize("rules", READINGS.values(), ids=READINGS.keys())
    def test_perft_counts_what_apply_counts(self, rules: dict[str, Any]) -> None:
        engine = _engine(_spec(**rules))
        for side in (0, 1):
            state = _position(engine, side, f2=("N", 0), b4=("G", 1))
            assert engine.perft(state, 3) == _apply_count(engine, state, 3)

    @pytest.mark.parametrize(
        "rules", [{"reach": LEAVE_OWN}, {"territory": {"mode": "repaint", "win_at": 12}}],
        ids=["leave", "repaint"],
    )
    def test_a_searching_game_replays(self, rules: dict[str, Any]) -> None:
        """The core's searching players make and unmake thousands of times a move; a colour or a
        departure their unmake left wrong would end the game elsewhere than `apply` does."""
        engine = _engine(_spec(**rules))
        for seed in range(2):
            record = play_game(engine, ("minimax:nodes=200", "minimax:nodes=200"), seed)
            state = engine.initial_state()
            for move in record.moves:
                assert move in engine.legal_moves(state)
                state = engine.apply(state, move)
            outcome = engine.terminal(state)
            assert outcome is not None
            assert (outcome.winner, outcome.reason) == (record.winner, record.reason)
