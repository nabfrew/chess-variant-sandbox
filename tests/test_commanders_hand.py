"""SR4h (`docs/plan-schema-next.md` §2.4 rows V5, V6, V9, box SR4h): the commander's hand, and a
commanded capture beside a strike or contacts.

The rulings (2026-10-27): a command (`pieces[].sight` effect `command`) with `captures_own` lets
the side to move make an opponent's piece take a piece of its own side. Beside a hand that fills
from captures the taken piece goes to the *commander's* hand, the player whose move took it (V9).
Beside a strike or contacts the capture is an ordinary removal: the taken piece leaves the board,
the commanded piece lands on its square, nothing is struck, no link is read and no side changes
(V5, V6).

**The probe** (the three pairs fed past the validator on SR4g's build `8f00698f034b1353`). The
hand was already the commander's: `deposit_capture` credits the side to move, so V9 is the
validator's lift alone. Beside every strike (rifle, circe, Andernach) the core panicked
(`strike_writes` read the strike's own-side table, which is empty where no own link exists), and
beside an own contact link it read the link (`~Qe2xb5/Q->R`). So `strike_writes` returns `None`
for a meeting whose two pieces share an owner who is not the side to move: the commanded
own-side capture is an ordinary one. `tests/test_readings_without_core.py` keeps the probe's
position (a strict xfail until this task).

The oracles are this file's: the pair's legal list is the reading's own moves (its game without
the command) and the command's foreign moves (its game without the reading), each applied as its
own game applies it; a hand changes by the demoted types of the pieces the mover does not own that
left the board (declared test oracles, rule 3). `make` is checked through `apply`, `unmake`
through `perft` against an `apply`-built count and through searching games replayed. Hand-built
specs, not chess.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_DROP, MOVE_FOREIGN, move_to, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.tournament.play import play_game
from tests.feature_inventory import _schema_seventy_two_uses

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
KNIGHT = {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}
# The preacher commands the first enemy piece on each of its rook lines, and may make it take
# its own side.
PREACH = {"vectors": [[1, 0]], "symmetry": "rotate4", "effect": "command", "whose": "enemy",
          "captures_own": True}

RIFLE = {"effect": "capture", "position": "stay", "immune": ["K"]}
CIRCE = {"effect": "capture", "position": "home", "immune": ["K"]}
ANDERNACH = {"effect": "capture", "immune": ["K"], "switches_sides": ["M", "R", "N", "G"]}
DEMOTE = {"effect": "demote", "position": "stay", "immune": ["K"],
          "chain": [{"from": "R", "to": "N"}, {"from": "N", "to": "G"}]}
ENEMY_LINKS = {"position": "stay", "immune": ["K"], "links": [
    {"mover": "N", "onto": "G", "onto_to": "N"}, {"mover": "R", "onto": "N", "mover_to": "N"}]}
# The link a commanded rook taking its own guard would fire, were it read.
OWN_LINKS = {"position": "advance", "immune": ["K"], "links": [
    {"mover": "R", "onto": "G", "own": True, "mover_to": "M"}]}
HAND = {"from_captures": True}

READINGS: dict[str, dict[str, Any]] = {
    "rifle": {"strike": RIFLE},
    "circe": {"strike": CIRCE},
    "andernach": {"strike": ANDERNACH},
    "demote": {"strike": DEMOTE},
    "enemy links": {"contacts": ENEMY_LINKS},
    "own links": {"contacts": OWN_LINKS},
    "hand": {"drops": HAND},
    "rifle and hand": {"strike": RIFLE, "drops": HAND},
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


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _with_hands(engine: Engine, state: GameState) -> GameState:
    """`state` as `engine` holds it: its own empty hands (none without a hand), and its key."""
    state = dataclasses.replace(state, hands=engine.initial_state().hands)
    return dataclasses.replace(state, key=engine._compute_hash(state))


def _as(engine: Engine, state: GameState) -> GameState:
    """``state`` as a twin keys it: the twins' boards encode alike; a twin without a hand
    holds none."""
    if not any(engine.initial_state().hands):
        state = dataclasses.replace(state, hands=((), ()))
    return dataclasses.replace(state, key=engine._compute_hash(state))


def _between(engine: Engine, state: GameState, origin: str, target: str) -> int:
    found = [m for m in engine.legal_moves(state)
             if (m & 0xFF, (m >> 8) & 0xFF) == (_sq(origin), _sq(target))]
    assert len(found) == 1, (origin, target, engine.move_texts(state))
    return found[0]


def _expected_hands(
    engine: Engine, before: GameState, move: int, after: GameState
) -> tuple[Any, Any]:
    """The hand oracle: the mover's hand gains the demoted type of every piece it does not own
    that left the board on its move, and loses the type it dropped; the other hand is
    unchanged."""
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
        hand[spec.demotions[piece_type]] += was - now
    if move_type(move) == MOVE_DROP:
        hand[spec.type_of(after.board[move_to(move)])] -= 1
    hands = list(before.hands)
    hands[mover] = tuple(hand)
    return hands[0], hands[1]


# The acceptance position: the preacher on c1 commands the enemy rook on c4 (the first piece
# on its line north), which may take its own guard on e4 or knight on c5.
def _turncoat_position(engine: Engine, side: int) -> GameState:
    other = 1 - side
    placement = {
        "a1": ("K", side), "c1": ("M", side), "f6": ("K", other),
        "c4": ("R", other), "e4": ("G", other), "c5": ("N", other),
    }
    state = engine.make_state(placement, side_to_move=side, moved=list(placement))
    return _with_hands(engine, state)


# --------------------------------------------------------------------------- the validator


class TestTheLift:
    @pytest.mark.parametrize("rules", READINGS.values(), ids=READINGS.keys())
    def test_each_pair_validates(self, rules: dict[str, Any]) -> None:
        assert validate_spec(_spec(**rules)) == []

    def test_the_inventory_names_each_pair(self) -> None:
        assert _schema_seventy_two_uses(_spec(strike=RIFLE)) == {
            "pieces.sight.command.captures_own.strike"}
        assert _schema_seventy_two_uses(_spec(contacts=OWN_LINKS)) == {
            "pieces.sight.command.captures_own.contacts"}
        assert _schema_seventy_two_uses(_spec(drops=HAND)) == {
            "pieces.sight.command.captures_own.drops"}
        assert _schema_seventy_two_uses(_spec(preach=False, strike=RIFLE, drops=HAND)) == set()
        assert _schema_seventy_two_uses(
            _spec(drops={"from_captures": False, "stock": {"G": 1}})) == set()

    def test_describe_says_what_the_turncoat_capture_does(self) -> None:
        hand = describe_spec(_spec(drops=HAND))
        assert "goes to the hand of the player who commanded it" in hand
        assert "makes an ordinary capture" not in hand
        struck = describe_spec(_spec(strike=RIFLE))
        assert "makes an ordinary capture" in struck
        assert "makes an ordinary capture" not in describe_spec(_spec(preach=False, strike=RIFLE))

    def test_fairy_stockfish_refuses_the_command(self) -> None:
        """Fairy-Stockfish moves no opponent's piece, so a command is refused as a sight."""
        for rules in READINGS.values():
            with pytest.raises(Untranslatable, match="sight"):
                translate(_spec(**rules))


# --------------------------------------------------------------------------- make


class TestTheAcceptance:
    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("rules", READINGS.values(), ids=READINGS.keys())
    def test_the_commanded_capture_is_an_ordinary_removal(
        self, rules: dict[str, Any], side: int
    ) -> None:
        engine = _engine(_spec(**rules))
        state = _turncoat_position(engine, side)
        for target in ("e4", "c5"):
            move = _between(engine, state, "c4", target)
            assert move & MOVE_FOREIGN
            text = engine.describe_move(state, move)
            assert text == f"~Rc4x{target}", text
            after = engine.apply(state, move)
            expected = bytearray(state.board)
            expected[_sq(target)] = state.board[_sq("c4")]
            expected[_sq("c4")] = 0
            assert after.board == bytes(expected), (target, engine.move_texts(state))
            assert after.key == engine._compute_hash(after)

    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("rules", [{"drops": HAND}, {"strike": RIFLE, "drops": HAND}],
                             ids=["hand", "rifle and hand"])
    def test_the_taken_piece_goes_to_the_commanders_hand(
        self, rules: dict[str, Any], side: int
    ) -> None:
        engine = _engine(_spec(**rules))
        state = _turncoat_position(engine, side)
        for target, pid in (("e4", "G"), ("c5", "N")):
            after = engine.apply(state, _between(engine, state, "c4", target))
            hand = [0] * len(state.hands[side])
            hand[engine.spec.piece_ids.index(pid)] = 1
            assert after.hands[side] == tuple(hand)
            assert after.hands[1 - side] == state.hands[1 - side]
            # The held piece is the commander's to drop.
            again = dataclasses.replace(after, side_to_move=side)
            again = dataclasses.replace(again, key=engine._compute_hash(again))
            drops = [t for t in engine.move_texts(again) if t.startswith(f"{pid}@")]
            assert drops

    def test_a_players_own_contact_still_reads_its_link(self) -> None:
        """The lift reads the side to move: a player's own rook onto its own guard is the
        link's merge, as before."""
        engine = _engine(_spec(contacts=OWN_LINKS))
        placement = {"a1": ("K", 0), "c4": ("R", 0), "e4": ("G", 0), "f6": ("K", 1)}
        state = _with_hands(engine, engine.make_state(placement, side_to_move=0, moved=list(placement)))
        move = _between(engine, state, "c4", "e4")
        assert not move & MOVE_FOREIGN
        assert engine.describe_move(state, move) == "Rc4xe4/R->M"
        after = engine.apply(state, move)
        assert engine.spec.type_of(after.board[_sq("e4")]) == engine.spec.piece_ids.index("M")


# --------------------------------------------------------------------------- walks


def _twins(rules: dict[str, Any]) -> tuple[Engine, Engine, Engine]:
    """The pair, its reading without the command, and the command without its reading."""
    return _engine(_spec(**rules)), _engine(_spec(preach=False, **rules)), _engine(_spec())


class TestTheWalks:
    @pytest.mark.parametrize("rules", READINGS.values(), ids=READINGS.keys())
    def test_the_pair_plays_as_each_rule_alone(self, rules: dict[str, Any]) -> None:
        """Seeded walks, both seats, leaning on commanded captures: the pair's legal list is
        the reading's own moves and the command's foreign moves; each move leaves the board
        its own game leaves, the hands the oracle's, and the key the full scan's."""
        both, alone, command = _twins(rules)
        turncoat = {0: 0, 1: 0}
        for seed in range(12):
            rng = random.Random(seed)
            state = both.initial_state()
            for _ in range(70):
                moves = both.legal_moves(state)
                if both.terminal(state, moves) is not None:
                    break
                own = [m for m in alone.legal_moves(_as(alone, state)) if not m & MOVE_FOREIGN]
                foreign = [m for m in command.legal_moves(_as(command, state)) if m & MOVE_FOREIGN]
                assert sorted(moves) == sorted(own + foreign), (seed, state.ply)
                turncoats = [m for m in moves if m & MOVE_FOREIGN and state.board[(m >> 8) & 0xFF]]
                takes = [m for m in moves if state.board[(m >> 8) & 0xFF]]
                draw = rng.random()
                move = (rng.choice(turncoats) if turncoats and draw < 0.7
                        else rng.choice(takes) if takes and draw < 0.85 else rng.choice(moves))
                after = both.apply(state, move)
                twin = command if move & MOVE_FOREIGN else alone
                assert after.board == twin.apply(_as(twin, state), move).board, (seed, state.ply)
                assert after.key == both._compute_hash(after), (seed, state.ply)
                if any(state.hands):
                    assert after.hands == _expected_hands(both, state, move, after), (seed, state.ply)
                if move in turncoats:
                    turncoat[state.side_to_move] += 1
                state = after
        assert turncoat[0] and turncoat[1], turncoat


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
        """`perft` makes and unmakes on one position; a board, a side or a hand an unmake
        left wrong after a commanded capture miscounts the next sibling."""
        engine = _engine(_spec(**rules))
        state = _turncoat_position(engine, 0)
        assert engine.perft(state, 3) == _apply_count(engine, state, 3)

    @pytest.mark.parametrize("rules", [{"drops": HAND}, {"strike": ANDERNACH}],
                             ids=["hand", "andernach"])
    def test_a_searching_game_replays(self, rules: dict[str, Any]) -> None:
        """The core's searching players make and unmake thousands of times a move; a hand or a
        side their unmake left wrong would put a move in the record `apply` refuses, or end the
        game elsewhere."""
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
