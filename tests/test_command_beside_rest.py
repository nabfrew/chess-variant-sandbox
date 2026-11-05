"""SR5e (`docs/plan-schema-next.md` §2.7 row V7, box SR5e): a command beside `pieces[].rest`.

The ruling (2026-10-27, the user's addition in quotes): a commanded move starts the piece's rest
as its owner's move would, counted on its owner's turns only; *"if it's in a resting state, it
can't be moved by either"*. The lift (a player whose every move is a resting piece's) reads each
player's own offered moves, commands included, and lifts the rest of those pieces for that
player's turn only.

Settled here on the plan's principles (the row says *as its owner's move would*): the entry goes
in the owner's list, so it counts down at the end of each of the owner's turns and is keyed as
the owner's; a resting piece of either list is offered to neither player, whoever's move rested
it; a commanded `captures_own` capture onto the owner's resting piece leaves one entry there, the
mover's afresh; a held piece (nobody's) rests in no list, as before.

**The probe** (the pair fed past the validator with V7's line gone and nothing else, on SR5c2's
build `3d02c0ea4016d635`): a commanded move entered no rest (`rest_after_turn` entered only the
side to move's own piece), so its owner was offered it at once; and a piece resting in its
owner's list was offered to the commander (`rest_filter` read the side to move's list alone).

The oracle is `tests/rest_oracle.py`, widened to the command (the owner's list for a commanded
move; both lists for the filter and the lift), a declared test oracle (rule 3). `unmake` is
checked through `perft` against an `apply`-built count and through searching games replayed.
Hand-built specs, not chess.
"""

from __future__ import annotations

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
from tests import rest_oracle
from tests.feature_inventory import _schema_eighty_three_uses

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
KNIGHT = {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}
# The preacher commands the first enemy piece on each of its rook lines.
PREACH = {"vectors": [[1, 0]], "symmetry": "rotate4", "effect": "command", "whose": "enemy"}


def _spec(
    preach: bool = True, rests: dict[str, int] | None = None, captures_own: bool = False,
) -> dict[str, Any]:
    """6x6 under checkmate: kings, a preacher `M` (a king's step, commanding along its rook
    lines), rooks, knights and guards `G`; ``rests`` per id (every type but the king 2 by
    default)."""
    rests = {"M": 2, "R": 2, "N": 2, "G": 2} if rests is None else rests
    sight = {**PREACH, **({"captures_own": True} if captures_own else {})}
    pieces: list[dict[str, Any]] = [
        {"id": "K", "royal": True, "value": 0, "moves": [KING]},
        {"id": "M", "value": 3, "moves": [KING], **({"sight": [sight]} if preach else {})},
        {"id": "R", "value": 5, "moves": [ROOK]},
        {"id": "N", "value": 3, "moves": [KNIGHT]},
        {"id": "G", "value": 2, "moves": [KING]},
    ]
    for piece in pieces:
        if rests.get(piece["id"]):
            piece["rest"] = rests[piece["id"]]
    return {
        "schema_version": 31, "name": "the weary preachers",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": pieces,
        "setup": {"placements": [
            {"piece": "K", "squares": ["c1"]}, {"piece": "M", "squares": ["d1"]},
            {"piece": "R", "squares": ["a1", "f1"]}, {"piece": "N", "squares": ["b1", "e1"]},
            {"piece": "G", "squares": ["b2", "d2", "e2"]},
        ]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 80},
    }


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _flip(name: str, side: int) -> str:
    return name if side == 0 else f"{name[0]}{WIDTH + 1 - int(name[1:])}"


def _sq(name: str, side: int = 0) -> int:
    name = _flip(name, side)
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _position(
    engine: Engine, side: int, rest: dict[str, int] | None = None, **extra: tuple[str, int],
) -> GameState:
    """``side`` to move (the commander), its preacher on c3 commanding the opponent's knight on
    c5 (as the first player sees it; the second player's is mirrored by rank). ``rest`` names
    resting squares (as the first player sees them) with their turns left; each goes in its
    piece's owner's list."""
    placement = {"a1": ("K", 0), "c3": ("M", 0), "f6": ("K", 1), "c5": ("N", 1), **extra}
    mirrored = {_flip(sq, side): (pid, owner ^ side) for sq, (pid, owner) in placement.items()}
    state = engine.make_state(mirrored, side_to_move=side, moved=list(mirrored))
    if rest:
        lists: list[list[tuple[int, int]]] = [[], []]
        for name, left in rest.items():
            lists[placement[name][1] ^ side].append((_sq(name, side), left))
        state = state.with_(rest=(tuple(sorted(lists[0])), tuple(sorted(lists[1]))))
        state = state.with_(key=engine._compute_hash(state))
    return state


def _from(engine: Engine, state: GameState, name: str, side: int) -> list[int]:
    return [m for m in engine.legal_moves(state) if move_from(m) == _sq(name, side)]


def _commanded(engine: Engine, state: GameState, side: int, target: str) -> int:
    found = [m for m in _from(engine, state, "c5", side) if move_to(m) == _sq(target, side)]
    assert len(found) == 1, engine.move_texts(state)
    assert found[0] & MOVE_FOREIGN
    return found[0]


def _check_oracle(engine: Engine, state: GameState) -> None:
    assert sorted(engine.legal_moves(state)) == sorted(rest_oracle.legal(engine, state))


# --------------------------------------------------------------------------- the validator


class TestTheLift:
    def test_the_pair_validates(self) -> None:
        assert validate_spec(_spec()) == []
        assert validate_spec(_spec(captures_own=True)) == []

    def test_the_inventory_names_the_pair(self) -> None:
        assert _schema_eighty_three_uses(_spec()) == {"pieces.sight.command.rest"}
        assert _schema_eighty_three_uses(_spec(preach=False)) == set()
        assert _schema_eighty_three_uses(_spec(rests={})) == set()

    def test_describe_says_whose_rest_it_is(self) -> None:
        said = describe_spec(_spec())
        assert "A commanded piece rests after its commanded move" in said
        assert "neither player may move it" in said
        assert "A commanded piece rests" not in describe_spec(_spec(rests={}))

    def test_fairy_stockfish_refuses_the_command(self) -> None:
        """Fairy-Stockfish moves no opponent's piece, so a command is refused as a sight."""
        with pytest.raises(Untranslatable, match="sight"):
            translate(_spec())


# --------------------------------------------------------------------------- the acceptance


class TestTheAcceptance:
    @pytest.mark.parametrize("side", [0, 1])
    def test_a_commanded_piece_rests_on_its_owners_turns(self, side: int) -> None:
        """The probe's first finding: a commanded move entered no rest. Now the knight enters its
        owner's list for its rest of 2, offered to its owner on neither of the owner's next two
        turns, nor to the commander between them, and counted down on the owner's turns only."""
        engine = _engine(_spec())
        state = _position(engine, side)
        knight = _sq("d3", side)
        after = engine.apply(state, _commanded(engine, state, side, "d3"))
        owner = 1 - side
        assert after.rest[owner] == ((knight, 2),) and after.rest[side] == ()
        assert after.key == engine._compute_hash(after)
        _check_oracle(engine, after)
        assert not _from(engine, after, "d3", side)  # the owner's turn: resting
        # The owner moves its king (which never rests); the knight counts down to 1.
        step = next(m for m in engine.legal_moves(after) if move_from(m) == _sq("f6", side))
        again = engine.apply(after, step)
        assert dict(again.rest[owner])[knight] == 1
        _check_oracle(engine, again)
        # The commander's turn: d3 is still on the preacher's rank, still resting, not offered.
        assert not _from(engine, again, "d3", side)
        # The commander moves; its own turn counts nothing of the owner's.
        quiet = next(m for m in engine.legal_moves(again) if move_from(m) == _sq("a1", side))
        third = engine.apply(again, quiet)
        assert dict(third.rest[owner])[knight] == 1
        assert not _from(engine, third, "d3", side)  # the owner's second turn: resting
        royal = next(m for m in engine.legal_moves(third)
                     if move_from(m) != knight and not m & MOVE_FOREIGN)
        fourth = engine.apply(third, royal)
        assert knight not in dict(fourth.rest[owner])
        assert any(m & MOVE_FOREIGN for m in _from(engine, fourth, "d3", side))

    @pytest.mark.parametrize("side", [0, 1])
    def test_the_owners_resting_piece_is_not_commanded(self, side: int) -> None:
        """The probe's second finding: a piece resting in its owner's list was offered to the
        commander."""
        engine = _engine(_spec())
        state = _position(engine, side, rest={"c5": 1})
        assert not _from(engine, state, "c5", side)
        _check_oracle(engine, state)
        free = _position(engine, side)
        assert _from(engine, free, "c5", side)
        assert engine.rest_turn(state, side).binds

    @pytest.mark.parametrize("side", [0, 1])
    def test_the_commanders_own_rest_does_not_bar_a_command(self, side: int) -> None:
        """The commander's preacher rests; the knight it commands does not, so it is offered and
        the lift does not fire."""
        engine = _engine(_spec())
        state = _position(engine, side, rest={"c3": 1})
        assert not _from(engine, state, "c3", side)
        assert _from(engine, state, "c5", side)
        assert not engine.rest_turn(state, side).lifted
        _check_oracle(engine, state)

    @pytest.mark.parametrize("side", [0, 1])
    def test_the_lift_reads_the_commanders_every_offer(self, side: int) -> None:
        """Every offer the commander has is a resting piece's: its king and preacher rest in its
        list, the commanded knight in its owner's. The lift fires for the commander's turn, and
        offers all three, the command included."""
        engine = _engine(_spec(rests={"K": 1, "M": 2, "N": 2}))
        state = _position(engine, side, rest={"a1": 1, "c3": 1, "c5": 2})
        moves = engine.legal_moves(state)
        froms = {move_from(m) for m in moves}
        assert {_sq("a1", side), _sq("c3", side), _sq("c5", side)} <= froms
        assert any(m & MOVE_FOREIGN for m in moves)
        turn = engine.rest_turn(state, side)
        assert turn.lifted and not turn.binds
        _check_oracle(engine, state)
        # A commanded move under the lift rests the knight afresh, in its owner's list.
        after = engine.apply(state, _commanded(engine, state, side, "d3"))
        assert dict(after.rest[1 - side]) == {_sq("d3", side): 2}
        assert after.key == engine._compute_hash(after)
        # The commander's own resting pieces leave on its count (rest 1 each): the lift is for
        # that turn only.
        assert after.rest[side] == ()

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_free_offer_keeps_the_rest(self, side: int) -> None:
        """The commander's own pieces all rest, but a command of a free piece remains: the lift
        does not fire, and the commander may move only the commanded knight."""
        engine = _engine(_spec(rests={"K": 1, "M": 2, "N": 2}))
        state = _position(engine, side, rest={"a1": 1, "c3": 1})
        moves = engine.legal_moves(state)
        assert moves and all(m & MOVE_FOREIGN for m in moves)
        assert engine.rest_turn(state, side).binds
        _check_oracle(engine, state)

    @pytest.mark.parametrize("side", [0, 1])
    def test_the_lift_for_the_owner_reads_its_own_offers(self, side: int) -> None:
        """The owner has a king and the knight, both resting (the king in its own list, which
        the commander's turn does not count down; the knight from the command): the lift fires
        for the owner and offers both."""
        engine = _engine(_spec(rests={"K": 1, "M": 2, "N": 2}))
        commander = _position(engine, side, rest={"f6": 1})
        state = engine.apply(commander, _commanded(engine, commander, side, "e4"))
        assert dict(state.rest[1 - side]) == {_sq("f6", side): 1, _sq("e4", side): 2}
        _check_oracle(engine, state)
        assert engine.rest_turn(state, 1 - side).lifted
        froms = {move_from(m) for m in engine.legal_moves(state)}
        assert {_sq("f6", side), _sq("e4", side)} <= froms

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_commanded_capture_onto_a_resting_twin_leaves_one_entry(self, side: int) -> None:
        """Under `captures_own` the commanded knight takes its owner's resting knight on d3: the
        owner's list holds one entry there, the mover's, afresh."""
        engine = _engine(_spec(captures_own=True))
        state = _position(engine, side, rest={"d3": 1}, d3=("N", 1))
        # d3 stands on the preacher's rank, so that knight is commanded too, but resting.
        after = engine.apply(state, _commanded(engine, state, side, "d3"))
        assert after.rest[1 - side] == ((_sq("d3", side), 2),)
        assert after.key == engine._compute_hash(after)


# --------------------------------------------------------------------------- walks


class TestTheWalks:
    @pytest.mark.parametrize("captures_own", [False, True], ids=["plain", "captures_own"])
    def test_every_list_and_rest_is_the_oracles(self, captures_own: bool) -> None:
        """Seeded walks leaning on commanded moves: every list is the oracle's, every rest list
        the oracle's, every key the full scan's."""
        engine = _engine(_spec(captures_own=captures_own))
        commanded = {0: 0, 1: 0}
        for seed in range(8):
            rng = random.Random(seed)
            state = engine.initial_state()
            for _ in range(50):
                moves = engine.legal_moves(state)
                assert sorted(moves) == sorted(rest_oracle.legal(engine, state)), (seed, state.ply)
                if engine.terminal(state, moves) is not None:
                    break
                foreign = [m for m in moves if m & MOVE_FOREIGN]
                move = rng.choice(foreign) if foreign and rng.random() < 0.7 else rng.choice(moves)
                after, lists = rest_oracle.apply(engine, state, move)
                assert after.rest == rest_oracle.as_rest(lists), (seed, state.ply)
                assert after.key == engine._compute_hash(after), (seed, state.ply)
                if move & MOVE_FOREIGN:
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
    @pytest.mark.parametrize("captures_own", [False, True], ids=["plain", "captures_own"])
    def test_perft_counts_what_apply_counts(self, captures_own: bool) -> None:
        engine = _engine(_spec(captures_own=captures_own))
        for side in (0, 1):
            state = _position(engine, side, rest={"c3": 1}, f2=("N", 0), b4=("G", 1))
            assert engine.perft(state, 3) == _apply_count(engine, state, 3)

    def test_a_searching_game_replays(self) -> None:
        """The core's searching players make and unmake thousands of times a move; a rest list
        their unmake left wrong would offer a move `apply` does not."""
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
