"""V1 of `docs/plan-rest.md` §6.3: what the rest does in its three showcases, without players.

The showcases (Q983d, §6.1, each one change from a measured parent): U1
`archive/gardner-with-weary-heavies` (Gardner, checkmate, the queen rests 2 and the rooks 1), U2
`archive/breather-by-weight` (breather in rounds with the pawns freed, knights, bishops and king
at 1, rooks and queen at 2), U3 `archive/chess-with-a-weary-queen` (chess, the queen rests 1).

V1 is pre-registered as: on hand-built positions and on 20 seeded random games each of U1 to U3
(`random.Random(seed)` over the legal moves, at most 200 plies), no resting piece moves outside a
lift; every lift happens exactly when the rest would leave no legal move; every rest list equals
the oracle's after every move or round; a resting piece taken, displaced or turned over never
rests again; in U1, a pawn promoting to a queen rests 2 turns as a queen; in U1 and U3, a
position where a resting queen gives check offers exactly the moves that answer it, and a
position where only a resting piece can answer a check gets the lift.

The rest lists are held to `tests/rest_oracle.py` (alternating) and `tests/simultaneous_oracle.py`
(rounds, which uses the same rest oracle), the declared test oracles; the list *with nothing
resting* is the core's own generator on the position with its rest cleared. Nothing here
decides a move's legality (CLAUDE.md, rule 3).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest
from rest_oracle import NO_REST, after_turn, as_rest, end_of_move, legal, lists_of
from retired_specs import RETIRED_PREVIEWS, retired_preview_path
from simultaneous_oracle import legal_for, resolve_round
from test_rest import froms, move, place, play, sq

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_DROP, MOVE_FOLD, decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import load_spec, spec_hash

ARCHIVE = Path(__file__).resolve().parent.parent / "archive"
SHOWCASES = {
    "U1": "gardner-with-weary-heavies",
    "U2": "breather-by-weight",
    "U3": "chess-with-a-weary-queen",
}
# The registered change of each (§6.1): every type's rest, 0 for none.
REGISTERED = {
    "U1": {"P": 0, "N": 0, "B": 0, "R": 1, "Q": 2, "K": 0},
    "U2": {"P": 0, "N": 1, "B": 1, "R": 2, "Q": 2, "K": 1},
    "U3": {"P": 0, "N": 0, "B": 0, "R": 0, "Q": 1, "K": 0},
}
PARENTS = {"U1": "975dde173f01", "U2": "b8687024cfc6", "U3": "4c1922ffae26"}
GAMES, MAX_PLIES = 20, 200


@cache
def engine(label: str) -> Engine:
    return Engine(load_spec(showcase_path(label)))


def showcase_path(label: str) -> Path:
    """Where the showcase `label` is kept: U3 joined the library, U1 and U2 left the previews
    for `tests/_retired_*.json` (Q1005, `docs/reference/preview-call-2026-10.md`)."""
    name = SHOWCASES[label]
    return retired_preview_path(name) if name in RETIRED_PREVIEWS else ARCHIVE / f"{name}.json"


def resting_squares(state: GameState, player: int) -> set[int]:
    return {square for square, _ in state.rest[player]}


def rests(move_word: int, squares: set[int]) -> bool:
    """Whether a move is made by a piece standing on one of these squares (never a drop)."""
    origin, _, kind, _ = decode_move(move_word)
    return kind not in (MOVE_DROP, MOVE_FOLD) and origin in squares


# ------------------------------------------------------------------ the files, as registered


@pytest.mark.parametrize("label", sorted(SHOWCASES))
def test_each_showcase_is_its_parent_with_the_registered_rests_and_nothing_else(label: str) -> None:
    spec = load_spec(showcase_path(label))
    assert {p["id"]: p.get("rest", 0) for p in spec["pieces"]} == REGISTERED[label]
    assert spec["provenance"]["parents"] == [PARENTS[label]]
    # U3 joined the library (Q1005); U1 and U2 were retired, their files moved unchanged.
    assert ("preview" in spec["provenance"]) == (SHOWCASES[label] in RETIRED_PREVIEWS)
    parent_path = {
        "U1": ARCHIVE.parent / "rulesets" / "gardner.json",
        "U2": ARCHIVE / "breather.json",
        "U3": ARCHIVE.parent / "rulesets" / "chess.json",
    }[label]
    parent = load_spec(parent_path)
    assert spec_hash(parent) == PARENTS[label]
    # Strip the rest from both, and declare the parent's version 29: the same game.
    stripped = {k: spec[k] for k in ("board", "setup", "rules")}
    unrested = {k: parent[k] for k in ("board", "setup", "rules")}
    assert stripped == unrested
    assert [{k: v for k, v in p.items() if k != "rest"} for p in spec["pieces"]] == [
        {k: v for k, v in p.items() if k != "rest"} for p in parent["pieces"]
    ]


# --------------------------------------------------------- random play, move by move


@dataclass
class Walked:
    turns: int = 0
    bound: int = 0
    lifted: int = 0
    freed: int = 0
    entered: dict[int, int] | None = None


def check_list(eng: Engine, state: GameState, player: int | None, walked: Walked) -> list[int]:
    """One side-turn's list: the core's, held to the oracle's, the lift exactly when the rest
    would leave no move, and no resting piece's move outside it."""
    mover = state.side_to_move if player is None else player
    if player is None:
        moves = eng.legal_moves(state)
        free = eng.legal_moves(state.with_(rest=NO_REST))
    else:
        moves = eng.legal_moves_for(state, player)
        free = eng.legal_moves_for(state.with_(rest=NO_REST), player)
    assert moves == (legal(eng, state) if player is None else legal_for(eng, state, player))
    squares = resting_squares(state, mover)
    unrested = [m for m in free if not rests(m, squares)]
    walked.turns += 1
    if unrested:
        # No lift: exactly the free list less every resting piece's moves.
        assert moves == unrested
        walked.bound += len(unrested) < len(free)
    elif free:
        # The rest would leave no move: lifted, the whole free list.
        assert moves == free
        walked.lifted += 1
    return moves


def check_entries(eng: Engine, before: GameState, after: GameState, walked: Walked) -> None:
    """A rest entry only begins on a square a piece moved to; an entry carried over keeps its
    piece; so a resting piece taken, displaced or turned over never rests again."""
    assert walked.entered is not None
    for player in (0, 1):
        old = dict(before.rest[player])
        for square, left in after.rest[player]:
            if square in old and left < old[square] and before.board[square] == after.board[square]:
                continue  # carried, counted down, the same piece
            if square in old and left == old[square] and before.board[square] == after.board[square]:
                continue  # the other player's turn (alternating): carried, uncounted
            # A fresh entry: the type standing there now, at its own rest.
            kind = eng.spec.type_of(after.board[square])
            assert left == eng.spec.rest[kind]
            walked.entered[kind] = walked.entered.get(kind, 0) + 1
        for square in old:
            if before.board[square] != after.board[square]:
                walked.freed += 1
                fresh = dict(after.rest[player]).get(square)
                if fresh is not None:
                    # A new piece on the square rests afresh only if it is its player's mover.
                    assert fresh == eng.spec.rest[eng.spec.type_of(after.board[square])]


def walk_turns(eng: Engine, seed: int, walked: Walked) -> None:
    rng = random.Random(seed)
    state = eng.initial_state()
    for _ in range(MAX_PLIES):
        moves = eng.legal_moves(state)
        if eng.terminal(state, moves) is not None:
            return
        check_list(eng, state, None, walked)
        word = rng.choice(moves)
        after = eng.apply(state, word)
        expected = after_turn(
            eng, lists_of(state), after.board, state.side_to_move,
            end_of_move(eng, state, after, word),
        )
        assert after.rest == as_rest(expected)
        check_entries(eng, state, after, walked)
        state = after


def walk_rounds(eng: Engine, seed: int, walked: Walked) -> None:
    rng = random.Random(seed)
    state = eng.initial_state()
    for _ in range(MAX_PLIES // 2):
        if eng.terminal(state) is not None:
            return
        lists = [check_list(eng, state, player, walked) for player in (0, 1)]
        first, second = rng.choice(lists[0]), rng.choice(lists[1])
        after = eng.apply_round(state, first, second)
        assert after == resolve_round(eng, state, first, second)
        check_entries(eng, state, after, walked)
        state = after


@pytest.mark.parametrize("label", sorted(SHOWCASES))
def test_random_games_rest_as_the_oracle_says_and_lift_only_when_nothing_else_moves(label: str) -> None:
    eng = engine(label)
    walked = Walked(entered={})
    for seed in range(GAMES):
        (walk_rounds if label == "U2" else walk_turns)(eng, seed, walked)
    assert walked.bound > 0, "the rest never removed a move"
    assert walked.freed > 0, "no resting piece was ever taken or displaced"
    ids = eng.spec.piece_ids
    # Exactly the resting types ever enter the list, each at its own rest.
    assert {ids[k] for k in walked.entered or {}} == {
        name for name, n in REGISTERED[label].items() if n
    }


# --------------------------------------------------------- hand-built positions


def test_u1_a_pawn_promoted_to_a_queen_rests_two_turns_as_the_queen() -> None:
    eng = engine("U1")
    state = place(eng, {"a1": ("K", 0), "d4": ("P", 0), "b1": ("N", 0), "e3": ("K", 1)})
    promote = [
        m for m in eng.legal_moves(state)
        if decode_move(m)[:2] == (sq(eng, "d4"), sq(eng, "d5"))
        and eng.spec.piece_ids[decode_move(m)[3]] == "Q"
    ]
    assert len(promote) == 1
    queen = sq(eng, "d5")
    state = play(eng, state, promote[0])
    assert state.rest == (((queen, 2),), ())
    state = play(eng, state, move(eng, state, "e3", "e2"))
    assert queen not in froms(eng, state)  # White's first turn after it
    state = play(eng, state, move(eng, state, "b1", "a3"))
    assert state.rest == (((queen, 1),), ())
    state = play(eng, state, move(eng, state, "e2", "e3"))
    assert queen not in froms(eng, state)  # the second
    state = play(eng, state, move(eng, state, "a3", "b1"))
    assert state.rest == ((), ())
    state = play(eng, state, move(eng, state, "e3", "e2"))
    assert queen in froms(eng, state)  # the third: free


# The checks: a resting queen giving check, and a check only a resting piece can answer.
CHECKS = {
    "U1": (
        {"a1": ("K", 0), "c2": ("R", 0), "e5": ("K", 1), "a4": ("Q", 1)}, ({}, {"a4": 2}),
        {"a1": ("K", 0), "a2": ("P", 0), "b2": ("P", 0), "c2": ("Q", 0),
         "e1": ("Q", 1), "e5": ("K", 1)}, ({"c2": 2}, {}),
    ),
    "U3": (
        {"e1": ("K", 0), "c2": ("R", 0), "h8": ("K", 1), "a5": ("Q", 1)}, ({}, {"a5": 1}),
        {"h1": ("K", 0), "g2": ("P", 0), "h2": ("P", 0), "e2": ("Q", 0),
         "a1": ("R", 1), "a8": ("K", 1)}, ({"e2": 1}, {}),
    ),
}


@pytest.mark.parametrize("label", ["U1", "U3"])
def test_a_resting_queens_check_is_answered_by_exactly_the_moves_that_answer_it(label: str) -> None:
    eng = engine(label)
    pieces, rest, _, _ = CHECKS[label]
    state = place(eng, pieces, rest=rest)
    assert eng.in_check(state, 0)
    answers = eng.legal_moves(state)
    assert answers and answers == legal(eng, state)
    # Every offered move answers the check, and every move that answers it is offered: White
    # has nothing resting, so the list is the free list, the moves that leave White unchecked.
    assert answers == eng.legal_moves(state.with_(rest=NO_REST))
    for word in answers:
        assert not eng.in_check(eng.apply(state, word), 0), eng.describe_move(state, word)


@pytest.mark.parametrize("label", ["U1", "U3"])
def test_a_check_only_a_resting_piece_can_answer_gets_the_lift(label: str) -> None:
    eng = engine(label)
    _, _, pieces, rest = CHECKS[label]
    state = place(eng, pieces, rest=rest)
    assert eng.in_check(state, 0)
    resting = resting_squares(state, 0)
    moves = eng.legal_moves(state)
    assert moves and all(rests(m, resting) for m in moves)
    assert moves == eng.legal_moves(state.with_(rest=NO_REST)) == legal(eng, state)
    assert eng.terminal(state, moves) is None
    for word in moves:
        assert not eng.in_check(eng.apply(state, word), 0)
