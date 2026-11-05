"""SR7h: a spy can be gained, by a promotion into one (Y3, `docs/plan-schema-next.md` §2.12,
*The gained spy, said once*).

The phase stays a function of who has a spy on the board and moves both ways. Each transition
on a scratch position of `_spy_promoting` (spy small, its pawns promoting into R, N or S), for
both seats: a second spy is a buffer; a round in which the spyless side gains returns to turns,
the other player first (both gaining: the first player); a gain in a blind round starts
announced rounds; a gain on the reply turn returns to turns; under `tie: "initiative"` the token
is dealt again where announced rounds begin. Every step is held to `tests/spy_oracle.py`, the
key to the slow one, and `perft_spy` across each gain to the oracle's count (the core's unmake
of a round that wrote the side is what that count walks back through).
"""

from __future__ import annotations

import json
import random
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.tournament.play import play_game, play_game_python
from tests import spy_oracle as oracle

HERE = Path(__file__).parent
NAME = "_spy_promoting"


def _load(name: str = NAME) -> dict[str, Any]:
    return json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))


@cache
def _engine(tie: str | None = None, plies: int | None = None) -> Engine:
    spec = _load()
    if tie is not None:
        spec["rules"]["simultaneous"] = {"tie": tie}
    if plies is not None:
        spec["rules"]["max_plies"] = plies
    return Engine(spec)


def _flip(square: str) -> str:
    """The square seen from the other side of the 5x5 board (a rank flip, as the mirror)."""
    return f"{square[0]}{6 - int(square[1:])}"


def _position(
    engine: Engine, mine: dict[str, str], theirs: dict[str, str], seat: int, side: int,
    initiative: int | None = None,
) -> GameState:
    """``mine`` for ``seat`` and ``theirs`` for the other, written from player 0's side of the
    board and flipped for player 1, so one scenario reads for both seats; ``side`` is
    ``seat``'s to move when 0."""
    placement: dict[str, tuple[str, int]] = {}
    for squares, owner in ((mine, seat), (theirs, 1 - seat)):
        for square, piece in squares.items():
            placement[square if seat == 0 else _flip(square)] = (piece, owner)
    state = engine.make_state(placement, side_to_move=seat if side == 0 else 1 - seat)
    if initiative is not None:
        state = state.with_(initiative=initiative)
        key = engine._compute_hash(state)
        state = state.with_(key=key, repetition=(key,))
    assert engine.terminal(state) is None
    return state


def _square(seat: int, square: str) -> str:
    return square if seat == 0 else _flip(square)


def _find(engine: Engine, moves: list[int], seat: int, origin: str, target: str, into: str = "") -> int:
    a = engine._square_index(_square(seat, origin))
    b = engine._square_index(_square(seat, target))
    found = [
        m for m in moves
        if decode_move(m)[:2] == (a, b)
        and (not into or engine.spec.piece_ids[decode_move(m)[3]] == into)
    ]
    assert len(found) == 1, (origin, target, into)
    return found[0]


def _phase(engine: Engine, state: GameState) -> tuple[str, int]:
    phase = engine.phase(state)
    assert (phase.name, phase.announcer) == oracle.phase(engine, state)
    return phase.name, phase.announcer


def _turn(engine: Engine, state: GameState, seat: int, origin: str, target: str, into: str = "") -> GameState:
    move = _find(engine, engine.legal_moves(state), seat, origin, target, into)
    after = engine.apply(state, move)
    assert after == oracle.turn(engine, state, move)
    assert after.key == engine._compute_hash(after)
    return after


def _round(
    engine: Engine, state: GameState, seat: int,
    mine: tuple[str, ...], theirs: tuple[str, ...],
) -> GameState:
    """One round, ``mine`` for ``seat`` and ``theirs`` for the other, each ``(origin, target)``
    or ``(origin, target, into)``."""
    moves = [0, 0]
    for player, words in ((seat, mine), (1 - seat, theirs)):
        moves[player] = _find(engine, engine.legal_moves_for(state, player), seat, *words)
    after = engine.apply_round(state, moves[0], moves[1])
    assert after == oracle.round_(engine, state, moves[0], moves[1])
    assert after.key == engine._compute_hash(after)
    return after


def _perft_agrees(engine: Engine, state: GameState, depth: int) -> None:
    from tests.test_spy import _oracle_perft

    assert engine.perft(state, depth) == _oracle_perft(engine, state, depth)


SEATS = pytest.mark.parametrize("seat", (0, 1))


def test_the_spec_validates_and_its_promotion_reaches_the_spy() -> None:
    assert validate_spec(_load()) == []
    assert "S" in _load()["rules"]["promotion"]["to"]


def test_only_lost_offers_the_spy_back_only_to_a_side_that_lost_it() -> None:
    """The settled reading of `promotion.only_lost` beside a spy: a target while the side has
    fewer than it started with, so a lost spy can be won back and a standing one not doubled."""
    spec = _load()
    spec["rules"]["promotion"]["only_lost"] = True
    assert validate_spec(spec) == []
    engine = Engine(spec)

    def into(white: dict[str, str]) -> set[str]:
        state = _position(engine, white, {"a5": "K", "e3": "S"}, 0, side=0)
        return {
            engine.spec.piece_ids[decode_move(m)[3]] for m in engine.legal_moves(state)
            if decode_move(m)[0] == engine._square_index("d4")
        }

    assert into({"a1": "K", "d4": "P"}) == {"R", "N", "S"}
    assert into({"a1": "K", "d4": "P", "c1": "S"}) == {"R", "N"}


@SEATS
def test_a_second_spy_is_a_buffer(seat: int) -> None:
    engine = _engine()
    mine = {"a1": "K", "c1": "S", "d4": "P"}
    theirs = {"a5": "K", "c5": "R", "e3": "S"}
    state = _position(engine, mine, theirs, seat, side=0)
    assert _phase(engine, state) == ("alternating", -1)
    gained = _turn(engine, state, seat, "d4", "d5", "S")
    taken = _turn(engine, gained, seat, "c5", "c1")
    # One spy taken of two: still turns, nothing owed.
    assert _phase(engine, taken) == ("alternating", -1)
    # Without the second spy the same capture owes the reply turn.
    plain = _turn(engine, state, seat, "a1", "b1")
    assert _phase(engine, _turn(engine, plain, seat, "c5", "c1")) == ("reply", -1)
    _perft_agrees(engine, state, 3)


@SEATS
def test_a_gain_on_the_reply_turn_returns_to_turns(seat: int) -> None:
    engine = _engine()
    state = _position(engine, {"a1": "K", "d4": "P"}, {"a5": "K", "e3": "S"}, seat, side=0)
    assert _phase(engine, state) == ("reply", -1)
    after = _turn(engine, state, seat, "d4", "d5", "S")
    assert _phase(engine, after) == ("alternating", -1)
    assert after.side_to_move == 1 - seat  # the other player moves next, as after any turn
    # Not gaining: announced rounds, the reply's player announcing.
    plain = _turn(engine, state, seat, "d4", "d5", "R")
    assert _phase(engine, plain) == ("announced", seat)
    _perft_agrees(engine, state, 3)


@SEATS
def test_a_round_in_which_the_spyless_side_gains_returns_to_turns_the_other_first(seat: int) -> None:
    engine = _engine()
    # The other player holds the only spy; ``seat`` announces, and promotes into a spy.
    state = _position(engine, {"a1": "K", "d4": "P"}, {"a5": "K", "e3": "S"}, seat, side=1)
    assert _phase(engine, state) == ("announced", seat)
    after = _round(engine, state, seat, ("d4", "d5", "S"), ("a5", "b5"))
    assert _phase(engine, after) == ("alternating", -1)
    assert after.side_to_move == 1 - seat  # the player who did not gain answers the gain
    _perft_agrees(engine, state, 2)


@SEATS
def test_a_gain_in_a_blind_round_starts_announced_rounds(seat: int) -> None:
    engine = _engine()
    for side in (0, 1):
        state = _position(engine, {"a1": "K", "d4": "P"}, {"a5": "K"}, seat, side=side)
        assert _phase(engine, state) == ("blind", -1)
        after = _round(engine, state, seat, ("d4", "d5", "S"), ("a5", "b5"))
        # The gainer holds the spy and is to move; the other, still without, announces.
        assert _phase(engine, after) == ("announced", 1 - seat)
        assert after.side_to_move == seat
    _perft_agrees(engine, state, 2)


def test_both_gaining_in_one_round_returns_to_turns_the_first_player_first() -> None:
    engine = _engine()
    for side in (0, 1):
        state = _position(
            engine, {"a1": "K", "b4": "P"}, {"a5": "K", "d2": "P"}, seat=0, side=side,
        )
        # Both pawns one step from promotion, neither side with a spy.
        assert _phase(engine, state) == ("blind", -1)
        moves = (
            _find(engine, engine.legal_moves_for(state, 0), 0, "b4", "b5", "S"),
            _find(engine, engine.legal_moves_for(state, 1), 0, "d2", "d1", "S"),
        )
        after = engine.apply_round(state, *moves)
        assert after == oracle.round_(engine, state, *moves)
        assert after.key == engine._compute_hash(after)
        assert _phase(engine, after) == ("alternating", -1) and after.side_to_move == 0
    _perft_agrees(engine, state, 2)


@SEATS
def test_a_spy_changing_hands_in_a_round_keeps_announced_rounds(seat: int) -> None:
    engine = _engine("initiative")
    # The holder's only spy stands on d1; the other's (``seat``'s) pawn takes it there,
    # promoting into a spy of its own: the holder changes, the rounds stay announced, and
    # nothing is dealt.
    holder = 1 - seat
    state = _position(
        engine, {"a3": "K", "d1": "S"}, {"e5": "K", "e2": "P"}, seat=holder, side=0,
        initiative=seat,
    )
    assert _phase(engine, state) == ("announced", seat)
    moves = [0, 0]
    moves[holder] = _find(engine, engine.legal_moves_for(state, holder), holder, "a3", "b3")
    moves[seat] = _find(engine, engine.legal_moves_for(state, seat), holder, "e2", "d1", "S")
    after = engine.apply_round(state, *moves)
    assert after == oracle.round_(engine, state, *moves)
    assert after.key == engine._compute_hash(after)
    assert _phase(engine, after) == ("announced", holder) and after.side_to_move == seat
    assert after.initiative == seat


@SEATS
def test_the_initiative_is_dealt_again_where_announced_rounds_begin(seat: int) -> None:
    engine = _engine("initiative")
    # Blind rounds, the token with the player about to gain; announced rounds begin and it
    # goes to the announcer, the other player.
    state = _position(engine, {"a1": "K", "d4": "P"}, {"a5": "K"}, seat, side=0, initiative=seat)
    after = _round(engine, state, seat, ("d4", "d5", "S"), ("a5", "b5"))
    assert _phase(engine, after) == ("announced", 1 - seat)
    assert after.initiative == 1 - seat
    # A second entry from turns deals again: back to turns by the other's gain, the spy taken
    # back, the reply played, and the turn into announced rounds deals once more.
    state = _position(
        engine, {"a1": "K", "c1": "S"}, {"a5": "K", "c5": "R", "d2": "P"}, seat, side=0,
        initiative=1 - seat,
    )
    assert _phase(engine, state) == ("announced", 1 - seat)
    back = _round(engine, state, seat, ("a1", "b1"), ("d2", "d1", "S"))
    assert _phase(engine, back) == ("alternating", -1) and back.side_to_move == seat
    moved = _turn(engine, back, seat, "b1", "b2")
    lost = _turn(engine, moved, seat, "d1", "e1")
    taken = _turn(engine, lost, seat, "c1", "c2")
    assert _phase(engine, taken) == ("alternating", -1)
    again = _turn(engine, taken, seat, "c5", "c2")
    assert _phase(engine, again) == ("reply", -1)
    entered = _turn(engine, again, seat, "b2", "a2")
    assert _phase(engine, entered) == ("announced", seat)
    assert entered.initiative == seat
    _perft_agrees(engine, state, 2)


def _walk(engine: Engine, seed: int, limit: int = 200) -> list[str]:
    """A random walk through the core, each step the oracle's; the phases it passed."""
    rng = random.Random(seed)
    state = engine.initial_state()
    seen: list[str] = []
    for _ in range(limit):
        if engine.terminal(state) is not None:
            break
        name = _phase(engine, state)[0]
        seen.append(name)
        if engine.phase(state).is_round:
            lists = engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1)
            assert lists == oracle.legal(engine, state)
            pair = rng.choice(lists[0]), rng.choice(lists[1])
            after = engine.apply_round(state, *pair)
            assert after == oracle.round_(engine, state, *pair)
        else:
            moves = engine.legal_moves(state)
            assert moves == oracle.legal(engine, state)
            move = rng.choice(moves)
            after = engine.apply(state, move)
            assert after == oracle.turn(engine, state, move)
        assert after.key == engine._compute_hash(after)
        state = after
    return seen


def test_random_walks_cross_the_phase_both_ways_as_the_oracle_does() -> None:
    order = {"alternating": 0, "reply": 1, "announced": 2, "blind": 3}
    back = 0
    for tie in (None, "initiative"):
        engine = _engine(tie)
        for seed in range(12):
            seen = _walk(engine, seed)
            back += sum(1 for a, b in zip(seen, seen[1:], strict=False) if order[b] < order[a])
    assert back > 0, "no walk went back a phase"


_PAIRINGS = (("greedy", "greedy"), ("minimax:nodes=150", "greedy"), ("random", "minimax:nodes=200"))


@pytest.mark.parametrize("agents", _PAIRINGS, ids="-".join)
def test_the_core_plays_a_gaining_game_as_the_oracles_do(agents: tuple[str, str]) -> None:
    engine = _engine(plies=40)
    native = play_game(engine, agents, 3)
    python = play_game_python(engine, agents, 3)
    assert native.moves == python.moves
    assert (native.winner, native.reason) == (python.winner, python.reason)
    assert native.nodes == python.nodes


@pytest.mark.parametrize("agent", ("greedy", "minimax:nodes=100", "minimax:nodes=1600"))
@SEATS
def test_the_players_search_across_a_spy_changing_hands_as_their_oracles(agent: str, seat: int) -> None:
    """The round after which the side to move changes (the holder's spy taken by a pawn
    promoting into a spy, so the search negates the child): the announcer's and the
    answerer's choices, held to the oracles."""
    from tests.test_spy_players import _choose

    engine = _engine()
    holder = 1 - seat
    state = _position(
        engine, {"a3": "K", "d1": "S", "b1": "N"}, {"e5": "K", "e2": "P", "c4": "R"},
        seat=holder, side=0,
    )
    assert _phase(engine, state) == ("announced", seat)
    announced, _ = _choose(engine, state, seat, agent)
    _choose(engine, state, holder, agent, announced)


def test_a_blind_root_stops_its_matrix_where_a_gain_leaves_the_blind_rounds() -> None:
    """`minimax` at a blind root plays the matrix search, which values a child that gained a
    spy as a leaf, in the core as in its oracle."""
    from tests.test_spy_players import _choose

    engine = _engine()
    state = _position(engine, {"a1": "K", "d4": "P", "b1": "N"}, {"a5": "K", "e2": "P"}, 0, side=0)
    assert _phase(engine, state) == ("blind", -1)
    for agent in ("greedy", "minimax:nodes=400"):
        for seat in (0, 1):
            _choose(engine, state, seat, agent)
