"""SR7k: a spy placed for nobody (O6, `docs/plan-schema-next.md` §2.11 and §2.12).

Ruled 2026-10-28, lifted: a spy placed for nobody gives nobody secrecy, since the phase counts
the players' spies on the board only. A player gains it by a rule that makes a taken piece
theirs (a capture hand, Y8; a side switch, Y5, never reaches it, since a switch changes a
capturer's side and a nobody's spy never captures: it is never wild, Y2); taken by any other
rule it is gone and no count changes.

Two twins of `_spy_promoting` (spy small, the phase both ways since SR7h), each with a spy of
the shared type placed for nobody on c3: **bare**, and **hand** (`drops.from_captures`). Each
transition on a scratch position, both seats, held to `tests/spy_oracle.py` (the turn by the
alternating twin, the round by the round oracle, the phase off the board); then random walks
and the core's players against their Python oracles.
"""

from __future__ import annotations

import copy
import json
import random
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_DROP, decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.schema import foreign_spy_gains, validate_spec
from chess_variant_sandbox.tournament.analysis import SpyStep, blank_spy_counts, count_spy_holders
from chess_variant_sandbox.tournament.play import play_game, play_game_python
from tests import spy_oracle as oracle
from tests.test_spy_gained import _flip, _perft_agrees, _phase
from tests.test_spy_in_hand import _held, _round, _spies, _turn, _with_hands, _word

HERE = Path(__file__).parent
NOBODY = 2


def _spec(name: str) -> dict[str, Any]:
    spec = json.loads((HERE / "_spy_promoting.json").read_text(encoding="utf-8"))
    spec["setup"]["placements"].append({"piece": "S", "squares": ["c3"], "player": "nobody"})
    if name == "hand":
        spec["rules"]["drops"] = {"from_captures": True}
    spec["name"] = f"spy for nobody ({name})"
    return spec


@cache
def _engine(name: str, plies: int | None = None) -> Engine:
    spec = _spec(name)
    if plies is not None:
        spec["rules"]["max_plies"] = plies
    return Engine(spec)


def _position(
    engine: Engine, mine: dict[str, str], theirs: dict[str, str], seat: int, side: int,
    nobodys: tuple[str, ...] = ("b3",),
) -> GameState:
    """`test_spy_gained._position` with a nobody's spy on each of ``nobodys`` (written from
    player 0's side, flipped for player 1)."""
    placement: dict[str, tuple[str, int]] = {}
    for squares, owner in ((mine, seat), (theirs, 1 - seat)):
        for square, piece in squares.items():
            placement[square if seat == 0 else _flip(square)] = (piece, owner)
    for square in nobodys:
        placement[square if seat == 0 else _flip(square)] = ("S", NOBODY)
    state = engine.make_state(placement, side_to_move=seat if side == 0 else 1 - seat)
    assert engine.terminal(state) is None
    return state


def _without_nobody(engine: Engine, state: GameState) -> GameState:
    board = bytes(
        0 if value and engine.spec.owner_of(value) == NOBODY else value for value in state.board
    )
    bare = state.with_(board=board)
    key = engine._compute_hash(bare)
    return bare.with_(key=key, repetition=(key,))


def _nobodys_spies(engine: Engine, state: GameState) -> int:
    spy = engine.spec.piece_ids.index("S")
    return sum(
        1 for value in state.board
        if value and engine.spec.owner_of(value) == NOBODY and engine.spec.type_of(value) == spy
    )


SEATS = pytest.mark.parametrize("seat", (0, 1))
NAMES = pytest.mark.parametrize("name", ("bare", "hand"))


@NAMES
def test_each_twin_validates(name: str) -> None:
    spec = _spec(name)
    assert validate_spec(spec) == []
    assert foreign_spy_gains(spec) == []


def test_the_refusal_is_gone_and_a_strike_may_make_a_nobody_s_spy() -> None:
    spec = _spec("bare")
    spec["rules"]["strike"] = {"chain": [{"from": "N", "to": "S"}]}
    spec["setup"]["placements"].append({"piece": "N", "squares": ["a3"], "player": "nobody"})
    assert not [error for error in validate_spec(spec) if "never gained" in error]


def test_a_nobody_s_spy_of_a_type_no_player_starts_with_reaches_no_hand() -> None:
    """Taken into a hand it would be its taker's to drop, as a type the taker does not start
    with: a spy for nobody, refused by the drops' own-type rule (SR7j), by name."""
    spec = _spec("bare")
    lurker = copy.deepcopy(next(piece for piece in spec["pieces"] if piece["id"] == "S"))
    lurker.update(id="T", name="lurker")
    spec["pieces"].append(lurker)
    spec["setup"]["placements"][-1]["piece"] = "T"
    assert validate_spec(spec) == []
    spec["rules"]["drops"] = {"from_captures": True}
    errors = validate_spec(spec)
    assert len(errors) == 2 and all("the spy 'T' it takes" in error for error in errors)


# ---------------------------------------------------------------- standing


@NAMES
@SEATS
def test_a_standing_nobody_s_spy_changes_no_phase(name: str, seat: int) -> None:
    engine = _engine(name)
    cases = (
        ({"a1": "K", "e1": "S"}, {"a5": "K", "e5": "S"}, 0, ("alternating", -1)),
        ({"a1": "K"}, {"a5": "K", "e5": "S"}, 0, ("reply", -1)),
        ({"a1": "K"}, {"a5": "K", "e5": "S"}, 1, ("announced", seat)),
        ({"a1": "K"}, {"a5": "K"}, 0, ("blind", -1)),
    )
    for mine, theirs, side, expected in cases:
        state = _position(engine, mine, theirs, seat, side)
        assert _phase(engine, state) == expected
        assert _phase(engine, _without_nobody(engine, state)) == expected


# ---------------------------------------------------------------- taken, gone


@SEATS
def test_a_nobody_s_spy_taken_on_a_turn_changes_no_phase(seat: int) -> None:
    engine = _engine("bare")
    state = _position(engine, {"a1": "K", "b1": "R", "e1": "S"}, {"a5": "K", "e5": "S"}, seat, 0)
    after = _turn(engine, state, seat, "b1", "b3")
    assert _nobodys_spies(engine, after) == 0
    assert _phase(engine, after) == ("alternating", -1) and after.side_to_move == 1 - seat
    # On the reply turn: the taker still has no spy, so announced rounds begin as after a
    # quiet move, the spyless taker announcing.
    state = _position(engine, {"a1": "K", "b1": "R"}, {"a5": "K", "e5": "S"}, seat, 0)
    assert _phase(engine, state) == ("reply", -1)
    taken = _turn(engine, state, seat, "b1", "b3")
    quiet = _turn(engine, state, seat, "b1", "b2")
    assert _phase(engine, taken) == _phase(engine, quiet) == ("announced", seat)
    _perft_agrees(engine, state, 3)


@SEATS
def test_a_nobody_s_spy_taken_in_a_blind_round_changes_no_phase(seat: int) -> None:
    engine = _engine("bare")
    state = _position(engine, {"a1": "K", "b1": "R"}, {"a5": "K"}, seat, 0)
    assert _phase(engine, state) == ("blind", -1)
    after = _round(engine, state, seat, ("b1", "b3"), ("a5", "b5"))
    assert _nobodys_spies(engine, after) == 0
    assert _phase(engine, after) == ("blind", -1)


# ---------------------------------------------------------------- taken into a hand


@SEATS
def test_a_nobody_s_spy_taken_into_the_hand_is_the_taker_s_buffer_when_dropped(seat: int) -> None:
    engine = _engine("hand")
    spy = engine.spec.piece_ids.index("S")
    state = _with_hands(
        engine,
        _position(engine, {"a1": "K", "b1": "R", "e1": "S"}, {"a5": "K", "e5": "S"}, seat, 0),
        seat, 0, 0,
    )
    after = _turn(engine, state, seat, "b1", "b3")
    assert _held(engine, after, seat) == 1 and _phase(engine, after) == ("alternating", -1)
    after = _turn(engine, after, seat, "a5", "b5")
    dropped = _turn(engine, after, seat, "drop", "c3")
    square = engine._square_index("c3" if seat == 0 else _flip("c3"))
    assert engine.spec.owner_of(dropped.board[square]) == seat
    assert engine.spec.type_of(dropped.board[square]) == spy
    assert _spies(engine, dropped, seat) == 2 and _held(engine, dropped, seat) == 0


@SEATS
def test_a_spyless_taker_drops_a_nobody_s_spy_and_turns_resume(seat: int) -> None:
    """Taken on the reply turn it counts for nobody: announced rounds begin, the spyless
    taker announcing; dropped in that round it is a gain, and turns resume with the other
    player first."""
    engine = _engine("hand")
    state = _with_hands(
        engine, _position(engine, {"a1": "K", "b1": "R"}, {"a5": "K", "e5": "S"}, seat, 0),
        seat, 0, 0,
    )
    after = _turn(engine, state, seat, "b1", "b3")
    assert _held(engine, after, seat) == 1
    assert _phase(engine, after) == ("announced", seat)
    after = _round(engine, after, seat, ("drop", "c3"), ("a5", "b5"))
    assert _spies(engine, after, seat) == 1
    assert _phase(engine, after) == ("alternating", -1) and after.side_to_move == 1 - seat


@SEATS
def test_the_pass_reads_no_loss_at_a_nobody_s_spy_and_a_drop_s_gain(seat: int) -> None:
    engine = _engine("hand")
    state = _with_hands(
        engine, _position(engine, {"a1": "K", "b1": "R"}, {"a5": "K"}, seat, 0), seat, 0, 0,
    )
    move = _word(engine, engine.legal_moves_for(state, seat), seat, ("b1", "b3"))
    other = _word(engine, engine.legal_moves_for(state, 1 - seat), seat, ("a5", "b5"))
    pair = (move, other) if seat == 0 else (other, move)
    after, _, events = engine.apply_round_with_phase(state, *pair)
    lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
    spy = blank_spy_counts()
    count_spy_holders(engine, spy, SpyStep(0, state, engine.phase(state), pair, lists, after, events), 7)
    assert {key: value for key, value in spy.items() if value not in (None, 0)} == {}


def test_the_description_says_a_spy_nobody_owns_counts_for_neither_player() -> None:
    from chess_variant_sandbox.spec.describe import describe_spec

    bare = describe_spec(_spec("bare"))
    hand = describe_spec(_spec("hand"))
    assert "A spy nobody owns counts for neither player and gives nobody secrecy; taken, it is gone." in bare
    assert "taken, it goes to its taker's hand, and dropping it gains it." in hand
    plain = describe_spec(json.loads((HERE / "_spy_promoting.json").read_text(encoding="utf-8")))
    assert "A spy nobody owns" not in plain


# ---------------------------------------------------------------- the operators


def test_gather_a_crowd_places_a_spy_game_s_spy_for_nobody() -> None:
    """The second of the two edits by which the loop reaches a spy nobody owns (no stored game
    is a spy game: `tests/test_operators.py::_CANNOT_ENTER`), and its removal leaves it."""
    from chess_variant_sandbox.loop.operators import OPERATORS
    from tests.feature_inventory import _schema_one_hundred_six_uses

    crowd = next(operator for operator in OPERATORS if operator.name == "gather-a-crowd")
    parent = json.loads((HERE / "_spy_promoting.json").read_text(encoding="utf-8"))
    # The operator places in the first half of the board, which `_spy_promoting` fills: one
    # pawn fewer leaves it a square.
    pawns = next(row for row in parent["setup"]["placements"] if row["piece"] == "P")
    pawns["squares"].remove("b2")
    assert validate_spec(parent) == []
    for seed in range(40):
        child = copy.deepcopy(parent)
        if crowd.apply(child, random.Random(seed)) is None:
            continue
        assert validate_spec(child) == []
        if "setup.placements.nobody.spy" in _schema_one_hundred_six_uses(child):
            assert [piece["id"] for piece in child["pieces"]] == [
                piece["id"] for piece in parent["pieces"]
            ]
            break
    else:
        pytest.fail("gather-a-crowd never placed the spy for nobody in 40 seeds")
    for seed in range(40):
        left = copy.deepcopy(child)
        if crowd.apply(left, random.Random(seed)) is not None and not _schema_one_hundred_six_uses(left):
            return
    pytest.fail("gather-a-crowd never took the nobody's spy away in 40 seeds")


# ---------------------------------------------------------------- walks and players


def _walk(engine: Engine, seed: int, limit: int = 160) -> tuple[int, int]:
    """A random walk through the core, each step the oracle's: whether a nobody's spy was
    taken, and the spy drops played."""
    rng = random.Random(seed)
    state = engine.initial_state()
    spy = engine.spec.piece_ids.index("S")
    drops = 0
    for _ in range(limit):
        if engine.terminal(state) is not None:
            break
        _phase(engine, state)
        if engine.phase(state).is_round:
            lists = engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1)
            assert lists == oracle.legal(engine, state)
            pair = rng.choice(lists[0]), rng.choice(lists[1])
            played: tuple[int, ...] = pair
            after = engine.apply_round(state, *pair)
            assert after == oracle.round_(engine, state, *pair)
        else:
            moves = engine.legal_moves(state)
            assert moves == oracle.legal(engine, state)
            move = rng.choice(moves)
            played = (move,)
            after = engine.apply(state, move)
            assert after == oracle.turn(engine, state, move)
        drops += sum(
            1 for m in played if decode_move(m)[2] == MOVE_DROP and decode_move(m)[3] == spy
        )
        assert after.key == engine._compute_hash(after)
        state = after
    return int(_nobodys_spies(engine, state) == 0), drops


@NAMES
def test_random_walks_take_the_nobody_s_spy_as_the_oracle_does(name: str) -> None:
    engine = _engine(name)
    taken = drops = 0
    for seed in range(16):
        gone, dropped = _walk(engine, seed)
        taken += gone
        drops += dropped
    assert taken > 0, "no walk took the nobody's spy"
    if name == "hand":
        assert drops > 0, "no walk dropped a spy"


@NAMES
@pytest.mark.parametrize(
    "agents", (("random", "minimax:nodes=150"), ("greedy", "random")), ids="-".join,
)
def test_the_core_plays_each_twin_as_the_oracles_do(name: str, agents: tuple[str, str]) -> None:
    engine = _engine(name, plies=40)
    native = play_game(engine, agents, 5)
    python = play_game_python(engine, agents, 5)
    assert native.moves == python.moves
    assert (native.winner, native.reason) == (python.winner, python.reason)
    assert native.nodes == python.nodes
