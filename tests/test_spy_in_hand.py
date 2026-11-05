"""SR7j: a spy in the hand (Y7, Y8, `docs/plan-schema-next.md` §2.12, *The gained spy, said
once*).

Two twins of `_spy_promoting` (spy small, the phase both ways since SR7h), each with a hand:

* **stock** -- each player starts with a spy in its stock (`drops.stock`, `from_captures`
  false): a spy in the hand is held, not on the board, so it counts for nobody until it is
  dropped, and dropping it gains it (Y7);
* **hand** -- captures fill the hand (`from_captures`, the default): a taken spy is lost by its
  side at once and goes to the taker's hand as the taker's, and dropping it gains it (Y8).

Each transition on a scratch position, both seats, held to `tests/spy_oracle.py` step by step
(the turn by the alternating twin, the round by the round oracle, the phase off the board),
the key to the slow one, `perft_spy` to the oracle's count; a drop into an announced round
that meets a piece collides as any drop does; then random walks, the core's players against
their Python oracles, the pass's tallies, and the refusal of a spy for nobody.
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
from chess_variant_sandbox.spec.schema import (
    foreign_spy_gains,
    spy_drop_ids,
    spy_gain_routes,
    validate_spec,
)
from chess_variant_sandbox.tournament.analysis import SpyStep, blank_spy_counts, count_spy_holders
from chess_variant_sandbox.tournament.play import play_game, play_game_python
from tests import spy_oracle as oracle
from tests.test_spy_gained import _find, _flip, _perft_agrees, _phase, _square
from tests.test_spy_gained import _position as _bare

HERE = Path(__file__).parent

_DROPS: dict[str, dict[str, Any]] = {
    "stock": {"stock": {"S": 1}, "from_captures": False},
    "hand": {"from_captures": True},
}


def _spec(name: str) -> dict[str, Any]:
    spec = json.loads((HERE / "_spy_promoting.json").read_text(encoding="utf-8"))
    spec["rules"]["drops"] = copy.deepcopy(_DROPS[name])
    spec["name"] = f"spy in the {name}"
    return spec


@cache
def _engine(name: str, plies: int | None = None) -> Engine:
    spec = _spec(name)
    if plies is not None:
        spec["rules"]["max_plies"] = plies
    return Engine(spec)


def _spy(engine: Engine) -> int:
    return engine.spec.piece_ids.index("S")


def _spies(engine: Engine, state: GameState, player: int) -> int:
    return sum(
        1 for value in state.board
        if value and engine.spec.owner_of(value) == player
        and engine.spec.type_of(value) == _spy(engine)
    )


def _held(engine: Engine, state: GameState, player: int) -> int:
    return state.hands[player][_spy(engine)]


def _with_hands(
    engine: Engine, state: GameState, seat: int, mine: int, theirs: int,
) -> GameState:
    """``state`` with ``mine`` spies in ``seat``'s hand and ``theirs`` in the other's."""
    hands = [[0] * engine.spec.n_types, [0] * engine.spec.n_types]
    hands[seat][_spy(engine)] = mine
    hands[1 - seat][_spy(engine)] = theirs
    held = state.with_(hands=(tuple(hands[0]), tuple(hands[1])))
    key = engine._compute_hash(held)
    held = held.with_(key=key, repetition=(key,))
    assert engine.terminal(held) is None
    return held


def _position(
    engine: Engine, mine: dict[str, str], theirs: dict[str, str], seat: int, side: int,
) -> GameState:
    """`test_spy_gained._position` with both hands empty (`make_state` sizes them since the
    fix after SR7j; set again here so the key and the terminal check are read once more)."""
    return _with_hands(engine, _bare(engine, mine, theirs, seat, side), seat, 0, 0)


def _drop(engine: Engine, moves: list[int], seat: int, square: str) -> int:
    target = engine._square_index(_square(seat, square))
    found = [
        m for m in moves
        if decode_move(m)[2] == MOVE_DROP and decode_move(m)[1] == target
        and decode_move(m)[3] == _spy(engine)
    ]
    assert len(found) == 1, square
    return found[0]


def _word(engine: Engine, moves: list[int], seat: int, word: tuple[str, ...]) -> int:
    """``("drop", square)`` for a spy's drop, else ``(origin, target)``."""
    if word[0] == "drop":
        return _drop(engine, moves, seat, word[1])
    return _find(engine, moves, seat, *word)


def _turn(engine: Engine, state: GameState, seat: int, *word: str) -> GameState:
    move = _word(engine, engine.legal_moves(state), seat, word)
    after = engine.apply(state, move)
    assert after == oracle.turn(engine, state, move)
    assert after.key == engine._compute_hash(after)
    return after


def _moves(
    engine: Engine, state: GameState, seat: int, mine: tuple[str, ...], theirs: tuple[str, ...],
) -> list[int]:
    moves = [0, 0]
    for player, word in ((seat, mine), (1 - seat, theirs)):
        moves[player] = _word(engine, engine.legal_moves_for(state, player), seat, word)
    return moves


def _round(
    engine: Engine, state: GameState, seat: int, mine: tuple[str, ...], theirs: tuple[str, ...],
) -> GameState:
    moves = _moves(engine, state, seat, mine, theirs)
    after = engine.apply_round(state, moves[0], moves[1])
    assert after == oracle.round_(engine, state, moves[0], moves[1])
    assert after.key == engine._compute_hash(after)
    return after


SEATS = pytest.mark.parametrize("seat", (0, 1))


@pytest.mark.parametrize("name", sorted(_DROPS))
def test_each_twin_validates_and_names_the_drop_as_a_route(name: str) -> None:
    spec = _spec(name)
    assert validate_spec(spec) == []
    assert foreign_spy_gains(spec) == []
    assert spy_drop_ids(spec) == ["S"]
    assert spy_gain_routes(spec) == ["promotion", "drops"]


def test_a_demoted_spy_reaches_no_hand_as_a_spy() -> None:
    spec = _spec("hand")
    spy = next(piece for piece in spec["pieces"] if piece["id"] == "S")
    spy["promoted_from"] = "P"
    assert validate_spec(spec) == []
    assert spy_drop_ids(spec) == []
    assert spy_gain_routes(spec) == ["promotion"]


# ---------------------------------------------------------------- Y7: a spy in the stock


@pytest.mark.parametrize("name", sorted(_DROPS))
@SEATS
def test_a_spy_in_the_hand_counts_for_nobody_and_a_drop_on_the_reply_gains_it(
    name: str, seat: int,
) -> None:
    engine = _engine(name)
    board = _position(engine, {"a1": "K"}, {"a5": "K", "e5": "S"}, seat, side=0)
    state = _with_hands(engine, board, seat, 1, 0)
    # A spy in the hand is no spy: ``seat`` has none, and it is its turn, the reply.
    assert _phase(engine, state) == ("reply", -1)
    after = _turn(engine, state, seat, "drop", "c3")
    assert _spies(engine, after, seat) == 1 and _held(engine, after, seat) == 0
    assert _phase(engine, after) == ("alternating", -1) and after.side_to_move == 1 - seat
    _perft_agrees(engine, state, 3)


@SEATS
def test_a_drop_in_an_announced_round_returns_to_turns_the_other_first(seat: int) -> None:
    engine = _engine("stock")
    board = _position(engine, {"a1": "K"}, {"a5": "K", "e5": "S"}, seat, side=1)
    state = _with_hands(engine, board, seat, 1, 0)
    assert _phase(engine, state) == ("announced", seat)
    after = _round(engine, state, seat, ("drop", "c3"), ("a5", "b5"))
    assert _spies(engine, after, seat) == 1
    # The player who did not gain answers the gain.
    assert _phase(engine, after) == ("alternating", -1) and after.side_to_move == 1 - seat
    _perft_agrees(engine, state, 2)


@SEATS
def test_a_drop_in_a_blind_round_starts_announced_rounds(seat: int) -> None:
    engine = _engine("stock")
    for side in (0, 1):
        board = _position(engine, {"a1": "K"}, {"a5": "K"}, seat, side=side)
        state = _with_hands(engine, board, seat, 1, 1)
        # Both hold a spy in the hand and neither has one on the board: blind rounds.
        assert _phase(engine, state) == ("blind", -1)
        after = _round(engine, state, seat, ("drop", "c3"), ("a5", "b5"))
        assert _phase(engine, after) == ("announced", 1 - seat) and after.side_to_move == seat
    _perft_agrees(engine, state, 2)


def test_both_dropping_in_one_round_returns_to_turns_the_first_player_first() -> None:
    engine = _engine("stock")
    board = _position(engine, {"a1": "K"}, {"a5": "K"}, 0, side=1)
    state = _with_hands(engine, board, 0, 1, 1)
    after = _round(engine, state, 0, ("drop", "c2"), ("drop", "c4"))
    assert _spies(engine, after, 0) == 1 and _spies(engine, after, 1) == 1
    assert _phase(engine, after) == ("alternating", -1) and after.side_to_move == 0


@pytest.mark.parametrize("name", sorted(_DROPS))
@SEATS
def test_a_drop_into_an_announced_round_collides_as_any_drop_does(name: str, seat: int) -> None:
    engine = _engine(name)
    # ``seat`` announces, its spy in the hand; the holder's rook comes down onto the square
    # the spy is dropped on. The two meet as a drop meets any move (`docs/reference/rounds.md`
    # §4): the spy captures nothing and the rook may take it, so the rook stands there, the
    # spy is spent from the hand and never stood, and the rounds stay announced. Under a
    # capture hand the taken spy goes to the rook's owner's hand (Y8).
    board = _position(engine, {"a1": "K"}, {"a5": "K", "c5": "R", "e5": "S"}, seat, side=1)
    state = _with_hands(engine, board, seat, 1, 0)
    assert _phase(engine, state) == ("announced", seat)
    moves = _moves(engine, state, seat, ("drop", "c3"), ("c5", "c3"))
    after, _, events = engine.apply_round_with_phase(state, moves[0], moves[1])
    assert after == oracle.round_(engine, state, moves[0], moves[1])
    met = engine._square_index(_square(seat, "c3"))
    assert any(square == met for _, square, _ in events)
    assert engine.spec.piece_ids[engine.spec.type_of(after.board[met])] == "R"
    assert engine.spec.owner_of(after.board[met]) == 1 - seat
    assert _spies(engine, after, seat) == 0 and _held(engine, after, seat) == 0
    assert _held(engine, after, 1 - seat) == (1 if name == "hand" else 0)
    assert _phase(engine, after) == ("announced", seat)
    _perft_agrees(engine, state, 2)


# ---------------------------------------------------------------- Y8: a spy taken into a hand


@SEATS
def test_a_taken_spy_is_lost_at_once_and_is_its_taker_s_to_drop(seat: int) -> None:
    engine = _engine("hand")
    state = _position(
        engine, {"a1": "K", "c1": "R", "e1": "S"}, {"a5": "K", "c4": "S"}, seat, side=0,
    )
    assert _phase(engine, state) == ("alternating", -1)
    taken = _turn(engine, state, seat, "c1", "c4")
    # Lost by its side at the capture: the other replies; the spy is the taker's, in hand.
    assert _spies(engine, taken, 1 - seat) == 0 and _held(engine, taken, seat) == 1
    assert _held(engine, taken, 1 - seat) == 0
    assert _phase(engine, taken) == ("reply", -1)
    replied = _turn(engine, taken, seat, "a5", "b5")
    assert _phase(engine, replied) == ("announced", 1 - seat)
    # The taker drops it beside its own: a buffer, the rounds still announced.
    after = _round(engine, replied, seat, ("drop", "c3"), ("b5", "a5"))
    assert _spies(engine, after, seat) == 2 and _held(engine, after, seat) == 0
    assert _phase(engine, after) == ("announced", 1 - seat)
    _perft_agrees(engine, state, 3)


@SEATS
def test_a_spyless_taker_drops_the_taken_spy_out_of_the_blind_rounds(seat: int) -> None:
    engine = _engine("hand")
    # ``seat`` announces, without a spy; its rook takes the holder's only spy in the round.
    state = _position(engine, {"a1": "K", "c1": "R"}, {"a5": "K", "c4": "S"}, seat, side=1)
    assert _phase(engine, state) == ("announced", seat)
    taken = _round(engine, state, seat, ("c1", "c4"), ("a5", "b5"))
    assert _held(engine, taken, seat) == 1
    assert _phase(engine, taken) == ("blind", -1)
    after = _round(engine, taken, seat, ("drop", "c2"), ("b5", "a5"))
    # The taker holds the only spy, so the other announces.
    assert _phase(engine, after) == ("announced", 1 - seat) and after.side_to_move == seat
    _perft_agrees(engine, state, 2)


# ---------------------------------------------------------------- walks and players


def _walk(engine: Engine, seed: int, limit: int = 160) -> tuple[list[str], int]:
    """A random walk through the core, each step the oracle's: the phases it passed, and the
    spy drops played."""
    rng = random.Random(seed)
    state = engine.initial_state()
    seen: list[str] = []
    drops = 0
    spy = _spy(engine)
    for _ in range(limit):
        if engine.terminal(state) is not None:
            break
        seen.append(_phase(engine, state)[0])
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
    return seen, drops


@pytest.mark.parametrize("name", sorted(_DROPS))
def test_random_walks_drop_spies_and_cross_the_phase_both_ways_as_the_oracle_does(
    name: str,
) -> None:
    order = {"alternating": 0, "reply": 1, "announced": 2, "blind": 3}
    engine = _engine(name)
    back = drops = 0
    for seed in range(24):
        seen, dropped = _walk(engine, seed)
        drops += dropped
        back += sum(1 for a, b in zip(seen, seen[1:], strict=False) if order[b] < order[a])
    assert drops > 0, "no walk dropped a spy"
    assert back > 0, "no walk went back a phase"


@pytest.mark.parametrize("name", sorted(_DROPS))
@pytest.mark.parametrize(
    "agents", (("greedy", "minimax:nodes=150"), ("random", "minimax:nodes=200")), ids="-".join,
)
def test_the_core_plays_each_twin_as_the_oracles_do(name: str, agents: tuple[str, str]) -> None:
    engine = _engine(name, plies=40)
    native = play_game(engine, agents, 5)
    python = play_game_python(engine, agents, 5)
    assert native.moves == python.moves
    assert (native.winner, native.reason) == (python.winner, python.reason)
    assert native.nodes == python.nodes


# ---------------------------------------------------------------- the pass's tallies


def _tally(engine: Engine, state: GameState, seat: int, *words: tuple[str, ...]) -> dict[str, Any]:
    phase = engine.phase(state)
    if phase.is_round:
        moves = _moves(engine, state, seat, words[0], words[1])
        lists: tuple[list[int], ...] = (
            engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1),
        )
        after, _, events = engine.apply_round_with_phase(state, moves[0], moves[1])
        step = SpyStep(0, state, phase, tuple(moves), lists, after, events)
    else:
        move = _word(engine, engine.legal_moves(state), seat, words[0])
        step = SpyStep(
            0, state, phase, (move,), (engine.legal_moves(state),), engine.apply(state, move), (),
        )
    spy = blank_spy_counts()
    count_spy_holders(engine, spy, step, 7)
    return {key: value for key, value in spy.items() if value not in (None, 0)}


@SEATS
def test_the_pass_reads_a_drop_s_gain_and_a_capture_into_the_hand_s_loss(seat: int) -> None:
    them = 1 - seat
    engine = _engine("hand")
    state = _position(
        engine, {"a1": "K", "c1": "R", "e1": "S"}, {"a5": "K", "c4": "S"}, seat, side=0,
    )
    assert _tally(engine, state, seat, ("c1", "c4")) == {
        f"lost_ply_{them}": 7, f"lost_how_{them}": "capture",
    }
    board = _position(engine, {"a1": "K"}, {"a5": "K", "e5": "S"}, seat, side=0)
    state = _with_hands(engine, board, seat, 1, 0)
    assert _tally(engine, state, seat, ("drop", "c3")) == {
        f"gained_ply_{seat}": 7, f"gained_how_{seat}": "drop",
    }
    board = _position(engine, {"a1": "K"}, {"a5": "K"}, seat, side=0)
    state = _with_hands(engine, board, seat, 1, 0)
    assert _tally(engine, state, seat, ("drop", "c3"), ("a5", "b5")) == {
        f"gained_ply_{seat}": 7, f"gained_how_{seat}": "drop",
    }


def test_the_description_says_a_spy_in_the_hand_counts_for_nobody() -> None:
    from chess_variant_sandbox.spec.describe import describe_spec

    stock = describe_spec(_spec("stock"))
    hand = describe_spec(_spec("hand"))
    assert "a spy dropped from the hand" in stock
    assert "A spy in a hand counts for nobody until it is dropped." in stock
    assert "lost by its side at the capture and is its taker's to drop" in hand


# ---------------------------------------------------------------- the operators


def test_give_a_handicap_stocks_a_spy_game_s_spy() -> None:
    """The second of the two edits by which the loop reaches a spy in a stock (no stored game
    is a spy game: `tests/test_operators.py::_CANNOT_ENTER`)."""
    from chess_variant_sandbox.loop.operators import OPERATORS
    from tests.feature_inventory import _schema_one_hundred_five_uses

    handicap = next(operator for operator in OPERATORS if operator.name == "give-a-handicap")
    parent = json.loads((HERE / "_spy_promoting.json").read_text(encoding="utf-8"))
    for seed in range(40):
        child = copy.deepcopy(parent)
        if handicap.apply(child, random.Random(seed)) is None:
            continue
        assert validate_spec(child) == []
        if "rules.drops.stock.spy" in _schema_one_hundred_five_uses(child):
            return
    pytest.fail("give-a-handicap never stocked the spy in 40 seeds")


# ---------------------------------------------------------------- a spy for nobody


def _unmirrored(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` with its mirrored setup written out per player, player 1's spy a type of its
    own (``T``), so the two players' spy types differ."""
    other = copy.deepcopy(next(piece for piece in spec["pieces"] if piece["id"] == "S"))
    other.update(id="T", name="other spy")
    spec["pieces"].append(other)
    placements = []
    for placement in spec["setup"]["placements"]:
        placements.append({**placement, "player": 0})
        piece = "T" if placement["piece"] == "S" else placement["piece"]
        placements.append(
            {"piece": piece, "squares": [_flip(sq) for sq in placement["squares"]], "player": 1}
        )
    spec["setup"] = {"mirror": False, "placements": placements}
    for rule in [spec["rules"]["promotion"]]:
        rule["to_by_player"] = {"0": ["R", "N", "S"], "1": ["R", "N", "T"]}
        rule.pop("to")
    return spec


def test_a_drop_is_refused_by_name_only_where_it_makes_a_spy_for_nobody() -> None:
    spec = _unmirrored(_spec("stock"))
    spec["rules"]["drops"] = {"stock_by_player": {"0": {"S": 1}, "1": {"T": 1}}, "from_captures": False}
    assert validate_spec(spec) == []
    assert spy_drop_ids(spec) == ["S", "T"]
    spec["rules"]["drops"] = {"stock": {"S": 1}, "from_captures": False}
    errors = validate_spec(spec)
    assert len(errors) == 1 and "gives player 1 the spy 'S' to drop" in errors[0]
    assert foreign_spy_gains(spec) == errors
    # A hand filled from captures takes the other's spy type: refused for both players.
    spec["rules"]["drops"] = {"from_captures": True}
    errors = validate_spec(spec)
    assert len(errors) == 2
    assert "fills player 0's hand with the spy 'T'" in errors[0]
    assert "fills player 1's hand with the spy 'S'" in errors[1]
    # Demoted in the hand, a taken spy comes back as a pawn: nothing to refuse.
    for piece in spec["pieces"]:
        if piece["id"] in ("S", "T"):
            piece["promoted_from"] = "P"
    assert validate_spec(spec) == []
