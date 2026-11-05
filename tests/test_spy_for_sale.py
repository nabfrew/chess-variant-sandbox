"""SR7m: a spy for sale (O27, `docs/plan-schema-next.md` §2.11, ruled again 2026-10-29: lifted;
the user: *"if there's no longer a reason to refuse a spy to be priced and bought it should be
allowed"*).

A bought spy is a spy **gained** at the moment it stands on the board, and its effect on secrecy
is §2.12's for any gain: bought while its buyer has one, a buffer; bought on the buyer's reply
turn, ordinary turns with the other player next; bought by the announcer in an announced round
and standing after it, ordinary turns with the player who did not buy first; bought in a blind
round and standing after it, announced rounds with the other side announcing (the token dealt
again under `tie: "initiative"`). In a round the buy is an arrival with the round's moves (SR7g):
it collides by `beats` and the tie, its price is paid whether or not it survives, and a bought
spy that loses its collision never stood, so nothing is gained. Supply is unlimited.

One twin of `_spy_promoting` (spy small, the phase both ways since SR7h) with a purse whose
market sells the spy onto the middle rank. Each transition on a scratch position, both seats,
held to `tests/spy_oracle.py` step by step (the turn by the alternating twin, the round by the
round oracle, which keeps the purse and the buy since SR7g), `perft_spy` to the oracle's count
(make and unmake across the buy); then random walks that buy spies, the core's players against
their Python oracles, a searching player to a game's end, the pass's tallies and the refusals.
"""

from __future__ import annotations

import json
import random
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_BUY, decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.schema import (
    foreign_spy_gains,
    spy_gain_routes,
    spy_market_ids,
    validate_spec,
)
from chess_variant_sandbox.tournament.analysis import SpyStep, blank_spy_counts, count_spy_holders
from chess_variant_sandbox.tournament.play import play_game, play_game_python
from tests import spy_oracle as oracle
from tests.test_spy_gained import _find, _flip, _perft_agrees, _phase

HERE = Path(__file__).parent
REGION = ["a3", "b3", "c3", "d3", "e3"]
PRICE, START = 4, 6


def _spec(tie: str | None = None) -> dict[str, Any]:
    spec = json.loads((HERE / "_spy_promoting.json").read_text(encoding="utf-8"))
    spec["rules"]["purse"] = {"start": START, "market": ["S"], "region": list(REGION)}
    if tie is not None:
        spec["rules"]["simultaneous"] = {"tie": tie}
    next(piece for piece in spec["pieces"] if piece["id"] == "S")["price"] = PRICE
    spec["name"] = "spy for sale" + (f" ({tie})" if tie else "")
    return spec


@cache
def _engine(tie: str | None = None, plies: int | None = None) -> Engine:
    spec = _spec(tie)
    if plies is not None:
        spec["rules"]["max_plies"] = plies
    return Engine(spec)


def _spy(engine: Engine) -> int:
    return engine.spec.piece_ids.index("S")


def _spies(engine: Engine, state: GameState, player: int) -> int:
    return sum(
        1 for value in state.board
        if value and engine.spec.owner_of(value) == player and engine.spec.type_of(value) == _spy(engine)
    )


def _position(
    engine: Engine, mine: dict[str, str], theirs: dict[str, str], seat: int, side: int,
) -> GameState:
    placement: dict[str, tuple[str, int]] = {}
    for squares, owner in ((mine, seat), (theirs, 1 - seat)):
        for square, piece in squares.items():
            placement[square if seat == 0 else _flip(square)] = (piece, owner)
    state = engine.make_state(placement, side_to_move=seat if side == 0 else 1 - seat)
    assert engine.terminal(state) is None and state.purse == (START, START)
    return state


def _buy(engine: Engine, moves: list[int], seat: int, square: str) -> int:
    target = engine._square_index(square if seat == 0 else _flip(square))
    found = [
        m for m in moves
        if decode_move(m)[2] == MOVE_BUY and decode_move(m)[1] == target
        and decode_move(m)[3] == _spy(engine)
    ]
    assert len(found) == 1, square
    return found[0]


def _word(engine: Engine, moves: list[int], seat: int, word: tuple[str, ...]) -> int:
    """``("buy", square)`` for a spy's buy, else ``(origin, target)``."""
    if word[0] == "buy":
        return _buy(engine, moves, seat, word[1])
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


def _coins(state: GameState, seat: int) -> tuple[int, int]:
    return state.purse[seat], state.purse[1 - seat]


SEATS = pytest.mark.parametrize("seat", (0, 1))


# ---------------------------------------------------------------- the validator


def test_the_spec_validates_and_names_the_market_as_a_route() -> None:
    spec = _spec()
    assert validate_spec(spec) == []
    assert foreign_spy_gains(spec) == []
    assert spy_market_ids(spec) == ["S"]
    assert spy_gain_routes(spec) == ["promotion", "purse.market"]
    assert validate_spec(_spec("initiative")) == []


def test_a_market_spy_needs_its_price() -> None:
    spec = _spec()
    del next(piece for piece in spec["pieces"] if piece["id"] == "S")["price"]
    assert validate_spec(spec) == ["rules.purse.market 'S' has no price: give pieces[].price, 1 to 40"]


def test_a_market_selling_the_other_player_s_spy_type_is_refused_by_name() -> None:
    """Disjoint spy types: a market sells to both players, so a buyer that does not start with
    the type would buy a spy for nobody."""
    from tests.test_spy_in_hand import _unmirrored

    spec = _unmirrored(_spec())
    errors = validate_spec(spec)
    assert "rules.purse.market sells player 1 the spy 'S', which player 1 does not start with" in " ".join(errors)
    spec["rules"]["purse"]["market"] = ["S", "T"]
    next(piece for piece in spec["pieces"] if piece["id"] == "T")["price"] = PRICE
    errors = validate_spec(spec)
    assert any("sells player 0 the spy 'T'" in error for error in errors)


# ---------------------------------------------------------------- turns


@SEATS
def test_a_bought_second_spy_is_a_buffer(seat: int) -> None:
    engine = _engine()
    state = _position(engine, {"a1": "K", "e1": "S"}, {"a5": "K", "e5": "S"}, seat, side=0)
    after = _turn(engine, state, seat, "buy", "c3")
    assert _spies(engine, after, seat) == 2
    assert _coins(after, seat) == (START - PRICE, START)
    assert _phase(engine, after) == ("alternating", -1) and after.side_to_move == 1 - seat
    _perft_agrees(engine, state, 3)


@SEATS
def test_a_buy_on_the_reply_turn_returns_to_turns(seat: int) -> None:
    engine = _engine()
    state = _position(engine, {"a1": "K"}, {"a5": "K", "e5": "S"}, seat, side=0)
    assert _phase(engine, state) == ("reply", -1)
    after = _turn(engine, state, seat, "buy", "c3")
    assert _spies(engine, after, seat) == 1
    assert _phase(engine, after) == ("alternating", -1) and after.side_to_move == 1 - seat
    _perft_agrees(engine, state, 3)


# ---------------------------------------------------------------- rounds


@SEATS
def test_the_announcer_s_buy_resumes_turns_with_the_other_player_first(seat: int) -> None:
    engine = _engine()
    state = _position(engine, {"a1": "K"}, {"a5": "K", "e5": "S"}, seat, side=1)
    assert _phase(engine, state) == ("announced", seat)
    after = _round(engine, state, seat, ("buy", "c3"), ("a5", "b5"))
    assert _spies(engine, after, seat) == 1
    assert _coins(after, seat) == (START - PRICE, START)
    assert _phase(engine, after) == ("alternating", -1) and after.side_to_move == 1 - seat
    _perft_agrees(engine, state, 2)


def test_both_buying_in_one_round_returns_to_turns_the_first_player_first() -> None:
    engine = _engine()
    state = _position(engine, {"a1": "K"}, {"a5": "K"}, 0, side=0)
    assert _phase(engine, state) == ("blind", -1)
    after = _round(engine, state, 0, ("buy", "b3"), ("buy", "d3"))
    assert _spies(engine, after, 0) == _spies(engine, after, 1) == 1
    assert after.purse == (START - PRICE, START - PRICE)
    assert _phase(engine, after) == ("alternating", -1) and after.side_to_move == 0


@pytest.mark.parametrize("tie", (None, "initiative"))
@SEATS
def test_a_buy_in_a_blind_round_starts_announced_rounds(seat: int, tie: str | None) -> None:
    engine = _engine(tie)
    state = _position(engine, {"a1": "K"}, {"a5": "K"}, seat, side=0)
    assert _phase(engine, state) == ("blind", -1)
    after = _round(engine, state, seat, ("buy", "c3"), ("a5", "b5"))
    assert _spies(engine, after, seat) == 1
    # The other side, still without, announces; under the initiative the token is dealt to it.
    assert _phase(engine, after) == ("announced", 1 - seat)
    if tie == "initiative":
        assert after.initiative == 1 - seat
    _perft_agrees(engine, state, 2)


@SEATS
def test_a_bought_spy_that_loses_its_collision_gains_nothing_and_its_price_is_spent(seat: int) -> None:
    engine = _engine()
    state = _position(engine, {"a1": "K"}, {"a5": "K", "c4": "R"}, seat, side=0)
    assert _phase(engine, state) == ("blind", -1)
    # The rook moves onto the square the spy is bought onto: the rook beats the spy, which
    # beats nothing, so the bought spy never stands.
    after = _round(engine, state, seat, ("buy", "c3"), ("c4", "c3"))
    assert _spies(engine, after, seat) == 0
    assert _coins(after, seat) == (START - PRICE, START)
    assert _phase(engine, after) == ("blind", -1)


@SEATS
def test_the_pass_reads_a_buy_s_gain(seat: int) -> None:
    engine = _engine()
    state = _position(engine, {"a1": "K"}, {"a5": "K", "e5": "S"}, seat, side=0)
    move = _buy(engine, engine.legal_moves(state), seat, "c3")
    after = engine.apply(state, move)
    step = SpyStep(0, state, engine.phase(state), (move,), (engine.legal_moves(state),), after, ())
    spy = blank_spy_counts()
    count_spy_holders(engine, spy, step, 7)
    assert {key: value for key, value in spy.items() if value not in (None, 0)} == {
        f"gained_ply_{seat}": 7, f"gained_how_{seat}": "buy",
    }


# ---------------------------------------------------------------- walks and players


def _walk(engine: Engine, seed: int, limit: int = 160) -> tuple[list[str], int]:
    """A random walk through the core, each step the oracle's: the phases it passed, and the
    spy buys played."""
    rng = random.Random(seed)
    state = engine.initial_state()
    seen: list[str] = []
    buys = 0
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
        buys += sum(
            1 for m in played if decode_move(m)[2] == MOVE_BUY and decode_move(m)[3] == _spy(engine)
        )
        assert after.key == engine._compute_hash(after)
        state = after
    return seen, buys


@pytest.mark.parametrize("tie", (None, "initiative"))
def test_random_walks_buy_spies_and_cross_the_phase_both_ways_as_the_oracle_does(
    tie: str | None,
) -> None:
    order = {"alternating": 0, "reply": 1, "announced": 2, "blind": 3}
    engine = _engine(tie)
    back = buys = 0
    for seed in range(16):
        seen, bought = _walk(engine, seed)
        buys += bought
        back += sum(1 for a, b in zip(seen, seen[1:], strict=False) if order[b] < order[a])
    assert buys > 0, "no walk bought a spy"
    assert back > 0, "no walk went back a phase"


@pytest.mark.parametrize(
    "agents", (("random", "minimax:nodes=150"), ("greedy", "random")), ids="-".join,
)
def test_the_core_plays_a_market_spy_game_as_the_oracles_do(agents: tuple[str, str]) -> None:
    engine = _engine(plies=40)
    native = play_game(engine, agents, 5)
    python = play_game_python(engine, agents, 5)
    assert native.moves == python.moves
    assert (native.winner, native.reason) == (python.winner, python.reason)
    assert native.nodes == python.nodes


def test_a_searching_announced_round_player_plays_a_market_spy_game_to_its_end() -> None:
    """Both seats searching (`minimax:nodes=N` plays every phase of a spy game, the announced
    round included), the game played to its ending, each seat's spy bought back somewhere."""
    from chess_variant_sandbox.tournament.analysis import spy_steps

    engine = _engine(plies=60)
    announced = bought = 0
    for seed in range(4):
        record = play_game(engine, ("minimax:nodes=200", "minimax:nodes=200"), seed)
        assert record.reason
        for step in spy_steps(engine, record, strict=True):
            announced += step.phase.name == "announced"
            bought += sum(
                1 for m in step.moves
                if decode_move(m)[2] == MOVE_BUY and decode_move(m)[3] == _spy(engine)
            )
    assert announced > 0, "no searched game reached an announced round"
    assert bought > 0, "no searched game bought a spy"


def test_open_a_market_sells_a_spy_game_s_spy() -> None:
    """The second of the two edits by which the loop reaches a spy for sale (no stored game is
    a spy game: `tests/test_operators.py::_CANNOT_ENTER`)."""
    from chess_variant_sandbox.loop.operators import OPERATORS
    from tests.feature_inventory import _schema_one_hundred_eight_uses

    market = next(operator for operator in OPERATORS if operator.name == "open-a-market")
    parent = json.loads((HERE / "_spy_promoting.json").read_text(encoding="utf-8"))
    for seed in range(60):
        child = json.loads(json.dumps(parent))
        if market.apply(child, random.Random(seed)) is None:
            continue
        assert validate_spec(child) == []
        if "rules.purse.market.spy" in _schema_one_hundred_eight_uses(child):
            return
    pytest.fail("open-a-market never put the spy on sale in 60 seeds")


def test_the_description_names_the_market() -> None:
    from chess_variant_sandbox.spec.describe import describe_spec

    assert "a spy bought from the market" in describe_spec(_spec())


def test_an_unlimited_supply_is_bounded_only_by_the_purse() -> None:
    """Supply is unlimited: a player buys a spy for as long as its purse holds the price."""
    engine = _engine()
    state = _position(engine, {"a1": "K", "e1": "S"}, {"a5": "K", "e5": "S"}, 0, side=0)
    state = state.with_(purse=(PRICE * 2, START))
    key = engine._compute_hash(state)
    state = state.with_(key=key, repetition=(key,))
    after = _turn(engine, state, 0, "buy", "b3")
    after = _turn(engine, after, 0, "a5", "b5")
    after = _turn(engine, after, 0, "buy", "d3")
    assert _spies(engine, after, 0) == 3 and after.purse[0] == 0
    after = _turn(engine, after, 0, "b5", "a5")
    assert not [m for m in engine.legal_moves(after) if decode_move(m)[2] == MOVE_BUY]
