"""SR7l: a price on a spy (O24, `docs/plan-schema-next.md` §2.11, ruled again 2026-10-29: lifted).

A price is what a player spends to buy, hire or bribe a piece of the type, and only that: it
leaves the buyer's purse and is paid to nobody. **Taken**, a priced spy pays its `bounty`, if it
has one, as any piece's taking does, never its price. **Gained** by a rule that makes a taken
piece the taker's (a side switch, Y5; a capture hand, Y8) or any other gain, no coins move: the
gain is §2.12's, and a dropped spy costs nothing. In a spy game only a market can charge the
price (SR7m sells it): a hire or a bribe stays refused beside rounds (R19), so a priced spy no
market sells is refused by the purse's own *nothing charges it*.

The validator's part is held on `_spy_promoting` with a purse. The coins are held on three of
its twins (taken; switched, SR7i's `switch`; taken into the hand, SR7j's `hand`), each with a
priced, bountied spy and a market that sells the knight: until SR7m a market cannot sell the
spy, so the validator refuses each twin (*nothing charges it*), and they are played past it,
as the core plays any spec it is handed (SR2's probe), every step held to `tests/spy_oracle.py`.
"""

from __future__ import annotations

import copy
import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine import compile as compile_module
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from tests.test_spy_gained import _flip, _phase
from tests.test_spy_in_hand import _held, _spies, _turn, _with_hands

HERE = Path(__file__).parent
REGION = ["a3", "b3", "c3", "d3", "e3"]
PRICE, BOUNTY, START = 4, 2, 6


def _priced(spec: dict[str, Any], market: tuple[str, ...] = ("N",)) -> dict[str, Any]:
    """``spec`` with a purse whose market sells ``market`` onto the middle rank, the knight at
    3, the spy at `PRICE` with a `BOUNTY`."""
    spec = copy.deepcopy(spec)
    spec["rules"]["purse"] = {"start": START, "market": list(market), "region": list(REGION)}
    by_id = {piece["id"]: piece for piece in spec["pieces"]}
    by_id["N"]["price"] = 3
    by_id["S"]["price"] = PRICE
    by_id["S"]["bounty"] = BOUNTY
    return spec


def _base() -> dict[str, Any]:
    return json.loads((HERE / "_spy_promoting.json").read_text(encoding="utf-8"))


def _twin(name: str) -> dict[str, Any]:
    if name == "switch":
        from tests.test_spy_gained_on_board import _spec as gained
    elif name == "hand":
        from tests.test_spy_in_hand import _spec as gained
    else:
        return _priced(_base())
    return _priced(gained(name))


@cache
def _engine(name: str) -> Engine:
    """The twin compiled past the validator, which refuses it only for its price until SR7m."""
    spec = _twin(name)
    errors = validate_spec(spec)
    assert errors and all("which nothing charges" in error for error in errors), errors
    return Engine(spec)


@pytest.fixture(autouse=True)
def _past_the_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every compile here, the oracle's alternating twin's included, past the validator."""
    monkeypatch.setattr(compile_module, "validate_or_raise", lambda spec: None)


def _position(
    engine: Engine, mine: dict[str, str], theirs: dict[str, str], seat: int,
) -> GameState:
    placement: dict[str, tuple[str, int]] = {}
    for squares, owner in ((mine, seat), (theirs, 1 - seat)):
        for square, piece in squares.items():
            placement[square if seat == 0 else _flip(square)] = (piece, owner)
    state = engine.make_state(placement, side_to_move=seat)
    assert engine.terminal(state) is None and state.purse == (START, START)
    return state


def _coins(state: GameState, seat: int) -> tuple[int, int]:
    """``seat``'s purse, then the other's."""
    return state.purse[seat], state.purse[1 - seat]


SEATS = pytest.mark.parametrize("seat", (0, 1))


# ---------------------------------------------------------------- the validator


def test_the_refusal_is_gone_and_a_priced_spy_no_market_sells_is_charged_by_nothing() -> None:
    spec = _priced(_base())
    errors = validate_spec(spec)
    assert not [error for error in errors if "never gained" in error]
    assert errors == [
        "pieces[4].price on 'S', which nothing charges: a price is what a market "
        "(rules.purse.market), a hire (a placement for nobody) or a bribe "
        "(rules.purse.bribe) costs, and none reads it here"
    ]


def test_a_spy_game_that_bribes_stays_refused_beside_rounds() -> None:
    spec = _priced(_base())
    spec["rules"]["purse"]["bribe"] = True
    errors = validate_spec(spec)
    assert "R19" in refused_ids(errors)
    assert not [error for error in errors if "never gained" in error]


def test_a_priced_spy_placed_for_nobody_stays_refused_beside_rounds() -> None:
    """A hire: R19's narrower row, never a spy refusal."""
    spec = _priced(_base())
    spec["setup"]["placements"].append({"piece": "S", "squares": ["c3"], "player": "nobody"})
    errors = validate_spec(spec)
    assert "R19" in refused_ids(errors)
    assert not [error for error in errors if "never gained" in error]


# ---------------------------------------------------------------- the coins


@SEATS
def test_a_priced_spy_taken_pays_its_bounty_and_not_its_price(seat: int) -> None:
    engine = _engine("taken")
    state = _position(
        engine, {"a1": "K", "c1": "R", "e1": "S"}, {"a5": "K", "c4": "S", "e5": "S"}, seat,
    )
    after = _turn(engine, state, seat, "c1", "c4")
    assert _spies(engine, after, 1 - seat) == 1
    assert _coins(after, seat) == (START + BOUNTY, START)
    points = after.points[seat] - state.points[seat]
    assert points == BOUNTY  # the tally: the bounty, never the price
    assert _phase(engine, after) == ("alternating", -1)


@SEATS
def test_a_priced_spy_that_switches_sides_moves_no_coins(seat: int) -> None:
    engine = _engine("switch")
    state = _position(
        engine, {"a1": "K", "c3": "S", "e1": "S"}, {"a5": "K", "c4": "P", "e5": "S"}, seat,
    )
    after = _turn(engine, state, seat, "c3", "c4")
    assert _spies(engine, after, 1 - seat) == 2 and _spies(engine, after, seat) == 1
    assert _coins(after, seat) == (START, START)


@SEATS
def test_a_priced_spy_taken_into_the_hand_pays_its_bounty_and_its_drop_costs_nothing(
    seat: int,
) -> None:
    engine = _engine("hand")
    state = _with_hands(
        engine,
        _position(engine, {"a1": "K", "c1": "R", "e1": "S"}, {"a5": "K", "c4": "S", "e5": "S"}, seat),
        seat, 0, 0,
    )
    taken = _turn(engine, state, seat, "c1", "c4")
    assert _held(engine, taken, seat) == 1
    assert _coins(taken, seat) == (START + BOUNTY, START)
    answered = _turn(engine, taken, seat, "a5", "b5")
    dropped = _turn(engine, answered, seat, "drop", "c3")
    assert _spies(engine, dropped, seat) == 2 and _held(engine, dropped, seat) == 0
    assert _coins(dropped, seat) == (START + BOUNTY, START)
