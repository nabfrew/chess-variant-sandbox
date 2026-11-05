"""SR7i: a spy gained by a strike, a side switch or a link (Y4, Y5, Y6,
`docs/plan-schema-next.md` §2.12, *The gained spy, said once*).

Three twins of `_spy_promoting` (spy small, the phase both ways since SR7h), each with one
rule that can make a spy on the board:

* **chain** -- a strike that demotes where it stands (`position: "stay"`), a knight struck
  into a spy and a spy struck into a pawn: the struck piece stays its owner's, so its owner
  gains a spy (or loses one);
* **switch** -- the spy captures and changes sides (`switches_sides: ["S"]`): one move, a spy
  lost by its old side and gained by its new one;
* **link** -- a knight taking a pawn becomes a spy (`mover_to`, the mover's gain) and a rook
  taking a knight turns it into one (`onto_to`, the target's owner's gain), both in place.

Each transition on a scratch position, both seats, held to `tests/spy_oracle.py` step by step
(the turn by the alternating twin, the round by the round oracle, the phase off the board),
the key to the slow one, `perft_spy` to the oracle's count; then random walks through every
phase and the core's players against their Python oracles.
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
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.schema import spy_gain_routes, validate_spec
from chess_variant_sandbox.tournament.analysis import (
    SPY_COUNTS,
    SPY_GAINS,
    SpyStep,
    blank_spy_counts,
    count_spy_holders,
)
from chess_variant_sandbox.tournament.play import play_game, play_game_python
from tests import spy_oracle as oracle
from tests.test_spy_gained import _find, _perft_agrees, _phase, _position, _round, _turn

HERE = Path(__file__).parent

_RULES: dict[str, dict[str, Any]] = {
    "chain": {"strike": {
        "effect": "demote", "position": "stay", "immune": ["K"],
        "chain": [{"from": "N", "to": "S"}, {"from": "S", "to": "P"}],
    }},
    "switch": {"strike": {"switches_sides": ["S"], "immune": ["K"]}},
    "link": {"contacts": {
        "position": "stay", "immune": ["K"],
        "links": [
            {"mover": "N", "onto": "P", "mover_to": "S"},
            {"mover": "R", "onto": "N", "onto_to": "S"},
        ],
    }},
}


def _spec(name: str) -> dict[str, Any]:
    spec = json.loads((HERE / "_spy_promoting.json").read_text(encoding="utf-8"))
    spec["rules"].update(copy.deepcopy(_RULES[name]))
    if name == "switch":
        # A spy that captures, so that it can switch.
        spy = next(piece for piece in spec["pieces"] if piece["id"] == "S")
        spy.pop("captures")
        spy["moves"][0]["mode"] = "both"
    spec["name"] = f"spy gained by a {name}"
    return spec


@cache
def _engine(name: str, plies: int | None = None) -> Engine:
    spec = _spec(name)
    if plies is not None:
        spec["rules"]["max_plies"] = plies
    return Engine(spec)


def _spies(engine: Engine, state: GameState, player: int) -> int:
    spy = engine.spec.piece_ids.index("S")
    return sum(
        1 for value in state.board
        if value and engine.spec.owner_of(value) == player and engine.spec.type_of(value) == spy
    )


SEATS = pytest.mark.parametrize("seat", (0, 1))


@pytest.mark.parametrize("name", sorted(_RULES))
def test_each_twin_validates_and_names_its_route(name: str) -> None:
    spec = _spec(name)
    assert validate_spec(spec) == []
    route = {"chain": "strike.chain", "switch": "strike.switches_sides", "link": "contacts"}[name]
    assert spy_gain_routes(spec) == ["promotion", route]


# ---------------------------------------------------------------- Y4: a strike chain


@SEATS
def test_a_piece_struck_into_a_spy_is_its_owner_s_buffer(seat: int) -> None:
    engine = _engine("chain")
    state = _position(
        engine, {"a1": "K", "c1": "R", "e1": "S"}, {"a5": "K", "c4": "N", "e5": "S"}, seat, side=0,
    )
    assert _phase(engine, state) == ("alternating", -1)
    struck = _turn(engine, state, seat, "c1", "c4")
    assert _spies(engine, struck, 1 - seat) == 2 and _spies(engine, struck, seat) == 1
    assert _phase(engine, struck) == ("alternating", -1)
    # The gained spy struck again turns into a pawn: one lost of two, still turns.
    moved = _turn(engine, struck, seat, "a5", "b5")
    again = _turn(engine, moved, seat, "c1", "c4")
    assert _spies(engine, again, 1 - seat) == 1
    assert _phase(engine, again) == ("alternating", -1)
    _perft_agrees(engine, state, 3)


@SEATS
def test_a_spy_struck_into_something_else_is_lost(seat: int) -> None:
    engine = _engine("chain")
    state = _position(engine, {"a1": "K", "c1": "R", "e1": "S"}, {"a5": "K", "c4": "S"}, seat, side=0)
    struck = _turn(engine, state, seat, "c1", "c4")
    # The other's last spy is a pawn now, and it is their turn: the reply.
    assert _phase(engine, struck) == ("reply", -1)
    _perft_agrees(engine, state, 3)


@SEATS
def test_a_strike_that_gains_the_spyless_side_a_spy_returns_to_turns(seat: int) -> None:
    engine = _engine("chain")
    # ``seat`` holds the only spy and answers; the other announces, and is struck a spy.
    state = _position(engine, {"a1": "K", "c1": "R", "e1": "S"}, {"a5": "K", "c4": "N"}, seat, side=0)
    assert _phase(engine, state) == ("announced", 1 - seat)
    after = _round(engine, state, seat, ("c1", "c4"), ("a5", "b5"))
    assert _phase(engine, after) == ("alternating", -1)
    assert after.side_to_move == seat  # the player who did not gain answers the gain
    _perft_agrees(engine, state, 2)


@SEATS
def test_a_strike_in_a_blind_round_starts_announced_rounds(seat: int) -> None:
    engine = _engine("chain")
    for side in (0, 1):
        state = _position(engine, {"a1": "K", "c1": "R"}, {"a5": "K", "c4": "N"}, seat, side=side)
        assert _phase(engine, state) == ("blind", -1)
        after = _round(engine, state, seat, ("c1", "c4"), ("a5", "b5"))
        # The struck side gained and holds the spy; the striker, still without, announces.
        assert _phase(engine, after) == ("announced", seat)
        assert after.side_to_move == 1 - seat
    _perft_agrees(engine, state, 2)


# ---------------------------------------------------------------- Y5: a side switch


@SEATS
def test_a_switched_spy_is_lost_by_one_side_and_gained_by_the_other_in_one_move(seat: int) -> None:
    engine = _engine("switch")
    state = _position(
        engine, {"a1": "K", "c3": "S"}, {"a5": "K", "d3": "N", "e5": "S"}, seat, side=0,
    )
    assert _phase(engine, state) == ("alternating", -1)
    after = _turn(engine, state, seat, "c3", "d3")
    assert _spies(engine, after, seat) == 0 and _spies(engine, after, 1 - seat) == 2
    # The mover lost its last spy on its own move, so no reply is its own: announced rounds,
    # the mover announcing, the other (the holder of both) to move.
    assert _phase(engine, after) == ("announced", seat)
    assert after.side_to_move == 1 - seat
    _perft_agrees(engine, state, 3)


@SEATS
def test_a_spy_switching_in_a_round_changes_the_holder(seat: int) -> None:
    engine = _engine("switch")
    # The other holds the only spy and answers; ``seat`` announces. The spy takes ``seat``'s
    # knight and joins ``seat``: the holder changes, the rounds stay announced.
    state = _position(engine, {"a1": "K", "c3": "N"}, {"a5": "K", "d3": "S"}, seat, side=1)
    assert _phase(engine, state) == ("announced", seat)
    after = _round(engine, state, seat, ("a1", "b1"), ("d3", "c3"))
    assert _spies(engine, after, seat) == 1 and _spies(engine, after, 1 - seat) == 0
    assert _phase(engine, after) == ("announced", 1 - seat) and after.side_to_move == seat
    _perft_agrees(engine, state, 2)


# ---------------------------------------------------------------- Y6: a contact link


@SEATS
def test_a_link_on_the_reply_turn_gains_the_mover_a_spy_and_returns_to_turns(seat: int) -> None:
    engine = _engine("link")
    state = _position(engine, {"a1": "K", "c1": "N"}, {"a5": "K", "d3": "P", "e5": "S"}, seat, side=0)
    assert _phase(engine, state) == ("reply", -1)
    after = _turn(engine, state, seat, "c1", "d3")  # mover_to: the knight becomes a spy
    assert _spies(engine, after, seat) == 1
    assert _phase(engine, after) == ("alternating", -1) and after.side_to_move == 1 - seat
    _perft_agrees(engine, state, 3)


@SEATS
def test_a_link_that_makes_the_target_a_spy_gains_it_for_the_target_s_owner(seat: int) -> None:
    engine = _engine("link")
    # ``seat`` holds the only spy; its rook takes the announcer's knight, which becomes the
    # announcer's spy (onto_to): turns again, ``seat`` (who did not gain) first.
    state = _position(engine, {"a1": "K", "c1": "R", "e1": "S"}, {"a5": "K", "c4": "N"}, seat, side=0)
    assert _phase(engine, state) == ("announced", 1 - seat)
    after = _round(engine, state, seat, ("c1", "c4"), ("a5", "b5"))
    assert _spies(engine, after, 1 - seat) == 1
    assert _phase(engine, after) == ("alternating", -1) and after.side_to_move == seat
    _perft_agrees(engine, state, 2)


@SEATS
def test_a_link_in_a_blind_round_starts_announced_rounds(seat: int) -> None:
    engine = _engine("link")
    state = _position(engine, {"a1": "K", "c1": "N"}, {"a5": "K", "d3": "P"}, seat, side=0)
    assert _phase(engine, state) == ("blind", -1)
    after = _round(engine, state, seat, ("c1", "d3"), ("a5", "b5"))
    assert _phase(engine, after) == ("announced", 1 - seat) and after.side_to_move == seat
    _perft_agrees(engine, state, 2)


# ---------------------------------------------------------------- walks and players


def _walk(engine: Engine, seed: int, limit: int = 160) -> list[str]:
    """A random walk through the core, each step the oracle's; the phases it passed."""
    rng = random.Random(seed)
    state = engine.initial_state()
    seen: list[str] = []
    for _ in range(limit):
        if engine.terminal(state) is not None:
            break
        seen.append(_phase(engine, state)[0])
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


@pytest.mark.parametrize("name", sorted(_RULES))
def test_random_walks_cross_the_phase_both_ways_as_the_oracle_does(name: str) -> None:
    order = {"alternating": 0, "reply": 1, "announced": 2, "blind": 3}
    engine = _engine(name)
    back = 0
    for seed in range(24):
        seen = _walk(engine, seed)
        back += sum(1 for a, b in zip(seen, seen[1:], strict=False) if order[b] < order[a])
    assert back > 0, "no walk went back a phase"


@pytest.mark.parametrize("name", sorted(_RULES))
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


def _step(engine: Engine, state: GameState, seat: int, *words: tuple[str, ...]) -> SpyStep:
    """One step from a hand-built position, as `spy_steps` would yield it: a turn's one move
    for the side to move, or a round's two, ``seat``'s first."""
    phase = engine.phase(state)
    if phase.is_round:
        moves = [0, 0]
        for player, word in zip((seat, 1 - seat), words, strict=True):
            moves[player] = _find(engine, engine.legal_moves_for(state, player), seat, *word)
        lists: tuple[list[int], ...] = (
            engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1),
        )
        after, _, events = engine.apply_round_with_phase(state, moves[0], moves[1])
        return SpyStep(0, state, phase, tuple(moves), lists, after, events)
    move = _find(engine, engine.legal_moves(state), seat, *words[0])
    return SpyStep(
        0, state, phase, (move,), (engine.legal_moves(state),), engine.apply(state, move), (),
    )


def _tally(engine: Engine, step: SpyStep) -> dict[str, Any]:
    spy = blank_spy_counts()
    count_spy_holders(engine, spy, step, 7)
    return {key: value for key, value in spy.items() if value not in (None, 0)}


@SEATS
def test_the_pass_counts_a_spy_won_back_and_how(seat: int) -> None:
    """`SPY_COUNTS`' gains beside its losses: the ply after the step and the route, for the
    player who had no spy before it and has one after (SR7h's promotion, SR7i's three)."""
    them = 1 - seat
    engine = _engine("chain")
    state = _position(engine, {"a1": "K", "c1": "R"}, {"a5": "K", "c4": "N"}, seat, side=0)
    step = _step(engine, state, seat, ("c1", "c4"), ("a5", "b5"))
    assert _tally(engine, step) == {f"gained_ply_{them}": 7, f"gained_how_{them}": "strike"}

    engine = _engine("switch")
    state = _position(engine, {"a1": "K", "c3": "N"}, {"a5": "K", "d3": "S"}, seat, side=1)
    step = _step(engine, state, seat, ("a1", "b1"), ("d3", "c3"))
    assert _tally(engine, step) == {
        f"gained_ply_{seat}": 7, f"gained_how_{seat}": "switch",
        f"lost_ply_{them}": 7, f"lost_how_{them}": "strike",
    }

    engine = _engine("link")
    state = _position(
        engine, {"a1": "K", "c1": "N"}, {"a5": "K", "d3": "P", "e5": "S"}, seat, side=0,
    )
    step = _step(engine, state, seat, ("c1", "d3"))
    assert _tally(engine, step) == {f"gained_ply_{seat}": 7, f"gained_how_{seat}": "link"}

    state = _position(engine, {"a1": "K", "d4": "P"}, {"a5": "K", "e3": "S"}, seat, side=0)
    step = _step(engine, state, seat, ("d4", "d5", "S"))
    assert _tally(engine, step) == {f"gained_ply_{seat}": 7, f"gained_how_{seat}": "promotion"}

    # A second spy beside a standing one changes no holder, so it is no gain here.
    engine = _engine("chain")
    state = _position(
        engine, {"a1": "K", "c1": "R", "e1": "S"}, {"a5": "K", "c4": "N", "e5": "S"}, seat, side=0,
    )
    assert _tally(engine, _step(engine, state, seat, ("c1", "c4"))) == {}


def test_the_gain_keys_and_readings() -> None:
    assert SPY_GAINS == ("promotion", "drop", "buy", "switch", "strike", "link", "other")
    assert [key for key in SPY_COUNTS if key.startswith("gained_")] == [
        "gained_ply_0", "gained_ply_1", "gained_how_0", "gained_how_1",
    ]
    assert all(blank_spy_counts()[key] is None for key in SPY_COUNTS if key.startswith("gained_"))
