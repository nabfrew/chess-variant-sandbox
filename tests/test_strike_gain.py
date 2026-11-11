"""T6 (`docs/plan-player-weaknesses.md` §T6, Q1174): the `strike_gain` search bit.

In a game with `rules.strike` or `rules.contacts`, move ordering reads a meeting's gain off the
compiled strike tables (`agents.rs::strike_gain`, oracle `MinimaxAgent._strike_gain`): what the
meeting leaves on the board, and whose it is, in place of the victim's whole value. Everywhere
else the bit is unread.

Specs, not chess: every archived strike and contacts game (demotion and promotion chains, `stay`,
`swap`, `choose`, `home`, a side switch, own links), and the committed rulesets for the inertness.
The gain's own oracle is the board: for a plain capture, the material balance the move leaves.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from chess_variant_sandbox import _core
from chess_variant_sandbox.agents import build_agent
from chess_variant_sandbox.agents.minimax import (
    DEFAULT_REPAIRS,
    REPAIR_OPTIONS,
    STRIKE_GAIN,
    MinimaxAgent,
)
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import WILD
from chess_variant_sandbox.engine.moves import MOVE_NORMAL, decode_move
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.play import play_game, play_game_python

ROOT = Path(__file__).resolve().parents[1]
STRIKE_GAMES = sorted(
    path for path in (ROOT / "archive").glob("*.json")
    if {"strike", "contacts"} & set(load_spec(path).get("rules", {}))
)
GIFT = ROOT / "archive" / "hierarchy-rise-stay.json"


def _engine(path: Path, plies: int) -> Engine:
    spec = load_spec(path)
    spec["rules"]["max_plies"] = plies
    return Engine(spec)


def test_the_bit_is_its_own_and_spelled_strike_gain() -> None:
    assert REPAIR_OPTIONS["strike_gain"] == STRIKE_GAIN == 262144
    assert _core.DEFAULT_REPAIRS == DEFAULT_REPAIRS


def test_every_strike_rule_is_reached() -> None:
    positions = {Engine(load_spec(path)).spec.strike.position for path in STRIKE_GAMES}
    assert len(STRIKE_GAMES) >= 12
    assert {"stay", "swap", "choose", "home", "advance"} <= positions


def _balance(engine: Engine, board: bytes, side: int, values: tuple[float, ...]) -> float:
    spec = engine.spec
    total = 0.0
    for byte in board:
        if byte and spec.owner_of(byte) != WILD:
            sign = 1.0 if spec.owner_of(byte) == side else -1.0
            total += sign * values[spec.type_of(byte)]
    return total


@pytest.mark.parametrize("path", STRIKE_GAMES, ids=lambda path: path.stem)
def test_the_gain_is_the_balance_the_meeting_leaves(path: Path) -> None:
    """Along a seeded walk that prefers captures, every plain capture's strike gain is the change
    in the mover's material balance once the move is played: the board is the oracle."""
    engine = _engine(path, 60)
    agent = build_agent("minimax:nodes=50,strike_gain=1", engine)
    assert isinstance(agent, MinimaxAgent)
    values = tuple(agent.evaluator.values)
    rng = random.Random(path.stem)
    state = engine.initial_state()
    checked = 0
    for _ in range(60):
        moves = engine.legal_moves(state)
        if engine.terminal(state, moves) is not None:
            break
        side = state.side_to_move
        before = _balance(engine, state.board, side, values)
        captures = []
        for move in moves:
            gain = agent._strike_gain(state.board, move)
            if gain is None or decode_move(move)[2] != MOVE_NORMAL:
                continue
            captures.append(move)
            after = _balance(engine, engine.apply(state, move).board, side, values)
            assert gain == pytest.approx(after - before, abs=1e-9), engine.describe_move(state, move)
            checked += 1
        state = engine.apply(state, rng.choice(captures or moves))
    assert checked > 0


@pytest.mark.parametrize("path", STRIKE_GAMES, ids=lambda path: path.stem)
def test_oracle_and_core_agree_under_strike_gain(path: Path) -> None:
    """Move for move and node for node, the candidate against the player without the bit."""
    engine = _engine(path, 24)
    agents = ("minimax:nodes=120,strike_gain=1", "minimax:nodes=120,strike_gain=0")
    native = play_game(engine, agents, seed=1)
    in_python = play_game_python(engine, agents, seed=1)
    assert native.moves == in_python.moves
    assert (native.winner, native.reason) == (in_python.winner, in_python.reason)
    assert native.nodes == in_python.nodes
    assert native.depth_sum == in_python.depth_sum


def test_the_core_reads_the_bit_on_gift_from_afar() -> None:
    engine = _engine(GIFT, 40)
    played = [
        play_game(engine, (f"minimax:nodes=400,strike_gain={bit}", "minimax:nodes=400"), seed=2)
        for bit in (1, 0)
    ]
    assert (played[0].moves, played[0].nodes) != (played[1].moves, played[1].nodes)


def test_every_game_without_a_strike_plays_as_before() -> None:
    """The bit is unread without a strike table: in every committed ruleset, `strike_gain=1`
    and `strike_gain=0` play the same game in the core, move for move and node for node."""
    played = 0
    for path in sorted((ROOT / "rulesets").glob("*.json")):
        spec = load_spec(path)
        spec.setdefault("rules", {})["max_plies"] = 6
        engine = Engine(spec)
        if engine.simultaneous or engine.spec.spy is not None:
            continue  # minimax plays no round
        assert engine.spec.strike is None, path.name
        records = [
            play_game(engine, (f"minimax:nodes=40,strike_gain={bit}", "minimax:nodes=40"), seed=3)
            for bit in (1, 0)
        ]
        assert (records[0].moves, records[0].nodes, records[0].depth_sum) == (
            records[1].moves, records[1].nodes, records[1].depth_sum
        ), path.name
        played += 1
    assert played > 30
