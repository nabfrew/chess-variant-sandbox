"""T1d's second arm (`docs/plan-player-weaknesses.md` §T1d, Q1186b): the `afford` search bit.

At a quiescence node with depth left, in a game with `rules.income` and a moves channel
(`rules.purse.upkeep`), where the side to move's purse is below the dearest cost among its
pseudo-legal moves as the purse filter reads it, the node takes no stand pat and searches every
legal move. Everywhere else the bit is unread: every committed game without income plays the
same moves and nodes with it on.

Specs, not chess: `rulesets/hunger.json` (income 8 a turn against move costs up to 14) with its
opening purse lowered, so the side to move is poor from the first ply, and the committed rulesets
as they are for the inertness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chess_variant_sandbox import _core
from chess_variant_sandbox.agents import build_agent
from chess_variant_sandbox.agents.minimax import (
    AFFORD,
    DEFAULT_REPAIRS,
    REPAIR_OPTIONS,
    MinimaxAgent,
)
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.play import (
    choose_move,
    play_game,
    play_game_python,
    seat_rng,
)

ROOT = Path(__file__).resolve().parents[1]
HUNGER = ROOT / "rulesets" / "hunger.json"
CANDIDATE = "minimax:nodes=300,scarcity=1,afford=1"


def _poor_hunger(start: int = 0, plies: int = 24) -> Engine:
    """Hunger's pieces and purse rules from a constructed opening: the queens and a rook each
    already out (a queen's step to a far row costs 2 a row), only the king on the first row (so
    income pays 1 a turn), and the purse `start`."""
    spec = load_spec(HUNGER)
    spec["setup"] = {
        "mirror": True,
        "placements": [
            {"piece": "K", "squares": ["e1"]},
            {"piece": "Q", "squares": ["d3"]},
            {"piece": "R", "squares": ["h3"]},
            {"piece": "P", "squares": ["a2", "b2", "c2"]},
        ],
    }
    spec["rules"].pop("castling", None)
    spec["rules"]["purse"]["start"] = start
    spec["rules"]["max_plies"] = plies
    return Engine(spec)


def test_the_bit_is_its_own_and_spelled_afford() -> None:
    assert REPAIR_OPTIONS["afford"] == AFFORD == 131072
    assert _core.DEFAULT_REPAIRS == DEFAULT_REPAIRS


def test_the_opening_of_a_poor_hunger_is_short_of_its_dearest_move() -> None:
    """The constructed position with an empty purse: the side to move cannot pay for its dearest
    pseudo-legal move, so the guard reads; with a full purse it does not."""
    engine = _poor_hunger()
    state = engine.initial_state()
    agent = build_agent(CANDIDATE, engine)
    assert isinstance(agent, MinimaxAgent) and agent._afford
    assert agent._short_of_dearest(state)
    rich = _poor_hunger(start=99)
    assert not build_agent(CANDIDATE, rich)._short_of_dearest(rich.initial_state())


def test_core_and_oracle_choose_alike_from_the_poor_position() -> None:
    engine = _poor_hunger()
    state = engine.initial_state()
    moves = engine.legal_moves(state)
    for seed in range(3):
        oracle = build_agent(CANDIDATE, engine).select_move(state, list(moves), seat_rng(seed, 0))
        assert choose_move(engine, CANDIDATE, state, seed=seed) == oracle


@pytest.mark.parametrize("seed", [1, 2])
def test_oracle_and_core_agree_under_afford(seed: int) -> None:
    """Move for move and node for node, from the poor position, against the reserve player."""
    engine = _poor_hunger()
    agents = (CANDIDATE, "minimax:nodes=300,reserve=1")
    native = play_game(engine, agents, seed=seed)
    in_python = play_game_python(engine, agents, seed=seed)
    assert native.moves == in_python.moves
    assert (native.winner, native.reason) == (in_python.winner, in_python.reason)
    assert native.nodes == in_python.nodes
    assert native.depth_sum == in_python.depth_sum


def test_the_core_reads_the_bit_on_a_poor_hunger() -> None:
    engine = _poor_hunger()
    played = [
        play_game(engine, (f"minimax:nodes=300,scarcity=1,afford={bit}", CANDIDATE), seed=4)
        for bit in (1, 0)
    ]
    assert (played[0].moves, played[0].nodes) != (played[1].moves, played[1].nodes)


def _without_income() -> list[Path]:
    paths = sorted((ROOT / "rulesets").glob("*.json"))
    return [path for path in paths if not load_spec(path).get("rules", {}).get("income")]


def test_every_game_without_income_plays_as_before() -> None:
    """The bit is unread without income: in every committed game without it, `afford=1` and
    `afford=0` play the same game in the core, move for move and node for node."""
    paths = _without_income()
    assert len(paths) > 30
    played = 0
    for path in paths:
        spec = load_spec(path)
        spec.setdefault("rules", {})["max_plies"] = 8
        engine = Engine(spec)
        if engine.simultaneous or engine.spec.spy is not None:
            continue  # minimax plays no round; neither pays income
        records = [
            play_game(engine, (f"minimax:nodes=60,afford={bit}", "minimax:nodes=60"), seed=3)
            for bit in (1, 0)
        ]
        assert (records[0].moves, records[0].nodes, records[0].depth_sum) == (
            records[1].moves, records[1].nodes, records[1].depth_sum
        ), path.name
        played += 1
    assert played > 30
