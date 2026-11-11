"""T5 (`docs/plan-player-weaknesses.md`, Q1173): the race over every racer a counted goal needs.

Under the `racers` search bit, where `rules.reach.count` is above 1, the race term sums each
side's `k = count - tally` nearest eligible racers' goal distances (a racer it no longer has
counts the goal diameter) over its own `k` diameters (Q1173b; `racers_fixed` keeps Q1173's
`count` diameters for both), in place of the nearest racer's over one. Everywhere
else the bit is unread: every committed game whose goal counts one arrival, or that has none,
evaluates to the same float and plays the same moves and nodes.

Specs, not chess: the committed rulesets as they are, and `rulesets/tessabi.json` (a goal
counted to four, banked) for the term itself; the oracle for the distances is the compiled
goal table read square by square in this file.
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import pytest

from chess_variant_sandbox import _core
from chess_variant_sandbox.agents.evaluation import Evaluator, reach_tally
from chess_variant_sandbox.agents.minimax import (
    DEFAULT_REPAIRS,
    RACERS,
    RACERS_FIXED,
    REPAIR_OPTIONS,
)
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.play import play_game, play_game_python

ROOT = Path(__file__).resolve().parents[1]
TESSABI = ROOT / "rulesets" / "tessabi.json"


def _counted(path: Path) -> bool:
    reach = load_spec(path).get("rules", {}).get("reach") or {}
    return int(reach.get("count", 1)) > 1


def test_the_bit_is_its_own_and_spelled_racers() -> None:
    assert REPAIR_OPTIONS["racers"] == RACERS == 32768
    assert REPAIR_OPTIONS["racers_fixed"] == RACERS_FIXED == 65536
    # Q1173b: on by default since T5b's duel (no cell lost); the fixed scale stays off.
    assert DEFAULT_REPAIRS & RACERS and _core.DEFAULT_REPAIRS & RACERS
    assert not DEFAULT_REPAIRS & RACERS_FIXED
    assert _core.DEFAULT_REPAIRS == DEFAULT_REPAIRS


def test_every_game_without_a_counted_goal_evaluates_bit_for_bit_as_before() -> None:
    """The oracle's half: in every committed game whose goal counts one arrival or that has
    none, the evaluator under `racers` reads each of a short random walk's positions to the same
    float as without it."""
    paths = [path for path in sorted((ROOT / "rulesets").glob("*.json")) if not _counted(path)]
    assert len(paths) > 30
    for path in paths:
        engine = Engine(load_spec(path))
        if engine.simultaneous or engine.spec.spy is not None:
            continue  # rounds and spy phases have no alternating walk; neither has a count
        off = Evaluator.for_spec(engine.spec)
        on = dataclasses.replace(off, racers=True)
        state = engine.initial_state()
        rng = random.Random(path.name)
        for _ply in range(4):
            moves = engine.legal_moves(state)
            assert on.evaluate(engine, state, 0, len(moves)) == off.evaluate(
                engine, state, 0, len(moves)
            ), path.name
            if not moves or engine.terminal(state, moves) is not None:
                break
            state = engine.apply(state, rng.choice(list(moves)))


@pytest.mark.parametrize("name", ["chess", "intransitive", "kingofthehill", "boulder"])
def test_the_core_plays_a_game_without_a_counted_goal_as_before(name: str) -> None:
    """The core's half: `racers=1` and `racers=0` play the same game, move for move and node for
    node, where the goal counts one arrival (a race among them) or there is none."""
    spec = load_spec(ROOT / "rulesets" / f"{name}.json")
    spec.setdefault("rules", {})["max_plies"] = 40
    engine = Engine(spec)
    records = [
        play_game(engine, (f"minimax:nodes=80,racers={bit}", "minimax:nodes=80"), seed=3)
        for bit in (1, 0)
    ]
    assert [(r.moves, r.nodes, r.depth_sum) for r in records][0] == (
        records[1].moves, records[1].nodes, records[1].depth_sum
    )


def _left(engine: Engine, state, player: int) -> float:  # type: ignore[no-untyped-def]
    """This file's oracle: the `count - tally` smallest capped goal distances, read off the
    compiled union table square by square, a missing racer at the diameter."""
    spec = engine.spec
    reach = spec.reach
    assert reach is not None
    diameter = int(spec.goal_diameter)
    needed = reach.count - int(reach_tally(spec, state, player))
    found = sorted(
        min(spec.goal_distance[player][spec.type_of(value)][square], diameter)
        for square, value in enumerate(state.board[: spec.n_squares])
        if value and spec.owner_of(value) == player and reach.admits(spec.type_of(value))
        and square not in reach.squares[player]
    )[:needed]
    return float(sum(found) + (needed - len(found)) * diameter)


def test_tessabi_races_every_runner_it_still_needs() -> None:
    engine = Engine(load_spec(TESSABI))
    off = Evaluator.for_spec(engine.spec)
    on = dataclasses.replace(off, racers=True)
    rng = random.Random(11)
    state = engine.initial_state()
    differ = False
    for _ply in range(30):
        moves = engine.legal_moves(state)
        if not moves or engine.terminal(state, moves) is not None:
            break
        for player in (0, 1):
            needed = engine.spec.reach.count - int(reach_tally(engine.spec, state, player))
            assert on._racers_to_goal(state, player, 0) == (_left(engine, state, player), needed)
        differ |= on.race_progress(state, 0) != off.race_progress(state, 0)
        state = engine.apply(state, rng.choice(list(moves)))
    assert differ


@pytest.mark.parametrize(("name", "seed"), [("tessabi", 1), ("tessabi", 2), ("crossings", 1)])
def test_oracle_and_core_agree_under_racers(name: str, seed: int) -> None:
    spec = load_spec(ROOT / "rulesets" / f"{name}.json")
    spec.setdefault("rules", {})["max_plies"] = 30
    engine = Engine(spec)
    agents = ("minimax:nodes=120,racers=1", "minimax:nodes=120,racers=1")
    native = play_game(engine, agents, seed=seed)
    in_python = play_game_python(engine, agents, seed=seed)
    assert native.moves == in_python.moves
    assert (native.winner, native.reason) == (in_python.winner, in_python.reason)
    assert native.nodes == in_python.nodes
    assert native.depth_sum == in_python.depth_sum


def test_the_core_reads_the_bit_on_tessabi() -> None:
    spec = load_spec(TESSABI)
    spec.setdefault("rules", {})["max_plies"] = 40
    engine = Engine(spec)
    played = [
        play_game(engine, (f"minimax:nodes=200,racers={bit}", "minimax:nodes=200,racers=0"), seed=4)
        for bit in (1, 0)
    ]
    assert (played[0].moves, played[0].nodes) != (played[1].moves, played[1].nodes)


def _race(engine: Engine, state, evaluator: Evaluator) -> float:  # type: ignore[no-untyped-def]
    """This file's oracle for the race's scale: per racer still needed (Q1173b), or over `count`
    diameters for both sides under `racers_fixed` (Q1173). The sums are the evaluator's own, read
    with the seal (`_left` holds them where nothing is sealed)."""
    spec = engine.spec
    reach = spec.reach
    assert reach is not None
    needed = [reach.count - int(reach_tally(spec, state, p)) for p in (0, 1)]
    sealed = evaluator.sealed_goals(state)
    left = [evaluator._racers_to_goal(state, p, sealed[p])[0] for p in (0, 1)]
    if evaluator.racers_fixed:
        progress = (left[1] - left[0]) / (reach.count * spec.goal_diameter)
    else:
        mean = [left[p] / needed[p] if needed[p] else 0.0 for p in (0, 1)]
        progress = (mean[1] - mean[0]) / spec.goal_diameter
    return max(-1.0, min(1.0, progress))


@pytest.mark.parametrize("fixed", [False, True])
def test_each_scale_prices_the_race_as_stated(fixed: bool) -> None:
    """Q1173b: the rescaled race is each side's mean racer distance over the diameter, so with
    one racer left on each side it is today's nearest-racer term; `racers_fixed` is Q1173's."""
    engine = Engine(load_spec(TESSABI))
    off = Evaluator.for_spec(engine.spec)
    on = dataclasses.replace(off, racers=True, racers_fixed=fixed)
    rng = random.Random(5)
    state = engine.initial_state()
    for _ply in range(40):
        moves = engine.legal_moves(state)
        if not moves or engine.terminal(state, moves) is not None:
            break
        assert on.race_progress(state, 0) == _race(engine, state, on)
        state = engine.apply(state, rng.choice(list(moves)))


def test_the_two_scales_are_two_players_on_tessabi() -> None:
    spec = load_spec(TESSABI)
    spec.setdefault("rules", {})["max_plies"] = 40
    engine = Engine(spec)
    played = [
        play_game(
            engine,
            (f"minimax:nodes=200,racers=1,racers_fixed={bit}", "minimax:nodes=200,racers=0"),
            seed=4,
        )
        for bit in (1, 0)
    ]
    assert (played[0].moves, played[0].nodes) != (played[1].moves, played[1].nodes)


@pytest.mark.parametrize("name", ["tessabi", "crossings"])
def test_oracle_and_core_agree_under_the_fixed_scale(name: str) -> None:
    spec = load_spec(ROOT / "rulesets" / f"{name}.json")
    spec.setdefault("rules", {})["max_plies"] = 30
    engine = Engine(spec)
    agents = ("minimax:nodes=120,racers=1,racers_fixed=1", "minimax:nodes=120,racers=1")
    native = play_game(engine, agents, seed=1)
    in_python = play_game_python(engine, agents, seed=1)
    assert native.moves == in_python.moves
    assert native.nodes == in_python.nodes
