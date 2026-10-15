"""Schema 26 (Q918b): the computer players for rounds (`docs/plan-simultaneous.md` §6.2, §7).

The solver's iteration count against the pre-registered matrices (*Q918b, pre-registered*);
the solver, the strategies and the node counts held to the declared oracle
(`agents/matrix.py`, `agents/simple.py`'s round arms) to the bit; a matching-pennies position;
a mutual capture-only standoff played out under the void bar; the refusals. Whole games
through both loops are `tests/test_backends.py`'s round arms.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
from matrix_family import CANDIDATES, exact_cases, failures, random_family, stake_family
from test_schema_twenty_six import VOID, place

import chess_variant_sandbox._core as _core
from chess_variant_sandbox.agents import build_agent, native_spec
from chess_variant_sandbox.agents.features import VOCABULARY
from chess_variant_sandbox.agents.matrix import REGRET_ITERATIONS, solve
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament import TournamentConfig, run_tournament
from chess_variant_sandbox.tournament.play import eval_values, play_game, play_game_python, seat_rng

HERE = Path(__file__).parent
RULESETS = HERE.parent / "rulesets"

# The test specs Q918a's box names: a mirrored Gardner-sized game, drops and promotion, an
# explosion, an own-merge contact (two files: the validator refuses the two together), a
# static portal pair beside a fold.
TEST_SPECS = ("gardner", "shogi", "atomic", "merge", "portals")


def load(name: str, max_plies: int | None = None) -> dict[str, Any]:
    spec = json.loads((HERE / f"_simultaneous_{name}.json").read_text(encoding="utf-8"))
    if max_plies is not None:
        spec["rules"]["max_plies"] = max_plies
    return spec


# ------------------------------------------------------------------- the solver


def test_the_iteration_count_is_the_cores() -> None:
    assert _core.REGRET_ITERATIONS == REGRET_ITERATIONS


def test_the_count_is_the_smallest_candidate_meeting_the_registration() -> None:
    """(a) to (d) at the count, and every smaller candidate misses one; (e) is the
    recorded weakness (the plan, *Q918b, as built*), so it is not asserted here."""
    missed, _ = failures(REGRET_ITERATIONS)
    assert missed == []
    assert REGRET_ITERATIONS in CANDIDATES
    for smaller in (count for count in CANDIDATES if count < REGRET_ITERATIONS):
        assert failures(smaller)[0], smaller


def test_the_oracles_solver_is_the_cores_to_the_bit() -> None:
    matrices = [matrix for _, matrix, _, _ in exact_cases()]
    matrices += random_family() + stake_family()
    for values, rows, cols in matrices:
        oracle = solve(values, rows, cols)
        core = _core.regret_solve(values, rows, cols)
        assert oracle == (list(core[0]), list(core[1])), (rows, cols)


# ------------------------------------------------- strategies, held to the oracle


def oracle_round(
    engine: Engine, agent: str, state: GameState, seat: int, seed: int
) -> tuple[int, list[float], int, int]:
    """What the declared oracle plays for ``seat`` here, as `Core.select_round` reports it."""
    player: Any = build_agent(agent, engine)
    lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
    move = player.select_round_move(state, seat, lists[seat], lists[1 - seat], seat_rng(seed, seat))
    strategy = list(getattr(player, "strategy", []))
    return move, strategy, player.stats.nodes, getattr(player, "depth", 0)


def core_round(
    engine: Engine, agent: str, state: GameState, seat: int, seed: int
) -> tuple[int | None, list[float], int, int]:
    native = native_spec(agent, engine.spec.hash)
    assert native is not None
    move, strategy, nodes, depth = engine.select_round(
        state, seat, native, eval_values(engine.spec), seed
    )
    return move, list(strategy), nodes, depth


def walk(engine: Engine, rounds: int, seed: int) -> list[GameState]:
    """Positions a random game passes through, the opening first."""
    rng = random.Random(seed)
    state = engine.initial_state()
    states = [state]
    for _ in range(rounds):
        if engine.terminal(state) is not None:
            break
        first = rng.choice(engine.legal_moves_for(state, 0))
        second = rng.choice(engine.legal_moves_for(state, 1))
        state = engine.apply_round(state, first, second)
        states.append(state)
    return [state for state in states if engine.terminal(state) is None]


def _assert_rounds_agree(name: str, agents: tuple[str, ...], rounds: int) -> None:
    engine = Engine(load(name))
    for index, state in enumerate(walk(engine, rounds, seed=11)):
        for agent in agents:
            for seat in (0, 1):
                core = core_round(engine, agent, state, seat, seed=index)
                assert core == oracle_round(engine, agent, state, seat, seed=index), (
                    name, agent, index, seat,
                )


def test_the_strategies_are_the_oracles_on_the_mirrored_spec() -> None:
    _assert_rounds_agree("gardner", ("greedy", "matrix:nodes=250"), rounds=2)


@pytest.mark.slow
@pytest.mark.parametrize("name", TEST_SPECS)
def test_the_strategies_are_the_oracles_on_every_test_spec(name: str) -> None:
    _assert_rounds_agree(name, ("greedy", "matrix:nodes=600"), rounds=8)


# ----------------------------------------------------------- matching pennies

# Player 1's royal K stands on b2 and must step to a1 or c1; player 0's A may step to
# either, or onto b2. Meeting K is a collision both pieces lose, which loses player 1 its
# royal: player 0 wins by guessing where K goes, player 1 by guessing wrong for it. The
# board is symmetric, so the two guesses are worth exactly the same: matching pennies.
PENNIES = {
    "schema_version": 26,
    "name": "matching pennies",
    "board": {"width": 3, "height": 2},
    "pieces": [
        {"id": "A", "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
        {"id": "K", "royal": True, "moves": [
            {"kind": "leap", "vectors": [[1, 1]], "symmetry": "rotate4"},
        ]},
    ],
    "setup": {"mirror": False, "placements": [
        {"piece": "A", "squares": ["b1"], "player": 0},
        {"piece": "K", "squares": ["b2"], "player": 1},
    ]},
    "rules": {"win": "royal_capture", "simultaneous": {}},
}


@pytest.mark.parametrize("agent", ["greedy", "matrix:nodes=1600"])
@pytest.mark.parametrize("seat", [0, 1])
def test_matching_pennies_is_played_half_and_half(agent: str, seat: int) -> None:
    """§7's box: within the stated tolerance, 0.01, of half and half."""
    engine = Engine(PENNIES)
    state = engine.initial_state()
    moves = engine.legal_moves_for(state, seat)
    _, strategy, _, depth = core = core_round(engine, agent, state, seat, seed=0)
    assert core == oracle_round(engine, agent, state, seat, seed=0)
    # On a board this small the budget finishes several passes: the deepening is held to
    # the oracle here, not only its first pass.
    assert depth >= (3 if agent.startswith("matrix") else 0)
    guesses = {
        engine.describe_move(state, move): weight for move, weight in zip(moves, strategy, strict=True)
    }
    side = "A" if seat == 0 else "K"
    for target in ("a1", "c1"):
        assert abs(guesses[f"{side}b2-{target}" if seat else f"{side}b1-{target}"] - 0.5) <= 0.01
    assert sum(guesses.values()) == pytest.approx(1.0)


# --------------------------------------------------------------- the void bar


def _void_pair(first: int, second: int) -> tuple[int, int] | None:
    """The pair a round would be void on: two capture-only moves at each other's square."""
    o0, t0, _, _ = decode_move(first)
    o1, t1, _, _ = decode_move(second)
    return (o0, o1) if (t0, t1) == (o1, o0) else None


@pytest.mark.parametrize(
    "agents", [("greedy", "matrix:nodes=400"), ("matrix:nodes=400", "greedy")],
    ids=lambda pair: f"{pair[0]}-vs-{pair[1]}",
)
def test_a_standoff_is_played_out_under_the_void_bar(agents: tuple[str, str]) -> None:
    """§7's box: from a mutual capture-only standoff the players never play a barred
    move, never three void rounds of one pair in a row, none of a barred pair again while
    it is barred, and the game ends within `max_plies`."""
    engine = Engine(VOID)
    start = place(engine, {"b2": ("P", 0), "c3": ("P", 1), "e1": ("W", 0), "a5": ("W", 1)})
    natives = tuple(native_spec(agent) for agent in agents)
    bars = 0
    for seed in range(6):
        record = engine.play_from(start, natives, eval_values(engine.spec), seed)
        moves, reason = record[0], record[2]
        assert reason and len(moves) + start.ply <= engine.spec.max_plies
        state, run = start, (None, 0)
        for index in range(0, len(moves), 2):
            first, second = moves[index], moves[index + 1]
            assert first in engine.legal_moves_for(state, 0)
            assert second in engine.legal_moves_for(state, 1)
            barred = set(state.bars)
            state, _, events = engine.apply_round_with_phase(state, first, second)
            voids = sum(1 for event in events if event[0] == 1)
            pair = _void_pair(first, second) if voids == 2 else None
            assert pair not in barred
            run = (pair, run[1] + 1 if pair is not None and pair == run[0] else int(pair is not None))
            assert run[1] <= 2
            bars += sum(1 for event in events if event[0] == 3)
    assert bars > 0, "no game reached the bar, so nothing was tested"


# ----------------------------------------------------------- what plays rounds


def test_matrix_at_one_pass_plays_greedys_game() -> None:
    """Pass 1 is greedy's matrix (§6.2), and the draw is the same draw."""
    engine = Engine(load("gardner", max_plies=40))
    one = play_game(engine, ("matrix:nodes=1", "random"), seed=3)
    other = play_game(engine, ("greedy", "random"), seed=3)
    assert one.moves == other.moves and one.nodes == other.nodes
    assert one.searches[0] > 0 and one.depth_sum[0] == one.searches[0]


def test_a_round_game_is_the_same_at_any_worker_count() -> None:
    spec = load("gardner", max_plies=30)
    config = TournamentConfig(
        agents=("random", "greedy", "matrix:nodes=200"), games_per_pairing=1, seed=5
    )
    serial = run_tournament(spec, TournamentConfig(**{**config.__dict__, "workers": 1}))
    parallel = run_tournament(spec, TournamentConfig(**{**config.__dict__, "workers": 2}))
    assert [(r.moves, r.nodes) for r in serial.records] == [
        (r.moves, r.nodes) for r in parallel.records
    ]


def test_matrix_is_refused_on_a_game_that_alternates() -> None:
    engine = Engine(load_spec(RULESETS / "gardner.json"))
    with pytest.raises(ValueError, match="matrix plays a simultaneous game"):
        play_game(engine, ("matrix:nodes=100", "random"), 0)
    with pytest.raises(ValueError, match="matrix plays a simultaneous game"):
        build_agent("matrix:nodes=100", engine)


def test_a_round_player_plays_the_built_in_objective_only() -> None:
    engine = Engine(load("gardner"))
    with pytest.raises(ValueError, match="built-in objective only"):
        native_spec("matrix:nodes=100,h=0123456789ab")
    weights = tuple([0.1] * len(VOCABULARY))
    greedy = ("greedy", 0, 0.05, 0, 0, 0.0, 16.0, weights, (), 0.0, b"", 0)
    random_seat = native_spec("random")
    with pytest.raises(ValueError, match="built-in objective only"):
        engine.core.play_game((greedy, random_seat), eval_values(engine.spec), 0)


def test_a_python_player_without_a_round_arm_is_named() -> None:
    engine = Engine(load("gardner"))
    with pytest.raises(ValueError, match="does not play a simultaneous game"):
        play_game_python(engine, ("random", "minimax:nodes=10"), 0)


def test_the_core_lists_matrix_among_its_players() -> None:
    assert "matrix" in _core.native_agent_kinds()
