"""The player-value teacher (Q665, plan-teacher §8.2 T2v): a label from the playing search.

What is held here, on gardner (a small board, not chess) and tiny budgets:

- ``Core.search_values`` is the minimax player's own search: its value is the Python
  oracle's (``agents/minimax.py``) to the bit, its move and depth are ``select_move``'s
  and ``select_move_depth``'s with the same seed, and a position with one legal move is
  searched too;
- its answer is a pure function of each position and its seed: the same at any thread
  count and in any input order;
- ``Procedure.label_search`` enters the digest only when it is not ``"probe"``, so the
  default (``db5ef1bc938e`` since Q671), Q637's (``1d160257b8e5``) and Q502f2's
  (``1c193cc56c76``) recipes keep their names;
- the harvest under ``"player"`` labels each sample with the squashed root value of
  ``minimax:nodes=<label_nodes>`` on that position, seeded by ``label_seed``, the same at
  1 and 3 worker processes; it records its teacher, and a label is never remade from
  another teacher's values.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from chess_variant_sandbox.agents import build_agent  # noqa: E402
from chess_variant_sandbox.agents.minimax import INFINITY, BudgetExhausted  # noqa: E402
from chess_variant_sandbox.agents.training.data import (  # noqa: E402
    Dataset,
    harvest,
    label_seed,
    labels,
    player_agent,
    self_play,
    squash,
)
from chess_variant_sandbox.agents.training.procedure import (  # noqa: E402
    DEFAULT,
    LABEL_SEARCHES,
    Q502F2,
    Q637,
    Procedure,
)
from chess_variant_sandbox.engine import Engine  # noqa: E402
from chess_variant_sandbox.engine.engine import _pack  # noqa: E402
from chess_variant_sandbox.spec import load_spec  # noqa: E402
from chess_variant_sandbox.tournament.play import eval_values, native_spec, seat_rng  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GARDNER = ROOT / "rulesets" / "gardner.json"

DEFAULT_DIGEST = "db5ef1bc938e"
Q637_DIGEST = "1d160257b8e5"
Q502F2_DIGEST = "1c193cc56c76"
# T2v's recipe: Q637's (the default then) with the player's value at 51,200 nodes
# (plan-teacher §8.2).
T2V = Q637.with_(label_search="player", label_nodes=51_200)
T2V_DIGEST = "bfca30cf2c4a"

TINY = Procedure(
    agents=("greedy",),
    games=4,
    skip_plies=2,
    result_weight=0.5,
    label_nodes=150,
    label_depth=3,
    label_search="player",
    td_lambda=0.0,
    accumulator=16,
    hidden1=8,
    hidden2=4,
    epochs=2,
    batch=32,
)


@pytest.fixture(scope="module")
def spec() -> dict:
    return load_spec(GARDNER)


@pytest.fixture(scope="module")
def engine(spec: dict) -> Engine:
    return Engine(spec)


def _positions(engine: Engine, count: int, seed: int = 5) -> list[Any]:
    """``count`` positions along one random line from the start, none of them over."""
    rng = random.Random(seed)
    state = engine.initial_state()
    out = []
    while len(out) < count:
        moves = engine.legal_moves(state)
        if not moves or engine.terminal(state, moves) is not None:
            state = engine.initial_state()
            continue
        out.append(state)
        state = engine.apply(state, rng.choice(moves))
    return out


def _values(engine: Engine, states: list[Any], agent: str, seeds: list[int], threads: int = 1):
    return engine.core.search_values(
        [_pack(state) for state in states],
        native_spec(agent, engine.spec.hash),
        eval_values(engine.spec),
        seeds,
        threads,
    )


# ------------------------------------------------------------------ the oracle


def _oracle_root(agent: Any, state: Any, moves: list[int], depth: int) -> tuple[int, float]:
    """``MinimaxAgent._search_root``, line for line, returning ``alpha`` beside the move."""
    alpha = -INFINITY
    best_move = moves[0]
    for move in moves:
        child = agent.engine.apply(state, move)
        if child.side_to_move != state.side_to_move:
            score = -agent._negamax(child, depth - 1, 1, -INFINITY, -alpha)
        else:
            score = agent._negamax(child, depth - 1, 1, alpha, INFINITY)
        if score > alpha:
            alpha, best_move = score, move
    return best_move, alpha


def _oracle_value(engine: Engine, agent_name: str, state: Any, seed: int):
    """``MinimaxAgent.select_move``'s deepening, line for line, without the one-move return."""
    agent = build_agent(agent_name, engine)
    moves = engine.legal_moves(state)
    rng = seat_rng(seed, 0)
    agent._nodes = 0
    ordered = list(moves)
    rng.shuffle(ordered)
    best_move, best_value, reached = ordered[0], -INFINITY, 0
    for depth in range(1, agent.max_depth + 1):
        try:
            candidate, value = _oracle_root(agent, state, ordered, depth)
        except BudgetExhausted:
            break
        best_move, best_value, reached = candidate, value, depth
        ordered.remove(best_move)
        ordered.insert(0, best_move)
        if agent._nodes >= agent.budget:
            break
    return (best_value, best_move, reached) if reached else None


@pytest.mark.parametrize("nodes", [60, 400])
def test_the_value_is_the_oracles_to_the_bit(engine: Engine, nodes: int) -> None:
    states = _positions(engine, 6)
    seeds = [label_seed(11, ply) for ply in range(len(states))]
    agent = f"minimax:nodes={nodes}"
    native = _values(engine, states, agent, seeds)
    oracle = [_oracle_value(engine, agent, s, seed) for s, seed in zip(states, seeds, strict=True)]
    assert native == oracle
    # 60 nodes runs out inside depth 1 on some of these (the oracle agrees: no value);
    # 400 finishes at least one iteration everywhere.
    if nodes >= 400:
        assert all(answer is not None and answer[2] >= 1 for answer in native)
    assert any(answer is not None for answer in native)


def test_the_move_and_depth_are_the_players(engine: Engine) -> None:
    """Where there is a choice, the value's move is the move the player plays."""
    agent = "minimax:nodes=300"
    native = native_spec(agent, engine.spec.hash)
    values = eval_values(engine.spec)
    states = _positions(engine, 8, seed=9)
    seeds = list(range(100, 108))
    answers = _values(engine, states, agent, seeds)
    for state, seed, (value, move, depth) in zip(states, seeds, answers, strict=True):
        assert math.isfinite(value)
        if len(engine.legal_moves(state)) > 1:
            assert engine.core.select_move(_pack(state), native, values, seed) == move
            assert engine.core.select_move_depth(_pack(state), native, values, seed) == (
                move,
                depth,
            )


def test_a_single_legal_move_is_searched_and_an_ended_game_is_not(engine: Engine) -> None:
    states = _positions(engine, 40, seed=3)
    forced = [s for s in states if len(engine.legal_moves(s)) == 1]
    for state in forced[:2]:
        (answer,) = _values(engine, [state], "minimax:nodes=200", [0])
        assert answer is not None and answer[1] == engine.legal_moves(state)[0]
    # A finished game: play random moves to the end.
    rng = random.Random(1)
    state = engine.initial_state()
    while (moves := engine.legal_moves(state)) and engine.terminal(state, moves) is None:
        state = engine.apply(state, rng.choice(moves))
    assert _values(engine, [state], "minimax:nodes=200", [0]) == [None]
    # A budget too small for depth 1 gives no value rather than a made-up one.
    wide = max(states, key=lambda s: len(engine.legal_moves(s)))
    assert _values(engine, [wide], "minimax:nodes=2", [0]) == [None]


def test_the_answer_does_not_depend_on_threads_or_order(engine: Engine) -> None:
    states = _positions(engine, 12, seed=4)
    seeds = [label_seed(7, ply) for ply in range(len(states))]
    agent = "minimax:nodes=250"
    one = _values(engine, states, agent, seeds, threads=1)
    three = _values(engine, states, agent, seeds, threads=3)
    assert one == three
    backwards = _values(engine, states[::-1], agent, seeds[::-1], threads=3)
    assert backwards == one[::-1]
    # The seed is the stream: another seed may reorder ties, never the call's shape.
    assert len(_values(engine, states, agent, [s + 1 for s in seeds])) == len(states)


def test_only_a_minimax_player_has_a_search_value(engine: Engine) -> None:
    state = engine.initial_state()
    for agent in ("greedy", "random"):
        with pytest.raises(ValueError, match="not one"):
            _values(engine, [state], agent, [0])
    with pytest.raises(ValueError, match="one seed per state"):
        _values(engine, [state, state], "minimax:nodes=100", [0])


# ------------------------------------------------------------------ the procedure


def test_label_search_enters_the_name_only_when_it_is_the_player() -> None:
    assert LABEL_SEARCHES == ("probe", "player", "parent")  # "parent": Q677, tests/test_generation.py
    assert DEFAULT.label_search == "probe" and "label_search" not in DEFAULT.as_dict()
    assert DEFAULT.digest == DEFAULT_DIGEST and Q502F2.digest == Q502F2_DIGEST
    assert Q637.digest == Q637_DIGEST
    assert DEFAULT.with_(label_search="probe").digest == DEFAULT_DIGEST
    assert Q637.with_(label_search="probe").digest == Q637_DIGEST
    assert T2V.as_dict()["label_search"] == "player"
    assert T2V.digest == T2V_DIGEST
    assert Procedure.from_dict(T2V.as_dict()) == T2V
    assert Procedure.from_dict(DEFAULT.as_dict()).label_search == "probe"
    with pytest.raises(ValueError, match="label_search"):
        DEFAULT.with_(label_search="oracle")
    assert player_agent(T2V) == "minimax:nodes=51200"


def test_the_label_seed_is_a_pure_function_under_two_to_the_62() -> None:
    assert label_seed(3, 4) == label_seed(3, 4)
    assert len({label_seed(seed, ply) for seed in range(20) for ply in range(20)}) == 400
    assert all(0 <= label_seed(seed, 7) < 2**62 for seed in range(50))


# ------------------------------------------------------------------ the harvest


def test_the_harvest_labels_each_sample_with_the_players_value(spec: dict, engine: Engine) -> None:
    records = self_play(spec, TINY, workers=1)
    data = harvest(spec, records, TINY)
    assert data.teacher == "player" and data.positions > 0
    w = TINY.result_weight
    at = 0
    for record in records:
        state = engine.initial_state()
        asked, seeds = [], []
        for ply, move in enumerate(record.moves):
            if ply >= TINY.skip_plies:
                asked.append(state)
                seeds.append(label_seed(record.seed, ply))
            state = engine.apply(state, move)
        for answer in _values(engine, asked, player_agent(TINY), seeds):
            assert answer is not None
            value = squash(answer[0], TINY.scale)
            assert data.values[at] == value
            assert data.targets[at] == np.float32(w * data.results[at] + (1 - w) * value)
            at += 1
    assert at == data.positions
    # Another teacher, another label.
    probe = harvest(spec, records, TINY.with_(label_search="probe"))
    assert probe.teacher == "probe" and probe.digest != data.digest


def test_the_player_harvest_does_not_depend_on_the_worker_count(spec: dict) -> None:
    procedure = TINY.with_(games=6)
    records = self_play(spec, procedure, workers=1)
    one = harvest(spec, records, procedure, workers=1)
    three = harvest(spec, records, procedure, workers=3)
    assert one.digest == three.digest
    for column in ("game", "ply", "results", "values"):
        assert np.array_equal(getattr(one, column), getattr(three, column), equal_nan=True)


def test_a_label_is_never_remade_from_another_teachers_values(spec: dict, tmp_path: Path) -> None:
    data = harvest(spec, self_play(spec, TINY, workers=1), TINY)
    again = Dataset.load(data.save(tmp_path / "player.npz"))
    assert again.teacher == "player" and again.digest == data.digest
    assert labels(again, TINY).tobytes() == data.targets.tobytes()
    with pytest.raises(ValueError, match="labelled by the player"):
        labels(again, TINY.with_(label_search="probe"))
    # Data saved before Q665 was the probe's: it trains under the probe, never the player.
    old = Dataset(
        data.n_inputs, data.offsets, data.inputs, data.sides, data.targets, data.games,
        data.game, data.ply, data.results, data.values,
    )
    assert old.teacher is None
    with pytest.raises(ValueError, match="labelled by the probe"):
        labels(old, TINY)
