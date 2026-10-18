"""Q918g: learned nets for round games (`docs/plan-simultaneous.md` §6.4, *Q918g, scoped*).

What is held here, on tiny budgets:

- the round segments (the initiative, each player's rested square, the void run, the bars)
  are on only where their rule is, so **every alternating layout is unchanged** (the reason
  `INPUTS_VERSION` did not move) and every stored net still reads as the net it was;
- along a round game's line the core's inputs and integer outputs are the oracle's
  (`agents/nets.py`) to the bit, every segment reached, a position a round;
- a net plays rounds: `matrix:net=` and `greedy:net=` choose the oracle's strategies and
  node counts, a whole game is the Python loop's, and the net is what played; a spy game
  still refuses one;
- the round teacher (`Core.round_values`) is the oracle's value to the bit, at any thread
  count, and greedy's is `matrix:nodes=1`'s;
- the harvest of a round game is one sample a round, all seat 0's, labelled by the round
  player's value, the same at any worker count; a round game's random opening is whole
  rounds from the seed's streams;
- the round recipe (`ROUND_DEFAULT`) and its control are pinned, and the combinations that
  cannot work are refused in words; the acceptance duel seats `matrix:` on a round game.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from chess_variant_sandbox.agents import build_agent, native_spec, nets  # noqa: E402
from chess_variant_sandbox.agents.evaluation import GOAL_PIECES, IMMUNE_PIECES  # noqa: E402
from chess_variant_sandbox.agents.matrix import (  # noqa: E402
    RoundSearch,
    round_evaluator,
    round_value_for_first,
)
from chess_variant_sandbox.agents.nets import (  # noqa: E402
    ACCEPTED,
    TURN_CAP,
    Layout,
    NetEvaluator,
    NetRecord,
    random_net,
    register_net,
)
from chess_variant_sandbox.agents.training.data import (  # noqa: E402
    _round_opening,
    diverse_game,
    harvest,
    opening_plies_for,
    self_play,
)
from chess_variant_sandbox.agents.training.procedure import (  # noqa: E402
    DEFAULT,
    ROUND_ACCEPT_NODES,
    ROUND_CONTROL,
    ROUND_DEFAULT,
    Procedure,
    builtin_player_arm,
    control,
)
from chess_variant_sandbox.agents.training.store import accept  # noqa: E402
from chess_variant_sandbox.engine import Engine, GameState  # noqa: E402
from chess_variant_sandbox.engine.engine import _pack  # noqa: E402
from chess_variant_sandbox.spec import load_spec  # noqa: E402
from chess_variant_sandbox.tournament.play import (  # noqa: E402
    eval_values,
    play_game,
    play_game_python,
    seat_rng,
)

ROOT = Path(__file__).resolve().parent.parent
PROCEDURE = "0e7000000918"

# The two library round games (the watering hole: `tie: impossible`; the chase: `rest`),
# and an archived one under the initiative, so every round segment has a game.
WATERING = "archive/watering-hole.json"
CHASE = "archive/gardner-in-rounds-against-the-chase.json"
INITIATIVE = "archive/gardner-in-rounds-with-the-initiative.json"
ROUND_GAMES = (WATERING, CHASE, INITIATIVE)

# The round recipe and its control, by name (`ROUND_DEFAULT`, `ROUND_CONTROL`): a change to
# either, the layout's version or the trainer must be seen.
ROUND_DEFAULT_DIGEST = "0f87b7f93850"
ROUND_CONTROL_DIGEST = "919d42469fbe"

TINY = ROUND_DEFAULT.with_(
    games=2, label_nodes=150, accumulator=16, hidden1=8, hidden2=4, epochs=1, batch=32
)


def _engine(path: str) -> Engine:
    return Engine(load_spec(ROOT / path))


@pytest.fixture
def registered() -> Iterator[None]:
    yield
    for key in list(nets._NETS):
        if key[0] == PROCEDURE:
            nets._NETS.pop(key)


def _register(engine: Engine, seed: int) -> bytes:
    blob = random_net(Layout.of(engine.spec).n_inputs, seed=seed, accumulator=16, hidden1=8)
    register_net(NetRecord(PROCEDURE, engine.spec.hash, ACCEPTED, blob))
    return blob


def _round_line(engine: Engine, seed: int) -> tuple[list[int], list[GameState]]:
    """A random round game's moves (each round's two in seat order) and every board."""
    record = play_game(engine, ("random", "random"), seed=seed, collect_series=False)
    state = engine.initial_state()
    states = [state]
    moves = list(record.moves)
    for index in range(0, len(moves), 2):
        state = engine.apply_round(state, moves[index], moves[index + 1])
        states.append(state)
    return moves, states


def _probes(engine: Engine) -> list[GameState]:
    """States no short random line is sure to reach: a void run, two bars sharing a
    square, a rested square for each player, either holder of the initiative."""
    start = engine.initial_state()
    a, b, c = 0, 1, engine.spec.n_squares - 1
    return [
        start.with_(void_run=(a, c)),
        start.with_(bars=((a, c), (b, c))),
        start.with_(rest=(b, c), initiative=0),
        start.with_(rest=(-1, a), initiative=1),
    ]


# ------------------------------------------------------------------ the layout


# The games a stored net is for (`nets/`), beside the kinds of state `test_nets.STATES`
# covers: their layouts are what every stored net was trained on.
STORED = {
    "975dde173f01": "rulesets/gardner.json",
    "29d5213b02b2": "rulesets/intransitive.json",
    "71912ea8ba7f": "rulesets/kingofthehill.json",
    "66b001f1b790": "rulesets/racingkings.json",
    "82a4788c02f0": "rulesets/rps-v2.json",
}
ALTERNATING = (*STORED.values(), "rulesets/chess.json", "rulesets/crazyhouse.json",
               "rulesets/threecheck.json", "archive/wolves-bounty.json")


def _alternating_layout_is_unchanged(path: Path) -> None:
    engine = Engine(load_spec(path))
    layout = Layout.of(engine.spec)
    assert not layout.rounds, path.name
    assert layout.n_inputs == layout.turn + TURN_CAP, path.name
    assert engine.core.net_inputs() == layout.n_inputs, path.name


@pytest.mark.parametrize("path", ALTERNATING)
def test_an_alternating_layout_is_unchanged(path: str) -> None:
    """No round segment on a game that alternates, so its layout, its stored nets and
    every procedure digest are what they were; the core says the same."""
    _alternating_layout_is_unchanged(ROOT / path)
    assert DEFAULT.digest == "db5ef1bc938e"


@pytest.mark.slow
def test_every_alternating_layout_is_unchanged() -> None:
    for path in sorted((ROOT / "rulesets").glob("*.json")):
        if not Engine(load_spec(path)).simultaneous:
            _alternating_layout_is_unchanged(path)


def test_every_stored_net_still_reads_as_its_game_s() -> None:
    seen = 0
    for weights in sorted((ROOT / "nets").glob("*.net")):
        path = STORED.get(weights.stem.split("-")[0])
        if path is None:
            continue
        engine = _engine(path)
        assert engine.spec.hash == weights.stem.split("-")[0]
        blob = weights.read_bytes()
        ((raw, _inputs, fresh),) = engine.core.net_line(_pack(engine.initial_state()), blob, [])
        assert raw == fresh == NetEvaluator(engine, blob, ()).raw(engine.initial_state())
        seen += 1
    assert seen >= 8


@pytest.mark.parametrize("path", ROUND_GAMES)
def test_a_round_line_is_the_oracles_to_the_bit(path: str) -> None:
    engine = _engine(path)
    layout = Layout.of(engine.spec)
    assert layout.rounds and engine.core.net_inputs() == layout.n_inputs
    moves, states = _round_line(engine, seed=7)
    start = engine.initial_state()
    for seed in (1, 2):
        blob = random_net(layout.n_inputs, seed=seed)
        oracle = NetEvaluator(engine, blob, ())
        answers = engine.core.net_line(_pack(start), blob, moves)
        assert len(answers) == len(states) == len(moves) // 2 + 1
        for index, ((raw, inputs, fresh), state) in enumerate(zip(answers, states, strict=True)):
            assert state.side_to_move == 0
            assert inputs == layout.features(state), (path, index)
            assert (raw, fresh) == (oracle.raw(state),) * 2, (path, seed, index)
        for probe in _probes(engine):
            ((raw, inputs, fresh),) = engine.core.net_line(_pack(probe), blob, [])
            assert inputs == layout.features(probe), (path, probe)
            assert raw == fresh == oracle.raw(probe)


def test_every_round_segment_is_reached_where_its_rule_is() -> None:
    hit: dict[str, set[str]] = {}
    for path in ROUND_GAMES:
        engine = _engine(path)
        layout = Layout.of(engine.spec)
        starts = {
            "initiative": (layout.initiative, layout.rest),
            "rest": (layout.rest, layout.void),
            "void": (layout.void, layout.bars),
            "bars": (layout.bars, layout.n_inputs),
        }
        states = [*_round_line(engine, seed=7)[1], *_probes(engine)]
        found = {
            name
            for state in states
            for feature in layout.features(state)
            for name, (low, high) in starts.items()
            if low <= feature < high
        }
        hit[path] = found
    assert hit[WATERING] == {"void", "bars"}
    assert hit[CHASE] == {"rest", "void", "bars"}
    assert hit[INITIATIVE] == {"initiative", "void", "bars"}


def test_a_round_line_is_whole_rounds_and_a_spy_game_has_none() -> None:
    engine = _engine(CHASE)
    blob = random_net(Layout.of(engine.spec).n_inputs, seed=1)
    moves, _ = _round_line(engine, seed=7)
    with pytest.raises(ValueError, match="odd count"):
        engine.core.net_line(_pack(engine.initial_state()), blob, moves[:3])
    spy = _engine("archive/gardner-with-spies.json")
    with pytest.raises(ValueError, match="spy game"):
        spy.core.net_line(
            _pack(spy.initial_state()), random_net(spy.core.net_inputs(), seed=1), []
        )


# ------------------------------------------------------------------ the players


def _walk(engine: Engine, rounds: int, seed: int) -> list[GameState]:
    rng = random.Random(seed)
    state = engine.initial_state()
    states = [state]
    for _ in range(rounds):
        if engine.terminal(state) is not None:
            break
        state = engine.apply_round(
            state,
            rng.choice(engine.legal_moves_for(state, 0)),
            rng.choice(engine.legal_moves_for(state, 1)),
        )
        states.append(state)
    return [state for state in states if engine.terminal(state) is None]


def _both(engine: Engine, agent: str, state: GameState, seat: int, seed: int) -> tuple[Any, Any]:
    native = native_spec(agent, engine.spec.hash)
    assert native is not None
    move, strategy, nodes, depth = engine.select_round(
        state, seat, native, eval_values(engine.spec), seed
    )
    player: Any = build_agent(agent, engine)
    lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
    oracle = player.select_round_move(
        state, seat, lists[seat], lists[1 - seat], seat_rng(seed, seat)
    )
    return (
        (move, list(strategy), nodes, depth),
        (oracle, list(player.strategy), player.stats.nodes, getattr(player, "depth", 0)),
    )


@pytest.mark.parametrize(("path", "rounds"), [(WATERING, 1), (CHASE, 3)])
def test_a_net_plays_rounds_as_the_oracle_does(path: str, rounds: int, registered: None) -> None:
    engine = _engine(path)
    _register(engine, seed=5)
    differs = False
    for index, state in enumerate(_walk(engine, rounds=rounds, seed=11)):
        for agent in (f"greedy:net={PROCEDURE}", f"matrix:net={PROCEDURE},nodes=300"):
            for seat in (0, 1):
                core, oracle = _both(engine, agent, state, seat, seed=index)
                assert core == oracle, (path, agent, index, seat)
                plain, _ = _both(engine, agent.replace(f"net={PROCEDURE}", "").replace(
                    ":,", ":").rstrip(":"), state, seat, seed=index)
                differs = differs or plain[1] != core[1]
    assert differs, "the net is what played: the built-in objective mixes otherwise"


def test_a_round_game_with_a_net_is_the_python_loops(registered: None) -> None:
    engine = _engine(CHASE)
    _register(engine, seed=6)
    agents = (f"matrix:net={PROCEDURE},nodes=250", "greedy")
    native = play_game(engine, agents, seed=3)
    in_python = play_game_python(engine, agents, seed=3)
    assert native.moves == in_python.moves
    assert (native.winner, native.reason, native.nodes) == (
        in_python.winner, in_python.reason, in_python.nodes,
    )


def test_a_spy_game_and_a_heuristic_still_refuse_a_round_net(registered: None) -> None:
    spy = _engine("archive/gardner-with-spies.json")
    _register(spy, seed=1)
    with pytest.raises(ValueError, match="no net plays a spy game"):
        build_agent(f"greedy:net={PROCEDURE}", spy)
    blob = random_net(spy.core.net_inputs(), seed=1, accumulator=16, hidden1=8)
    seat = ("greedy", 0, 0.05, 0, 0, 0.0, 16.0, (), (), 0.0, blob, 0)
    with pytest.raises(ValueError, match="spy game"):
        spy.core.play_game((seat, native_spec("random")), eval_values(spy.spec), 0)
    with pytest.raises(ValueError, match="built-in objective only"):
        native_spec("matrix:nodes=100,h=0123456789ab")


# ------------------------------------------------------------------ the teacher


def _oracle_value(engine: Engine, state: GameState, budget: int | None) -> tuple[float, int]:
    # The round players' defaults (`matrix._native_matrix`, `simple._native_greedy`).
    search = RoundSearch(engine, round_evaluator(engine, 0.05, GOAL_PIECES, IMMUNE_PIECES))
    lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
    return round_value_for_first(search, state, lists, budget)


@pytest.mark.parametrize("path", (WATERING, CHASE))
def test_the_round_teacher_is_the_oracles_at_any_thread_count(path: str) -> None:
    engine = _engine(path)
    states = _walk(engine, rounds=3, seed=4)
    packed = [_pack(state) for state in states]
    values = eval_values(engine.spec)
    for agent, budget in (("greedy", None), ("matrix:nodes=1", None), ("matrix:nodes=400", 400)):
        native = native_spec(agent, engine.spec.hash)
        one = engine.core.round_values(packed, native, values)
        assert engine.core.round_values(packed, native, values, threads=3) == one
        for state, answer in zip(states, one, strict=True):
            assert answer is not None
            value, depth = answer
            expected = _oracle_value(engine, state, budget)
            assert (value, depth) == expected, (path, agent)
    with pytest.raises(ValueError, match="greedy, matrix"):
        engine.core.round_values(packed, native_spec("random"), values)
    with pytest.raises(ValueError, match="rounds"):
        gardner = _engine("rulesets/gardner.json")
        gardner.core.round_values(
            [_pack(gardner.initial_state())], native_spec("greedy"), eval_values(gardner.spec)
        )


# ------------------------------------------------------------------ the data


def test_a_round_opening_is_whole_rounds_from_the_seeds_streams() -> None:
    engine = _engine(CHASE)
    for seed in range(12):
        state, moves = _round_opening(engine, seed, 8)
        assert len(moves) == 2 * (opening_plies_for(seed, 8) // 2)
        again = _round_opening(engine, seed, 8)
        assert again[1] == moves and _pack(again[0]) == _pack(state)
    record = diverse_game(engine, (5, "greedy", "greedy"), 8)
    assert record.plies == len(record.moves) and len(record.moves) % 2 == 0
    state = engine.initial_state()
    for index in range(0, len(record.moves), 2):
        assert record.moves[index] in engine.legal_moves_for(state, 0)
        assert record.moves[index + 1] in engine.legal_moves_for(state, 1)
        state = engine.apply_round(state, record.moves[index], record.moves[index + 1])
    assert engine.terminal(state) is not None


def test_a_round_harvest_is_seat_0_s_a_round_and_not_the_workers() -> None:
    spec = load_spec(ROOT / CHASE)
    records = self_play(spec, TINY, workers=1)
    assert [r.moves for r in records] == [r.moves for r in self_play(spec, TINY, workers=2)]
    one = harvest(spec, records, TINY, workers=1)
    two = harvest(spec, records, TINY, workers=2)
    assert one.digest == two.digest
    assert one.values is not None and two.values is not None
    assert np.array_equal(one.values, two.values, equal_nan=True)
    assert set(one.sides.tolist()) == {0}
    assert one.ply is not None and all(p % 2 == 0 and p >= 2 for p in one.ply.tolist())
    rounds = sum(len(r.moves) // 2 - 1 for r in records)
    assert one.positions == rounds
    assert one.teacher == "player" and not np.isnan(one.values).any()
    first = [r.score_for(0) for r in records]
    assert one.results is not None and one.game is not None
    assert one.results.tolist() == [first[g] for g in one.game.tolist()]


def test_a_round_label_is_the_round_players_value_squashed() -> None:
    from chess_variant_sandbox.agents.training.data import squash

    spec = load_spec(ROOT / CHASE)
    engine = Engine(spec)
    procedure = TINY.with_(games=1, td_lambda=0.0)
    records = self_play(spec, procedure, workers=1)
    data = harvest(spec, records, procedure, workers=1)
    moves = list(records[0].moves)
    state = engine.apply_round(engine.initial_state(), moves[0], moves[1])
    native = native_spec("matrix:nodes=150", engine.spec.hash)
    ((value, _depth),) = engine.core.round_values([_pack(state)], native, eval_values(engine.spec))
    assert data.values is not None
    assert data.values[0] == squash(value, procedure.scale)
    assert data.targets[0] == np.float32(squash(value, procedure.scale))


# ------------------------------------------------------------------ the recipe


def test_the_round_recipe_and_its_control_are_pinned() -> None:
    assert ROUND_DEFAULT.digest == ROUND_DEFAULT_DIGEST
    assert ROUND_CONTROL.digest == ROUND_CONTROL_DIGEST
    assert ROUND_DEFAULT.rounds and ROUND_DEFAULT.searcher == "matrix"
    assert not DEFAULT.rounds and DEFAULT.searcher == "minimax"
    assert (ROUND_DEFAULT.agents, ROUND_DEFAULT.label_search, ROUND_DEFAULT.label_nodes) == (
        ("matrix:nodes=1600",), "player", 6400,
    )
    assert ROUND_DEFAULT.with_(label_nodes=1) == ROUND_CONTROL
    assert control(ROUND_DEFAULT) == ROUND_CONTROL
    assert control(TINY) == TINY.with_(label_nodes=1)
    with pytest.raises(ValueError, match="generation 0"):
        builtin_player_arm(ROUND_DEFAULT)
    assert ROUND_ACCEPT_NODES == 1600


def test_a_round_recipe_that_cannot_work_is_refused_in_words() -> None:
    with pytest.raises(ValueError, match="no probe teacher"):
        ROUND_DEFAULT.with_(label_search="probe")
    with pytest.raises(ValueError, match="cannot sit beside it"):
        Procedure(agents=("matrix:nodes=100", "minimax:nodes=100"), label_search="player")
    ROUND_DEFAULT.with_(result_weight=1.0, td_lambda=0.0)  # results only needs no teacher


def test_the_acceptance_duel_seats_the_round_player(registered: None) -> None:
    spec = load_spec(ROOT / CHASE)
    engine = Engine(spec)
    procedure = TINY
    blob = random_net(Layout.of(engine.spec).n_inputs, seed=2, accumulator=16, hidden1=8)
    register_net(NetRecord(procedure.digest, engine.spec.hash, ACCEPTED, blob))
    try:
        found = accept(spec, procedure, pairs=1, nodes=200, workers=1)
    finally:
        nets._NETS.pop((procedure.digest, engine.spec.hash), None)
    assert found.candidate == f"matrix:net={procedure.digest},nodes=200"
    assert found.reference == "matrix:nodes=200"
    assert found.games == 2
