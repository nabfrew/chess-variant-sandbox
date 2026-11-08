"""Pins for the trainer's pieces that only ran end to end (Q1152, `agents/training`).

Each test holds one helper's whole output on a small inline input, so a change to a bound,
a default, a rounding place or a skip is seen: the dataset's size check, `squash`, the
replay set's nearest link, the refusal of an unresolved net, the harvest's chunks and its
progress lines (in a process pool and in a campaign), a round label the teacher could not
give, the float net's clips, padding, heads and gradients, a resumed and an unbroken
training's losses, the quantized biases' i32 bound, a procedure's width and count bounds,
the store's player and payload, the acceptance duel's node budget and node counts, and the
quantization error on a round game. The process pool, the duel's games and a resumed
run's saves are stood in for, so every test stays fast.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from chess_variant_sandbox.agents import nets  # noqa: E402
from chess_variant_sandbox.agents.nets import (  # noqa: E402
    MAX_ACCUMULATOR,
    MAX_HIDDEN,
    NetError,
    NetRecord,
    random_net,
)
from chess_variant_sandbox.agents.training import data as tdata  # noqa: E402
from chess_variant_sandbox.agents.training import model, store  # noqa: E402
from chess_variant_sandbox.agents.training.data import Dataset, squash  # noqa: E402
from chess_variant_sandbox.agents.training.procedure import (  # noqa: E402
    ROUND_DEFAULT,
    Parent,
    Procedure,
    Replay,
)
from chess_variant_sandbox.agents.training.store import register_parent  # noqa: E402
from chess_variant_sandbox.engine import Engine  # noqa: E402
from chess_variant_sandbox.spec import load_spec  # noqa: E402
from chess_variant_sandbox.tournament import checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GARDNER = ROOT / "rulesets" / "gardner.json"
ROUNDS = ROOT / "archive" / "breather.json"

SMALL = Procedure(
    agents=("greedy",),
    games=3,
    skip_plies=2,
    result_weight=1.0,
    label_nodes=0,
    td_lambda=0.0,
    accumulator=4,
    hidden1=3,
    hidden2=2,
    epochs=2,
    batch=2,
    learning_rate=0.01,
)
ROUND_SMALL = ROUND_DEFAULT.with_(
    agents=("matrix:nodes=20",),
    games=1,
    label_nodes=150,
    td_lambda=0.0,
    accumulator=4,
    hidden1=3,
    hidden2=2,
)


@pytest.fixture
def clean() -> Iterator[None]:
    before = dict(nets._NETS)
    yield
    nets._NETS.clear()
    nets._NETS.update(before)


def _dataset() -> Dataset:
    """Five positions over six inputs, both heads, a batch of two leaving one over."""
    return Dataset(
        n_inputs=6,
        offsets=np.array([0, 2, 3, 6, 7, 9], dtype=np.int64),
        inputs=np.array([0, 3, 1, 0, 2, 5, 4, 1, 5], dtype=np.int32),
        sides=np.array([0, 1, 0, 1, 1], dtype=np.int8),
        targets=np.array([1.0, 0.0, 0.5, 0.25, 0.75], dtype=np.float32),
        games=2,
    )


def _record(seed: int, moves: list[int] | None = None) -> Any:
    return SimpleNamespace(moves=list(moves or []), seed=seed, score_for=lambda seat: 0.5)


# ------------------------------------------------------------------ data


def test_a_dataset_refuses_targets_of_another_size() -> None:
    with pytest.raises(ValueError, match="disagree on its size"):
        Dataset(
            n_inputs=4,
            offsets=np.array([0, 1, 2], dtype=np.int64),
            inputs=np.array([0, 1], dtype=np.int32),
            sides=np.array([0, 1], dtype=np.int8),
            targets=np.array([0.5], dtype=np.float32),
            games=1,
        )


def test_squash_is_the_logistic_of_score_over_scale() -> None:
    assert squash(1.0, 1.0) == pytest.approx(0.7310585786300049, abs=1e-15)
    assert squash(3.0, 2.0) == pytest.approx(0.8175744761936437, abs=1e-15)
    assert squash(0.0, 5.0) == 0.5


def test_a_chained_replay_plays_its_nearest_links_bytes(clean: None) -> None:
    spec = load_spec(GARDNER)
    engine = Engine(spec)
    n_inputs = int(engine.core.net_inputs())
    links = []
    blobs = []
    for index, name in enumerate(("a00000000001", "a00000000002")):
        blob = random_net(n_inputs, seed=index + 3, accumulator=4, hidden1=3, hidden2=2)
        parent = Parent(procedure=name, weights_sha256=hashlib.sha256(blob).hexdigest())
        register_parent(engine.spec.hash, parent, blob)
        links.append(SimpleNamespace(net=parent))
        blobs.append(blob)
    assert blobs[0] != blobs[1]
    procedure = SimpleNamespace(replay=SimpleNamespace(links=tuple(links)))
    assert tdata.replay_bytes(spec, procedure) == blobs[0]  # type: ignore[arg-type]


def test_an_unresolved_net_is_refused_by_its_hash_prefix() -> None:
    spec = load_spec(GARDNER)
    parent = Parent(procedure="b00000000002", weights_sha256="0123456789abcdef" * 4)
    with pytest.raises(NetError) as raised:
        tdata._checked_bytes(spec, parent, "parent")
    assert "the weights 0123456789ab... the procedure names (no net);" in str(raised.value)


def test_chunks_cut_the_games_in_order_from_the_first() -> None:
    games = [(i, [], 0.5, i) for i in range(5)]
    assert tdata._chunks(games, 1) == [[game] for game in games]  # type: ignore[arg-type]
    # Eight chunks a worker: eight games are eight chunks of one.
    eight = [(i, [], 0.5, i) for i in range(8)]
    assert tdata._chunks(eight, 1) == [[game] for game in eight]  # type: ignore[arg-type]
    twenty = [(i, [], 0.5, i) for i in range(20)]
    cut = tdata._chunks(twenty, 1)  # type: ignore[arg-type]
    assert [[g[0] for g in chunk] for chunk in cut] == [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19]
    ]


class _InlinePool:
    """A process pool run in this process, in order: what `harvest` sees of one."""

    def __init__(self, max_workers: int, mp_context: Any, initializer: Any, initargs: Any):
        self.workers = max_workers
        initializer(*initargs)

    def __enter__(self) -> _InlinePool:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def map(self, fn: Any, items: Any) -> Iterator[Any]:
        return (fn(item) for item in items)


def test_a_pooled_harvest_reports_a_tenth_of_its_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = load_spec(GARDNER)
    monkeypatch.setattr(tdata, "ProcessPoolExecutor", _InlinePool)
    # Two workers is a pool: four games are four chunks, each reported.
    lines: list[str] = []
    found = tdata.harvest(spec, [_record(i) for i in range(4)], SMALL, workers=2, report=lines.append)
    assert found.positions == 0 and found.games == 4
    assert lines == [f"  harvested {i}/4 chunks, 0 positions" for i in (1, 2, 3, 4)]
    # 32 chunks report every third: 32 // 10.
    lines.clear()
    tdata.harvest(spec, [_record(i) for i in range(32)], SMALL, workers=4, report=lines.append)
    assert lines == [f"  harvested {i}/32 chunks, 0 positions" for i in range(3, 33, 3)]
    # Two games are already worth a pool.
    lines.clear()
    tdata.harvest(spec, [_record(i) for i in range(2)], SMALL, workers=2, report=lines.append)
    assert lines == [f"  harvested {i}/2 chunks, 0 positions" for i in (1, 2)]
    # One worker is no pool, however many games.
    lines.clear()
    tdata.harvest(spec, [_record(i) for i in range(4)], SMALL, workers=1, report=lines.append)
    assert lines == []


def test_a_campaign_harvest_reports_its_pieces(tmp_path: Path) -> None:
    spec = load_spec(GARDNER)
    lines: list[str] = []
    with checkpoint.active(tmp_path / "campaign", build="pins"):
        found = tdata.harvest(
            spec, [_record(i) for i in range(3)], SMALL, workers=1, report=lines.append
        )
    assert found.games == 3
    assert lines == [f"  harvested {i}/3 games" for i in (1, 2, 3)]


def test_a_campaign_harvest_of_many_games_reports_a_tenth_and_the_end(tmp_path: Path) -> None:
    spec = load_spec(GARDNER)
    lines: list[str] = []
    with checkpoint.active(tmp_path / "campaign", build="pins"):
        tdata.harvest(spec, [_record(i) for i in range(25)], SMALL, workers=1, report=lines.append)
    assert lines == CAMPAIGN_LINES
    # 18 games are a step of one: every piece reports.
    lines.clear()
    with checkpoint.active(tmp_path / "eighteen", build="pins"):
        tdata.harvest(spec, [_record(i) for i in range(18)], SMALL, workers=1, report=lines.append)
    assert lines == [f"  harvested {i}/18 games" for i in range(1, 19)]
    # 20 games are a step of two.
    lines.clear()
    with checkpoint.active(tmp_path / "twenty", build="pins"):
        tdata.harvest(spec, [_record(i) for i in range(20)], SMALL, workers=1, report=lines.append)
    assert lines == [f"  harvested {i}/20 games" for i in range(2, 21, 2)]


class _NoProcesses:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("a one-worker harvest started a process pool")


def test_a_one_worker_campaign_harvest_starts_no_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = load_spec(GARDNER)
    monkeypatch.setattr(tdata, "ProcessPoolExecutor", _NoProcesses)
    with checkpoint.active(tmp_path / "campaign", build="pins"):
        found = tdata.harvest(spec, [_record(i) for i in range(3)], SMALL, workers=1)
    assert found.games == 3


def test_replay_link_0_is_the_new_games_and_not_a_link() -> None:
    procedure = SimpleNamespace(replay=SimpleNamespace(links=("one back", "two back")))
    assert tdata._replay_link(procedure, 1) == "one back"  # type: ignore[arg-type]
    assert tdata._replay_link(procedure, 2) == "two back"  # type: ignore[arg-type]
    for link in (0, 3):
        with pytest.raises(ValueError, match=rf"^replay link {link} is not one of this procedure's \(1\.\.2; 0 is the new games\)$"):
            tdata._replay_link(procedure, link)  # type: ignore[arg-type]


# A step of 25 // 10 games, then the last game alone.
CAMPAIGN_LINES = [f"  harvested {i}/25 games" for i in (*range(2, 25, 2), 25)]


def test_an_empty_dataset_has_no_active_input() -> None:
    empty = Dataset(
        n_inputs=4,
        offsets=np.zeros(1, dtype=np.int64),
        inputs=np.zeros(0, dtype=np.int32),
        sides=np.zeros(0, dtype=np.int8),
        targets=np.zeros(0, dtype=np.float32),
        games=0,
    )
    assert (empty.positions, empty.most_active) == (0, 0)
    assert _dataset().most_active == 3


def test_the_zero_net_is_all_zeros_of_width_one() -> None:
    parsed = nets.Net.parse(tdata.zero_net(5))
    assert (parsed.n_inputs, parsed.accumulator, parsed.hidden1, parsed.hidden2) == (5, 1, 1, 1)
    assert parsed.w0 == (0,) * 5 and parsed.b0 == (0,)
    for head in parsed.heads:
        assert (head.w1, head.b1, head.w2, head.b2, head.w3, head.b3) == (
            (0,), (0,), (0,), (0,), (0,), 0
        )


def test_a_replay_links_schedule_collects_no_series() -> None:
    net = Parent(procedure="c00000000003", weights_sha256="ab" * 32)
    replay = Replay(
        agents=("minimax:net=c00000000003,nodes=10", "greedy"), games=2, seed=9,
        opening_plies=0, net=net,
    )
    config = tdata._link_config(replay, 3)
    assert (config.agents, config.games_per_pairing, config.seed, config.workers) == (
        replay.agents, 2, 9, 3
    )
    assert config.collect_series is False


class _Harvests:
    def __init__(self, datasets: list[Dataset]) -> None:
        self.datasets = list(datasets)

    def __call__(self, *args: Any, **kwargs: Any) -> Dataset:
        return self.datasets.pop(0)


def test_arms_whose_inputs_disagree_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    base = _dataset()
    other = replace(base, inputs=base.inputs[::-1].copy())
    monkeypatch.setattr(tdata, "self_play", lambda *args, **kwargs: [])
    monkeypatch.setattr(tdata, "harvest", _Harvests([base, base]))
    assert len(tdata.harvest_arms({}, [SMALL, SMALL])) == 2
    monkeypatch.setattr(tdata, "harvest", _Harvests([base, other]))
    with pytest.raises(AssertionError, match="disagree on the positions"):
        tdata.harvest_arms({}, [SMALL, SMALL])


def test_self_play_s_schedule_collects_no_series() -> None:
    config = tdata._config(SMALL.with_(seed=4), 3)
    assert (config.agents, config.games_per_pairing, config.seed, config.workers) == (
        ("greedy",), 3, 4, 3
    )
    assert config.collect_series is False


def test_the_opening_length_is_the_seeds_and_reaches_its_top() -> None:
    assert [tdata.opening_plies_for(seed, 1) for seed in range(12)] == OPENINGS_1
    assert [tdata.opening_plies_for(seed, 8) for seed in range(12)] == OPENINGS_8


OPENINGS_1 = [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1]
OPENINGS_8 = [8, 8, 0, 2, 6, 7, 5, 1, 3, 8, 2, 4]


class _CoreLosingFirst:
    def __init__(self, core: Any) -> None:
        self._core = core

    def __getattr__(self, name: str) -> Any:
        return getattr(self._core, name)

    def round_values(self, *args: Any, **kwargs: Any) -> list[Any]:
        answers = list(self._core.round_values(*args, **kwargs))
        return [None, *answers[1:]]


class _EngineLosingFirst:
    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self.core = _CoreLosingFirst(engine.core)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)


def test_a_round_the_teacher_cannot_value_keeps_its_result_and_the_rest_are_labelled() -> None:
    spec = load_spec(ROUNDS)
    engine = Engine(spec)
    state = engine.initial_state()
    moves: list[int] = []
    for _ in range(4):
        assert engine.terminal(state) is None
        first = engine.legal_moves_for(state, 0)[0]
        second = engine.legal_moves_for(state, 1)[0]
        moves += [first, second]
        state = engine.apply_round(state, first, second)
    tdata._setup(spec, ROUND_SMALL)
    try:
        tdata._WORKER["engine"] = _EngineLosingFirst(tdata._WORKER["engine"])
        _rows, sides, targets, _indices, plies, results, values = tdata._harvest_rounds(
            [(0, moves, 1.0, 7)]
        )
    finally:
        tdata._WORKER.clear()
    assert plies == [2, 4, 6] and sides == [0, 0, 0] and results == [1.0, 1.0, 1.0]
    assert math.isnan(values[0]) and targets[0] == 1.0
    assert not math.isnan(values[1]) and not math.isnan(values[2])
    assert targets[1:] == values[1:]


# ------------------------------------------------------------------ the float net


def test_a_new_net_starts_its_accumulator_half_open() -> None:
    net = model.initial(6, SMALL, 3)
    assert net.params["b0"].tolist() == [0.5, 0.5, 0.5, 0.5]


def test_the_accumulator_bias_is_clipped_to_one_each_way() -> None:
    net = model.initial(6, SMALL, 3)
    net.params["b0"][:] = [-3.0, -0.5, 0.25, 3.0]
    net.params["w0"][0, :2] = [5.0, -5.0]
    model._clip(net)
    assert net.params["b0"].tolist() == [-1.0, -0.5, 0.25, 1.0]
    # W0's bound is its integer limit (127 at three active inputs) over 127.
    assert net.w0_limit == 127 and net.params["w0"][0, :2].tolist() == [1.0, -1.0]


def test_padding_is_as_wide_as_the_widest_row_and_never_narrower_than_one() -> None:
    offsets = np.array([0, 1, 2], dtype=np.int64)
    inputs = np.array([7, 8], dtype=np.int32)
    assert model._padded(offsets, inputs, np.array([1, 0]), 99).tolist() == [[8], [7]]
    empty = np.array([0, 0, 0], dtype=np.int64)
    assert model._padded(empty, inputs[:0], np.array([0, 1]), 99).tolist() == [[99], [99]]


def test_padding_is_each_rows_own_inputs_then_the_sentinel() -> None:
    offsets = np.array([0, 2, 3, 6], dtype=np.int64)
    inputs = np.array([10, 11, 12, 13, 14, 15], dtype=np.int32)
    out = model._padded(offsets, inputs, np.array([2, 0, 1]), 99)
    assert out.tolist() == [[13, 14, 15], [10, 11, 99], [12, 99, 99]]


def test_the_second_head_answers_without_the_first() -> None:
    net = model.initial(6, SMALL, 3)
    alone = model.predict(net, [[0, 3]], [1])
    mixed = model.predict(net, [[1], [0, 3]], [0, 1])
    assert alone[0] == mixed[1] != 0.0


def _batch(sides: list[int]) -> tuple[Any, Any, Any]:
    data = _dataset()
    rows = np.arange(len(sides))
    index = model._padded(data.offsets, data.inputs, rows, data.n_inputs)
    return index, np.array(sides, dtype=np.int8), data.targets[rows]


def test_a_head_with_no_position_has_zero_gradients_and_the_other_has_its_own() -> None:
    net = model.initial(6, SMALL, 3)
    index, sides, targets = _batch([1, 1, 1])
    _loss, grads = model._gradients(net, index, sides, targets, 1.0)
    assert set(grads) == set(net.params)
    for key in model.HEAD_KEYS:
        assert not grads[f"{key}_0"].any(), key
    assert grads["w3_1"].any() and grads["b3_1"].any()


def test_a_hidden_unit_at_the_top_of_its_clip_passes_no_gradient() -> None:
    net = model.initial(6, SMALL, 3)
    net.params["w2_0"][:] = 0.0
    net.params["b2_0"][:] = 1.0
    index, sides, targets = _batch([0, 0, 0])
    _loss, grads = model._gradients(net, index, sides, targets, 1.0)
    assert grads["b3_0"].any()
    assert not grads["b2_0"].any() and not grads["w2_0"].any() and not grads["w1_0"].any()


def test_training_a_small_dataset_gives_these_losses() -> None:
    lines: list[str] = []
    net = model.train(_dataset(), SMALL, report=lines.append)
    assert net.history == TRAINED_HISTORY
    assert lines == ["  epoch 1/2: loss 0.115102", "  epoch 2/2: loss 0.108615"]


def test_a_dense_weights_integer_bound_is_i8_and_w0s_keeps_i16() -> None:
    assert model.w0_int_limit(1) == 127
    assert model.w0_int_limit(0) == 127
    assert model.w0_int_limit(300) == 108
    assert model.w0_int_limit(40_000) == 1
    # 32,640 = 7 x 4,663 - 1: one unit below the next step.
    assert model.w0_int_limit(4_663) == 6
    assert model.DENSE_LIMIT == 127 / 64


TRAINED_HISTORY = [0.11510221, 0.10861544]


class _Saves:
    def __init__(self, arrays: dict[str, Any], meta: dict[str, Any]) -> None:
        self.found = (arrays, meta)
        self.saved: list[dict[str, Any]] = []

    def latest(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.found

    def due(self) -> bool:
        return False

    def save(self, arrays: dict[str, Any], meta: dict[str, Any]) -> None:
        self.saved.append(meta)


def test_a_resumed_training_says_where_and_carries_its_partial_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = model.initial(6, SMALL, 3)
    arrays: dict[str, Any] = {}
    for key, value in start.params.items():
        arrays[f"p_{key}"] = value
        arrays[f"m_{key}"] = np.zeros_like(value)
        arrays[f"v_{key}"] = np.zeros_like(value)
    meta = {
        "step": 3,
        "history": [0.25],
        "epoch": 1,
        "begin": 2,
        "total": 0.5,
        "rng": dict(np.random.default_rng(9).bit_generator.state),
    }
    saves = _Saves(arrays, meta)
    monkeypatch.setattr(model, "_saves", lambda data, procedure: saves)
    lines: list[str] = []
    net = model.train(_dataset(), SMALL, report=lines.append)
    assert lines[0] == "  resumed at epoch 2, position 2, step 3"
    assert net.history[0] == 0.25 and len(net.history) == 2
    assert net.history[1] == RESUMED_LOSS
    assert saves.saved[-1]["epoch"] == 2 and saves.saved[-1]["step"] == 5
    assert saves.saved[-1]["where"] == "epoch 3, position 0, step 5"
    assert (saves.saved[-1]["begin"], saves.saved[-1]["total"]) == (0, 0.0)


def test_a_run_resumed_after_its_last_epoch_saves_nothing_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = model.initial(6, SMALL, 3)
    arrays: dict[str, Any] = {}
    for key, value in start.params.items():
        arrays[f"p_{key}"] = value
        arrays[f"m_{key}"] = np.zeros_like(value)
        arrays[f"v_{key}"] = np.zeros_like(value)
    meta = {"step": 6, "history": [0.25, 0.2], "epoch": 2, "begin": 0, "total": 0.0,
            "rng": dict(np.random.default_rng(9).bit_generator.state)}
    saves = _Saves(arrays, meta)
    monkeypatch.setattr(model, "_saves", lambda data, procedure: saves)
    net = model.train(_dataset(), SMALL)
    assert net.history == [0.25, 0.2] and saves.saved == []


RESUMED_LOSS = 0.18316454


def test_a_quantized_bias_stops_at_the_i32_bound() -> None:
    net = model.initial(6, SMALL, 3)
    net.params["b1_0"][0] = 3.0e5
    net.params["b1_0"][1] = -3.0e5
    parsed = nets.Net.parse(model.quantize(net))
    assert parsed.heads[0].b1[:2] == (2**31 - 1, -(2**31 - 1))


# ------------------------------------------------------------------ the procedure


def test_the_accumulator_width_is_one_to_the_cores_most() -> None:
    assert SMALL.with_(accumulator=MAX_ACCUMULATOR).accumulator == MAX_ACCUMULATOR
    assert SMALL.with_(accumulator=1).accumulator == 1
    with pytest.raises(ValueError, match="accumulator width 0 is outside 1"):
        SMALL.with_(accumulator=0)


@pytest.mark.parametrize("name", ["games", "stride", "epochs", "batch"])
def test_a_count_of_zero_is_refused(name: str) -> None:
    assert getattr(SMALL.with_(**{name: 1}), name) == 1
    with pytest.raises(ValueError, match=f"^{name} must be at least 1$"):
        SMALL.with_(**{name: 0})


@pytest.mark.parametrize("name", ["hidden1", "hidden2"])
def test_a_hidden_width_is_one_to_the_cores_most(name: str) -> None:
    assert getattr(SMALL.with_(**{name: MAX_HIDDEN}), name) == MAX_HIDDEN
    with pytest.raises(ValueError, match=f"^{name} width 0 is outside 1..{MAX_HIDDEN}$"):
        SMALL.with_(**{name: 0})
    with pytest.raises(ValueError, match=f"^{name} width {MAX_HIDDEN + 1} is outside"):
        SMALL.with_(**{name: MAX_HIDDEN + 1})


def test_result_weight_is_zero_to_one() -> None:
    assert SMALL.with_(result_weight=0.0, label_nodes=40).result_weight == 0.0
    with pytest.raises(ValueError, match="^result_weight -0.5 is outside"):
        SMALL.with_(result_weight=-0.5)


def test_a_procedure_cannot_be_changed_in_place() -> None:
    from dataclasses import FrozenInstanceError, replace

    with pytest.raises(FrozenInstanceError):
        SMALL.games = 7  # type: ignore[misc]
    assert SMALL.games == 3
    # Nor can its replay set, which its name carries.
    net = Parent(procedure="c00000000003", weights_sha256="ab" * 32)
    replay = Replay(
        agents=("minimax:net=c00000000003,nodes=10", "greedy"), games=1, seed=0,
        opening_plies=0, net=net,
    )
    with pytest.raises(FrozenInstanceError):
        replay.games = 2  # type: ignore[misc]
    assert replay.games == 1 and hash(replay) == hash(replace(replay))


def test_td_lambda_is_zero_to_one_inclusive() -> None:
    searched = SMALL.with_(result_weight=0.5, label_nodes=40)
    assert searched.with_(td_lambda=1.0).td_lambda == 1.0
    with pytest.raises(ValueError, match="^td_lambda 1.5 is outside"):
        searched.with_(td_lambda=1.5)


def test_a_negative_opening_is_refused() -> None:
    assert SMALL.with_(opening_plies=0).opening_plies == 0
    with pytest.raises(ValueError, match="^opening_plies must be 0 or more$"):
        SMALL.with_(opening_plies=-1)


def test_a_negative_skip_is_refused_whatever_the_scale() -> None:
    assert SMALL.with_(skip_plies=0).skip_plies == 0
    with pytest.raises(ValueError, match="^skip_plies, scale and learning_rate must be positive$"):
        SMALL.with_(skip_plies=-1)


# ------------------------------------------------------------------ the store


def _trained(procedure: Procedure) -> store.Trained:
    return store.Trained(
        record=None,  # type: ignore[arg-type]
        procedure=procedure,
        data=None,  # type: ignore[arg-type]
        history=(),
        quantization_error=0.0,
        acceptance=None,
        meta_path=Path("x.json"),
        seconds={},
    )


def test_a_trained_nets_player_is_its_searcher_at_the_duels_budget() -> None:
    assert _trained(SMALL).player == f"minimax:net={SMALL.digest},nodes=6400"
    assert _trained(ROUND_SMALL).player == f"matrix:net={ROUND_SMALL.digest},nodes=1600"
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        _trained(SMALL).quantization_error = 1.0  # type: ignore[misc]


def test_the_payload_keeps_six_places_of_the_quantization_error() -> None:
    record = NetRecord("abcabcabcabc", "0123", "unchecked", b"weights")
    payload = store.store_payload(record, SMALL, _dataset(), (0.5,), 7, 0.123456789, None)
    assert payload["training"] == {
        "loss_by_epoch": [0.5],
        "w0_limit": 7,
        "quantization_error": 0.123457,
    }


def test_the_duel_defaults_to_the_built_ins_budget_and_counts_each_seats_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chess_variant_sandbox.tournament.duel as duel

    seen: list[Any] = []

    def games(spec: Any, config: Any, most: int) -> list[Any]:
        seen.append(config)
        candidate, reference = config.agents
        return [
            SimpleNamespace(agents=(candidate, reference), nodes=(10, 20), moves=[1, 2], seed=5),
            SimpleNamespace(agents=(reference, candidate), nodes=(30, 45), moves=[1, 2], seed=5),
        ]

    monkeypatch.setattr(store, "_opening_duel", games)
    monkeypatch.setattr(
        duel,
        "grade",
        lambda records, candidate, reference: SimpleNamespace(
            score=0.5, error=0.0, games=2, identical=0, pairs=[0.5]
        ),
    )
    found = store.accept(load_spec(GARDNER), SMALL, pairs=1, workers=1, opening_plies=2)
    assert found.candidate == f"minimax:net={SMALL.digest},nodes=6400"
    assert found.reference == "minimax:nodes=6400"
    assert (found.candidate_nodes, found.reference_nodes) == (55, 50)
    assert len(seen) == 1
    # Unnamed, the pairs and the seed are the fixed rule's.
    seen.clear()
    opened: list[int] = []
    monkeypatch.setattr(
        store, "_opening_duel", lambda spec, config, most: opened.append(most) or games(spec, config, most)
    )
    found = store.accept(load_spec(GARDNER), SMALL, nodes=60, workers=1)
    assert opened == [8] and found.opening_plies == 8
    assert (seen[0].games_per_pairing, seen[0].seed) == (200, 1_000_000)
    assert (found.seed, found.planned) == (1_000_000, 200)
    assert found.as_dict()["planned_pairs"] == 200


def test_an_acceptance_without_a_plan_plans_what_it_decided() -> None:
    found = store.Acceptance(
        candidate="a", reference="b", score=0.61234567, error=0.04567891, games=6,
        identical=0, candidate_nodes=1, reference_nodes=2, accepted=False,
    )
    assert (found.as_dict()["score"], found.as_dict()["error"]) == (0.612346, 0.045679)
    assert found.as_dict()["planned_pairs"] == 3 == found.decided_after


def test_a_store_is_written_at_an_indent_of_two(tmp_path: Path) -> None:
    blob = model.quantize(model.initial(6, SMALL, 3))
    record = NetRecord("abcabcabcabc", "0123", "unchecked", blob)
    meta = store.write_store(record, {"a": [1], "b": {"c": None}}, tmp_path)
    assert meta.read_bytes() == b'{\n  "a": [\n    1\n  ],\n  "b": {\n    "c": null\n  }\n}\n'


def test_no_held_out_position_is_no_error(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = load_spec(GARDNER)
    n_inputs = int(Engine(spec).core.net_inputs())
    net = model.initial(n_inputs, SMALL, 8)
    asked: list[Any] = []

    def no_games(spec: Any, procedure: Procedure, workers: int = 0) -> list[Any]:
        asked.append(procedure)
        return []

    monkeypatch.setattr(store, "self_play", no_games)
    assert store.quantization_error(spec, net, model.quantize(net), SMALL.with_(seed=5)) == 0.0
    # Four new games of the next schedule: never the training's own.
    assert asked == [SMALL.with_(games=4, seed=6)]


def test_each_stage_is_timed_from_its_own_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean: None
) -> None:
    from dataclasses import replace

    spec = load_spec(GARDNER)
    n_inputs = int(Engine(spec).core.net_inputs())
    ticks = iter(float(t) for t in (100, 101, 200, 203, 300, 306))
    monkeypatch.setattr(store, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    monkeypatch.setattr(store, "self_play", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        store, "harvest", lambda *args, **kwargs: replace(_dataset(), n_inputs=n_inputs)
    )
    monkeypatch.setattr(store, "quantization_error", lambda *args: 0.0)
    lines: list[str] = []
    done = store.train_ruleset(
        spec, SMALL, root=tmp_path, workers=1, duel=False, report=lines.append
    )
    assert done.seconds == {"self_play": 1.0, "harvest": 3.0, "train": 6.0}
    assert "  5 positions harvested in 3.0s" in lines


def test_an_alternating_games_quantization_error_is_every_position() -> None:
    spec = load_spec(GARDNER)
    n_inputs = int(Engine(spec).core.net_inputs())
    net = model.initial(n_inputs, SMALL, 8)
    error = store.quantization_error(spec, net, model.quantize(net), SMALL)
    assert 0.0 < error < 0.1
    assert error == pytest.approx(ALTERNATING_ERROR, rel=1e-9)


ALTERNATING_ERROR = 0.013826734971899523


def test_the_duel_is_timed_from_its_own_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean: None
) -> None:
    from dataclasses import replace

    spec = load_spec(GARDNER)
    n_inputs = int(Engine(spec).core.net_inputs())
    ticks = iter(float(t) for t in (100, 102, 500, 505))
    monkeypatch.setattr(store, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    monkeypatch.setattr(store, "quantization_error", lambda *args: 0.0)
    found = store.Acceptance(
        candidate="a", reference="b", score=0.5, error=0.0, games=2, identical=0,
        candidate_nodes=1, reference_nodes=1, accepted=False,
    )
    monkeypatch.setattr(store, "accept", lambda *args, **kwargs: found)
    done = store.train_ruleset(
        spec, SMALL, root=tmp_path, workers=1, data=replace(_dataset(), n_inputs=n_inputs)
    )
    assert done.seconds == {"train": 2.0, "duel": 5.0}
    assert done.acceptance is found and done.record.verdict == "refused"


def test_a_round_games_quantization_error_is_seat_0s_every_round() -> None:
    spec = load_spec(ROUNDS)
    n_inputs = int(Engine(spec).core.net_inputs())
    net = model.initial(n_inputs, ROUND_SMALL, 8)
    error = store.quantization_error(spec, net, model.quantize(net), ROUND_SMALL)
    assert 0.0 < error < 0.1
    assert error == pytest.approx(ROUND_ERROR, rel=1e-9)


ROUND_ERROR = 0.018076215158490574


# ------------------------------------------------------------------ Q1153, the rest


def test_a_round_harvest_skips_an_odd_opening_and_takes_every_strideth_round() -> None:
    """One skipped ply skips round 0 (its board is at ply 0 < 1); a stride of 3 then takes
    rounds 1, 4 and 7 of a nine-round game, counted from the first, not from round 0."""
    spec = load_spec(ROUNDS)
    engine = Engine(spec)
    state = engine.initial_state()
    moves: list[int] = []
    for _ in range(9):
        assert engine.terminal(state) is None
        first = engine.legal_moves_for(state, 0)[0]
        second = engine.legal_moves_for(state, 1)[0]
        moves += [first, second]
        state = engine.apply_round(state, first, second)
    procedure = ROUND_SMALL.with_(
        label_nodes=0, result_weight=1.0, skip_plies=1, stride=3
    )
    tdata._setup(spec, procedure)
    try:
        _rows, sides, _targets, indices, plies, results, _values = tdata._harvest_rounds(
            [(5, moves, 0.0, 7)]
        )
    finally:
        tdata._WORKER.clear()
    assert plies == [2, 8, 14]
    assert sides == [0, 0, 0] and indices == [5, 5, 5] and results == [0.0, 0.0, 0.0]


def test_a_scale_of_zero_is_refused_and_a_small_one_is_not() -> None:
    assert SMALL.with_(scale=0.001).scale == 0.001
    for scale in (0.0, -0.5):
        with pytest.raises(
            ValueError, match="^skip_plies, scale and learning_rate must be positive$"
        ):
            SMALL.with_(scale=scale)


class _Stop(Exception):
    pass


def test_training_by_default_plays_on_all_cores_and_harvests_on_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``workers`` 0 is the tournament's "all but one core" for self-play; the harvest,
    which has no such reading, gets at least one."""
    seen: dict[str, int] = {}

    def self_play(_spec: Any, _procedure: Any, workers: int) -> list[Any]:
        seen["self_play"] = workers
        return []

    def harvest(_spec: Any, _records: Any, _procedure: Any, workers: int, report: Any) -> Any:
        seen["harvest"] = workers
        raise _Stop

    monkeypatch.setattr(store, "self_play", self_play)
    monkeypatch.setattr(store, "harvest", harvest)
    with pytest.raises(_Stop):
        store.train_ruleset(load_spec(GARDNER), SMALL, root=tmp_path)
    assert seen == {"self_play": 0, "harvest": 1}


# ------------------------------------------------------------------ Q1153, batch 7


def test_a_dataset_cannot_be_changed_in_place() -> None:
    from dataclasses import FrozenInstanceError

    data = _dataset()
    with pytest.raises(FrozenInstanceError):
        data.games = 3  # type: ignore[misc]
    assert data.games == 2


def test_no_games_are_no_chunks() -> None:
    assert tdata._chunks([], 1) == []
    assert tdata._chunks([], 4) == []


class _CoreDroppingLast:
    def __init__(self, core: Any) -> None:
        self._core = core

    def __getattr__(self, name: str) -> Any:
        return getattr(self._core, name)

    def search_values(self, *args: Any, **kwargs: Any) -> list[Any]:
        return list(self._core.search_values(*args, **kwargs))[:-1]


class _EngineDroppingLast:
    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self.core = _CoreDroppingLast(engine.core)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)


def test_a_teacher_answering_fewer_positions_than_asked_is_refused() -> None:
    spec = load_spec(GARDNER)
    engine = Engine(spec)
    state = engine.initial_state()
    moves: list[int] = []
    for _ in range(4):
        move = engine.legal_moves(state)[0]
        moves.append(move)
        state = engine.apply(state, move)
    procedure = SMALL.with_(
        result_weight=0.5, label_nodes=20, label_search="player", skip_plies=0
    )
    tdata._setup(spec, procedure)
    try:
        found = tdata._harvest_games([(0, moves, 1.0, 7)])
        assert len(found[0]) == 4
        tdata._WORKER["engine"] = _EngineDroppingLast(tdata._WORKER["engine"])
        with pytest.raises(ValueError, match="zip"):
            tdata._harvest_games([(0, moves, 1.0, 7)])
    finally:
        tdata._WORKER.clear()


def test_a_round_label_blends_the_result_and_the_round_value_by_the_weight() -> None:
    spec = load_spec(ROUNDS)
    engine = Engine(spec)
    state = engine.initial_state()
    moves: list[int] = []
    for _ in range(4):
        first = engine.legal_moves_for(state, 0)[0]
        second = engine.legal_moves_for(state, 1)[0]
        moves += [first, second]
        state = engine.apply_round(state, first, second)
    tdata._setup(spec, ROUND_SMALL.with_(result_weight=0.25))
    try:
        _rows, _sides, targets, _i, _plies, results, values = tdata._harvest_rounds(
            [(0, moves, 1.0, 7)]
        )
    finally:
        tdata._WORKER.clear()
    assert results == [1.0, 1.0, 1.0] and len(values) == 3
    for target, value in zip(targets, values, strict=True):
        assert 0.0 < value < 1.0
        assert target == pytest.approx(0.25 + 0.75 * value, abs=1e-12)


def test_w0s_integer_bound_is_i16_less_b0s_over_the_active_count() -> None:
    # 32,767 - 127 = 32,640 = 272 x 120 exactly.
    assert model.w0_int_limit(272) == 120
    assert model.w0_int_limit(273) == 119


def test_the_dense_weights_are_clipped_to_their_i8_limit() -> None:
    net = model.initial(6, SMALL, 3)
    for side in (0, 1):
        net.params[f"w1_{side}"][0, :2] = [5.0, -5.0]
        net.params[f"w2_{side}"][0, :2] = [-5.0, 5.0]
    model._clip(net)
    limit = 127 / 64
    for side in (0, 1):
        assert net.params[f"w1_{side}"][0, :2].tolist() == [limit, -limit]
        assert net.params[f"w2_{side}"][0, :2].tolist() == [-limit, limit]


def test_an_accumulator_above_one_passes_no_gradient() -> None:
    net = model.initial(6, SMALL, 3)
    net.params["w0"][:] = 0.25
    net.params["b0"][:] = 1.0  # every accumulator is 1.25 to 1.75: clipped at one
    index, sides, targets = _batch([0, 1, 0])
    _loss, grads = model._gradients(net, index, sides, targets, 1.0)
    assert grads["w3_0"].any()
    assert not grads["b0"].any() and not grads["w0"].any()


def test_a_hidden_width_of_one_is_allowed() -> None:
    assert SMALL.with_(hidden1=1, hidden2=1).hidden1 == 1


def test_the_acceptance_bar_is_two_errors_over_half() -> None:
    found = store.Acceptance(
        candidate="a", reference="b", score=0.5, error=0.0, games=2,
        identical=0, candidate_nodes=1, reference_nodes=2, accepted=False,
    )
    assert found.as_dict()["bar"] == "score - 2 x error > 0.5"


def test_training_by_default_runs_the_whole_duel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean: None
) -> None:
    from dataclasses import replace

    spec = load_spec(GARDNER)
    n_inputs = int(Engine(spec).core.net_inputs())
    monkeypatch.setattr(store, "quantization_error", lambda *args: 0.0)
    asked: dict[str, Any] = {}
    found = store.Acceptance(
        candidate="a", reference="b", score=0.5, error=0.0, games=2, identical=0,
        candidate_nodes=1, reference_nodes=1, accepted=False,
    )

    def accept(*args: Any, **kwargs: Any) -> Any:
        asked.update(kwargs)
        return found

    monkeypatch.setattr(store, "accept", accept)
    store.train_ruleset(
        spec, SMALL, root=tmp_path, workers=1, data=replace(_dataset(), n_inputs=n_inputs)
    )
    assert asked["curtail"] is False


# ------------------------------------------------------------------ Q1153, batch 8


def test_no_opening_is_zero_plies_whatever_the_seed() -> None:
    assert [tdata.opening_plies_for(seed, 0) for seed in range(6)] == [0] * 6
    assert [tdata.opening_plies_for(seed, -3) for seed in range(6)] == [0] * 6


def test_a_replay_plays_by_default_on_all_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    """``workers`` 0 is the tournament's "all but one core"; a replay is no exception."""
    seen: list[int] = []
    monkeypatch.setattr(tdata, "replay_bytes", lambda spec, procedure: None)
    monkeypatch.setattr(
        tdata, "_replay_link_games", lambda spec, replay, workers: seen.append(workers) or []
    )
    monkeypatch.setattr(
        tdata, "new_games", lambda spec, procedure, workers: seen.append(workers) or []
    )
    procedure = SimpleNamespace(replay=SimpleNamespace(links=("one back",)))
    assert tdata.replay_games({}, procedure) == []  # type: ignore[arg-type]
    assert tdata.link_games({}, procedure, 0) == []  # type: ignore[arg-type]
    assert tdata.link_games({}, procedure, 1) == []  # type: ignore[arg-type]
    assert seen == [0, 0, 0]


def test_a_resumed_campaign_harvest_loads_every_sealed_piece(tmp_path: Path) -> None:
    spec = load_spec(GARDNER)
    engine = Engine(spec)
    state = engine.initial_state()
    moves: list[int] = []
    for _ in range(6):
        move = engine.legal_moves(state)[0]
        moves.append(move)
        state = engine.apply(state, move)
    records = [_record(i, moves) for i in range(3)]
    with checkpoint.active(tmp_path / "campaign", build="pins"):
        first = tdata.harvest(spec, records, SMALL, workers=1)
    lines: list[str] = []
    with checkpoint.active(tmp_path / "campaign", build="pins"):
        again = tdata.harvest(spec, records, SMALL, workers=1, report=lines.append)
    # Every piece is on disk: one report, of the whole, and nothing played again.
    assert lines == ["  harvested 3/3 games"]
    assert first.positions == again.positions == 12
    assert again.ply.tolist() == first.ply.tolist() == [2, 3, 4, 5] * 3


def test_an_accumulator_at_exactly_one_passes_no_gradient() -> None:
    net = model.initial(6, SMALL, 3)
    net.params["w0"][:] = 0.0
    net.params["b0"][:] = 1.0  # every accumulator is exactly 1.0: the top of its clip
    index, sides, targets = _batch([0, 1, 0])
    _loss, grads = model._gradients(net, index, sides, targets, 1.0)
    assert grads["w3_0"].any()
    assert not grads["b0"].any() and not grads["w0"].any()


def test_the_held_out_games_are_played_on_one_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = load_spec(GARDNER)
    n_inputs = int(Engine(spec).core.net_inputs())
    net = model.initial(n_inputs, SMALL, 8)
    seen: list[int] = []

    def no_games(spec: Any, procedure: Procedure, workers: int = 0) -> list[Any]:
        seen.append(workers)
        return []

    monkeypatch.setattr(store, "self_play", no_games)
    store.quantization_error(spec, net, model.quantize(net), SMALL)
    assert seen == [1]


def test_a_duel_without_an_opening_is_the_runners_and_never_opens_below_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chess_variant_sandbox.tournament.checkpoint as played
    import chess_variant_sandbox.tournament.duel as duel

    def run_tournament(spec: Any, config: Any) -> Any:
        candidate, reference = config.agents
        return SimpleNamespace(records=[
            SimpleNamespace(agents=(candidate, reference), nodes=(1, 2), moves=[], seed=0),
        ])

    def opened(*args: Any) -> list[Any]:
        raise AssertionError("a duel asked for no opening was opened")

    monkeypatch.setattr(played, "run_tournament", run_tournament)
    monkeypatch.setattr(store, "_opening_duel", opened)
    monkeypatch.setattr(
        duel,
        "grade",
        lambda records, candidate, reference: SimpleNamespace(
            score=0.5, error=0.0, games=1, identical=0, pairs=[0.5]
        ),
    )
    for plies in (0, -2):
        found = store.accept(load_spec(GARDNER), SMALL, pairs=1, workers=1, opening_plies=plies)
        assert (found.opening_plies, found.openings) == (0, 1), plies
        assert (found.candidate_nodes, found.reference_nodes) == (1, 2)


# ------------------------------------------------------------------ Q1153 (queue training-9)


class _NoPool:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("a pool was started")


def test_a_single_game_is_harvested_without_a_pool_whatever_the_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_spec(GARDNER)
    monkeypatch.setattr(tdata, "ProcessPoolExecutor", _NoPool)
    lines: list[str] = []
    found = tdata.harvest(spec, [_record(0)], SMALL, workers=4, report=lines.append)
    assert (found.games, found.positions, lines) == (1, 0, [])


def test_a_hidden_unit_between_one_and_two_passes_no_gradient() -> None:
    net = model.initial(6, SMALL, 3)
    net.params["w1_0"][:] = 0.0
    net.params["b1_0"][:] = 1.5  # every first hidden unit is 1.5: clipped at one
    net.params["w2_0"][:] = 0.1
    net.params["b2_0"][:] = 0.2  # the second layer is 0.5: open, so it passes a gradient back
    index, sides, targets = _batch([0, 0, 0])
    _loss, grads = model._gradients(net, index, sides, targets, 1.0)
    assert grads["w2_0"].any() and grads["b2_0"].any()
    assert not grads["w1_0"].any() and not grads["b1_0"].any()


def _stub_duel(monkeypatch: pytest.MonkeyPatch, records: Any) -> list[int]:
    import chess_variant_sandbox.tournament.checkpoint as played
    import chess_variant_sandbox.tournament.duel as duel

    opened: list[int] = []

    def run_tournament(spec: Any, config: Any) -> Any:
        raise AssertionError("an opened duel ran the plain runner")

    def opening_duel(spec: Any, config: Any, most: int) -> list[Any]:
        opened.append(most)
        return records(config.agents)

    monkeypatch.setattr(played, "run_tournament", run_tournament)
    monkeypatch.setattr(store, "_opening_duel", opening_duel)
    monkeypatch.setattr(
        duel,
        "grade",
        lambda records, candidate, reference: SimpleNamespace(
            score=0.5, error=0.0, games=2, identical=0, pairs=[0.5]
        ),
    )
    return opened


def test_a_duel_of_one_opening_ply_is_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = _stub_duel(
        monkeypatch,
        lambda agents: [
            SimpleNamespace(agents=agents, nodes=(1, 2), moves=[3, 4], seed=4),
            SimpleNamespace(agents=agents[::-1], nodes=(2, 1), moves=[3, 5], seed=4),
        ],
    )
    found = store.accept(load_spec(GARDNER), SMALL, pairs=1, workers=1, opening_plies=1)
    assert opened == [1]
    assert (found.opening_plies, found.openings) == (1, 1)


def test_a_round_duels_openings_are_counted_in_whole_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Seed 11 opens with 4 of at most 8 plies: two rounds. The two games share their first
    # round and differ in the second, so they are two openings; seed 6 (5 plies) is two
    # rounds as well, its odd ply not part of one.
    def records(agents: Any) -> list[Any]:
        return [
            SimpleNamespace(agents=agents, nodes=(1, 2), moves=[1, 2, 3, 4, 9], seed=11),
            SimpleNamespace(agents=agents[::-1], nodes=(2, 1), moves=[1, 2, 7, 8, 9], seed=11),
            SimpleNamespace(agents=agents, nodes=(1, 2), moves=[1, 2, 3, 4, 5], seed=6),
            SimpleNamespace(agents=agents[::-1], nodes=(2, 1), moves=[1, 2, 7, 8, 6], seed=6),
        ]

    opened = _stub_duel(monkeypatch, records)
    found = store.accept(load_spec(ROUNDS), ROUND_SMALL, pairs=2, workers=1, opening_plies=8)
    assert opened == [8]
    assert found.candidate.startswith("matrix:")
    assert found.openings == 2


def test_the_output_weight_is_clipped_to_its_i8_limit() -> None:
    net = model.initial(6, SMALL, 3)
    for side in (0, 1):
        net.params[f"w3_{side}"].flat[:2] = [100.0, -100.0]
    model._clip(net)
    limit = 127 * 127 / 1024  # 15.751953125
    for side in (0, 1):
        assert net.params[f"w3_{side}"].flat[:2].tolist() == [limit, -limit]


def test_arms_default_to_one_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[tuple[str, int]] = []

    def self_play(spec: Any, procedure: Any, workers: int) -> list[Any]:
        asked.append(("self_play", workers))
        return []

    def harvest(spec: Any, records: Any, procedure: Any, workers: int, report: Any) -> Any:
        asked.append(("harvest", workers))
        return "dataset"

    monkeypatch.setattr(tdata, "self_play", self_play)
    monkeypatch.setattr(tdata, "harvest", harvest)
    assert tdata.harvest_arms({}, [SMALL]) == ["dataset"]
    assert asked == [("self_play", 1), ("harvest", 1)]


def _acceptance(**fields: Any) -> store.Acceptance:
    base: dict[str, Any] = dict(
        candidate="a", reference="b", score=0.5, error=0.0, games=5,
        identical=0, candidate_nodes=1, reference_nodes=2, accepted=False,
    )
    base.update(fields)
    return store.Acceptance(**base)


def test_an_acceptance_counts_one_opening_and_whole_pairs_by_default() -> None:
    found = _acceptance()
    assert found.openings == 1
    assert found.decided_after == 2
    assert found.as_dict()["decided_after"] == 2
    assert repr(found.as_dict()["planned_pairs"]) == "2"


def test_the_acceptance_bar_subtracts_two_errors() -> None:
    # [0, 1]: score 0.5, error 0.5; [0.6, 0.7]: score 0.65, error 0.05.
    assert store.accepts([0.0, 1.0]) is False
    assert store.accepts([0.6, 0.7]) is True
    assert store.accepts([0.5, 0.5]) is False


def test_a_hidden_unit_at_exactly_zero_passes_no_gradient() -> None:
    net = model.initial(6, SMALL, 3)
    net.params["w1_0"][:] = 0.0
    net.params["b1_0"][:] = 0.0  # every first hidden unit is exactly 0.0: the bottom of its clip
    net.params["b2_0"][:] = 0.5  # the second layer is open, so it passes a gradient back
    index, sides, targets = _batch([0, 0, 0])
    _loss, grads = model._gradients(net, index, sides, targets, 1.0)
    assert grads["b2_0"].any()
    assert not grads["w1_0"].any() and not grads["b1_0"].any()


def test_a_hidden_unit_at_exactly_one_passes_no_gradient() -> None:
    net = model.initial(6, SMALL, 3)
    net.params["w1_0"][:] = 0.0
    net.params["b1_0"][:] = 1.0  # every first hidden unit is exactly 1.0: the top of its clip
    net.params["w2_0"][:] = 0.1
    net.params["b2_0"][:] = 0.2  # the second layer is 0.5: open
    index, sides, targets = _batch([0, 0, 0])
    _loss, grads = model._gradients(net, index, sides, targets, 1.0)
    assert grads["b2_0"].any()
    assert not grads["w1_0"].any() and not grads["b1_0"].any()


def test_a_second_hidden_unit_at_exactly_zero_passes_no_gradient() -> None:
    net = model.initial(6, SMALL, 3)
    net.params["w2_0"][:] = 0.0
    net.params["b2_0"][:] = 0.0  # every second hidden unit is exactly 0.0
    # Four positions, so the errors do not cancel (the first three's sum to zero at 0.5).
    index, sides, targets = _batch([0, 0, 0, 0])
    _loss, grads = model._gradients(net, index, sides, targets, 1.0)
    assert grads["b3_0"].any()
    assert not grads["b2_0"].any() and not grads["w2_0"].any() and not grads["w1_0"].any()


def test_result_weight_above_one_is_refused() -> None:
    assert SMALL.with_(result_weight=1.0).result_weight == 1.0
    with pytest.raises(ValueError, match=r"^result_weight 1.5 is outside \[0, 1\]$"):
        SMALL.with_(result_weight=1.5)


def test_a_negative_td_lambda_is_refused() -> None:
    searched = SMALL.with_(result_weight=0.5, label_nodes=40)
    assert searched.with_(td_lambda=0.0).td_lambda == 0.0
    with pytest.raises(ValueError, match=r"^td_lambda -0.5 is outside \[0, 1\]$"):
        searched.with_(td_lambda=-0.5)


def test_an_alternating_harvest_takes_every_strideth_ply_counted_from_the_skip() -> None:
    """One skipped ply and a stride of three take plies 1, 4 and 7 of a nine-ply game."""
    spec = load_spec(GARDNER)
    engine = Engine(spec)
    state = engine.initial_state()
    moves: list[int] = []
    for _ in range(9):
        assert engine.terminal(state) is None
        move = engine.legal_moves(state)[0]
        moves.append(move)
        state = engine.apply(state, move)
    tdata._setup(spec, SMALL.with_(skip_plies=1, stride=3))
    try:
        _rows, sides, _targets, indices, plies, results, _values = tdata._harvest_games(
            [(5, moves, 1.0, 7)]
        )
    finally:
        tdata._WORKER.clear()
    assert plies == [1, 4, 7]
    assert sides == [1, 0, 1] and indices == [5, 5, 5] and results == [0.0, 1.0, 0.0]


def _branches(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Which way a schedule is played: ("tournament",) whole, or ("opened", most) per task."""
    taken: list[Any] = []
    monkeypatch.setattr(
        checkpoint, "run_tournament",
        lambda spec, config: taken.append(("tournament",)) or SimpleNamespace(records=[]),
    )
    monkeypatch.setattr(
        tdata, "_play_tasks",
        lambda spec, config, tasks, most: taken.append(("opened", most)) or [],
    )
    return taken


def test_new_games_of_one_opening_ply_are_opened_and_of_none_are_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    taken = _branches(monkeypatch)
    spec = load_spec(GARDNER)
    for plies in (0, 1, 2):
        tdata.new_games(spec, SMALL.with_(opening_plies=plies), workers=1)
    assert taken == [("tournament",), ("opened", 1), ("opened", 2)]


def test_a_replay_link_of_one_opening_ply_is_opened_and_of_none_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    taken = _branches(monkeypatch)
    monkeypatch.setattr(tdata, "_checked_bytes", lambda *args: b"")
    net = Parent(procedure="c00000000003", weights_sha256="ab" * 32)
    for plies in (0, 1, 2):
        replay = Replay(
            agents=("minimax:net=c00000000003,nodes=10", "greedy"), games=2, seed=9,
            opening_plies=plies, net=net,
        )
        tdata._replay_link_games(load_spec(GARDNER), replay, 1)
    assert taken == [("tournament",), ("opened", 1), ("opened", 2)]


class _RecordedPool(ThreadPoolExecutor):
    """A process pool run on one thread of this process; records that it was asked for."""

    made: list[int] = []

    def __init__(self, max_workers: int, mp_context: Any, initializer: Any, initargs: Any):
        super().__init__(max_workers=1)
        initializer(*initargs)
        _RecordedPool.made.append(max_workers)


def test_a_two_worker_campaign_harvest_is_a_process_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = load_spec(GARDNER)
    _RecordedPool.made = []
    monkeypatch.setattr(tdata, "ProcessPoolExecutor", _RecordedPool)
    with checkpoint.active(tmp_path / "two", build="pins"):
        found = tdata.harvest(spec, [_record(i) for i in range(3)], SMALL, workers=2)
    assert found.games == 3
    assert _RecordedPool.made == [2]
    with checkpoint.active(tmp_path / "one", build="pins"):
        tdata.harvest(spec, [_record(i) for i in range(3)], SMALL, workers=1)
    assert _RecordedPool.made == [2]


# ------------------------------------------------------------------ Q1153 batch 13


def test_a_dataset_whose_last_offset_misses_its_inputs_is_refused_and_an_empty_one_is_not() -> None:
    with pytest.raises(ValueError, match="last offset is not the number of inputs"):
        Dataset(
            n_inputs=4,
            offsets=np.array([0, 2], dtype=np.int64),
            inputs=np.array([0, 1, 2], dtype=np.int32),
            sides=np.array([0], dtype=np.int8),
            targets=np.array([0.5], dtype=np.float32),
            games=1,
        )
    # No sample: the offsets' one zero is not checked against stray inputs.
    empty = Dataset(
        n_inputs=4,
        offsets=np.array([0], dtype=np.int64),
        inputs=np.array([3], dtype=np.int32),
        sides=np.array([], dtype=np.int8),
        targets=np.array([], dtype=np.float32),
        games=0,
    )
    assert empty.positions == 0


def test_a_dataset_is_saved_into_a_directory_it_makes(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "er" / "set.npz"
    saved = _dataset().save(target)
    assert saved.exists() and saved.parent == tmp_path / "deep" / "er"


def test_a_harvest_by_default_starts_no_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = load_spec(GARDNER)
    monkeypatch.setattr(tdata, "ProcessPoolExecutor", _NoPool)
    found = tdata.harvest(spec, [_record(i) for i in range(3)], SMALL)
    assert (found.games, found.positions) == (3, 0)


def test_eighteen_pooled_chunks_report_every_one(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = load_spec(GARDNER)
    monkeypatch.setattr(tdata, "ProcessPoolExecutor", _InlinePool)
    lines: list[str] = []
    tdata.harvest(spec, [_record(i) for i in range(18)], SMALL, workers=3, report=lines.append)
    assert lines == [f"  harvested {i}/18 chunks, 0 positions" for i in range(1, 19)]


def test_padding_no_rows_is_an_empty_rectangle_one_wide() -> None:
    offsets = np.array([0, 2, 3], dtype=np.int64)
    inputs = np.array([4, 5, 6], dtype=np.int32)
    out = model._padded(offsets, inputs, np.array([], dtype=np.int64), 9)
    assert out.shape == (0, 1)


def test_a_learning_rate_of_zero_is_refused_and_a_small_one_is_not() -> None:
    assert SMALL.with_(learning_rate=0.0001).learning_rate == 0.0001
    for rate in (0.0, -0.5):
        with pytest.raises(
            ValueError, match="^skip_plies, scale and learning_rate must be positive$"
        ):
            SMALL.with_(learning_rate=rate)


def test_a_store_is_written_into_a_root_it_makes(tmp_path: Path) -> None:
    blob = model.quantize(model.initial(6, SMALL, 3))
    record = NetRecord("abcabcabcabc", "0123", "unchecked", blob)
    root = tmp_path / "new" / "store"
    meta = store.write_store(record, {"a": 1}, root)
    assert meta == root / "0123-abcabcabcabc.json"
    assert (root / "0123-abcabcabcabc.net").read_bytes() == blob


def test_an_acceptance_duel_collects_no_series(monkeypatch: pytest.MonkeyPatch) -> None:
    import chess_variant_sandbox.tournament.duel as duel

    seen: list[Any] = []

    def games(spec: Any, config: Any, most: int) -> list[Any]:
        seen.append(config)
        candidate, reference = config.agents
        return [SimpleNamespace(agents=(candidate, reference), nodes=(1, 2), moves=[1], seed=5)]

    monkeypatch.setattr(store, "_opening_duel", games)
    monkeypatch.setattr(
        duel,
        "grade",
        lambda records, candidate, reference: SimpleNamespace(
            score=0.5, error=0.0, games=1, identical=0, pairs=[0.5]
        ),
    )
    store.accept(load_spec(GARDNER), SMALL, pairs=1, workers=1, opening_plies=2)
    assert [(c.collect_series, c.include_self_play, c.paired) for c in seen] == [(False, False, True)]


def test_nine_games_on_one_worker_are_two_chunks_of_eight_a_worker() -> None:
    # ceil(9 / 8) = 2 games a chunk: a ninth game splits the eight-chunk cut.
    nine = [(i, [], 0.5, i) for i in range(9)]
    cut = tdata._chunks(nine, 1)  # type: ignore[arg-type]
    assert [[g[0] for g in chunk] for chunk in cut] == [[0, 1], [2, 3], [4, 5], [6, 7], [8]]
    # Seventeen on one worker: three a chunk.
    seventeen = [(i, [], 0.5, i) for i in range(17)]
    assert [len(c) for c in tdata._chunks(seventeen, 1)] == [3, 3, 3, 3, 3, 2]  # type: ignore[arg-type]


@pytest.mark.parametrize("most_active", [0, 1])
def test_one_or_no_active_input_spreads_w0_half_each_way(most_active: int) -> None:
    w0 = model.initial(40, SMALL, most_active).params["w0"]
    assert np.isfinite(w0).all()
    top = float(np.abs(w0).max())
    # 0.5 / sqrt(1); two active rows would be 0.5 / sqrt(2) = 0.354.
    assert 0.36 < top <= 0.5


def test_a_replay_sets_negative_opening_is_refused_and_none_is_not() -> None:
    net = Parent(procedure="c00000000003", weights_sha256="ab" * 32)
    agents = ("minimax:net=c00000000003,nodes=10", "greedy")
    assert Replay(agents=agents, games=1, seed=9, opening_plies=0, net=net).opening_plies == 0
    with pytest.raises(ValueError, match="^a replay set's opening_plies must be 0 or more$"):
        Replay(agents=agents, games=1, seed=9, opening_plies=-1, net=net)


def test_an_acceptance_duel_by_default_plays_on_all_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    import chess_variant_sandbox.tournament.duel as duel

    seen: list[Any] = []

    def games(spec: Any, config: Any, most: int) -> list[Any]:
        seen.append(config)
        candidate, reference = config.agents
        return [SimpleNamespace(agents=(candidate, reference), nodes=(1, 2), moves=[1], seed=5)]

    monkeypatch.setattr(store, "_opening_duel", games)
    monkeypatch.setattr(
        duel,
        "grade",
        lambda records, candidate, reference: SimpleNamespace(
            score=0.5, error=0.0, games=1, identical=0, pairs=[0.5]
        ),
    )
    store.accept(load_spec(GARDNER), SMALL, pairs=1, opening_plies=2)
    assert [c.workers for c in seen] == [0]


# ------------------------------------------------------------------ Q1153 batch training-15


def test_self_play_by_default_plays_new_and_replayed_games_on_all_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``workers`` 0 is the tournament's "all but one core"; self-play's own default too."""
    seen: list[tuple[str, int]] = []
    monkeypatch.setattr(tdata, "parent_bytes", lambda spec, procedure: b"")
    monkeypatch.setattr(tdata, "replay_bytes", lambda spec, procedure: b"")
    monkeypatch.setattr(
        tdata, "new_games", lambda spec, procedure, workers: seen.append(("new", workers)) or [1]
    )
    monkeypatch.setattr(
        tdata,
        "replay_games",
        lambda spec, procedure, workers: seen.append(("replay", workers)) or [2],
    )
    procedure = SimpleNamespace(replay=SimpleNamespace())
    assert tdata.self_play({}, procedure) == [1, 2]  # type: ignore[arg-type]
    assert seen == [("new", 0), ("replay", 0)]


def test_an_accumulator_at_exactly_zero_passes_no_gradient() -> None:
    net = model.initial(6, SMALL, 3)
    net.params["w0"][:] = 0.0
    net.params["b0"][:] = 0.0  # every accumulator is exactly 0.0: the bottom of its clip
    net.params["w1_0"][:] = 0.3
    net.params["b1_0"][:] = 0.5  # the first hidden layer is 0.5: open
    net.params["w2_0"][:] = 0.3
    net.params["b2_0"][:] = 0.2  # the second is 0.65: open
    net.params["w3_0"][:] = 0.5
    index, sides, targets = _batch([0, 0, 0])
    _loss, grads = model._gradients(net, index, sides, targets, 1.0)
    assert grads["b1_0"].any()  # a gradient reaches the accumulator...
    assert not grads["b0"].any() and not grads["w0"].any()  # ...and stops there


def test_a_round_game_with_a_parent_alone_is_refused() -> None:
    parent = Parent(procedure="c00000000003", weights_sha256="ab" * 32)
    assert ROUND_SMALL.parent is None and ROUND_SMALL.replay is None
    with pytest.raises(ValueError, match="^a round game's generations are not built"):
        ROUND_SMALL.with_(parent=parent)


def test_an_acceptance_cannot_be_changed_in_place() -> None:
    from dataclasses import FrozenInstanceError

    found = store.Acceptance(
        candidate="a", reference="b", score=0.5, error=0.1, games=2,
        identical=0, candidate_nodes=1, reference_nodes=2, accepted=False,
    )
    with pytest.raises(FrozenInstanceError):
        found.accepted = True  # type: ignore[misc]
    assert found.accepted is False


def test_a_duel_of_one_opening_ply_counts_its_distinct_first_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Seeds 4 and 7 each open with 1 ply of at most 1 (OPENINGS_1); their first moves differ.
    opened = _stub_duel(
        monkeypatch,
        lambda agents: [
            SimpleNamespace(agents=agents, nodes=(1, 2), moves=[3, 4], seed=4),
            SimpleNamespace(agents=agents[::-1], nodes=(2, 1), moves=[3, 6], seed=4),
            SimpleNamespace(agents=agents, nodes=(1, 2), moves=[5, 4], seed=7),
            SimpleNamespace(agents=agents[::-1], nodes=(2, 1), moves=[5, 6], seed=7),
        ],
    )
    found = store.accept(load_spec(GARDNER), SMALL, pairs=2, workers=1, opening_plies=1)
    assert opened == [1]
    assert (found.opening_plies, found.openings) == (1, 2)


def test_a_one_input_nets_padding_reads_no_row() -> None:
    # One input row: the padding slot (index 1 == n_inputs) must be masked, never looked up.
    f32 = np.float32
    params = {"w0": np.array([[0.25, 0.5]], dtype=f32), "b0": np.array([0.0, 0.25], dtype=f32)}
    for side in (0, 1):
        params[f"w1_{side}"] = np.array([[1.0, 1.0]], dtype=f32)
        params[f"b1_{side}"] = np.zeros(1, dtype=f32)
        params[f"w2_{side}"] = np.array([[0.5]], dtype=f32)
        params[f"b2_{side}"] = np.zeros(1, dtype=f32)
        params[f"w3_{side}"] = np.array([2.0], dtype=f32)
        params[f"b3_{side}"] = np.array([0.125], dtype=f32)
    net = model.FloatNet(params=params, w0_limit=127)
    found = model._forward(net, np.array([[0, 1]], dtype=np.int64), np.array([0], dtype=np.int8))
    assert found.x0.tolist() == [[0.25, 0.75]]
    assert found.y.tolist() == [1.125]
