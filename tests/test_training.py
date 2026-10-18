"""The learned evaluator's trainer and stores (Q502f): data through the core, a net out.

What is held here, on tiny budgets:

- self-play with a random opening (Q637) is the core's moves, a pure function of the
  schedule's seeds at any thread count, and at `opening_plies` 0 it is the runner's games;
- the data is the core's: its input lists are `Core.net_line`'s, equal to the layout's
  oracle; the labels are the game's results (and the probe's, squashed), for the side to
  move; and the dataset does not depend on how many processes harvested it;
- the TD(λ) label (Q658) is a pure function of each game's saved line: λ 0 is the probe
  (and the harvest's label bit for bit), λ 1 the result, λ 0.5 a hand computation, and
  `td_lambda` enters a procedure's name only when non-zero;
- the trainer's gradients are the loss's derivative, and quantizing is rounding: the
  core's integer output is the float net's within the quantization step;
- the same data gives the same store, byte for byte;
- `train` runs end to end, and what it stores loads as `minimax:net=...` and plays in the
  core once accepted, refusing to play before (Q709: never the built-in objective under
  the net's name);
- the basis stamps which net played (`Basis.net`), and a roster naming no net stamps
  nothing, so no library stamp moves.

Design: `docs/reference/learned-evaluator.md` §8.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from chess_variant_sandbox import measurement  # noqa: E402
from chess_variant_sandbox.agents import nets  # noqa: E402
from chess_variant_sandbox.agents.nets import (  # noqa: E402
    ACCEPTED,
    REFUSED,
    Layout,
    NetRecord,
    load_net,
    register_net,
    store_paths,
    unregister_net,
)
from chess_variant_sandbox.agents.training import model  # noqa: E402
from chess_variant_sandbox.agents.training.data import (  # noqa: E402
    Dataset,
    harvest,
    labels,
    opening_plies_for,
    relabel,
    self_play,
    squash,
    td_labels,
)
from chess_variant_sandbox.agents.training.procedure import (  # noqa: E402
    DEFAULT,
    Q502F2,
    Q637,
    TRAINER_VERSION,
    Parent,
    Procedure,
)
from chess_variant_sandbox.agents.training.store import (  # noqa: E402
    UNCHECKED,
    _opening_duel,
    accept,
    train_ruleset,
)
from chess_variant_sandbox.engine import Engine  # noqa: E402
from chess_variant_sandbox.spec import load_spec  # noqa: E402
from chess_variant_sandbox.tournament.play import play_game  # noqa: E402
from chess_variant_sandbox.tournament.runner import TournamentConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GARDNER = ROOT / "rulesets" / "gardner.json"

# The default recipe (Q671's: Q637's at TD(lambda 0.7), the first whose net was accepted),
# the default before it (Q637's: diverse self-play, the probe's label alone) and the first
# one (Q502f2's); their digests are the `net=` names the stores carry, so a change to a
# recipe, the layout or the trainer must be seen.
DEFAULT_DIGEST = "db5ef1bc938e"
Q637_DIGEST = "1d160257b8e5"
Q502F2_DIGEST = "1c193cc56c76"
# The accepted net (archive/plan-teacher §8.6.3-§8.6.4): Intransitive's, under the default's name.
ACCEPTED_NET = ROOT / "nets" / "d93e28d09a33-db5ef1bc938e.json"

TINY = Procedure(
    agents=("greedy",),
    games=3,
    skip_plies=2,
    result_weight=0.5,
    label_nodes=40,
    label_depth=3,
    td_lambda=0.0,
    accumulator=16,
    hidden1=8,
    hidden2=4,
    epochs=2,
    batch=32,
)


@pytest.fixture
def spec() -> dict:
    return load_spec(GARDNER)


@pytest.fixture
def clean() -> Iterator[None]:
    """Forget every net this test registered, so no other test meets one."""
    before = dict(nets._NETS)
    yield
    nets._NETS.clear()
    nets._NETS.update(before)


def _data(spec: dict, procedure: Procedure = TINY) -> Dataset:
    return harvest(spec, self_play(spec, procedure, workers=1), procedure)


# ------------------------------------------------------------------ the procedure


def test_the_default_procedure_is_pinned() -> None:
    assert DEFAULT.digest == DEFAULT_DIGEST, (
        "the default training procedure's digest moved; every stored net under the old "
        "name is now another procedure's. If that was the point, bump TRAINER_VERSION "
        "or the recipe on purpose and pin the new digest here"
    )
    assert Q637.digest == Q637_DIGEST, "Q637's stores no longer name their recipe"
    assert Q502F2.digest == Q502F2_DIGEST, "Q502f2's stores no longer name their recipe"
    assert TRAINER_VERSION == 1
    assert DEFAULT.result_weight == 0.0 and DEFAULT.opening_plies > 0
    # Q671: the default is Q637's recipe at TD(lambda 0.7), and nothing else.
    assert DEFAULT.td_lambda == 0.7 and Q637.with_(td_lambda=0.7) == DEFAULT
    assert DEFAULT.with_(td_lambda=0.0) == Q637


def test_the_accepted_net_is_the_default_procedures() -> None:
    """`nets/d93e28d09a33-db5ef1bc938e.json`, the first accepted net, is the default's."""
    payload = json.loads(ACCEPTED_NET.read_text(encoding="utf-8"))
    assert payload["procedure"] == DEFAULT.digest == DEFAULT_DIGEST
    assert payload["verdict"] == ACCEPTED
    assert payload["recipe"] == DEFAULT.as_dict()
    assert Procedure.from_dict(payload["recipe"]) == DEFAULT


def test_every_stored_net_still_names_its_recipe() -> None:
    """A change to the defaults must not rename a recipe a store already carries."""
    stores = sorted((ROOT / "nets").glob("*.json"))
    assert stores
    for path in stores:
        payload = json.loads(path.read_text(encoding="utf-8"))
        recipe = Procedure.from_dict(payload["recipe"])
        assert recipe.digest == payload["procedure"] == path.stem.split("-")[1], path.name


def test_a_recipe_written_before_the_random_opening_reads_back_without_one() -> None:
    old = Q502F2.as_dict()
    assert "opening_plies" not in old
    assert Procedure.from_dict(old) == Q502F2


def test_every_field_of_a_procedure_is_in_its_name() -> None:
    changed = {
        "agents": ("greedy",),
        "games": 7,
        "seed": 3,
        "opening_plies": 3,
        "skip_plies": 5,
        "stride": 2,
        "result_weight": 0.25,
        "label_nodes": 99,
        "label_depth": 4,
        "td_lambda": 0.5,
        "label_search": "player",
        "scale": 2.0,
        "accumulator": 64,
        "hidden1": 16,
        "hidden2": 8,
        "epochs": 3,
        "batch": 256,
        "learning_rate": 0.01,
        "init_seed": 9,
        "parent": Parent("db5ef1bc938e", "0" * 64),
    }
    assert set(changed) == set(DEFAULT.as_dict()) | {"td_lambda", "label_search", "parent"}
    names = {DEFAULT.with_(**{key: value}).digest for key, value in changed.items()}
    assert len(names) == len(changed) and DEFAULT.digest not in names
    assert Procedure.from_dict(DEFAULT.as_dict()) == DEFAULT


def test_td_lambda_enters_the_name_only_when_selected() -> None:
    """Q658: at 0 the field is absent, so every recipe named before it keeps its name."""
    assert "td_lambda" not in Q637.as_dict() and "td_lambda" not in Q502F2.as_dict()
    assert Q637.with_(td_lambda=0.0).digest == Q637_DIGEST
    assert Q502F2.with_(td_lambda=0.0).digest == Q502F2_DIGEST
    td = Q637.with_(td_lambda=0.7)
    assert td.as_dict()["td_lambda"] == 0.7 and td.digest == DEFAULT_DIGEST != Q637_DIGEST
    assert Procedure.from_dict(td.as_dict()) == td
    # A recipe without the key was written at 0, never today's default.
    assert Procedure.from_dict(Q637.as_dict()).td_lambda == 0.0
    with pytest.raises(ValueError, match="td_lambda"):
        DEFAULT.with_(td_lambda=1.5)
    with pytest.raises(ValueError, match="td_lambda"):
        DEFAULT.with_(result_weight=1.0, td_lambda=0.5)
    with pytest.raises(ValueError, match="td_lambda"):
        DEFAULT.with_(result_weight=1.0)
    assert DEFAULT.with_(result_weight=1.0, td_lambda=0.0).td_lambda == 0.0


def test_a_procedure_the_core_cannot_play_is_refused() -> None:
    with pytest.raises(ValueError, match="accumulator"):
        DEFAULT.with_(accumulator=4096)
    with pytest.raises(ValueError, match="generation"):
        DEFAULT.with_(agents=("minimax:net=abcdefabcdef,nodes=400",))
    with pytest.raises(ValueError, match="label_nodes"):
        DEFAULT.with_(label_nodes=0)


# ------------------------------------------------------------------ the data


def test_the_data_is_the_cores_inputs_and_the_games_results(spec: dict) -> None:
    """Every sample's inputs are the layout's; results-only labels are the game's result."""
    procedure = TINY.with_(result_weight=1.0)
    engine = Engine(spec)
    records = self_play(spec, procedure, workers=1)
    data = harvest(spec, records, procedure)
    layout = Layout.of(engine.spec)
    assert data.n_inputs == layout.n_inputs == engine.core.net_inputs()
    assert data.games == len(records)

    at = 0
    for record in records:
        state = engine.initial_state()
        for ply, move in enumerate(record.moves):
            if ply >= procedure.skip_plies:
                row = data.inputs[data.offsets[at] : data.offsets[at + 1]].tolist()
                assert row == layout.features(state)
                side = state.side_to_move
                assert data.sides[at] == side
                assert data.targets[at] == pytest.approx(record.score_for(side))
                at += 1
            state = engine.apply(state, move)
    assert at == data.positions > 0


def test_a_searched_label_is_the_probes_score_squashed(spec: dict) -> None:
    data = _data(spec)
    assert data.positions > 0
    assert np.all((data.targets >= 0.0) & (data.targets <= 1.0))
    # Half the result, so a label that is not 0, 0.5 or 1 shows the probe spoke.
    assert not np.all(np.isin(data.targets, [0.0, 0.25, 0.5, 0.75, 1.0]))
    assert squash(0.0, 1.0) == 0.5
    assert squash(1e6, 1.0) == 1.0 and squash(-1e6, 1.0) == 0.0


def test_a_dataset_round_trips_and_is_named_by_its_content(spec: dict, tmp_path: Path) -> None:
    data = _data(spec)
    again = Dataset.load(data.save(tmp_path / "d.npz"))
    assert again.digest == data.digest
    changed = Dataset(
        data.n_inputs, data.offsets, data.inputs, data.sides, data.targets * 0.5, data.games
    )
    assert changed.digest != data.digest
    # The line round-trips too, and data saved before it existed still loads.
    assert again.has_line
    for column in ("game", "ply", "results", "values"):
        assert np.array_equal(getattr(again, column), getattr(data, column), equal_nan=True)
    bare = Dataset(data.n_inputs, data.offsets, data.inputs, data.sides, data.targets, data.games)
    old = Dataset.load(bare.save(tmp_path / "old.npz"))
    assert not old.has_line and old.digest == data.digest


# ------------------------------------------------------------------ TD(λ) (Q658)

# Three tiny games, by hand. Game 0: four samples, the side to move alternating, seat 0
# won (so the side to move's result is 1, 0, 1, 0). Game 1: one sample, a draw. Game 2:
# one side moves twice running (no flip between its samples) and lost; its second
# sample's probe gave nothing (NaN), so its value is its result.
SIDES = np.array([0, 1, 0, 1, 1, 0, 0], dtype=np.int8)
GAME = np.array([0, 0, 0, 0, 1, 2, 2], dtype=np.int32)
RESULTS = np.array([1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0])
VALUES = np.array([0.6, 0.3, 0.8, 0.1, 0.4, 0.2, np.nan])


def _td(lam: float) -> np.ndarray:
    return td_labels(VALUES, RESULTS, SIDES, GAME, lam)


def test_td_lambda_zero_is_the_probe_and_one_is_the_result() -> None:
    probe = np.where(np.isnan(VALUES), RESULTS, VALUES)
    assert np.array_equal(_td(0.0), probe)
    assert np.array_equal(_td(1.0), RESULTS)


def test_td_lambda_matches_a_hand_computation() -> None:
    # λ = 0.5, each game backwards: L = 0.5 V + 0.5 x (the next L from this side's view,
    # or the result at the game's last sample).
    # game 0: L3 = .05 + 0 = .05; L2 = .4 + .5(1 - .05) = .875;
    #         L1 = .15 + .5(1 - .875) = .2125; L0 = .3 + .5(1 - .2125) = .69375
    # game 1: L = .2 + .25 = .45 (the draw ends it)
    # game 2: L6 = 0 (NaN: the result is the value, and the result ends it);
    #         L5 = .1 + .5 x 0 = .1 (same side to move: no flip)
    expected = [0.69375, 0.2125, 0.875, 0.05, 0.45, 0.1, 0.0]
    assert _td(0.5).tolist() == pytest.approx(expected, abs=1e-12)
    # Unrolled: L_i = (1 - λ) Σ_k λ^k V'_{i+k} + λ^(n-i) R_i, V' each seen from i's side.
    lam = 0.7
    seen = [0.6, 1 - 0.3, 0.8, 1 - 0.1]
    unrolled = (1 - lam) * sum(lam**k * v for k, v in enumerate(seen)) + lam**4 * 1.0
    assert _td(lam)[0] == pytest.approx(unrolled, abs=1e-12)


def test_td_labels_are_a_pure_function_of_the_line() -> None:
    first = _td(0.7)
    assert np.array_equal(first, _td(0.7))
    # Each game is its own line: a game's labels do not depend on its neighbours.
    alone = td_labels(VALUES[:4], RESULTS[:4], SIDES[:4], GAME[:4], 0.7)
    assert np.array_equal(first[:4], alone)


def test_a_td_label_at_lambda_zero_is_the_harvests_bit_for_bit(spec: dict) -> None:
    """The control arm of T3b: remade from the line, λ 0 is today's label exactly."""
    for procedure in (TINY, TINY.with_(result_weight=0.0), TINY.with_(stride=2)):
        data = _data(spec, procedure)
        assert data.has_line and data.positions > 0
        assert labels(data, procedure).tobytes() == data.targets.tobytes()
        assert relabel(data, procedure).digest == data.digest
    # Results-only data has no probe values, and its labels are the results.
    plain = TINY.with_(result_weight=1.0)
    data = _data(spec, plain)
    assert bool(np.isnan(data.values).all())
    assert labels(data, plain).tobytes() == data.targets.tobytes()


def test_the_harvest_keeps_each_samples_game_ply_result_and_value(spec: dict) -> None:
    procedure = TINY.with_(stride=2)
    records = self_play(spec, procedure, workers=1)
    data = harvest(spec, records, procedure)
    assert data.game.tolist() == sorted(data.game.tolist())
    for index, record in enumerate(records):
        here = data.game == index
        expected = list(range(procedure.skip_plies, len(record.moves), procedure.stride))
        assert data.ply[here].tolist() == expected
        for side, result in zip(data.sides[here], data.results[here], strict=True):
            assert result == record.score_for(int(side))
    # The searched part of the label is the value: label = w x result + (1 - w) x value.
    w = procedure.result_weight
    assert np.allclose(data.targets, w * data.results + (1 - w) * data.values, atol=1e-6)


def test_a_td_label_blends_the_line_and_needs_one(spec: dict) -> None:
    td = TINY.with_(td_lambda=0.7)
    data = _data(spec, TINY)
    remade = relabel(data, td)
    assert remade.digest != data.digest and np.all((remade.targets >= 0) & (remade.targets <= 1))
    searched = td_labels(data.values, data.results, data.sides, data.game, 0.7)
    w = td.result_weight
    assert np.allclose(remade.targets, w * data.results + (1 - w) * searched, atol=1e-6)
    # The line is kept, so a TD-labelled dataset relabels back to the probe's labels.
    assert relabel(remade, TINY).digest == data.digest
    bare = Dataset(data.n_inputs, data.offsets, data.inputs, data.sides, data.targets, data.games)
    with pytest.raises(ValueError, match="game line"):
        labels(bare, td)
    assert labels(bare, TINY) is bare.targets


def test_a_random_opening_is_the_cores_and_the_seeds_and_not_the_threads(spec: dict) -> None:
    from chess_variant_sandbox.agents.training.data import opening_plies_for, opening_seed
    from chess_variant_sandbox.tournament.play import choose_move, finish_game
    from chess_variant_sandbox.tournament.runner import TournamentConfig, schedule

    procedure = TINY.with_(games=8, opening_plies=6)
    one = self_play(spec, procedure, workers=1)
    four = self_play(spec, procedure, workers=4)
    assert [r.moves for r in one] == [r.moves for r in four]
    assert [r.winner for r in one] == [r.winner for r in four]
    assert len({tuple(r.moves) for r in one}) > 1

    # Each game is the random agent's opening from its own streams, then the roster's.
    engine = Engine(spec)
    tasks = schedule(TournamentConfig(agents=procedure.agents, games_per_pairing=8, seed=0))
    assert [r.seed for r in one] == [seed for seed, _a, _b in tasks]
    for record, (seed, first, second) in zip(one, tasks, strict=True):
        state = engine.initial_state()
        k = opening_plies_for(seed, procedure.opening_plies)
        assert 0 <= k <= procedure.opening_plies
        for ply in range(k):
            move = choose_move(engine, "random", state, opening_seed(seed, ply))
            assert record.moves[ply] == move
            state = engine.apply(state, move)
        rest = finish_game(engine, (first, second), state, seed)
        assert record.moves[k:] == rest.moves and record.winner == rest.winner
        assert record.plies == len(record.moves)

    # No opening is the ordinary runner's schedule, game for game.
    from chess_variant_sandbox.tournament.runner import run_tournament

    plain = self_play(spec, procedure.with_(opening_plies=0), workers=1)
    config = TournamentConfig(
        agents=procedure.agents, games_per_pairing=8, seed=0, workers=1, collect_series=False
    )
    assert [r.moves for r in plain] == [r.moves for r in run_tournament(spec, config).records]


def test_the_data_does_not_depend_on_the_worker_count(spec: dict) -> None:
    records = self_play(spec, TINY.with_(games=6), workers=1)
    one = harvest(spec, records, TINY, workers=1)
    three = harvest(spec, records, TINY, workers=3)
    assert one.digest == three.digest
    # The line too: each chunk's games carry their index in the whole harvest.
    for column in ("game", "ply", "results", "values"):
        assert np.array_equal(getattr(one, column), getattr(three, column), equal_nan=True)


# ------------------------------------------------------------------ the trainer


def test_the_gradients_are_the_derivative_of_the_loss(spec: dict) -> None:
    data = _data(spec)
    net = model.initial(data.n_inputs, TINY, data.most_active)
    for key in net.params:
        net.params[key] = net.params[key].astype(np.float64)
    rows = np.arange(min(48, data.positions))
    index = model._padded(data.offsets, data.inputs, rows, data.n_inputs)
    sides, targets = data.sides[rows], data.targets[rows].astype(np.float64)
    _loss, grads = model._gradients(net, index, sides, targets, 0.5)
    active = int(data.inputs[data.offsets[0]])
    rng = np.random.default_rng(0)
    checked = 0
    for key, value in net.params.items():
        for _ in range(3):
            if key == "w0":
                at: tuple[int, ...] = (active, int(rng.integers(value.shape[1])))
            else:
                at = tuple(int(rng.integers(size)) for size in value.shape)
            old = value[at]
            value[at] = old + 1e-6
            up, _ = model._gradients(net, index, sides, targets, 0.5)
            value[at] = old - 1e-6
            down, _ = model._gradients(net, index, sides, targets, 0.5)
            value[at] = old
            numeric = (up - down) / 2e-6
            assert grads[key][at] == pytest.approx(numeric, rel=1e-4, abs=1e-9), key
            checked += 1
    assert checked == 3 * len(net.params)


def test_quantizing_is_rounding_and_the_core_reads_the_bytes_as_meant(spec: dict) -> None:
    """The core's integer output is the float net's, within a few quantization steps."""
    from chess_variant_sandbox.agents.training.store import quantization_error

    data = _data(spec)
    net = model.train(data, TINY)
    blob = model.quantize(net)
    parsed = nets.Net.parse(blob)
    assert (parsed.n_inputs, parsed.accumulator, parsed.hidden1, parsed.hidden2) == (
        data.n_inputs, 16, 8, 4
    )
    assert max(abs(w) for w in parsed.w0) <= net.w0_limit
    # A step of the output weight is 1024/127 of a 1024th, times at most 127 per unit and
    # 4 units: a bound on rounding, not a tolerance on a bug.
    assert quantization_error(spec, net, blob, TINY) < 0.1


def test_the_trainer_learns_the_labels(spec: dict) -> None:
    data = _data(spec, TINY.with_(games=6))
    net = model.train(data, TINY.with_(epochs=6, learning_rate=0.01))
    assert net.history[-1] < net.history[0]


# ------------------------------------------------------------------ stores


def test_the_same_data_gives_the_same_store_byte_for_byte(
    spec: dict, tmp_path: Path, clean: None
) -> None:
    data = _data(spec)
    first = train_ruleset(spec, TINY, root=tmp_path / "a", workers=1, duel=False, data=data)
    second = train_ruleset(spec, TINY, root=tmp_path / "b", workers=1, duel=False, data=data)
    for one, two in zip(
        store_paths(first.record.ruleset_hash, TINY.digest, tmp_path / "a"),
        store_paths(second.record.ruleset_hash, TINY.digest, tmp_path / "b"),
        strict=True,
    ):
        assert one.read_bytes() == two.read_bytes(), one.name
    assert first.record.digest == second.record.digest


def test_a_td_procedure_trains_on_its_own_labels_from_saved_data(
    spec: dict, tmp_path: Path, clean: None
) -> None:
    """T3b's path: positions saved once, each λ arm trained from them under its own name."""
    data = Dataset.load(_data(spec).save(tmp_path / "d.npz"))
    td = TINY.with_(td_lambda=0.7)
    control = train_ruleset(spec, TINY, root=tmp_path, workers=1, duel=False, data=data)
    arm = train_ruleset(spec, td, root=tmp_path, workers=1, duel=False, data=data)
    assert control.data.digest == data.digest
    assert arm.data.digest == relabel(data, td).digest != data.digest
    payload = json.loads(arm.meta_path.read_text(encoding="utf-8"))
    assert payload["procedure"] == td.digest != TINY.digest
    assert payload["recipe"]["td_lambda"] == 0.7
    assert payload["data"]["digest"] == arm.data.digest


def test_a_store_is_named_by_its_procedure_and_carries_its_weights_hash_and_build(
    spec: dict, tmp_path: Path, clean: None
) -> None:
    done = train_ruleset(spec, TINY, root=tmp_path, workers=1, duel=False)
    meta, weights = store_paths(done.record.ruleset_hash, TINY.digest, tmp_path)
    assert done.meta_path == meta and weights.exists()
    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["procedure"] == TINY.digest
    assert payload["verdict"] == UNCHECKED
    assert payload["build"] == measurement.built_fingerprint()
    assert payload["digest"] == done.record.digest
    assert payload["recipe"] == TINY.as_dict()
    assert payload["data"]["digest"] == done.data.digest
    assert payload["acceptance"] is None
    unregister_net(TINY.digest, done.record.ruleset_hash)
    loaded = load_net(done.record.ruleset_hash, TINY.digest, tmp_path)
    assert loaded is not None and loaded.weights == weights.read_bytes()


def test_a_trained_net_plays_in_the_core_once_accepted_and_not_before(
    spec: dict, tmp_path: Path, clean: None
) -> None:
    engine = Engine(spec)
    done = train_ruleset(spec, TINY, root=tmp_path, workers=1, duel=False)
    name = f"minimax:net={TINY.digest},nodes=300"
    plain = play_game(engine, ("minimax:nodes=300", "greedy"), seed=1, collect_series=False)
    # Unchecked, it refuses to sit down (Q709): never the built-in objective under its name.
    with pytest.raises(nets.NetError, match="unchecked, not accepted"):
        play_game(engine, (name, "greedy"), seed=1, collect_series=False)

    register_net(NetRecord(TINY.digest, engine.spec.hash, ACCEPTED, done.record.weights))
    played = play_game(engine, (name, "greedy"), seed=1, collect_series=False)
    assert played.moves != plain.moves


def test_the_duel_judges_the_net_and_the_verdict_is_what_plays(
    spec: dict, tmp_path: Path, clean: None
) -> None:
    done = train_ruleset(spec, TINY, root=tmp_path, workers=1, pairs=1, nodes=60)
    found = done.acceptance
    assert found is not None and found.games == 2
    assert done.record.verdict == (ACCEPTED if found.accepted else REFUSED)
    payload = json.loads(done.meta_path.read_text(encoding="utf-8"))
    assert payload["acceptance"]["games"] == 2
    assert payload["verdict"] == done.record.verdict
    blob = nets.net_bytes({"net": TINY.digest}, done.record.ruleset_hash)
    assert blob == (done.record.weights if found.accepted else b"")


def test_the_opening_duel_samples_positions_and_not_the_threads(
    spec: dict, clean: None
) -> None:
    """§8.11: each pair opens as self-play does, both games from one opening, seats swapped."""
    engine = Engine(spec)
    weights = model.quantize(model.train(_data(spec), TINY))
    register_net(NetRecord(TINY.digest, engine.spec.hash, ACCEPTED, weights))
    one = accept(spec, TINY, pairs=6, nodes=60, workers=1, seed=500, opening_plies=4)
    three = accept(spec, TINY, pairs=6, nodes=60, workers=3, seed=500, opening_plies=4)
    assert one == three
    assert one.games == 12 and one.opening_plies == 4 and one.seed == 500
    assert 1 < one.openings <= 6

    candidate = f"minimax:net={TINY.digest},nodes=60"
    config = TournamentConfig(
        agents=(candidate, "minimax:nodes=60"), games_per_pairing=6, seed=500, workers=2,
        include_self_play=False, collect_series=False, paired=True,
    )
    by_seed: dict[int, list] = {}
    for record in _opening_duel(spec, config, 4):
        by_seed.setdefault(record.seed, []).append(record)
    assert sorted(by_seed) == list(range(500, 506))
    for first, second in by_seed.values():
        assert first.agents == second.agents[::-1]
        k = opening_plies_for(first.seed, 4)
        assert first.moves[:k] == second.moves[:k]

    plain = accept(spec, TINY, pairs=2, nodes=60, workers=1, seed=500, opening_plies=0)
    assert plain.openings == 1 and plain.opening_plies == 0


@pytest.mark.parametrize("opening_plies", [0, 4])
def test_a_curtailed_acceptance_is_the_full_duel_s_verdict_at_any_worker_count(
    spec: dict, clean: None, opening_plies: int
) -> None:
    """reference/compute-economy.md §4.2 (Q632): the verdict by construction, the score of the prefix."""
    engine = Engine(spec)
    weights = model.quantize(model.train(_data(spec), TINY))
    register_net(NetRecord(TINY.digest, engine.spec.hash, ACCEPTED, weights))
    options = {"pairs": 8, "nodes": 60, "seed": 500, "opening_plies": opening_plies}
    whole = accept(spec, TINY, workers=1, **options)
    one = accept(spec, TINY, workers=1, curtail=True, **options)
    four = accept(spec, TINY, workers=4, curtail=True, **options)
    assert one == four
    assert one.accepted == whole.accepted
    assert whole.decided_after == whole.planned == 8
    assert one.planned == 8 and one.decided_after < 8, "the fixture never stops early"
    assert one.games == 2 * one.decided_after
    stored = one.as_dict()
    assert (stored["decided_after"], stored["planned_pairs"]) == (one.decided_after, 8)


def test_train_runs_end_to_end_from_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean: None, capsys: pytest.CaptureFixture
) -> None:
    from chess_variant_sandbox.cli import main

    monkeypatch.chdir(tmp_path)
    args = [
        "train", str(GARDNER), "--agents", "greedy", "--games", "2", "--label-nodes", "40",
        "--label-depth", "3", "--accumulator", "16", "--hidden1", "8", "--hidden2", "4",
        "--epochs", "1", "--batch", "32", "--duel-pairs", "1", "--duel-nodes", "60",
        "--workers", "1", "--save-data", "d.npz",
    ]
    assert main(args) == 0
    procedure = DEFAULT.with_(
        agents=("greedy",), games=2, label_nodes=40, label_depth=3, accumulator=16,
        hidden1=8, hidden2=4, epochs=1, batch=32,
    )
    digest = Engine(load_spec(GARDNER)).spec.hash
    meta, weights = store_paths(digest, procedure.digest, "nets")
    assert meta.exists() and weights.exists() and Path("d.npz").exists()
    out = capsys.readouterr().out
    assert procedure.digest in out

    # A fresh process would find it on disk, as the player does: from `nets/` here.
    nets._NETS.clear()
    record = load_net(digest, procedure.digest)
    assert record is not None
    engine = Engine(load_spec(GARDNER))
    seats = (f"minimax:net={procedure.digest},nodes=60", "greedy")
    if record.verdict == ACCEPTED:
        assert play_game(engine, seats, seed=0, collect_series=False).plies > 0
    else:
        # Q709: a refused net does not sit down as the built-in objective.
        with pytest.raises(nets.NetError, match="refused"):
            play_game(engine, seats, seed=0, collect_series=False)

    # Again, without --force: the store stands.
    assert main(args) == 0
    assert "exists" in capsys.readouterr().out

    # From the saved positions, with no duel: the same data, the same weights.
    assert main([*args[:-2], "--data", "d.npz", "--no-duel", "--force"]) == 0
    assert weights.read_bytes() == record.weights

    # T3b's path (Q658): the positions alone, then a TD(λ) arm from them under its own name.
    assert main([*args[:-2], "--save-data", "h.npz", "--harvest-only"]) == 0
    # The run saved its positions with its own (TD) labels; the harvest's, relabelled, are those.
    harvested = Dataset.load("h.npz")
    assert relabel(harvested, procedure).digest == Dataset.load("d.npz").digest
    # Q671: the default is TD(lambda 0.7), and `--td-lambda 0` is the default before it.
    assert procedure.td_lambda == 0.7
    assert main([*args[:-2], "--data", "h.npz", "--no-duel", "--td-lambda", "0"]) == 0
    old = procedure.with_(td_lambda=0.0)
    assert old == Q637.with_(
        agents=("greedy",), games=2, label_nodes=40, label_depth=3, accumulator=16,
        hidden1=8, hidden2=4, epochs=1, batch=32,
    )
    assert old.digest != procedure.digest
    assert store_paths(digest, old.digest, "nets")[0].exists()


# ------------------------------------------------------------------ the basis


def test_a_roster_without_a_net_stamps_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every library stamp of a game with no rated net (Q502h): no `net` field, so none is
    rewritten and none reads stale. Gardner's own roster is its net ladder; with no store
    it is `CANONICAL_AGENTS`."""
    from chess_variant_sandbox import corpus
    from chess_variant_sandbox.corpus import CANONICAL_AGENTS, current_basis

    monkeypatch.setattr(corpus, "NETS_DIR", tmp_path / "no-nets")
    assert not any("net=" in agent for agent in CANONICAL_AGENTS)
    basis = current_basis(Engine(load_spec(GARDNER)).spec.hash)
    assert basis.net == ""
    assert "net" not in basis.as_dict()
    stored = measurement.Basis.from_dict(basis.as_dict())
    assert stored is not None and stored.net == ""
    assert not any("net" in reason for reason in stored.stale_against(basis).reasons)


def test_the_basis_stamps_the_net_that_played(clean: None) -> None:
    ruleset = Engine(load_spec(GARDNER)).spec.hash
    procedure = "abcdef012345"
    agents = (f"minimax:net={procedure},nodes=400", "greedy")
    assert measurement.net_players(agents, ruleset) == f"{procedure}=builtin"

    blob = nets.random_net(Engine(load_spec(GARDNER)).core.net_inputs(), seed=1)
    register_net(NetRecord(procedure, ruleset, ACCEPTED, blob))
    accepted = NetRecord(procedure, ruleset, ACCEPTED, blob).digest
    assert measurement.net_players(agents, ruleset) == f"{procedure}=net:{accepted}"

    def basis() -> measurement.Basis:
        return measurement.Basis.current(ruleset, "v11", agents, 2, (0,))

    played = basis()
    assert played.as_dict()["net"] == f"{procedure}=net:{accepted}"
    stored = measurement.Basis.from_dict(played.as_dict())
    assert stored is not None and stored.net == played.net

    register_net(NetRecord(procedure, ruleset, REFUSED, blob))
    now = basis()
    assert now.net == f"{procedure}=builtin"
    staleness = stored.stale_against(now)
    assert staleness.needs == "replay"
    assert any("nets were" in reason for reason in staleness.reasons)


def test_the_trainer_is_in_no_fingerprint_set() -> None:
    """It makes nets, which the basis names by digest; it cannot move a game without one."""
    training = (ROOT / "chess_variant_sandbox" / "agents" / "training").resolve()
    for kind in ("engine", "metrics", "analysis"):
        assert not any(
            training in path.resolve().parents for path in measurement.fingerprinted(kind)
        ), kind
