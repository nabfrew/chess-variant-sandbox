"""A generation's code (Q677, plan-teacher §9.1, T5-code): the parent net as teacher and seat.

What is held here, on gardner (a small board, not chess), a random net standing in for the
parent and tiny budgets:

- **the no-``rust/`` claim, first**: ``Core.search_values`` on ``minimax:net=<parent>``
  searches on the net -- its move and depth are ``select_move``'s and
  ``select_move_depth``'s for the same seed, its values are not the built-in's, and the
  answer does not depend on the thread count;
- ``Procedure.parent`` enters the digest only when set, so every existing name holds
  (``db5ef1bc938e``, ``1d160257b8e5``, ``1c193cc56c76``, T2v's ``bfca30cf2c4a``), and a
  parent-named digest moves with the parent's weights;
- the refusals that stay: a ``net=`` roster without a parent, or naming any net but
  exactly the parent; ``label_search="parent"`` without one;
- the generation's roster gives the parent half the seats, a pure function of the
  schedule's seed;
- the plumbing: the parent must resolve to the bytes its name carries before a game is
  played; the harvest labels with the parent player's value, the same at 1 and 3 worker
  processes (spawned workers register the parent); the child, C and K share every
  position; the store records the lineage; the duel takes another opponent.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from chess_variant_sandbox.agents import nets  # noqa: E402
from chess_variant_sandbox.agents.nets import NetError, random_net  # noqa: E402
from chess_variant_sandbox.agents.training.data import (  # noqa: E402
    harvest,
    harvest_arms,
    label_seed,
    labels,
    parent_bytes,
    player_agent,
    self_play,
    squash,
)
from chess_variant_sandbox.agents.training.procedure import (  # noqa: E402
    DEFAULT,
    GENERATION_GAMES,
    GENERATION_LABEL_NODES,
    Q502F2,
    Q637,
    Parent,
    Procedure,
    builtin_player_arm,
    control,
    generation,
)
from chess_variant_sandbox.agents.training.store import (  # noqa: E402
    accept,
    read_parent,
    register_parent,
    train_ruleset,
)
from chess_variant_sandbox.engine import Engine  # noqa: E402
from chess_variant_sandbox.engine.engine import _pack  # noqa: E402
from chess_variant_sandbox.spec import load_spec  # noqa: E402
from chess_variant_sandbox.tournament.play import eval_values, native_spec  # noqa: E402
from chess_variant_sandbox.tournament.runner import schedule  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GARDNER = ROOT / "rulesets" / "gardner.json"

# The pins this task must not move (plan-teacher §9.4, T5-code).
PINNED = {
    "db5ef1bc938e": DEFAULT,
    "1d160257b8e5": Q637,
    "1c193cc56c76": Q502F2,
    "bfca30cf2c4a": Q637.with_(label_search="player", label_nodes=51_200),
}
# The parent's procedure name in these tests: a made-up digest, a random net's bytes.
PARENT = "abcdefabcdef"

BASE = Procedure(
    agents=("greedy",),
    games=1,
    opening_plies=2,
    skip_plies=2,
    td_lambda=0.7,
    label_nodes=1_500,
    accumulator=16,
    hidden1=8,
    hidden2=4,
    epochs=1,
    batch=32,
)


@pytest.fixture(scope="module")
def spec() -> dict:
    return load_spec(GARDNER)


@pytest.fixture(scope="module")
def engine(spec: dict) -> Engine:
    return Engine(spec)


@pytest.fixture(scope="module")
def blob(engine: Engine) -> bytes:
    return random_net(engine.core.net_inputs(), 3)


@pytest.fixture
def clean() -> Iterator[None]:
    """Forget every net this test registered, so no other test meets one."""
    before = dict(nets._NETS)
    yield
    nets._NETS.clear()
    nets._NETS.update(before)


@pytest.fixture
def parent(engine: Engine, blob: bytes, clean: None) -> Parent:
    found = Parent(PARENT, hashlib.sha256(blob).hexdigest())
    register_parent(engine.spec.hash, found, blob)
    return found


def _tiny_child(parent: Parent, games: int = 1) -> Procedure:
    """``generation(parent)`` on tiny budgets: the roster at 100 nodes a seat."""
    return generation(parent, BASE).with_(
        agents=(f"minimax:net={parent.procedure},nodes=100", "minimax:nodes=100"),
        games=games,
        label_nodes=1_500,
    )


def _positions(engine: Engine, count: int, seed: int = 5) -> list[Any]:
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


# ------------------------------------------------------------------ the no-rust/ claim


def test_search_values_searches_on_a_net_player(engine: Engine, parent: Parent) -> None:
    """§9.4: the call carries the net (``AgentIn``'s bytes), so a generation needs no ``rust/``."""
    agent = f"minimax:net={parent.procedure},nodes=1500"
    native = native_spec(agent, engine.spec.hash)
    values = eval_values(engine.spec)
    states = _positions(engine, 10)
    seeds = [label_seed(17, ply) for ply in range(len(states))]
    answers = _values(engine, states, agent, seeds)
    assert all(answer is not None for answer in answers)
    for state, seed, (_value, move, depth) in zip(states, seeds, answers, strict=True):
        if len(engine.legal_moves(state)) > 1:
            assert engine.core.select_move_depth(_pack(state), native, values, seed) == (
                move,
                depth,
            )
            assert engine.core.select_move(_pack(state), native, values, seed) == move
    # The net is what is searched: its values are not the built-in objective's.
    builtin = _values(engine, states, "minimax:nodes=1500", seeds)
    assert [a[0] for a in answers] != [b[0] for b in builtin]
    # A pure function of the position and the seed, whatever the threads or the order.
    assert _values(engine, states, agent, seeds, threads=3) == answers
    assert _values(engine, states[::-1], agent, seeds[::-1], threads=3) == answers[::-1]


# ------------------------------------------------------------------ the digest


def test_every_existing_digest_holds_and_parent_is_absent_until_set() -> None:
    for digest, procedure in PINNED.items():
        assert procedure.digest == digest
        assert "parent" not in procedure.as_dict()
        assert procedure.with_(parent=None).digest == digest


def test_a_parent_named_digest_moves_with_the_parents_weights() -> None:
    one = generation(Parent("db5ef1bc938e", "a" * 64))
    two = generation(Parent("db5ef1bc938e", "b" * 64))
    other = generation(Parent("1d160257b8e5", "a" * 64))
    assert len({one.digest, two.digest, other.digest, DEFAULT.digest}) == 4
    assert one.as_dict()["parent"] == {"procedure": "db5ef1bc938e", "weights_sha256": "a" * 64}
    assert Procedure.from_dict(json.loads(json.dumps(one.as_dict()))) == one
    # The parent alone, on a recipe otherwise the default's, is a new name too.
    assert DEFAULT.with_(parent=Parent("db5ef1bc938e", "a" * 64)).digest != DEFAULT.digest


def test_the_generation_recipe_is_section_9_1s() -> None:
    parent = Parent("db5ef1bc938e", "c" * 64)
    child = generation(parent)
    assert child.agents == ("minimax:net=db5ef1bc938e,nodes=1600", "minimax:nodes=1600")
    assert child.games == GENERATION_GAMES == 5_000
    assert child.label_search == "parent" and child.label_nodes == GENERATION_LABEL_NODES == 6_400
    assert player_agent(child) == "minimax:net=db5ef1bc938e,nodes=6400"
    assert child.teacher == "parent:" + "c" * 12
    # Everything else is generation 0's: openings, samples, TD(0.7), the student, from scratch.
    for name in ("opening_plies", "skip_plies", "stride", "td_lambda", "result_weight",
                 "accumulator", "hidden1", "hidden2", "epochs", "batch", "init_seed", "seed"):
        assert getattr(child, name) == getattr(DEFAULT, name), name
    # C: the default's teacher (the probe at 1,200) on the child's games; K: the built-in
    # player at the child's budget.
    c, k = control(child), builtin_player_arm(child)
    assert (c.label_search, c.label_nodes, c.label_depth) == ("probe", 1_200, 8)
    assert (k.label_search, k.label_nodes) == ("player", 6_400)
    assert player_agent(k) == "minimax:nodes=6400"
    assert len({child.digest, c.digest, k.digest}) == 3


def test_the_refusals_that_stay() -> None:
    parent = Parent("db5ef1bc938e", "d" * 64)
    with pytest.raises(ValueError, match="generation"):
        DEFAULT.with_(agents=("minimax:net=abcdefabcdef,nodes=400",))
    with pytest.raises(ValueError, match="only its parent"):
        DEFAULT.with_(parent=parent, agents=("minimax:net=abcdefabcdef,nodes=400",))
    with pytest.raises(ValueError, match="only its parent"):
        generation(parent).with_(
            agents=("minimax:net=db5ef1bc938e,nodes=1600", "greedy:net=1d160257b8e5")
        )
    with pytest.raises(ValueError, match="parent"):
        DEFAULT.with_(label_search="parent")
    with pytest.raises(ValueError, match="hex"):
        Parent("db5ef1bc938", "d" * 64)
    with pytest.raises(ValueError, match="64 hex"):
        Parent("db5ef1bc938e", "D" * 64)
    # The parent's net in either seat or both is allowed.
    assert DEFAULT.with_(parent=parent, agents=("greedy:net=db5ef1bc938e",)).parent == parent


# ------------------------------------------------------------------ the schedule


def test_the_parent_takes_half_the_seats_a_pure_function_of_the_seed() -> None:
    from chess_variant_sandbox.agents.training.data import _config

    child = generation(Parent("db5ef1bc938e", "e" * 64)).with_(games=7, seed=40)
    tasks = schedule(_config(child, 1))
    assert len(tasks) == 4 * 7
    seats = [agent for _seed, first, second in tasks for agent in (first, second)]
    net = child.agents[0]
    assert seats.count(net) == len(seats) // 2
    assert {(first, second) for _seed, first, second in tasks} == {
        (a, b) for a in child.agents for b in child.agents
    }
    assert [seed for seed, _first, _second in tasks] == list(range(40, 40 + 28))
    assert schedule(_config(child, 5)) == tasks


# ------------------------------------------------------------------ the plumbing


def test_the_parent_must_be_the_bytes_its_name_carries(spec: dict, engine: Engine, blob: bytes, clean: None) -> None:
    sha = hashlib.sha256(blob).hexdigest()
    child = _tiny_child(Parent(PARENT, sha))
    with pytest.raises(NetError, match="no net"):
        self_play(spec, child, workers=1)
    with pytest.raises(NetError, match="not the parent"):
        register_parent(engine.spec.hash, Parent(PARENT, "0" * 64), blob)
    other = random_net(engine.core.net_inputs(), 4)
    nets.register_net(nets.NetRecord(PARENT, engine.spec.hash, nets.ACCEPTED, other))
    with pytest.raises(NetError, match="other weights"):
        parent_bytes(spec, child)
    register_parent(engine.spec.hash, Parent(PARENT, sha), blob)
    assert parent_bytes(spec, child) == blob


def test_the_harvest_labels_with_the_parents_player(spec: dict, engine: Engine, parent: Parent) -> None:
    child = _tiny_child(parent).with_(td_lambda=0.0, result_weight=0.5)
    records = self_play(spec, child, workers=1)
    assert len(records) == 4
    assert sum(r.agents.count(child.agents[0]) for r in records) == 4
    data = harvest(spec, records, child)
    assert data.teacher == child.teacher == f"parent:{parent.weights_sha256[:12]}"
    at = 0
    for record in records:
        state = engine.initial_state()
        asked, seeds = [], []
        for ply, move in enumerate(record.moves):
            if ply >= child.skip_plies:
                asked.append(state)
                seeds.append(label_seed(record.seed, ply))
            state = engine.apply(state, move)
        for answer in _values(engine, asked, player_agent(child), seeds):
            if answer is not None:
                value = squash(answer[0], child.scale)
                assert data.values[at] == value
                assert data.targets[at] == np.float32(0.5 * data.results[at] + 0.5 * value)
            at += 1
    assert at == data.positions and not np.isnan(data.values).all()
    # Another parent's values never train under this one's name, nor the built-in's.
    stranger = child.with_(parent=Parent(PARENT, "f" * 64))
    with pytest.raises(ValueError, match="labelled by the parent"):
        labels(data, stranger)
    with pytest.raises(ValueError, match="labelled by the parent"):
        labels(data, builtin_player_arm(child))


def test_the_parent_harvest_does_not_depend_on_the_worker_count(spec: dict, parent: Parent) -> None:
    child = _tiny_child(parent, games=2)
    records = self_play(spec, child, workers=2)
    assert [r.moves for r in records] == [r.moves for r in self_play(spec, child, workers=1)]
    one = harvest(spec, records, child, workers=1)
    three = harvest(spec, records, child, workers=3)
    assert one.digest == three.digest
    for column in ("game", "ply", "results", "values"):
        assert np.array_equal(getattr(one, column), getattr(three, column), equal_nan=True)


def test_the_child_c_and_k_share_every_position(spec: dict, parent: Parent) -> None:
    child = _tiny_child(parent)
    c, k = control(child, BASE).with_(label_nodes=40, label_depth=3), builtin_player_arm(child)
    found = harvest_arms(spec, [child, c, k], workers=1)
    assert [d.teacher for d in found] == [child.teacher, "probe", "player"]
    first = found[0]
    for data in found[1:]:
        assert np.array_equal(data.offsets, first.offsets)
        assert np.array_equal(data.inputs, first.inputs)
        assert np.array_equal(data.sides, first.sides)
        assert np.array_equal(data.results, first.results)
    assert len({d.digest for d in found}) == 3
    with pytest.raises(ValueError, match="share one self-play"):
        harvest_arms(spec, [child, child.with_(seed=1)], workers=1)


def test_read_parent_checks_the_name_and_the_bytes(engine: Engine, blob: bytes, tmp_path: Path) -> None:
    sha = hashlib.sha256(blob).hexdigest()
    weights = tmp_path / f"{engine.spec.hash}-{PARENT}.net"
    weights.write_bytes(blob)
    assert read_parent(weights, engine.spec.hash) == (Parent(PARENT, sha), blob)
    meta = weights.with_suffix(".json")
    meta.write_text(json.dumps({"weights_sha256": sha}), encoding="utf-8")
    assert read_parent(meta, engine.spec.hash)[0] == Parent(PARENT, sha)
    with pytest.raises(NetError, match="not"):
        read_parent(weights, "0" * len(engine.spec.hash))
    meta.write_text(json.dumps({"weights_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(NetError, match="does not hash"):
        read_parent(weights, engine.spec.hash)


def test_a_generation_trains_and_its_store_records_the_lineage(
    spec: dict, parent: Parent, tmp_path: Path
) -> None:
    child = _tiny_child(parent)
    done = train_ruleset(spec, child, root=tmp_path, workers=1, duel=False)
    payload = json.loads(done.meta_path.read_text(encoding="utf-8"))
    assert payload["procedure"] == child.digest
    assert payload["lineage"] == {
        "parent": PARENT,
        "parent_weights_sha256": parent.weights_sha256,
        "teacher": child.teacher,
    }
    assert Procedure.from_dict(payload["recipe"]) == child
    # The duel takes another opponent in the built-in's seat (P: the parent; A25: a
    # deeper built-in), on the same paired schedule.
    nets.register_net(nets.NetRecord(child.digest, Engine(spec).spec.hash, nets.ACCEPTED,
                                     done.record.weights))
    p = accept(spec, child, pairs=1, nodes=60, workers=1,
               reference=f"minimax:net={PARENT},nodes=60")
    assert p.reference == f"minimax:net={PARENT},nodes=60" and p.games == 2
    with pytest.raises(ValueError, match="two players"):
        accept(spec, child, pairs=1, nodes=60, reference=f"minimax:net={child.digest},nodes=60")


def test_a_stored_net_without_a_parent_carries_no_lineage(spec: dict, tmp_path: Path, clean: None) -> None:
    done = train_ruleset(spec, BASE.with_(td_lambda=0.0, label_nodes=40, label_depth=3),
                         root=tmp_path, workers=1, duel=False)
    assert "lineage" not in json.loads(done.meta_path.read_text(encoding="utf-8"))


def test_a_generation_runs_from_the_command_line(
    engine: Engine, blob: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    clean: None, capsys: pytest.CaptureFixture,
) -> None:
    """T5-g1's commands in miniature: harvest three arms once, train C, duel it against the parent."""
    from chess_variant_sandbox.cli import main

    monkeypatch.chdir(tmp_path)
    weights = tmp_path / f"{engine.spec.hash}-{PARENT}.net"
    weights.write_bytes(blob)
    tiny = [
        "--agents", f"minimax:net={PARENT},nodes=100", "minimax:nodes=100", "--games", "1",
        "--opening-plies", "2", "--label-nodes", "1500", "--accumulator", "16", "--hidden1", "8",
        "--hidden2", "4", "--epochs", "1", "--batch", "32", "--workers", "1",
    ]
    assert main(["train", str(GARDNER), "--parent", str(weights), *tiny, "--harvest-only",
                 "--save-data", "g1/d.npz", "--also-harvest", "C", "K"]) == 0
    for name in ("d.npz", "d.C.npz", "d.K.npz"):
        assert (tmp_path / "g1" / name).exists()
    # C trains from its harvest: the default's teacher, the probe at 1,200, on the child's games.
    c_flags = ["--label-search", "probe", "--label-nodes", "1200"]
    c_args = ["train", str(GARDNER), "--parent", str(weights), *tiny, *c_flags]
    assert main([*c_args, "--data", "g1/d.C.npz", "--store", "g1/nets", "--no-duel"]) == 0
    sha = hashlib.sha256(blob).hexdigest()
    c = control(_tiny_child(Parent(PARENT, sha)))
    assert c.digest in capsys.readouterr().out
    assert (tmp_path / "g1" / "nets" / f"{engine.spec.hash}-{c.digest}.json").exists()
    assert main([*c_args, "--store", "g1/nets", "--duel-only", "--duel-pairs", "1",
                 "--duel-nodes", "60", "--duel-seed", "2000000",
                 "--duel-against", f"minimax:net={PARENT},nodes=60"]) == 0
    row = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert row["duel"]["seed"] == 2_000_000 and row["duel"]["games"] == 2
    assert row["duel"]["reference"] == f"minimax:net={PARENT},nodes=60"


def test_the_duel_flags_go_with_duel_only(capsys: pytest.CaptureFixture) -> None:
    from chess_variant_sandbox.cli import main

    assert main(["train", str(GARDNER), "--duel-seed", "2000000"]) == 2
    assert main(["train", str(GARDNER), "--also-harvest", "C"]) == 2
    assert "--duel-only" in capsys.readouterr().err
