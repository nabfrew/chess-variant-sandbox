"""E1's replay-buffer field (Q712, plan-teacher §9.13.3): the parent's own training games.

What is held here, on gardner (a small board, not chess), random nets standing in for the
parent and its parent, and tiny budgets; nothing trains at scale and no store is written
under `nets/`:

- ``Procedure.replay`` enters the digest only when set, so every existing name holds
  (the default ``db5ef1bc938e`` and the stored generations, ``70b6ece08506`` among them),
  and a set field is a new recipe whose name moves with the replay net's weights;
- the refusals: a replay without a parent, a replay roster naming any net but its own, or
  none; a recipe that is not the parent's;
- E1's own procedure, read from the stores under ``29d5213b02b2``: its replay set is
  generation 1's schedule, its net generation 0's bytes, and the block it leaves out is
  exactly the new games' built-in-only block (schedules only, no game);
- the plumbing: the replay net must resolve to the bytes its name carries; the union is
  the new games then the replay schedule's net games; its replay half is a direct harvest
  of the parent's schedule under the child's teacher; the same at 1 and 3 workers; the
  child and C share every position; a store records the field and reads back as it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from chess_variant_sandbox.agents import nets  # noqa: E402
from chess_variant_sandbox.agents.nets import NetError, random_net  # noqa: E402
from chess_variant_sandbox.agents.training.data import (  # noqa: E402
    _config,
    harvest,
    harvest_arms,
    new_game_count,
    replay_bytes,
    replay_games,
    replay_tasks,
    self_play,
)
from chess_variant_sandbox.agents.training.procedure import (  # noqa: E402
    DEFAULT,
    Parent,
    Procedure,
    Replay,
    control,
    generation,
    replayed,
)
from chess_variant_sandbox.agents.training.store import (  # noqa: E402
    register_parent,
    train_ruleset,
)
from chess_variant_sandbox.engine import Engine  # noqa: E402
from chess_variant_sandbox.spec import load_spec  # noqa: E402
from chess_variant_sandbox.tournament.runner import schedule  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GARDNER = ROOT / "rulesets" / "gardner.json"
INTRANSITIVE = "29d5213b02b2"
# E1's names (plan-teacher §9.13.3, written there before the run): the child, generation
# 2R, and its control C, both over generation 1's union. Computed from the stores alone.
E1_CHILD = "0eaad64c4586"
E1_C = "91c0316f9a09"
# E1's gardner guard child (archive/plan-teacher §9.13.9-10, Q715): the same field over gardner's
# generation 1, accepted and stored whole.
E1_GARDNER_CHILD = "5fe1cb87d4c9"
# E1's chess guard child (archive/plan-teacher §9.13.11-12, Q716): the same field over chess's
# refused generation 1; refused itself, its JSON under nets/.
E1_CHESS_CHILD = "2f81b3ec2c36"
# Generation 3 (plan-teacher §9.14-9.15, Q724): the same field from 2R on Intransitive and
# from E1's gardner child on gardner; both stored whole as accepted, not kept.
G3_CHILD = "56026eae103c"
G3_GARDNER_CHILD = "7444909265ed"
# Generation 3c (plan-teacher §9.19-9.20, Q969b, Q969g): the chained replay from 2R, from E1's
# gardner child and from E1's chess child; the first two stored whole, chess's JSON refused.
G3C_CHILDREN = ("b14f1d190426", "ae241123e674", "0bbc0c08b1ee")

# Made-up procedure names for the random nets standing in for a parent and its parent.
PARENT = "abcdefabcdef"
GRANDPARENT = "bcdefabcdefa"

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


@pytest.fixture
def clean() -> Iterator[None]:
    """Forget every net this test registered, so no other test meets one."""
    before = dict(nets._NETS)
    yield
    nets._NETS.clear()
    nets._NETS.update(before)


def _net(engine: Engine, name: str, seed: int, register: bool = True) -> Parent:
    blob = random_net(engine.core.net_inputs(), seed)
    found = Parent(name, hashlib.sha256(blob).hexdigest())
    if register:
        register_parent(engine.spec.hash, found, blob)
    return found


def _tiny_generation(parent: Parent, games: int = 1) -> Procedure:
    """``generation(parent)`` on tiny budgets: the roster at 100 nodes a seat."""
    return generation(parent, BASE).with_(
        agents=(f"minimax:net={parent.procedure},nodes=100", "minimax:nodes=100"),
        games=games,
        label_nodes=1_500,
    )


@pytest.fixture
def lineage(engine: Engine, clean: None) -> tuple[Procedure, Procedure]:
    """The parent's recipe (a generation of the grandparent) and E1's child of it."""
    grandparent = _net(engine, GRANDPARENT, 5)
    parent_recipe = _tiny_generation(grandparent, games=2).with_(seed=7)
    # The parent's name is its recipe's digest, as a store's is.
    blob = random_net(engine.core.net_inputs(), 3)
    parent = Parent(parent_recipe.digest, hashlib.sha256(blob).hexdigest())
    register_parent(engine.spec.hash, parent, blob)
    return parent_recipe, replayed(_tiny_generation(parent), parent_recipe)


# ------------------------------------------------------------------ the digest


def test_the_field_is_absent_until_set_and_no_stored_name_moves() -> None:
    assert DEFAULT.digest == "db5ef1bc938e" and "replay" not in DEFAULT.as_dict()
    for meta in sorted((ROOT / "nets").glob("*.json")):
        recipe = json.loads(meta.read_text("utf-8")).get("recipe")
        if recipe is None:
            continue
        procedure = Procedure.from_dict(recipe)
        name = meta.stem.partition("-")[2]
        # Q713, Q715, Q716, Q724: E1's child and its two guard children, and generation 3's
        # child and its gardner guard child, and generation 3c's three (Q969g), are the stored
        # recipes with a replay set; every other reads back without the field, under its old name.
        if name in (E1_CHILD, E1_GARDNER_CHILD, E1_CHESS_CHILD, G3_CHILD, G3_GARDNER_CHILD,
                    *G3C_CHILDREN):
            assert procedure.replay is not None and "replay" in recipe, meta.name
        else:
            assert procedure.replay is None and "replay" not in procedure.as_dict(), meta.name
        assert procedure.digest == name, meta.name


def test_a_replay_is_a_new_recipe_named_by_its_schedule_and_its_nets_weights() -> None:
    grandparent = Parent("db5ef1bc938e", "a" * 64)
    parent_recipe = generation(grandparent)
    child = generation(Parent(parent_recipe.digest, "b" * 64))
    one = replayed(child, parent_recipe)
    assert one.replay == Replay(
        agents=parent_recipe.agents, games=5_000, seed=0, opening_plies=8, net=grandparent
    )
    assert one.digest != child.digest
    assert Procedure.from_dict(json.loads(json.dumps(one.as_dict()))) == one
    # The replay net's weights alone move the name (the parent's recipe moves with them).
    other_recipe = generation(Parent("db5ef1bc938e", "c" * 64))
    other_weights = replayed(generation(Parent(other_recipe.digest, "b" * 64)), other_recipe)
    names = {
        one.digest,
        other_weights.digest,
        one.with_(replay=Replay.of(parent_recipe.with_(seed=1))).digest,
        child.digest,
    }
    assert len(names) == 4
    # C keeps the field: the same union under the default's teacher, another name.
    assert control(one).replay == one.replay and control(one).digest != one.digest


def test_the_refusals() -> None:
    grandparent = Parent("db5ef1bc938e", "a" * 64)
    parent_recipe = generation(grandparent)
    replay = Replay.of(parent_recipe)
    with pytest.raises(ValueError, match="generation's"):
        DEFAULT.with_(replay=replay)
    with pytest.raises(ValueError, match="only its own net"):
        Replay(agents=("minimax:net=1d160257b8e5,nodes=1600",), games=1, seed=0,
               opening_plies=0, net=grandparent)
    with pytest.raises(ValueError, match="nothing to replay"):
        Replay(agents=("minimax:nodes=1600",), games=1, seed=0, opening_plies=0, net=grandparent)
    with pytest.raises(ValueError, match="at least 1"):
        Replay(agents=parent_recipe.agents, games=0, seed=0, opening_plies=0, net=grandparent)
    with pytest.raises(ValueError, match="no parent of its own"):
        Replay.of(DEFAULT)
    child = generation(Parent(parent_recipe.digest, "b" * 64))
    with pytest.raises(ValueError, match="not the parent"):
        replayed(child, parent_recipe.with_(seed=3))
    with pytest.raises(ValueError, match="names no parent"):
        replayed(DEFAULT, parent_recipe)


# ------------------------------------------------------------------ E1's procedure


def _store(name: str) -> dict:
    return json.loads((ROOT / "nets" / f"{INTRANSITIVE}-{name}.json").read_text("utf-8"))


def test_e1s_child_is_generation_one_with_its_own_games_replayed() -> None:
    """§9.13.3's table: the parent and teacher generation 1, the new games generation 2's,
    the replay set generation 1's schedule with generation 0's net by its bytes."""
    one, zero = _store("df1148b26ff6"), _store("db5ef1bc938e")
    parent_recipe = Procedure.from_dict(one["recipe"])
    plain = generation(Parent("df1148b26ff6", one["weights_sha256"]))
    assert plain.digest == "70b6ece08506"  # generation 2's name, unmoved by the field
    child = replayed(plain, parent_recipe)
    assert (child.digest, control(child).digest) == (E1_CHILD, E1_C)
    assert child.replay == Replay(
        agents=("minimax:net=db5ef1bc938e,nodes=1600", "minimax:nodes=1600"),
        games=5_000,
        seed=0,
        opening_plies=8,
        net=Parent("db5ef1bc938e", zero["weights_sha256"]),
    )
    assert control(child).label_search == "probe" and control(child).label_nodes == 1_200
    # The block left out is the new games' own built-in-only block, seed for seed.
    played = replay_tasks(child)
    whole = schedule(_config(parent_recipe, 1))
    kept = set(played)
    left_out = [task for task in whole if task not in kept]
    assert played == [task for task in whole if task in kept]  # in schedule order
    new = schedule(_config(child, 1))
    assert len(played) == 15_000 and new_game_count(child) == 20_000
    assert left_out == [task for task in new if "net=" not in task[1] + task[2]]
    assert left_out == [(seed, "minimax:nodes=1600", "minimax:nodes=1600")
                        for seed in range(15_000, 20_000)]


# ------------------------------------------------------------------ the plumbing


def test_the_replay_net_must_be_the_bytes_its_name_carries(
    spec: dict, engine: Engine, clean: None
) -> None:
    grandparent = _net(engine, GRANDPARENT, 5, register=False)
    parent_recipe = _tiny_generation(grandparent)
    parent = _net(engine, parent_recipe.digest, 3)
    child = replayed(_tiny_generation(parent), parent_recipe)
    with pytest.raises(NetError, match="replay roster's net=.*no net"):
        self_play(spec, child, workers=1)
    other = random_net(engine.core.net_inputs(), 4)
    nets.register_net(nets.NetRecord(GRANDPARENT, engine.spec.hash, nets.ACCEPTED, other))
    with pytest.raises(NetError, match="other weights"):
        replay_bytes(spec, child)


def test_the_union_is_the_new_games_then_the_parents_net_games(
    spec: dict, lineage: tuple[Procedure, Procedure]
) -> None:
    parent_recipe, child = lineage
    games = self_play(spec, child, workers=1)
    new = new_game_count(child)
    assert new == 4 and len(games) == 4 + 3 * 2  # the parent's 4 x 2, its built-in block out
    plain = self_play(spec, child.with_(replay=None), workers=1)
    assert [g.moves for g in games[:new]] == [g.moves for g in plain]
    parents = [g for g in self_play(spec, parent_recipe, workers=1)
               if any("net=" in agent for agent in g.agents)]
    assert [(g.seed, g.agents, g.moves) for g in games[new:]] == [
        (g.seed, g.agents, g.moves) for g in parents
    ]
    assert all("net=" in "".join(g.agents) for g in games[new:])


def test_the_replay_half_is_a_direct_harvest_of_the_parents_schedule(
    spec: dict, lineage: tuple[Procedure, Procedure]
) -> None:
    """Labelled by the child's teacher (the parent), sampled as the child samples."""
    parent_recipe, child = lineage
    union = harvest(spec, self_play(spec, child, workers=1), child)
    parents = [g for g in self_play(spec, parent_recipe, workers=1)
               if any("net=" in agent for agent in g.agents)]
    direct = harvest(spec, parents, child)
    assert union.teacher == direct.teacher == child.teacher
    new = new_game_count(child)
    start = int(np.searchsorted(union.game, new))
    assert union.positions - start == direct.positions > 0
    first = int(union.offsets[start])
    assert np.array_equal(union.offsets[start:] - first, direct.offsets)
    assert np.array_equal(union.inputs[first:], direct.inputs)
    for column in ("sides", "targets", "ply", "results", "values"):
        assert np.array_equal(getattr(union, column)[start:], getattr(direct, column),
                              equal_nan=True), column
    assert np.array_equal(union.game[start:] - new, direct.game)
    assert union.games == new + direct.games


def test_the_union_does_not_depend_on_the_worker_count(
    spec: dict, lineage: tuple[Procedure, Procedure]
) -> None:
    _parent_recipe, child = lineage
    records = self_play(spec, child, workers=3)
    assert [r.moves for r in records] == [r.moves for r in self_play(spec, child, workers=1)]
    assert [r.moves for r in replay_games(spec, child, workers=3)] == [
        r.moves for r in records[new_game_count(child):]
    ]
    one = harvest(spec, records, child, workers=1)
    three = harvest(spec, records, child, workers=3)
    assert one.digest == three.digest
    for column in ("game", "ply", "results", "values"):
        assert np.array_equal(getattr(one, column), getattr(three, column), equal_nan=True)


def test_the_child_and_c_share_every_position_of_the_union(
    spec: dict, lineage: tuple[Procedure, Procedure]
) -> None:
    _parent_recipe, child = lineage
    c = control(child, BASE).with_(label_nodes=40, label_depth=3)
    found = harvest_arms(spec, [child, c], workers=1)
    assert [d.teacher for d in found] == [child.teacher, "probe"]
    assert found[0].games == found[1].games == new_game_count(child) + 6
    for column in ("offsets", "inputs", "sides", "game", "ply", "results"):
        assert np.array_equal(getattr(found[0], column), getattr(found[1], column)), column
    assert found[0].digest != found[1].digest
    with pytest.raises(ValueError, match="share one self-play"):
        harvest_arms(spec, [child, child.with_(replay=None)], workers=1)


def test_train_replay_names_e1s_child_and_c(
    tmp_path: Path, clean: None, capsys: pytest.CaptureFixture
) -> None:
    """Q713: `train --parent <gen 1> --replay <gen 1's .json>` is E1's child, and with the
    probe flags its C, by name; no game is played (an empty store makes --duel-only stop)."""
    from chess_variant_sandbox.cli import main

    ruleset = str(ROOT / "rulesets" / "intransitive.json")
    one = ROOT / "nets" / f"{INTRANSITIVE}-df1148b26ff6"
    args = ["train", ruleset, "--parent", f"{one}.net", "--replay", f"{one}.json",
            "--store", str(tmp_path), "--duel-only"]
    assert main(args) == 2
    assert f"{INTRANSITIVE}-{E1_CHILD}.net" in capsys.readouterr().err
    assert main([*args, "--label-search", "probe", "--label-nodes", "1200"]) == 2
    assert f"{INTRANSITIVE}-{E1_C}.net" in capsys.readouterr().err
    # --replay-net registers the replay roster's net by its bytes (a guard's refused
    # generation 0 lies outside nets/); the name does not move.
    zero = ROOT / "nets" / f"{INTRANSITIVE}-db5ef1bc938e.net"
    assert main([*args, "--replay-net", str(zero)]) == 2
    assert f"{INTRANSITIVE}-{E1_CHILD}.net" in capsys.readouterr().err
    assert main([*args, "--replay-net", f"{one}.net"]) == 2
    assert "not the replay roster's net" in capsys.readouterr().err


def test_train_replay_refusals(tmp_path: Path, clean: None, capsys: pytest.CaptureFixture) -> None:
    from chess_variant_sandbox.cli import main

    ruleset = str(ROOT / "rulesets" / "intransitive.json")
    one = ROOT / "nets" / f"{INTRANSITIVE}-df1148b26ff6"
    zero = ROOT / "nets" / f"{INTRANSITIVE}-db5ef1bc938e"
    assert main(["train", ruleset, "--replay", f"{one}.json", "--duel-only"]) == 2
    assert "--replay goes with --parent" in capsys.readouterr().err
    assert main(["train", ruleset, "--parent", f"{one}.net", "--replay-net", f"{zero}.net",
                 "--duel-only"]) == 2
    assert "--replay-net goes with --replay" in capsys.readouterr().err
    # Generation 0's recipe is not generation 1's: the replay set must be the parent's games.
    assert main(["train", ruleset, "--parent", f"{one}.net", "--replay", f"{zero}.json",
                 "--duel-only"]) == 2
    assert "not the parent" in capsys.readouterr().err
    blank = tmp_path / "blank.json"
    blank.write_text("{}", encoding="utf-8")
    assert main(["train", ruleset, "--parent", f"{one}.net", "--replay", str(blank),
                 "--duel-only"]) == 2
    assert "no recipe" in capsys.readouterr().err


def test_a_replayed_generation_trains_and_its_store_records_the_field(
    spec: dict, lineage: tuple[Procedure, Procedure], tmp_path: Path
) -> None:
    _parent_recipe, child = lineage
    done = train_ruleset(spec, child, root=tmp_path, workers=1, duel=False)
    payload = json.loads(done.meta_path.read_text(encoding="utf-8"))
    assert payload["procedure"] == child.digest
    assert payload["data"]["games"] == new_game_count(child) + 6
    assert Procedure.from_dict(payload["recipe"]) == child
    assert payload["recipe"]["replay"]["net"]["procedure"] == GRANDPARENT
