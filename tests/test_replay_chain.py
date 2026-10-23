"""E3b's chained replay (archive/plan-teacher §9.17.4): every generation's games kept in the union.

What is held here:

- the names: ``Replay.prior`` enters the digest only when set, so every one-link recipe
  keeps its name (``replayed_deep`` of a parent without a replay set is ``replayed``'s
  child), and E3b's chained gardner child of ``5fe1cb87d4c9`` (F3b) at ``init_seed`` 0, 1, 2
  is named from the stores alone, its links generation 1's games and then generation 0's;
- ``data.joined`` on tiny hand-built arrays (no engine): the game ids are renumbered past
  the parts before, so a TD(λ) line never runs across a seam, and the join survives a save;
- the plumbing, on gardner with random nets standing in for three generations and tiny
  budgets (as ``test_replay.py``): the chained union is the new games, then each link in
  order without its built-in-only block; each part harvested alone (``link_games``) and
  joined is the whole chained harvest, array for array, at 1 and 3 workers (3 in the slow tier);
- a one-link replay is main's (E1's) as it was: its harvest is byte for byte the one built
  by Q712's own steps, transcribed below;
- ``train --chain`` names F3b, and ``--harvest-link`` and ``--chain`` refuse what they
  cannot do, with no game played.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from chess_variant_sandbox.agents import nets  # noqa: E402
from chess_variant_sandbox.agents.nets import random_net  # noqa: E402
from chess_variant_sandbox.agents.training.data import (  # noqa: E402
    Dataset,
    _play_tasks,
    harvest,
    joined,
    labels,
    link_count,
    link_games,
    new_game_count,
    replay_tasks,
    self_play,
)
from chess_variant_sandbox.agents.training.procedure import (  # noqa: E402
    Parent,
    Procedure,
    Replay,
    generation,
    replayed,
    replayed_deep,
)
from chess_variant_sandbox.agents.training.store import register_parent  # noqa: E402
from chess_variant_sandbox.engine import Engine  # noqa: E402
from chess_variant_sandbox.spec import load_spec  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GARDNER = ROOT / "rulesets" / "gardner.json"
GARDNER_HASH = "975dde173f01"
# Generation 3's gardner guard child F3 (Q724), the one-link child of E1's guard child.
F3 = "7444909265ed"
# E3b's F3b (§9.17.4): the chained child of `5fe1cb87d4c9`, by `init_seed`. Computed from the
# stores alone, before any game.
F3B = {0: "ae241123e674", 1: "2598368ca9de", 2: "9e054c892dc1"}

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

# Made-up names for the random nets standing in for generation 0.
GEN0 = "bcdefabcdefa"
# The teacher of the hand-built datasets: a parent whose weights hash to b...b.
TEACHER = "parent:" + "b" * 12


def _stored(name: str) -> dict:
    return json.loads((ROOT / "nets" / f"{GARDNER_HASH}-{name}.json").read_text("utf-8"))


def _f3b(init_seed: int = 0) -> Procedure:
    stored = _stored("5fe1cb87d4c9")
    child = generation(Parent("5fe1cb87d4c9", stored["weights_sha256"]))
    return replayed_deep(child, Procedure.from_dict(stored["recipe"])).with_(init_seed=init_seed)


# ------------------------------------------------------------------ the names


def test_a_one_link_replay_keeps_its_name() -> None:
    stored = _stored("5fe1cb87d4c9")
    child = generation(Parent("5fe1cb87d4c9", stored["weights_sha256"]))
    one = replayed(child, Procedure.from_dict(stored["recipe"]))
    assert one.digest == F3 and one.replay is not None and one.replay.prior is None
    assert "prior" not in one.as_dict()["replay"]
    assert one.replay.links == (one.replay,)
    # A parent without a replay set of its own: the chain is one link, E1's child.
    e1 = _stored("e03365fe43e7")
    plain = generation(Parent("e03365fe43e7", e1["weights_sha256"]))
    recipe = Procedure.from_dict(e1["recipe"])
    assert recipe.replay is None
    assert replayed_deep(plain, recipe) == replayed(plain, recipe)
    assert replayed_deep(plain, recipe).digest == "5fe1cb87d4c9"


def test_f3b_chains_generation_one_then_generation_zero() -> None:
    f3b = _f3b()
    assert f3b.replay is not None
    one, zero = f3b.replay.links
    assert one == Replay(
        agents=("minimax:net=e03365fe43e7,nodes=1600", "minimax:nodes=1600"),
        games=5_000, seed=0, opening_plies=8,
        net=Parent("e03365fe43e7", _stored("e03365fe43e7")["weights_sha256"]),
        prior=zero,
    )
    # Generation 0 is the refused `db5ef1bc938e`, by the bytes `5fe1cb87d4c9`'s replay named.
    assert zero.as_dict() == _stored("5fe1cb87d4c9")["recipe"]["replay"]
    assert zero == Replay(
        agents=("minimax:net=db5ef1bc938e,nodes=1600", "minimax:nodes=1600"),
        games=5_000, seed=0, opening_plies=8,
        net=Parent("db5ef1bc938e",
                   "b85a969f6131b250b3eea58e11693b3a5f7e480026ec5bc3076abae0cbf352fa"),
    )
    assert f3b.teacher == "parent:317025dfb035"
    assert Procedure.from_dict(json.loads(json.dumps(f3b.as_dict()))) == f3b
    assert {s: _f3b(s).digest for s in F3B} == F3B
    assert len({*F3B.values(), F3}) == 4


def test_each_link_names_only_its_own_net() -> None:
    zero = Parent("db5ef1bc938e", "a" * 64)
    link = Replay(agents=("minimax:net=db5ef1bc938e,nodes=1600",), games=1, seed=0,
                  opening_plies=0, net=zero)
    with pytest.raises(ValueError, match="only its own net"):
        Replay(agents=("minimax:net=db5ef1bc938e,nodes=1600",), games=1, seed=0,
               opening_plies=0, net=Parent("e03365fe43e7", "b" * 64), prior=link)
    chained = Replay(agents=("minimax:net=e03365fe43e7,nodes=1600",), games=1, seed=0,
                     opening_plies=0, net=Parent("e03365fe43e7", "b" * 64), prior=link)
    assert chained.links == (chained, link)
    assert Replay.from_dict(chained.as_dict()) == chained


# ------------------------------------------------------------------ joined, on tiny arrays


def _tiny(games: list[list[tuple[int, list[int]]]], teacher: str | None = TEACHER) -> Dataset:
    """A dataset of ``games``, each a list of (side, inputs) samples; a fixed made-up line."""
    rows = [row for game in games for _side, row in game]
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum([len(row) for row in rows], out=offsets[1:])
    count = len(rows)
    rng = np.random.default_rng(len(rows))
    return Dataset(
        n_inputs=10,
        offsets=offsets,
        inputs=np.array([i for row in rows for i in row], dtype=np.int32),
        sides=np.array([side for game in games for side, _row in game], dtype=np.int8),
        targets=rng.random(count).astype(np.float32),
        games=len(games),
        game=np.array([g for g, game in enumerate(games) for _ in game], dtype=np.int32),
        ply=np.array([p + 2 for game in games for p in range(len(game))], dtype=np.int32),
        results=rng.choice([0.0, 0.5, 1.0], count),
        values=rng.random(count),
        teacher=teacher,
    )


def test_joined_renumbers_the_games_and_keeps_each_line(tmp_path: Path) -> None:
    first = _tiny([[(0, [1, 2]), (1, [3])], [(0, [4])]])
    # Its first sampled game is numbered 1 there, as the first part's last game is.
    second = _tiny([[], [(1, [5, 6, 7])], [(0, [8]), (1, [9])]])
    both = joined([first, second])
    assert both.games == 5 and both.positions == 6
    assert both.game.tolist() == [0, 0, 1, 3, 4, 4]
    assert both.offsets.tolist() == [0, 2, 3, 4, 7, 8, 9]
    assert both.inputs.tolist() == list(range(1, 10))
    for column in ("sides", "targets", "ply", "results", "values"):
        assert np.array_equal(getattr(both, column),
                              np.concatenate([getattr(first, column), getattr(second, column)]))
    # The TD(λ) labels are each part's own: no line crosses the seam, though the first
    # part's last game and the second's first sampled one were both numbered 1.
    td = BASE.with_(label_search="parent", parent=Parent("a" * 12, "b" * 64))
    assert td.teacher == TEACHER and td.td_lambda == 0.7
    parts = np.concatenate([labels(first, td), labels(second, td)])
    assert labels(both, td).tobytes() == parts.tobytes()
    stuck = replace(both, game=np.concatenate([first.game, second.game]))
    assert labels(stuck, td).tobytes() != parts.tobytes()  # what the renumbering prevents
    loaded = Dataset.load(both.save(tmp_path / "both.npz"))
    assert loaded.digest == both.digest and loaded.teacher == both.teacher
    for column in ("game", "ply", "results", "values"):
        assert np.array_equal(getattr(loaded, column), getattr(both, column))
    assert joined([first]).digest == first.digest


def test_joined_refuses_what_is_not_one_harvest() -> None:
    first = _tiny([[(0, [1])]])
    with pytest.raises(ValueError, match="at least one"):
        joined([])
    with pytest.raises(ValueError, match="different teachers"):
        joined([first, _tiny([[(0, [2])]], teacher="probe")])
    other = replace(first, n_inputs=11)
    with pytest.raises(ValueError, match="input layouts"):
        joined([first, other])
    bare = replace(first, game=None, ply=None, results=None, values=None)
    with pytest.raises(ValueError, match="game line"):
        joined([first, bare])


# ------------------------------------------------------------------ the plumbing, on gardner


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


def _net(engine: Engine, name: str, seed: int) -> Parent:
    blob = random_net(engine.core.net_inputs(), seed)
    found = Parent(name, hashlib.sha256(blob).hexdigest())
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
def chain(engine: Engine, clean: None) -> tuple[Procedure, Procedure, Procedure]:
    """Three generations as gardner's are: generation 1's recipe (a child of generation 0),
    generation 2's (E1's child of generation 1, replaying generation 0's games), and E3b's
    chained child of generation 2. Each net is random, each name its recipe's digest."""
    zero = _net(engine, GEN0, 5)
    one_recipe = _tiny_generation(zero, games=2).with_(seed=7)
    one = _net(engine, one_recipe.digest, 3)
    two_recipe = replayed(_tiny_generation(one, games=2).with_(seed=11), one_recipe)
    two = _net(engine, two_recipe.digest, 9)
    return one_recipe, two_recipe, replayed_deep(_tiny_generation(two), two_recipe)


def _net_games(records: list) -> list:
    return [record for record in records if any("net=" in agent for agent in record.agents)]


def test_the_chained_union_is_the_new_games_then_each_link_in_order(
    spec: dict, chain: tuple[Procedure, Procedure, Procedure]
) -> None:
    one_recipe, two_recipe, child = chain
    assert child.replay is not None and link_count(child) == 3
    assert [link.net.procedure for link in child.replay.links] == [one_recipe.digest, GEN0]
    games = self_play(spec, child, workers=1)
    new = new_game_count(child)
    assert new == 4 and len(games) == 4 + 6 + 6  # each link's 4 x 2, its built-in block out
    expected = [
        *self_play(spec, child.with_(replay=None), workers=1),
        *_net_games(self_play(spec, two_recipe.with_(replay=None), workers=1)),
        *_net_games(self_play(spec, one_recipe, workers=1)),
    ]
    assert [(g.seed, g.agents, g.moves) for g in games] == [
        (g.seed, g.agents, g.moves) for g in expected
    ]
    assert len(replay_tasks(child, 1)) == len(replay_tasks(child, 2)) == 6
    assert replay_tasks(child, 1) == replay_tasks(child) != replay_tasks(child, 2)
    with pytest.raises(ValueError, match="replay link 3"):
        replay_tasks(child, 3)


# At 3 workers each harvest spawns its processes (about 5 s on Windows), so it is the
# weekly slow tier's; `test_replay.py` holds a harvest at 3 workers in the fast suite.
@pytest.mark.parametrize("workers", [1, pytest.param(3, marks=pytest.mark.slow)])
def test_each_part_harvested_alone_and_joined_is_the_chained_harvest(
    spec: dict, chain: tuple[Procedure, Procedure, Procedure], workers: int
) -> None:
    _one_recipe, _two_recipe, child = chain
    # Labelling is 3.1 s of the 3.5 s at 1,500 nodes a position (Q997b); the claim is that
    # the parts joined are the whole, array for array, whatever the labels are, so a small
    # budget proves the same (`label_nodes` is not in the games' names or the links' nets).
    child = child.with_(label_nodes=100)
    whole = harvest(spec, self_play(spec, child, workers=workers), child, workers=workers)
    parts = [
        harvest(spec, link_games(spec, child, link, workers=workers), child, workers=workers)
        for link in range(link_count(child))
    ]
    assert [part.games for part in parts] == [4, 6, 6]
    assert all(part.teacher == child.teacher and part.positions for part in parts)
    assert all(int(part.game[0]) == 0 for part in parts)  # each alone numbers from 0
    found = joined(parts)
    assert found.digest == whole.digest and found.games == whole.games
    for column in ("offsets", "inputs", "sides", "targets", "game", "ply", "results", "values"):
        mine, theirs = getattr(found, column), getattr(whole, column)
        assert mine.dtype == theirs.dtype, column
        assert np.array_equal(mine, theirs, equal_nan=True), column
    assert found.teacher == whole.teacher
    assert labels(found, child).tobytes() == labels(whole, child).tobytes()


def _main_replay_games(spec: dict, procedure: Procedure) -> list:
    """Q712's replay games as main built them before the chain (24f18ab5), transcribed:
    the parent's schedule's tasks naming a net, opened as self-play's are, in task order."""
    from chess_variant_sandbox.tournament.runner import TournamentConfig, schedule

    replay = procedure.replay
    assert replay is not None and replay.prior is None and replay.opening_plies > 0
    config = TournamentConfig(agents=replay.agents, games_per_pairing=replay.games,
                              seed=replay.seed, workers=1, collect_series=False)
    tasks = [task for task in schedule(config) if "net=" in task[1] + task[2]]
    return _play_tasks(spec, config, tasks, replay.opening_plies)


def test_a_one_link_replay_harvests_main_s_arrays_byte_for_byte(
    spec: dict, chain: tuple[Procedure, Procedure, Procedure]
) -> None:
    """Existing recipes are unchanged: E1's union, as the chained code plays it, is the
    union main's code played, every array byte for byte."""
    _one_recipe, two_recipe, _child = chain
    # Generation 2's recipe is E1's: one link, generation 0's games behind generation 1's.
    assert two_recipe.replay is not None and two_recipe.replay.prior is None
    # Labelled at 100 nodes, not 1,500: both harvests label alike, and that was 1.6 s (Q997b).
    labelled = two_recipe.with_(label_nodes=100)
    union = harvest(spec, self_play(spec, two_recipe, workers=1), labelled)
    main = harvest(
        spec,
        [*self_play(spec, two_recipe.with_(replay=None), workers=1),
         *_main_replay_games(spec, two_recipe)],
        labelled,
    )
    for column in ("offsets", "inputs", "sides", "targets", "game", "ply", "results", "values"):
        assert getattr(union, column).tobytes() == getattr(main, column).tobytes(), column
    assert union.digest == main.digest and union.games == main.games == 4 * 2 + 6


# ------------------------------------------------------------------ train --chain


def test_train_chain_names_f3b(tmp_path: Path, clean: None, capsys: pytest.CaptureFixture) -> None:
    """`train --parent <5fe1cb87d4c9> --replay <its .json> --chain` is F3b, by name; no game
    is played (an empty store makes --duel-only stop)."""
    from chess_variant_sandbox.cli import main

    parent = ROOT / "nets" / f"{GARDNER_HASH}-5fe1cb87d4c9"
    args = ["train", str(GARDNER), "--parent", f"{parent}.net", "--replay", f"{parent}.json",
            "--store", str(tmp_path), "--duel-only"]
    assert main([*args, "--chain"]) == 2
    captured = capsys.readouterr()
    assert f"{GARDNER_HASH}-{F3B[0]}.net" in captured.err
    assert "replay link 2: net=db5ef1bc938e" in captured.out
    assert main([*args, "--chain", "--init-seed", "2"]) == 2
    assert f"{GARDNER_HASH}-{F3B[2]}.net" in capsys.readouterr().err
    assert main(args) == 2
    assert f"{GARDNER_HASH}-{F3}.net" in capsys.readouterr().err
    # --replay-net registers any link's net by its bytes, and refuses a net no link names.
    one = ROOT / "nets" / f"{GARDNER_HASH}-e03365fe43e7.net"
    assert main([*args, "--chain", "--replay-net", str(one)]) == 2
    assert f"{GARDNER_HASH}-{F3B[0]}.net" in capsys.readouterr().err
    assert main([*args, "--chain", "--replay-net", str(one), f"{parent}.net"]) == 2
    assert "not the replay roster's net" in capsys.readouterr().err


def test_train_chain_and_harvest_link_refusals(
    tmp_path: Path, clean: None, capsys: pytest.CaptureFixture
) -> None:
    from chess_variant_sandbox.cli import main

    parent = ROOT / "nets" / f"{GARDNER_HASH}-5fe1cb87d4c9"
    assert main(["train", str(GARDNER), "--parent", f"{parent}.net", "--chain",
                 "--duel-only"]) == 2
    assert "--chain goes with --replay" in capsys.readouterr().err
    chained = ["train", str(GARDNER), "--parent", f"{parent}.net", "--replay",
               f"{parent}.json", "--chain", "--store", str(tmp_path)]
    assert main([*chained, "--harvest-link", "2", "--duel-only"]) == 2
    assert "--harvest-link goes with --harvest-only" in capsys.readouterr().err
    save = ["--harvest-only", "--save-data", str(tmp_path / "g.npz")]
    assert main([*chained, *save, "--harvest-link", "2", "--also-harvest", "C"]) == 2
    assert "without --also-harvest" in capsys.readouterr().err
    for link in ("3", "-1"):
        assert main([*chained, *save, "--harvest-link", link]) == 2
        assert f"--harvest-link {link} is not a part" in capsys.readouterr().err
    assert not (tmp_path / "g.npz").exists()
