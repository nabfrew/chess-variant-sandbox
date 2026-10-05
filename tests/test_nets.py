"""The learned evaluator (Q502e): the core's integer inference against its Python oracle.

`rust/src/net.rs` computes; `agents/nets.py` is the declared oracle. They must agree to
the bit on random nets and positions -- the incremental accumulator included -- and a
search on each must play the same game. A `net=` player whose store is absent, refused or
unreadable refuses to play (Q709). Design: `docs/reference/learned-evaluator.md`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import chess_variant_sandbox._core as _core
from chess_variant_sandbox.agents import build_agent, native_spec, nets
from chess_variant_sandbox.agents.nets import (
    ACCEPTED,
    REFUSED,
    Layout,
    Net,
    NetEvaluator,
    NetRecord,
    load_net,
    random_net,
    register_net,
    save_net,
    unregister_net,
)
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.engine import _pack
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.play import play_game, play_game_python

ROOT = Path(__file__).resolve().parent.parent
PROCEDURE = "0e7000000001"

# One ruleset per kind of state the layout reads: plain, facing, drops, territory,
# check counts, scoring, en passant, wild pieces, a hexagon.
STATES = [
    "rulesets/chess.json",
    "rulesets/makruk.json",
    "rulesets/crazyhouse.json",
    "rulesets/intransitive.json",
    "rulesets/threecheck.json",
    "rulesets/rps-v5.json",
    "rulesets/berolina.json",
    "archive/wolves-bounty.json",
    "archive/hex-glinski-hexagon.json",
]


def _engine(path: str) -> Engine:
    return Engine(load_spec(ROOT / path))


def _turned(path: str) -> Engine:
    """Two moves a turn, so the turn counter has something to say."""
    spec: dict[str, Any] = load_spec(ROOT / path)
    spec["schema_version"] = max(4, int(spec.get("schema_version", 1)))
    spec["rules"]["turn"] = {"moves": 2}
    return Engine(spec)


def _line(engine: Engine, seed: int) -> tuple[GameState, list[int], list[GameState]]:
    """A random game's start, moves and every position along it."""
    record = play_game(engine, ("random", "random"), seed=seed, collect_series=False)
    state = engine.initial_state()
    states = [state]
    for move in record.moves:
        state = engine.apply(state, move)
        states.append(state)
    return engine.initial_state(), list(record.moves), states


@pytest.fixture
def registered() -> Iterator[None]:
    yield
    for key in list(nets._NETS):
        if key[0] == PROCEDURE:
            nets._NETS.pop(key)


def test_the_constants_and_the_layout_are_the_cores() -> None:
    assert _core.NET_INPUTS_VERSION == nets.INPUTS_VERSION
    for path in STATES:
        engine = _engine(path)
        assert engine.core.net_inputs() == Layout.of(engine.spec).n_inputs, path


@pytest.mark.parametrize("path", STATES)
def test_the_core_and_the_oracle_agree_to_the_bit_along_a_game(path: str) -> None:
    """Every position of a random game: inputs, the carried accumulator, a fresh one."""
    engine = _engine(path)
    n_inputs = Layout.of(engine.spec).n_inputs
    start, moves, states = _line(engine, seed=3)
    for seed in (1, 2):
        blob = random_net(n_inputs, seed=seed)
        oracle = NetEvaluator(engine, blob, ())
        answers = engine.core.net_line(_pack(start), blob, moves)
        assert len(answers) == len(states)
        for ply, ((raw, inputs, fresh), state) in enumerate(zip(answers, states, strict=True)):
            assert inputs == oracle.layout.features(state), (path, ply)
            expected = oracle.raw(state)
            assert (raw, fresh) == (expected, expected), (path, seed, ply)
        # The counters no random line reaches: a wild-piece bounty, a check count past
        # its cap, a clock past its limit. Fractional points floor, as both sides say.
        probe = start.with_(points=(2.5, 17.0), checks=(1, 9), halfmove=97, ply=400)
        ((raw, inputs, fresh),) = engine.core.net_line(_pack(probe), blob, [])
        assert inputs == oracle.layout.features(probe)
        assert raw == fresh == oracle.raw(probe), path


SEGMENTS = ("pieces", "unmoved", "ep", "colour", "hands", "checks", "points", "halfmove", "ply")


def test_the_agreement_lines_reach_every_segment() -> None:
    """Otherwise the bit-for-bit test proves less than it says. (`turn` is the next test's.)"""
    hit: set[str] = set()
    for path in STATES:
        engine = _engine(path)
        layout = Layout.of(engine.spec)
        starts = [0, layout.unmoved, layout.ep, layout.colour, layout.hands, layout.checks,
                  layout.points, layout.halfmove, layout.ply, layout.turn]
        states = [*_line(engine, seed=3)[2], engine.initial_state().with_(points=(2.0, 0.0))]
        for state in states:
            for feature in layout.features(state):
                index = max(i for i, start in enumerate(starts[:-1]) if feature >= start)
                hit.add(SEGMENTS[index])
    assert hit == set(SEGMENTS)


def test_the_turn_counter_and_both_heads_are_read() -> None:
    engine = _turned("rulesets/gardner.json")
    layout = Layout.of(engine.spec)
    start, moves, states = _line(engine, seed=4)
    assert any(layout.turn in layout.features(s) for s in states)
    assert {s.side_to_move for s in states} == {0, 1}
    blob = random_net(layout.n_inputs, seed=9)
    oracle = NetEvaluator(engine, blob, ())
    answers = engine.core.net_line(_pack(start), blob, moves)
    assert [a[0] for a in answers] == [oracle.raw(s) for s in states]


def test_every_clip_is_exercised_by_the_random_nets() -> None:
    """Otherwise agreement proves less than it says."""
    engine = _engine("rulesets/chess.json")
    net = Net.parse(random_net(Layout.of(engine.spec).n_inputs, seed=1))
    acc = net.accumulate(Layout.of(engine.spec).features(engine.initial_state()))
    assert min(acc) < 0 < max(acc) and max(acc) > nets.ACTIVATION_MAX


def test_a_damaged_or_foreign_net_is_refused_at_the_boundary() -> None:
    chess = _engine("rulesets/chess.json")
    foreign = random_net(Layout.of(_engine("rulesets/gardner.json").spec).n_inputs, seed=1)
    with pytest.raises(ValueError, match="trained on another game"):
        chess.core.net_line(_pack(chess.initial_state()), foreign, [])
    good = random_net(Layout.of(chess.spec).n_inputs, seed=1)
    with pytest.raises(ValueError, match="truncated"):
        chess.core.net_line(_pack(chess.initial_state()), good[:-1], [])
    with pytest.raises(nets.NetError):
        NetEvaluator(chess, foreign, ())


def _net_player(engine: Engine, seed: int, verdict: str = ACCEPTED) -> None:
    blob = random_net(Layout.of(engine.spec).n_inputs, seed=seed, accumulator=16, hidden1=8)
    register_net(NetRecord(PROCEDURE, engine.spec.hash, verdict, blob))


@pytest.mark.parametrize(
    "agents",
    [
        (f"minimax:net={PROCEDURE},nodes=150", "greedy"),
        (f"greedy:net={PROCEDURE}", f"minimax:net={PROCEDURE},nodes=120"),
    ],
    ids=["minimax", "greedy"],
)
def test_a_net_player_plays_the_same_game_in_the_core_and_the_oracle(
    agents: tuple[str, str], registered: None
) -> None:
    engine = _engine("rulesets/gardner.json")
    _net_player(engine, seed=5)
    native = play_game(engine, agents, seed=5)
    in_python = play_game_python(engine, agents, seed=5)
    assert native.moves == in_python.moves
    assert (native.winner, native.reason) == (in_python.winner, in_python.reason)
    assert native.nodes == in_python.nodes
    assert native.depth_sum == in_python.depth_sum
    # And the net is what played: the built-in objective plays another game.
    plain = play_game(engine, tuple(a.replace(f"net={PROCEDURE},", "").replace(
        f":net={PROCEDURE}", "") for a in agents), seed=5)  # type: ignore[arg-type]
    assert plain.moves != native.moves


@pytest.mark.parametrize("verdict", [None, REFUSED], ids=["absent", "refused"])
def test_an_absent_or_refused_net_refuses_to_play(
    verdict: str | None, registered: None
) -> None:
    """Q709: a `net=` player with no net refuses, in a sentence, wherever it is built --
    the core's parameters (the schedule, the practice player, the site) and the Python
    oracle -- and never plays the built-in objective under the learned player's name."""
    engine = _engine("rulesets/gardner.json")
    if verdict is not None:
        _net_player(engine, seed=5, verdict=verdict)
    agent = f"minimax:net={PROCEDURE},nodes=300"
    why = "no net is stored" if verdict is None else "refused, not accepted"
    for build in (
        lambda: native_spec(agent, engine.spec.hash),
        lambda: native_spec(f"greedy:net={PROCEDURE}", engine.spec.hash),
        lambda: build_agent(agent, engine),
        lambda: play_game(engine, (agent, "greedy"), seed=2),
        lambda: play_game_python(engine, (agent, "greedy"), seed=2),
    ):
        with pytest.raises(nets.NetError, match=why) as error:
            build()
        assert "built-in evaluation" in str(error.value) and PROCEDURE in str(error.value)
    # The permissive reading stays for the basis stamp: which player would the name be.
    assert nets.net_bytes({"net": PROCEDURE}, engine.spec.hash) == b""


def test_a_net_beside_another_objective_is_refused(registered: None) -> None:
    engine = _engine("rulesets/gardner.json")
    _net_player(engine, seed=5)
    with pytest.raises(ValueError, match="at once"):
        native_spec(f"minimax:net={PROCEDURE},fit=98f172aeac49,nodes=100", engine.spec.hash)
    with pytest.raises(nets.NetError, match="which ruleset"):
        native_spec(f"minimax:net={PROCEDURE},nodes=100")


def test_a_store_round_trips_and_is_checked_against_its_record(tmp_path: Path) -> None:
    engine = _engine("rulesets/gardner.json")
    blob = random_net(Layout.of(engine.spec).n_inputs, seed=7)
    record = NetRecord(PROCEDURE, engine.spec.hash, ACCEPTED, blob, build="x")
    try:
        meta = save_net(record, tmp_path)
        loaded = load_net(engine.spec.hash, PROCEDURE, tmp_path)
        assert loaded == record and loaded.playing == blob
        assert load_net(engine.spec.hash, "ffffffffffff", tmp_path) is None
        meta.with_suffix(".net").write_bytes(blob[:-1] + b"\x00")
        with pytest.raises(nets.NetError, match="does not hash"):
            load_net(engine.spec.hash, PROCEDURE, tmp_path)
    finally:
        unregister_net(PROCEDURE, engine.spec.hash)
