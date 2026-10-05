"""Q502i: the generic search repairs, held to their oracle and to the search before them.

Three repairs (`docs/reference/general-player.md` §2 F), each one bit on the agent's
twelfth native slot and one option in its name: ``keep=1`` (the transposition table kept
between moves), ``threats=1`` (threat-aware quiescence) and ``loud_drops=1``. E9 graded
them (§6.7): ``threats`` won and is on by default, the other two lost and are off, and the
player from before Q502i is spelled ``threats=0``. What this file holds:

* the default is the core's, and a player spelling every repair at its default is the
  unspelled player, bit for bit;
* the player spelling all three off is the search before Q502i;
* each repair is played identically by the core and by `agents/minimax.py`, move for move
  and node for node, on a game where it is not inert (the default is also held by every
  minimax arm of `test_backends.py`);
* each one changes play somewhere, so the identity above is not two copies of nothing;
* a repaired player's games do not depend on how many threads played them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import chess_variant_sandbox._core as _core
from chess_variant_sandbox.agents import native_spec
from chess_variant_sandbox.agents.minimax import (
    DEFAULT_REPAIRS,
    KEEP_TABLE,
    LOUD_DROPS,
    THREATS,
    repairs_of,
)
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament import TournamentConfig, run_tournament
from chess_variant_sandbox.tournament.play import play_game, play_game_python

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"
REPAIRS_SLOT = 11
# The search before Q502i, every repair spelled off.
BEFORE = "keep=0,loud_drops=0,threats=0"


def _engine(name: str) -> Engine:
    return Engine(load_spec(RULESETS / f"{name}.json"))


def _same(engine: Engine, agents: tuple[str, str], seed: int = 5) -> None:
    native = play_game(engine, agents, seed=seed)
    in_python = play_game_python(engine, agents, seed=seed)
    assert native.moves == in_python.moves
    assert (native.winner, native.reason) == (in_python.winner, in_python.reason)
    assert native.nodes == in_python.nodes
    assert native.depth_sum == in_python.depth_sum
    assert native.searches == in_python.searches


def test_the_default_is_the_cores_and_is_the_duels_winner() -> None:
    assert _core.DEFAULT_REPAIRS == DEFAULT_REPAIRS == THREATS
    plain = native_spec("minimax:nodes=150")
    spelled = native_spec("minimax:keep=0,loud_drops=0,nodes=150,threats=1")
    assert plain is not None and plain == spelled
    assert plain[REPAIRS_SLOT] == THREATS


def test_every_repair_off_is_the_search_before_them() -> None:
    before = native_spec(f"minimax:{BEFORE},nodes=150")
    assert before is not None and before[REPAIRS_SLOT] == 0
    for kind in ("random", "greedy", "mcts"):
        spec = native_spec(kind)
        assert spec is not None and spec[REPAIRS_SLOT] == 0, kind


def test_each_option_is_its_own_bit() -> None:
    off = {"keep": 0, "threats": 0, "loud_drops": 0}
    assert repairs_of({**off, "keep": 1}) == KEEP_TABLE
    assert repairs_of({**off, "threats": 1}) == THREATS
    assert repairs_of({**off, "loud_drops": 1}) == LOUD_DROPS
    assert repairs_of({"keep": 1, "threats": 1, "loud_drops": 1}) == 7
    assert repairs_of({}) == DEFAULT_REPAIRS


def test_a_repair_is_spelled_zero_or_one() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        native_spec("minimax:keep=2,nodes=100")


# Each repair on a game where it acts, against the search before them: the table on any
# game, checks on a royal game, arrivals on a race to a square, drops on a game with a
# hand.
ORACLE_ARMS = [
    # `dull` because its narrow tree lets 300 nodes reach far enough for the next search
    # to start inside this one's stored entries; on a wide board the table is inert at
    # any budget the oracle can afford.
    ("dull", "minimax:keep=1,loud_drops=0,nodes=300,threats=0"),
    ("gardner", "minimax:keep=0,loud_drops=0,nodes=150,threats=1"),
    ("kingofthehill", "minimax:keep=0,loud_drops=0,nodes=150,threats=1"),
    ("dobutsu", "minimax:keep=0,loud_drops=1,nodes=150,threats=0"),
    ("dobutsu", "minimax:keep=1,loud_drops=1,nodes=150,threats=1"),
]


def _budget(player: str) -> str:
    return player.split("nodes=")[1].split(",")[0]


@pytest.mark.parametrize(("name", "player"), ORACLE_ARMS, ids=lambda x: str(x))
def test_the_core_and_the_oracle_play_the_same_repaired_game(name: str, player: str) -> None:
    for seed in (0, 5):
        _same(_engine(name), (player, f"minimax:{BEFORE},nodes={_budget(player)}"), seed)


def test_the_threats_search_agrees_with_its_oracle_when_a_side_moves_twice() -> None:
    """A long turn is where negamax may not negate, and where a check given by the first
    move of a turn is a win the terminal check reports before the second: both paths the
    threats search walks and a normal turn never reaches."""
    spec = load_spec(RULESETS / "gardner.json")
    spec["schema_version"] = 4
    spec["rules"]["turn"] = {"moves": 2}
    engine = Engine(spec)
    for seed in (0, 5):
        _same(engine, ("minimax:nodes=150", f"minimax:{BEFORE},nodes=150"), seed)


@pytest.mark.parametrize(
    ("name", "player"),
    [
        # The table is inert at a few hundred nodes: a search two or three plies deep
        # stores nothing two plies on, where the next search starts.
        ("gardner", "minimax:keep=1,loud_drops=0,nodes=6400,threats=0"),
        ("gardner", "minimax:keep=0,loud_drops=0,nodes=400,threats=1"),
        ("dobutsu", "minimax:keep=0,loud_drops=1,nodes=400,threats=0"),
    ],
    ids=lambda x: str(x),
)
def test_each_repair_changes_play(name: str, player: str) -> None:
    engine = _engine(name)
    before = f"minimax:{BEFORE},nodes={_budget(player)}"
    differ = any(
        play_game(engine, (player, before), seed=seed).moves
        != play_game(engine, (before, before), seed=seed).moves
        for seed in range(4)
    )
    assert differ, f"{player} played the search before it on {name}"


def test_a_repaired_players_games_do_not_depend_on_threads() -> None:
    spec = load_spec(RULESETS / "dobutsu.json")
    player = "minimax:keep=1,loud_drops=1,nodes=300,threats=1"

    def played(workers: int) -> list[tuple[int, list[int]]]:
        result = run_tournament(
            spec,
            TournamentConfig(
                agents=(player, "minimax:nodes=300"),
                games_per_pairing=2,
                seed=7,
                workers=workers,
                include_self_play=False,
                collect_series=False,
            ),
        )
        return sorted((record.seed, list(record.moves)) for record in result.records)

    assert played(1) == played(4)
