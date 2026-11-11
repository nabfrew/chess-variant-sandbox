"""T7's P6 for the players (`docs/plan-player-weaknesses.md` §T7, Q1175b): the `portal_reach` bit.

In a portal game a minimax player on the built-in objective or a net plays with the derived table
counting each priced type's capture reach through the ends where they start
(`eval.rs::portal_reach_values`, oracle `agents/portal_reach.py`). Everywhere else it is unread.

Specs, not chess: every archived portal game (pairs, passages, carries, a map with a way out only,
moving doors, a straight-back bar, a seam, folds), a hex pair and a fold on the hexagon, a region
that blocks an exit; the committed rulesets (none has portals) for the inertness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import _core
from chess_variant_sandbox.agents import build_agent
from chess_variant_sandbox.agents.minimax import (
    DEFAULT_REPAIRS,
    PORTAL_REACH,
    REPAIR_OPTIONS,
    MinimaxAgent,
)
from chess_variant_sandbox.agents.nets import NetEvaluator
from chess_variant_sandbox.agents.portal_reach import portal_extra, portal_reach_values
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.play import eval_values, play_game, play_game_python
from tests.test_hex_portals import hex_portals
from tests.test_portal_exit_claims import _spec as region_spec

ROOT = Path(__file__).resolve().parents[1]
PORTAL_GAMES = sorted(
    path for path in (ROOT / "archive").glob("*.json")
    if "portals" in load_spec(path).get("rules", {})
)
FERRY = ROOT / "archive" / "ferry.json"
WORMHOLE = ROOT / "archive" / "wormhole.json"
CREASE = ROOT / "archive" / "crease.json"


def _engine(path: Path, plies: int) -> Engine:
    spec = load_spec(path)
    spec["rules"]["max_plies"] = plies
    return Engine(spec)


def _arms() -> list[tuple[str, dict[str, Any]]]:
    arms = [(path.stem, load_spec(path)) for path in PORTAL_GAMES]
    arms += [("hex pair", hex_portals()), ("hex fold", hex_portals("fold"))]
    arms += [
        ("region", region_spec("momentum", region=["a1", "b1", "a2", "c1", "d1", "e1"])),
        ("region fold", region_spec("fold", region=["a1", "b1", "a2", "c1", "d1", "e1"])),
    ]
    return arms


def test_the_bit_is_its_own_and_spelled_portal_reach() -> None:
    assert REPAIR_OPTIONS["portal_reach"] == PORTAL_REACH == 524288
    assert _core.DEFAULT_REPAIRS == DEFAULT_REPAIRS


def test_every_portal_rule_is_reached() -> None:
    kinds: set[str] = set()
    for path in PORTAL_GAMES:
        portals = Engine(load_spec(path)).spec.portals
        assert portals is not None
        kinds.update(portals.kinds)
        if portals.no_back:
            kinds.add("no_back")
    assert len(PORTAL_GAMES) >= 10
    assert {"momentum", "fold", "no_back"} <= kinds


@pytest.mark.parametrize("name,spec", _arms(), ids=[name for name, _ in _arms()])
def test_core_and_oracle_read_one_table(name: str, spec: dict[str, Any]) -> None:
    """Bit for bit: the core's table is the oracle's, in every portal game and constructed arm."""
    engine = Engine(spec)
    derived = eval_values(engine.spec)
    oracle = portal_reach_values(engine.spec, derived)
    assert engine.core is not None
    assert engine.core.portal_reach_values(list(derived)) == list(oracle), name


@pytest.mark.parametrize("name,spec", _arms(), ids=[name for name, _ in _arms()])
def test_the_table_only_adds_and_spares_what_no_door_reaches(name: str, spec: dict[str, Any]) -> None:
    """A priced type no transit reaches keeps its value bit for bit, a royal stays 0, and a type
    a door reaches gains; somewhere in the arm a door reaches something."""
    engine = Engine(spec)
    derived = eval_values(engine.spec)
    table = portal_reach_values(engine.spec, derived)
    extra = portal_extra(engine.spec)
    assert any(extra), name
    for piece_type, (before, after) in enumerate(zip(derived, table, strict=True)):
        if piece_type in engine.spec.royal_types or not extra[piece_type]:
            assert after == before, (name, piece_type)
        else:
            assert after > before, (name, piece_type)


def test_a_door_closed_to_a_player_counts_for_the_other_alone() -> None:
    """Shut wormhole's passage is closed to the second player: each type gains half of what it
    gains in wormhole, the same pair open to both."""
    shut = portal_extra(Engine(load_spec(ROOT / "archive" / "shut-wormhole.json")).spec)
    open_ = portal_extra(Engine(load_spec(WORMHOLE)).spec)
    assert any(open_)
    assert shut == tuple(value / 2 for value in open_)


def test_a_region_that_blocks_the_exit_adds_nothing() -> None:
    """The rook's region keeps it off the exit's side: its transit takes nothing (SR5i's stand)."""
    blocked = region_spec("momentum", region=["a1", "b1", "a2", "c1", "d1", "e1"])
    free = region_spec("momentum")
    rook = Engine(free).spec.piece_ids.index("R")
    assert portal_extra(Engine(free).spec)[rook] > portal_extra(Engine(blocked).spec)[rook]


def test_without_portals_the_table_is_the_derived_one() -> None:
    engine = Engine(load_spec(ROOT / "rulesets" / "chess.json"))
    derived = eval_values(engine.spec)
    assert engine.core is not None
    assert engine.core.portal_reach_values(list(derived)) == list(derived)
    assert portal_reach_values(engine.spec, derived) == tuple(derived)


def test_the_oracle_player_reads_the_table() -> None:
    engine = Engine(load_spec(FERRY))
    agent = build_agent("minimax:nodes=50,portal_reach=1", engine)
    plain = build_agent("minimax:nodes=50", engine)
    assert isinstance(agent, MinimaxAgent) and isinstance(plain, MinimaxAgent)
    assert tuple(agent.evaluator.values) == portal_reach_values(engine.spec, eval_values(engine.spec))
    assert tuple(plain.evaluator.values) == tuple(eval_values(engine.spec))


def test_a_net_player_orders_by_the_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """A net's search orders captures by the derived table, so the bit hands it the portal one."""
    engine = Engine(load_spec(FERRY))
    built: list[tuple[float, ...]] = []

    def fake(self: NetEvaluator, engine: Engine, net: bytes, values: tuple[float, ...]) -> None:
        built.append(tuple(values))
        self._values = values

    monkeypatch.setattr(NetEvaluator, "__init__", fake)
    agent = MinimaxAgent(engine, nodes=50, net=b"stand-in", repairs=DEFAULT_REPAIRS | PORTAL_REACH)
    assert isinstance(agent.evaluator, NetEvaluator)
    assert built[-1] == portal_reach_values(engine.spec, eval_values(engine.spec))


@pytest.mark.parametrize(
    "path", [FERRY, WORMHOLE, CREASE, ROOT / "archive" / "carousel.json"], ids=lambda path: path.stem
)
def test_oracle_and_core_agree_under_portal_reach(path: Path) -> None:
    """Move for move and node for node, the candidate against the player without the bit."""
    engine = _engine(path, 24)
    agents = ("minimax:nodes=80,portal_reach=1", "minimax:nodes=80,portal_reach=0")
    native = play_game(engine, agents, seed=1)
    in_python = play_game_python(engine, agents, seed=1)
    assert native.moves == in_python.moves
    assert (native.winner, native.reason) == (in_python.winner, in_python.reason)
    assert native.nodes == in_python.nodes
    assert native.depth_sum == in_python.depth_sum


def test_the_core_reads_the_bit_on_ferry() -> None:
    engine = _engine(FERRY, 40)
    played = [
        play_game(engine, (f"minimax:nodes=400,portal_reach={bit}", "minimax:nodes=400"), seed=2)
        for bit in (1, 0)
    ]
    assert (played[0].moves, played[0].nodes) != (played[1].moves, played[1].nodes)


def test_every_game_without_portals_plays_as_before() -> None:
    """The bit is unread without portals: in every committed ruleset, `portal_reach=1` and
    `portal_reach=0` play the same game in the core, move for move and node for node."""
    played = 0
    for path in sorted((ROOT / "rulesets").glob("*.json")):
        spec = load_spec(path)
        spec.setdefault("rules", {})["max_plies"] = 6
        engine = Engine(spec)
        if engine.simultaneous or engine.spec.spy is not None:
            continue  # minimax plays no round
        assert engine.spec.portals is None, path.name
        records = [
            play_game(engine, (f"minimax:nodes=40,portal_reach={bit}", "minimax:nodes=40"), seed=3)
            for bit in (1, 0)
        ]
        assert (records[0].moves, records[0].nodes, records[0].depth_sum) == (
            records[1].moves, records[1].nodes, records[1].depth_sum
        ), path.name
        played += 1
    assert played > 30
