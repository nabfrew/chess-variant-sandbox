"""Input layout 2 (Q1172, W4, `docs/plan-player-weaknesses.md` §T4): the purse, the bank and
the filled-parts mask, each only where the spec carries its rule.

Asserted against specs: which rule opens which segment, what each input reads off the
state, the core and the oracle to the bit along random games and at the counts no random
line reaches, every layout without the three rules unchanged, and a layout-1 store read
where nothing was added and refused where something was.
"""

from __future__ import annotations

import copy
import json
import struct
from pathlib import Path
from typing import Any

import pytest

import chess_variant_sandbox._core as _core
from chess_variant_sandbox.agents import nets
from chess_variant_sandbox.agents.nets import (
    BANK_CAP,
    INPUTS_VERSION,
    MAGIC,
    OLDEST_INPUTS_VERSION,
    PURSE_BUCKETS,
    PURSE_UNITS,
    Layout,
    Net,
    NetError,
    NetEvaluator,
    load_net,
    random_net,
)
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.engine import _pack
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.spec.schema import PURSE_MAX
from chess_variant_sandbox.tournament.play import play_game

ROOT = Path(__file__).resolve().parent.parent

# Every library spec that carries one of the three rules today: a purse with upkeep and
# income, a purse with bribes, a bank with any-order parts, a bank alone, parts alone.
CARRIERS = [
    "rulesets/hunger.json",
    "archive/the-free-companies.json",
    "rulesets/tessabi.json",
    "rulesets/crossings.json",
    "rulesets/drover.json",
]
# Specs without any of them: plain, drops, a round game, a game that rests.
PLAIN = [
    "rulesets/chess.json",
    "rulesets/crazyhouse.json",
    "rulesets/intransitive.json",
    "archive/wolves-bounty.json",
]


def _spec(path: str) -> dict[str, Any]:
    return load_spec(ROOT / path)


def _rules(spec: dict[str, Any]) -> tuple[bool, bool, int]:
    """What the spec says layout 2 must carry: a purse, a bank, and the parts it masks."""
    rules = spec.get("rules", {})
    reach = rules.get("reach") or {}
    turn = rules.get("turn") or {}
    parts = turn.get("parts") or []
    return (
        "purse" in rules,
        bool(reach.get("removes")),
        len(parts) if turn.get("order") == "any" else 0,
    )


def _round_width(layout: Layout) -> int:
    return 2 * layout.n_squares if layout.rounds else 0


def test_the_constants_are_the_cores() -> None:
    assert _core.NET_INPUTS_VERSION == INPUTS_VERSION == 2
    assert OLDEST_INPUTS_VERSION == 1


@pytest.mark.parametrize("path", CARRIERS + PLAIN)
def test_each_segment_is_there_exactly_where_its_rule_is(path: str) -> None:
    spec = _spec(path)
    has_purse, has_bank, n_parts = _rules(spec)
    engine = Engine(spec)
    layout = Layout.of(engine.spec)
    assert (layout.has_purse, layout.has_bank, layout.n_parts) == (has_purse, has_bank, n_parts)
    assert layout.purse == layout.bars + _round_width(layout)
    assert layout.bank - layout.purse == (2 * (PURSE_UNITS + PURSE_BUCKETS) if has_purse else 0)
    assert layout.parts - layout.bank == (2 * BANK_CAP if has_bank else 0)
    assert layout.n_inputs == layout.parts + n_parts
    assert layout.since_v1 == (has_purse or has_bank or n_parts > 0)
    assert engine.core.net_inputs() == layout.n_inputs, path


@pytest.mark.parametrize("path", PLAIN)
def test_a_spec_without_the_three_rules_keeps_layout_one(path: str) -> None:
    """Layout 1 ended at the round segments; nothing follows them where no rule is on."""
    layout = Layout.of(Engine(_spec(path)).spec)
    assert layout.n_inputs == layout.bars + _round_width(layout) == layout.purse
    assert not layout.since_v1


def test_fixed_order_parts_add_no_mask() -> None:
    """Under order "fixed" the mask is the moves left, which the turn segment carries."""
    spec = copy.deepcopy(_spec("rulesets/drover.json"))
    spec["rules"]["turn"]["order"] = "fixed"
    layout = Layout.of(Engine(spec).spec)
    assert layout.n_parts == 0 and not layout.since_v1


def _expected_tail(layout: Layout, state: GameState) -> list[int]:
    """Layout 2's inputs as §T4 states them, written out here rather than called."""
    out: list[int] = []
    if layout.has_purse:
        for player in range(2):
            base = layout.purse + player * (PURSE_UNITS + PURSE_BUCKETS)
            coins = state.purse[player]
            out += [base + k for k in range(PURSE_UNITS) if coins > k]
            out += [base + PURSE_UNITS + k for k in range(PURSE_BUCKETS)
                    if coins * PURSE_BUCKETS >= (k + 1) * PURSE_MAX]
    if layout.has_bank:
        for player in range(2):
            out += [layout.bank + player * BANK_CAP + k for k in range(BANK_CAP)
                    if state.banked[player] > k]
    out += [layout.parts + j for j in range(layout.n_parts) if state.parts >> j & 1]
    return out


def _probes(layout: Layout, start: GameState) -> list[GameState]:
    """The counts no random line is sure to reach: an empty purse, the unit cap and one
    past it, a full purse; a bank past its cap; every parts mask a turn can stand at (all
    parts filled is the next turn's start, which the core reads as no part filled)."""
    out = []
    if layout.has_purse:
        for purse in ((0, 1), (16, 15), (17, 50), (PURSE_MAX, 13)):
            out.append(start.with_(purse=purse))
    if layout.has_bank:
        for banked in ((0, 3), (16, 17)):
            out.append(start.with_(banked=banked))
    for mask in range((1 << layout.n_parts) - 1 if layout.n_parts else 1):
        out.append(start.with_(parts=mask))
    return out


@pytest.mark.parametrize("path", CARRIERS)
def test_the_core_and_the_oracle_read_the_new_planes_to_the_bit(path: str) -> None:
    engine = Engine(_spec(path))
    layout = Layout.of(engine.spec)
    record = play_game(engine, ("random", "random"), seed=5, collect_series=False)
    start = engine.initial_state()
    states = [start]
    for move in record.moves:
        states.append(engine.apply(states[-1], move))
    blob = random_net(layout.n_inputs, seed=7)
    oracle = NetEvaluator(engine, blob, ())
    answers = engine.core.net_line(_pack(start), blob, list(record.moves))
    for ply, ((raw, inputs, fresh), state) in enumerate(zip(answers, states, strict=True)):
        assert inputs == oracle.layout.features(state), (path, ply)
        assert [i for i in inputs if i >= layout.purse] == _expected_tail(layout, state)
        assert raw == fresh == oracle.raw(state), (path, ply)
    for probe in _probes(layout, start):
        ((raw, inputs, fresh),) = engine.core.net_line(_pack(probe), blob, [])
        assert inputs == oracle.layout.features(probe), path
        assert [i for i in inputs if i >= layout.purse] == _expected_tail(layout, probe)
        assert raw == fresh == oracle.raw(probe), path


def test_the_purse_reads_units_then_buckets() -> None:
    """Written out once: 17 coins light all 16 units and the first bucket (17 x 8 >= 99)."""
    engine = Engine(_spec("rulesets/hunger.json"))
    layout = Layout.of(engine.spec)
    state = engine.initial_state().with_(purse=(17, 0))
    tail = [i - layout.purse for i in layout.features(state) if i >= layout.purse]
    assert tail == [*range(16), 16]


def _as_version(blob: bytes, version: int) -> bytes:
    return MAGIC + struct.pack("<I", version) + blob[12:]


def test_a_layout_one_store_plays_where_nothing_was_added() -> None:
    engine = Engine(_spec("rulesets/chess.json"))
    n_inputs = Layout.of(engine.spec).n_inputs
    new = random_net(n_inputs, seed=11)
    old = _as_version(new, 1)
    assert Net.parse(old).version == 1
    start = engine.initial_state()
    assert engine.core.net_line(_pack(start), old, []) == engine.core.net_line(
        _pack(start), new, []
    )
    assert NetEvaluator(engine, old, ()).raw(start) == NetEvaluator(engine, new, ()).raw(start)


@pytest.mark.parametrize("path", CARRIERS)
def test_a_layout_one_store_is_refused_where_a_segment_was_added(path: str) -> None:
    engine = Engine(_spec(path))
    old = _as_version(random_net(Layout.of(engine.spec).n_inputs, seed=11), 1)
    with pytest.raises(ValueError, match="layout 2 added"):
        engine.core.net_line(_pack(engine.initial_state()), old, [])
    with pytest.raises(NetError, match="layout 2 added"):
        NetEvaluator(engine, old, ())


@pytest.mark.parametrize("version", [0, 3])
def test_a_store_of_an_unread_version_is_refused(version: int) -> None:
    engine = Engine(_spec("rulesets/chess.json"))
    blob = _as_version(random_net(Layout.of(engine.spec).n_inputs, seed=11), version)
    with pytest.raises(ValueError, match="refused rather than reinterpreted"):
        engine.core.net_line(_pack(engine.initial_state()), blob, [])
    with pytest.raises(NetError, match=f"input layout {version}"):
        Net.parse(blob)


@pytest.mark.parametrize("version", [0, 1, 2, 3])
def test_a_store_record_is_read_at_the_versions_the_core_reads(
    tmp_path: Path, version: int
) -> None:
    weights = _as_version(random_net(4, seed=1), max(min(version, 2), 1))
    (tmp_path / "abc-proc.net").write_bytes(weights)
    (tmp_path / "abc-proc.json").write_text(json.dumps({
        "procedure": "proc", "ruleset_hash": "abc", "inputs_version": version,
        "verdict": "refused", "weights_sha256": "",
    }), encoding="utf-8")
    try:
        if version in (1, 2):
            record = load_net("abc", "proc", tmp_path)
            assert record is not None and record.verdict == "refused"
        else:
            with pytest.raises(NetError, match="input layout"):
                load_net("abc", "proc", tmp_path)
    finally:
        nets.unregister_net("proc", "abc")
