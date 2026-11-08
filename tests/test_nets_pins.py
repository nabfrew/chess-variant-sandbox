"""Pins on `agents/nets.py`'s edges (Q1152, the `agents/` mutation group).

The width bounds a store is read under, on each side of each bound; the turn and rest
segments of the input layout on a state that reaches them; the store's JSON as written.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from chess_variant_sandbox.agents.nets import (
    INPUTS_VERSION,
    MAGIC,
    Layout,
    Net,
    NetError,
    NetRecord,
    encode,
    save_net,
)


def _header(acc: int, h1: int, h2: int) -> bytes:
    return MAGIC + struct.pack("<5I", INPUTS_VERSION, 1, acc, h1, h2)


@pytest.mark.parametrize(
    ("acc", "h1", "h2", "message"),
    [
        (2048, 1, 1, "truncated"),
        (2049, 1, 1, "out of range"),
        (1, 128, 1, "truncated"),
        (1, 129, 1, "out of range"),
        (1, 1, 128, "truncated"),
        (0, 1, 1, "out of range"),
        (1, 0, 1, "out of range"),
        (1, 1, 0, "out of range"),
    ],
)
def test_the_widths_a_store_may_declare(acc: int, h1: int, h2: int, message: str) -> None:
    with pytest.raises(NetError, match=message):
        Net.parse(_header(acc, h1, h2))


def _smallest() -> bytes:
    head = ([3], [4], [5], [6], [7], 8)
    return encode(1, 1, 1, 1, [1], [2], [head, head])


def test_a_net_one_wide_everywhere_parses() -> None:
    net = Net.parse(_smallest())
    assert (net.n_inputs, net.accumulator, net.hidden1, net.hidden2) == (1, 1, 1, 1)
    assert (net.w0, net.b0, net.heads[1].b3) == ((1,), (2,), 8)


def test_the_turn_and_rest_segments_of_a_resting_layout() -> None:
    spec = SimpleNamespace(
        n_squares=4, n_types=1, n_owners=2, n_facings=1, simultaneous=None,
        rest=True, rest_layers=2, no_progress_draw=0, max_plies=0,
    )
    layout = Layout.of(spec)
    assert (layout.turn, layout.rest, layout.n_inputs) == (120, 124, 140)
    state = SimpleNamespace(
        board=[0, 0, 0, 0], unmoved=0, ep_target=-1, colour=[0, 0, 0, 0],
        hands=[[0], [0]], checks=[0, 0], points=[0.0, 0.0], halfmove=0, ply=0,
        remaining=4, initiative=-1,
        # The square one past the board comes first and is dropped; the rest are kept.
        rest=[[(4, 2), (1, 1)], [(2, 2)]],
    )
    assert layout.features(state) == [120, 121, 122, 125, 138]  # type: ignore[arg-type]


def test_a_store_is_written_as_two_space_json(tmp_path: Path) -> None:
    weights = _smallest()
    record = NetRecord(
        procedure="000000000abc", ruleset_hash="h", verdict="accepted", weights=weights
    )
    meta = save_net(record, tmp_path)
    digest = hashlib.sha256(weights).hexdigest()
    assert meta.read_text(encoding="utf-8") == (
        "{\n"
        '  "procedure": "000000000abc",\n'
        '  "ruleset_hash": "h",\n'
        f'  "inputs_version": {INPUTS_VERSION},\n'
        '  "verdict": "accepted",\n'
        f'  "weights_sha256": "{digest}",\n'
        '  "build": ""\n'
        "}\n"
    )


def test_a_rest_entry_off_its_layers_is_dropped() -> None:
    spec = SimpleNamespace(
        n_squares=4, n_types=1, n_owners=2, n_facings=1, simultaneous=None,
        rest=True, rest_layers=2, no_progress_draw=0, max_plies=0,
    )
    state = SimpleNamespace(
        board=[0, 0, 0, 0], unmoved=0, ep_target=-1, colour=[0, 0, 0, 0],
        hands=[[0], [0]], checks=[0, 0], points=[0.0, 0.0], halfmove=0, ply=0,
        remaining=1, initiative=-1, rest=[[(1, 0), (2, 3)], [(3, 1)]],
    )
    assert Layout.of(spec).features(state) == [124 + 1 * 4 + 3]  # type: ignore[arg-type]


def test_a_random_net_is_this_net() -> None:
    from chess_variant_sandbox.agents.nets import random_net

    blob = random_net(2, 3, accumulator=2, hidden1=2, hidden2=3)
    assert hashlib.sha256(blob).hexdigest() == (
        "e7dc04b1ca7b5f33cd292204e46bceece700b1178ec43003133feb04dc11b5d4"
    )


def test_a_store_filed_under_another_name_is_not_read(tmp_path: Path) -> None:
    """``h-other.json`` says it is ``proc``'s store and that ``proc`` is its parent: read,
    it would make ``proc`` a parent and leave no leaf."""
    import json

    from chess_variant_sandbox.agents.nets import rated_net, unregister_net

    procedure = "000000000abc"
    meta = save_net(
        NetRecord(procedure=procedure, ruleset_hash="h", verdict="accepted",
                  weights=_smallest()),
        tmp_path,
    )
    data = json.loads(meta.read_text(encoding="utf-8"))
    meta.write_text(json.dumps({**data, "kept": True}), encoding="utf-8")
    stray = {**data, "kept": True, "lineage": {"parent": procedure}}
    (tmp_path / "h-other.json").write_text(json.dumps(stray), encoding="utf-8")
    try:
        assert rated_net("h", tmp_path) == procedure
    finally:
        unregister_net(procedure, "h")


def test_a_layout_has_slots() -> None:
    spec = SimpleNamespace(
        n_squares=4, n_types=1, n_owners=2, n_facings=1, simultaneous=None,
        rest=False, rest_layers=0, no_progress_draw=0, max_plies=0,
    )
    assert not hasattr(Layout.of(spec), "__dict__")


def test_a_store_is_saved_into_a_root_that_does_not_exist_yet(tmp_path: Path) -> None:
    record = NetRecord(
        procedure="000000000abc", ruleset_hash="h", verdict="refused", weights=_smallest()
    )
    meta = save_net(record, tmp_path / "a" / "b")
    assert meta == tmp_path / "a" / "b" / "h-000000000abc.json"
    assert meta.with_suffix(".net").read_bytes() == _smallest()


def test_a_clock_with_no_limit_has_no_inputs() -> None:
    from chess_variant_sandbox.agents.nets import _clock

    assert _clock(10, 5, 0) == []
    assert _clock(10, 5, 8) == [10, 11, 12, 13, 14]


def test_a_random_net_reaches_the_top_of_its_output_bias() -> None:
    """Seed 3767 draws the output bias's upper end, 5000, for the second head."""
    from chess_variant_sandbox.agents.nets import random_net

    blob = random_net(1, 3767, accumulator=1, hidden1=1, hidden2=1)
    net = Net.parse(blob)
    assert (net.heads[0].b3, net.heads[1].b3) == (-3973, 5000)
    assert hashlib.sha256(blob).hexdigest() == (
        "c26b9dedb5fc22f015864447bebecb291e8954859a9894eef9f3738385062157"
    )


def test_a_bar_one_past_the_board_is_dropped() -> None:
    spec = SimpleNamespace(
        n_squares=4, n_types=1, n_owners=2, n_facings=1,
        simultaneous=SimpleNamespace(tie="none"),
        rest=False, rest_layers=0, no_progress_draw=0, max_plies=0,
    )
    layout = Layout.of(spec)
    assert (layout.void, layout.bars, layout.n_inputs) == (124, 132, 140)
    state = SimpleNamespace(
        board=[0, 0, 0, 0], unmoved=0, ep_target=-1, colour=[0, 0, 0, 0],
        hands=[[0], [0]], checks=[0, 0], points=[0.0, 0.0], halfmove=0, ply=0,
        remaining=1, void_run=[-1, -1], bars=[(4, 1)],
    )
    assert layout.features(state) == [137]  # type: ignore[arg-type]


def test_a_head_has_slots() -> None:
    assert not hasattr(Net.parse(_smallest()).heads[0], "__dict__")


def test_rated_net_reads_a_store_without_an_acceptance_as_score_zero(tmp_path: Path) -> None:
    """Two kept leaves at score 0: the tie goes to the later procedure, which is the one
    whose store records no acceptance at all."""
    import json

    from chess_variant_sandbox.agents.nets import rated_net, unregister_net

    late, early = "000000000abd", "000000000abc"
    for procedure, extra in ((late, {}), (early, {"acceptance": {"score": 0.0}})):
        meta = save_net(
            NetRecord(procedure=procedure, ruleset_hash="h", verdict="accepted",
                      weights=_smallest()),
            tmp_path,
        )
        data = json.loads(meta.read_text(encoding="utf-8"))
        meta.write_text(json.dumps({**data, "kept": True, **extra}), encoding="utf-8")
    try:
        assert rated_net("h", tmp_path) == late
    finally:
        unregister_net(late, "h")
        unregister_net(early, "h")
