"""`bridge.build_tables` is one flattener per rule, merged (CP1, `docs/plan-smells-python.md` PY8).

What the split has to keep true: no two flatteners write one key (a later one would
silently overwrite an earlier one's answer in the merge), each rule's inert rows name only
keys its own flattener writes, and `_INERT_DEFAULTS` is exactly their union. That the
tables themselves did not move is `tests/test_schema_nine.py::FROZEN`'s job, over every
committed and archived spec's `canonical_tables`.
"""

from __future__ import annotations

from typing import Any

import pytest

from chess_variant_sandbox.engine import bridge
from chess_variant_sandbox.engine.compile import compile_spec


@pytest.fixture(scope="module")
def flattened(chess_spec: dict[str, Any]) -> list[tuple[str, set[str], dict[str, Any]]]:
    compiled = compile_spec(chess_spec)
    return [
        (flatten.__name__, set(flatten(compiled)), inert)
        for flatten, inert in bridge._FLATTENERS
    ]


def test_no_two_flatteners_write_one_key(
    flattened: list[tuple[str, set[str], dict[str, Any]]],
) -> None:
    owner: dict[str, str] = {}
    clashes = []
    for name, keys, _inert in flattened:
        for key in keys:
            if key in owner:
                clashes.append(f"{key}: {owner[key]} and {name}")
            owner[key] = name
    assert not clashes, clashes


def test_each_rules_inert_rows_sit_beside_its_flattener(
    flattened: list[tuple[str, set[str], dict[str, Any]]],
) -> None:
    stray = {name: sorted(set(inert) - keys) for name, keys, inert in flattened}
    assert not any(stray.values()), stray


def test_inert_defaults_is_the_union_of_the_rows() -> None:
    rows = [inert for _flatten, inert in bridge._FLATTENERS]
    assert sum(len(inert) for inert in rows) == len(bridge._INERT_DEFAULTS)
    for inert in rows:
        for key, value in inert.items():
            assert bridge._INERT_DEFAULTS[key] == value


def test_build_tables_is_the_merge(
    chess_spec: dict[str, Any], flattened: list[tuple[str, set[str], dict[str, Any]]]
) -> None:
    tables = bridge.build_tables(compile_spec(chess_spec))
    assert set(tables) == set().union(*(keys for _name, keys, _inert in flattened))
