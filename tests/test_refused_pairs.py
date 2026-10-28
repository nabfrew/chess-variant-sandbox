"""The refused pairs' table (`spec/refusals.py`, CP4): one statement of each pair, by its id.

The validator emits every row through `refusal`, `rust_supports` names its portal holds by
the same ids, and a test asks `refused_ids` instead of matching prose. The plan's own rows
are held to the table in `docs_tests/test_refused_pair_ids.py`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from chess_variant_sandbox.engine.bridge import PORTAL_CORE_HOLDS
from chess_variant_sandbox.spec import refusals
from chess_variant_sandbox.spec.refusals import REFUSALS, refusal, refusal_rows, refused_ids

SCHEMA = Path(refusals.__file__).with_name("schema.py")


def test_each_row_is_one_pair_with_its_own_words() -> None:
    assert len({(row.id, row.second) for row in REFUSALS}) == len(REFUSALS)
    assert len({row.message for row in REFUSALS}) == len(REFUSALS)
    for row in REFUSALS:
        assert re.fullmatch(r"[A-Z]{1,2}\d+", row.id), row.id
        assert row.first and row.second and row.first != row.second, row


def test_every_id_is_emitted_by_the_validator_and_no_message_is_written_twice() -> None:
    source = SCHEMA.read_text(encoding="utf-8")
    emitted = set(re.findall(r'refusal\("([A-Z]{1,2}\d+)"', source))
    # The loops that emit by a key's id hold it as a string beside the key.
    held = set(re.findall(r'"([A-Z]{1,2}\d+)"', source))
    assert {row.id for row in REFUSALS} <= emitted | held
    for row in REFUSALS:
        # A message's distinctive tail never reappears as a literal in the validator.
        tail = row.message[-60:]
        assert tail not in source, (row.id, tail)


def test_a_message_reads_back_as_its_id() -> None:
    for row in REFUSALS:
        assert refused_ids([row.message, "something else entirely"]) == {row.id}
        assert refusal(row.id, row.second) == row.message
    assert refused_ids([]) == set()


def test_an_id_of_several_rows_needs_its_second_key() -> None:
    assert len(refusal_rows("K1")) == 3
    with pytest.raises(KeyError):
        refusal("K1")
    with pytest.raises(KeyError):
        refusal("N20")  # lifted by SR2: not in the table
    assert refusal("K1", "rules.forbid_check").startswith('rules.duck beside "forbid_check"')


def test_the_bridges_portal_holds_are_rows_of_the_table() -> None:
    for ids, _ in PORTAL_CORE_HOLDS:
        for rid in ids:
            rows = refusal_rows(rid)
            assert rows, rid
            assert any("rules.portals" in (row.first, row.second) for row in rows), rid
