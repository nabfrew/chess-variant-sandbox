"""Every lifted pair has been put to Fairy-Stockfish (Q1218).

`spec/refusals.toml`'s `[[lifted]]` rows are pairs the rule language once refused and now plays.
U2 lifted `gating.duck` and the translator wrote it, though Fairy-Stockfish offers no gate
beside the duck. `fairy.lifted.TREATMENT` says how Fairy-Stockfish treats each row: refused by
name, or expressed and verified. A new `[[lifted]]` row with no entry fails here, so the next
lift is probed (`uv run python -m chess_variant_sandbox.fairy.lifted --rows <id>`) before it
can reach an outside opponent unasked.
"""

from __future__ import annotations

from typing import Any

import pytest

from chess_variant_sandbox.fairy.lifted import TREATMENT, check
from chess_variant_sandbox.fuzz_lifted.keys import lifted


def _row(row_id: str) -> Any:
    return next(row for row in lifted() if row.id == row_id)


def test_every_lifted_row_says_how_fairy_treats_it() -> None:
    rows = {row.id for row in lifted()}
    missing = sorted(rows - set(TREATMENT))
    assert not missing, (
        f"[[lifted]] rows with no entry in fairy.lifted.TREATMENT: {missing}. Probe each "
        "(`uv run python -m chess_variant_sandbox.fairy.lifted --rows <id>`), refuse a "
        "diverging pair by name in fairy/variant.py, and add the row's entry."
    )
    stale = sorted(set(TREATMENT) - rows)
    assert not stale, f"TREATMENT names rows refusals.toml no longer lifts: {stale}"


def test_every_entry_is_refused_by_name_or_verified() -> None:
    for row_id, (verdict, evidence) in TREATMENT.items():
        assert verdict in ("refused", "verified"), row_id
        assert evidence.strip(), row_id


@pytest.mark.parametrize("row_id", ["gating.duck", "L7", "PA4"])
def test_a_refused_row_is_refused_by_the_reason_it_names(row_id: str) -> None:
    assert check(_row(row_id), None) == []


@pytest.mark.slow
def test_every_row_holds_its_entry() -> None:
    pytest.importorskip("pyffish", reason="pyffish is an optional dev dependency")
    from chess_variant_sandbox.fairy.verify import PyffishBackend

    backend = PyffishBackend()
    problems = [p for row in lifted() for p in check(row, backend)]
    assert not problems, "\n".join(problems)
