"""Every lifted pair is fuzzed against a check, and stays so (R-L6d).

`spec/refusals.toml`'s `[[lifted]]` rows are pairs the rule language once refused and now plays.
The lifted-pair fuzz (`chess_variant_sandbox/fuzz_lifted/`) is the only independent net beyond
each lift's own test, so a row it does not watch is a pair nobody re-checks. Each row has a
rule-obedience check that names it (`checks.CHECKS`' `pairs`), or a stated reason for having
none (`checks.UNCHECKED`). A new `[[lifted]]` row in neither place fails here: write its check
(`docs/reference/fuzz-lifted-pairs.md` §2 says how), or say why it has none. The same shape as
`tests/test_fairy_lifted.py::test_every_lifted_row_says_how_fairy_treats_it`.
"""

from __future__ import annotations

from chess_variant_sandbox.fuzz_lifted.checks import UNCHECKED, checked_pairs
from chess_variant_sandbox.fuzz_lifted.generate import buildable
from chess_variant_sandbox.fuzz_lifted.keys import lifted


def test_every_lifted_row_has_a_check_or_says_why_not() -> None:
    rows = {row.id for row in lifted()}
    assert len(rows) >= 107, len(rows)
    missing = sorted(rows - checked_pairs() - set(UNCHECKED))
    assert not missing, (
        f"[[lifted]] rows no fuzz check watches: {missing}. Write a check in "
        "chess_variant_sandbox/fuzz_lifted/checks.py naming the row (with a planted-fault test "
        "in tests/test_fuzz_lifted.py), or give the reason it has none in checks.UNCHECKED."
    )
    stale = sorted(set(UNCHECKED) - rows)
    assert not stale, f"UNCHECKED names rows refusals.toml no longer lifts: {stale}"
    both = sorted(set(UNCHECKED) & checked_pairs())
    assert not both, f"UNCHECKED names rows a check does watch: {both}"
    assert all(reason.strip() for reason in UNCHECKED.values())


def test_every_lifted_row_has_a_maker_for_both_keys() -> None:
    """A checked row the generator cannot build is a check that never runs."""
    built = {row.id for row in buildable()}
    assert sorted(row.id for row in lifted() if row.id not in built) == []
