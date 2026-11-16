"""Every refusal the validator makes carries its reason, in `spec/refusals.toml` (R-L1a, R-L1b).

The classification is the one source of why each refused combination is refused: package
data beside the refusal tables, whose counts the generated `RULES.md` prints. The record's
script (`docs/reference/refusal-debt.py`) reads the refusals from the code (the tables of
`spec/refusals.py`, every message-like literal of `spec/schema.py`) and holds them one-to-one
against it, so a refusal added without an entry, or an entry left behind by a lift, fails
here and the count cannot rot. The fix is in the data file, never in this test: give the new
refusal its reason (`refusal-debt.md` says how), or delete the entry a lift made stale.
"""

from __future__ import annotations

import private_paths

from chess_variant_sandbox.spec import refusals
from chess_variant_sandbox.spec.reference import render_reference

SCRIPT = "docs/reference/refusal-debt.py"

debt = private_paths.script(SCRIPT, "refusal_debt")
pytestmark = private_paths.needs(SCRIPT)


def test_every_refusal_is_classified_exactly_once() -> None:
    reading = debt.read()
    assert reading.problems == [], "\n".join(reading.problems)


def test_the_reader_sees_the_tables_and_the_validator() -> None:
    """The count is of the code: if the walk found nothing it would cover nothing."""
    reading = debt.read()
    assert len(reading.rows) >= 30
    assert len(reading.literals) >= 500
    assert any(literal.function == "_check_hop_runs" for literal in reading.literals)


def test_the_script_reads_the_packages_classification() -> None:
    """One source: the record's script and the package read the same file."""
    assert debt.DATA.resolve() == refusals.REASONS_FILE.resolve()


def _without(tmp_path, start_marker: str):  # type: ignore[no-untyped-def]
    text = refusals.REASONS_FILE.read_text(encoding="utf-8")
    start = text.index(start_marker)
    end = text.index("[[refusal]]", start + 1)
    cut = tmp_path / "refusals.toml"
    cut.write_text(text[:start] + text[end:], encoding="utf-8")
    return cut


def test_a_refusal_with_no_reason_is_named(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Take one table row's entry out and the reading says which refusal has no reason."""
    problems = debt.read(_without(tmp_path, '[[refusal]]\nid = "pass.P1"')).problems
    assert any("unclassified table row pass.P1" in problem for problem in problems), problems


def test_a_refusal_of_the_validator_with_no_reason_is_named(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The same for a refusal written inline in `schema.py`, even in a function whose other
    messages are covered by one default: the default's stated count no longer holds."""
    problems = debt.read(_without(tmp_path, '[[refusal]]\nid = "R1"')).problems
    assert any("_check_simultaneous_coherence covers 4 literals" in problem
               for problem in problems), problems


def test_a_reason_with_no_refusal_is_named(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """An entry naming nothing the code refuses (a lift that left its row) is a problem."""
    text = refusals.REASONS_FILE.read_text(encoding="utf-8")
    stale = (
        '\n[[refusal]]\nid = "stale.table"\nfamily = "test"\nat = "table:NO_SUCH_ROW"\n'
        'pair = ["rules.a", "rules.b"]\nreason = "contradictory"\nwhy = "lifted"\n'
        '\n[[refusal]]\nid = "stale.literal"\nfamily = "test"\n'
        'at = "_validate_rules: words no message of the validator says"\n'
        'pair = ["rules.a", "rules.c"]\nreason = "contradictory"\nwhy = "lifted"\n'
    )
    head, sep, tail = text.partition("\n[[lifted]]")
    path = tmp_path / "refusals.toml"
    path.write_text(head + stale + sep + tail, encoding="utf-8")
    problems = debt.read(path).problems
    assert any("NO_SUCH_ROW" in problem and "(lifted?)" in problem for problem in problems), problems
    assert any("words no message of the validator says" in problem for problem in problems), problems


def test_every_unbuilt_refusal_says_who_holds_it() -> None:
    reading = debt.read()
    for entry in reading.refusals:
        if entry["reason"] == "not_built":
            assert entry["held_by"], entry["id"]
        if entry["reason"] == "too_expensive":
            assert entry["cost"] and entry["cost_basis"], entry["id"]
    numbers = debt.summary(reading)
    assert sum(numbers["by_reason"].values()) == numbers["refusals"]
    assert numbers["under_plan_rows"] + numbers["new_since"] == numbers["refusals"]


def test_rules_md_prints_the_counts_the_record_counts() -> None:
    """The three counts on `/rules` and in `RULES.md` are the record's, reason by reason."""
    by_reason = debt.summary(debt.read())["by_reason"]
    assert refusals.reason_counts() == by_reason
    assert tuple(by_reason) == refusals.REASONS
    text = render_reference()
    assert f"Today {sum(by_reason.values())} pairs of rules are refused" in text
    for words, reason in (("Contradictory", "contradictory"), ("Too expensive", "too_expensive"),
                          ("Not built yet", "not_built")):
        assert f"**{words} ({by_reason[reason]}).**" in text, words
