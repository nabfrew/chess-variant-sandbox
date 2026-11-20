"""Which refusal a validator sentence is, and the reason `validate` says beside it (R-L3, B1).

`refusals.entry_of` turns a message of `validate_spec` into its `[[refusal]]` entry of
`spec/refusals.toml`: a table row by its exact message, any other refusal by its entry's
`at` words, the longest that occur (`docs/reference/refusal-offers.md` §3). The reason is
read from that entry, never written per refusal. These tests hold, through the record's
reader of `schema.py` (`docs/reference/refusal-debt.py`), that every entry resolves from its
own sentence and from no other (§8 test 4), that the check fails on words that would not,
and that `validate` prints the reason line (§6.1) while saying every sentence as before.
"""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path
from typing import Any

import private_paths
import pytest
from test_spy import _disjoint

from chess_variant_sandbox import cli
from chess_variant_sandbox.spec import refusals, validate_spec
from chess_variant_sandbox.spec.refusals import REASON_SAYS, entry_of, reason_line

SCRIPT = "docs/reference/refusal-debt.py"
debt = private_paths.script(SCRIPT, "refusal_debt")
RULESETS = Path(__file__).resolve().parents[1] / "rulesets"


@functools.cache
def _reading() -> Any:
    """The record's reading of the code, once per run (it parses all of `schema.py`)."""
    return debt.read()


def _entries() -> dict[str, dict[str, Any]]:
    return {str(entry["id"]): entry for entry in _reading().refusals}


@functools.cache
def _word_sites() -> tuple[tuple[str, str, str], ...]:
    """Every entry's words of a literal: (function, words, id)."""
    return tuple(
        (*debt._split(site), str(entry["id"]))
        for entry in _reading().refusals for site in debt._sites(entry)
        if not site.startswith("table:")
    )


def _expected(literal: Any) -> str | None:
    """The entry that classifies ``literal``: the refusal whose words of the literal's own
    function it holds, the longest; None where no refusal names it."""
    hits = [
        (debt._length(says), rid) for function, says, rid in _word_sites()
        if function == literal.function and debt._says(says, literal.text)
    ]
    return max(hits)[1] if hits else None


@private_paths.needs(SCRIPT)
def test_every_refusal_resolves_to_its_entry_and_no_other() -> None:
    """§8 test 4: each entry is what `entry_of` says of the sentence its `at` names, and no
    other literal of `schema.py`, refusal or not, resolves to it."""
    reading = _reading()
    assert reading.problems == [], "\n".join(reading.problems)
    resolved: set[str] = set()
    by_id = _entries()
    for row in reading.rows:
        entry = entry_of(row.message)
        assert entry is not None, row
        owner = next(
            rid for rid, raw in by_id.items()
            if f"table:{row.id}|{row.second}" in debt._sites(raw)
            or f"table:{row.id}" in debt._sites(raw)
        )
        assert entry.id == owner, (row, entry.id)
        resolved.add(entry.id)
    for literal in reading.literals:
        expected = _expected(literal)
        entry = entry_of(literal.text)
        assert (entry.id if entry else None) == expected, (literal, entry)
        if expected is not None:
            resolved.add(expected)
    assert resolved == set(by_id), sorted(set(by_id) - resolved)


@private_paths.needs(SCRIPT)
def test_an_entry_carries_its_reason_from_the_file() -> None:
    reading = _reading()
    by_id = _entries()
    sentences = {
        "R1": next(lit.text for lit in reading.literals if _expected(lit) == "R1"),
        "R7": refusals.refusal("R7"),
        "pass.P7": refusals.refusal("pass.P7"),
        "hold.C1": refusals.refusal("hold.C1"),
    }
    for rid, sentence in sentences.items():
        raw, entry = by_id[rid], entry_of(sentence)
        assert entry is not None and entry.id == rid
        assert (entry.reason, entry.why, list(entry.pair)) == (raw["reason"], raw["why"], raw["pair"])
        assert entry.held_by == raw.get("held_by") and entry.cost == raw.get("cost")


def test_a_message_no_entry_names_is_none() -> None:
    assert entry_of("board.width must be an integer") is None
    assert entry_of("") is None


# ------------------------------------------------------------------ the check fails on

def _with_at(tmp_path: Path, rid: str, at: str) -> Path:
    text = refusals.REASONS_FILE.read_text(encoding="utf-8")
    start = text.index(f'[[refusal]]\nid = "{rid}"')
    line = text.index("\nat = ", start) + 1
    end = text.index("\n", line)
    path = tmp_path / "refusals.toml"
    path.write_text(text[:line] + at + text[end:], encoding="utf-8")
    return path


@private_paths.needs(SCRIPT)
def test_the_check_fails_on_words_that_span_an_interpolation(tmp_path: Path) -> None:
    path = _with_at(
        tmp_path, "X11.radius",
        'at = "_check_hexagon_coherence: rules.explode_on_capture.radius {} reaches across"',
    )
    problems = debt.read(path).problems
    assert any("X11.radius" in p and "span an interpolated part" in p for p in problems), problems


@private_paths.needs(SCRIPT)
def test_the_check_fails_on_words_in_another_refusals_literal(tmp_path: Path) -> None:
    """R16's words set to a longer run of R1's sentence: R1's would resolve to R16."""
    path = _with_at(
        tmp_path, "R16",
        "at = \"_check_npc_pounce: there is no check when both players move at once; use "
        "'royal_capture'\"",
    )
    reading = debt.read(path)
    problems = reading.problems
    assert any("would name it at run time" in p and "R16" in p for p in problems), problems


@private_paths.needs(SCRIPT)
def test_the_check_fails_on_one_literal_claimed_by_two_refusals(tmp_path: Path) -> None:
    path = _with_at(
        tmp_path, "gating.either_face",
        'at = "_check_gating_coherence: rules.drops.gating beside rules.drops.placement:"',
    )
    problems = debt.read(path).problems
    assert any("is claimed by" in p for p in problems), problems


# ------------------------------------------------------------------ split literals

def test_each_spy_route_names_its_own_refusal() -> None:
    """Y4, Y5 and Y6 were one literal told apart only by an interpolated part; each route
    now says its own words, and each sentence resolves to its own entry."""
    spec = _disjoint()
    spec["rules"]["strike"] = {
        "effect": "demote", "position": "stay", "chain": [{"from": "N", "to": "S"}],
        "immune": ["K"],
    }
    chain = spec
    switch = _disjoint()
    for spy in ("S", "T"):
        piece = next(piece for piece in switch["pieces"] if piece["id"] == spy)
        piece["captures"] = ["P"]
        piece["moves"][0]["mode"] = "both"
    switch["rules"]["strike"] = {"switches_sides": ["S"], "immune": ["K"]}
    link = _disjoint()
    link["rules"]["contacts"] = {"links": [{"mover": "N", "onto": "P", "mover_to": "T"}], "immune": ["K"]}
    for built, rid, words in (
        (chain, "Y4.foreign", "rules.strike.chain gives player 1 the spy 'S', which player 1 "
                              "does not start with: a spy counts for the player whose spy type "
                              "it is, so make each player's piece into its own spy"),
        (switch, "Y5.foreign", "rules.strike.switches_sides gives player 1 the spy 'S', which "
                               "player 1 does not start with"),
        (link, "Y6.foreign", "a rules.contacts link's mover_to gives player 0 the spy 'T', "
                             "which player 0 does not start with"),
    ):
        said = [error for error in validate_spec(built) if words in error]
        assert said, validate_spec(built)
        assert {getattr(entry_of(error), "id", None) for error in said} == {rid}


# ------------------------------------------------------------------ validate prints it

def _validate(tmp_path: Path, spec: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    code = cli.main(["validate", str(path)])
    return code, capsys.readouterr().out


def test_validate_says_the_reason_under_each_refusal_of_chess_in_rounds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """§9 B1's witness, the one the design names: chess (the committed file, used because the
    design pins it) with `rules.simultaneous` added refuses two things, R1 (inline) and R7 (a
    table row), and each problem gets its reason line, in the problem's own order."""
    spec = json.loads((RULESETS / "chess.json").read_text(encoding="utf-8"))
    spec["rules"]["simultaneous"] = {}
    errors = validate_spec(spec)
    assert [getattr(entry_of(error), "id", None) for error in errors] == ["R1", "R7"]
    code, out = _validate(tmp_path, spec, capsys)
    assert code == 1
    lines = out.splitlines()
    assert lines[1:] == [
        f"  - {errors[0]}",
        f"      contradictory: {REASON_SAYS['contradictory']}",
        f"  - {errors[1]}",
        f"      contradictory: {REASON_SAYS['contradictory']}",
    ]


def test_validate_says_a_problem_with_no_entry_as_before(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    spec = json.loads((RULESETS / "chess.json").read_text(encoding="utf-8"))
    spec["board"]["width"] = "wide"
    errors = validate_spec(spec)
    assert errors and all(entry_of(error) is None for error in errors)
    code, out = _validate(tmp_path, spec, capsys)
    assert code == 1
    assert out.splitlines()[1:] == [f"  - {error}" for error in errors]


def test_the_reason_line_says_cost_and_holder() -> None:
    expensive = refusals.Entry("x", "too_expensive", "why", ("a", "b"), cost="a wider word")
    assert reason_line(expensive) == (
        f"too expensive: {REASON_SAYS['too_expensive']} Cost: a wider word."
    )
    planned = refusals.Entry("y", "not_built", "why", ("a", "b"), held_by="SR9a")
    assert reason_line(planned) == f"not built: {REASON_SAYS['not_built']} It is planned. Held by SR9a."
    unheld = copy.copy(planned)._replace(held_by="none")
    assert reason_line(unheld) == f"not built: {REASON_SAYS['not_built']} Held by no task yet."
    assert reason_line(refusals.Entry("z", "inert", "why", ("a", "b"))) is None
