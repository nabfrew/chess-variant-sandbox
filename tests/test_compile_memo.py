"""`conftest.CompileMemo` keeps what comes back, within a count and a size (Q749)."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import CompileMemo

from chess_variant_sandbox.engine import compile as compile_module
from chess_variant_sandbox.spec import load_spec

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"


def memo_key(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def _memo(entries: int) -> tuple[CompileMemo, list[str]]:
    made: list[str] = []

    def real(spec):  # type: ignore[no-untyped-def]
        made.append(spec["name"])
        return compile_module.compile_spec(spec)

    memo = CompileMemo(real)
    memo.ENTRIES_KEPT = entries
    return memo, made


def test_a_spec_is_kept_the_second_time_and_answered_the_third() -> None:
    memo, made = _memo(10**9)
    dull = load_spec(RULESETS / "dull.json")
    first, second, third = memo(dull), memo(dull), memo(dull)
    assert made == ["dull", "dull"]
    assert third is second and first == second


def test_what_it_keeps_never_holds_more_entries_than_its_bound() -> None:
    dull, chess = load_spec(RULESETS / "dull.json"), load_spec(RULESETS / "chess.json")
    small = CompileMemo.size(compile_module.compile_spec(dull))
    big = CompileMemo.size(compile_module.compile_spec(chess))
    assert big > 10 * small, "a size that does not grow with the tables bounds nothing"
    memo, made = _memo(big + small // 2)  # room for chess or dull, not both
    for spec in (dull, dull, chess, chess):
        memo(spec)
    # Chess came in second and pushed dull, the least recently used, out.
    assert [key for key in memo.kept] == [memo_key(chess)]
    assert memo.entries == big <= memo.ENTRIES_KEPT
    memo(dull)
    assert made[-1] == "dull", "dull was out, so it was compiled again"


def test_a_spec_bigger_than_the_bound_is_never_kept() -> None:
    chess = load_spec(RULESETS / "chess.json")
    memo, made = _memo(10)
    for _ in range(3):
        memo(chess)
    assert made == ["chess"] * 3 and not memo.kept and memo.entries == 0
