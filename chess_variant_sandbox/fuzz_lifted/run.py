"""Run generated cases against the checks, with a coverage report and a checkpoint.

A run walks seeds in order. Each seed builds a case (`generate.case`), plays it
(`record.play`), and hands the record to every check whose keys the spec holds, and to the
existing fuzz oracles (`fuzz.RULE_OBEDIENCE` and the cheap half of `fuzz.SELF_CONSISTENCY`) on
its positions played by turns. A failure is written down -- the seed, the row, the check, the
message and the spec, enough to rebuild it with `generate.case(seed)` -- and the run goes on:
a soak that stops at its first find covers nothing after it.

Under `--checkpoint` the run writes its state (the next seed, the coverage counters, the
failures so far) atomically every `CHECKPOINT_EVERY` cases and on exit, and a run started with
the same file resumes from it (memory: *long runs must resume*). The state is a pure function
of the seeds walked, so a resumed run ends where an unbroken one would.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_variant_sandbox.fuzz_lifted.checks import (
    CHECKS,
    KNOWN_PREFIX,
    Check,
    CheckFailure,
    checked_pairs,
)
from chess_variant_sandbox.fuzz_lifted.generate import buildable, case
from chess_variant_sandbox.fuzz_lifted.keys import BATCH_ORDER, lifted
from chess_variant_sandbox.fuzz_lifted.record import PLIES, Record, play
from chess_variant_sandbox.spec.schema import SpecError

CHECKPOINT_EVERY = 50
HELD_KEPT = 400  # how many core-refused cases the checkpoint keeps, with the core's reason


@dataclass
class Coverage:
    """What a run built, played and checked, by `[[lifted]]` row."""

    cases: int = 0
    played: int = 0
    unbuilt: Counter[str] = field(default_factory=Counter)  # row -> seeds that built nothing
    # row -> valid cases the core refused to play (`Engine` raises NotImplementedError): a
    # spec the validator accepts and `bridge.rust_supports` refuses, a validator gap (§5).
    held: Counter[str] = field(default_factory=Counter)
    held_cases: list[dict[str, Any]] = field(default_factory=list)  # the first few, with why
    refused: int = 0  # children the validator refused on the way (discarded, never played)
    occurred: Counter[str] = field(default_factory=Counter)  # row -> played cases holding it
    exercised: Counter[str] = field(default_factory=Counter)  # row -> (case, check) runs on it
    runs: Counter[str] = field(default_factory=Counter)  # check -> records it read
    judged: Counter[str] = field(default_factory=Counter)  # check -> events of its rule judged
    failures: list[dict[str, Any]] = field(default_factory=list)
    known: Counter[str] = field(default_factory=Counter)  # known fault -> cases it recurred in
    seconds: float = 0.0
    next_seed: int = 0

    def as_json(self) -> dict[str, Any]:
        return {
            "cases": self.cases, "played": self.played, "unbuilt": dict(self.unbuilt),
            "held": dict(self.held), "held_cases": self.held_cases,
            "refused": self.refused, "occurred": dict(self.occurred),
            "exercised": dict(self.exercised), "runs": dict(self.runs), "judged": dict(self.judged),
            "failures": self.failures, "known": dict(self.known), "seconds": self.seconds, "next_seed": self.next_seed,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Coverage:
        return cls(
            cases=data["cases"], played=data["played"], unbuilt=Counter(data["unbuilt"]),
            held=Counter(data.get("held", {})), held_cases=list(data.get("held_cases", [])),
            refused=data["refused"], occurred=Counter(data["occurred"]),
            exercised=Counter(data["exercised"]), runs=Counter(data["runs"]), judged=Counter(data.get("judged", {})),
            failures=list(data["failures"]), known=Counter(data.get("known", {})), seconds=data["seconds"], next_seed=data["next_seed"],
        )

    def report(self) -> str:
        """Which lifted pairs occurred and how often, which have a check, which have neither."""
        rows = lifted()
        checked = checked_pairs()
        rate = self.played / self.seconds * 3600 if self.seconds else 0.0
        lines = [
            f"{self.cases} seeds, {self.played} cases played, {self.refused} generated specs "
            f"refused by the validator (discarded), {len(self.failures)} failures; "
            f"{rate:,.0f} cases an hour",
            "",
            "row     batch  occurred  checked  check runs",
        ]
        for batch in BATCH_ORDER:
            for row in rows:
                if row.batch != batch:
                    continue
                lines.append(
                    f"{row.id:<8}{batch:<7}{self.occurred[row.id]:>8}  "
                    f"{'yes' if row.id in checked else 'no':>7}  {self.exercised[row.id]:>10}"
                )
        never = [r.id for r in rows if not self.occurred[r.id]]
        unchecked = [r.id for r in rows if r.id not in checked]
        neither = [r.id for r in rows if not self.occurred[r.id] and r.id not in checked]
        idle = [r.id for r in rows if r.id in checked and not self.exercised[r.id]]
        lines += [
            "",
            f"never occurred in a case ({len(never)}): " + (", ".join(never) or "none"),
            f"no check written ({len(unchecked)}): " + (", ".join(unchecked) or "none"),
            f"neither a case nor a check ({len(neither)}): " + (", ".join(neither) or "none"),
            f"a check, never run on a case holding it ({len(idle)}): " + (", ".join(idle) or "none"),
            "checks (records read / events judged): " + ", ".join(
                f"{name} {n}/{self.judged[name]}" for name, n in sorted(self.runs.items())),
        ]
        if self.known:
            lines.append("known faults, recurring (not counted as failures): " + ", ".join(
                f"{fault} x{n}" for fault, n in self.known.most_common()))
        if self.held:
            lines.append("valid cases the core refused to play, by row (a validator gap): " + ", ".join(
                f"{row} x{n}" for row, n in self.held.most_common()))
            reasons = Counter(case["why"] for case in self.held_cases)
            lines.append("  the core's reasons (first cases kept): " + "; ".join(
                f"{why} x{n}" for why, n in reasons.most_common()))
        if self.unbuilt:
            lines.append("seeds that built no case, by row: " + ", ".join(
                f"{row} x{n}" for row, n in self.unbuilt.most_common()))
        for failure in self.failures[:20]:
            lines.append(f"FAILED seed {failure['seed']} row {failure['row']} "
                         f"{failure['check']}: {failure['message']}")
        return "\n".join(lines)


def _fuzz_oracles(record: Record) -> None:
    """The fuzz module's own oracles on every position played by turns."""
    from chess_variant_sandbox import fuzz

    cheap = (
        fuzz.legal_is_a_subset_of_pseudo_legal,
        fuzz.the_key_matches_the_slow_definition,
        fuzz.the_verdict_does_not_depend_on_being_handed_the_moves,
    )
    for step in record.steps:
        if step.round:
            continue
        for oracle in cheap + fuzz.RULE_OBEDIENCE:
            try:
                oracle(record.engine, step.before)
            except fuzz.OracleFailure as failure:
                raise CheckFailure(f"step {step.index}: {failure}") from failure


def check_case(seed: int, coverage: Coverage, plies: int = PLIES) -> None:
    """Build, play and check one seed, adding what happened to ``coverage``."""
    coverage.cases += 1
    built = case(seed)
    coverage.refused += built.refused
    if built.spec is None:
        coverage.unbuilt[built.row] += 1
        return
    try:
        record = play(built.spec, seed, plies)
    except (NotImplementedError, SpecError) as held:
        # A valid spec the core will not play: a validator gap. R-L6d met one, F5 (`purse.upkeep`
        # beside the duck), which U1b made the validator's (`hold.H7`). The compiler refusing a
        # spec the validator accepted (`SpecError`, R-L6b's seed 2117229) is the same gap one
        # step earlier, and once killed the soak. Kept so the next gap is counted apart from the
        # unbuilt, with the reason, and never as a played case.
        coverage.held[built.row] += 1
        if len(coverage.held_cases) < HELD_KEPT:
            why = str(held) if isinstance(held, NotImplementedError) else f"the compiler refuses it: {held}"
            coverage.held_cases.append({"seed": seed, "row": built.row, "why": why,
                                        "case": built.describe()})
        return
    coverage.played += 1
    for row in built.present:
        coverage.occurred[row] += 1
    named: list[tuple[str, Check | None]] = [
        (check.name, check) for check in CHECKS if check.applies(built.spec)
    ]
    named.append(("fuzz_oracles", None))
    for name, check in named:
        coverage.runs[name] += 1
        if check is not None:
            for row in built.present:
                if row in check.pairs:
                    coverage.exercised[row] += 1
        try:
            if check is None:
                _fuzz_oracles(record)
            else:
                coverage.judged[name] += check(record)
        except CheckFailure as failure:
            message = str(failure)
            if KNOWN_PREFIX in message:
                fault = message.split(KNOWN_PREFIX, 1)[1].split("]", 1)[0]
                coverage.known[fault] += 1
                continue
            coverage.failures.append(_failure(seed, built, name, message))
        except Exception as error:  # noqa: BLE001 - a crash inside a check is a find too
            coverage.failures.append(_failure(
                seed, built, name, f"{type(error).__name__}: {error}",
                trace=traceback.format_exc(limit=4),
            ))


def _failure(seed: int, built: Any, check: str, message: str, trace: str = "") -> dict[str, Any]:
    found = {"seed": seed, "row": built.row, "check": check, "message": message,
             "case": built.describe(), "spec": built.spec}
    if trace:
        found["trace"] = trace
    return found


def run(
    seeds: range, plies: int = PLIES, checkpoint: Path | None = None,
    deadline: float | None = None,
) -> Coverage:
    """Walk ``seeds`` (resuming from ``checkpoint`` if it exists), stopping early at
    ``deadline`` (a `time.monotonic` value)."""
    coverage = Coverage(next_seed=seeds.start)
    if checkpoint is not None and checkpoint.exists():
        coverage = Coverage.from_json(json.loads(checkpoint.read_text(encoding="utf-8")))
    started = time.monotonic()
    spent = coverage.seconds
    since = 0
    for seed in range(max(coverage.next_seed, seeds.start), seeds.stop):
        if deadline is not None and time.monotonic() >= deadline:
            break
        check_case(seed, coverage, plies)
        coverage.next_seed = seed + 1
        coverage.seconds = spent + time.monotonic() - started
        since += 1
        if checkpoint is not None and since >= CHECKPOINT_EVERY:
            save(coverage, checkpoint)
            since = 0
    coverage.seconds = spent + time.monotonic() - started
    if checkpoint is not None:
        save(coverage, checkpoint)
    return coverage


def save(coverage: Coverage, path: Path) -> None:
    """Write the run's state atomically: a kill mid-write leaves the last good file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(path.suffix + ".tmp")
    scratch.write_text(json.dumps(coverage.as_json()), encoding="utf-8")
    os.replace(scratch, path)


def rows_built() -> list[str]:
    """The rows the generator has a maker for on both sides."""
    return [row.id for row in buildable()]
