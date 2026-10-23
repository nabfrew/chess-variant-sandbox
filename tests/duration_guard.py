"""No test in the fast suite may cost more than two seconds (Q749, `docs/archive/plan-tests.md` F7).

`.claude/rules/tests.md` says to mark anything over a second or two `slow`, and nothing
checked it: Q734 counted 19 fast tests over 2 s, F4's after run 50. This plugin times every
test that is not `slow`, names each one over `LIMIT_S` at the end of the run, and fails the
run on one whose own cost is over twice that.

**Named by the wall clock, judged by the smaller of two, with room for a busy machine.** A
test's own wall time (its setup, call and teardown, less any shared fixture's, below) is
what the gate waits for on its account, and every test over `LIMIT_S` of it is named in the
summary, with the whole time pytest reports beside it. But a gate must not go red
because the machine was busy. So what fails the run is a test's *charge*: the smaller of its
wall time and the processor time its own process spent on it (`time.process_time`, every
thread, the Rust core's included). Another lease's load stretches the wall far more than the
processor time; a serial run's fifteen threads (only an xdist worker is held to one core,
`conftest.one_core_per_worker`) multiply the processor time and not the wall. The smaller is
the test's own cost either way. Load still stretches processor time some -- two threads on
one core share its throughput, so up to about twice -- and the run fails on a charge over
`FAIL_S`, twice the rule: a test that keeps the rule on a quiet machine cannot fail it on a
full one, and one that has drifted to twice the rule fails it anywhere. Every fast test was
brought under `LIMIT_S` on a quiet machine when this was written.

A test whose time goes to a child process -- git, a nested pytest, Fairy-Stockfish, node --
spends little processor time of its own, so it is named and never failed; the summary shows
both clocks, which says which kind a test is.

**What a test is charged.** Its own setup, call and teardown, less the setup of any fixture
wider than a function (`session`, `package`, `module`, `class`): that is paid once for the
tests that share it, by whichever runs first, and is named on its own line when it is over
the limit. So `warm_library`'s ten seconds on one core fail no test, and a module fixture
that grows to a minute (as `test_operators.py`'s sweep did) is still named.

Not judged when the run is traced (coverage, a profiler), which multiplies both clocks; the
names are still printed. A test the run deselects or marks `slow` is not timed.

Not judged either when `--skip-if-green` narrowed the run to what a change reached
(`green_stamp`, 2026-10-22): that is mostly corpus sweeps, each paying warm-up a whole run
spreads over thousands of tests, and one at 3.3 s alone took 6.4 s by both clocks there. The
names are printed, and the next whole run, at most a few gates away, judges.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from typing import Any

import pytest

# Mark anything over a second or two `slow` (`.claude/rules/tests.md`).
LIMIT_S = 2.0
# What fails the run: twice the rule, the most a shared core stretches processor time.
FAIL_S = 2 * LIMIT_S
WIDE = ("session", "package", "module", "class")
_KEY = "duration_guard"


def traced() -> bool:
    """Whether this process runs under a tracer or profiler, which inflates both clocks."""
    if sys.gettrace() is not None or sys.getprofile() is not None:
        return True
    monitoring = getattr(sys, "monitoring", None)
    if monitoring is None:
        return False
    return any(
        monitoring.get_tool(tool) is not None
        for tool in (monitoring.COVERAGE_ID, monitoring.PROFILER_ID, monitoring.DEBUGGER_ID)
    )


def _clocks() -> tuple[float, float]:
    return time.perf_counter(), time.process_time()


class DurationGuard:
    def __init__(self, config: pytest.Config) -> None:
        self.worker = hasattr(config, "workerinput")
        # In the process that runs the tests: the current test's clocks.
        self.started: tuple[float, float] | None = None
        self.shared = [0.0, 0.0]
        self.wide_depth = 0
        self.wide: list[tuple[str, str, float]] = []
        # In the process that reports (the controller, or a serial run).
        self.wall: dict[str, float] = {}
        self.own: dict[str, tuple[float, float]] = {}
        self.fixtures: dict[str, tuple[str, float]] = {}
        self.judged = True
        self.failed: list[str] = []
        self.narrowed: list[str] = []  # what would fail a whole run, in a run that is not one

    # The process that runs the tests.

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item: pytest.Item) -> Iterator[None]:
        self.started = _clocks() if item.get_closest_marker("slow") is None else None
        self.shared = [0.0, 0.0]
        self.wide = []
        yield
        self.started = None

    @pytest.hookimpl(hookwrapper=True)
    def pytest_fixture_setup(self, fixturedef: Any) -> Iterator[None]:
        if fixturedef.scope not in WIDE or self.wide_depth:
            yield
            return
        self.wide_depth += 1
        wall, cpu = _clocks()
        try:
            yield
        finally:
            self.wide_depth -= 1
            now_wall, now_cpu = _clocks()
            self.shared[0] += now_wall - wall
            self.shared[1] += now_cpu - cpu
            self.wide.append((fixturedef.argname, fixturedef.baseid, now_wall - wall))

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call: Any) -> Iterator[None]:
        outcome = yield
        if call.when != "teardown" or self.started is None:
            return
        now = _clocks()
        outcome.get_result().user_properties.append(
            (
                _KEY,
                {
                    "wall": now[0] - self.started[0] - self.shared[0],
                    "cpu": now[1] - self.started[1] - self.shared[1],
                    "wide": [list(each) for each in self.wide if each[2] > LIMIT_S],
                    "traced": traced(),
                },
            )
        )

    # The process that reports.

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.keywords.get("slow"):
            return
        self.wall[report.nodeid] = self.wall.get(report.nodeid, 0.0) + report.duration
        for name, said in report.user_properties:
            if name != _KEY:
                continue
            self.own[report.nodeid] = (said["wall"], said["cpu"])
            self.judged = self.judged and not said["traced"]
            for argname, baseid, spent in said["wide"]:
                key = f"{baseid}::{argname}" if baseid else argname
                if spent > self.fixtures.get(key, ("", 0.0))[1]:
                    self.fixtures[key] = (report.nodeid, spent)

    def charge(self, nodeid: str) -> float:
        return min(self.own[nodeid])

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtestloop(self, session: pytest.Session) -> Iterator[None]:
        """Judge when the loop ends, before `pytest_sessionfinish`: a run this fails then
        ends `TESTS_FAILED`, which `green_stamp` reads, so it is never stamped green."""
        yield
        if self.worker:
            return  # an xdist worker: its reports reach the controller, which decides
        over = sorted(n for n in self.own if self.charge(n) > FAIL_S)
        stamp = session.config.pluginmanager.get_plugin("green-stamp")
        if stamp is not None and stamp.narrowed():
            # A run `--skip-if-green` narrowed is mostly the corpus sweeps a change reached,
            # each paying warm-up a whole run spreads over thousands of tests: a sweep at 3.3 s
            # alone took 6.4 s by both clocks there (2026-10-22). The next whole run judges.
            self.narrowed = over
            return
        if self.judged:
            self.failed = over
        session.testsfailed += len(self.failed)

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        if self.worker:
            return
        def own_wall(nodeid: str) -> float:
            return self.own[nodeid][0] if nodeid in self.own else self.wall[nodeid]

        over = sorted(
            (nodeid for nodeid in self.wall if own_wall(nodeid) > LIMIT_S),
            key=lambda nodeid: -own_wall(nodeid),
        )
        over += [nodeid for nodeid in self.failed + self.narrowed if nodeid not in over]
        if not over and not self.fixtures:
            return
        write = terminalreporter.write_line
        terminalreporter.section(f"fast tests over {LIMIT_S:g} s", sep="-")
        for nodeid in over:
            mine = ""
            if nodeid in self.own:
                wall, cpu = self.own[nodeid]
                mine = f" (its own: {wall:.1f} s wall, {cpu:.1f} s processor)"
            verdict = (
                f"  FAILS, over {FAIL_S:g} s by both clocks: speed it or mark it slow"
                if nodeid in self.failed
                else f"  over {FAIL_S:g} s by both clocks, in a narrowed run"
                if nodeid in self.narrowed else ""
            )
            write(f"{self.wall.get(nodeid, 0.0):6.1f} s{mine}  {nodeid}{verdict}")
        for key, (nodeid, spent) in sorted(self.fixtures.items(), key=lambda kv: -kv[1][1]):
            write(f"{spent:6.1f} s  shared fixture {key}, set up for {nodeid}")
        if self.failed:
            write(
                f"{len(self.failed)} fast test(s) cost over {FAIL_S:g} s by both clocks:"
                " the run fails (tests/duration_guard.py)"
            )
        elif self.narrowed:
            write(
                f"{len(self.narrowed)} fast test(s) cost over {FAIL_S:g} s by both clocks in a run"
                " --skip-if-green narrowed: named, not judged; the next whole run judges"
            )
        elif not self.judged:
            write("a traced run: named, not judged")


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(DurationGuard(config), "duration-guard")
