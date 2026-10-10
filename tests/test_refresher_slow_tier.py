"""The refresher's weekly slow tier (Q743, `docs/plan-tests.md` §8 F1).

`chess_variant_sandbox/refresher.py`, point 7: a watching refresher with nothing owed and the
machine free runs `pytest -m slow` once a week under a `heavy` lease and records the result in
`runs/slow-tier.json`, and never acts on it. Held down here over `test_refresher`'s fake
`Machine`; `GitMachine.slow_tier` with a fake process runner; the report parsing; and the
file-reading script the coordinator runs.
"""

from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path
from typing import Any

from test_refresher import FakeMachine, repo  # noqa: F401  (`repo` is a fixture)

from chess_variant_sandbox import refresher
from chess_variant_sandbox.refresher import SlowTier

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / ".claude" / "scripts" / "slow_tier_status.py"

NOW = datetime.datetime(2026, 10, 5, 12, 0, tzinfo=datetime.timezone.utc)

RED = SlowTier(commit="c1", passed=740, failed=2, skipped=5,
               failing=("tests.test_a::test_one", "tests.test_b.TestX::test_two"),
               seconds=900.0, exit_code=1)


def watching(machine: FakeMachine, runs: Path, passes: int = 1, **fields: Any
             ) -> refresher.Refresher:
    fields.setdefault("clock", lambda: NOW)
    return refresher.Refresher(machine, refresher.Ledger(runs), watch=True, passes=passes,
                               **fields)


def written(runs: Path, finished: datetime.datetime, **fields: Any) -> None:
    refresher.write_slow_tier(runs, SlowTier(commit="c0", passed=1, **fields),
                              started=finished.isoformat(), finished=finished.isoformat())


class TestWhenItRuns:
    def test_a_watching_idle_refresher_runs_it_once_under_a_heavy_lease(
        self, tmp_path: Path
    ) -> None:
        machine = FakeMachine()
        watching(machine, tmp_path, passes=3).run()
        work = [c for c in machine.calls if c[0] in ("acquire", "slow tier", "release")]
        assert work == [("acquire", "heavy"), ("slow tier", "c1"), ("release", "heavy")]
        record = refresher.read_slow_tier(tmp_path)
        assert record is not None and record["outcome"] == "green"
        assert record["commit"] == "c1" and record["passed"] == 3

    def test_games_owed_come_first(self, tmp_path: Path) -> None:
        machine = FakeMachine(owed={"a": ("rescore", "s")})
        watching(machine, tmp_path, passes=3).run()
        order = [c[0] for c in machine.calls if c[0] in ("repair", "slow tier")]
        assert order == ["repair", "slow tier"]

    def test_never_in_one_pass_or_with_only(self, tmp_path: Path) -> None:
        machine = FakeMachine()
        refresher.Refresher(machine, refresher.Ledger(tmp_path), passes=3).run()
        refresher.Refresher(machine, refresher.Ledger(tmp_path), watch=True, passes=3,
                            only=frozenset({"a"})).run()
        refresher.Refresher(machine, refresher.Ledger(tmp_path), watch=True, passes=3,
                            slow_tier=False).run()
        assert machine.did("slow tier") == []
        assert refresher.read_slow_tier(tmp_path) is None

    def test_never_while_another_job_has_the_machine(self, tmp_path: Path) -> None:
        machine = FakeMachine(busy_with="a heavy lease is held by T5")
        watching(machine, tmp_path, passes=3).run()
        assert machine.did("slow tier") == [] and machine.did("acquire") == []

    def test_a_lease_not_granted_is_waited_for(self, tmp_path: Path) -> None:
        machine = FakeMachine(grant=False)
        watching(machine, tmp_path, passes=2).run()
        assert machine.did("slow tier") == []
        status = refresher.read_status(tmp_path)
        assert status is not None
        assert status["waiting_for"] == "a heavy lease for the slow tier"

    def test_weekly(self, tmp_path: Path) -> None:
        machine = FakeMachine()
        written(tmp_path, NOW - datetime.timedelta(days=6, hours=23))
        watching(machine, tmp_path, passes=2).run()
        assert machine.did("slow tier") == []
        written(tmp_path, NOW - datetime.timedelta(days=7))
        watching(machine, tmp_path, passes=2).run()
        assert len(machine.did("slow tier")) == 1

    def test_a_run_that_did_not_finish_waits_its_week_too(self, tmp_path: Path) -> None:
        machine = FakeMachine(tier=RuntimeError("no interpreter"))
        watching(machine, tmp_path, passes=3).run()
        assert len(machine.did("slow tier")) == 1
        record = refresher.read_slow_tier(tmp_path)
        assert record is not None and record["outcome"] == "unfinished"
        assert "no interpreter" in record["note"]
        assert machine.holding is None


class TestWhatItRecords:
    def test_a_red_run_names_its_failures_and_changes_nothing_else(
        self, tmp_path: Path
    ) -> None:
        machine = FakeMachine(tier=RED)
        watching(machine, tmp_path).run()
        record = refresher.read_slow_tier(tmp_path)
        assert record is not None
        assert record["outcome"] == "red"
        assert record["failing"] == list(RED.failing) and record["failing_total"] == 2
        assert record["summary"].startswith("red on 2026-")
        assert "740 passed, 2 failed, 5 skipped" in record["summary"]
        assert "tests.test_a::test_one" in record["summary"]
        # Recorded, never acted on: no refusal, no delivery, no kept commit, no halt.
        status = refresher.read_status(tmp_path)
        assert status is not None
        assert status["refusals"] == [] and "last_refusal" not in status
        assert machine.did("deliver") == [] and machine.did("keep") == []
        assert status["slow_tier"]["outcome"] == "red"
        assert any(line.startswith("slow tier: red on ")
                   for line in refresher.plain_status(status))
        events = refresher.read_log(tmp_path)
        assert [e["event"] for e in events] == ["slow tier"]
        assert events[0]["failing"] == list(RED.failing)

    def test_committed_files_it_changed_are_put_back_and_named(self, tmp_path: Path) -> None:
        machine = FakeMachine(tier_changes=["rulesets/chess.json"])
        watching(machine, tmp_path).run()
        assert machine.did("discard") == [("discard", "rulesets/chess.json")]
        record = refresher.read_slow_tier(tmp_path)
        assert record is not None
        assert "put back: rulesets/chess.json" in record["note"]

    def test_the_names_kept_are_capped_and_the_rest_counted(self, tmp_path: Path) -> None:
        many = tuple(f"tests.t::test_{n}" for n in range(refresher.SLOW_TIER_NAMES + 7))
        record = refresher.write_slow_tier(
            tmp_path, SlowTier(commit="c", failed=len(many), failing=many, exit_code=1),
            started=NOW.isoformat(), finished=NOW.isoformat())
        assert len(record["failing"]) == refresher.SLOW_TIER_NAMES
        assert record["failing_total"] == len(many)
        assert "and 54 more" in record["summary"]

    def test_status_says_when_it_has_not_run(self) -> None:
        lines = refresher.plain_status({"state": "idle", "age_s": 1})
        assert lines[-1] == "slow tier: not run here yet (weekly, when watching and idle)"


JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="1" failures="1" skipped="1" tests="5">
<testcase classname="tests.test_a" name="test_ok" time="1.0"/>
<testcase classname="tests.test_a.TestX" name="test_ok_too" time="1.0"/>
<testcase classname="tests.test_a" name="test_red" time="1.0"><failure message="x">x</failure></testcase>
<testcase classname="tests.test_b" name="test_broken" time="1.0"><error message="y">y</error></testcase>
<testcase classname="tests.test_b" name="test_later" time="0.0"><skipped message="z">z</skipped></testcase>
</testsuite></testsuites>
"""


def test_the_report_is_read_for_counts_and_names(tmp_path: Path) -> None:
    (tmp_path / "slow.xml").write_text(JUNIT, encoding="utf-8")
    tier = refresher.parse_junit(tmp_path / "slow.xml")
    assert (tier.passed, tier.failed, tier.errors, tier.skipped) == (2, 1, 1, 1)
    assert tier.failing == ("tests.test_a::test_red", "tests.test_b::test_broken")
    assert tier.outcome == "red"


class TestTheRealMachine:
    def test_it_runs_pytest_slow_on_the_worktrees_interpreter(self, repo) -> None:  # noqa: F811
        seen: list[list[str]] = []
        real = repo.run

        def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            if command[0] == "git":
                return real(command, **kwargs)
            seen.append(command)
            report = next(arg for arg in command if arg.startswith("--junitxml="))
            Path(report.split("=", 1)[1]).write_text(JUNIT, encoding="utf-8")
            assert kwargs["cwd"] == repo.worktree
            assert kwargs["timeout"] == refresher.SLOW_TIER_TIMEOUT_S
            return subprocess.CompletedProcess(command, 1, "", "")

        repo.run = run
        tier = repo.slow_tier()
        assert seen[0][:5] == [str(refresher._python(repo.worktree)), "-m", "pytest", "-m",
                               "slow"]
        assert tier.commit == repo.head() and tier.exit_code == 1
        assert (tier.passed, tier.failed, tier.errors) == (2, 1, 1)
        assert tier.note == ""
        # The report went to a scratch directory: nothing is left in the worktree.
        assert repo.changed() == []

    def test_no_report_is_a_run_that_did_not_finish(self, repo) -> None:  # noqa: F811
        real = repo.run

        def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            if command[0] == "git":
                return real(command, **kwargs)
            return subprocess.CompletedProcess(command, 4, "", "ERROR: usage")

        repo.run = run
        tier = repo.slow_tier()
        assert tier.outcome == "unfinished" and "ERROR: usage" in tier.note

    def test_a_run_past_its_time_is_stopped_and_said(self, repo) -> None:  # noqa: F811
        real = repo.run

        def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            if command[0] == "git":
                return real(command, **kwargs)
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

        repo.run = run
        tier = repo.slow_tier()
        assert tier.outcome == "unfinished" and "stopped after" in tier.note


class TestTheScript:
    """`.claude/scripts/slow_tier_status.py`: reads the file, imports nothing."""

    def run(self, runs: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), str(runs)],
                              capture_output=True, text=True, check=False)

    def test_none_recorded(self, tmp_path: Path) -> None:
        done = self.run(tmp_path)
        assert done.returncode == 1 and "no run recorded" in done.stdout

    def test_green_and_recent_is_zero(self, tmp_path: Path) -> None:
        now = datetime.datetime.now(datetime.timezone.utc)
        written(tmp_path, now - datetime.timedelta(days=1))
        done = self.run(tmp_path)
        assert done.returncode == 0, done.stdout
        assert done.stdout.startswith("slow tier: green on ")

    def test_red_names_its_failures(self, tmp_path: Path) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        refresher.write_slow_tier(tmp_path, RED, started=now, finished=now)
        done = self.run(tmp_path)
        assert done.returncode == 1
        assert "  tests.test_b.TestX::test_two" in done.stdout.splitlines()

    def test_green_but_old_is_one(self, tmp_path: Path) -> None:
        now = datetime.datetime.now(datetime.timezone.utc)
        written(tmp_path, now - datetime.timedelta(days=9))
        done = self.run(tmp_path)
        assert done.returncode == 1 and "9 days old" in done.stdout

    def test_it_imports_nothing_of_the_package(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        assert "chess_variant_sandbox" not in text.split('"""', 2)[2]
