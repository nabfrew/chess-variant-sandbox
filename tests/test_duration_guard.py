"""`tests/duration_guard.py` names a fast test over the limit and fails one over twice it.

Each case runs a small suite inside this one (`pytester`), with the limits shrunk to tenths
of a second so the suite inside spends a fraction of a second: a test that burns processor
time, one that only waits, one marked `slow`, and a shared fixture that is expensive.
"""

from __future__ import annotations

import json

import duration_guard
import pytest

pytest_plugins = ("pytester",)

INNER = '''
import time

import pytest


def _burn(seconds):
    end = time.process_time() + seconds
    while time.process_time() < end:
        pass


@pytest.fixture(scope="module")
def shared():
    _burn(0.12)


def test_quick():
    pass


def test_waits():
    time.sleep(0.12)


def test_burns():
    _burn(0.12)


@pytest.mark.slow
def test_burns_but_is_slow():
    _burn(0.12)


def test_reads_the_shared_fixture(shared):
    pass
'''


@pytest.fixture
def guarded(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> pytest.Pytester:
    monkeypatch.setattr(duration_guard, "LIMIT_S", 0.04)
    monkeypatch.setattr(duration_guard, "FAIL_S", 0.08)
    # No re-time unless a case asks for one: each is a pytest started in a child process.
    monkeypatch.setattr(duration_guard, "RETIME_MAX", 0)
    monkeypatch.delenv(duration_guard.RETIME_ENV, raising=False)
    pytester.makeini("[pytest]\nmarkers =\n    slow: slow\n")
    return pytester


# Load, simulated: a test that burns only while it is not the re-time's child is one the
# machine made slow. One real child process, since each is a pytest start; the verdicts on
# a test still over alone, or not timeable alone, are asked of a stand-in below.
BUSY = f'''
import os
import time


def test_burns_when_busy():
    if not os.environ.get("{duration_guard.RETIME_ENV}"):
        end = time.process_time() + 0.12
        while time.process_time() < end:
            pass
'''


def test_a_trip_only_load_made_is_re_timed_alone_and_not_failed(
    guarded: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q1157, end to end: the child process, its clocks file and the nodeid it is run by."""
    monkeypatch.setattr(duration_guard, "RETIME_MAX", 8)
    guarded.makepyfile(test_inner=BUSY)
    result = guarded.runpytest("-p", "duration_guard")
    said = _summary(result)
    assert result.ret == pytest.ExitCode.OK
    line = next(line for line in said.splitlines() if "::test_burns_when_busy" in line)
    assert "not failed; re-timed alone: " in line
    assert "1 fast test(s) over 0.08 s by both clocks in the run came under it" in said


def test_a_trip_over_alone_too_or_not_timeable_alone_fails(
    guarded: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    alone = {"test_inner.py::test_burns": (0.12, 0.12), "test_inner.py::test_waits": None}
    monkeypatch.setattr(duration_guard, "RETIME_MAX", 8)
    monkeypatch.setattr(duration_guard, "retime_alone", lambda nodeid, rootdir: alone.get(nodeid))
    guarded.makepyfile(test_inner=(
        "import time\n\n"
        "def _burn(seconds):\n"
        "    end = time.process_time() + seconds\n"
        "    while time.process_time() < end:\n"
        "        pass\n\n"
        "def test_burns():\n    _burn(0.12)\n\n"
        "def test_waits():\n    _burn(0.12)\n"
    ))
    result = guarded.runpytest("-p", "duration_guard")
    said = _summary(result)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    burns = next(line for line in said.splitlines() if "::test_burns" in line)
    waits = next(line for line in said.splitlines() if "::test_waits" in line)
    assert "FAILS" in burns and "re-timed alone: 0.1 s wall" in burns
    assert "FAILS" in waits and "could not be timed" in waits
    assert "2 fast test(s) cost over 0.08 s by both clocks: the run fails" in said


def test_only_the_cheapest_trips_are_re_timed(
    guarded: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(duration_guard, "RETIME_MAX", 1)
    monkeypatch.setattr(
        duration_guard, "retime_alone", lambda nodeid, rootdir: calls.append(nodeid) or (0.0, 0.0)
    )
    guarded.makepyfile(test_inner=(
        "import time\n\n"
        "def _burn(seconds):\n"
        "    end = time.process_time() + seconds\n"
        "    while time.process_time() < end:\n"
        "        pass\n\n"
        "def test_dear():\n    _burn(0.3)\n\n"
        "def test_cheap():\n    _burn(0.12)\n"
    ))
    result = guarded.runpytest("-p", "duration_guard")
    assert calls == ["test_inner.py::test_cheap"]
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    assert "1 fast test(s) cost over 0.08 s by both clocks: the run fails" in _summary(result)


def test_a_re_time_child_reports_its_clocks_and_judges_nothing(
    guarded: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = guarded.path / "clocks.json"
    monkeypatch.setenv(duration_guard.RETIME_ENV, str(out))
    guarded.makepyfile(test_inner=INNER)
    result = guarded.runpytest("-p", "duration_guard")
    assert result.ret == pytest.ExitCode.OK
    clocks = json.loads(out.read_text())
    wall, cpu = clocks["test_inner.py::test_burns"]
    assert cpu >= 0.1 and wall >= 0.1  # 0.12 burnt, less the clocks' tick on Windows
    assert "test_inner.py::test_burns_but_is_slow" not in clocks


def _summary(result: pytest.RunResult) -> str:
    text = result.stdout.str()
    return text[text.index("fast tests over"):]


def test_a_test_that_spends_its_own_time_fails_the_run(guarded: pytest.Pytester) -> None:
    guarded.makepyfile(test_inner=INNER)
    result = guarded.runpytest("-p", "duration_guard")
    said = _summary(result)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(passed=5)
    burns = next(line for line in said.splitlines() if "::test_burns " in line + " ")
    assert "FAILS" in burns
    assert "1 fast test(s) cost over 0.08 s by both clocks: the run fails" in said


def test_a_test_that_only_waits_is_named_and_not_failed(guarded: pytest.Pytester) -> None:
    """Its processor time is its own cost; the wall is what the machine made of it."""
    guarded.makepyfile(test_inner=INNER)
    said = _summary(guarded.runpytest("-p", "duration_guard"))
    waits = next(line for line in said.splitlines() if "::test_waits" in line)
    assert "FAILS" not in waits


def test_a_slow_test_and_a_shared_fixture_are_not_charged(guarded: pytest.Pytester) -> None:
    """The fixture is named on a line of its own, and not as the test that set it up."""
    guarded.makepyfile(test_inner=INNER)
    said = _summary(guarded.runpytest("-p", "duration_guard"))
    assert "test_burns_but_is_slow" not in said
    named = [line for line in said.splitlines() if "shared fixture" not in line]
    assert not any("::test_reads_the_shared_fixture" in line for line in named)
    assert "shared fixture test_inner.py::shared, set up for test_inner.py::test_reads" in said


def test_a_run_the_green_stamp_narrowed_names_and_does_not_judge(guarded: pytest.Pytester) -> None:
    """A gate that ran only what a change reached is mostly corpus sweeps paying warm-up a
    whole run shares (2026-10-22); the next whole run judges."""
    guarded.makeconftest(
        "class Narrowed:\n"
        "    def narrowed(self):\n"
        "        return True\n\n"
        "def pytest_configure(config):\n"
        "    config.pluginmanager.register(Narrowed(), 'green-stamp')\n"
    )
    guarded.makepyfile(test_inner=INNER)
    result = guarded.runpytest("-p", "duration_guard")
    said = _summary(result)
    assert result.ret == pytest.ExitCode.OK
    burns = next(line for line in said.splitlines() if "::test_burns " in line + " ")
    assert "FAILS" not in burns
    assert "in a narrowed run" in burns
    assert "--skip-if-green narrowed: named, not judged" in said


def test_a_run_with_nothing_over_says_nothing(guarded: pytest.Pytester) -> None:
    guarded.makepyfile(test_inner="def test_quick():\n    pass\n")
    result = guarded.runpytest("-p", "duration_guard")
    assert result.ret == pytest.ExitCode.OK
    assert "fast tests over" not in result.stdout.str()


def test_the_rule_is_two_seconds_and_a_failure_twice_it() -> None:
    assert (duration_guard.LIMIT_S, duration_guard.FAIL_S) == (2.0, 4.0)
