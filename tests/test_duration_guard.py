"""`tests/duration_guard.py` names a fast test over the limit and fails one over twice it.

Each case runs a small suite inside this one (`pytester`), with the limits shrunk to tenths
of a second so the suite inside spends a fraction of a second: a test that burns processor
time, one that only waits, one marked `slow`, and a shared fixture that is expensive.
"""

from __future__ import annotations

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
    pytester.makeini("[pytest]\nmarkers =\n    slow: slow\n")
    return pytester


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


def test_a_run_with_nothing_over_says_nothing(guarded: pytest.Pytester) -> None:
    guarded.makepyfile(test_inner="def test_quick():\n    pass\n")
    result = guarded.runpytest("-p", "duration_guard")
    assert result.ret == pytest.ExitCode.OK
    assert "fast tests over" not in result.stdout.str()


def test_the_rule_is_two_seconds_and_a_failure_twice_it() -> None:
    assert (duration_guard.LIMIT_S, duration_guard.FAIL_S) == (2.0, 4.0)
