"""`.claude/scripts/basis.py` prints the status file's Basis from code (R-T4a).

Each fact is read from where it lives, never retyped: the weights version from the scorer, the
core's fingerprint from the measurement module, the blind count from R-P4's coverage file, and
main against GitHub from git. These tests hold each reading to its source and the report to
its shape; none runs the child under `uv run` or reads the main checkout's `runs/`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import private_paths
import pytest

from chess_variant_sandbox import cli, measurement
from chess_variant_sandbox.metrics.scoring import DEFAULT_WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ".claude/scripts/basis.py"
COVERAGE_SCRIPT = ".claude/scripts/evaluator_coverage.py"
COVERAGE_FILE = "docs/reference/evaluator-coverage.toml"

basis = private_paths.script(SCRIPT, "basis")
evaluator_coverage = private_paths.script(COVERAGE_SCRIPT, "evaluator_coverage")
pytestmark = private_paths.needs(SCRIPT, COVERAGE_SCRIPT, COVERAGE_FILE)

CHECK_OUTPUT = """\
  wolves                 [ceb4916bdabc] current, 4 runs, re-certified: sample 128/1024 on c6a32df2d078aad3
  wormhole               [3732e55621b1] stale: the core moved

and the core itself: the loaded core was built from different sources than rust/src/

1 of 140 need work
"""


def test_the_library_verdict_is_measure_checks_last_line() -> None:
    read = basis.library_check(CHECK_OUTPUT)
    assert read["stale"] == 1 and read["games"] == 140
    assert read["core_behind"].startswith("the loaded core was built")
    assert read["certified_on"] == ["c6a32df2d078aad3"]
    assert basis.library_check("0 of 140 need work\n")["core_behind"] is None
    assert "error" in basis.library_check("nothing to measure\n")


def test_the_package_facts_come_from_the_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "main", lambda argv: print("0 of 3 need work") or 0)
    facts = basis.package_facts(ROOT)
    assert facts["weights"] == DEFAULT_WEIGHTS
    assert facts["core"]["sources"] == measurement.rust_source_fingerprint()
    assert facts["core"]["built"] == measurement.built_fingerprint()
    assert facts["library"]["stale"] == 0 and facts["library"]["games"] == 3
    expected = evaluator_coverage.counts(evaluator_coverage.load(ROOT), evaluator_coverage.rule_keys())
    assert facts["blind"]["counts"] == expected
    assert facts["blind"]["summary"] == evaluator_coverage.summary(expected)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_main_against_github_counts_both_ways(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "base")
    _git(repo, "update-ref", "refs/remotes/Hub/main", "HEAD")
    _git(repo, "config", "branch.main.remote", "Hub")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "local one")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "local two")
    assert basis.github(repo) == {"remote": "Hub/main", "ahead": 2, "behind": 0}
    _git(repo, "config", "branch.main.remote", "Nowhere")
    assert "error" in basis.github(repo)


def _facts(**over: Any) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "main_checkout": "C:/main",
        "read_at": "2026-11-20T12:00+01:00",
        "disk": {"free_mb": 48_000, "floor_mb": 16_000},
        "github": {"remote": "GameDesign/main", "ahead": 0, "behind": 0},
        "weights": "v15",
        "core": {"sources": "c6a32df2d078aad3", "built": "c6a32df2d078aad3"},
        "library": {"stale": 0, "games": 140, "core_behind": None, "certified_on": []},
        "blind": {
            "counts": {"blind": 113, "partly": 167, "covered": 18, "n/a": 24},
            "summary": "blind 113, partly 167, covered 18 (n/a 24)",
        },
    }
    facts.update(over)
    return facts


def test_the_report_states_every_basis_fact() -> None:
    text = "\n".join(basis.lines(_facts()))
    for fact in ("48,000 MB", "0 ahead of GameDesign/main", "v15", "c6a32df2d078aad3",
                 "the built core matches", "current on it: 0 of 140", "blind to 113"):
        assert fact in text
    assert "unread" not in text
    assert basis.unread(_facts()) == []


def test_the_report_flags_what_is_owed() -> None:
    text = "\n".join(basis.lines(_facts(
        disk={"free_mb": 9_000, "floor_mb": 16_000},
        github={"remote": "GameDesign/main", "ahead": 3, "behind": 0},
        core={"sources": "aaaa", "built": "bbbb"},
        library={"stale": 2, "games": 140, "core_behind": "rebuild", "certified_on": []},
    )))
    assert "UNDER THE FLOOR" in text
    assert "a push is owed" in text
    assert "the built core is bbbb" in text
    assert "NOT current: 2 of 140" in text


def test_a_fact_that_cannot_be_read_is_named_and_fails_the_exit() -> None:
    facts = _facts(blind={"error": "no file"})
    del facts["weights"]
    facts["weights_error"] = "ImportError"
    gaps = basis.unread(facts)
    assert any(g.startswith("weights") for g in gaps)
    assert any(g.startswith("blind") for g in gaps)
    whole = {k: v for k, v in _facts().items() if k not in ("weights", "core", "library", "blind")}
    assert basis.unread({**whole, "error": "uv missing"}) == ["the package: uv missing"]


def test_the_script_declares_itself_light() -> None:
    assert private_paths.path(SCRIPT).read_text(encoding="utf-8").startswith("# compute-lease: light\n")
