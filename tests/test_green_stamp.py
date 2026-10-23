"""The green stamp: a tree the fast suite passed on is not run again, and nothing else is spared.

Each case runs an inner pytest in a scratch git repository with `green_stamp` as its plugin,
so the stamp store is that repository's own `.claude/coordinator/green/`. The core checks
are replaced by stand-ins: whether this checkout's core happens to be current is not what
these tests are about.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import green_stamp
import pytest

REPOSITORY = Path(__file__).resolve().parents[1]

SUITE = """
import pytest

def test_passes():
    pass

@pytest.mark.slow
def test_slow():
    pass

def test_wants_a_database():
    pytest.skip("no database")
"""

FAST = ("-m", "not slow", "-p", "green_stamp")
GATE = (*FAST, "--skip-if-green")


def git(root: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=root, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def repository(root: Path, files: dict[str, str]) -> Path:
    files = {
        ".gitignore": ".claude/coordinator/\n__pycache__/\n.pytest_cache/\n",
        "pyproject.toml": (
            '[tool.pytest.ini_options]\ntestpaths = ["tests"]\nmarkers = ["slow: slow"]\n'
        ),
        **files,
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # LF on disk, so the repository reads clean whether or not git converts line
        # endings. A template is built before `pytester` hides the user's git config (and
        # its `core.autocrlf=true` on this machine) from what runs inside a test; with CRLF
        # on disk its copy read as three modified files and no stamp ever matched.
        path.write_text(text, encoding="utf-8", newline="\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "start")
    return root


def stamp(root: Path) -> dict | None:
    return green_stamp.read_stamp(root / green_stamp.STORE, git(root, "rev-parse", "HEAD^{tree}"))


@pytest.fixture(autouse=True)
def a_current_core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(green_stamp, "stale_core", lambda: None)
    monkeypatch.setattr(green_stamp, "built_core", lambda: "test-core")


def copied(template: Path, root: Path) -> Path:
    """A fresh repository from one built earlier: a copy of the directory, `.git` and all.

    Building one is three git processes, 0.3-0.5 s apiece on this machine, and most tests
    here start from the same one (Q377).
    """
    shutil.copytree(template, root, dirs_exist_ok=True)
    return root


@pytest.fixture(scope="module")
def suite_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return repository(tmp_path_factory.mktemp("suite"), {"tests/test_suite.py": SUITE})


@pytest.fixture
def suite(pytester: pytest.Pytester, suite_template: Path) -> Path:
    return copied(suite_template, pytester.path)


class TestTheGate:
    def test_a_green_run_is_stamped_and_the_same_tree_reruns_only_what_skipped(
        self, pytester: pytest.Pytester, suite: Path
    ) -> None:
        first = pytester.runpytest(*FAST)
        first.assert_outcomes(passed=1, skipped=1, deselected=1)
        first.stdout.fnmatch_lines(["green stamp: stamped tree * green"])
        written = stamp(suite)
        assert written is not None
        assert written["skipped"] == ["tests/test_suite.py::test_wants_a_database"]
        assert written["passed"] == 1
        assert written["core"] == "test-core"

        again = pytester.runpytest(*GATE)
        again.assert_outcomes(skipped=1, deselected=2)
        assert again.ret == pytest.ExitCode.OK
        again.stdout.fnmatch_lines(["green stamp: tree * re-ran only the 1 tests that run skipped"])

    def test_nothing_left_to_rerun_is_a_pass_not_an_empty_run(
        self, pytester: pytest.Pytester
    ) -> None:
        root = repository(pytester.path, {"tests/test_suite.py": "def test_passes():\n    pass\n"})
        pytester.runpytest(*FAST).assert_outcomes(passed=1)
        assert stamp(root) is not None
        result = pytester.runpytest(*GATE)
        assert result.ret == pytest.ExitCode.OK
        result.assert_outcomes(deselected=1)

    def test_an_uncommitted_edit_is_a_new_tree(
        self, pytester: pytest.Pytester, suite: Path
    ) -> None:
        pytester.runpytest(*FAST)
        (suite / "tests" / "test_suite.py").write_text(
            SUITE + "\ndef test_added():\n    pass\n", encoding="utf-8"
        )
        result = pytester.runpytest(*GATE)
        result.assert_outcomes(passed=2, skipped=1, deselected=1)
        result.stdout.fnmatch_lines(["green stamp: no stamp for tree *; ran the whole fast suite*"])

    def test_a_stamp_from_before_files_vouches_only_for_its_own_collection(
        self, pytester: pytest.Pytester, suite: Path
    ) -> None:
        """Until 2026-10-22 a stamp had one digest for the whole collection."""
        pytester.runpytest(*FAST)
        written = stamp(suite)
        assert written is not None
        del written["files"]
        written["collection"] = "elsewhere"
        green_stamp.write_stamp(suite / green_stamp.STORE, written)
        result = pytester.runpytest(*GATE)
        result.assert_outcomes(passed=1, skipped=1, deselected=1)
        result.stdout.fnmatch_lines(
            ["green stamp: no stamp for tree * vouches for any of its tests; ran the whole fast suite*"]
        )


class TestWhatIsNotStamped:
    def test_a_failing_run_stopped_early(self, pytester: pytest.Pytester) -> None:
        """`-x` never ran the rest, so its red says nothing about them."""
        root = repository(pytester.path, {"tests/test_suite.py": "def test_fails():\n    assert 0\n"})
        pytester.runpytest(*FAST, "-x").assert_outcomes(failed=1)
        assert stamp(root) is None

    def test_a_run_on_a_stale_core(
        self, pytester: pytest.Pytester, suite: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(green_stamp, "stale_core", lambda: "built from other sources")
        result = pytester.runpytest(*FAST)
        result.assert_outcomes(passed=1, skipped=1, deselected=1)
        assert stamp(suite) is None
        result.stdout.fnmatch_lines(["green stamp: not stamped: built from other sources"])

    def test_a_narrowed_run(self, pytester: pytest.Pytester, suite: Path) -> None:
        pytester.runpytest(*FAST, "tests/test_suite.py").assert_outcomes(passed=1, skipped=1)
        assert stamp(suite) is None

    def test_another_marker_expression(self, pytester: pytest.Pytester, suite: Path) -> None:
        pytester.runpytest("-p", "green_stamp").assert_outcomes(passed=2, skipped=1)
        assert stamp(suite) is None


EXPOSED = """
import os
import pytest

def test_passes():
    pass

def test_reads_the_environment():
    assert os.environ.get("GREEN_STAMP_STATE", "good") == "good"

def test_wants_a_database():
    if "GREEN_STAMP_DATABASE" not in os.environ:
        pytest.skip("no database")
    assert os.environ["GREEN_STAMP_DATABASE"] == "good"
"""


@pytest.fixture(scope="module")
def exposed_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A suite whose verdict the environment sets, as `runs/` and a database set the real one's,
    beside Markdown nothing reads and Markdown a module names."""
    return repository(
        tmp_path_factory.mktemp("exposed"),
        {
            "tests/test_suite.py": EXPOSED,
            "app.py": 'CHANGELOG = "docs/changelog.md"\n',
            "docs/changelog.md": "# Changes\n",
            "docs/notes.md": "# Notes\n",
        },
    )


@pytest.fixture
def exposed(
    pytester: pytest.Pytester, exposed_template: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.delenv("GREEN_STAMP_STATE", raising=False)
    monkeypatch.delenv("GREEN_STAMP_DATABASE", raising=False)
    return copied(exposed_template, pytester.path)


class TestRedStamps:
    """A red tree re-checked costs its failures, not the suite (2026-10-09)."""

    def test_a_red_run_is_stamped_and_its_gate_reruns_only_what_failed_or_skipped(
        self, pytester: pytest.Pytester, exposed: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GREEN_STAMP_STATE", "bad")
        first = pytester.runpytest(*FAST)
        first.assert_outcomes(passed=1, failed=1, skipped=1)
        first.stdout.fnmatch_lines(["green stamp: stamped tree * red (1 failed)"])
        written = stamp(exposed)
        assert written is not None
        assert written["verdict"] == "red"
        assert written["failed"] == ["tests/test_suite.py::test_reads_the_environment"]

        again = pytester.runpytest(*GATE)
        again.assert_outcomes(failed=1, skipped=1, deselected=1)
        assert again.ret == pytest.ExitCode.TESTS_FAILED
        again.stdout.fnmatch_lines(
            ["green stamp: tree * was red in * (1 failed); re-ran only the 2 tests that run"
             " failed or skipped; stamped tree * red"]
        )

        # the cause outside the tree is fixed: the rerun is green, and so is the tree now
        monkeypatch.delenv("GREEN_STAMP_STATE")
        fixed = pytester.runpytest(*GATE)
        fixed.assert_outcomes(passed=1, skipped=1, deselected=1)
        assert fixed.ret == pytest.ExitCode.OK
        fixed.stdout.fnmatch_lines(["green stamp: * stamped tree * green"])
        written = stamp(exposed)
        assert written is not None
        assert written["verdict"] == "green"
        assert written["failed"] == []
        assert written["skipped"] == ["tests/test_suite.py::test_wants_a_database"]
        assert written["passed"] == 2
        assert written["rerun_of"]["verdict"] == "red"

        last = pytester.runpytest(*GATE)
        last.assert_outcomes(skipped=1, deselected=2)
        last.stdout.fnmatch_lines(["green stamp: tree * re-ran only the 1 tests that run skipped"])

    def test_a_green_stamp_whose_rerun_fails_turns_red(
        self, pytester: pytest.Pytester, exposed: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A worker's green, then the main checkout's database test fails: the tree is red."""
        pytester.runpytest(*FAST).assert_outcomes(passed=2, skipped=1)
        monkeypatch.setenv("GREEN_STAMP_DATABASE", "bad")
        gate = pytester.runpytest(*GATE)
        gate.assert_outcomes(failed=1, deselected=2)
        gate.stdout.fnmatch_lines(["green stamp: * stamped tree * red"])
        written = stamp(exposed)
        assert written is not None
        assert written["verdict"] == "red"
        assert written["failed"] == ["tests/test_suite.py::test_wants_a_database"]


class TestMarkdownDrift:
    """A committed edit to Markdown nothing reads doesn't cost a full run (2026-10-09)."""

    def test_a_committed_note_nothing_reads_still_matches(
        self, pytester: pytest.Pytester, exposed: Path
    ) -> None:
        pytester.runpytest(*FAST).assert_outcomes(passed=2, skipped=1)
        stamped = git(exposed, "rev-parse", "HEAD^{tree}")
        (exposed / "docs" / "notes.md").write_text("# Notes\n- after the run\n", encoding="utf-8")
        git(exposed, "commit", "-qam", "a note after the run")
        # a stamp named for a tree this repository never had is passed over, not an error
        green_stamp.write_stamp(
            exposed / green_stamp.STORE, {"tree": "0" * 40, "collection": "x", "skipped": []}
        )

        result = pytester.runpytest(*GATE)
        result.assert_outcomes(skipped=1, deselected=2)
        result.stdout.fnmatch_lines(
            [f"green stamp: tree * differs from tree {stamped[:10]} only in Markdown nothing"
             " reads (docs/notes.md); * re-ran only the 1 tests that run skipped"]
        )

    def test_a_committed_edit_to_markdown_code_reads_is_a_change_not_drift(
        self, pytester: pytest.Pytester, exposed: Path
    ) -> None:
        """`app.py` reads the changelog, so the stamp is not the same code's: it vouches only
        where the change can't reach, and the run that leans on it owes debt. No test here
        imports `app.py`, so that is every test that passed."""
        pytester.runpytest(*FAST)
        (exposed / "docs" / "changelog.md").write_text("# Changes\n- one\n", encoding="utf-8")
        git(exposed, "commit", "-qam", "the changelog, which app.py reads")
        result = pytester.runpytest(*GATE)
        result.assert_outcomes(skipped=1, deselected=2)
        result.stdout.fnmatch_lines(["green stamp: tree *: 2 tests vouched for, *; debt 1 of 5;*"])
        written = stamp(exposed)
        assert written is not None
        assert written["debt"] == 1


GRAPH = {
    "tests/alpha.py": "VALUE = 1\n",
    "tests/beta.py": "VALUE = 1\n",
    "tests/reader.py": 'CHANGELOG = "docs/changelog.md"\n',
    "tests/test_alpha.py": "import alpha\n\ndef test_alpha():\n    assert alpha.VALUE == 1\n",
    "tests/test_beta.py": "import beta\n\ndef test_beta():\n    assert beta.VALUE == 1\n",
    "tests/test_reader.py": "import reader\n\ndef test_reader():\n    assert reader.CHANGELOG\n",
    "docs/changelog.md": "# Changes\n",
}


@pytest.fixture(scope="module")
def graph_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Three test files, each reached by one module of its own."""
    return repository(tmp_path_factory.mktemp("graph"), GRAPH)


@pytest.fixture
def graph(pytester: pytest.Pytester, graph_template: Path) -> Path:
    return copied(graph_template, pytester.path)


def commit(root: Path, path: str, text: str) -> str:
    """Commit `text` as `path`; the tree it makes."""
    (root / path).write_text(text, encoding="utf-8", newline="\n")
    git(root, "add", "-A")
    git(root, "commit", "-qm", f"edit {path}")
    return git(root, "rev-parse", "HEAD^{tree}")


class TestLeaningOnAnotherTree:
    """A stamp vouches for what the change from its tree cannot reach (2026-10-22)."""

    def test_a_change_reruns_only_the_tests_it_reaches(
        self, pytester: pytest.Pytester, graph: Path
    ) -> None:
        pytester.runpytest(*FAST).assert_outcomes(passed=3)
        before = git(graph, "rev-parse", "HEAD^{tree}")
        commit(graph, "tests/alpha.py", "VALUE = 1\nOTHER = 2\n")
        result = pytester.runpytest(*GATE)
        result.assert_outcomes(passed=1, deselected=2)
        result.stdout.fnmatch_lines(
            [f"green stamp: tree *: 2 tests vouched for, 2 by tree {before[:10]} (*; 1 paths"
             " apart, reaching 1 test files); ran the other 1; debt 1 of 5; stamped tree * green"]
        )
        written = stamp(graph)
        assert written is not None
        assert written["debt"] == 1
        assert written["vouched_by"] == {before: 2}
        assert written["passed"] == 3
        assert written["full_run"]["tree"] == before

    def test_what_the_change_breaks_is_run_and_found(
        self, pytester: pytest.Pytester, graph: Path
    ) -> None:
        pytester.runpytest(*FAST).assert_outcomes(passed=3)
        commit(graph, "tests/alpha.py", "VALUE = 2\n")
        result = pytester.runpytest(*GATE)
        result.assert_outcomes(failed=1, deselected=2)
        written = stamp(graph)
        assert written is not None
        assert written["verdict"] == "red"
        assert written["failed"] == ["tests/test_alpha.py::test_alpha"]

    def test_markdown_code_reads_reaches_its_readers(
        self, pytester: pytest.Pytester, graph: Path
    ) -> None:
        pytester.runpytest(*FAST).assert_outcomes(passed=3)
        commit(graph, "docs/changelog.md", "# Changes\n- one\n")
        result = pytester.runpytest(*GATE)
        result.assert_outcomes(passed=1, deselected=2)
        assert "test_reader.py" in result.stdout.str().split("green stamp:")[0]

    def test_a_merge_reruns_only_what_both_sides_reach(
        self, pytester: pytest.Pytester, graph: Path
    ) -> None:
        """Each side's stamp covers what the other side changed, so neither change alone
        sends a test back; only a test both reach would run."""
        main = git(graph, "branch", "--show-current")
        git(graph, "checkout", "-qb", "side")
        side = commit(graph, "tests/alpha.py", "VALUE = 1\nOTHER = 2\n")
        pytester.runpytest(*FAST).assert_outcomes(passed=3)
        git(graph, "checkout", "-q", main)
        ours = commit(graph, "tests/beta.py", "VALUE = 1\nOTHER = 3\n")
        pytester.runpytest(*FAST).assert_outcomes(passed=3)
        git(graph, "merge", "-q", "--no-edit", "side")

        result = pytester.runpytest(*GATE)
        result.assert_outcomes(deselected=3)
        assert result.ret == pytest.ExitCode.OK
        written = stamp(graph)
        assert written is not None
        assert set(written["vouched_by"]) == {side, ours}
        assert written["debt"] == 1

    def test_a_test_that_failed_on_a_stamp_leaned_on_is_run(
        self, pytester: pytest.Pytester, graph: Path
    ) -> None:
        pytester.runpytest(*FAST).assert_outcomes(passed=3)
        written = stamp(graph)
        assert written is not None
        written.update(verdict="red", failed=["tests/test_beta.py::test_beta"])
        green_stamp.write_stamp(graph / green_stamp.STORE, written)
        commit(graph, "tests/alpha.py", "VALUE = 1\nOTHER = 2\n")
        result = pytester.runpytest(*GATE)
        result.assert_outcomes(passed=2, deselected=1)

    def test_a_test_a_worker_skipped_runs_whatever_an_older_stamp_says(
        self, pytester: pytest.Pytester, graph: Path
    ) -> None:
        """A worker skips what reads the main checkout's `runs/`; an older main stamp that
        passed it must not vouch for it on the worker's tree."""
        pytester.runpytest(*FAST).assert_outcomes(passed=3)  # main, earlier: all passed
        commit(graph, "tests/alpha.py", "VALUE = 1\nOTHER = 2\n")
        pytester.runpytest(*FAST).assert_outcomes(passed=3)  # the worker's run...
        written = stamp(graph)
        assert written is not None
        written["skipped"] = ["tests/test_beta.py::test_beta"]  # ...which had no `runs/`
        green_stamp.write_stamp(graph / green_stamp.STORE, written)
        result = pytester.runpytest(*GATE)
        result.assert_outcomes(passed=1, deselected=2)
        assert "test_beta.py" in result.stdout.str().split("green stamp:")[0]

    def test_a_test_that_skipped_on_a_recent_stamp_runs_on_other_code(
        self, pytester: pytest.Pytester, graph: Path
    ) -> None:
        """A worktree skips what reads the main checkout's `runs/`, so a skip anywhere lately
        marks a test no change between trees can speak for (a delivered record, 2026-10-20)."""
        main = git(graph, "branch", "--show-current")
        pytester.runpytest(*FAST).assert_outcomes(passed=3)  # main: all passed
        git(graph, "checkout", "-qb", "worktree")
        commit(graph, "tests/conftest.py", "")  # a change that reaches everything: no base
        pytester.runpytest(*FAST).assert_outcomes(passed=3)
        written = stamp(graph)
        assert written is not None
        written["skipped"] = ["tests/test_beta.py::test_beta"]  # it had no `runs/`
        green_stamp.write_stamp(graph / green_stamp.STORE, written)
        git(graph, "checkout", "-q", main)
        commit(graph, "tests/reader.py", 'CHANGELOG = "docs/changelog.md"\nMORE = 1\n')
        result = pytester.runpytest(*GATE)
        result.assert_outcomes(passed=2, deselected=1)
        assert "test_beta.py" in result.stdout.str().split("green stamp:")[0]

    def test_a_file_that_collects_differently_runs_whole(
        self, pytester: pytest.Pytester, graph: Path
    ) -> None:
        """Same tree, other ids in one file: what `runs/puzzles/` does to `test_site_proven.py`
        between a worktree and the main checkout."""
        pytester.runpytest(*FAST).assert_outcomes(passed=3)
        written = stamp(graph)
        assert written is not None
        written["files"]["tests/test_beta.py"] = "elsewhere"
        green_stamp.write_stamp(graph / green_stamp.STORE, written)
        result = pytester.runpytest(*GATE)
        result.assert_outcomes(passed=1, deselected=2)
        result.stdout.fnmatch_lines(
            ["green stamp: tree * re-ran only the 1 tests that run skipped (and 1 whose file"
             " collects differently here)"]
        )

    def test_at_the_debt_limit_a_stamp_vouches_only_for_its_own_tree(
        self, pytester: pytest.Pytester, graph: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(green_stamp, "DEBT_LIMIT", 1)
        pytester.runpytest(*FAST).assert_outcomes(passed=3)
        first = git(graph, "rev-parse", "HEAD^{tree}")
        leaning = commit(graph, "tests/alpha.py", "VALUE = 1\nOTHER = 2\n")
        pytester.runpytest(*GATE).assert_outcomes(passed=1, deselected=2)
        (graph / green_stamp.STORE / f"{first}.json").unlink()  # the full run it leaned on
        # the same code: debt is no reason to run it again
        pytester.runpytest(*GATE).assert_outcomes(deselected=3)
        assert git(graph, "rev-parse", "HEAD^{tree}") == leaning

        commit(graph, "tests/beta.py", "VALUE = 1\nOTHER = 3\n")
        result = pytester.runpytest(*GATE)
        result.assert_outcomes(passed=3)
        result.stdout.fnmatch_lines([f"green stamp: no stamp for tree *; ran the whole fast suite (tree {leaning[:10]} is at debt 1, the limit)*"])
        written = stamp(graph)
        assert written is not None
        assert written["debt"] == 0

    def test_a_full_run_too_old_vouches_only_for_its_own_tree(
        self, pytester: pytest.Pytester, graph: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(green_stamp, "FULL_EVERY_HOURS", 0)
        pytester.runpytest(*FAST).assert_outcomes(passed=3)
        pytester.runpytest(*GATE).assert_outcomes(deselected=3)
        commit(graph, "tests/alpha.py", "VALUE = 1\nOTHER = 2\n")
        result = pytester.runpytest(*GATE)
        result.assert_outcomes(passed=3)
        result.stdout.fnmatch_lines(["green stamp: no stamp for tree *; ran the whole fast suite (*over 0 h ago)*"])

    def test_a_red_full_run_names_where_to_look(
        self, pytester: pytest.Pytester, graph: Path
    ) -> None:
        pytester.runpytest(*FAST).assert_outcomes(passed=3)
        green = git(graph, "rev-parse", "HEAD")
        commit(graph, "tests/alpha.py", "VALUE = 2\n")
        result = pytester.runpytest(*FAST)
        result.assert_outcomes(passed=2, failed=1)
        result.stdout.fnmatch_lines(
            [f"green stamp: stamped tree * red (1 failed); the last full green run on this line"
             f" was {green[:10]} (*): look in git log {green[:10]}..HEAD"]
        )

    def test_the_plan_says_what_the_gate_would_run(
        self, pytester: pytest.Pytester, graph: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pytester.runpytest(*FAST).assert_outcomes(passed=3)
        commit(graph, "tests/alpha.py", "VALUE = 1\nOTHER = 2\n")
        capsys.readouterr()
        assert green_stamp.show_plan(graph) == 0
        said = capsys.readouterr().out
        assert "would re-run 1 of 3 test files" in said
        assert "tests/test_alpha.py" in said
        assert "debt after: 1 of 5" in said


class TestATravellingStamp:
    """A cloud clone's stamp, carried home on a branch of its own (2026-10-09)."""

    def test_a_stamp_exported_from_a_clone_is_imported_and_matched(
        self, pytester: pytest.Pytester, suite: Path, tmp_path: Path
    ) -> None:
        branch = git(suite, "branch", "--show-current")
        head = git(suite, "rev-parse", "HEAD")
        cloud = tmp_path / "cloud"
        git(tmp_path, "clone", "-q", str(suite), str(cloud))
        # the clone's run, stood in for by moving this checkout's stamp into its store
        pytester.runpytest(*FAST).assert_outcomes(passed=1, skipped=1, deselected=1)
        shutil.move(str(suite / green_stamp.STORE), str(cloud / green_stamp.STORE))
        assert stamp(suite) is None

        assert green_stamp.export(cloud, push="origin") == 0
        assert green_stamp.import_(suite, f"{branch}-green") == 0

        # the code is untouched: the stamp rode outside its history and tree
        assert git(suite, "rev-parse", "HEAD") == head
        assert git(suite, "status", "--porcelain") == ""
        carried = git(suite, "ls-tree", "--name-only", f"{branch}-green")
        assert carried == green_stamp.CARRIED

        result = pytester.runpytest(*GATE)
        result.assert_outcomes(skipped=1, deselected=2)
        result.stdout.fnmatch_lines(
            [f"green stamp: tree * passed the fast suite in {branch}-green at *;"
             " re-ran only the 1 tests that run skipped"]
        )

    def test_nothing_to_export_says_so(self, suite: Path) -> None:
        assert green_stamp.export(suite, push=None) == 1

    def test_a_branch_without_a_stamp_imports_nothing(self, suite: Path) -> None:
        assert green_stamp.import_(suite, "HEAD") == 1
        assert not (suite / green_stamp.STORE).exists()


class TestTheGateRefuses:
    @pytest.mark.parametrize(
        "narrowing",
        [("tests/test_suite.py",), ("-k", "passes"), ("--deselect", "tests/test_suite.py::test_passes")],
    )
    def test_a_narrowed_run(
        self, pytester: pytest.Pytester, suite: Path, narrowing: tuple[str, ...]
    ) -> None:
        result = pytester.runpytest(*GATE, *narrowing)
        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(["*--skip-if-green vouches only for the whole fast suite*"])

    def test_a_stale_core(
        self, pytester: pytest.Pytester, suite: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(green_stamp, "stale_core", lambda: "built from other sources")
        result = pytester.runpytest(*GATE)
        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(["*will not run on a stale core: built from other sources*"])


class TestTheKey:
    """What moves the key. Plain git, no inner pytest."""

    @pytest.fixture(scope="class")
    @classmethod
    def template(cls, tmp_path_factory: pytest.TempPathFactory) -> Path:
        return repository(
            tmp_path_factory.mktemp("key"),
            {
                "app.py": 'CHANGELOG = "docs/changelog.md"\n',
                "docs/changelog.md": "# Changes\n",
                "docs/add-to-queue.md": "# Inbox\n",
            },
        )

    @pytest.fixture
    def root(self, template: Path, tmp_path: Path) -> Path:
        return copied(template, tmp_path / "repository")

    def test_a_clean_checkout_is_its_head(self, root: Path) -> None:
        tree = green_stamp.tree_of(root)
        assert tree.key == git(root, "rev-parse", "HEAD^{tree}")
        assert tree.exempt == ()

    def test_an_inbox_line_nothing_reads_is_left_out(self, root: Path) -> None:
        (root / "docs" / "add-to-queue.md").write_text("# Inbox\n- a task\n", encoding="utf-8")
        (root / "docs" / "notes.md").write_text("untracked\n", encoding="utf-8")
        tree = green_stamp.tree_of(root)
        assert tree.key == git(root, "rev-parse", "HEAD^{tree}")
        assert set(tree.exempt) == {"docs/add-to-queue.md", "docs/notes.md"}

    @pytest.mark.parametrize(
        "path", ["docs/changelog.md", "app.py", "new.py", "rulesets/new.json"]
    )
    def test_anything_code_could_read_moves_it(self, root: Path, path: str) -> None:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("changed\n", encoding="utf-8")
        (root / "docs" / "add-to-queue.md").write_text("# Inbox\n- a task\n", encoding="utf-8")
        tree = green_stamp.tree_of(root)
        assert tree.key != git(root, "rev-parse", "HEAD^{tree}")
        assert tree.exempt == ()

    def test_the_real_index_is_untouched(self, root: Path) -> None:
        (root / "new.py").write_text("x = 1\n", encoding="utf-8")
        before = git(root, "status", "--porcelain")
        green_stamp.tree_of(root)
        assert git(root, "status", "--porcelain") == before == "?? new.py"


# Slow (Q749): each run starts real xdist workers, 3.5-3.8 s a test on one core and 4-6 s
# in the gate; the gate itself stamps through the same controller every run.
@pytest.mark.slow
class TestUnderXdist:
    """How the fast suite runs since Q377c: `-n`, with only the controller stamping.

    The inner workers are processes of their own, so they find the plugin through a
    conftest that puts this directory on their path; the controller runs here, in-process,
    where the core stand-ins above apply -- and it is the only side that consults the core.
    """

    WORKERS = ("-n", "2", "-m", "not slow")

    @staticmethod
    def conftest() -> str:
        here = str(Path(__file__).resolve().parent)
        return f"import sys\nsys.path.insert(0, {here!r})\npytest_plugins = ('green_stamp',)\n"

    @pytest.fixture(scope="class")
    @classmethod
    def parallel_template(cls, tmp_path_factory: pytest.TempPathFactory) -> Path:
        return repository(
            tmp_path_factory.mktemp("parallel"),
            {"conftest.py": cls.conftest(), "tests/test_suite.py": SUITE},
        )

    def test_the_controller_stamps_once_and_either_mode_reads_the_other(
        self, pytester: pytest.Pytester, parallel_template: Path
    ) -> None:
        root = copied(parallel_template, pytester.path)
        first = pytester.runpytest(*self.WORKERS)
        first.assert_outcomes(passed=1, skipped=1)
        first.stdout.fnmatch_lines(["green stamp: stamped tree * green"])
        written = stamp(root)
        assert written is not None
        # the controller counted what the workers reported, as a serial run counts its own
        assert written["passed"] == 1
        assert written["skipped"] == ["tests/test_suite.py::test_wants_a_database"]
        assert len(list((root / green_stamp.STORE).glob("*.json"))) == 1

        # the digest is the serial run's, so a serial gate trusts a parallel stamp...
        serial = pytester.runpytest("-m", "not slow", "--skip-if-green")
        serial.assert_outcomes(skipped=1)
        serial.stdout.fnmatch_lines(["green stamp: tree * re-ran only the 1 tests that run skipped"])
        # ...and a parallel gate re-runs only what skipped, deselected on every worker
        again = pytester.runpytest(*self.WORKERS, "--skip-if-green")
        again.assert_outcomes(skipped=1)
        assert again.ret == pytest.ExitCode.OK
        again.stdout.fnmatch_lines(["green stamp: tree * re-ran only the 1 tests that run skipped"])

    def test_a_worker_whose_share_passed_does_not_stamp_a_red_run(
        self, pytester: pytest.Pytester
    ) -> None:
        """The defect this class exists for: each worker used to stamp its own share.

        `--dist loadfile` gives each file its own worker, so one worker's share is all
        green while the other's fails. Before Q377c that worker wrote the stamp, and the
        next gate on the same tree skipped the failing test.
        """
        root = repository(
            pytester.path,
            {
                "conftest.py": self.conftest(),
                "tests/test_green.py": "def test_one():\n    pass\n\ndef test_two():\n    pass\n",
                "tests/test_red.py": "def test_fails():\n    assert 0\n",
            },
        )
        result = pytester.runpytest(*self.WORKERS, "--dist", "loadfile")
        result.assert_outcomes(passed=2, failed=1)
        written = stamp(root)
        assert written is not None
        assert written["verdict"] == "red"
        assert written["failed"] == ["tests/test_red.py::test_fails"]
        assert len(list((root / green_stamp.STORE).glob("*.json"))) == 1

        # ...and the next gate on the tree runs the failure again rather than skipping it
        again = pytester.runpytest(*self.WORKERS, "--skip-if-green")
        again.assert_outcomes(failed=1)
        assert again.ret == pytest.ExitCode.TESTS_FAILED


def test_the_inbox_is_one_nothing_reads() -> None:
    """The main checkout usually has an uncommitted line here, and the coordinator's gate runs
    there. Were the inbox named in any Python file, every post-merge check would run in full.
    If code comes to read it on purpose, delete this test: the gate is then right to run.
    """
    assert green_stamp.unread(REPOSITORY, "docs/add-to-queue.md")


@pytest.mark.parametrize("path", ["docs/queue.md", "docs/coordinator-status.md"])
def test_the_coordinators_own_documents_are_ones_nothing_reads(path: str) -> None:
    """The coordinator commits these between a worker's run and its gate. Were they read, the
    gate after a queue commit would run in full (2026-10-09). Same remedy as the inbox's test."""
    assert green_stamp.unread(REPOSITORY, path)


def test_the_store_is_ignored_so_a_stamp_cannot_move_a_tree() -> None:
    """Were it tracked, writing a stamp would change the next run's content tree."""
    probe = (green_stamp.STORE / "0123.json").as_posix()
    ignored = subprocess.run(["git", "check-ignore", "-q", probe], cwd=REPOSITORY)
    assert ignored.returncode == 0
