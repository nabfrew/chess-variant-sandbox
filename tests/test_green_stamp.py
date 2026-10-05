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

    def test_a_different_collection_runs_everything(
        self, pytester: pytest.Pytester, suite: Path
    ) -> None:
        """Same tree, other tests: what a suite parametrised over a checkout's files does."""
        pytester.runpytest(*FAST)
        written = stamp(suite)
        assert written is not None
        written["collection"] = "elsewhere"
        green_stamp.write_stamp(suite / green_stamp.STORE, written)
        result = pytester.runpytest(*GATE)
        result.assert_outcomes(passed=1, skipped=1, deselected=1)
        result.stdout.fnmatch_lines(["*collects a different set of tests*"])


class TestWhatIsNotStamped:
    def test_a_failing_run(self, pytester: pytest.Pytester) -> None:
        root = repository(pytester.path, {"tests/test_suite.py": "def test_fails():\n    assert 0\n"})
        pytester.runpytest(*FAST).assert_outcomes(failed=1)
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
        assert stamp(root) is None
        assert not (root / green_stamp.STORE).exists() or not list(
            (root / green_stamp.STORE).glob("*.json")
        )


def test_the_inbox_is_one_nothing_reads() -> None:
    """The main checkout usually has an uncommitted line here, and the coordinator's gate runs
    there. Were the inbox named in any Python file, every post-merge check would run in full.
    If code comes to read it on purpose, delete this test: the gate is then right to run.
    """
    assert green_stamp.unread(REPOSITORY, "docs/add-to-queue.md")


def test_the_store_is_ignored_so_a_stamp_cannot_move_a_tree() -> None:
    """Were it tracked, writing a stamp would change the next run's content tree."""
    probe = (green_stamp.STORE / "0123.json").as_posix()
    ignored = subprocess.run(["git", "check-ignore", "-q", probe], cwd=REPOSITORY)
    assert ignored.returncode == 0
