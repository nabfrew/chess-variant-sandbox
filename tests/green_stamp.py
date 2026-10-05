"""Remember which trees the fast suite passed on, so an identical tree is not run twice.

A task used to end with `pytest -m "not slow"` three or four times over the same code: once
before merging main, once after, again when main moved during that run, and once more in
the main checkout after the coordinator fast-forwarded it. Each is about seven minutes on a
loaded machine, and the last one tested, byte for byte, a tree the worker had just tested.

So every green run of the whole fast suite leaves a *stamp*: a small JSON file named after
the git tree of the files it ran against, under the main checkout's
`.claude/coordinator/green/`, where every worktree sees it. `--skip-if-green` reads it back.
On a matching tree it re-runs only the tests the stamped run skipped and deselects the
rest. In a worktree those are the tests that want a database or the main checkout's
`runs/`, and a skip of the second kind is how a worker's green once hid a red main.

What a stamp is keyed on, and why it can be trusted:

- **The content tree**: what `git add -A` would commit from the working copy, built in a
  scratch index so the real one is untouched. Uncommitted and untracked files count, so a
  stamp vouches for what was on disk rather than for HEAD. The tree is taken before and
  after the run, and a run whose files moved underneath it is not stamped.
- **Except Markdown under `docs/` that no tracked Python file names.** The main checkout
  usually carries an uncommitted line in the queue's inbox, and a key that counted it would
  never match there. `docs/changelog.md` is served by the site and read by its tests, so
  it counts; the name search is what tells the two apart. When every uncommitted path is
  of this kind the key is HEAD's tree, and the summary line names what it left out.
- **The core the run loaded was built from this tree's `rust/src/`.** A run on a stale core
  is not stamped, and `--skip-if-green` refuses to start on one.
- **The same collection**: a digest of the selected test ids. A test parametrised over a
  checkout's files collects differently in another checkout, and then the whole suite runs.

A stamp says the code was green. It does not vouch for anything outside the tree that has
changed since, such as a delivered measurement or an environment variable, beyond
re-running what skipped.

**Under xdist (`-n 4`, how the fast suite runs since Q377c)** the work is split, and so is
this plugin. The *controller* keys the tree once, hands it to each worker, counts the
passes and skips the workers report back, and alone decides and writes the stamp, from
the whole run's exit status. Each *worker* collects the whole suite, computes the same
collection digest, applies the gate's deselection to its own copy of the collection, and
reports what it did through `workeroutput`. A worker never stamps: its part of a run can
be green while another worker's is red.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

FAST_SUITE = "not slow"
STORE = Path(".claude") / "coordinator" / "green"
KEEP_DAYS = 30
# The key this plugin uses in xdist's `workerinput` (controller to worker) and
# `workeroutput` (worker to controller).
XDIST_KEY = "green_stamp"
# Files that name docs only as examples. Searched, they found the inbox in this module's own
# tests and made its line count in the main checkout, so the coordinator's gate never matched.
NOT_READERS = ("tests/green_stamp.py", "tests/test_green_stamp.py")


class NotAGitCheckout(Exception):
    """git could not answer: no repository, no HEAD, no git."""


def _git(root: Path, *args: str, env: dict[str, str] | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True,
            env=env, timeout=60,
        )
    except (subprocess.SubprocessError, OSError) as error:
        raise NotAGitCheckout(" ".join(args)) from error
    return out.stdout.strip()


def _absolute(root: Path, answer: str) -> Path:
    path = Path(answer)
    return path if path.is_absolute() else (root / path).resolve()


def content_tree(root: Path, index: Path) -> str:
    """The tree `git add -A` would commit now, without touching the real index."""
    with tempfile.TemporaryDirectory() as scratch:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(scratch) / "index")}
        if index.exists():
            # A copy keeps the stat cache, so only files that changed are hashed.
            shutil.copyfile(index, env["GIT_INDEX_FILE"])
        _git(root, "add", "-A", env=env)
        return _git(root, "write-tree", env=env)


def unread(root: Path, path: str) -> bool:
    """Whether an uncommitted path is one the suite cannot be reading.

    Markdown under `docs/` whose file name appears in no tracked Python file outside
    `NOT_READERS`. A substring search errs towards *read*: `queue.md` is read if a file
    names `add-to-queue.md`.
    """
    if not (path.startswith("docs/") and path.endswith(".md")):
        return False
    name = path.rsplit("/", 1)[-1]
    excluded = [f":(exclude){reader}" for reader in NOT_READERS]
    found = subprocess.run(
        ["git", "grep", "-l", "-F", "-e", name, "--", "*.py", *excluded],
        cwd=root, capture_output=True, text=True, timeout=60,
    )
    return found.returncode == 1  # 0 is a match; anything above 1 is git failing


@dataclass(frozen=True)
class Tree:
    key: str  # the tree a stamp is filed under
    head: str  # the commit checked out
    exempt: tuple[str, ...]  # uncommitted paths the key leaves out


def tree_of(root: Path) -> Tree:
    # One `rev-parse` for all three: git is a process per call, and slow to start on Windows.
    index, head, committed = _git(
        root, "rev-parse", "--git-path", "index", "HEAD", "HEAD^{tree}"
    ).splitlines()
    content = content_tree(root, _absolute(root, index))
    if content == committed:
        return Tree(content, head, ())
    changed = _git(root, "diff-tree", "-r", "--name-only", committed, content).splitlines()
    if all(unread(root, path) for path in changed):
        return Tree(committed, head, tuple(changed))
    return Tree(content, head, ())


def store_for(root: Path) -> Path:
    """The main checkout's stamp directory, which every worktree shares."""
    common = _absolute(root, _git(root, "rev-parse", "--git-common-dir"))
    return common.parent / STORE


def read_stamp(store: Path, key: str) -> dict[str, Any] | None:
    try:
        return dict(json.loads((store / f"{key}.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def write_stamp(store: Path, stamp: dict[str, Any]) -> None:
    store.mkdir(parents=True, exist_ok=True)
    target = store / f"{stamp['tree']}.json"
    scratch = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    scratch.write_text(json.dumps(stamp, indent=1), encoding="utf-8")
    os.replace(scratch, target)
    cutoff = time.time() - KEEP_DAYS * 86400
    for old in store.glob("*.json"):
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except OSError:
            pass


def collection_digest(items: list[pytest.Item]) -> str:
    digest = hashlib.sha256()
    for nodeid in sorted(item.nodeid for item in items):
        digest.update(nodeid.encode("utf-8") + b"\n")
    return digest.hexdigest()[:16]


def stale_core() -> str | None:
    from chess_variant_sandbox import measurement

    return measurement.stale_build()


def built_core() -> str:
    from chess_variant_sandbox import measurement

    return measurement.built_fingerprint()


def not_the_fast_suite(config: pytest.Config) -> str | None:
    """Why this run is not the whole fast suite, or ``None`` when it is."""
    option = config.option
    marker = " ".join((option.markexpr or "").split())
    if marker != FAST_SUITE:
        return f'its marker expression is "{marker}", not "{FAST_SUITE}"'
    if config.args_source is pytest.Config.ArgsSource.ARGS:
        named = {(config.invocation_params.dir / arg).resolve() for arg in config.args}
        testpaths = {(config.rootpath / path).resolve() for path in config.getini("testpaths")}
        if named != testpaths:
            return "it names paths of its own"
    elif config.args_source is not pytest.Config.ArgsSource.TESTPATHS:
        return "no testpaths are configured"
    narrowing = {
        "-k": option.keyword,
        "--deselect": getattr(option, "deselect", None),
        "--ignore": getattr(option, "ignore", None),
        "--ignore-glob": getattr(option, "ignore_glob", None),
        "--last-failed": getattr(option, "lf", False),
        "--stepwise": getattr(option, "stepwise", False),
        "--collect-only": option.collectonly,
    }
    named_flags = [flag for flag, value in narrowing.items() if value]
    if named_flags:
        return f"{named_flags[0]} narrows it"
    return None


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--skip-if-green",
        action="store_true",
        default=False,
        help=(
            'with -m "not slow" and no paths: if this exact tree already passed the fast '
            "suite, re-run only the tests that run skipped"
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(GreenStamp(config), "green-stamp")


class GreenStamp:
    def __init__(self, config: pytest.Config) -> None:
        self.gate: bool = config.getoption("skip_if_green")
        self.root = config.rootpath
        self.tree: Tree | None = None
        self.store: Path | None = None
        self.collection: str | None = None
        self.matched: dict[str, Any] | None = None
        self.rerun = 0
        self.passed = 0
        self.skipped: set[str] = set()
        self.started = time.time()
        self.note: str | None = None
        # xdist: set on a worker, which only collects and reports; the controller gathers
        # one report per worker and decides.
        self.workeroutput: dict[str, Any] | None = getattr(config, "workeroutput", None)
        self.reports: list[dict[str, Any]] = []

        if self.workeroutput is not None:
            handed = getattr(config, "workerinput", {}).get(XDIST_KEY)
            if handed is not None:  # the controller found a whole fast suite on a git tree
                self.tree = Tree(handed["tree"], handed["head"], tuple(handed["exempt"]))
                self.store = Path(handed["store"])
            return

        why_not = not_the_fast_suite(config)
        if why_not is not None:
            if self.gate:
                raise pytest.UsageError(
                    f'--skip-if-green vouches only for the whole fast suite (-m "{FAST_SUITE}"'
                    f" and no paths), and this run is not it: {why_not}"
                )
            return
        if self.gate:
            stale = stale_core()
            if stale is not None:
                raise pytest.UsageError(f"--skip-if-green will not run on a stale core: {stale}")
        try:
            self.tree = tree_of(self.root)
            self.store = store_for(self.root)
        except NotAGitCheckout:
            self.tree = None
            self.note = "no git tree to key a stamp on; nothing skipped or stamped"

    @pytest.hookimpl(optionalhook=True)
    def pytest_configure_node(self, node: Any) -> None:
        """xdist, on the controller: hand each worker the tree, keyed once for them all."""
        if self.tree is not None and self.store is not None:
            node.workerinput[XDIST_KEY] = {
                "tree": self.tree.key,
                "head": self.tree.head,
                "exempt": list(self.tree.exempt),
                "store": str(self.store),
            }

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(
        self, config: pytest.Config, items: list[pytest.Item]
    ) -> None:
        if self.tree is None or self.store is None:
            return
        self.collection = collection_digest(items)
        try:
            self._gate(config, items)
        finally:
            if self.workeroutput is not None:
                self.workeroutput[XDIST_KEY] = {
                    "collection": self.collection,
                    "matched": self.matched is not None,
                    "rerun": self.rerun,
                    "note": self.note,
                }

    def _gate(self, config: pytest.Config, items: list[pytest.Item]) -> None:
        if not self.gate or self.tree is None or self.store is None:
            return
        short = self.tree.key[:10]
        stamp = read_stamp(self.store, self.tree.key)
        if stamp is None:
            self.note = f"no stamp for tree {short}; ran the whole fast suite"
            return
        if stamp.get("collection") != self.collection:
            self.note = (
                f"tree {short} was green, but this checkout collects a different set of"
                " tests; ran the whole fast suite"
            )
            return
        skipped = set(stamp.get("skipped", ()))
        keep = [item for item in items if item.nodeid in skipped]
        drop = [item for item in items if item.nodeid not in skipped]
        if drop:
            config.hook.pytest_deselected(items=drop)
        items[:] = keep
        self.matched = stamp
        self.rerun = len(keep)

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node: Any, error: object) -> None:
        """xdist, on the controller: what a worker collected, and whether its gate matched."""
        said = getattr(node, "workeroutput", None) or {}
        if XDIST_KEY in said:
            self.reports.append(said[XDIST_KEY])

    def _gather(self) -> None:
        """xdist, on the controller: the workers' reports, read as one collection would be.

        Every worker collects the whole suite, so every report carries the same digest and
        the same verdict on the stamp; xdist itself refuses to run collections that differ.
        A disagreement here is read as no collection at all, which stamps nothing.
        """
        digests = {report["collection"] for report in self.reports}
        verdicts = {report["matched"] for report in self.reports}
        if len(digests) != 1 or len(verdicts) != 1:
            self.collection = None
            self.note = "the workers disagreed about the collection; nothing stamped"
            return
        self.collection = digests.pop()
        first = self.reports[0]
        if not verdicts.pop():
            self.note = first["note"]
            return
        stamp = read_stamp(self.store, self.tree.key) if self.tree and self.store else None
        if stamp is None:  # gone since the workers read it: this run was a subset, so no stamp
            self.collection = None
            self.note = "the workers matched a stamp that is gone now; nothing stamped"
            return
        self.matched = stamp
        self.rerun = int(first["rerun"])

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.passed and report.when == "call":
            self.passed += 1
        elif report.skipped and not hasattr(report, "wasxfail"):
            self.skipped.add(report.nodeid)

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        if self.workeroutput is not None:
            return  # a worker never stamps: its part can be green while another's is red
        if self.tree is None or self.store is None:
            return
        if self.reports:
            self._gather()
        short = self.tree.key[:10]
        left_out = ""
        if self.tree.exempt:
            left_out = f" (left out, uncommitted: {', '.join(self.tree.exempt)})"
        if self.matched is not None:
            if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
                session.exitstatus = pytest.ExitCode.OK
            where = Path(str(self.matched.get("checkout", "?"))).name
            self.note = (
                f"tree {short} passed the fast suite in {where} at {self.matched.get('finished')}"
                f" ({self.matched.get('passed')} passed); re-ran only the {self.rerun} tests"
                f" that run skipped{left_out}"
            )
            return
        if exitstatus != pytest.ExitCode.OK or self.collection is None:
            return
        stale = stale_core()
        if stale is not None:
            self.note = f"not stamped: {stale}"
            return
        try:
            after = tree_of(self.root)
        except NotAGitCheckout:
            return
        if after.key != self.tree.key:
            self.note = f"not stamped: the files changed during the run (tree {short})"
            return
        write_stamp(
            self.store,
            {
                "tree": self.tree.key,
                "head": self.tree.head,
                "exempt": list(self.tree.exempt),
                "checkout": str(self.root),
                "finished": time.strftime("%Y-%m-%d %H:%M"),
                "seconds": round(time.time() - self.started, 1),
                "core": built_core(),
                "collection": self.collection,
                "passed": self.passed,
                "skipped": sorted(self.skipped),
            },
        )
        prefix = self.note + "; " if self.note else ""
        self.note = f"{prefix}stamped tree {short} green{left_out}"

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        if self.note and self.workeroutput is None:
            terminalreporter.write_line(f"green stamp: {self.note}")
