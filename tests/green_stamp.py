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
- **Except Markdown under `docs/` or `.claude/` that no tracked Python file names.** The main checkout
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

Three things widen what a stamp is found for (2026-10-09, after a day of gates that re-ran the
whole suite on code a worker had just tested):

- **A tree that differs from a stamped one only in Markdown nothing reads**, committed or
  not. The exemption above covers only uncommitted paths, so a worker that committed a
  `docs/reference/` note after its run, or a coordinator commit to the queue, used to cost a
  full run. `find_stamp` looks among the newest stamps for one whose tree differs from this
  one only in paths `unread` clears, and the summary line names them. `unread` also stopped
  counting `docs_tests/` as a reader (the fast suite never collects it), and clears skills
  and rules under `.claude/`: the coordinator's queue, status and policy commits land
  between a worker's run and its gate.
- **A red run is stamped too**, with its failures (`"verdict": "red"`), when it ran to the
  end (no `-x`/`--maxfail`). A gate on the same tree re-runs only what failed or skipped, so
  re-checking a known-red main costs the failures, not the suite. A rerun that comes out
  green stamps the tree green: every other test passed in the first run. A gate that matched
  a green stamp and fails on what it re-ran stamps the tree red.
- **A stamp can travel.** A cloud worker's stamp is written in its own clone and used to die
  with it, so every cloud merge re-ran the whole suite here. `export` puts it on a branch of
  its own (`<branch>-green`, an orphan commit holding `stamp.json`, never part of the code's
  tree); `import` files it in this machine's store. The gate then re-runs what the cloud
  skipped: the tests that need `runs/`, the database or Fairy-Stockfish. What it gives up,
  by the user's choice (2026-10-09): code tested only on Linux reaches main without one full
  run on Windows.

        uv run python tests/green_stamp.py export --push origin     # in the cloud, after the suite
        uv run python tests/green_stamp.py import origin/<branch>-green   # here, before the gate

**Under xdist (`-n 4`, how the fast suite runs since Q377c)** the work is split, and so is
this plugin. The *controller* keys the tree once, hands it to each worker, counts the
passes and skips the workers report back, and alone decides and writes the stamp, from
the whole run's exit status. Each *worker* collects the whole suite, computes the same
collection digest, applies the gate's deselection to its own copy of the collection, and
reports what it did through `workeroutput`. A worker never stamps: its part of a run can
be green while another worker's is red.
"""

# compute-lease: light
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
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
#
# `docs_tests/` reads documents on purpose, but the fast suite never collects it and every
# merge runs it anyway. The other two name the queue as a path to classify and as a file in a
# scratch repository. Counted, they made every queue commit on main a new tree, so a gate
# after one never matched the worker's stamp (2026-10-09).
#
# `prune_worktrees.py branches` reads the *main checkout's* queue when a coordinator closes,
# never the tree under test, and its tests give it inline text or a scratch repository's
# (Q845), so neither makes the suite's outcome depend on this tree's queue.
NOT_READERS = (
    "tests/green_stamp.py",
    "tests/test_green_stamp.py",
    "docs_tests/",
    "tests/test_docs_only.py",
    "tests/test_export_public.py",
    ".claude/scripts/prune_worktrees.py",
    "tests/test_prune_worktrees.py",
)
# Where Markdown can be cleared as unread (`unread`).
PROSE = ("docs/", ".claude/")
# How many of the newest stamps `find_stamp` compares when none has this tree's own key.
SEARCH = 100
TREE_NAME = re.compile(r"[0-9a-f]{40}")
# The file a travelling stamp is carried in, alone in its orphan commit's tree.
CARRIED = "stamp.json"
GREEN, RED = "green", "red"


class NotAGitCheckout(Exception):
    """git could not answer: no repository, no HEAD, no git."""


def _git(
    root: Path, *args: str, env: dict[str, str] | None = None, input: str | None = None
) -> str:
    # Bytes, not text mode: on Windows a text pipe writes "\n" as "\r\n", and git would read
    # the "\r" into an object or file name.
    try:
        out = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, check=True, env=env,
            input=None if input is None else input.encode("utf-8"), timeout=60,
        )
    except (subprocess.SubprocessError, OSError) as error:
        raise NotAGitCheckout(" ".join(args)) from error
    return out.stdout.decode("utf-8", errors="replace").strip()


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

    Markdown under `docs/` or `.claude/` (the skills and rules, since 2026-10-09) whose file name
    appears in no tracked Python file outside `NOT_READERS`. A substring search errs towards
    *read*: `queue.md` is read if a file names `add-to-queue.md`. Markdown elsewhere is
    never cleared, since `pyproject.toml` can name a README where no Python file does.
    """
    if not (path.startswith(PROSE) and path.endswith(".md")):
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


@dataclass(frozen=True)
class Found:
    key: str  # the stamp's own tree, which names its file
    drift: tuple[str, ...]  # Markdown nothing reads, by which that tree differs from this one


def _present_trees(root: Path, names: list[str]) -> set[str]:
    """Which of these are trees in this repository: a cloud stamp's may never have arrived."""
    answer = _git(root, "cat-file", "--batch-check", input="".join(f"{n}\n" for n in names))
    return {line.split()[0] for line in answer.splitlines() if line.split()[1:2] == ["tree"]}


def _changed_paths(root: Path, stamped: list[str], key: str) -> dict[str, list[str]]:
    """The paths by which each stamped tree differs from `key`, from one `git diff-tree`."""
    answer = _git(
        root, "-c", "core.quotepath=off", "diff-tree", "--stdin", "-r", "--name-only",
        input="".join(f"{name} {key}\n" for name in stamped),
    )
    changed: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in answer.splitlines():
        pair = line.split(" ")
        if len(pair) == 2 and all(TREE_NAME.fullmatch(part) for part in pair):
            current = changed.setdefault(pair[0], [])
        elif current is not None and line:
            current.append(line)
    return changed


def find_stamp(root: Path, store: Path, key: str) -> Found | None:
    """The stamp that vouches for tree `key`: its own, or the newest one taken on a tree that
    differs from it only in Markdown no test reads (`unread`), committed or not."""
    if (store / f"{key}.json").exists():
        return Found(key, ())
    stamps = sorted(store.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    names = [path.stem for path in stamps[:SEARCH] if TREE_NAME.fullmatch(path.stem)]
    if not names:
        return None
    present = _present_trees(root, names)
    names = [name for name in names if name in present]
    if not names:
        return None
    changed = _changed_paths(root, names, key)
    read: dict[str, bool] = {}

    def nothing_reads(path: str) -> bool:
        if path not in read:
            read[path] = unread(root, path)
        return read[path]

    for name in names:  # newest first
        paths = changed.get(name)
        if not paths:
            continue
        if all(p.startswith(PROSE) and p.endswith(".md") for p in paths) and all(
            nothing_reads(p) for p in paths
        ):
            return Found(name, tuple(paths))
    return None


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
        self.found: Found | None = None
        self.collection: str | None = None
        self.matched: dict[str, Any] | None = None
        self.rerun = 0
        self.passed = 0
        self.skipped: set[str] = set()
        self.failed: set[str] = set()
        # A run stopped early by -x or --maxfail never ran the rest, so its red says nothing
        # about them.
        self.ran_to_the_end = not config.getoption("maxfail", 0)
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
                if handed.get("found") is not None:
                    self.found = Found(handed["found"][0], tuple(handed["found"][1]))
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
            if self.gate:
                # Once, here: the workers are handed the answer rather than each searching.
                self.found = find_stamp(self.root, self.store, self.tree.key)
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
                "found": None if self.found is None else [self.found.key, list(self.found.drift)],
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
        stamp = None if self.found is None else read_stamp(self.store, self.found.key)
        if stamp is None:
            self.note = f"no stamp for tree {short}; ran the whole fast suite"
            return
        if stamp.get("collection") != self.collection:
            self.note = (
                f"tree {short} was stamped, but this checkout collects a different set of"
                " tests; ran the whole fast suite"
            )
            return
        rerun = set(stamp.get("skipped", ())) | set(stamp.get("failed", ()))
        keep = [item for item in items if item.nodeid in rerun]
        drop = [item for item in items if item.nodeid not in rerun]
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
        stamp = read_stamp(self.store, self.found.key) if self.found and self.store else None
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
        elif report.failed:
            self.failed.add(report.nodeid)

    def _verdict(self, exitstatus: int) -> str | None:
        """What this run says about its tree, or ``None`` when it says nothing whole."""
        if exitstatus == pytest.ExitCode.OK:
            return GREEN
        if exitstatus == pytest.ExitCode.TESTS_FAILED and self.failed and self.ran_to_the_end:
            return RED
        return None

    def _why_not_stamp(self) -> str | None:
        stale = stale_core()
        if stale is not None:
            return stale
        assert self.tree is not None
        try:
            after = tree_of(self.root)
        except NotAGitCheckout:
            return "no git tree after the run"
        if after.key != self.tree.key:
            return f"the files changed during the run (tree {self.tree.key[:10]})"
        return None

    def _stamp(self, verdict: str, **more: Any) -> None:
        assert self.tree is not None and self.store is not None
        write_stamp(
            self.store,
            {
                "tree": self.tree.key,
                "head": self.tree.head,
                "exempt": list(self.tree.exempt),
                "verdict": verdict,
                "checkout": str(self.root),
                "finished": time.strftime("%Y-%m-%d %H:%M"),
                "seconds": round(time.time() - self.started, 1),
                "core": built_core(),
                "collection": self.collection,
                "passed": self.passed,
                "skipped": sorted(self.skipped),
                "failed": sorted(self.failed),
                **more,
            },
        )

    def _matched_note(self, left_out: str) -> str:
        assert self.matched is not None and self.tree is not None and self.found is not None
        stamped = self.tree.key[:10]
        said = f"tree {stamped}"
        if self.found.drift:
            stamped = self.found.key[:10]
            said = (
                f"tree {self.tree.key[:10]} differs from tree {stamped} only in Markdown"
                f" nothing reads ({', '.join(self.found.drift)}); tree {stamped}"
            )
        where = Path(str(self.matched.get("checkout", "?"))).name
        when = self.matched.get("finished")
        if self.matched.get("verdict", GREEN) == RED:
            failed = len(self.matched.get("failed", ()))
            return (
                f"{said} was red in {where} at {when} ({failed} failed); re-ran only the"
                f" {self.rerun} tests that run failed or skipped{left_out}"
            )
        return (
            f"{said} passed the fast suite in {where} at {when}"
            f" ({self.matched.get('passed')} passed); re-ran only the {self.rerun} tests"
            f" that run skipped{left_out}"
        )

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
                session.exitstatus = exitstatus = pytest.ExitCode.OK
            self.note = self._matched_note(left_out)
            verdict = self._verdict(exitstatus)
            # A green stamp that stays green is left as it is. Anything else is news: a red
            # tree come green, or what the rerun found failing.
            if verdict is None or (verdict == GREEN and self.matched.get("verdict", GREEN) == GREEN):
                return
            why_not = self._why_not_stamp()
            if why_not is not None:
                self.note += f"; not stamped: {why_not}"
                return
            self._stamp(
                verdict,
                passed=int(self.matched.get("passed", 0)) + self.passed,
                rerun_of={
                    key: self.matched.get(key)
                    for key in ("tree", "verdict", "checkout", "finished")
                },
            )
            self.note += f"; stamped tree {short} {verdict}"
            return
        verdict = self._verdict(exitstatus)
        if verdict is None or self.collection is None:
            return
        why_not = self._why_not_stamp()
        if why_not is not None:
            self.note = f"not stamped: {why_not}"
            return
        self._stamp(verdict)
        prefix = self.note + "; " if self.note else ""
        failed = f" ({len(self.failed)} failed)" if verdict == RED else ""
        self.note = f"{prefix}stamped tree {short} {verdict}{failed}{left_out}"

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        if self.note and self.workeroutput is None:
            terminalreporter.write_line(f"green stamp: {self.note}")


# A stamp's journey from a cloud clone to this machine. The orphan commit's only file is the
# stamp, so the branch shares no history or tree with the code it vouches for.


def export(root: Path, push: str | None) -> int:
    """Carry this checkout's stamp on `<branch>-green`; push it to `push` when named."""
    tree = tree_of(root)
    store = store_for(root)
    found = find_stamp(root, store, tree.key)
    stamp = None if found is None else read_stamp(store, found.key)
    if found is None or stamp is None:
        print(f"no stamp for tree {tree.key[:10]}: nothing to export", file=sys.stderr)
        return 1
    branch = _git(root, "branch", "--show-current")
    if not branch:
        print("no branch checked out to name the stamp's branch after", file=sys.stderr)
        return 1
    blob = _git(root, "hash-object", "-w", "--stdin", input=json.dumps(stamp, indent=1))
    holder = _git(root, "mktree", input=f"100644 blob {blob}\t{CARRIED}\n")
    commit = _git(
        root, "-c", "user.name=green-stamp", "-c", "user.email=green-stamp@localhost",
        "commit-tree", holder, "-m", f"GREEN STAMP {stamp['tree']} ({stamp.get('verdict', GREEN)})",
    )
    target = f"{branch}-green"
    _git(root, "branch", "-f", target, commit)
    if push:
        _git(root, "push", "-f", push, f"{target}:{target}")
    drift = f", which differs from this tree only in {', '.join(found.drift)}" if found.drift else ""
    print(f"exported the {stamp.get('verdict', GREEN)} stamp of tree {found.key[:10]}{drift} on {target}")
    return 0


def import_(root: Path, ref: str) -> int:
    """File the stamp carried on `ref` in this checkout's store."""
    try:
        stamp = dict(json.loads(_git(root, "show", f"{ref}:{CARRIED}")))
    except (NotAGitCheckout, ValueError):
        print(f"no stamp on {ref}", file=sys.stderr)
        return 1
    if not TREE_NAME.fullmatch(str(stamp.get("tree", ""))) or "collection" not in stamp:
        print(f"{ref} carries something that is not a stamp", file=sys.stderr)
        return 1
    stamp["imported_from"] = {"ref": ref, "checkout": stamp.get("checkout")}
    stamp["checkout"] = ref  # what the gate's summary line names
    write_stamp(store_for(root), stamp)
    print(f"imported the {stamp.get('verdict', GREEN)} stamp of tree {stamp['tree'][:10]} from {ref}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Carry a green stamp between clones.")
    commands = parser.add_subparsers(dest="command", required=True)
    out = commands.add_parser("export", help="put this tree's stamp on <branch>-green")
    out.add_argument("--push", metavar="REMOTE", help="and push that branch there")
    into = commands.add_parser("import", help="file the stamp carried on REF here")
    into.add_argument("ref")
    args = parser.parse_args(argv)
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    if args.command == "export":
        return export(root, args.push)
    return import_(root, args.ref)


if __name__ == "__main__":
    sys.exit(main())
