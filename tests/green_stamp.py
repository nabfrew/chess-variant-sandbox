"""Remember which trees the fast suite passed on, so a test is not run again where nothing reaches it.

A task used to end with `pytest -m "not slow"` three or four times over the same code: once
before merging main, once after, again when main moved during that run, and once more in
the main checkout after the coordinator fast-forwarded it. Each is about seven minutes on a
loaded machine, and the last one tested, byte for byte, a tree the worker had just tested.

So every run of the whole fast suite that runs to the end leaves a *stamp*: a small JSON file
named after the git tree of the files it ran against, under the main checkout's
`.claude/coordinator/green/`, where every worktree sees it. `--skip-if-green` reads them back
and runs only the tests no stamp vouches for.

What a stamp is keyed on, and why it can be trusted:

- **The content tree**: what `git add -A` would commit from the working copy, built in a
  scratch index so the real one is untouched. Uncommitted and untracked files count, so a
  stamp vouches for what was on disk rather than for HEAD. The tree is taken before and
  after the run, and a run whose files moved underneath it is not stamped.
- **Except Markdown under `docs/` or `.claude/` that no tracked Python file names.** The main
  checkout usually carries an uncommitted line in the queue's inbox, and a key that counted it
  would never match there. `docs/changelog.md` is served by the site and read by its tests, so
  it counts; the name search is what tells the two apart. When every uncommitted path is
  of this kind the key is HEAD's tree, and the summary line names what it left out.
- **The core the run loaded was built from this tree's `rust/src/`.** A run on a stale core
  is not stamped, and `--skip-if-green` refuses to start on one.
- **The test ids, file by file** (`files`: a digest of each test file's ids). A test
  parametrised over a checkout's files (`runs/puzzles/`) collects differently in another
  checkout, and a module skipped at import (`importorskip`) collects nothing. Either re-runs
  that file, not the suite. Until 2026-10-22 one digest covered the whole collection, and
  either difference made nearly every gate after a worker's run a full one.

A stamp vouches for a test on this tree when the test's file collects the same ids there, the
test neither failed nor skipped there, and (`docs/reference/affected-gate.md`):

- **the two trees are the same**, or differ only in Markdown nothing reads, committed or not
  (*exact*: the coordinator's queue commits land between a worker's run and its gate); or
- **the change between them cannot reach the test's file** (`affected.py`: imports to two
  steps, files named in strings, conftest fixtures). Any recent stamp may vouch, so after a
  merge the worker's stamp covers what main changed and main's covers what the worker
  changed, and only tests both changes reach run. This leans on a judgement the import graph
  makes, so it is bounded:
  - a run that leaned on another tree writes `debt` = 1 + the largest debt it leaned on, and
    a stamp at `DEBT_LIMIT` vouches for nothing on another tree;
  - a stamp names the full run its verdict rests on, and vouches for nothing on another tree
    once that run is `FULL_EVERY_HOURS` old.
  So a failure the graph missed is found by a full run within a few gates and a few hours, and
  a red full run names the last whole green run on its line, which bounds the search.
- A test that failed or skipped on any stamp the gate leans on is re-run, whatever the
  others say: a worker's skipped `runs/` readers still run in the main checkout. And a test
  that skipped on any of the newest stamps reads something outside the tree (`runs/`, a
  database), so no stamp on other code vouches for it.

A stamp says the code was green. It does not vouch for anything outside the tree that has
changed since, such as a delivered measurement or an environment variable, beyond
re-running what skipped.

- **A red run is stamped too**, with its failures (`"verdict": "red"`), when it ran to the
  end (no `-x`/`--maxfail`). A gate on the same tree re-runs only what failed or skipped, so
  re-checking a known-red main costs the failures, not the suite.
- **A stamp can travel.** A cloud worker's stamp is written in its own clone and used to die
  with it, so every cloud merge re-ran the whole suite here. `export` puts it on a branch of
  its own (`<branch>-green`, an orphan commit holding `stamp.json`, never part of the code's
  tree); `import` files it in this machine's store. What it gives up, by the user's choice
  (2026-10-09): code tested only on Linux reaches main without one full run on Windows.

        uv run python tests/green_stamp.py export --push origin     # in the cloud, after the suite
        uv run python tests/green_stamp.py import origin/<branch>-green   # here, before the gate
        uv run python tests/green_stamp.py plan     # what the gate would run here, and why; seconds

**Under xdist (`-n 4`, how the fast suite runs since Q377c)** the work is split, and so is
this plugin. The *controller* keys the tree and chooses the stamps that may vouch once, hands
them to each worker, counts the passes and skips the workers report back, and alone decides
and writes the stamp, from the whole run's exit status. Each *worker* collects the whole
suite, applies the same deselection to its own copy of the collection, and reports what it did
through `workeroutput`. A worker never stamps: its part of a run can be green while another
worker's is red.
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
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import affected
import pytest

FAST_SUITE = "not slow"
STORE = Path(".claude") / "coordinator" / "green"
KEEP_DAYS = 30
# How far a judgement may run before the whole suite runs again (docs/reference/affected-gate.md).
DEBT_LIMIT = 5
FULL_EVERY_HOURS = 8
# How many stamps a gate leans on at most: the exact ones and the derived ones that reach least.
BASES = 8
# Of the newest stamps, how many on other code a gate weighs, and how far apart they may be:
# an older stamp's change is larger and reaches more, so it seldom vouches for anything the
# newer ones don't, and reaching through hundreds of paths costs seconds.
DERIVE_SEARCH = 20
DERIVE_PATHS = 300
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
# How many of the newest stamps a gate compares with this tree.
SEARCH = 100
TREE_NAME = re.compile(r"[0-9a-f]{40}")
# The file a travelling stamp is carried in, alone in its orphan commit's tree.
CARRIED = "stamp.json"
GREEN, RED = "green", "red"
# What `affected` parsed, by blob, beside the store.
CACHE = "affected-cache.json"
WHEN = "%Y-%m-%d %H:%M"


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


def full_run_of(stamp: dict[str, Any]) -> dict[str, Any]:
    """The full run a stamp's verdict rests on. A stamp from before 2026-10-22 is one."""
    rests = stamp.get("full_run")
    if isinstance(rests, dict) and "at" in rests:
        return rests
    try:
        at = time.mktime(time.strptime(str(stamp.get("finished")), WHEN))
    except ValueError:
        at = 0.0
    return {"tree": stamp.get("tree"), "head": stamp.get("head"), "finished": stamp.get("finished"), "at": at}


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
    if not stamped:
        return {}
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


def _newest(store: Path, key: str) -> list[str]:
    """This tree's own stamp first, then the newest others."""
    stamps = sorted(store.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    names = [path.stem for path in stamps[:SEARCH] if TREE_NAME.fullmatch(path.stem)]
    if (store / f"{key}.json").exists():
        names = [key, *(name for name in names if name != key)]
    return names


def find_stamp(root: Path, store: Path, key: str) -> Found | None:
    """The stamp that vouches for tree `key` exactly: its own, or the newest one taken on a tree
    that differs from it only in Markdown no test reads (`unread`), committed or not."""
    if (store / f"{key}.json").exists():
        return Found(key, ())
    plan = plan_for(root, store, key, derive=False)
    exact = [base for base in plan.bases if base.exact]
    return Found(exact[0].key, exact[0].drift) if exact else None


@dataclass(frozen=True)
class Base:
    """A stamp a gate may lean on, and what the change from its tree reaches."""

    key: str  # the stamp's tree
    exact: bool  # the same code: the trees differ at most in Markdown nothing reads
    drift: tuple[str, ...]  # the paths they differ in
    reached: frozenset[str]  # the test files the change reaches; none when exact
    debt: int  # what a run leaning on it owes: its own debt, plus one unless exact

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key, "exact": self.exact, "drift": list(self.drift),
            "reached": sorted(self.reached), "debt": self.debt,
        }

    @staticmethod
    def from_json(data: dict[str, Any]) -> Base:
        return Base(
            data["key"], bool(data["exact"]), tuple(data["drift"]),
            frozenset(data["reached"]), int(data["debt"]),
        )


@dataclass
class Plan:
    bases: list[Base]
    why_not: str | None = None  # why the nearest stamp that could not vouch could not
    tests: frozenset[str] | None = None  # the tree's test files, when the graph was built
    # Tests that skipped on one of the newest stamps: they read something outside the tree
    # (the main checkout's `runs/`, a database), which no change between trees shows.
    skipped_lately: frozenset[str] = frozenset()


def plan_for(
    root: Path, store: Path, key: str, *, derive: bool = True, now: float | None = None
) -> Plan:
    """The stamps that may vouch for tests on tree `key`: exact ones first, then the ones
    whose change reaches fewest test files."""
    now = time.time() if now is None else now
    names = _newest(store, key)
    if not names:
        return Plan([])
    present = _present_trees(root, names)
    names = [name for name in names if name in present]
    changed = _changed_paths(root, [name for name in names if name != key], key)
    read: dict[str, bool] = {}

    def nothing_reads(path: str) -> bool:
        if path not in read:
            read[path] = unread(root, path)
        return read[path]

    graph: affected.Graph | None = None
    exact: list[Base] = []
    derived: list[Base] = []
    why_not: str | None = None
    considered = 0
    for name in names:
        stamp = read_stamp(store, name)
        if stamp is None or "collection" not in stamp:
            continue
        debt = int(stamp.get("debt", 0))
        paths = changed.get(name, [])
        if name == key:
            exact.append(Base(name, True, (), frozenset(), debt))
            continue
        if not paths:
            continue
        # `all` stops at the first path that isn't Markdown, so only a stamp that may be the
        # same code pays a `git grep` per note.
        if all(p.startswith(PROSE) and p.endswith(".md") for p in paths) and all(
            nothing_reads(p) for p in paths
        ):
            exact.append(Base(name, True, tuple(paths), frozenset(), debt))
            continue
        if not derive or considered >= DERIVE_SEARCH or len(paths) > DERIVE_PATHS:
            continue
        considered += 1
        if "files" not in stamp:
            why_not = why_not or "the stamps near it are from before 2026-10-22, which vouch only for their own tree"
            continue
        if debt >= DEBT_LIMIT:
            why_not = why_not or f"tree {name[:10]} is at debt {debt}, the limit"
            continue
        rests = full_run_of(stamp)
        if now - float(rests["at"]) > FULL_EVERY_HOURS * 3600:
            why_not = why_not or (
                f"tree {name[:10]} rests on a full run of {rests.get('finished')}, over"
                f" {FULL_EVERY_HOURS} h ago"
            )
            continue
        if graph is None:
            graph = affected.graph_of(str(root), key, cache=store.parent / CACHE, not_readers=NOT_READERS)
        reach = graph.reach(paths)
        if reach.everything is not None:
            why_not = why_not or f"from tree {name[:10]}, {reach.everything}"
            continue
        derived.append(Base(name, False, tuple(paths), reach.tests, debt + 1))
    derived.sort(key=lambda base: (base.debt, len(base.reached)))
    chosen = exact[:3] + derived[: max(0, BASES - min(len(exact), 3))]
    skipped_lately: set[str] = set()
    for name in names[:DERIVE_SEARCH]:
        skipped_lately.update((read_stamp(store, name) or {}).get("skipped", ()))
    return Plan(chosen, why_not, None if graph is None else graph.tests, frozenset(skipped_lately))


def collection_digest(items: list[pytest.Item]) -> str:
    digest = hashlib.sha256()
    for nodeid in sorted(item.nodeid for item in items):
        digest.update(nodeid.encode("utf-8") + b"\n")
    return digest.hexdigest()[:16]


def file_of(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def file_digests(items: list[pytest.Item]) -> dict[str, str]:
    """Each test file's own digest of the ids it collects."""
    ids: dict[str, list[str]] = defaultdict(list)
    for item in items:
        ids[file_of(item.nodeid)].append(item.nodeid)
    return {
        path: hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()[:16]
        for path, names in sorted(ids.items())
    }


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
            'with -m "not slow" and no paths: run only the tests no stamp vouches for (a stamp'
            " on this tree, or on one whose change cannot reach them)"
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(GreenStamp(config), "green-stamp")


@dataclass
class Gated:
    """What the deselection did, as a worker reports it."""

    collection: str
    files: dict[str, str]
    vouched: dict[str, int] = field(default_factory=dict)  # stamp tree -> tests it vouched for
    rerun: int = 0
    # tests kept that the one exact stamp leaned on had passed: their file collects differently
    recollected: int = 0
    note: str | None = None


class GreenStamp:
    def __init__(self, config: pytest.Config) -> None:
        self.gate: bool = config.getoption("skip_if_green")
        self.root = config.rootpath
        self.tree: Tree | None = None
        self.store: Path | None = None
        self.plan = Plan([])
        self.gated: Gated | None = None
        self.passed = 0
        self.skipped: set[str] = set()
        self.failed: set[str] = set()
        self.durations: Counter[str] = Counter()
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
                self.plan = Plan(
                    [Base.from_json(base) for base in handed["bases"]],
                    skipped_lately=frozenset(handed["skipped_lately"]),
                )
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
                self.plan = plan_for(self.root, self.store, self.tree.key)
        except NotAGitCheckout:
            self.tree = None
            self.note = "no git tree to key a stamp on; nothing skipped or stamped"

    @pytest.hookimpl(optionalhook=True)
    def pytest_configure_node(self, node: Any) -> None:
        """xdist, on the controller: hand each worker the tree and the stamps, chosen once."""
        if self.tree is not None and self.store is not None:
            node.workerinput[XDIST_KEY] = {
                "tree": self.tree.key,
                "head": self.tree.head,
                "exempt": list(self.tree.exempt),
                "store": str(self.store),
                "bases": [base.to_json() for base in self.plan.bases],
                "skipped_lately": sorted(self.plan.skipped_lately),
            }

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(
        self, config: pytest.Config, items: list[pytest.Item]
    ) -> None:
        if self.tree is None or self.store is None:
            return
        self.gated = Gated(collection_digest(items), file_digests(items))
        try:
            self._gate(config, items)
        finally:
            if self.workeroutput is not None:
                self.workeroutput[XDIST_KEY] = {
                    "collection": self.gated.collection,
                    "files": self.gated.files,
                    "vouched": self.gated.vouched,
                    "rerun": self.gated.rerun,
                    "recollected": self.gated.recollected,
                    "note": self.gated.note,
                }

    def _gate(self, config: pytest.Config, items: list[pytest.Item]) -> None:
        assert self.gated is not None and self.tree is not None and self.store is not None
        if not self.gate:
            return
        short = self.tree.key[:10]
        stamps = {
            base.key: stamp
            for base in self.plan.bases
            if (stamp := read_stamp(self.store, base.key)) is not None
        }
        bases = [base for base in self.plan.bases if base.key in stamps]
        if not bases:
            self.gated.note = f"no stamp for tree {short}; ran the whole fast suite" + (
                f" ({self.plan.why_not})" if self.plan.why_not else ""
            )
            return
        # What failed or skipped on any stamp leaned on runs: a failure is news until it
        # passes, and a skip is a test this environment may be the first to run (a worker's
        # `runs/` readers, run in the main checkout whatever an older stamp there says).
        unsure = set().union(
            *(set(s.get("skipped", ())) | set(s.get("failed", ())) for s in stamps.values())
        )
        now = self.gated.files
        keep: list[pytest.Item] = []
        drop: list[pytest.Item] = []
        vouched: Counter[str] = Counter()
        for item in items:
            by = None
            if item.nodeid not in unsure:
                path = file_of(item.nodeid)
                for base in bases:
                    stamp = stamps[base.key]
                    if path in base.reached:
                        continue
                    if not base.exact and (
                        not affected.is_test_file(path) or item.nodeid in self.plan.skipped_lately
                    ):
                        # a file the graph never looked at, or a test that reads outside the
                        # tree: no change between trees says what reaches it
                        continue
                    files = stamp.get("files")
                    if files is None:  # a stamp from before 2026-10-22: one digest for the lot
                        if stamp.get("collection") != self.gated.collection:
                            continue
                    elif files.get(path) != now.get(path):
                        continue
                    by = base.key
                    break
            if by is None:
                keep.append(item)
            else:
                drop.append(item)
                vouched[by] += 1
        if not drop:
            self.gated.note = (
                f"no stamp for tree {short} vouches for any of its tests; ran the whole fast suite"
            )
            return
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
        self.gated.vouched = dict(vouched)
        self.gated.rerun = len(keep)
        self.gated.recollected = sum(item.nodeid not in unsure for item in keep)

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node: Any, error: object) -> None:
        """xdist, on the controller: what a worker collected, and what its gate vouched for."""
        said = getattr(node, "workeroutput", None) or {}
        if XDIST_KEY in said:
            self.reports.append(said[XDIST_KEY])

    def _gather(self) -> None:
        """xdist, on the controller: the workers' reports, read as one collection would be.

        Every worker collects the whole suite, so every report carries the same digests and
        the same deselection; xdist itself refuses to run collections that differ. A
        disagreement here is read as no collection at all, which stamps nothing.
        """
        digests = {report["collection"] for report in self.reports}
        vouched = {json.dumps(report["vouched"], sort_keys=True) for report in self.reports}
        if len(digests) != 1 or len(vouched) != 1:
            self.gated = None
            self.note = "the workers disagreed about the collection; nothing stamped"
            return
        first = self.reports[0]
        self.gated = Gated(
            first["collection"], dict(first["files"]), dict(first["vouched"]),
            int(first["rerun"]), int(first.get("recollected", 0)), first["note"],
        )

    def narrowed(self) -> bool:
        """Whether a stamp vouched for some of this run's tests, so it ran only part of the
        suite. On the controller this holds once the workers are down (`duration_guard` asks
        when the test loop ends)."""
        if self.workeroutput is not None:
            return False
        if self.reports:
            return any(report.get("vouched") for report in self.reports)
        return self.gated is not None and bool(self.gated.vouched)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.durations[file_of(report.nodeid)] += report.duration
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
        assert self.tree is not None and self.store is not None and self.gated is not None
        finished = time.strftime(WHEN)
        stamp = {
            "tree": self.tree.key,
            "head": self.tree.head,
            "exempt": list(self.tree.exempt),
            "verdict": verdict,
            "checkout": str(self.root),
            "finished": finished,
            "seconds": round(time.time() - self.started, 1),
            "core": built_core(),
            "collection": self.gated.collection,
            "files": self.gated.files,
            "passed": self.passed,
            "skipped": sorted(self.skipped),
            "failed": sorted(self.failed),
            "durations": {path: round(seconds, 2) for path, seconds in sorted(self.durations.items())},
            "debt": 0,
            "full_run": {"tree": self.tree.key, "head": self.tree.head, "finished": finished, "at": time.time()},
        }
        stamp.update(more)
        write_stamp(self.store, stamp)

    def _matched_note(self, base: Base, stamp: dict[str, Any], left_out: str) -> str:
        assert self.tree is not None and self.gated is not None
        stamped = self.tree.key[:10]
        said = f"tree {stamped}"
        if base.drift:
            stamped = base.key[:10]
            said = (
                f"tree {self.tree.key[:10]} differs from tree {stamped} only in Markdown"
                f" nothing reads ({', '.join(base.drift)}); tree {stamped}"
            )
        where = Path(str(stamp.get("checkout", "?"))).name
        when = stamp.get("finished")
        also = ""
        if self.gated.recollected:
            also = f" (and {self.gated.recollected} whose file collects differently here)"
        if stamp.get("verdict", GREEN) == RED:
            failed = len(stamp.get("failed", ()))
            return (
                f"{said} was red in {where} at {when} ({failed} failed); re-ran only the"
                f" {self.gated.rerun} tests that run failed or skipped{also}{left_out}"
            )
        return (
            f"{said} passed the fast suite in {where} at {when}"
            f" ({stamp.get('passed')} passed); re-ran only the {self.gated.rerun} tests"
            f" that run skipped{also}{left_out}"
        )

    def _derived_note(self, used: list[Base], debt: int, left_out: str) -> str:
        assert self.tree is not None and self.store is not None and self.gated is not None
        parts = []
        for base in used:
            stamp = read_stamp(self.store, base.key) or {}
            where = Path(str(stamp.get("checkout", "?"))).name
            how = "the same code" if base.exact else (
                f"{len(base.drift)} paths apart, reaching {len(base.reached)} test files"
            )
            parts.append(
                f"{self.gated.vouched[base.key]} by tree {base.key[:10]} ({where},"
                f" {stamp.get('finished')}; {how})"
            )
        total = sum(self.gated.vouched.values())
        return (
            f"tree {self.tree.key[:10]}: {total} tests vouched for, {'; '.join(parts)}; ran the"
            f" other {self.gated.rerun}; debt {debt} of {DEBT_LIMIT}{left_out}"
        )

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        if self.workeroutput is not None:
            return  # a worker never stamps: its part can be green while another's is red
        if self.tree is None or self.store is None:
            return
        if self.reports:
            self._gather()
        if self.gated is None:
            return
        short = self.tree.key[:10]
        left_out = ""
        if self.tree.exempt:
            left_out = f" (left out, uncommitted: {', '.join(self.tree.exempt)})"
        if self.gated.vouched:
            self._finish_leaning(session, exitstatus, left_out)
            return
        self.note = self.gated.note
        verdict = self._verdict(exitstatus)
        if verdict is None:
            return
        why_not = self._why_not_stamp()
        if why_not is not None:
            self.note = f"not stamped: {why_not}"
            return
        self._stamp(verdict)
        prefix = self.note + "; " if self.note else ""
        failed = f" ({len(self.failed)} failed)" if verdict == RED else ""
        self.note = f"{prefix}stamped tree {short} {verdict}{failed}{left_out}"
        if verdict == RED:
            self.note += self._last_whole_green()

    def _finish_leaning(self, session: pytest.Session, exitstatus: int, left_out: str) -> None:
        """A run that some stamps vouched for: say which, and stamp what it found."""
        assert self.gated is not None and self.store is not None and self.tree is not None
        if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
            session.exitstatus = exitstatus = pytest.ExitCode.OK
        used = [base for base in self.plan.bases if base.key in self.gated.vouched]
        stamps = {base.key: read_stamp(self.store, base.key) or {} for base in used}
        debt = max(base.debt for base in used)
        if len(used) == 1 and used[0].exact:
            self.note = self._matched_note(used[0], stamps[used[0].key], left_out)
        else:
            self.note = self._derived_note(used, debt, left_out)
        verdict = self._verdict(exitstatus)
        # The same code, green before and green now, is left as it is. Anything else is news:
        # a red tree come green, what the rerun found failing, or a verdict leaned on.
        all_green = all(stamp.get("verdict", GREEN) == GREEN for stamp in stamps.values())
        if verdict is None or (verdict == GREEN and all_green and all(b.exact for b in used)):
            return
        why_not = self._why_not_stamp()
        if why_not is not None:
            self.note += f"; not stamped: {why_not}"
            return
        rests = min((full_run_of(stamp) for stamp in stamps.values()), key=lambda run: float(run["at"]))
        more: dict[str, Any] = {
            "passed": sum(self.gated.vouched.values()) + self.passed,
            "debt": debt,
            "full_run": rests,
            "vouched_by": dict(self.gated.vouched),
            "durations": self._durations(stamps),
        }
        if len(used) == 1 and used[0].exact:
            more["rerun_of"] = {
                key: stamps[used[0].key].get(key) for key in ("tree", "verdict", "checkout", "finished")
            }
        self._stamp(verdict, **more)
        self.note += f"; stamped tree {self.tree.key[:10]} {verdict}"
        if verdict == RED:
            self.note += self._last_whole_green()

    def _durations(self, stamps: dict[str, dict[str, Any]]) -> dict[str, float]:
        """Each file's seconds: this run's where it ran, else the stamp that vouched for it."""
        merged: dict[str, float] = {}
        for stamp in stamps.values():
            for path, seconds in dict(stamp.get("durations", {})).items():
                merged.setdefault(path, float(seconds))
        for path, seconds in self.durations.items():
            merged[path] = round(seconds, 2)
        return dict(sorted(merged.items()))

    def _last_whole_green(self) -> str:
        """Where to look when this red is news: commits since the last full green run here."""
        assert self.store is not None
        try:
            for name in _newest(self.store, "")[:40]:
                stamp = read_stamp(self.store, name) or {}
                head = str(stamp.get("head", ""))
                if stamp.get("verdict", GREEN) != GREEN or int(stamp.get("debt", 0)) or not head:
                    continue
                if subprocess.run(
                    ["git", "merge-base", "--is-ancestor", head, "HEAD"],
                    cwd=self.root, capture_output=True, timeout=60,
                ).returncode == 0:
                    return (
                        f"; the last full green run on this line was {head[:10]}"
                        f" ({stamp.get('finished')}): look in git log {head[:10]}..HEAD"
                    )
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        return ""

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        if self.note and self.workeroutput is None:
            terminalreporter.write_line(f"green stamp: {self.note}")


def show_plan(root: Path) -> int:
    """What `--skip-if-green` would run on this checkout, and why: a screen, in seconds.

    By file, before collecting: a file whose ids collect differently here runs whole in the
    gate, and this does not see that.
    """
    tree = tree_of(root)
    store = store_for(root)
    plan = plan_for(root, store, tree.key)
    head = f"tree {tree.key[:10]}"
    if tree.exempt:
        head += f" (left out, uncommitted: {', '.join(tree.exempt)})"
    print(head)
    if not plan.bases:
        print(f"would run the whole fast suite: {plan.why_not or 'no stamp to lean on'}")
        return 0
    stamps = {base.key: read_stamp(store, base.key) or {} for base in plan.bases}
    tests = plan.tests
    if tests is None:
        tests = frozenset(affected.graph_of(str(root), tree.key, cache=store.parent / CACHE, not_readers=NOT_READERS).tests)
    rerun: set[str] = set(tests)
    leaned: list[Base] = []
    for base in plan.bases:
        files = stamps[base.key].get("files")
        covers = {t for t in rerun if t not in base.reached and (files is None or t in files)}
        if covers:
            leaned.append(base)
            rerun -= covers
    for base in leaned:
        stamp = stamps[base.key]
        how = "same code" if base.exact else f"{len(base.drift)} paths apart, reaching {len(base.reached)} test files"
        print(
            f"  leans on {base.key[:10]}  {Path(str(stamp.get('checkout', '?'))).name:<16}"
            f" {stamp.get('finished')}  {stamp.get('verdict', GREEN):<5} debt {stamp.get('debt', 0)}  {how}"
        )
    durations: dict[str, float] = {}
    for stamp in stamps.values():
        for path, seconds in dict(stamp.get("durations", {})).items():
            durations.setdefault(path, float(seconds))
    known = [durations[t] for t in rerun if t in durations]
    cost = f", about {round(sum(known) / 4)} s of tests at -n 4" if known else ""
    print(
        f"would re-run {len(rerun)} of {len(tests)} test files{cost}, plus what those stamps"
        f" skipped or failed and the {len(plan.skipped_lately)} tests that skipped on a recent"
        " stamp (they read outside the tree)"
    )
    if rerun:
        print("  " + ", ".join(sorted(rerun)[:12]) + (" ..." if len(rerun) > 12 else ""))
    if leaned:
        debt = max(base.debt for base in leaned)
        rests = min((full_run_of(stamps[b.key]) for b in leaned), key=lambda run: float(run["at"]))
        due = time.strftime(WHEN, time.localtime(float(rests["at"]) + FULL_EVERY_HOURS * 3600))
        print(f"debt after: {debt} of {DEBT_LIMIT}; rests on the full run of {rests.get('finished')}; a full run is due by {due}")
    if plan.why_not:
        print(f"not leaned on: {plan.why_not}")
    return 0


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
    parser = argparse.ArgumentParser(description="Carry a green stamp between clones, or say what the gate would run.")
    commands = parser.add_subparsers(dest="command", required=True)
    out = commands.add_parser("export", help="put this tree's stamp on <branch>-green")
    out.add_argument("--push", metavar="REMOTE", help="and push that branch there")
    into = commands.add_parser("import", help="file the stamp carried on REF here")
    into.add_argument("ref")
    commands.add_parser("plan", help="what --skip-if-green would run here, and which stamps it leans on")
    args = parser.parse_args(argv)
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    if args.command == "export":
        return export(root, args.push)
    if args.command == "plan":
        return show_plan(root)
    return import_(root, args.ref)


if __name__ == "__main__":
    sys.exit(main())
