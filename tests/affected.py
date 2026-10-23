"""What a change can reach: the test files a gate re-runs when another tree's stamp vouches for the rest.

`green_stamp` lets a stamp taken on one tree vouch for a test on another when the change
between them cannot reach that test's file. This module says what a change reaches, from
the tree's own Python and nothing else: no coverage run, no engine, a second or two.

A changed path reaches **every test** when it is under `rust/`, is `pyproject.toml`,
`uv.lock` or `tests/conftest.py`, does not parse, or is named by no Python file (there is
then nothing to follow), except under `docs/` and `.claude/`, where a file nothing names is
a note or a screenshot and reaches nothing. Otherwise it reaches the test files among:

- the file itself, the files that import it, and the files that name its module in a string
  (`"chess_variant_sandbox.x.y"` for a monkeypatch, `"green_stamp"` for `pytest_plugins`);
- the files that name its file name in a string, or its directory as a path component
  (`ROOT / "rulesets"` reads a new ruleset);
- for any Python change, the test files that glob `"*.py"` and read source as text, or
  import a helper beside them that does;
- for a note under `docs/` or `.claude/`, only what names it by file name (no directory);
- then the files that import any of those, to `DEPTH` steps; and when that reaches
  `tests/conftest.py`, the test files that ask for one of its fixtures.

Docstrings are left out: they name files the code never opens.

Depth, not the whole closure: the median test file imports 189 of the package's 228 modules
transitively, so the closure reaches everything. Depth 2 was measured against every red run
on record (`docs/reference/affected-gate.md`); what it misses is caught by the next full run,
which `green_stamp.DEBT_LIMIT` keeps a few gates away.
"""

# compute-lease: light
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Any

DEPTH = 2
PACKAGE = "chess_variant_sandbox"
CONFTEST = "tests/conftest.py"
# Changes every test can see, whatever imports what.
EVERYWHERE = ("rust/", "pyproject.toml", "uv.lock", ".python-version", CONFTEST)
# Where a file no Python names is prose or a picture, not something a test can read.
PROSE = ("docs/", ".claude/")
# A test file naming this in a string globs source as text.
SOURCE_GLOB = "*.py"
SEPARATOR = "\x00"
PATH_SPLIT = re.compile(r"[/\\]")


def module_name(path: str) -> str | None:
    """The name a Python file is imported by: the package's dotted path, or a bare name for a
    file directly in `tests/`, which pytest puts on `sys.path`."""
    if not path.endswith(".py"):
        return None
    if path.startswith(f"{PACKAGE}/"):
        return path[: -len(".py")].replace("/", ".").removesuffix(".__init__")
    if path.startswith("tests/") and path.count("/") == 1:
        return path[len("tests/") : -len(".py")]
    return None


def import_name(path: str) -> str | None:
    """`module_name`, or for a Python file elsewhere (`.claude/scripts/`, the root) the bare
    name a test imports it by once its directory is on `sys.path`."""
    name = module_name(path)
    if name is None and path.endswith(".py"):
        stem = PurePosixPath(path).stem
        name = None if stem == "__init__" else stem
    return name


def is_test_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return path.startswith("tests/") and name.startswith("test_") and name.endswith(".py")


@dataclass(frozen=True)
class Parsed:
    imports: frozenset[str]  # every name imported, made absolute
    strings: str  # the string constants, docstrings left out, joined by SEPARATOR
    components: frozenset[str]  # every path component of those strings: "docs", "page", ...
    parameters: frozenset[str]  # every function parameter: the fixtures a test asks for
    defined: frozenset[str]  # the functions defined at the top: a conftest's fixtures


def _absolute(path: str, node: ast.ImportFrom) -> str:
    base = node.module or ""
    if not node.level:
        return base
    parts = (module_name(path) or "").split(".")
    if not path.endswith("__init__.py"):
        parts = parts[:-1]
    parts = parts[: len(parts) - (node.level - 1)]
    return ".".join([*parts, base] if base else parts)


def parse(path: str, text: str) -> Parsed | None:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    imported: set[str] = set()
    strings: list[str] = []
    parameters: set[str] = set()
    prose = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute(path, node)
            imported.add(base)
            imported.update(f"{base}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in prose and len(node.value) < 400:
                strings.append(node.value)
        elif isinstance(node, ast.arguments):
            parameters.update(a.arg for a in (*node.posonlyargs, *node.args, *node.kwonlyargs))
    return Parsed(
        imports=frozenset(name for name in imported if name),
        strings=SEPARATOR.join(strings),
        components=frozenset(
            part for text in strings for part in PATH_SPLIT.split(text) if 0 < len(part) < 80
        ),
        parameters=frozenset(parameters),
        defined=frozenset(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
    )


@dataclass(frozen=True)
class Reach:
    everything: str | None  # why every test is reached, or None
    tests: frozenset[str]  # the test files reached; every one when `everything` says why


class Graph:
    """One tree's Python: who imports what, and what each file names in a string."""

    def __init__(self, parsed: dict[str, Parsed | None], not_readers: tuple[str, ...] = ()) -> None:
        # Files that name notes only as examples (`green_stamp.NOT_READERS`).
        self.not_readers = not_readers
        self.files = frozenset(parsed)
        self.tests = frozenset(path for path in parsed if is_test_file(path))
        self.parsed = {path: one for path, one in parsed.items() if one is not None}
        self.unparsed = {path for path, one in parsed.items() if one is None}
        self._memo: dict[tuple[str, int], Reach] = {}

    @cached_property
    def importers(self) -> dict[str, frozenset[str]]:
        """Module name -> the files that import it or anything under it."""
        found: dict[str, set[str]] = defaultdict(set)
        for path, parsed in self.parsed.items():
            for name in parsed.imports:
                parts = name.split(".")
                for i in range(1, len(parts) + 1):
                    found[".".join(parts[:i])].add(path)
        return {name: frozenset(paths) for name, paths in found.items()}

    @cached_property
    def scanners(self) -> frozenset[str]:
        """The test files that glob Python source and read it as text, themselves or through a
        helper beside them (`green_stamp.unread` greps every tracked `*.py`)."""
        helpers = {
            name
            for path, parsed in self.parsed.items()
            if path.startswith("tests/") and not is_test_file(path)
            and SOURCE_GLOB in parsed.strings and (name := import_name(path)) is not None
        }
        return frozenset(
            t
            for t in self.tests
            if t in self.parsed
            and (SOURCE_GLOB in self.parsed[t].strings or self.parsed[t].imports & helpers)
        )

    @cached_property
    def fixture_users(self) -> frozenset[str]:
        """The test files that ask for a fixture `tests/conftest.py` defines.

        What conftest imports reaches these, and only these. Its autouse fixtures are the
        session's plumbing (a table cache, a scratch store); `green_stamp.DEBT_LIMIT` and
        the next full run stand behind that.
        """
        conftest = self.parsed.get(CONFTEST)
        if conftest is None:
            return self.tests
        return frozenset(
            t
            for t in self.tests
            if t in self.parsed and self.parsed[t].parameters & conftest.defined
        )

    def naming(self, needle: str) -> set[str]:
        """The files that name `needle` in a string, docstrings aside."""
        return {path for path, parsed in self.parsed.items() if needle in parsed.strings}

    def _naming_directory(self, name: str) -> set[str]:
        return {path for path, parsed in self.parsed.items() if name in parsed.components}

    def reach_one(self, path: str, depth: int = DEPTH) -> Reach:
        key = (path, depth)
        if key not in self._memo:
            self._memo[key] = self._reach_one(path, depth)
        return self._memo[key]

    def _reach_one(self, path: str, depth: int) -> Reach:
        if path.startswith(EVERYWHERE):
            return Reach(f"{path} reaches every test", self.tests)
        if path in self.unparsed:
            return Reach(f"{path} does not parse", self.tests)
        pure = PurePosixPath(path)
        module = import_name(path)
        frontier: set[str] = set()
        if path in self.files:
            frontier.add(path)
        elif module is not None:  # deleted: what still imports it
            frontier |= self.importers.get(module, frozenset())
        named = self.naming(pure.name)
        if module is not None:  # a monkeypatch's "pkg.mod.attr", a `pytest_plugins` entry
            named |= self.naming(module) if "." in module else self._naming_directory(module)
        directory = pure.parent.name
        if path.startswith(PROSE) and not path.endswith(".py"):
            # A note is read by name or not at all (`green_stamp.unread`): every coordinator
            # commit touches the queue, and `"docs"` is a component half the package names.
            named = {p for p in named if not p.startswith(self.not_readers)}
        elif not path.endswith(".py"):
            if directory:
                named |= self._naming_directory(directory)
        elif not is_test_file(path):
            # Source read as text: by a glob, or by a path a test builds from directory names.
            named |= self.scanners
            if directory:
                named |= self._naming_directory(directory) & self.tests
        frontier |= named
        if not frontier:
            if path.startswith(PROSE):  # a note or a screenshot: what `green_stamp.unread` clears
                return Reach(None, frozenset())
            return Reach(f"nothing names {path}", self.tests)
        seen = set(frontier)
        for _ in range(depth):
            frontier = {
                importer
                for f in frontier
                if (name := import_name(f)) is not None
                for importer in self.importers.get(name, frozenset())
            } - seen
            seen |= frontier
        tests = seen & self.tests
        if CONFTEST in seen:
            tests |= self.fixture_users
        return Reach(None, frozenset(tests))

    def reach(self, paths: Iterable[str], depth: int = DEPTH) -> Reach:
        tests: set[str] = set()
        for path in paths:
            one = self.reach_one(path, depth)
            if one.everything is not None:
                return one
            tests |= one.tests
        return Reach(None, frozenset(tests))


def _git(root: str, *args: str, input: bytes | None = None) -> bytes:
    # Bytes both ways: a Windows text pipe writes "\n" as "\r\n", and git reads the "\r".
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, check=True, input=input, timeout=120
    ).stdout


def python_blobs(root: str, tree: str) -> dict[str, str]:
    """Every Python file in a git tree, and its blob."""
    listing = _git(root, "-c", "core.quotepath=off", "ls-tree", "-r", tree).decode("utf-8")
    blobs: dict[str, str] = {}
    for line in listing.splitlines():
        meta, path = line.split("\t", 1)
        kind, sha = meta.split()[1:3]
        if kind == "blob" and path.endswith(".py"):
            blobs[path] = sha
    return blobs


def read_blobs(root: str, blobs: dict[str, str]) -> dict[str, str]:
    """These blobs' text, read in one `git cat-file`."""
    if not blobs:
        return {}
    request = "".join(f"{sha}\n" for sha in blobs.values()).encode()
    data = _git(root, "cat-file", "--batch", input=request)
    texts: dict[str, str] = {}
    at = 0
    for path in blobs:
        newline = data.index(b"\n", at)
        size = int(data[at:newline].split()[2])
        texts[path] = data[newline + 1 : newline + 1 + size].decode("utf-8", errors="replace")
        at = newline + 1 + size + 1
    return texts


# What parsing found, by blob, so a gate parses only what changed: 681 files take 5-15 s.
# Keyed by this module's own text too, so a change to how files are read starts it afresh.
VERSION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
CACHE_ENTRIES = 4000


def _to_json(one: Parsed | None) -> list[Any] | None:
    if one is None:
        return None
    return [sorted(one.imports), one.strings, sorted(one.components), sorted(one.parameters), sorted(one.defined)]


def _from_json(data: list[Any] | None) -> Parsed | None:
    if data is None:
        return None
    imports, strings, components, parameters, defined = data
    return Parsed(frozenset(imports), strings, frozenset(components), frozenset(parameters), frozenset(defined))


def graph_of(root: str, tree: str, cache: Path | None = None, not_readers: tuple[str, ...] = ()) -> Graph:
    """The graph of a git tree's Python, parsing only the blobs `cache` has not seen."""
    blobs = python_blobs(root, tree)
    known: dict[str, Any] = {}
    if cache is not None:
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved.get("version") == VERSION:
                known = dict(saved["entries"])
        except (OSError, ValueError, KeyError, AttributeError):
            known = {}
    keys = {path: f"{sha} {path}" for path, sha in blobs.items()}
    new = {path: blobs[path] for path, key in keys.items() if key not in known}
    for path, text in read_blobs(root, new).items():
        known[keys[path]] = _to_json(parse(path, text))
    graph = Graph({path: _from_json(known[key]) for path, key in keys.items()}, not_readers)
    if cache is not None and new:
        mine = set(keys.values())
        others = [key for key in known if key not in mine][: max(0, CACHE_ENTRIES - len(mine))]
        kept = {key: known[key] for key in [*mine, *others]}
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            scratch = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
            scratch.write_text(json.dumps({"version": VERSION, "entries": kept}), encoding="utf-8")
            os.replace(scratch, cache)
        except OSError:
            pass  # a cache that can't be written costs time, not answers
    return graph
