"""Paths this checkout has and the public copy does not, and the one way a test reads them (Q1210).

The public repository is an export of an allow-listed part of this one
(`docs/reference/public-fork.md`): `.claude/`, `deploy/`, `docs/` (all but the changelog),
`docs_tests/` and `CLAUDE.md` never leave. Most of `tests/` does, so a test that reads one of
those paths would fail there for a reason that is not a defect of the code it tests. Such a
test reads the path through this module and carries its mark:

    clean = private_paths.script(".claude/scripts/clean_runs.py", "clean_runs")
    pytestmark = private_paths.needs(".claude/scripts/clean_runs.py")

`needs` skips **only when a named path is absent**. In this repository every one is present,
so every such test runs here and asserts exactly what it always did
(`tests/test_private_paths.py` holds both halves: that each named path exists here, and that
no exported test reaches a private path except through this module).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]

# What the export never carries, as `export_public.NEVER` names it, restricted to what a test
# could reach in a checkout. `tests/test_private_paths.py` checks each against the exporter.
PRIVATE: tuple[str, ...] = (".claude/", "deploy/", "docs/", "docs_tests/", "CLAUDE.md")
# The one path under a private root that the export does carry.
EXPORTED: tuple[str, ...] = ("docs/changelog.md",)

REASON = "this checkout only: the public copy carries no {}"


def path(relative: str) -> Path:
    """`relative` (a repository path in `/` form) in this checkout. It does not check that the
    path is there; the test that reads it carries `needs` for it, which does."""
    return REPOSITORY / relative


def private(relative: str) -> bool:
    """Whether the export leaves `relative` (a repository path in `/` form) behind."""
    if relative in EXPORTED:
        return False
    return any(relative.startswith(root) if root.endswith("/") else relative == root
               for root in PRIVATE)


def absent(*relatives: str) -> list[str]:
    return [relative for relative in relatives if not path(relative).exists()]


def needs(*relatives: str) -> pytest.MarkDecorator:
    """A mark that skips the test (or class, or module, as `pytestmark`) when any of
    `relatives` is absent from this checkout, and only then."""
    missing = absent(*relatives)
    return pytest.mark.skipif(bool(missing), reason=REASON.format(", ".join(missing)))


class Missing:
    """What `script` returns for a script the checkout lacks. A test that reaches it without
    `needs` for the same path is a bug in the test; it fails here naming the mark."""

    def __init__(self, relative: str) -> None:
        self._relative = relative

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            f"{self._relative} is absent and the test reading `{name}` of it lacks "
            f'@private_paths.needs("{self._relative}")'
        )


def script(relative: str, name: str | None = None) -> Any:
    """The Python script at `relative`, loaded as a module (and registered in `sys.modules`
    as `name`, when given), or a `Missing` stand-in where the checkout lacks it. Loading at
    import time is safe either way, so a module can hold it at top level and mark only the
    tests that use it."""
    where = path(relative)
    if not where.is_file():
        return Missing(relative)
    module_name = name or where.stem
    found = importlib.util.spec_from_file_location(module_name, where)
    assert found is not None and found.loader is not None
    module = importlib.util.module_from_spec(found)
    if name is not None:
        sys.modules[name] = module
    found.loader.exec_module(module)
    return module
