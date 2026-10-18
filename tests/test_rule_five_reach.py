"""Rule 5's constraint 5c, enforced over the import graph rather than promised.

`docs/reference/human-data.md` §3 reads rule 5 (*human results are a labelled quantity beside the
index, never an input to it*, `docs/plan-website.md`) as four constraints. The tests in
`tests/test_web.py` and `tests/test_site_correspondence.py::TestTheRules` hold 5a: nothing
under `metrics/` names the site's store or `web.human`. They say nothing about **5c**: *no
number in a score's basis depends on a person's game, directly or through a computer player
whose selection or acceptance a person's game decided*. A computer player is chosen and
accepted by `tournament/` (duels, ratings, fitting), built by `agents/`, and trained by the
net procedure (`agents/training/`); any of them reading people's games would break 5c with
every 5a test green (`docs/reference/human-data.md` §3, the brief's gap; task H5).

So this file holds that no module under those three places reaches, through any chain of the
package's own imports, the site (`chess_variant_sandbox.site`, which owns the database) or the
local human record (`chess_variant_sandbox.web.human`). Transitive on purpose: an import of a
helper that imports the store is the same dependency one step removed. And beside the graph,
the source is read for the store's names, so a module cannot open the database or the
record's file by path without importing either.

Routes B and C (`docs/reference/human-data.md` §3.2, §3.3) live under `site/` and `/admin`, reading
the rating table and writing beside the score; this test is what keeps them there.
"""

from __future__ import annotations

import ast
from pathlib import Path

import chess_variant_sandbox

PACKAGE = Path(chess_variant_sandbox.__file__).parent
ROOT = PACKAGE.name

# Where a computer player is built, trained, chosen and accepted (5c).
GUARDED = ("agents", "tournament")
# The net procedure lives under `agents/training/`; named so a move out of `agents/` is caught.
NET_PROCEDURE = "agents/training/procedure.py"

# What a guarded module may not reach: the site's store and the local human record.
FORBIDDEN = (f"{ROOT}.site", f"{ROOT}.web.human")
# What a guarded module's source may not name: the store's variable and the record's file.
FORBIDDEN_NAMES = ("DATABASE_URL", "HumanRecord", "HumanGame", '"human"', "'human'",
                   "web.human", "web import human")


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(PACKAGE.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _modules() -> dict[str, Path]:
    return {_module_name(p): p for p in PACKAGE.rglob("*.py")}


def _imports(name: str, path: Path, known: dict[str, Path]) -> set[str]:
    """Every module of the package that `path` imports, at any depth of the file."""
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        targets: list[str] = []
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)]
                stem = ".".join(base + ([node.module] if node.module else []))
            else:
                stem = node.module or ""
            targets = [stem] + [f"{stem}.{alias.name}" for alias in node.names]
        for target in targets:
            # `import a.b.c` loads `a` and `a.b` too: each is a dependency.
            parts = target.split(".")
            for i in range(1, len(parts) + 1):
                prefix = ".".join(parts[:i])
                if prefix in known:
                    found.add(prefix)
    found.discard(name)
    return found


def _reach(start: str, graph: dict[str, set[str]]) -> dict[str, str | None]:
    """Every module reachable from `start`, each with the module that first imported it."""
    seen: dict[str, str | None] = {start: None}
    frontier = [start]
    while frontier:
        current = frontier.pop()
        for nxt in sorted(graph[current]):
            if nxt not in seen:
                seen[nxt] = current
                frontier.append(nxt)
    return seen


def _chain(module: str, seen: dict[str, str | None]) -> str:
    steps = [module]
    while seen[steps[-1]] is not None:
        steps.append(seen[steps[-1]])  # type: ignore[arg-type]
    return " <- ".join(steps)


def _forbidden(module: str) -> bool:
    return any(module == f or module.startswith(f + ".") for f in FORBIDDEN)


def _guarded(modules: dict[str, Path]) -> list[str]:
    return sorted(
        name for name, path in modules.items()
        if path.relative_to(PACKAGE).parts[0] in GUARDED
    )


def test_the_guarded_places_exist() -> None:
    """A guard over an empty set passes for nothing: the places it watches are there."""
    modules = _modules()
    guarded = _guarded(modules)
    for place in GUARDED:
        assert any(name.startswith(f"{ROOT}.{place}.") for name in guarded), place
    assert (PACKAGE / NET_PROCEDURE).exists(), NET_PROCEDURE
    assert _module_name(PACKAGE / NET_PROCEDURE) in guarded
    for target in FORBIDDEN:
        assert target in modules, f"{target} moved: the guard would watch nothing"


def test_no_computer_player_reaches_peoples_games() -> None:
    """Rule 5c (`docs/reference/human-data.md` §3, task H5): nothing under `agents/`, the net
    procedure or `tournament/` imports the site's store or `web.human`, at any depth."""
    modules = _modules()
    graph = {name: _imports(name, path, modules) for name, path in modules.items()}
    breaches = []
    for start in _guarded(modules):
        seen = _reach(start, graph)
        # The first step into the forbidden set, one chain each: what to cut.
        breaches += [
            _chain(m, seen) for m in seen
            if _forbidden(m) and not _forbidden(seen[m] or "")
        ]
    assert not breaches, "rule 5c: a computer player reaches people's games:\n" + "\n".join(
        sorted(set(breaches))
    )


def test_no_computer_player_names_the_store() -> None:
    """The same, for a reach without an import: no guarded file names the database's
    variable or the human record's directory and types."""
    modules = _modules()
    for name in _guarded(modules):
        source = modules[name].read_text(encoding="utf-8")
        for word in FORBIDDEN_NAMES:
            assert word not in source, f"{modules[name].relative_to(PACKAGE)} names {word}"
