"""The `new_game` mark and `--games`: the checks a new or edited ruleset trips (Q1156).

A test marked `new_game` reads the committed corpus (`rulesets/`, `archive/`) and is one a
game added, edited or retired can turn red: a frozen identity, a coverage rule, a ceiling, a
claim about which games do what (`docs/reference/corpus-pins.md` §3 and §5 say which and why).
`tests/check_new_game.py` runs exactly these, in seconds:

    uv run python tests/check_new_game.py

`--games a.json,b.json` narrows the marked tests that take one game as a parameter to those
games, so a designer pays for their own game and not for the 130 others; the marked tests
that read the whole corpus at once (does `FROZEN` cover it, is the grid crowded, do two games
measure identical) always run. `--games all` keeps every game. Without `--games` nothing is
narrowed: the fast suite runs these tests as it always did.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
FOLDERS = ("rulesets", "archive")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--games",
        default=None,
        help="comma-separated ruleset files (or `all`): the per-game `new_game` tests to keep",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "new_game: reads the committed corpus; a game added, edited or retired can trip it "
        "(run them with `uv run python tests/check_new_game.py`)",
    )


def game_file(value: Any) -> Path | None:
    """The corpus file a test parameter names, or None when it names none."""
    if isinstance(value, str) and value.endswith(".json"):
        value = Path(value)
    if not isinstance(value, Path) or value.suffix != ".json":
        return None
    path = value if value.is_absolute() else REPOSITORY / value
    path = path.resolve()
    return path if path.parent.name in FOLDERS else None


def chosen(raw: str) -> set[Path] | None:
    """The files `--games` names, resolved; None for `all`."""
    if raw.strip() == "all":
        return None
    named = set()
    for part in raw.split(","):
        if part.strip():
            path = Path(part.strip())
            named.add((path if path.is_absolute() else REPOSITORY / path).resolve())
    return named


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    raw = config.getoption("--games")
    if raw is None:
        return
    games = chosen(raw)
    if games is None:
        return
    keep: list[pytest.Item] = []
    drop: list[pytest.Item] = []
    for item in items:
        callspec = getattr(item, "callspec", None)
        files = (
            {game_file(value) for value in callspec.params.values()} - {None}
            if callspec is not None and item.get_closest_marker("new_game") is not None
            else set()
        )
        (drop if files and not files & games else keep).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
