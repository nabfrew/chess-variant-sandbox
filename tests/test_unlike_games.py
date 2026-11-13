"""The designed games of the search for strong games unlike chess (`unlike-search.md` §2, §9).

Each `unlike/*.json` is a session's admitted game: its hash is the one §9 records (and is what
gets measured), it holds the brief's limits (a) to (c) from the spec alone (`unlike/limits.py`),
and its session record `unlike/sessions/<name>.md` sits beside it. Nothing here plays a game
or reads a score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.spec import load_spec, spec_hash

ROOT = Path(__file__).resolve().parents[1]
UNLIKE = ROOT / "unlike"

# Each admitted game and the hash §9's table records for it, in session order.
ADMITTED = {
    "drove": "7c0bc4016910",  # session 2, majority control
    "smugglers": "73da5f8f7bd1",  # session 4, a banked race with a hand
    "triad": "8716890978a9",  # session 5, a circle of captures lost by extinction
    "thresh": "a52971eab2d1",  # session 6, hex hoppers and sweepers
    "gorgons": "43e5b764b9a8",  # session 7, lines of sight
    "wardens": "0ee8d3ecb57b",  # session 8, a turn of two parts
}


def _games() -> list[Path]:
    return sorted(UNLIKE.glob("*.json"))


@pytest.fixture(scope="module")
def library() -> tuple[Any, list[str], list[dict[str, Any]]]:
    """`unlike/limits.py` (a script, not a package module) and the library it reads against."""
    import importlib.util

    loaded = importlib.util.spec_from_file_location("unlike_limits", UNLIKE / "limits.py")
    assert loaded is not None and loaded.loader is not None
    module = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(module)
    hashes, specs = module.library_specs()
    return module, hashes, specs


def test_every_unlike_game_is_an_admitted_game_at_its_recorded_hash() -> None:
    found = {path.stem: spec_hash(load_spec(path)) for path in _games()}
    assert found == ADMITTED


@pytest.mark.parametrize("name", sorted(ADMITTED))
def test_each_has_its_session_record(name: str) -> None:
    record = UNLIKE / "sessions" / f"{name}.md"
    assert record.exists()
    text = record.read_text(encoding="utf-8")
    assert "design reason" in text and "Blind read" in text
    assert (UNLIKE / "sessions" / name).is_dir()


@pytest.mark.parametrize("name", sorted(ADMITTED))
def test_each_holds_the_briefs_limits(name: str, library: Any) -> None:
    module, hashes, specs = library
    read = module.read(load_spec(UNLIKE / f"{name}.json"), hashes, specs)
    assert read["a"], f"likeness to chess {read['likeness']:.4f} is not below {module.CUT}"
    assert read["b"], f"{read['chess_pieces']} pieces of chess"
    assert read["c"], f"nearest library game {read['nearest']}"
