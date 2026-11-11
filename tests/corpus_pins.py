"""The per-game freeze rows a new game owes, written by a command rather than by hand (Q1154).

Two freezes hold a row for every square-board game in `rulesets/` and `archive/`:
`test_hex_readers._DESCRIPTIONS_BEFORE_THE_HEXAGON` (the digest of `describe_spec`) and
`test_hex_resolver._FEATURES_BEFORE_THE_RESOLVER` (the digest of `distance.features`). A row
that was there before a change guards that change: a describer or feature edit that moves an
existing game's text is caught, and re-freezing that row stays a hand edit with its reason in
a comment. A *new* game's row guards nothing yet -- it is whatever the code says on the day
the game arrives -- so it is generated:

    uv run python tests/corpus_pins.py

adds a row to `tests/corpus_pins.json` for every game neither freeze has, and drops a row of
that file whose game has left the corpus. It never changes a row that exists, in the file or
in the test modules: a frozen digest that moved is a failing test to argue, not a snapshot
to refresh. `--check` says what it would write and exits 1 if anything, writing nothing.

`docs/reference/corpus-pins.md` lists every pin that used to restate the corpus and what
became of it.
"""

# compute-lease: light

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SNAPSHOT = Path(__file__).resolve().parent / "corpus_pins.json"
COMMAND = "uv run python tests/corpus_pins.py"
FREEZES = ("descriptions", "features")


def load() -> dict[str, dict[str, str]]:
    """The generated rows, one table per freeze (empty tables when the file is absent)."""
    data: dict[str, Any] = (
        json.loads(SNAPSHOT.read_text(encoding="utf-8")) if SNAPSHOT.exists() else {}
    )
    return {name: dict(data.get(name, {})) for name in FREEZES}


def coverage_message(freeze: str, frozen: set[str], library: set[str]) -> str:
    """What a freeze's coverage test says when the corpus and its rows disagree."""
    missing, gone = sorted(library - frozen), sorted(frozen - library)
    return (
        f"the {freeze} freeze and the corpus disagree: no row for {missing}, a row for no game "
        f"{gone}. A game added or retired: run `{COMMAND}` (it writes tests/corpus_pins.json, "
        "and never changes an existing row). A hand-written row whose game left: delete it "
        "with a comment saying why."
    )


def _rows() -> dict[str, tuple[dict[str, str], dict[str, str], Any]]:
    import test_hex_readers as readers
    import test_hex_resolver as resolver

    snapshot = load()
    return {
        "descriptions": (
            readers._DESCRIPTIONS_BEFORE_THE_HEXAGON, snapshot["descriptions"], readers
        ),
        "features": (resolver._FEATURES_BEFORE_THE_RESOLVER, snapshot["features"], resolver),
    }


def sync(write: bool | None = None) -> dict[str, dict[str, list[str]]]:
    """The rows to add and drop; writes them unless `--check` (or `write=False`)."""
    if write is None:
        write = "--check" not in sys.argv[1:]
    changes: dict[str, dict[str, list[str]]] = {}
    written: dict[str, dict[str, str]] = {}
    for freeze, (by_hand, generated, module) in _rows().items():
        keys = {module._key(path): path for path in module.LIBRARY}
        add = sorted(set(keys) - set(by_hand) - set(generated))
        drop = sorted(set(generated) - set(keys))
        table = {key: value for key, value in generated.items() if key not in drop}
        for key in add:
            table[key] = module.frozen_digest(keys[key])
        written[freeze] = dict(sorted(table.items()))
        changes[freeze] = {"add": add, "drop": drop}
    if write and any(
        change["add"] or change["drop"] for change in changes.values()
    ):
        SNAPSHOT.write_text(
            json.dumps({"_written_by": COMMAND, **written}, indent=2) + "\n", encoding="utf-8"
        )
    return changes


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    result = sync()
    for freeze, change in result.items():
        for key in change["add"]:
            print(f"{freeze}: added {key}")
        for key in change["drop"]:
            print(f"{freeze}: dropped {key}")
    pending = any(change["add"] or change["drop"] for change in result.values())
    if not pending:
        print("corpus_pins.json is current")
    sys.exit(1 if pending and "--check" in sys.argv[1:] else 0)
