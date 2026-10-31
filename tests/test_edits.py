"""A mechanics edit in place cannot orphan a child's lineage silently (Q1075).

`spec.edits.EDITED` maps the hash a committed file held before an in-place mechanics edit to
the hash it held after, and the lineage readers (`graphing._lineage`, `Game.ancestry`) resolve a
recorded parent through it. Two holds: every edit re-frozen in `tests/test_schema_nine.py::FROZEN`
with its old hash in a `was <hash>` comment (the convention, CLAUDE.md) has its row here, and
every `provenance.parents` entry of a committed or archived ruleset resolves to a game -- one in
the tree, or one retired on purpose and kept by `tests/retired_specs.py`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from chess_variant_sandbox.spec import edits, load_spec, spec_hash
from tests import retired_specs

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SOURCE = ROOT / "tests" / "test_schema_nine.py"
_ROW = re.compile(r'^"([^"]+)": \("([0-9a-f]{12})", "[0-9a-f]{16}"\),?\s*(#.*)?$')
_WAS = re.compile(r"\bwas ([0-9a-f]{12})\b")


def _frozen_edits() -> list[tuple[str, str, str]]:
    """`(path, old hash, hash now)` for every FROZEN row whose comments name an old hash.

    A row's comments are the comment lines above it, back to the previous row, and its own
    trailing comment; a 12-hex `was` is a mechanics hash (a tables fingerprint is 16).
    """
    text = FROZEN_SOURCE.read_text(encoding="utf-8")
    block = text[text.index("FROZEN: dict"):]
    block = block[:block.index("\n}\n")]
    found: list[tuple[str, str, str]] = []
    pending: list[str] = []
    for line in block.splitlines()[1:]:
        stripped = line.strip()
        row = _ROW.match(stripped)
        if row:
            said = " ".join([*pending, row.group(3) or ""])
            found.extend((row.group(1), old, row.group(2)) for old in _WAS.findall(said))
            pending = []
        elif stripped.startswith("#"):
            pending.append(stripped)
    return found


def _known_games() -> set[str]:
    """Every game a parent may name: the tree's rulesets and archive, and the kept retirees."""
    paths = sorted((ROOT / "rulesets").glob("*.json")) + sorted((ROOT / "archive").glob("*.json"))
    known = {spec_hash(load_spec(path)) for path in paths}
    known |= set(retired_specs.RETIRED_HASHES.values())
    known |= set(retired_specs.RETIRED_PREVIEWS.values())
    known |= set(retired_specs.REFUSED_PREVIEWS.values())
    return known


def _declared_parents() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for folder in ("rulesets", "archive"):
        for path in sorted((ROOT / folder).glob("*.json")):
            spec = json.loads(path.read_text(encoding="utf-8"))
            for parent in (spec.get("provenance") or {}).get("parents") or []:
                digest = parent.get("hash") if isinstance(parent, dict) else parent
                found.append((f"{folder}/{path.name}", str(digest)))
    return found


class TestTheEditsRecord:
    def test_the_frozen_comments_name_some_edits(self) -> None:
        """The parser reads the convention, or the next hold would pass on nothing."""
        assert ("rulesets/chess.json", "4c1922ffae26", "831c77e75650") in _frozen_edits()
        assert len(_frozen_edits()) >= 20

    def test_every_edit_frozen_with_its_old_hash_is_recorded(self) -> None:
        """Re-freezing a row with `was <old>` and no `EDITED` row fails here, by name."""
        missing = [
            f"{path}: was {old}, now {now} (EDITED gives {edits.current(old)})"
            for path, old, now in _frozen_edits()
            if edits.current(old) != now
        ]
        assert not missing, missing

    def test_every_provenance_parent_resolves_to_a_game(self) -> None:
        """An edit that would orphan a child fails here instead of dropping an arrow."""
        known = _known_games()
        orphans = [
            f"{path}: {digest}" for path, digest in _declared_parents()
            if edits.current(digest) not in known
        ]
        assert not orphans, orphans

    def test_every_edit_ends_at_a_game(self) -> None:
        """Every row's target is a game today or the start of a later row: no dead ends."""
        known = _known_games()
        dead = [old for old, new in edits.EDITED.items() if new not in known
                and new not in edits.EDITED]
        assert not dead, dead


class TestCurrent:
    def test_a_hash_never_edited_is_itself(self) -> None:
        assert edits.current("000000000000") == "000000000000"

    def test_chess_s_old_hash_is_chess_now(self) -> None:
        assert edits.current("4c1922ffae26") == spec_hash(load_spec(ROOT / "rulesets/chess.json"))

    def test_a_file_edited_twice_resolves_to_its_last_hash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(edits, "EDITED", {"aaa": "bbb", "bbb": "ccc"})
        assert edits.current("aaa") == "ccc"

    def test_an_edit_undone_stops_rather_than_looping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(edits, "EDITED", {"aaa": "bbb", "bbb": "aaa"})
        assert edits.current("aaa") == "bbb"
