"""The day each library game joined the library (Q826), and the order that reads it.

The user, 2026-10-10: *"Make the highlighting of recently added games update automatically
(currently it's out of date) and make it possible to sort games in the `/games` section by
date created."* The shelf was a hand-kept list; a library file now records the day it joined
as `provenance.added`, outside the mechanics hash, written for every existing file from git
by `.claude/scripts/backfill_added.py` and for a new one by `corpus.archive_game` and
`corpus.promote`. The shelf's window is `tests/test_launch.py`'s; the site's member days
are `tests/test_site_stages.py`'s.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from test_corpus import SPEC, _find, _measured, world  # noqa: F401  (`world` is a fixture)

from chess_variant_sandbox import corpus
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.web import library

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".claude" / "scripts" / "backfill_added.py"
_spec = importlib.util.spec_from_file_location("backfill_added", SCRIPT)
assert _spec is not None and _spec.loader is not None
backfill = importlib.util.module_from_spec(_spec)
sys.modules["backfill_added"] = backfill
_spec.loader.exec_module(backfill)

LIBRARY = sorted(path for folder in ("rulesets", "archive") for path in (ROOT / folder).glob("*.json"))


class TestEveryLibraryFileSaysWhenItJoined:
    @pytest.mark.parametrize("path", LIBRARY, ids=lambda p: f"{p.parent.name}/{p.stem}")
    def test_a_day_and_a_real_one(self, path: Path) -> None:
        """A file added without one fails here by name: run the backfill with `--write`,
        which stamps a file git has not seen yet with today."""
        added = (load_spec(path).get("provenance") or {}).get("added")
        assert isinstance(added, str), (
            f"{path.name} has no provenance.added: "
            "python .claude/scripts/backfill_added.py --write")
        day = date.fromisoformat(added)
        assert date(2026, 8, 1) <= day <= date.today(), added

    @pytest.mark.parametrize(
        ("name", "day"),
        [
            # The first commit of the corpus.
            ("rulesets/chess.json", "2026-08-19"),
            # Q788's four of 9 October, and the portal games of the tenth.
            ("archive/circe.json", "2026-10-09"),
            ("archive/rifle.json", "2026-10-09"),
            ("archive/carousel.json", "2026-10-10"),
            ("archive/wormhole.json", "2026-10-10"),
            # The schema-5 boards of the corpus's second week. Two renamed files kept the
            # day of the name they arrived under (`sardines-torus`, `tunnel-vertical`); both
            # left the library on 2026-10-12 (Q868).
            ("rulesets/cylinder.json", "2026-08-28"),
            ("archive/onewaytorus.json", "2026-08-30"),
            # One of the five files that had no provenance block at all.
            ("rulesets/rps-v2.json", "2026-08-21"),
        ],
    )
    def test_the_backfill_wrote_the_day_git_first_saw_it(self, name: str, day: str) -> None:
        assert load_spec(ROOT / name)["provenance"]["added"] == day

    def test_the_card_carries_it(self, tmp_path: Path) -> None:
        # The two cards asked about, not all of them: a card's summary runs the library's
        # smoke gate on its game, 23 s of processor time over the whole corpus (Q749).
        # Listed over an empty `runs/` (Q893): the day does not come from a measurement,
        # and a `runs/` that holds them makes the listing ask `tables_fingerprint` of every
        # game, which compiles the library in whichever test lists first (15.7 s here
        # with no seed; 9.6 s in a gate, 2026-10-13).
        found = library.discover(library.RULESET_DIR, tmp_path, library.ARCHIVE_DIR)
        games = {game.name: game for game in found if game.source != "evolved"}
        assert games["chess"].summary()["added"] == "2026-08-19"
        assert games["circe"].summary()["added"] == "2026-10-09"


class TestTheFieldMovesNoGame:
    def test_the_hash_ignores_it(self) -> None:
        spec = load_spec(ROOT / "rulesets" / "chess.json")
        moved = {**spec, "provenance": {**spec["provenance"], "added": "2030-01-01"}}
        assert spec_hash(moved) == spec_hash(spec) == "831c77e75650"

    def test_the_validator_takes_a_day_and_refuses_anything_else(self) -> None:
        spec = load_spec(ROOT / "rulesets" / "chess.json")
        assert validate_spec(spec) == []
        for bad in ("10 October 2026", "2026-13-01", "2026-1-9", 20261009):
            wrong = {**spec, "provenance": {**spec["provenance"], "added": bad}}
            assert any("provenance.added" in problem for problem in validate_spec(wrong)), bad

    def test_only_a_library_file_speaks_for_its_day(self, tmp_path: Path) -> None:
        game = library.discover(library.RULESET_DIR, tmp_path, library.ARCHIVE_DIR)[0]
        assert library.added(game) is not None
        for source in ("designed", "evolved", "retired"):
            assert library.added(_Stand(source, game.spec)) is None, source


class _Stand:
    def __init__(self, source: str, spec: dict[str, Any]) -> None:
        self.source = source
        self.spec = spec


class TestTheBackfill:
    def test_the_oldest_day_git_names(self) -> None:
        assert backfill.oldest_day("2026-10-09\n\n2026-08-20\n") == "2026-08-20"
        assert backfill.oldest_day("") is None

    def test_a_rename_is_followed_and_a_copy_is_not(self) -> None:
        log = "R069\trulesets/old.json\trulesets/new.json\nC084\ta.json\tb.json\nM\tc.json\n"
        assert backfill.parse_renames(log) == {"rulesets/new.json": "rulesets/old.json"}

    @pytest.mark.parametrize(
        "text",
        [
            '{\n  "name": "x",\n  "provenance": {\n    "origin": "human"\n  }\n}\n',
            '{\n  "name": "x",\n  "provenance": {}\n}\n',
            '{"name":"x","provenance":{"origin":"human"},"rules":{}}',
            '{"name": "x", "provenance": {"origin": "human"}}',
            '{\n  "name": "x",\n  "rules": {\n    "win": "checkmate"\n  }\n}\n',
            '{"name":"x","description":"its provenance: {odd}","rules":{}}',
        ],
    )
    def test_the_day_goes_in_as_text_in_the_files_own_layout(self, text: str) -> None:
        after = backfill.insert_added(text, "2026-10-10")
        before = json.loads(text)
        expected = {**before, "provenance": {**before.get("provenance", {}),
                                             "added": "2026-10-10"}}
        assert json.loads(after) == expected
        # Nothing else in the file moved: every line of it is still there.
        kept = [line for line in text.splitlines() if "provenance" not in line]
        assert all(line.rstrip(",") in after for line in kept)

    def test_a_day_already_there_is_never_rewritten(self) -> None:
        with pytest.raises(ValueError, match="already"):
            backfill.insert_added('{"provenance": {"added": "2026-08-19"}}', "2026-10-10")


class TestANewGameIsStampedOnTheWayIn:
    def test_archiving_stamps_today_and_promoting_keeps_it(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        monkeypatch.setattr(corpus, "_today", lambda: "2026-10-10")
        kept = corpus.archive_game(_find(world), archive_dir=world["archive"],
                                   runs_dir=world["runs"])
        assert load_spec(kept)["provenance"]["added"] == "2026-10-10"
        assert spec_hash(load_spec(kept)) == spec_hash(SPEC)

        monkeypatch.setattr(corpus, "_today", lambda: "2026-10-20")
        _measured(world)
        promoted = corpus.promote(_find(world), reason="it pins something down",
                                  ruleset_dir=world["rulesets"], runs_dir=world["runs"])
        assert load_spec(promoted)["provenance"]["added"] == "2026-10-10"

    def test_a_candidates_own_claim_is_not_believed(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        monkeypatch.setattr(corpus, "_today", lambda: "2026-10-10")
        candidate = _find(world)
        claimed = {**candidate.spec, "provenance": {"added": "2020-01-01"}}
        object.__setattr__(candidate, "spec", claimed)
        kept = corpus.archive_game(candidate, archive_dir=world["archive"],
                                   runs_dir=world["runs"])
        assert load_spec(kept)["provenance"]["added"] == "2026-10-10"


class TestTheGamesPageSortsByIt:
    def test_date_added_is_in_the_first_tier_and_reads_newest_first(self) -> None:
        from page_text import PAGE

        assert "['added', 'date added']" in PAGE
        assert "const FIRST_SORTS = ['name', 'index', 'added', 'stars', 'humanGames'];" in PAGE
        body = PAGE[PAGE.index("function dirWords()"):]
        body = body[:body.index("\n}\n")]
        assert "'newest first'" in body and "'oldest first'" in body
        # Picking it orders newest first, as a measurement orders best first.
        assert "BROWSE.descending = BROWSE.sort !== 'name';" in PAGE

    def test_a_card_says_its_day_when_the_list_is_ordered_by_it(self) -> None:
        from page_text import PAGE

        body = PAGE[PAGE.index("function sortedByHtml(game)"):]
        body = body[:body.index("\n}\n")]
        assert "key === 'added'" in body and "dayWords(game.added)" in body

    def test_the_site_takes_it_as_a_query(self) -> None:
        from chess_variant_sandbox.site import browse

        cards = [
            {"hash": "a", "name": "old", "added": "2026-08-19"},
            {"hash": "b", "name": "none", "added": None},
            {"hash": "c", "name": "new", "added": "2026-10-10"},
            {"hash": "d", "name": "also new", "added": "2026-10-10"},
        ]
        assert browse.ADDED in browse.SORTS
        assert [card["name"] for card in browse.order(cards, browse.ADDED)] == [
            "also new", "new", "old", "none"]
