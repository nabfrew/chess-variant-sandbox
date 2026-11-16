"""Every library score says whether stronger players would be likely to change it (R-S4c).

The reading is R-S4b's (`docs/reference/strength-check-all-games.md` §2.3, §14), carried in
the tracked file `chess_variant_sandbox/strength_check.json` and read by
`measurement.strength_reading`. What is held down here:

- **The file**: every library game has a reading or a stated reason for none; its counts are
  §14's (39 *depends*, 1 *held*, 91 *held by its own players only*) and the one game without
  a reading is the one that joined after the check; where the main checkout's readings are on
  disk, the file equals them row for row.
- **The words**: one per reading, plain, with no internal name (clause, c, f1) in what a
  visitor reads, *held by its own players* kept apart from *held*, and the two numbers only
  in the game page's details.
- **How it combines**: with *unread* (no second line), with *provisional* (shown, as a
  reading of that measurement), with a replay onto other players (§3's asymmetry: a flag
  stays, dated; a *held* reads *not checked*), and with the tiers (a flagged game keeps its
  tier, marked) and the front page's stories (kept, the line marked).
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
import strength_check_file
from test_provisional_estimate import EARLIER, _library, _measure

from chess_variant_sandbox import corpus, measurement
from chess_variant_sandbox.web import library, stories
from chess_variant_sandbox.web.page import game as game_page

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "chess_variant_sandbox" / "web" / "static" / "page"
HELD = measurement.strength_file()
MAIN_RUNS = strength_check_file.main_runs()
INTERNAL = re.compile(r"\bclause\b|\bf1\b|\bf4\b|\bc\b|\bc ?[<>=]|R-S4|depends on the players")


def _row(game: library.Game, word: str, holding: list[str], c: float = 0.3,
         f1: float | None = None) -> dict[str, Any]:
    basis = game.measurement.basis if game.measurement else None
    assert basis is not None
    return {"name": game.name, "word": word, "holding": holding, "c": c, "c_error": 0.05,
            "f1": f1, "f1_error": None if f1 is None else 0.01, "f4": None, "f4_error": None,
            "fairy_plays": f1 is not None, "read_on": "release-1", "beside": [],
            "player": basis.agents[-1], "engine": f"{basis.engine}+{basis.agents_source}",
            "tables": basis.tables, "build": "b", "fairy_engine": "", "rule": "r",
            "script": "s", "read_at": "2026-11-13"}


def _readings(monkeypatch: pytest.MonkeyPatch, rows: dict[str, Any],
              missing: dict[str, Any] | None = None) -> None:
    found = {"weights": "v13", "games": rows, "not_checked": missing or {}}
    monkeypatch.setattr(measurement, "strength_file", lambda *_: found)


# ------------------------------------------------------------------ the file


class TestTheFile:
    def test_every_library_game_has_a_reading_or_a_reason(self) -> None:
        """A library game with no reading and no stated reason fails here."""
        rows, missing = HELD["games"], HELD["not_checked"]
        for digest in corpus.corpus_hashes():
            assert digest in rows or (missing.get(digest) or {}).get("why"), digest
        assert not set(rows) & set(missing)

    def test_the_counts_are_the_records(self) -> None:
        """§14.3: 39 / 1 / 91 of 131, less atomic's reading, withdrawn by Q1219 (§17): 38
        *depends* (22 on clause B alone); the 132nd game is not checked yet, nor is atomic."""
        words = [row["word"] for row in HELD["games"].values()]
        assert words.count("depends on the players") == 38
        assert words.count("held") == 1
        assert words.count("held, by its own players only") == 91
        holding = [tuple(row["holding"]) for row in HELD["games"].values()]
        assert holding.count(("B",)) == 22 and holding.count(("A",)) == 11
        assert holding.count(("A", "C")) == 5
        assert sorted(row["name"] for row in HELD["not_checked"].values()) == [
            "atomic", "rear guard torus chess with even promotion"]

    def test_atomics_reading_is_withdrawn_and_says_why(self) -> None:
        """Q1219: its word rested on games against Fairy-Stockfish, which plays another
        atomic where the kings stand close; the page says so in plain words."""
        made = library.strength_payload(measurement.strength_reading("6bd4d9c5769e", None))
        assert made["word"] == measurement.NOT_CHECKED and made["tag"] == "not checked yet"
        assert "different rule when the two kings stand close" in made["said"]
        assert not INTERNAL.search(made["said"]), made["said"]

    def test_every_reading_carries_its_basis(self) -> None:
        for row in HELD["games"].values():
            assert row["read_on"] == "release-1" and row["build"] == "80dee3d13bd9330d"
            assert row["read_at"] and row["engine"] and row["tables"] and row["player"]
        assert HELD["weights"] == "v13"

    @pytest.mark.skipif(not strength_check_file.READINGS.is_dir(),
                        reason="R-S4b's readings are only in the main checkout's runs/")
    def test_the_file_equals_the_readings(self) -> None:
        assert strength_check_file.main(["check", "--rows-only"]) == 0

    def test_the_replay_left_127_readings_standing_by_identity(self) -> None:
        """R-S4d (§18): after SR8r's replay onto core 7892b1922fa4cf8d, every reading whose
        game's records came out byte-identical carries the proof; the three read games that
        play differently (`replay-2026-11-15.md`; atomic's is withdrawn) carry none."""
        rows = HELD["games"]
        same = {digest for digest, row in rows.items() if row.get("identical")}
        assert len(same) == 127
        assert sorted(rows[d]["name"] for d in set(rows) - same) == [
            "queen bomb", "queens wake", "the shepherds duck"]
        for digest in same:
            proof = rows[digest]["identical"]
            assert proof["engine"].startswith("f0b28a2e4f883c15-7892b1922fa4cf8d+")
            assert len(proof["records"]) == 64 and proof["read_beside"].startswith("previous/")
            assert proof["rule"].endswith("#18")


@pytest.mark.slow  # 4 s: hashes every library game's records, twice
@pytest.mark.skipif(not strength_check_file.READINGS.is_dir(),
                    reason="R-S4b's readings are only in the main checkout's runs/")
def test_the_identity_proofs_are_the_records() -> None:
    """R-S4d: each `identical` entry is re-made from the main checkout's records."""
    assert strength_check_file.main(["check"]) == 0


# ------------------------------------------------------------------ R-S4d's rule


def _run(root: Path, digest: str, seed: int, lines: list[dict[str, Any]],
         waiting: list[dict[str, Any]] | None = None) -> None:
    run = root / f"{digest}-{seed}"
    run.mkdir(parents=True)
    (run / "games.jsonl").write_text(
        "".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines),
        encoding="utf-8")
    if waiting is not None:
        (run / "waiting.jsonl").write_text(
            "".join(json.dumps(line, separators=(",", ":")) + "\n" for line in waiting),
            encoding="utf-8")


def _game(moves: list[int], ms: float) -> dict[str, Any]:
    return {"ruleset_hash": "abc", "moves": moves, "winner": 0, "duration_ms": ms}


def _stored(engine: str, tables: str = "t") -> dict[str, Any]:
    return {"hash": "abc", "basis": {"ruleset_hash": "abc", "seeds": [0, 1000],
                                     "engine": engine, "agents_source": "a", "tables": tables,
                                     "agents": ["random", "minimax:nodes=25600"],
                                     "weights_version": "v14", "games_per_pairing": 4,
                                     "metrics": "m", "analysis": "n",
                                     "verdict_rule": "r"}}


class TestTheIdentityRule:
    """A reading stands while the records it was taken on are byte-identical to the current
    ones, the wall clock (`duration_ms`) aside (`strength-check-all-games.md` §18)."""

    def _library_runs(self, tmp_path: Path, *, moved: bool) -> Path:
        runs = tmp_path / "runs"
        kept = runs / "abc" / "previous" / "3"
        kept.mkdir(parents=True)
        (kept / "canonical.json").write_text(json.dumps(_stored("old")), encoding="utf-8")
        (runs / "abc" / "canonical.json").write_text(json.dumps(_stored("new")),
                                                     encoding="utf-8")
        for seed in (0, 1000):
            _run(kept, "abc", seed, [_game([1, 2, 3], 12.5)], [_game([4], 1.0)])
            _run(runs, "abc", seed, [_game([1, 2, 3 + (moved and seed == 1000)], 99.25)],
                 [_game([4], 7.0)])
        return runs

    def _row(self) -> dict[str, Any]:
        return {"name": "abc", "word": "held, by its own players only", "engine": "old+a",
                "tables": "t", "player": "minimax:nodes=25600"}

    def test_the_wall_clock_aside_identical_records_have_one_digest(
            self, tmp_path: Path) -> None:
        runs = self._library_runs(tmp_path, moved=False)
        old = measurement.run_directories(runs / "abc" / "previous" / "3" / "canonical.json",
                                          _stored("old"))
        new = measurement.run_directories(runs / "abc" / "canonical.json", _stored("new"))
        assert old and new and old[0].parent != new[0].parent
        assert (old[0] / "games.jsonl").read_bytes() != (new[0] / "games.jsonl").read_bytes()
        assert measurement.records_digest(old) == measurement.records_digest(new)

    def test_one_move_moved_is_another_digest(self, tmp_path: Path) -> None:
        runs = self._library_runs(tmp_path, moved=True)
        assert strength_check_file.identity(runs, "abc", self._row()) is None

    def test_a_run_without_its_records_proves_nothing(self, tmp_path: Path) -> None:
        runs = self._library_runs(tmp_path, moved=False)
        (runs / "abc-1000" / "games.jsonl").unlink()
        assert strength_check_file.identity(runs, "abc", self._row()) is None

    def test_identical_records_carry_the_proof_and_the_reading_stands(
            self, tmp_path: Path) -> None:
        runs = self._library_runs(tmp_path, moved=False)
        proof = strength_check_file.identity(runs, "abc", self._row())
        assert proof is not None
        assert proof["engine"] == "new+a" and proof["tables"] == "t"
        assert proof["read_beside"] == "previous/3"
        row = {**self._row(), "identical": proof}
        now = measurement.Basis.from_dict(_stored("new")["basis"])
        reading = measurement.strength_reading("abc", now, readings={"games": {"abc": row}})
        assert reading["word"] == measurement.HELD_OWN and reading["belongs"]
        assert reading["standsBy"] == "identity" and reading["identical"] == proof
        # Without the proof the same reading is not checked on the new stamp (§16.2).
        bare = measurement.strength_reading("abc", now,
                                            readings={"games": {"abc": self._row()}})
        assert bare["word"] == measurement.NOT_CHECKED and bare["standsBy"] is None

    def test_the_proof_holds_only_for_the_stamp_it_names(self) -> None:
        """A later replay onto yet another core is a new stamp: the proof made for the last
        one does not carry over, and the reading reads *not checked* until it is made again."""
        row = {**self._row(), "identical": {"engine": "new+a", "tables": "t", "records": "d"}}
        later = measurement.Basis.from_dict(_stored("later")["basis"])
        reading = measurement.strength_reading("abc", later, readings={"games": {"abc": row}})
        assert reading["word"] == measurement.NOT_CHECKED and not reading["belongs"]
        other_tables = measurement.Basis.from_dict(_stored("new", tables="u")["basis"])
        assert measurement.reading_stands_by(row, other_tables) is None

    def test_a_reading_taken_on_the_score_stands_as_read(self) -> None:
        row = {**self._row(), "identical": {"engine": "new+a", "tables": "t", "records": "d"}}
        old = measurement.Basis.from_dict(_stored("old")["basis"])
        assert measurement.reading_stands_by(row, old) == "reading"


# ------------------------------------------------------------------ the words


class TestTheWords:
    @pytest.mark.parametrize("word", measurement.STRENGTH_WORDS)
    def test_each_word_has_plain_words(self, word: str) -> None:
        said = library.STRENGTH_SAID[word] + " " + library.STRENGTH_TAGS[word]
        assert not INTERNAL.search(said), said
        assert "chess" not in said.lower()

    def test_held_by_its_own_players_is_not_held(self) -> None:
        own, held = (library.STRENGTH_TAGS[measurement.HELD_OWN],
                     library.STRENGTH_TAGS[measurement.HELD])
        assert own != held and "own" in own
        assert "weaker check" in library.STRENGTH_SAID[measurement.HELD_OWN]
        assert "not a guarantee" in library.STRENGTH_SAID[measurement.HELD]

    def test_the_numbers_are_details_with_what_they_are(self) -> None:
        reading = {"word": "depends", "recorded": "depends", "belongs": True,
                   "c": 0.40625, "f1": 0.0, "readAt": "2026-11-13"}
        made = library.strength_payload(reading)
        assert made["tag"] == "may change with stronger players" and made["shown"]
        assert made["details"][0] == ("Checked on 13 November 2026, on the computer players "
                                      "that measured this score.")
        assert "0.41 of 1" in made["details"][1] and "four times" in made["details"][1]
        assert "0.00 of 1" in made["details"][2] and "Fairy-Stockfish" in made["details"][2]
        assert "0.41" not in made["said"]
        assert made["tierSaid"] == library.STRENGTH_TIER_SAID

    def test_not_checked_says_why(self) -> None:
        made = library.strength_payload(measurement.strength_reading(
            "0df9ea99bd82", None))
        assert made["word"] == measurement.NOT_CHECKED and made["tag"] == "not checked yet"
        assert made["said"].endswith("joined the library after the check was made on "
                                     "13 November 2026.")
        assert made["details"] == [] and made["tierSaid"] is None


# ------------------------------------------------------------------ how it combines


class TestHowItCombines:
    def test_a_flagged_current_score_keeps_its_place_marked(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        digests = {name: _measure(tmp_path, name, index)
                   for name, index in (("chess", 0.8), ("gardner", 0.95), ("dull", 0.2))}
        games = _library(tmp_path)
        _readings(monkeypatch, {digests["gardner"]: _row(
            games[digests["gardner"]], "depends on the players", ["B"], f1=0.1)})
        games = _library(tmp_path)
        gardner = games[digests["gardner"]]
        assert gardner.status == library.CURRENT
        card = gardner.summary()
        assert card["strength"]["word"] == measurement.DEPENDS and card["strength"]["shown"]
        # Still a score: compared, banded and in its tier, the mark beside it.
        assert gardner.current_score is not None
        tier = library.tiers(games.values())[gardner.hash]
        assert tier in library.TIERS
        assert card["strength"]["tierSaid"] == library.STRENGTH_TIER_SAID
        line = game_page.score_line(gardner.score, gardner.stale)
        assert line.startswith("Score 0.950 out of 1") and "may change with stronger" in line

    def test_on_a_provisional_score_it_is_that_measurements_reading(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Today's case: every library score is v13's, provisional under v14. A new weights
        version does not move the players, so the reading still belongs to the score."""
        digest = _measure(tmp_path, "gardner", 0.8, weights_version=EARLIER)
        game = _library(tmp_path)[digest]
        _readings(monkeypatch, {digest: _row(game, "held, by its own players only", [])})
        game = _library(tmp_path)[digest]
        assert game.status == library.PROVISIONAL
        strength = game.summary()["strength"]
        assert strength["word"] == measurement.HELD_OWN and strength["belongs"]
        assert strength["details"][0].endswith("that measured this score.")
        line = game_page.score_line(game.score, game.stale)
        assert "provisional" in line and "held when its own strongest" in line

    def test_an_unread_score_shows_one_line(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from test_unread_score import _make_unread

        digest = _measure(tmp_path, "gardner", 0.8)
        game = _library(tmp_path)[digest]
        _readings(monkeypatch, {digest: _row(game, "depends on the players", ["A", "C"],
                                             c=0.5)})
        _make_unread(tmp_path, digest, blind=(True,))
        game = _library(tmp_path)[digest]
        assert game.status == library.UNREAD
        strength = game.summary()["strength"]
        assert strength["word"] == measurement.DEPENDS and not strength["shown"]
        line = game_page.score_line(game.score, game.stale)
        assert "cannot be read yet" in line and "stronger players" not in line
        said = stories.verdict(game.score, game.name)
        assert said["players"] == "" and "may change" not in said["line"]

    def test_a_replay_onto_other_players_keeps_a_flag_and_drops_a_hold(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """§3's asymmetry: a flag found on earlier players stays, dated; a *held* found on
        them is shown as not checked until it is checked again."""
        flagged = _measure(tmp_path, "gardner", 0.8)
        held = _measure(tmp_path, "dull", 0.2)
        games = _library(tmp_path)
        rows = {digest: {**_row(games[digest], word, holding),
                         "engine": "earlier-core+earlier-agents"}
                for digest, word, holding in (
                    (flagged, "depends on the players", ["A"]),
                    (held, "held, by its own players only", []))}
        _readings(monkeypatch, rows)
        games = _library(tmp_path)
        one, two = games[flagged].summary()["strength"], games[held].summary()["strength"]
        assert one["word"] == measurement.DEPENDS and not one["belongs"]
        assert "This was found on 13 November 2026" in one["said"]
        assert one["details"][0].endswith("that measured an earlier score.")
        assert two["word"] == measurement.NOT_CHECKED and two["recorded"] == "held_own"
        assert two["tag"] == "not checked yet" and "earlier score" in two["said"]

    def test_a_flagged_story_keeps_its_story_and_carries_the_mark(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        digest = _measure(tmp_path, "gardner", 0.8, weights_version=EARLIER)
        game = _library(tmp_path)[digest]
        _readings(monkeypatch, {digest: _row(game, "depends on the players", ["A"])})
        game = _library(tmp_path)[digest]
        said = stories.verdict(copy.deepcopy(game.score), game.name)
        assert said["status"] == stories.PROVISIONAL and said["verdict"] == "stays in"
        assert said["line"].endswith(" · may change with stronger players")
        assert said["players"] == "may change with stronger players"


class TestThePage:
    def test_the_scripts_draw_it_beside_the_score(self) -> None:
        core = (STATIC / "core.js").read_text(encoding="utf-8")
        detail = (STATIC / "detail.js").read_text(encoding="utf-8")
        browse = (STATIC / "browse.js").read_text(encoding="utf-8")
        assert core.count("strengthCheck: ['") == 2  # the maintainer's and a visitor's
        assert "function strengthMarkHtml(estimate, details)" in core
        assert "${strengthMarkHtml(score, true)}" in detail
        assert "strengthTag(game)" in browse and "data-tier-strength" in browse


# ------------------------------------------------------------------ the library today


@pytest.mark.slow  # U1b: 16 s alone (every library score read off the main checkout's runs)
@pytest.mark.skipif(not (MAIN_RUNS / "831c77e75650" / "canonical.json").exists(),
                    reason="reads the library's stored runs, which only the main checkout has")
def test_the_library_today_shows_the_records_counts() -> None:
    """On the main checkout's runs: every library game with a score shows a line, and the
    lines add up to §14's 39 / 1 / 91, five of the 39 shown as unread instead, and atomic's
    reading withdrawn (Q1219), so 33 *depends* and two *not checked*.

    Moved 2026-11-16 (SR8r): the library replay put every score on core `7892b1922fa4cf8d`, so
    no reading taken on `80dee3d13bd9330d` belongs any more (§16.2: a *held* reading is never
    shown across a replay; a *depends* one stays, dated). The 91 *held by its own players* and
    kyoto's *held* now read *not checked yet*: 94 with the two that were. Was
    `{"depends": 33, "unread": 5, "held": 1, "held_own": 91, "not_checked": 2}`.

    Moved again 2026-11-16 (R-S4d, the record's §18): a reading stands while the records it
    was taken on are byte-identical to the current ones. 127 are, so kyoto reads *held* and 88
    *held by its own players* again; *not checked* is the five whose records changed or that
    have no reading (atomic, queen bomb, queens wake, the shepherd's duck, rear guard torus
    chess with even promotion). Was `{"depends": 33, "unread": 5, "not_checked": 94}`."""
    games = {game.hash: game for game in library.discover(
        ROOT / "rulesets", MAIN_RUNS, ROOT / "archive")}
    shown: dict[str, int] = {}
    for digest in corpus.corpus_hashes():
        game = games[digest]
        score = game.score
        if score is None:
            continue
        strength = score["strength"]
        assert strength["tag"] and strength["said"], game.name
        key = strength["word"] if strength["shown"] else "unread"
        shown[key] = shown.get(key, 0) + 1
    assert shown == {"depends": 33, "unread": 5, "held": 1, "held_own": 88,
                     "not_checked": 5}, json.dumps(shown)
