"""A score its players could not read is stamped *unread* and shown so (R-S8).

The rule is R-S4a's clause C as registered (`docs/reference/strength-check-all-games.md`
§2.3): "in the score's own file, at least half of its runs carry a non-empty `unreadable`,
or `verdict.measured` is false". What is held down here, on fixture scores and the
rulesets' own files:

- **The rule** (`measurement.unread_reasons`) on a readable score, one with half its runs
  unreadable, one with one of four, a single-run score each way, a round game's shape (no
  verdict reached) and a spy game's shape; and its reasons are words a second clause can
  join.
- **The status** (`library.Game.status`): an up-to-date unread score reads `unread`, never
  `current`; an out-of-date one reads `unread` too, with its staleness beside it and words
  that say both, so it is never more current than either alone; after a replay it stays
  unread while its runs still cannot be read, and reads `current` once they can.
- **What it never reaches**: a comparison (`current_only`), a band, a tier against chess
  (its own group, with plain words), a trust level, the site's current headline.
- **Where it is said**: the card, the game page's sentence, the landing's story (kept,
  marked, with no pair of bars), and the page's scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_provisional_estimate import EARLIER, _library, _measure

from chess_variant_sandbox import measurement
from chess_variant_sandbox.web import landing, library, stories
from chess_variant_sandbox.web.page import game as game_page

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "chess_variant_sandbox" / "web" / "static" / "page"
BLIND = ["lead_conversion", "skill_gradient"]


def _run(seed: int, unreadable: list[str] | None = None, measured: bool = True,
         index: float | None = 0.5) -> dict[str, Any]:
    return {"seed": seed, "index": index if not unreadable else None, "sd": 0.01,
            "viable": not unreadable, "violations": [], "unreadable": list(unreadable or []),
            "measured": measured}


def _stored(*unreadable: bool, measured: bool = True, **extra: Any) -> dict[str, Any]:
    seeds = [0, 1000, 2000, 3000][: len(unreadable)]
    return {"runs": [_run(seed, BLIND if blind else None) for seed, blind in
                     zip(seeds, unreadable, strict=True)],
            "verdict": {"rule": "vote-tie-refuses+pooled-seat+blind-abstains",
                        "measured": measured}, **extra}


class TestTheRule:
    def test_a_readable_score_reads(self) -> None:
        assert measurement.unread_reasons(_stored(False, False, False, False)) == ()

    def test_half_its_runs_unreadable_is_unread(self) -> None:
        """*At least half*: two of four is enough, as registered."""
        assert measurement.unread_reasons(_stored(True, False, True, False)) == (
            measurement.ROSTER,)
        assert measurement.unread_reasons(_stored(True, True, True, True)) == (
            measurement.ROSTER,)

    def test_one_run_of_four_unreadable_still_reads(self) -> None:
        assert measurement.unread_reasons(_stored(False, True, False, False)) == ()

    def test_a_single_run_reads_by_its_own_run(self) -> None:
        assert measurement.unread_reasons(_stored(True)) == (measurement.ROSTER,)
        assert measurement.unread_reasons(_stored(False)) == ()
        # The single-run form, with no `runs` list, is its own one run.
        assert measurement.unread_reasons({"index": None, "unreadable": BLIND}) == (
            measurement.ROSTER,)
        assert measurement.unread_reasons({"index": 0.5, "unreadable": []}) == ()

    def test_a_round_game_whose_runs_reached_no_verdict(self) -> None:
        """The watering hole's shape on 2026-11-14: three of four runs blind, and the vote
        reached no verdict. Either half of clause C names it alone."""
        stored = _stored(True, True, False, True, measured=False, agents=[
            "random", "greedy", "matrix:nodes=1600", "matrix:nodes=102400"])
        assert measurement.unread_reasons(stored) == (measurement.ROSTER,)
        no_verdict = _stored(False, False, False, False, measured=False)
        assert measurement.unread_reasons(no_verdict) == (measurement.ROSTER,)

    def test_a_spy_game_is_read_by_the_same_rule(self) -> None:
        """A spy game's score carries `spy_readings` beside its runs; the rule reads only
        the runs and the verdict, so it is named exactly as any other game is."""
        spy = {"spy_readings": {"rounds": 12}}
        assert measurement.unread_reasons(_stored(True, True, False, False, **spy)) == (
            measurement.ROSTER,)
        assert measurement.unread_reasons(_stored(False, False, False, True, **spy)) == ()

    def test_anything_that_is_not_a_score_reads(self) -> None:
        for nothing in (None, [], "x", {}, {"runs": []}, {"runs": "x"}):
            assert measurement.unread_reasons(nothing) == ()

    def test_the_reasons_are_words_a_second_clause_can_join(self) -> None:
        assert measurement.UNREAD_REASONS == (measurement.ROSTER,)
        assert library.unread_sentence([]) is None
        assert library.unread_sentence([measurement.ROSTER]) == library.UNREAD_SAID
        assert library.unread_sentence([measurement.ROSTER, "another"]) == library.UNREAD_SAID


def _make_unread(root: Path, digest: str, *, blind: tuple[bool, ...] = (True, True, False,
                                                                         True)) -> None:
    """Rewrite a fixture measurement's runs so that clause C holds (or not), keeping its
    basis: the runs are the only thing that changes."""
    path = root / "runs" / digest / "canonical.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    seeds = [0, 1000, 2000, 3000][: len(blind)]
    stored["runs"] = [_run(seed, BLIND if b else None) for seed, b in zip(seeds, blind,
                                                                           strict=True)]
    stored["seeds"] = seeds if len(blind) > 1 else stored["seeds"]
    path.write_text(json.dumps(stored), encoding="utf-8")


class TestTheStatus:
    def test_an_up_to_date_unread_score_is_never_current(self, tmp_path: Path) -> None:
        digest = _measure(tmp_path, "gardner", 0.8)
        _make_unread(tmp_path, digest, blind=(True,))
        game = _library(tmp_path)[digest]
        assert game.measurement is not None and game.measurement.current
        assert game.status == library.UNREAD
        assert game.current_score is None
        assert library.current_only(game.score) is None
        assert library.trust_card(game) is None
        card = game.summary()
        assert card["status"] == "unread" and card["unread"] == [measurement.ROSTER]
        assert card["unreadSaid"] == library.UNREAD_SAID
        # The number, where it has one, is kept, as a provisional one's is.
        assert card["index"] == 0.8

    def test_unread_and_out_of_date_says_both(self, tmp_path: Path) -> None:
        """Both states at once (today: every library score is v13's under a v14 basis). It
        reads `unread`, never `provisional` alone and never `current`, and its words say
        it is out of date too."""
        digest = _measure(tmp_path, "gardner", 0.8, weights_version=EARLIER)
        _make_unread(tmp_path, digest, blind=(True,))
        game = _library(tmp_path)[digest]
        assert game.measurement is not None and not game.measurement.current
        assert game.status == library.UNREAD
        card = game.summary()
        assert card["staleness"]["current"] is False
        assert card["unreadSaid"] == library.UNREAD_SAID + library.UNREAD_STALE_SAID
        assert library.tier_of(game.score, None) == library.UNREAD_GROUP

    def test_a_readable_stale_score_is_still_provisional(self, tmp_path: Path) -> None:
        digest = _measure(tmp_path, "gardner", 0.8, weights_version=EARLIER)
        game = _library(tmp_path)[digest]
        assert game.status == library.PROVISIONAL
        assert game.summary()["unread"] == [] and game.summary()["unreadSaid"] is None

    def test_after_a_replay_it_stays_unread_until_its_runs_read(self, tmp_path: Path) -> None:
        """A replay is a new file under the current basis. Its runs decide: still blind,
        still unread; readable, current."""
        digest = _measure(tmp_path, "gardner", 0.8, weights_version=EARLIER)
        _make_unread(tmp_path, digest, blind=(True,))
        assert _library(tmp_path)[digest].status == library.UNREAD
        replayed = _measure(tmp_path, "gardner", 0.8)  # the replay: a current stamp
        assert replayed == digest
        _make_unread(tmp_path, digest, blind=(True,))
        assert _library(tmp_path)[digest].status == library.UNREAD
        _make_unread(tmp_path, digest, blind=(False,))
        assert _library(tmp_path)[digest].status == library.CURRENT


class TestWhatItNeverReaches:
    def test_no_band_and_no_tier_against_chess(self, tmp_path: Path) -> None:
        digests = {name: _measure(tmp_path, name, index)
                   for name, index in {"gardner": 0.8, "losalamos": 0.5}.items()}
        _make_unread(tmp_path, digests["gardner"])
        games = list(_library(tmp_path).values())
        assert digests["gardner"] not in library.bands(games)
        assert library.tiers(games)[digests["gardner"]] == library.UNREAD_GROUP
        assert all(m.game != digests["gardner"] for m in library._comparable(games))

    def test_the_group_is_its_own_with_plain_words(self) -> None:
        legend = library.tier_legend()
        assert library.UNREAD_GROUP in legend["order"]
        assert library.UNREAD_GROUP not in legend["tiers"]
        words = legend["groups"][library.UNREAD_GROUP]
        prose = json.dumps(words) + library.UNREAD_SAID + library.UNREAD_STALE_SAID
        assert "told apart" in prose
        for internal in ("clause", "unreadable", "roster", "R-S", "verdict", "v13", "v14"):
            assert internal not in prose

    def test_the_site_shows_it_marked_and_keeps_no_current_headline(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """On the site (a database): the card and the page read `unread`, and the store's
        row holds state `unread` with no headline and no range, so no badge or round reads
        it as a current number."""
        from test_site_admin import SITE_URL, _running_site
        from test_trust_line import DIGEST, SPEC
        from test_web import TestTheIndexBand

        from chess_variant_sandbox.site import load, mail

        monkeypatch.setenv(mail.URL_VAR, SITE_URL)
        monkeypatch.delenv(mail.KEY_VAR, raising=False)
        for name in ("rulesets", "runs", "archive"):
            (tmp_path / name).mkdir()
        (tmp_path / "rulesets" / "lancers.json").write_text(json.dumps(SPEC), encoding="utf-8")
        TestTheIndexBand._score(SPEC, DIGEST, tmp_path / "runs", 0.61, 0.58, 0.64)
        _make_unread(tmp_path, DIGEST, blind=(True,))
        with _running_site(tmp_path) as running:
            running.game(DIGEST, SPEC)
            browser = running.browser()
            [card] = [game for game in browser.get("/api/library").payload["games"]
                      if game["hash"] == DIGEST]
            assert card["status"] == "unread" and card["trust"] is None
            assert card["tier"] == library.UNREAD_GROUP
            page = browser.get(f"/api/game/{DIGEST}").payload
            assert page["status"] == "unread" and page["trust"] is None
            load.load_measurements(running.store, {DIGEST: (SPEC, load.CORPUS)},
                                   tmp_path / "runs")
            with running.store.cursor() as cursor:
                cursor.execute("select state, headline, ci_low from measurements"
                               " where hash = %s", (DIGEST,))
                state, headline, ci_low = cursor.fetchone()
            assert state == load.UNREAD and headline is None and ci_low is None


class TestWhereItIsSaid:
    def test_the_game_page_sentence(self) -> None:
        score = {"status": "unread", "index": None, "unreadSaid": library.UNREAD_SAID}
        assert game_page.score_line(score, stale=False) == library.UNREAD_SAID
        numbered = {**score, "index": 0.355}
        said = game_page.score_line(numbered, stale=False)
        assert said.startswith("Score 0.355 out of 1, which cannot be read yet.")
        assert "provisional" not in said

    def test_a_story_whose_game_is_unread_stays_marked_with_no_bars(self) -> None:
        side = {"name": "sweeper", "hash": "h", "status": stories.UNREAD, "line": "x",
                "words": [], "qualities": {q: 0.5 for q in stories.QUALITIES}}
        fixed = {**side, "name": "kept", "status": stories.CURRENT}
        drawn = landing._verdict(side, good=False)
        assert "cannot be read yet" in drawn and "ruled out" in drawn
        assert landing._bars(side, fixed) == (
            f'<p class="small muted pairWait">{landing._e(landing.BARS_UNREAD_SAID)}</p>')

    def test_a_storys_line_leads_with_the_mark_and_says_provisional_too(self) -> None:
        score = {"index": None, "seeds": [0, 1000, 2000, 3000], "viable": False,
                 "status": "unread", "revision": "v13", "measured_at": "2026-11-05",
                 "staleness": {"current": False}, "verdict": {"split": ""},
                 "families": {}}
        line = stories.verdict(score, "sweeper")["line"]
        assert line.startswith(stories.UNREAD_LINE)
        assert stories.PROVISIONAL_LINE in line
        current = stories.verdict({**score, "staleness": {"current": True}}, "s")["line"]
        assert stories.PROVISIONAL_LINE not in current

    def test_the_page_draws_the_mark_and_withholds_band_and_trust(self) -> None:
        core = (STATIC / "core.js").read_text(encoding="utf-8")
        detail = (STATIC / "detail.js").read_text(encoding="utf-8")
        upkeep = (STATIC / "upkeep.js").read_text(encoding="utf-8")
        assert "function isUnread(estimate)" in core
        assert "status === 'unread'" in core and "unreadScore: ['" in core
        assert core.count("unreadScore: ['") == 2  # the maintainer's and a visitor's
        assert "isNotCurrent(score) ? '' : bandHtml(band)" in detail
        assert "if (isUnread(game))" in upkeep


@pytest.mark.skipif(not (ROOT / "runs" / "417c19c2f6bd" / "canonical.json").exists(),
                    reason="reads the library's stored runs, which only the main checkout has")
def test_the_library_games_clause_c_names() -> None:
    """On the main checkout's stored runs: the watering hole reads unread (2026-11-14:
    three of four runs blind and no verdict reached)."""
    stored = json.loads((ROOT / "runs" / "417c19c2f6bd" / "canonical.json").read_text(
        encoding="utf-8"))
    assert measurement.unread_reasons(stored) == (measurement.ROSTER,)
