"""A score as an evolving estimate: what a reader holds while a score is measured again.

EV2 of `docs/archive/plan-evolving-score.md` (§2.1, §3, §5.1), the user's answer of 2026-10-03 (§9
Q1, A). A stale measurement used to be no number at all; it is now the game's last number,
marked *provisional* with the revision it came from and its date. What is held down here,
on the rulesets' own files (never chess):

- **The status.** `library.Game.status` is `current`, `provisional` or `none`, and every
  payload (`summary`, `detail`, `/api/library`, `/api/game`) carries it with `revision` and
  `measuredAt`. A weights bump makes every game provisional with its old number.
- **What a provisional number never reaches**: a band (`bands`, `_comparable`), a
  neighbour's gap, the map's ramp, a draft's parent, the ladder, a trust level, and any
  written file -- the `canonical.json` it was read from stays stale and byte for byte.
- **A design score follows the same rule**: a number under an earlier weights version
  than the current one is provisional.
- **The page** draws the number with the mark and the sentence, never bare, and draws no
  trust chip, trust line or band beside it; on the site, in a visitor's words (the words
  guard reads the new gloss, `provisionalScore`, with every other).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import site_words
from test_web import _in_page, _one_run

from chess_variant_sandbox import corpus
from chess_variant_sandbox.loop.archive import Archive, Entry
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS
from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.server import App

RULESETS = library.RULESET_DIR
EARLIER = "v11"
assert EARLIER != DEFAULT_WEIGHTS


def _measure(root: Path, name: str, index: float, sd: float = 0.01,
             **basis_overrides: Any) -> str:
    """One library game's measurement on disk, its basis current unless overridden."""
    spec = load_spec(RULESETS / f"{name}.json")
    digest = spec_hash(spec)
    rulesets = root / "rulesets"
    rulesets.mkdir(exist_ok=True)
    (rulesets / f"{name}.json").write_text(json.dumps(spec), encoding="utf-8")
    out = corpus.measurement_dir(digest, root / "runs")
    out.mkdir(parents=True, exist_ok=True)
    basis = {**corpus.current_basis(digest, spec=spec).as_dict(), **basis_overrides}
    weights = str(basis.get("weights_version") or DEFAULT_WEIGHTS)
    (out / "canonical.json").write_text(json.dumps({
        "hash": digest, "name": name, "index": index, "sd": sd,
        "viable": True, "violations": [], "weights_version": weights,
        "agents": list(corpus.CANONICAL_AGENTS), "games": 256,
        "families": {}, "basis": basis,
        "seeds": [0], "totals_used": 1, "runs": _one_run(index, sd),
        "standings": [{"name": "random", "rating": -700.0},
                      {"name": "minimax:nodes=6400", "rating": 620.0}],
    }), encoding="utf-8")
    (out / f"index-{weights}.json").write_text(json.dumps({
        "total": index, "scores": {}, "families": {},
        "interval": {"resamples": 200, "mean": index, "sd": sd,
                     "ci_low": index - 2 * sd, "ci_high": index + 2 * sd,
                     "deterministic_pairings": 0.0},
    }), encoding="utf-8")
    return digest


def _library(root: Path) -> dict[str, library.Game]:
    return {game.hash: game for game in library.discover(root / "rulesets", root / "runs")}


# Far apart, so every current pair separates and a band always has something to count.
SCORES = {"gardner": 0.8, "losalamos": 0.5, "dull": 0.2}


class TestTheStatus:
    def test_a_current_measurement_is_current(self, tmp_path) -> None:
        digest = _measure(tmp_path, "gardner", 0.8)
        game = _library(tmp_path)[digest]
        assert game.status == library.CURRENT
        card = game.summary()
        assert card["status"] == "current" and card["index"] == 0.8
        assert card["revision"] == DEFAULT_WEIGHTS
        assert game.current_score is game.score

    def test_a_weights_bump_makes_every_game_provisional_with_its_old_number(
        self, tmp_path
    ) -> None:
        """§2.2 step 2: the revision lands and every estimate it touches is provisional at
        once, with no compute, keeping its number, its revision and its date."""
        digests = {name: _measure(tmp_path, name, index, weights_version=EARLIER)
                   for name, index in SCORES.items()}
        games = _library(tmp_path)
        for name, digest in digests.items():
            game = games[digest]
            card = game.summary()
            assert game.status == library.PROVISIONAL, name
            assert card["status"] == "provisional"
            assert card["index"] == SCORES[name]
            assert card["revision"] == EARLIER
            assert isinstance(card["measuredAt"], str) and len(card["measuredAt"]) == 10
            assert card["staleness"]["current"] is False
            assert card["staleness"]["needs"] == "rescore"
            assert game.score is not None and game.score["status"] == "provisional"
            assert game.current_score is None

    def test_never_measured_is_none(self, tmp_path) -> None:
        spec = load_spec(RULESETS / "dull.json")
        (tmp_path / "rulesets").mkdir()
        (tmp_path / "rulesets" / "dull.json").write_text(json.dumps(spec), encoding="utf-8")
        card = _library(tmp_path)[spec_hash(spec)].summary()
        assert card["status"] == "none" and card["index"] is None
        assert card["revision"] is None and card["measuredAt"] is None

    def test_a_file_that_cannot_state_its_basis_is_none_not_provisional(
        self, tmp_path
    ) -> None:
        """The basis is always stated: a file with no basis shows no number at all."""
        digest = _measure(tmp_path, "gardner", 0.8)
        path = corpus.measurement_dir(digest, tmp_path / "runs") / "canonical.json"
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored["basis"] = None
        path.write_text(json.dumps(stored), encoding="utf-8")
        game = _library(tmp_path)[digest]
        assert game.stale and game.status == library.NONE
        assert game.score is None and game.summary()["index"] is None
        assert game.summary()["staleness"]["needs"] == "replay"


class TestAProvisionalNumberIsNeverCompared:
    def _mixed(self, root: Path) -> dict[str, str]:
        """Two current games and one provisional one, the provisional far above both."""
        return {
            "gardner": _measure(root, "gardner", 0.8),
            "losalamos": _measure(root, "losalamos", 0.5),
            "dull": _measure(root, "dull", 0.95, weights_version=EARLIER),
        }

    def test_it_reaches_no_band_and_no_one_elses(self, tmp_path) -> None:
        digests = self._mixed(tmp_path)
        games = list(_library(tmp_path).values())
        bands = library.bands(games)
        assert digests["dull"] not in bands
        assert bands[digests["gardner"]] == {"clears": 1, "clearedBy": 0, "close": 0, "of": 1}
        assert {entry.game for entry in library._comparable(games)} == {
            digests["gardner"], digests["losalamos"]}

    def test_it_has_no_gap_to_a_neighbour(self, tmp_path) -> None:
        digests = self._mixed(tmp_path)
        games = _library(tmp_path)
        assert library._gap(games[digests["dull"]], games[digests["gardner"]]) is None
        assert library._gap(games[digests["gardner"]], games[digests["dull"]]) is None
        assert library._gap(games[digests["gardner"]], games[digests["losalamos"]]) is not None

    def test_a_score_naming_no_status_is_read_as_it_always_was(self) -> None:
        assert library.current_only({"index": 0.5}) == {"index": 0.5}
        assert library.current_only({"index": 0.5, "status": "provisional"}) is None
        assert library.current_only(None) is None

    def test_it_has_no_ladder_and_no_trust(self, tmp_path) -> None:
        digests = self._mixed(tmp_path)
        game = _library(tmp_path)[digests["dull"]]
        assert game.ratings is None
        assert library.trust_card(game) is None and library.trust_line(game) is None
        assert game.summary()["trust"] is None

    def test_trust_is_withheld_on_the_status_whatever_the_reading(self) -> None:
        """Belt and braces: even a reading that has a level is not drawn beside a
        provisional estimate, since it was read against a basis the estimate lacks."""
        from test_trust_line import ORDERED, _trust

        found = _trust(ORDERED)
        current = SimpleNamespace(trust=found, status="current", history=[], score={},
                                  spec={}, premise=None)
        provisional = SimpleNamespace(**{**vars(current), "status": "provisional"})
        assert library.trust_card(current) is not None
        assert library.trust_line(current) is not None
        assert library.trust_card(provisional) is None
        assert library.trust_line(provisional) is None

    def test_reading_it_writes_nothing(self, tmp_path) -> None:
        """Never written into a current file: the stale file stays stale, byte for byte."""
        digests = self._mixed(tmp_path)
        path = corpus.measurement_dir(digests["dull"], tmp_path / "runs") / "canonical.json"
        before = path.read_bytes()
        game = _library(tmp_path)[digests["dull"]]
        game.summary()
        library.detail(game, _library(tmp_path).values())
        assert path.read_bytes() == before
        spec = load_spec(RULESETS / "dull.json")
        again = corpus.read_measurement(digests["dull"], tmp_path / "runs",
                                        DEFAULT_WEIGHTS, spec,
                                        corpus.expected_seeds("committed"))
        assert again is not None and not again.current

    def test_the_map_and_the_designer_read_current_only(self, tmp_path) -> None:
        from chess_variant_sandbox.web import candidacy, graphing

        digests = self._mixed(tmp_path)
        game = _library(tmp_path)[digests["dull"]]
        assert graphing._card([game]) is None
        assert graphing._measurements([game]) == {}
        assert candidacy.measured(game)["state"] == "stale"
        assert "index" not in candidacy.measured(game)


class TestTheDesignScore:
    def test_a_design_number_under_an_earlier_version_is_provisional(self, tmp_path) -> None:
        specs = {name: load_spec(RULESETS / f"{name}.json") for name in ("gardner", "dull")}
        archive = Archive(tmp_path / "runs" / "design")
        for name, version in (("gardner", DEFAULT_WEIGHTS), ("dull", EARLIER)):
            archive.record(Entry(hash=spec_hash(specs[name]), name=name, generation=1,
                                 parent=None, status="evaluated", index=0.6, index_sd=0.02,
                                 viable=True, seeds=[0], weights_version=version),
                           spec=specs[name])
        games = _library(tmp_path)
        now = games[spec_hash(specs["gardner"])].summary()
        then = games[spec_hash(specs["dull"])].summary()
        assert now["designStatus"] == "current" and now["designRevision"] == DEFAULT_WEIGHTS
        assert then["designStatus"] == "provisional" and then["designRevision"] == EARLIER
        assert then["designIndex"] == 0.6 and then["index"] is None
        pill = _in_page("", f"scoreTag({json.dumps(then)})")
        assert "0.600 rough" in pill and "provisional" in pill
        assert "provisional" not in _in_page("", f"scoreTag({json.dumps(now)})")


class TestThePayloads:
    def test_the_library_and_the_game_carry_the_status(self, tmp_path) -> None:
        digest = _measure(tmp_path, "dull", 0.4, weights_version=EARLIER)
        app = App(tmp_path / "rulesets", tmp_path / "runs", archive_dir=tmp_path / "archive")
        (card,) = [g for g in app.get("/api/library")["games"] if g["hash"] == digest]
        assert card["status"] == "provisional" and card["index"] == 0.4
        assert card["revision"] == EARLIER and card["measuredAt"]
        page = app.get(f"/api/game/{digest}")
        assert page["status"] == "provisional"
        assert page["score"]["status"] == "provisional"
        assert page["score"]["revision"] == EARLIER and page["score"]["measured_at"]
        assert page["trust"] is None and page.get("band") is None
        assert page["ratings"] is None


def _provisional_card() -> dict[str, Any]:
    return {"hash": "x", "name": "x", "index": 0.612, "status": "provisional",
            "revision": EARLIER, "measuredAt": "2026-10-03", "seeds": [0, 1000, 2000, 3000],
            "weights": EARLIER, "viable": True,
            "staleness": {"current": False, "needs": "rescore",
                          "summary": "stale (re-score): weights changed"}}


class TestThePage:
    def test_the_card_draws_the_number_with_its_mark_never_bare(self) -> None:
        card = _provisional_card()
        for setup in ("", "ACCOUNTS = true;"):
            pill = _in_page(setup, f"scoreTag({json.dumps(card)})")
            assert "0.612" in pill and "provisional" in pill
            assert "4 runs" in pill
            assert "3 October 2026" in pill
            assert 'data-gloss="provisionalScore"' in pill
        site = _in_page("ACCOUNTS = true;", f"scoreTag({json.dumps(card)})")
        assert "scoring rules version 11" in site
        assert site_words.hits(site_words.visible(site)) == []

    def test_a_stale_card_with_no_number_still_reads_out_of_date(self) -> None:
        card = {**_provisional_card(), "index": None, "status": "none"}
        assert ">out of date</span>" in _in_page("", f"scoreTag({json.dumps(card)})")

    def test_no_trust_chip_or_line_beside_a_provisional_number(self) -> None:
        trust = {"level": "early estimate", "said": "x", "more": []}
        assert _in_page("", f"trustTag({json.dumps(trust)}, 'provisional')") == ""
        assert _in_page("", f"trustHtml({json.dumps(trust)}, 'provisional')") == ""
        assert "early estimate" in _in_page("", f"trustTag({json.dumps(trust)}, 'current')")
        assert "early estimate" in _in_page("", f"trustHtml({json.dumps(trust)})")

    def test_the_score_panel_marks_it_and_draws_no_band(self) -> None:
        score = {"total": 0.612, "status": "provisional", "revision": EARLIER,
                 "measured_at": "2026-10-03", "weights_version": EARLIER,
                 "families": {}, "scores": {}, "contributions": {}, "seeds": [0]}
        band = {"clears": 3, "clearedBy": 0, "close": 0, "of": 3}
        trust = {"level": "early estimate", "said": "x", "more": []}
        panel = _in_page("", f"scoreHtml({json.dumps(score)}, {{}}, {json.dumps(band)},"
                             f" {json.dumps(trust)})")
        assert "0.612" in panel and "Provisional: worked out before" in panel
        assert "clearly above" not in panel and "How far to trust it" not in panel
        current = {**score, "status": "current"}
        drawn = _in_page("", f"scoreHtml({json.dumps(current)}, {{}}, {json.dumps(band)},"
                             f" {json.dumps(trust)})")
        assert "Provisional" not in drawn and "clearly above" in drawn

    def test_the_site_glossary_says_provisional_in_a_visitors_words(self) -> None:
        site = _in_page("ACCOUNTS = true;", "glossEntry('provisionalScore')")
        local = _in_page("", "glossEntry('provisionalScore')")
        assert site[0] == local[0] == "Provisional"
        assert site[1] != local[1]
        assert site_words.hits(site[1]) == []
        # The browser-measured number keeps its own key and its own words (Q312).
        assert _in_page("", "GLOSS.provisional[0]") == "Measured on its designer's computer"


# ------------------------------------------------------------------ the site (a database)


class TestTheSite:
    """A visitor on the site: the number marked, no trust line, and the store's ranking
    column still holds current numbers only (`site/load.py`)."""

    def test_a_visitor_sees_the_number_marked_and_no_trust(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        from test_premise import _synthetic
        from test_site_admin import SITE_URL, _running_site
        from test_trust_line import DIGEST, SPEC
        from test_web import TestTheIndexBand

        from chess_variant_sandbox import premise
        from chess_variant_sandbox.site import load, mail

        monkeypatch.setenv(mail.URL_VAR, SITE_URL)
        monkeypatch.delenv(mail.KEY_VAR, raising=False)
        for name in ("rulesets", "runs", "archive"):
            (tmp_path / name).mkdir()
        (tmp_path / "rulesets" / "lancers.json").write_text(json.dumps(SPEC), encoding="utf-8")
        TestTheIndexBand._score(SPEC, DIGEST, tmp_path / "runs", 0.61, 0.58, 0.64, stale=True)
        check = _synthetic(premise.PASSES)
        check["fairy"] = {"played": True, "score": 0.40, "error": 0.03, "games": 32,
                          "floor": 0.30}
        premise.write(check, tmp_path / "runs")
        with _running_site(tmp_path) as running:
            running.game(DIGEST, SPEC)
            browser = running.browser()
            [card] = [game for game in browser.get("/api/library").payload["games"]
                      if game["hash"] == DIGEST]
            assert card["status"] == "provisional" and card["index"] == 0.61
            assert card["revision"] == "v0" and card["measuredAt"]
            assert card["trust"] is None and card.get("band") is None
            page = browser.get(f"/api/game/{DIGEST}").payload
            assert page["status"] == "provisional" and page["trust"] is None
            load.load_measurements(running.store, {DIGEST: (SPEC, load.CORPUS)},
                                   tmp_path / "runs")
            with running.store.cursor() as cursor:
                cursor.execute("select state, headline, ci_low from measurements"
                               " where hash = %s", (DIGEST,))
                state, headline, ci_low = cursor.fetchone()
            assert state == "rescore" and headline is None and ci_low is None
