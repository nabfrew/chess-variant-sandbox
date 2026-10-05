"""One leaderboard, the current revision's, saying how many of its rows are provisional.

EV3 of `docs/plan-evolving-score.md` (§3, §5.2). Exactly one weights version is current and
every leaderboard is its: a score under another version is placed by its number and marked,
and never compared. What is held down here, on the rulesets' own files (never chess):

- **The bands** read the current version only: a library half on the earlier version and
  half on `DEFAULT_WEIGHTS` gives bands from the current half alone, and a score that calls
  itself current under another version (built by hand, naming no status) is left out too.
- **The header**: `/api/library` carries `provisionalScores`, *n of m scores are
  provisional while they are measured again*, and nothing at all when none is; the page
  draws the sentence and nothing when it is null; the site's listing carries the same count
  over the cards it offers (a database).
- **The `leaderboard` command**: a design run's rows under the earlier version are placed
  by their number and marked, never *tied for first*, and the header counts them; the
  design loop's own summary, which reads every row as its run pinned it, is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from test_provisional_estimate import EARLIER, _library, _measure
from test_web import _in_page

from chess_variant_sandbox import cli
from chess_variant_sandbox.loop.archive import Archive, Entry, provisional_sentence
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS
from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.server import App

# Far apart, so every current pair separates. The earlier half sits above the current
# half, so a band that let it in would count it.
CURRENT_HALF = {"gardner": 0.5, "losalamos": 0.3}
EARLIER_HALF = {"dull": 0.9, "almost": 0.7}


def _half_and_half(root: Path) -> dict[str, str]:
    digests = {name: _measure(root, name, index) for name, index in CURRENT_HALF.items()}
    digests.update({name: _measure(root, name, index, weights_version=EARLIER)
                    for name, index in EARLIER_HALF.items()})
    return digests


class TestTheBands:
    def test_a_library_half_on_each_version_bands_the_current_half_only(
        self, tmp_path: Path
    ) -> None:
        digests = _half_and_half(tmp_path)
        bands = library.bands(_library(tmp_path).values())
        assert set(bands) == {digests["gardner"], digests["losalamos"]}
        assert bands[digests["gardner"]] == {"clears": 1, "clearedBy": 0, "close": 0, "of": 1}
        assert bands[digests["losalamos"]] == {"clears": 0, "clearedBy": 1, "close": 0, "of": 1}

    def test_a_current_score_under_another_version_never_shares_a_band(self) -> None:
        """A score built by hand names no status, and `current_only` reads it as current:
        the version is what keeps it off the current leaderboard."""
        def game(digest: str, version: str, total: float) -> Any:
            return SimpleNamespace(hash=digest, score={
                "weights_version": version, "total": total,
                "interval": {"ci_low": total - 0.01, "ci_high": total + 0.01}})

        games = [game("a", DEFAULT_WEIGHTS, 0.5), game("b", DEFAULT_WEIGHTS, 0.3),
                 game("c", EARLIER, 0.9)]
        bands = library.bands(games)  # type: ignore[arg-type]
        assert set(bands) == {"a", "b"}
        assert bands["a"]["of"] == 1


class TestTheHeader:
    def test_the_listing_says_how_many_scores_are_provisional(self, tmp_path: Path) -> None:
        _half_and_half(tmp_path)
        app = App(tmp_path / "rulesets", tmp_path / "runs", archive_dir=tmp_path / "archive")
        payload = app.get("/api/library")
        assert payload["provisionalScores"] == {
            "provisional": 2, "of": 4,
            "sentence": "2 of 4 scores are provisional while they are measured again.",
        }
        # Placed by their number, marked: the cards keep the provisional numbers.
        indices = {card["index"] for card in payload["games"] if card["status"] == "provisional"}
        assert indices == set(EARLIER_HALF.values())

    def test_it_says_nothing_when_none_is_provisional(self, tmp_path: Path) -> None:
        for name, index in CURRENT_HALF.items():
            _measure(tmp_path, name, index)
        app = App(tmp_path / "rulesets", tmp_path / "runs", archive_dir=tmp_path / "archive")
        assert app.get("/api/library")["provisionalScores"] is None

    def test_a_game_with_no_number_is_not_a_score(self) -> None:
        cards = [{"status": "provisional"}, {"status": "current"}, {"status": "none"}, {}]
        assert library.provisional_count(cards) == {
            "provisional": 1, "of": 2,
            "sentence": "1 of 2 scores is provisional while it is measured again.",
        }
        assert library.provisional_count([{"status": "none"}]) is None

    def test_the_page_draws_the_sentence_and_nothing_without_one(self) -> None:
        found = {"provisional": 4, "of": 73, "sentence": provisional_sentence(4, 73)}
        drawn = _in_page("", f"provisionalScoresHtml({json.dumps(found)})")
        assert "4 of 73 scores are provisional while they are measured again." in drawn
        assert _in_page("", "provisionalScoresHtml(null)") == ""


def _design_run(root: Path) -> Path:
    """A design run whose two rows above are on the earlier version."""
    out = root / "design"
    archive = Archive(out)
    rows = (("old-best", 0.90, EARLIER), ("old-next", 0.80, ""),
            ("new-best", 0.60, DEFAULT_WEIGHTS), ("new-close", 0.59, DEFAULT_WEIGHTS),
            ("new-far", 0.20, DEFAULT_WEIGHTS))
    for number, (name, index, version) in enumerate(rows):
        archive.record(Entry(hash=f"{number:012x}", name=name, generation=1, parent=None,
                             status="evaluated", index=index, index_sd=0.01,
                             index_ci_low=index - 0.02, index_ci_high=index + 0.02,
                             viable=True, seeds=[0], weights_version=version))
    return out


def _line(text: str, name: str) -> str:
    (found,) = [line for line in text.splitlines() if f" {name} [" in line]
    return found


class TestTheLeaderboardCommand:
    def test_rows_under_another_revision_are_placed_marked_and_never_compared(
        self, tmp_path: Path
    ) -> None:
        said = Archive(_design_run(tmp_path)).summary(revision=DEFAULT_WEIGHTS)
        assert "2 of 5 scores are provisional while they are measured again." in said
        assert _line(said, "old-best").startswith("  1. ")
        assert f"(provisional: scored under {EARLIER})" in _line(said, "old-best")
        assert "(provisional: no scorer revision recorded)" in _line(said, "old-next")
        for name in ("old-best", "old-next"):
            assert "tied for first" not in _line(said, name)
        # The current leader is compared with current rows only.
        assert "tied for first" not in _line(said, "new-best")
        assert "tied for first" in _line(said, "new-close")
        assert "tied for first" not in _line(said, "new-far")
        for name in ("new-best", "new-close", "new-far"):
            assert "provisional" not in _line(said, name)

    def test_the_command_prints_the_current_revisions_board(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        assert cli.main(["leaderboard", "--out", str(_design_run(tmp_path))]) == 0
        printed = capsys.readouterr().out
        assert "2 of 5 scores are provisional" in printed
        assert f"(provisional: scored under {EARLIER})" in printed

    def test_no_header_when_every_row_is_current(self, tmp_path: Path) -> None:
        archive = Archive(tmp_path / "design")
        archive.record(Entry(hash="0" * 12, name="only", generation=1, parent=None,
                             status="evaluated", index=0.5, viable=True, seeds=[0],
                             weights_version=DEFAULT_WEIGHTS))
        assert "provisional" not in archive.summary(revision=DEFAULT_WEIGHTS)

    def test_the_design_loops_summary_is_unchanged(self, tmp_path: Path) -> None:
        """The loop pins its own revision (§5.3) and reads its rows as the run did."""
        said = Archive(_design_run(tmp_path)).summary()
        assert "provisional" not in said
        assert "tied for first" not in _line(said, "old-best")


# ------------------------------------------------------------------ the site (a database)


class TestTheSite:
    def test_the_sites_listing_carries_the_count(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        from test_site_admin import SITE_URL, _running_site
        from test_trust_line import DIGEST, SPEC
        from test_web import TestTheIndexBand

        from chess_variant_sandbox.site import mail

        monkeypatch.setenv(mail.URL_VAR, SITE_URL)
        monkeypatch.delenv(mail.KEY_VAR, raising=False)
        for name in ("rulesets", "runs", "archive"):
            (tmp_path / name).mkdir()
        (tmp_path / "rulesets" / "lancers.json").write_text(json.dumps(SPEC), encoding="utf-8")
        TestTheIndexBand._score(SPEC, DIGEST, tmp_path / "runs", 0.61, 0.58, 0.64, stale=True)
        with _running_site(tmp_path) as running:
            running.game(DIGEST, SPEC)
            payload = running.browser().get("/api/library").payload
            assert payload["provisionalScores"] == {
                "provisional": 1, "of": 1,
                "sentence": "1 of 1 scores is provisional while it is measured again.",
            }
