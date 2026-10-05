"""The refresher's status drawn on `/admin` and in the local application (Q705).

`docs/plan-evolving-score.md`, EV5's checklist: *the status drawn on `/admin` and in the
local app*. `web/refreshing.py` words `refresher.read_status`; both pages draw its lines
and decide nothing. One test per state the task names (running, idle, refused, a file
that stopped being written, no file), the two routes that carry it, and the page wiring.
The site's own HTTP check (an admin is sent it, a visitor never) is in
`tests/test_site_admin.py`, since it needs a database.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from page_text import PAGE

from chess_variant_sandbox import measurement, refresher
from chess_variant_sandbox.site import pages
from chess_variant_sandbox.web import refreshing
from chess_variant_sandbox.web.server import App

GAME = {"hash": "a" * 12, "name": "alpha", "needs": "replay", "lease": "heavy"}


def _ledger(runs: Path, **fields) -> None:
    """A status file written the refresher's own way, just now."""
    refresher.Ledger(runs).update(**fields)


def _written(runs: Path, seconds_ago: float, **fields) -> None:
    """A status file last written `seconds_ago`, as a refresher that stopped leaves it."""
    then = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        seconds=seconds_ago)
    runs.mkdir(parents=True, exist_ok=True)
    (runs / refresher.STATUS_FILE).write_text(json.dumps(
        {"refusals": [], "owed_merge": [], **fields,
         "updated": then.isoformat(timespec="seconds")}), encoding="utf-8")


class TestTheStates:
    def test_no_file_says_it_is_not_running_never_an_error(self, tmp_path: Path) -> None:
        view = refreshing.view(tmp_path / "runs")
        assert view["state"] == "none" and view["running"] is False
        assert view["said"] == [
            "The refresher is not running here: it has never written a status."]
        assert view["refused"] == []

    def test_an_unreadable_file_is_no_file(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        runs.mkdir()
        (runs / refresher.STATUS_FILE).write_text("{ half", encoding="utf-8")
        assert refreshing.view(runs)["state"] == "none"

    def test_running_names_the_game_the_repair_and_what_is_owed(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        _ledger(runs, state="working", doing={**GAME, "started": refresher._now()},
                owed_counts={"rescore": 3, "replay": 1}, refused_owed=0)
        view = refreshing.view(runs)
        assert view["state"] == "working" and view["running"] is True
        assert view["doing"] == {"hash": GAME["hash"], "name": "alpha",
                                 "needs": "playing again"}
        assert view["said"][0] == "The refresher is working on alpha: playing again, since just now."
        assert view["said"][1] == ("4 games out of date at its last look: "
                                   "3 scoring again, 1 playing again.")
        assert view["owed"] == 4

    def test_idle_says_nothing_is_owed(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        _ledger(runs, state="idle", doing=None, owed_counts={}, refused_owed=0,
                last_done={**GAME, "needs": "rescore", "index": 0.61234})
        view = refreshing.view(runs)
        assert view["state"] == "idle" and view["running"] is True
        assert view["doing"] is None
        assert view["said"][0].startswith("The refresher is running and idle")
        assert "No game was out of date at its last look." in view["said"]
        assert "Last repaired: alpha, scoring again, now 0.6123." in view["said"]

    def test_a_refusal_says_the_game_and_why(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        ledger = refresher.Ledger(runs)
        item = refresher.Item(hash=GAME["hash"], name="alpha", needs="replay",
                              reason="", tier=3, measured="", stored="d1")
        ledger.refuse(item, refresher.Outcome(
            hash=GAME["hash"], needs="replay", refusal=refresher.GATE_NEWLY_FAILED, reason="draw rate 0.9 > 0.8"))
        ledger.update(state="idle", owed_counts={}, refused_owed=1)
        view = refreshing.view(runs)
        [refused] = view["refused"]
        assert refused["name"] == "alpha" and refused["hash"] == GAME["hash"]
        assert refused["needs"] == "playing again"
        assert refused["why"] == "it passed its checks before and fails them now"
        assert refused["reason"] == "draw rate 0.9 > 0.8"
        assert view["lastRefusal"]["name"] == "alpha"
        assert ("1 more is out of date and refused, so it does not try them again."
                in view["said"])

    def test_every_refusal_kind_has_plain_words(self) -> None:
        kinds = {getattr(refresher, name) for name in (
            "STALE_BUILD", "FIT_NOT_REPRODUCED", "GATE_NEWLY_FAILED", "COMMIT_MOVED",
            "CHANGED_COMMITTED_FILES", "ERROR")}
        assert kinds == set(refreshing.REFUSALS)

    def test_a_file_that_stopped_being_written_is_not_running(self, tmp_path: Path) -> None:
        """Whatever its last state says: `age_s` decides (`read_status`'s reason for it)."""
        runs = tmp_path / "runs"
        _written(runs, 2 * 3600, state="idle")
        view = refreshing.view(runs)
        assert view["state"] == "quiet" and view["running"] is False
        assert view["said"][0].startswith(
            "The refresher is not running: it last wrote 2 hours ago")

    def test_a_long_game_is_still_running(self, tmp_path: Path) -> None:
        """A replay writes once at its start; an hour into one is not a stopped refresher."""
        runs = tmp_path / "runs"
        _written(runs, 3600, state="working", doing=GAME)
        assert refreshing.view(runs)["running"] is True
        _written(runs, 5 * 3600, state="working", doing=GAME)
        assert refreshing.view(runs)["state"] == "quiet"

    def test_stopped_and_halted(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        _ledger(runs, state="stopped", doing=None)
        view = refreshing.view(runs)
        assert view["running"] is False
        assert view["said"][0] == "The refresher is not running: it stopped just now."
        _ledger(runs, state="halted", waiting_for="main to move",
                last_refusal={"kind": refresher.STALE_BUILD, "reason": "rust/ is newer"})
        view = refreshing.view(runs)
        assert view["running"] is True and "rust/ is newer" in view["said"][0]

    def test_ago(self) -> None:
        assert [refreshing.ago(s) for s in (None, 5, 60, 150, 3600, 7300, 86400 * 3)] == [
            "at a time it did not record", "just now", "1 minute ago", "2 minutes ago",
            "1 hour ago", "2 hours ago", "3 days ago"]


class TestWhereItIsDrawn:
    def test_the_local_application_sends_it_with_the_scores(self, tmp_path: Path) -> None:
        for name in ("rulesets", "runs", "archive"):
            (tmp_path / name).mkdir()
        app = App(tmp_path / "rulesets", tmp_path / "runs", archive_dir=tmp_path / "archive")
        assert app.get("/api/scoring")["refresher"]["state"] == "none"
        _ledger(tmp_path / "runs", state="idle", owed_counts={})
        assert app.get("/api/scoring")["refresher"]["state"] == "idle"

    def test_the_local_page_draws_it_in_the_scores_panel(self) -> None:
        assert "${refresherHtml(s.refresher)}" in PAGE
        assert "function refresherHtml(r)" in PAGE

    def test_admin_draws_it_from_the_payload(self) -> None:
        page = pages.admin("me")
        for mark in ('id="refresher-said"', 'id="refresher-refused"',
                     'id="refresher-refused-empty"'):
            assert mark in page
        script = (measurement.ROOT / "chess_variant_sandbox/site/static/admin.js").read_text(
            encoding="utf-8")
        assert "refresherStatus(payload.refresher);" in script
        assert "innerHTML" not in script

    def test_nothing_under_metrics_reads_it(self) -> None:
        metrics = measurement.ROOT / "chess_variant_sandbox" / "metrics"
        for path in metrics.rglob("*.py"):
            assert "refreshing" not in path.read_text(encoding="utf-8"), path
