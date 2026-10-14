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


class TestTheLadders:
    """Q860: the ladder work Q822 put in the status file, worded beside the score work.

    One fixture file per case the task names: no ladder owed, ladders owed, a refused
    ladder, a file whose ladders wait on a score, and an old file without the keys. The
    panel reads the keys as written and works out no ladder state itself.
    """

    def test_none_owed(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        _ledger(runs, state="idle", owed_counts={}, refused_owed=0, ladders_owed=[],
                ladder_counts={"ladder": 0, "top-up": 0, "re-rate": 0}, ladders_refused=0)
        view = refreshing.view(runs)
        assert view["ladders"] == "read"
        assert view["laddersOwed"] == 0 and view["ladderCounts"] == {}
        assert view["laddersRefused"] == 0
        assert "No computer-player ladder was owed at its last look." in view["said"]
        assert not any("ladders are owed and refused" in line for line in view["said"])

    def test_owed_says_how_many_of_each_kind_in_order(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        _ledger(runs, state="working", owed_counts={}, refused_owed=0,
                doing={**GAME, "needs": "top-up", "started": refresher._now()},
                ladders_owed=[{**GAME, "needs": "top-up"}],
                ladder_counts={"re-rate": 1, "ladder": 2, "top-up": 3}, ladders_refused=0)
        view = refreshing.view(runs)
        assert view["ladderCounts"] == {"ladder": 2, "top-up": 3, "re-rate": 1}
        assert view["laddersOwed"] == 6
        assert ("6 computer-player ladders were owed at its last look: "
                "2 missing, 3 short, 1 to rate again.") in view["said"]
        assert view["said"][0] == ("The refresher is working on alpha: topping up its "
                                   "computer players, since just now.")
        assert view["doing"]["needs"] == "topping up its computer players"
        # The score lines are unchanged beside them, and come first.
        said = view["said"]
        assert said.index("No game was out of date at its last look.") < said.index(
            "6 computer-player ladders were owed at its last look: "
            "2 missing, 3 short, 1 to rate again.")
        _ledger(runs, ladder_counts={"ladder": 1, "top-up": 0, "re-rate": 0})
        assert ("1 computer-player ladder was owed at its last look: 1 missing."
                in refreshing.view(runs)["said"])

    def test_a_refused_ladder_says_the_game_the_work_and_why(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        ledger = refresher.Ledger(runs)
        score = refresher.Item(hash=GAME["hash"], name="alpha", needs="replay",
                               reason="", tier=3, measured="", stored="d1")
        ladder = refresher.Item(hash=GAME["hash"], name="alpha", needs="re-rate",
                                reason="old players", tier=3, measured="",
                                stored=refresher.LADDER_STORED + "absent")
        ledger.refuse(score, refresher.Outcome(
            hash=GAME["hash"], needs="replay", refusal=refresher.ERROR, reason="boom"))
        ledger.refuse(ladder, refresher.Outcome(
            hash=GAME["hash"], needs="re-rate", refusal=refresher.ERROR,
            reason="rated, and its ladder still owes top-up: short: minimax"))
        ledger.update(state="idle", owed_counts={}, refused_owed=1, ladders_owed=[],
                      ladder_counts={"ladder": 0, "top-up": 0, "re-rate": 0},
                      ladders_refused=1)
        view = refreshing.view(runs)
        assert view["laddersRefused"] == 1 and view["laddersOwed"] == 0
        assert ("1 more ladder is owed and refused, so it does not try them again."
                in view["said"])
        rows = {row["ladder"]: row for row in view["refused"]}
        assert set(rows) == {True, False}
        assert rows[True]["needs"] == "rating its computer players again"
        assert rows[True]["why"] == "measuring it failed"
        assert rows[True]["reason"].startswith("rated, and its ladder still owes")
        assert rows[False]["needs"] == "playing again"
        assert view["lastRefusal"]["ladder"] is True

    def test_ladders_wait_while_a_score_is_owed(self, tmp_path: Path) -> None:
        """The refresher writes the keys as `None` when a score is owed; that is said."""
        runs = tmp_path / "runs"
        _ledger(runs, state="idle", owed_counts={"rescore": 1}, refused_owed=0,
                ladders_owed=None, ladder_counts=None, ladders_refused=2)
        view = refreshing.view(runs)
        assert view["ladders"] == "waiting"
        assert view["laddersOwed"] is None and view["ladderCounts"] is None
        assert view["laddersRefused"] is None
        assert ("Computer players' ladders: not read while a score is owed (scores first)."
                in view["said"])

    def test_an_old_file_without_the_keys_says_nothing_of_ladders(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        _written(runs, 10, state="idle", owed_counts={"rescore": 2}, refused_owed=0)
        view = refreshing.view(runs)
        assert view["ladders"] == "unknown"
        assert view["laddersOwed"] is None and view["ladderCounts"] is None
        assert view["laddersRefused"] is None
        assert not any("ladder" in line for line in view["said"])
        assert view["said"][1] == "2 games out of date at its last look: 2 scoring again."
        assert all(row["ladder"] is False for row in view["refused"])

    def test_no_file_carries_the_ladder_keys_empty(self, tmp_path: Path) -> None:
        view = refreshing.view(tmp_path / "runs")
        assert view["ladders"] == "unknown" and view["laddersOwed"] is None

    def test_a_ladder_done_last_shows_no_score(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        _ledger(runs, state="idle", owed_counts={}, ladder_counts={},
                last_done={**GAME, "needs": "ladder", "index": None})
        assert ("Last done: alpha, rating its computer players."
                in refreshing.view(runs)["said"])

    def test_every_ladder_kind_has_plain_words(self) -> None:
        assert set(refreshing.LADDERS) == set(refresher.LADDER_ORDER)
        assert set(refreshing.LADDER_COUNTED) == set(refresher.LADDER_ORDER)

    def test_the_panel_works_out_no_ladder_state(self) -> None:
        """It reads the status file only: never `ladder_items` or `deploy/`."""
        source = (measurement.ROOT / "chess_variant_sandbox/web/refreshing.py").read_text(
            encoding="utf-8")
        for call in ("ladder_items", "ladder_work", "ladders_check", "_deploy", "opponents"):
            assert call not in source, call


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
