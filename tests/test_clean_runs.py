"""The standing delete rule, `.claude/scripts/clean_runs.py` (Q1188), on a synthetic `runs/`.

A wrong yes deletes something that cannot be re-made, so each refusal the rule promises is
pinned here on its own, beside the one campaign that does qualify and is deleted. Everything
the decision reads besides the tree is a field of `Facts`; the tree is built per test.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".claude" / "scripts" / "clean_runs.py"
spec = importlib.util.spec_from_file_location("clean_runs", SCRIPT)
assert spec is not None and spec.loader is not None
clean = importlib.util.module_from_spec(spec)
sys.modules["clean_runs"] = clean
spec.loader.exec_module(clean)

DAY = 24 * 3600
NOW = time.time() + 60 * DAY  # every file written by the test is two months old
OLD = NOW - 30 * DAY  # a commit well outside the fourteen days


def write(path: Path, data: bytes | str = b"x" * 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return path


def campaign_dir(path: Path, last: str = "end train x: returned") -> Path:
    write(path / "manifest.json", json.dumps({"kind": "campaign", "stages": {}}))
    write(
        path / "progress.log",
        f"[2026-01-01T00:00:00+01:00] start train x\n[2026-01-01T01:00:00+01:00] {last}\n",
    )
    write(path / "games-abc" / "piece-0000.ckpt", b"p" * 1000)
    return path


def finished(runs: Path, name: str) -> Path:
    """A campaign that qualifies: ended, delivered, old; a checkpoint and a harvest to go."""
    top = runs / name
    campaign_dir(top / "checkpoint")
    write(top / "data" / "game.npz", b"n" * 5000)
    write(top / "campaign.log", "done\n")
    return top


def facts(runs: Path, *names: str, **overrides: object):
    base = dict(
        runs=runs,
        now=NOW,
        commits=[(OLD, f"{n}: delivered") for n in names],
        open_queue="# The queue\n- **Q999** something else\n",
        finished_queue="\n".join(f"- **{n}** done" for n in names),
    )
    base.update(overrides)
    return clean.Facts(**base)


def verdict(runs: Path, name: str, f) -> object:
    return clean.decide(runs / name, f)


def test_a_finished_delivered_old_campaign_loses_its_working_files_and_nothing_else(tmp_path):
    runs = tmp_path / "runs"
    top = finished(runs, "Q100")
    write(top / "nets" / "child.net")
    write(top / "nets" / "kept.npz")
    write(top / "results.jsonl", "{}\n")
    write(top / "duels" / "a.json", "{}")
    v = verdict(runs, "Q100", facts(runs, "Q100"))
    assert v.keep is None
    assert sorted(clean.rel(i.path, runs) for i in v.working) == [
        "Q100/checkpoint",
        "Q100/data/game.npz",
    ]
    assert all(i.keep is None for i in v.working)
    freed = clean.apply([v])
    assert freed == v.freed > 5000
    assert not (top / "checkpoint").exists() and not (top / "data" / "game.npz").exists()
    for stays in (
        "nets/child.net",
        "nets/kept.npz",
        "results.jsonl",
        "duels/a.json",
        "campaign.log",
    ):
        assert (top / stays).exists(), stays


def test_dry_run_deletes_nothing_and_says_what_would_go(tmp_path, capsys):
    runs = tmp_path / "runs"
    top = finished(runs, "Q100")
    f = facts(runs, "Q100")
    verdicts = clean.plan(f)
    text = (
        clean.show(verdicts, f, everything=False, act=False)
        + "\n"
        + clean.summary(verdicts, act=False)
    )
    assert "would delete: Q100/checkpoint" in text
    assert "clean_runs: would delete 2 working files from 1 campaigns (runs/Q100)" in text
    assert (top / "checkpoint").exists() and (top / "data" / "game.npz").exists()


@pytest.mark.parametrize(
    "name",
    [
        "0123456789ab",
        "0123456789ab-1000",
        "tournament",
        "puzzles",
        "opponents",
        "endgames",
        "human",
        "v12-calibrate",
        "v11",
        "q478-calibration",
        "v12-rescore",
        "replay-2026-10-20",
        "batch-0925-replay",
        "fresh-seeds-2026-09",
        "seat-fresh-2026-09",
        "recertify-2026-10-20",
        "restamp-Q847",
    ],
)
def test_measurements_anchors_and_shipped_directories_are_never_campaigns(tmp_path, name):
    runs = tmp_path / "runs"
    finished(runs, name)
    v = verdict(runs, name, facts(runs, name))
    assert v.keep and not v.freed
    assert clean.apply([v]) == 0
    assert (runs / name / "checkpoint").exists()


def test_a_campaign_younger_than_fourteen_days_is_kept_with_its_due_date(tmp_path):
    runs = tmp_path / "runs"
    finished(runs, "Q101")
    recent = NOW - 3 * DAY
    v = verdict(
        runs,
        "Q101",
        facts(runs, "Q101", commits=[(OLD, "Q101: start"), (recent, "Q101: written up")]),
    )
    assert v.keep == f"younger than 14 days: falls due {clean.day(recent + 14 * DAY)}"


def test_the_clock_starts_at_the_last_write_when_that_is_later(tmp_path):
    runs = tmp_path / "runs"
    top = finished(runs, "Q101")
    late = NOW - 2 * DAY
    os.utime(top / "campaign.log", (late, late))
    v = verdict(runs, "Q101", facts(runs, "Q101"))
    assert v.clock == pytest.approx(late)
    assert v.keep and v.keep.startswith("younger than 14 days")


def test_no_campaign_manifest_is_no_proof_of_finishing(tmp_path):
    runs = tmp_path / "runs"
    write(runs / "Q102" / "data.npz")
    write(runs / "Q102" / "checkpoint" / "loose.ckpt")
    v = verdict(runs, "Q102", facts(runs, "Q102"))
    assert v.keep == "no campaign manifest: no proof it finished"


def test_a_stalled_or_raised_campaign_inside_is_not_finished(tmp_path):
    runs = tmp_path / "runs"
    finished(runs, "Q103")
    campaign_dir(runs / "Q103" / "checkpoint-old", last="games-abc piece 3/10 done")
    v = verdict(runs, "Q103", facts(runs, "Q103"))
    assert v.keep == "not finished: Q103/checkpoint-old reads STALLED"
    campaign_dir(runs / "Q103" / "checkpoint-old", last="end train x: raised")
    assert (
        verdict(runs, "Q103", facts(runs, "Q103")).keep
        == "not finished: Q103/checkpoint-old reads RAISED"
    )


def test_a_superseded_stalled_campaign_counts_as_finished(tmp_path):
    runs = tmp_path / "runs"
    finished(runs, "Q109")
    campaign_dir(runs / "Q109" / "checkpoint-old", last="games-abc piece 3/10 done")
    write(
        runs / "Q109" / "checkpoint-old" / "SUPERSEDED",
        "killed across a build change; handover-Q109.md\n",
    )
    v = verdict(runs, "Q109", facts(runs, "Q109"))
    assert v.keep is None
    assert "Q109/checkpoint-old" in [clean.rel(i.path, runs) for i in v.working]


def test_a_superseded_marker_written_late_does_not_move_the_clock(tmp_path):
    runs = tmp_path / "runs"
    finished(runs, "Q112")
    old = campaign_dir(runs / "Q112" / "checkpoint-old", last="games-abc piece 3/10 done")
    for file in runs.rglob("*"):
        if file.is_file():
            os.utime(file, (NOW - 40 * DAY, NOW - 40 * DAY))
    write(old / "SUPERSEDED", "killed across a build change; handover-Q112.md\n")
    os.utime(old / "SUPERSEDED", (NOW - DAY, NOW - DAY))
    v = verdict(runs, "Q112", facts(runs, "Q112"))
    assert v.keep is None
    assert v.last_write == pytest.approx(NOW - 40 * DAY)


def test_a_superseded_campaign_is_not_live_when_gathered(tmp_path):
    """Q1189: `gather` once listed a superseded checkpoint as an unfinished campaign, so the
    campaign holding it was kept as "an unfinished campaign is in it" for ever."""
    runs = tmp_path / "runs"
    finished(runs, "Q110")
    old = campaign_dir(runs / "Q110" / "checkpoint-old", last="games-abc piece 3/10 done")
    write(old / "SUPERSEDED", "killed across a build change; docs/archive/handover-Q110.md\n")
    stalled = campaign_dir(runs / "Q111" / "checkpoint", last="games-abc piece 3/10 done")
    for path in (old, stalled):
        for file in path.rglob("*"):
            os.utime(file, (NOW - 40 * DAY, NOW - 40 * DAY))
    gathered = clean.gather(tmp_path, runs, NOW)
    assert [campaign for campaign, _ in gathered.live] == [stalled]
    gathered.commits = [(OLD, "Q110: delivered")]
    gathered.finished_queue = "- **Q110** done"
    v = verdict(runs, "Q110", gathered)
    assert v.keep is None
    assert "Q110/checkpoint-old" in [clean.rel(i.path, runs) for i in v.working]
    assert verdict(runs, "Q111", gathered).keep.startswith("an unfinished campaign is in it")


def test_gather_reads_the_queues_of_the_root_it_is_given(tmp_path):
    """The proof of delivery comes from the checkout `gather` is handed, so no test here reads
    the real queue (Q1191: why the green stamp may count this file among `NOT_READERS`)."""
    runs = tmp_path / "runs"
    finished(runs, "Q113")
    write(tmp_path / "docs" / "queue.md", "- **Q113b** reads `runs/Q113/x`\n")
    write(tmp_path / "docs" / "archive" / "queue-2026-01.md", "- **Q113** done\n")
    gathered = clean.gather(tmp_path, runs, NOW)
    assert "Q113b" in gathered.open_queue and "Q113" in gathered.finished_queue
    gathered.commits = [(OLD, "Q113: delivered")]
    assert verdict(runs, "Q113", gathered).keep == "not delivered: docs/queue.md still names it"


def test_a_held_lease_naming_it_keeps_it(tmp_path):
    runs = tmp_path / "runs"
    finished(runs, "Q104")
    v = verdict(runs, "Q104", facts(runs, "Q104", lease_tags=["Q104 replay"]))
    assert v.keep == "a held lease's tag names it (`Q104 replay`)"
    assert v.working, "a kept campaign still lists what it holds"


def test_a_live_campaign_inside_or_pointing_at_it_keeps_it(tmp_path):
    runs = tmp_path / "runs"
    finished(runs, "Q105")
    elsewhere = tmp_path / "worktree" / "runs" / "Q200" / "checkpoint"
    commands = json.dumps([["train", "x.json", "--data", "runs/Q105/data/game.npz"]])
    v = verdict(runs, "Q105", facts(runs, "Q105", live=[(elsewhere, commands)]))
    assert v.keep and v.keep.startswith("an unfinished campaign names it")
    inside = runs / "Q105" / "checkpoint"
    v = verdict(runs, "Q105", facts(runs, "Q105", live=[(inside, "[]")]))
    assert v.keep and v.keep.startswith("an unfinished campaign is in it")


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        (
            dict(finished_queue="- **Q1060** done"),
            "not delivered: no finished task in docs/archive/queue-*.md names it",
        ),
        (
            dict(open_queue="- **Q106b** reads `runs/Q106/x`"),
            "not delivered: docs/queue.md still names it",
        ),
        (dict(commits=[(OLD, "Q1060: other")]), "not delivered: no commit on main names it"),
    ],
)
def test_no_proof_of_delivery_keeps_it(tmp_path, override, expected):
    runs = tmp_path / "runs"
    finished(runs, "Q106")
    assert verdict(runs, "Q106", facts(runs, "Q106", **override)).keep == expected


def test_a_lowercase_or_suffixed_directory_is_known_by_its_task(tmp_path):
    runs = tmp_path / "runs"
    finished(runs, "q107-probe")
    v = verdict(
        runs,
        "q107-probe",
        facts(runs, finished_queue="- **Q107** done", commits=[(OLD, "Q107: x")]),
    )
    assert v.keep is None


def test_inside_a_qualifying_campaign_each_protected_item_stays(tmp_path):
    runs = tmp_path / "runs"
    top = finished(runs, "Q108")
    write(top / "with-net" / "checkpoint" / "store" / "a.net")
    campaign_dir(top / "with-net" / "checkpoint")
    write(top / "with-record" / "checkpoint" / "games.jsonl", "{}\n")
    campaign_dir(top / "with-record" / "checkpoint")
    write(top / "with-score" / "checkpoint" / "x" / "tournament.json", "{}")
    campaign_dir(top / "with-score" / "checkpoint")
    write(top / "with-previous" / "checkpoint" / "previous" / "1" / "a.txt")
    campaign_dir(top / "with-previous" / "checkpoint")
    write(top / "named" / "positions.npz")
    write(top / "tracked" / "positions.npz")
    write(top / "previous" / "1" / "old.npz")
    f = facts(
        runs, "Q108", code_refs=["Q108/named/positions.npz"], tracked={"Q108/tracked/positions.npz"}
    )
    v = verdict(runs, "Q108", f)
    assert v.keep is None
    kept = {clean.rel(i.path, runs): i.keep for i in v.working}
    assert kept["Q108/with-net/checkpoint"] == "holds a net file"
    assert kept["Q108/with-record/checkpoint"] == "holds a game record or score"
    assert kept["Q108/with-score/checkpoint"] == "holds a game record or score"
    assert kept["Q108/with-previous/checkpoint"] == "holds a previous/ directory"
    assert (
        kept["Q108/named/positions.npz"]
        == "named by tracked code (`runs/Q108/named/positions.npz`)"
    )
    assert kept["Q108/tracked/positions.npz"] == "holds a file tracked by git"
    assert "Q108/previous/1/old.npz" not in kept
    assert kept["Q108/checkpoint"] is None and kept["Q108/data/game.npz"] is None
    clean.apply([v])
    assert not (top / "checkpoint").exists()
    for stays in (
        "with-net/checkpoint",
        "with-record/checkpoint",
        "with-score/checkpoint",
        "with-previous/checkpoint",
        "named/positions.npz",
        "tracked/positions.npz",
        "previous/1/old.npz",
    ):
        assert (top / stays).exists(), stays


def test_code_naming_the_whole_campaign_keeps_every_working_file(tmp_path):
    runs = tmp_path / "runs"
    finished(runs, "Q110")
    v = verdict(runs, "Q110", facts(runs, "Q110", code_refs=["Q110"]))
    assert v.keep is None and v.freed == 0
    assert all(i.keep == "named by tracked code (`runs/Q110`)" for i in v.working)
