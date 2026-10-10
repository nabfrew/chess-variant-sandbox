"""Campaigns that survive a crash (Q732, ``tournament/checkpoint.py``).

What is held here:

- a file written atomically is the old one or the whole new one, and a sealed one that was
  torn or altered reads as missing;
- a campaign directory remembers its build and refuses another; a stage is named by what
  it computes and cut into pieces by its units alone, never the worker count;
- a schedule, a harvest and a training run inside a campaign give what they give outside
  one, byte for byte, and again after a crash that left only some pieces or saves on disk;
- a torn training save is skipped for the one before it;
- (slow) a real process killed mid-run and run again gives the uninterrupted run's
  aggregate: a tournament's ratings and games, a net's store.

Records differ only in ``duration_ms``, wall time, which is never part of an answer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament import checkpoint
from chess_variant_sandbox.tournament.checkpoint import (
    Campaign,
    CampaignError,
    atomic_write,
    read_sealed,
    run_tournament,
    seal,
    unseal,
)
from chess_variant_sandbox.tournament.runner import TournamentConfig
from chess_variant_sandbox.tournament.runner import run_tournament as runner_run

ROOT = Path(__file__).resolve().parent.parent
GARDNER = ROOT / "rulesets" / "gardner.json"
BUILD = "testbuild0000000"


def _rows(records: list[Any]) -> list[dict[str, Any]]:
    """Records as their answer: everything but wall time."""
    rows = []
    for record in records:
        row = record.as_dict()
        row.pop("duration_ms")
        rows.append(row)
    return rows


@pytest.fixture
def campaign_dir(tmp_path: Path) -> Iterator[Path]:
    yield tmp_path / "campaign"
    assert checkpoint.current() is None


# ------------------------------------------------------------------ files


def test_an_atomic_write_leaves_no_temp_and_replaces_whole(tmp_path: Path) -> None:
    path = tmp_path / "a" / "file.bin"
    atomic_write(path, b"one")
    atomic_write(path, b"two, longer")
    assert path.read_bytes() == b"two, longer"
    assert [p.name for p in path.parent.iterdir()] == ["file.bin"]


def test_a_torn_or_altered_seal_reads_as_missing(tmp_path: Path) -> None:
    blob = seal(b"payload")
    assert unseal(blob) == b"payload"
    assert unseal(blob[:-1]) is None
    assert unseal(blob[:-1] + b"X") is None
    assert unseal(b"short") is None
    path = tmp_path / "p.ckpt"
    assert read_sealed(path) is None
    path.write_bytes(blob[: len(blob) // 2])
    assert read_sealed(path) is None
    path.write_bytes(blob)
    assert read_sealed(path) == b"payload"


# ------------------------------------------------------------------ the manifest


def test_a_campaign_remembers_its_build_and_refuses_another(campaign_dir: Path) -> None:
    Campaign.open(campaign_dir, build=BUILD, command=["tournament", "x"])
    manifest = json.loads((campaign_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "campaign" and manifest["build"] == BUILD
    assert manifest["commands"] == [["tournament", "x"]]
    Campaign.open(campaign_dir, build=BUILD, command=["tournament", "x"])  # resumes
    with pytest.raises(CampaignError, match="begun on build"):
        Campaign.open(campaign_dir, build="another0000000000")
    log = (campaign_dir / "progress.log").read_text(encoding="utf-8")
    assert "begin on build" in log and "resume on build" in log


def test_a_stage_is_named_by_what_it_computes_and_cut_by_its_units(campaign_dir: Path) -> None:
    campaign = Campaign.open(campaign_dir, build=BUILD)
    units = list(range(2500))
    one = campaign.stage("games", {"ruleset_hash": "r", "seed": 1}, units)
    again = campaign.stage("games", {"ruleset_hash": "r", "seed": 1}, units)
    other = campaign.stage("games", {"ruleset_hash": "r", "seed": 2}, units)
    assert one.name == again.name != other.name
    assert len(one.pieces) <= checkpoint.PIECES
    assert [u for piece in one.pieces for u in piece] == units
    stages = json.loads((campaign_dir / "manifest.json").read_text(encoding="utf-8"))["stages"]
    assert stages[one.name] == {"kind": "games", "pieces": len(one.pieces), "units": 2500, "label": ""}


def test_a_stage_plays_only_the_pieces_not_on_disk(campaign_dir: Path) -> None:
    campaign = Campaign.open(campaign_dir, build=BUILD)
    stage = campaign.stage("squares", {"n": 9}, list(range(9)), size=2)
    played: list[list[int]] = []

    def run() -> list[list[int]]:
        with ThreadPoolExecutor(max_workers=3) as pool:
            def submit(piece: list[int]) -> Future[list[int]]:
                played.append(piece)
                return pool.submit(lambda: [u * u for u in piece])

            return stage.run(
                submit,
                lambda value: json.dumps(value).encode(),
                lambda payload, piece: json.loads(payload),
            )

    first = run()
    assert first == [[0, 1], [4, 9], [16, 25], [36, 49], [64]] and len(played) == 5
    stage.path(1).unlink()
    stage.path(3).write_bytes(stage.path(3).read_bytes()[:-3])  # torn
    played.clear()
    assert run() == first
    assert played == [[2, 3], [6, 7]]
    assert "torn or foreign" in (campaign_dir / "progress.log").read_text(encoding="utf-8")


# ------------------------------------------------------------------ games


def _config(workers: int) -> TournamentConfig:
    return TournamentConfig(
        agents=("random", "greedy"), games_per_pairing=5, seed=7, workers=workers,
        include_self_play=False,
    )


def test_a_schedule_in_a_campaign_is_the_schedule_and_resumes_to_it(campaign_dir: Path) -> None:
    spec = load_spec(GARDNER)
    plain = runner_run(spec, _config(2))
    assert _rows(run_tournament(spec, _config(2)).records) == _rows(plain.records)  # no campaign
    with checkpoint.active(campaign_dir, build=BUILD):
        inside = run_tournament(spec, _config(3))
    assert _rows(inside.records) == _rows(plain.records)
    assert inside.ratings.as_dict() == plain.ratings.as_dict()
    # A crash that left half the pieces: the rest are played, the answer is the same.
    [stage] = [p for p in campaign_dir.iterdir() if p.is_dir()]
    pieces = sorted(stage.glob("piece-*.ckpt"))
    assert len(pieces) == 10
    for path in pieces[::2]:
        path.unlink()
    with checkpoint.active(campaign_dir, build=BUILD):
        resumed = run_tournament(spec, _config(1))
    assert _rows(resumed.records) == _rows(plain.records)
    assert "resumed: 5/10 pieces" in (campaign_dir / "progress.log").read_text(encoding="utf-8")


def test_an_opening_schedule_in_a_campaign_is_the_schedule(campaign_dir: Path) -> None:
    pytest.importorskip("numpy")
    from chess_variant_sandbox.agents.training.data import _play_tasks
    from chess_variant_sandbox.tournament.runner import schedule

    spec = load_spec(GARDNER)
    config = _config(2)
    plain = _play_tasks(spec, config, schedule(config), 4)
    with checkpoint.active(campaign_dir, build=BUILD):
        inside = _play_tasks(spec, config, schedule(config), 4)
    assert _rows(inside) == _rows(plain)


# ------------------------------------------------------------------ the harvest and training


@pytest.fixture
def tiny() -> Any:
    pytest.importorskip("numpy")
    from chess_variant_sandbox.agents.training.procedure import Procedure

    return Procedure(
        agents=("greedy",), games=3, skip_plies=2, result_weight=0.5, label_nodes=40,
        label_depth=3, td_lambda=0.0, accumulator=16, hidden1=8, hidden2=4, epochs=3, batch=32,
    )


def test_a_harvest_in_a_campaign_is_the_harvest(campaign_dir: Path, tiny: Any) -> None:
    from chess_variant_sandbox.agents.training.data import harvest, self_play

    spec = load_spec(GARDNER)
    games = self_play(spec, tiny, workers=1)
    plain = harvest(spec, games, tiny, workers=1)
    with checkpoint.active(campaign_dir, build=BUILD):
        inside = harvest(spec, games, tiny, workers=1)
        again = harvest(spec, games, tiny, workers=1)  # every piece from disk
    for found in (inside, again):
        assert found.digest == plain.digest
        for column in ("game", "ply", "results", "values"):
            assert getattr(found, column).tobytes() == getattr(plain, column).tobytes()


class _Crash(Exception):
    pass


def test_training_resumes_from_its_last_save_to_the_same_bytes(
    campaign_dir: Path, tiny: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chess_variant_sandbox.agents.training import model
    from chess_variant_sandbox.agents.training.data import harvest, self_play

    spec = load_spec(GARDNER)
    data = harvest(spec, self_play(spec, tiny, workers=1), tiny, workers=1)
    plain = model.train(data, tiny)
    real_save = checkpoint.Saves.save
    for crash_after in (1, 4, 7):

        def saving(
            self: Any, arrays: Any, meta: Any, limit: int = crash_after, calls: list[int] = [0]  # noqa: B006
        ) -> Any:
            path = real_save(self, arrays, meta)
            calls[0] += 1
            if calls[0] == limit:
                raise _Crash
            return path

        directory = campaign_dir / str(crash_after)
        monkeypatch.setattr(checkpoint.Saves, "save", saving)
        with pytest.raises(_Crash), checkpoint.active(directory, save_minutes=0, build=BUILD):
            model.train(data, tiny)
        monkeypatch.setattr(checkpoint.Saves, "save", real_save)
        with checkpoint.active(directory, save_minutes=0, build=BUILD):
            resumed = model.train(data, tiny)
        assert model.quantize(resumed) == model.quantize(plain), crash_after
        assert resumed.history == plain.history
        for key, value in plain.params.items():
            assert resumed.params[key].tobytes() == value.tobytes(), key


def test_a_torn_training_save_is_skipped_for_the_one_before(campaign_dir: Path) -> None:
    np = pytest.importorskip("numpy")
    campaign = Campaign.open(campaign_dir, build=BUILD, save_minutes=0)
    saves = campaign.saves({"data": "d"})
    assert saves.latest() is None
    saves.save({"w": np.arange(4, dtype=np.float32)}, {"step": 1})
    newest = saves.save({"w": np.arange(4, dtype=np.float32) * 2}, {"step": 2})
    arrays, meta = saves.latest() or ({}, {})
    assert meta["step"] == 2 and arrays["w"].tolist() == [0, 2, 4, 6]
    newest.write_bytes(newest.read_bytes()[:-10])
    arrays, meta = saves.latest() or ({}, {})
    assert meta["step"] == 1 and arrays["w"].tolist() == [0, 1, 2, 3]
    for step in range(3, 8):
        saves.save({"w": np.zeros(1, dtype=np.float32)}, {"step": step})
    assert len(list(saves.stage.directory.glob("train-*.ckpt"))) == checkpoint.KEEP_SAVES


# ------------------------------------------------------------------ the stall check


def _status_script() -> Any:
    import importlib.util

    path = ROOT / ".claude" / "scripts" / "campaign_status.py"
    found = importlib.util.spec_from_file_location("campaign_status", path)
    assert found is not None and found.loader is not None
    module = importlib.util.module_from_spec(found)
    found.loader.exec_module(module)
    return module


def test_the_stall_check_tells_running_stalled_and_ended_apart(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    status = _status_script()
    root = tmp_path / "runs"
    campaign = Campaign.open(root / "q1" / "checkpoint", build=BUILD, command=["tournament"])
    stage = campaign.stage("games", {"ruleset_hash": "r"}, list(range(4)))
    stage.save(0, b"x")
    assert status.main([str(root)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("RUNNING") and "1/4 pieces" in out

    log = root / "q1" / "checkpoint" / "progress.log"
    old = time.time() - 3600
    os.utime(log, (old, old))
    assert status.main([str(root), "--minutes", "20"]) == 1
    assert capsys.readouterr().out.startswith("STALLED")

    campaign.log("end tournament: returned")
    os.utime(log, (old, old))
    assert status.main([str(root)]) == 0
    assert capsys.readouterr().out.startswith("ENDED")
    assert status.main([str(tmp_path / "nothing")]) == 0
    assert "no campaign" in capsys.readouterr().out


# ------------------------------------------------------------------ killed for real (slow)


def _cli(*args: str) -> list[str]:
    return [sys.executable, "-m", "chess_variant_sandbox.cli", *args]


def _kill_when(command: list[str], log: Path, pieces: int, cwd: Path) -> None:
    """Start ``command``, kill it once ``log`` records ``pieces`` pieces, and check it died
    before it finished (else the test proves nothing)."""
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 300
    try:
        while time.monotonic() < deadline:
            text = log.read_text(encoding="utf-8") if log.exists() else ""
            if text.count(" done (") + text.count(" saved train-") >= pieces:
                break
            if process.poll() is not None:
                break
            time.sleep(0.05)
        process.kill()
        process.wait()
    finally:
        if process.poll() is None:
            process.kill()
    text = log.read_text(encoding="utf-8")
    assert ": returned" not in text, "the run finished before it was killed"


@pytest.mark.slow
def test_a_killed_tournament_resumes_to_the_uninterrupted_result(tmp_path: Path) -> None:
    agents = ["--agents", "greedy", "minimax:nodes=300", "--games", "12", "--seed", "3",
              "--workers", "2"]
    plain = tmp_path / "plain"
    subprocess.run(
        _cli("tournament", str(GARDNER), *agents, "--out", str(plain)), cwd=ROOT, check=True,
        stdout=subprocess.DEVNULL,
    )
    campaign = tmp_path / "campaign"
    resumed = tmp_path / "resumed"
    command = _cli("--checkpoint", str(campaign), "tournament", str(GARDNER), *agents,
                   "--out", str(resumed))
    _kill_when(command, campaign / "progress.log", 6, ROOT)
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    log = (campaign / "progress.log").read_text(encoding="utf-8")
    assert "resumed:" in log

    def answer(directory: Path) -> tuple[Any, list[Any]]:
        summary = json.loads((directory / "tournament.json").read_text(encoding="utf-8"))
        games = [json.loads(line) for line in (directory / "games.jsonl").read_text().splitlines()]
        for game in games:
            game.pop("duration_ms")
        return summary, games

    assert answer(resumed) == answer(plain)


@pytest.mark.slow
def test_a_killed_training_resumes_to_the_uninterrupted_store(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    train = [
        "train", str(GARDNER), "--agents", "greedy", "--games", "6", "--label-nodes", "40",
        "--label-depth", "3", "--accumulator", "16", "--hidden1", "8", "--hidden2", "4",
        "--epochs", "40", "--batch", "16", "--duel-pairs", "4", "--duel-nodes", "60",
        "--workers", "2",
    ]
    plain = tmp_path / "plain"
    subprocess.run(_cli(*train, "--store", str(plain)), cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    campaign = tmp_path / "campaign"
    store = tmp_path / "resumed"
    command = _cli("--checkpoint", str(campaign), "--checkpoint-minutes", "0", *train,
                   "--store", str(store))
    # Killed in the middle of training: after the games, the harvest and some saves.
    _kill_when(command, campaign / "progress.log", 60, ROOT)
    assert " saved train-" in (campaign / "progress.log").read_text(encoding="utf-8")
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    assert "resumes from train-" in (campaign / "progress.log").read_text(encoding="utf-8")
    made = sorted(p.name for p in plain.iterdir())
    assert made == sorted(p.name for p in store.iterdir()) and len(made) == 2
    for name in made:
        assert (store / name).read_bytes() == (plain / name).read_bytes(), name
