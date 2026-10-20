"""Does a library measurement really resume? (Q900, `docs/reference/campaign-checkpoints.md`).

Every path that measures a library game is interrupted here partway through a small
measurement (a five-square game, a three-player roster, one game a pairing, two seeds),
and the second call is counted: how many games the core plays again. The count is the
answer, not a reading of the code.

- `corpus.measure` inside a campaign (`checkpoint.active`) plays only what was not on disk,
  and its records and `canonical.json` are the uninterrupted measurement's, byte for byte
  outside wall time. Outside a campaign it replays from the start, which is all it can do.
- the CLI's `--checkpoint DIR measure`, which is also the call every library-replay
  script's `slice` makes, resumes the same way; the launcher's own resume point is a game.
- the refresher's child (`refresh <hash>`) is given no campaign, so a game it lost is played
  from its start; a named repair given `--checkpoint` resumes; and `refresh --library
  --checkpoint` refuses, since its children could never see the directory.

Until Q900, `corpus` called the runner's `run_tournament` directly, so a campaign directory
given to `measure` was opened, logged and ignored: the second call replayed every game.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chess_variant_sandbox import cli, corpus, refresher
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.tournament import checkpoint

ROOT = Path(__file__).resolve().parent.parent

SPEC = {
    "schema_version": 1,
    "name": "lancers",
    "description": "A small skirmish game.",
    "board": {"width": 5, "height": 5},
    "pieces": [
        {"id": "K", "name": "captain", "royal": True, "value": 0, "moves": [
            {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"}
        ]},
        {"id": "L", "name": "lancer", "value": 5, "moves": [
            {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both"}
        ]},
    ],
    "setup": {"mirror": True, "placements": [
        {"piece": "K", "squares": ["c1"]},
        {"piece": "L", "squares": ["a1", "e1"]},
    ]},
    "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 120,
              "no_progress_draw": 40},
}
DIGEST = spec_hash(SPEC)
# A searching rung, so the analysis pass has games to grade; `threats=0` as in
# `tests/test_corpus.py`, so the tiny ladder has a gradient to score.
AGENTS = ("random", "greedy", "minimax:nodes=50,threats=0")
SEEDS = (0, 1000)
# Every ordered pairing, self-play included, at one game a pairing, for each seed.
GAMES = len(AGENTS) ** 2 * len(SEEDS)
# Killed inside the second seed's tournament: the first seed is whole on disk.
KILL_AT = len(AGENTS) ** 2 + 4


class Killed(Exception):
    """The crash: raised from inside the tournament, as a power cut would stop it."""


class Games:
    """The games the core plays, counted where each becomes a record, and the kill.

    `play.build_record` turns each finished core game into its record, on the runner's
    path and the campaign's alike, so a count of its calls is a count of games played.
    The analysis pass's blunder trials finish games too (`play.finish_game`); those are
    the pass's, not the tournament's, and are neither counted nor killed.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from chess_variant_sandbox.tournament import play, runner

        self.played = 0
        self.kill_after: int | None = None
        lock = threading.Lock()
        real = play.build_record

        def counted(*args: Any, **kwargs: Any) -> Any:
            if sys._getframe(1).f_code is play.finish_game.__code__:
                return real(*args, **kwargs)
            with lock:
                if self.kill_after is not None and self.played >= self.kill_after:
                    raise Killed
                self.played += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(play, "build_record", counted)
        monkeypatch.setattr(runner, "build_record", counted)

    def reset(self, kill_after: int | None = None) -> None:
        self.played = 0
        self.kill_after = kill_after


@pytest.fixture
def library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Games:
    """A checkout with one committed game, on a roster small enough to play in seconds.

    The working directory is the checkout, as `measure` and the replay scripts expect: its
    `rulesets/` holds the game and its `runs/` is where the measurement lands, and the
    library expects `SEEDS`.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rulesets").mkdir()
    (tmp_path / "rulesets" / "lancers.json").write_text(json.dumps(SPEC), encoding="utf-8")
    monkeypatch.setattr(corpus, "CANONICAL_AGENTS", AGENTS)
    monkeypatch.setattr(corpus, "CANONICAL_GAMES", 1)
    monkeypatch.setattr(corpus, "EXPECTED_LIBRARY_SEEDS", SEEDS)
    return Games(monkeypatch)


def _pieces(campaign: Path) -> int:
    """The sealed pieces a campaign holds: the games the second call need not play."""
    return sum(1 for _ in campaign.glob("games-*/piece-*.ckpt"))


def _measure(runs: Path, campaign: Path | None = None) -> None:
    candidate = corpus.find(DIGEST, runs_dir=runs)
    if campaign is None:
        corpus.refresh(candidate, runs_dir=runs, workers=1)
        return
    with checkpoint.active(campaign, build="testbuild"):
        corpus.refresh(candidate, runs_dir=runs, workers=1)


def _answer(runs: Path) -> dict[str, Any]:
    """Every file a measurement leaves, as its answer: wall time aside.

    `duration_ms` in a record is the clock's. Nothing else is set aside: if a resumed
    measurement differs from an uninterrupted one in any other byte, this says where.
    """
    out: dict[str, Any] = {}
    for path in sorted(runs.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(runs).as_posix()
        if path.name == "games.jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                row.pop("duration_ms")
            out[name] = rows
        else:
            out[name] = path.read_bytes()
    return out


def _interrupted(games: Games, call: Any, kill_at: int = KILL_AT) -> None:
    games.reset(kill_after=kill_at)
    with pytest.raises(Killed):
        call()
    assert games.played == kill_at


# ------------------------------------------------------------------ corpus.measure


def test_outside_a_campaign_a_measurement_replays_from_the_start(
    library: Games, tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    _interrupted(library, lambda: _measure(runs))
    library.reset()
    _measure(runs)
    assert library.played == GAMES


def test_inside_a_campaign_a_measurement_resumes_to_the_uninterrupted_one(
    library: Games, tmp_path: Path
) -> None:
    plain = tmp_path / "plain"
    library.reset()
    _measure(plain)
    assert library.played == GAMES

    runs, campaign = tmp_path / "runs", tmp_path / "campaign"
    _interrupted(library, lambda: _measure(runs, campaign))
    saved = _pieces(campaign)
    assert saved == KILL_AT
    library.reset()
    _measure(runs, campaign)
    assert library.played == GAMES - saved
    assert _answer(runs) == _answer(plain)


def test_a_stale_library_measurement_resumes_to_the_uninterrupted_replay(
    library: Games, tmp_path: Path
) -> None:
    """The library replay's own case: a stored measurement whose engine moved.

    What it replaces is kept under `previous/1` once, and its history line written once,
    whether or not the replay was interrupted on the way.
    """
    stale = tmp_path / "stale"
    _measure(stale)
    canonical = corpus.measurement_dir(DIGEST, stale) / "canonical.json"
    stored = json.loads(canonical.read_text(encoding="utf-8"))
    stored["basis"]["engine"] = "0" * 16
    canonical.write_text(json.dumps(stored), encoding="utf-8")

    import shutil

    plain, runs = tmp_path / "plain", tmp_path / "runs"
    shutil.copytree(stale, plain)
    shutil.copytree(stale, runs)
    library.reset()
    _measure(plain)
    assert library.played == GAMES

    campaign = tmp_path / "campaign"
    _interrupted(library, lambda: _measure(runs, campaign))
    saved = _pieces(campaign)
    assert saved == KILL_AT
    library.reset()
    _measure(runs, campaign)
    assert library.played == GAMES - saved
    assert (corpus.measurement_dir(DIGEST, runs) / "previous" / "1").is_dir()
    assert not (corpus.measurement_dir(DIGEST, runs) / "previous" / "2").exists()
    assert _answer(runs) == _answer(plain)


# ------------------------------------------------------------------ the CLI and the launchers


def test_the_cli_measure_resumes_inside_its_checkpoint(library: Games, tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    argv = ["--checkpoint", str(campaign), "measure", "rulesets/lancers.json", "--workers", "1"]
    _interrupted(library, lambda: cli.main(argv))
    saved = _pieces(campaign)
    assert saved == KILL_AT
    library.reset()
    assert cli.main(argv) == 0
    assert library.played == GAMES - saved
    log = (campaign / "progress.log").read_text(encoding="utf-8")
    assert "resumed:" in log


def _newest_replay_script() -> Path:
    """The newest library-replay script: each is a copy of the one before it."""
    scripts = [
        path for path in sorted((ROOT / "docs" / "reference").glob("replay-*.py"))
        if "def slice_(" in path.read_text(encoding="utf-8")
    ]
    assert scripts
    return scripts[-1]


def test_the_replay_launcher_resumes_inside_a_game(
    library: Games, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replay script's `slice`, killed mid-game, then run again as its launcher would.

    The launcher's own resume point is a game (`progress.jsonl`): a game with a line is not
    run again. Inside a game it is `--checkpoint`, which `slice` passes to `measure`. The
    script asks for ten workers; the analysis pass is held to one here, since threads never
    change an answer and ten spawned processes would cost more than nine games.
    """
    script = _newest_replay_script()
    found = importlib.util.spec_from_file_location("replay_launcher", script)
    assert found is not None and found.loader is not None
    module = importlib.util.module_from_spec(found)
    found.loader.exec_module(module)
    assert not Path(module.HERE).is_absolute(), "the script must work in its own checkout"
    monkeypatch.setattr(module, "_library", lambda: [DIGEST])
    analyse = corpus.analyse_corpus
    monkeypatch.setattr(corpus, "analyse_corpus",
                        lambda spec, records, seed=0, workers=0: analyse(spec, records,
                                                                        seed=seed, workers=1))

    _interrupted(library, lambda: module.slice_(60.0))
    saved = _pieces(Path(module.CHECKPOINT))
    assert 0 < saved <= KILL_AT
    library.reset()
    assert module.slice_(60.0) == 0
    assert library.played == GAMES - saved
    library.reset()
    assert module.slice_(60.0) == 0
    assert library.played == 0, "a game the launcher finished is not run again"


# ------------------------------------------------------------------ the refresher


def test_the_refreshers_child_is_given_no_campaign_and_replays_a_lost_game(
    library: Games, tmp_path: Path
) -> None:
    """What `refresh --library` runs per game, and what it does with one it lost.

    The child's command names no `--checkpoint`, and `stage_game` empties the game's
    staging before each repair, so a game the refresher lost is played from its start.
    """
    seen: list[list[str]] = []

    def run(argv: list[str], **_: Any) -> Any:
        seen.append([str(part) for part in argv])
        return SimpleNamespace(returncode=0, stdout=json.dumps({"hash": DIGEST, "needs": ""}),
                               stderr="")

    machine = refresher.GitMachine(main_checkout=tmp_path, worktree=tmp_path / "wt", run=run)
    machine.repair(refresher.Item.from_dict({
        "hash": DIGEST, "name": "lancers", "needs": "replay", "tier": 0, "reason": "",
    }))
    child = next(argv for argv in seen if "chess_variant_sandbox.cli" in argv)
    assert "--checkpoint" not in child
    assert child[child.index("chess_variant_sandbox.cli") + 1] == "refresh"

    runs = tmp_path / "runs"
    library.reset(kill_after=KILL_AT)
    lost = refresher.repair(DIGEST, runs, workers=1)
    assert lost.refused and "Killed" in lost.reason
    library.reset()
    assert not refresher.repair(DIGEST, runs, workers=1).refused
    assert library.played == GAMES


def test_a_named_repair_resumes_inside_its_checkpoint(library: Games, tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    argv = ["--checkpoint", str(campaign), "refresh", DIGEST, "--runs", "runs", "--workers", "1"]
    library.reset(kill_after=KILL_AT)
    assert cli.main(argv) == 1, "the repair refuses what raised, and says so"
    saved = _pieces(campaign)
    assert saved == KILL_AT
    library.reset()
    assert cli.main(argv) == 0
    assert library.played == GAMES - saved


def test_the_refresher_refuses_a_checkpoint_it_could_not_pass_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`refresh --library --checkpoint DIR` would open DIR and play every game elsewhere."""
    started: list[Any] = []

    class Loop:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            started.append(args)

        def run(self) -> int:
            return 0

    monkeypatch.setattr(refresher, "Refresher", Loop)
    monkeypatch.setattr(refresher, "main_checkout_of", lambda _path: tmp_path)
    monkeypatch.setattr(refresher.GitMachine, "exists", lambda self: True)
    campaign = tmp_path / "campaign"
    code = cli.main(["--checkpoint", str(campaign), "refresh", "--library",
                     "--worktree", str(tmp_path / "wt")])
    assert code == 2
    assert not started
    assert not campaign.exists()
    assert "--checkpoint" in capsys.readouterr().err
