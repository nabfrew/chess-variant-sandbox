"""Proven positions keep themselves up to date (Q993d, `docs/archive/plan-proven.md` §5, §6).

The refresher's third tier: once no score and no ladder is owed, it reads which Library and
Baseline games are owed a proven-positions record (`puzzles.survey`), and proves one game a
`heavy` lease (`refresher.prove_game`, `puzzles.make`), writing the record in place. What
these hold down, on `test_puzzles`' five-square game (not chess: a royal, a stepper and a
walker a side) with fixture library runs its own players play here:

- **What is owed**, and why: no record; a record whose positions no longer replay; one in
  an older format or proved by an older probe (`PROOF_VERSION`; none written is version 1);
  never a game of rounds, nor one whose four runs are not on disk; a record from newer code
  is not owed.
- **The sample** skips the last two plies of a won game (`NEAR_MIN` = 3, §3.5), and the
  record's basis names it.
- **`make`**: a record whole, T1 to T5 applied to every new position, each stage resumable
  from its unit files and the same however many threads; a game whose runs hold no small
  ending gets the empty record, which the site reads as no record and which is not owed
  again; work sampled another way is moved aside, never written over.
- **The loop**, over the refresher's fake machine: proofs after ladders after scores, one
  game a `heavy` lease, nothing delivered, a refusal keyed on the game's record and never
  retried, three refusals per game kept apart.
- **What it says**: `refresh --plan` and `--status`, `proofs --check`, and the real
  machine's child commands.
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_puzzles import SMALL, SPEC, _game
from test_refresher import FakeMachine, repo  # noqa: F401  (`repo` is a fixture)

from chess_variant_sandbox import measurement, puzzles, refresher
from chess_variant_sandbox.corpus import LIBRARY_SEEDS
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.refresher import Item, Outcome
from chess_variant_sandbox.site import proven
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.tournament.record import GameRecord, write_records

REPOSITORY = Path(__file__).resolve().parent.parent
GAMES = 2  # a seed: enough won games for a few dozen positions near their ends

# A game nobody can win early: no royal, nothing captures (a stepper only onto empty
# squares is still a capture rule, so the pieces are many and the board full). Its runs
# hold no position with six pieces or fewer in play, so its record is the empty one.
CROWDED: dict[str, Any] = {
    **SPEC, "name": "puzzle crowd",
    "setup": {"mirror": False, "placements": [
        *SPEC["setup"]["placements"],
        {"piece": "S", "player": 0, "squares": ["a2", "b2"]},
        {"piece": "S", "player": 1, "squares": ["d4", "e4"]},
    ]},
}


@pytest.fixture(scope="module")
def played() -> dict[int, list[GameRecord]]:
    """Each library seed's games, played once for the module by the core's own players."""
    engine = Engine(SPEC)
    return {seed: [_game(engine, seed + n, ("minimax:nodes=400", "random"))
                   for n in range(GAMES)] for seed in LIBRARY_SEEDS}


def lay(root: Path, spec: dict[str, Any], played: dict[int, list[GameRecord]] | None,
        stem: str, archived: bool = True) -> str:
    """A library game under ``root`` with its four runs in ``root``'s `runs/`."""
    folder = root / ("archive" if archived else "rulesets")
    folder.mkdir(parents=True, exist_ok=True)
    (root / ("rulesets" if archived else "archive")).mkdir(exist_ok=True)
    (folder / f"{stem}.json").write_text(json.dumps(spec), encoding="utf-8")
    digest = spec_hash(spec)
    if played is not None:
        for seed, games in played.items():
            records = [GameRecord(**{**game.__dict__, "ruleset_hash": digest}) for game in games]
            write_records(records, root / "runs" / f"{digest}-{seed}" / "games.jsonl")
    return digest


def crowd_runs(root: Path, digest: str) -> None:
    """The crowded game's runs: its own players' games, none small."""
    engine = Engine(CROWDED)
    for seed in LIBRARY_SEEDS:
        games = [_game(engine, seed, ("random", "random"), limit=12)]
        write_records([GameRecord(**{**g.__dict__, "ruleset_hash": digest}) for g in games],
                      root / "runs" / f"{digest}-{seed}" / "games.jsonl")


@pytest.fixture
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """`make` at a test's budget, wherever it is called from; the build fresh."""
    monkeypatch.setattr(puzzles, "make", functools.partial(
        puzzles.make, nodes=SMALL, defence_nodes=SMALL))
    monkeypatch.setattr(measurement, "stale_build", lambda: None)


@pytest.fixture(scope="module")
def made(tmp_path_factory: pytest.TempPathFactory,
         played: dict[int, list[GameRecord]]) -> tuple[Path, str, dict[str, Any]]:
    """The fixture game proved once for the module: its root, hash and what `make` said."""
    root = tmp_path_factory.mktemp("made")
    digest = lay(root, SPEC, played, "probe")
    said = puzzles.make(SPEC, root / "runs", workers=4, nodes=SMALL, defence_nodes=SMALL)
    return root, digest, said


# ------------------------------------------------------------------ what is owed


class TestWhatIsOwed:
    def test_the_refreshers_kinds_are_the_upkeeps(self) -> None:
        assert refresher.PROOF_ORDER == puzzles.NEEDS
        assert all(refresher.LEASE_FOR[needs] == "heavy" for needs in puzzles.NEEDS)
        assert refresher.QUEUE_ORDER[-2:] == puzzles.NEEDS

    def test_a_game_with_runs_and_no_record_is_owed_a_proof(
        self, tmp_path: Path, played: Any
    ) -> None:
        digest = lay(tmp_path, SPEC, played, "probe")
        [found] = puzzles.survey(tmp_path / "runs", tmp_path)
        assert (found.hash, found.needs, found.reason) == (digest, puzzles.PROVE, "no record")
        assert found.stored == puzzles.STORED + "absent"

    def test_no_runs_no_rounds_and_no_preview_are_owed(self, tmp_path: Path, played) -> None:
        # Three games apart in their mechanics (a name moves no hash).
        lay(tmp_path, {**SPEC, "name": "no runs yet",
                       "rules": {**SPEC["rules"], "max_plies": 101}}, None, "bare")
        lay(tmp_path, {**SPEC, "name": "rounds", "schema_version": 26,
                       "rules": {"win": "royal_capture", "simultaneous": {}}}, None, "rounds")
        lay(tmp_path, {**SPEC, "name": "shown", "rules": {**SPEC["rules"], "max_plies": 102},
                       "provenance": {"preview": "A preview."}}, played, "shown")
        found = {f.name: f for f in puzzles.survey(tmp_path / "runs", tmp_path)}
        assert set(found) == {"no runs yet", "rounds"}  # an archived preview is not library
        assert not found["no runs yet"].owed
        assert "not all on disk (seeds 0, 1000, 2000, 3000)" in found["no runs yet"].reason
        assert not found["rounds"].owed and "rounds" in found["rounds"].reason

    def test_a_record_that_holds_is_owed_nothing(self, made) -> None:
        root, digest, said = made
        [found] = puzzles.survey(root / "runs", root)
        assert not found.owed and found.reason.startswith("current: ")
        assert found.stored == puzzles.stored_of(root / "runs", digest) != (
            puzzles.STORED + "absent")

    def test_each_way_a_record_stops_holding(self, made) -> None:
        root, digest, _said = made
        engine = Engine(SPEC)
        data = json.loads(puzzles.record_path(root / "runs", digest).read_text("utf-8"))
        assert puzzles.owes(engine, data) is None
        assert "format 3" in puzzles.owes(engine, {**data, "format": 3})
        assert puzzles.owes(engine, {**data, "format": puzzles.FORMAT + 1}) is None
        older = {**data, "basis": {**data["basis"], "proof_version": 0}}
        assert "probe version 0" in puzzles.owes(engine, older)
        # Before the field existed: version 1, the probe as R9a ran it.
        unstamped = {**data, "basis": {k: v for k, v in data["basis"].items()
                                       if k != "proof_version"}}
        assert puzzles.owes(engine, unstamped) is None
        moved = json.loads(json.dumps(data))
        moved["positions"][0]["moves"] = [10 ** 6]
        assert puzzles.owes(engine, moved) == (
            f"1 of its {len(data['positions'])} proved positions no longer replay")
        twice = {**data, "positions": [data["positions"][0], data["positions"][0]]}
        assert puzzles.owes(engine, twice) == "1 of its 2 proved positions no longer replay"
        assert puzzles.owes(engine, {**data, "ruleset_hash": "x"}) is not None
        assert puzzles.owes(engine, None) is not None

    def test_a_reproof_is_owed_only_where_the_runs_are(self, made, tmp_path: Path) -> None:
        root, digest, _said = made
        copy = tmp_path / "copy"
        shutil.copytree(root, copy)
        path = puzzles.record_path(copy / "runs", digest)
        path.write_text(json.dumps({**json.loads(path.read_text("utf-8")), "format": 2}),
                        encoding="utf-8")
        [found] = puzzles.survey(copy / "runs", copy)
        assert found.needs == puzzles.REPROVE and "format 2" in found.reason
        assert found.made == json.loads(path.read_text("utf-8"))["made"]
        shutil.rmtree(copy / "runs" / f"{digest}-1000")
        [found] = puzzles.survey(copy / "runs", copy)
        assert not found.owed and "seeds 1000" in found.reason


# ------------------------------------------------------------------ the sample


def test_the_sample_skips_the_last_two_plies(played) -> None:
    engine = Engine(SPEC)
    runs = list(played.items())
    every = list(puzzles.candidates(engine, runs, near=puzzles.NEAR))
    near = list(puzzles.candidates(engine, runs, near=puzzles.NEAR, near_min=puzzles.NEAR_MIN))
    ends = {(seed, n): len(game.moves) for seed, games in played.items()
            for n, game in enumerate(games)}
    left = [ends[(c.seed, c.game)] - c.ply for c in every]
    assert min(left) < puzzles.NEAR_MIN == 3  # the fixture has some, or this tests nothing
    assert near == [c for c, plies in zip(every, left, strict=True) if plies >= 3]


# ------------------------------------------------------------------ one game made


class TestMake:
    def test_the_record_is_whole_with_every_test_applied(self, made) -> None:
        root, digest, said = made
        data = json.loads(puzzles.record_path(root / "runs", digest).read_text("utf-8"))
        assert data["format"] == puzzles.FORMAT and data["ruleset_hash"] == digest
        basis = data["basis"]
        assert basis["near"] == puzzles.NEAR and basis["near_min"] == puzzles.NEAR_MIN
        assert basis["proof_version"] == puzzles.PROOF_VERSION
        assert basis["runs"] == [f"{digest}-{seed}" for seed in LIBRARY_SEEDS]
        for test in puzzles.TESTS:
            assert basis[f"{test}_nodes"] == SMALL and basis[f"{test}_built"]
        assert data["positions"], "the fixture proves something"
        for row in data["positions"]:
            assert row["result"] == puzzles.WON
            # T3 and T4 are asked of exactly what they apply to, T5 of what they keep.
            assert ("defence" in row) == puzzles.asks_defence(row)
            assert ("hasten" in row) == ("defence" in row)
            assert ("shallow" in row) == puzzles.asks_shallow(row)
        offered = [row for row in data["positions"] if not puzzles.trivial(row)]
        assert said["offered"] == len(offered) and said["proved"] == len(data["positions"])
        assert set(said["seconds"]) == {"count", "prove", *puzzles.TESTS}

    def test_the_site_offers_what_the_tests_keep(self, made) -> None:
        root, digest, said = made
        record = proven.load(root / "runs", digest, SPEC)
        assert record is not None and len(record.positions) == said["offered"]
        assert record.shallow and record.tests == "T5"

    def test_a_killed_run_picks_up_and_threads_change_nothing(
        self, made, tmp_path: Path
    ) -> None:
        root, digest, _said = made
        copy = tmp_path / "copy"
        shutil.copytree(root, copy)
        work = puzzles.work_dir(copy / "runs", digest)
        # Killed half way: the record never written, a proof batch and a T5 unit lost.
        puzzles.record_path(copy / "runs", digest).unlink()
        (work / "batch-0008.json").unlink()
        for unit in sorted(work.glob("shallow-*.json"))[:1]:
            unit.unlink()
        again = puzzles.make(SPEC, copy / "runs", workers=1, nodes=SMALL,
                             defence_nodes=SMALL)
        assert "count" not in again["seconds"]  # the count was on disk
        first = json.loads(puzzles.record_path(root / "runs", digest).read_text("utf-8"))
        second = json.loads(puzzles.record_path(copy / "runs", digest).read_text("utf-8"))
        assert second == first
        assert not list(work.glob("record-*.json"))  # nothing was there to keep

    def test_a_game_replayed_under_the_same_run_names_is_sampled_again(
        self, made, tmp_path: Path
    ) -> None:
        """Q1032: a re-measure keeps the run directories' names and changes their games, so
        the work must not be picked up: the count is taken again and the old work kept."""
        root, digest, _said = made
        copy = tmp_path / "copy"
        shutil.copytree(root, copy)
        games = copy / "runs" / f"{digest}-0" / "games.jsonl"
        lines = games.read_text(encoding="utf-8").splitlines(keepends=True)
        games.write_text("".join(lines[:-1]), encoding="utf-8")
        again = puzzles.make(SPEC, copy / "runs", workers=1, nodes=SMALL,
                             defence_nodes=SMALL)
        assert "count" in again["seconds"]
        assert list(puzzles.work_dir(copy / "runs", digest).parent.glob(f"{digest}.before-*"))

    def test_a_record_replaced_is_kept_and_other_work_moved_aside(
        self, made, tmp_path: Path
    ) -> None:
        root, digest, _said = made
        copy = tmp_path / "copy"
        shutil.copytree(root, copy)
        work = puzzles.work_dir(copy / "runs", digest)
        listed = json.loads((work / "candidates.json").read_text("utf-8"))
        listed["sampling"] = {**listed["sampling"], "near_min": 1}
        (work / "candidates.json").write_text(json.dumps(listed), encoding="utf-8")
        before = puzzles.record_path(copy / "runs", digest).read_bytes()
        puzzles.make(SPEC, copy / "runs", workers=4, nodes=SMALL, defence_nodes=SMALL)
        [aside] = work.parent.glob(f"{digest}.before-*")
        assert (aside / "batch-0000.json").exists()  # moved, not written over
        [kept] = work.glob("record-*.json")
        assert kept.read_bytes() == before
        assert puzzles.record_path(copy / "runs", digest).read_bytes() == before

    def test_a_game_with_no_small_ending_gets_the_empty_record(self, tmp_path: Path) -> None:
        digest = lay(tmp_path, CROWDED, None, "crowd")
        crowd_runs(tmp_path, digest)
        said = puzzles.make(CROWDED, tmp_path / "runs", nodes=SMALL, defence_nodes=SMALL)
        assert said["proved"] == said["offered"] == 0
        data = json.loads(puzzles.record_path(tmp_path / "runs", digest).read_text("utf-8"))
        assert data["positions"] == [] and data["counts"]["candidates"] == 0
        assert data["basis"]["shallow_nodes"] == SMALL  # applied, to none
        # Asked once: it is owed nothing now, and the site reads it as no record.
        [found] = puzzles.survey(tmp_path / "runs", tmp_path)
        assert not found.owed and "no small ending" in found.reason
        assert proven.load(tmp_path / "runs", digest, CROWDED) is None
        assert proven.offered(tmp_path / "runs", digest, CROWDED) == 0

    def test_a_game_of_rounds_or_without_runs_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            puzzles.make(SPEC, tmp_path / "runs", nodes=SMALL)
        rounds = {**SPEC, "schema_version": 26,
                  "rules": {"win": "royal_capture", "simultaneous": {}}}
        with pytest.raises(ValueError):
            puzzles.make(rounds, tmp_path / "runs", nodes=SMALL)


# ------------------------------------------------------------------ one game, here


class TestProveGame:
    def test_an_owed_game_is_proved_and_then_owes_nothing(
        self, tmp_path: Path, played, fast
    ) -> None:
        digest = lay(tmp_path, SPEC, played, "probe")
        outcome = refresher.prove_game(digest, tmp_path / "runs", tmp_path, workers=4)
        assert not outcome.refused, outcome
        assert outcome.needs == puzzles.PROVE
        assert any("offered;" in line for line in outcome.lines)
        assert refresher.proof_items(tmp_path / "runs", tmp_path) == []
        again = refresher.prove_game(digest, tmp_path / "runs", tmp_path)
        assert not again.refused and again.needs == ""

    def test_a_stale_build_is_refused_before_anything_runs(
        self, tmp_path: Path, played, fast, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        digest = lay(tmp_path, SPEC, played, "probe")
        monkeypatch.setattr(measurement, "stale_build", lambda: "the core is behind")
        outcome = refresher.prove_game(digest, tmp_path / "runs", tmp_path)
        assert outcome.refusal == refresher.STALE_BUILD
        assert not puzzles.work_dir(tmp_path / "runs", digest).exists()

    def test_a_failure_is_a_refusal_keyed_on_the_record(
        self, tmp_path: Path, played, fast, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        digest = lay(tmp_path, SPEC, played, "probe")

        def broken(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("the probe fell over")

        monkeypatch.setattr(puzzles, "make", broken)
        outcome = refresher.prove_game(digest, tmp_path / "runs", tmp_path)
        assert outcome.refusal == refresher.ERROR and "fell over" in outcome.reason
        assert outcome.needs == puzzles.PROVE and outcome.stored == ""
        assert refresher.prove_game("0" * 12, tmp_path / "runs", tmp_path).refusal == (
            refresher.ERROR)

    def test_a_proof_that_leaves_it_owed_is_refused_as_the_file_it_left(
        self, tmp_path: Path, played, fast, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        digest = lay(tmp_path, SPEC, played, "probe")

        def old_format(spec: Any, runs: Path, **_k: Any) -> dict[str, Any]:
            puzzles.write(puzzles.record_path(runs, digest),
                          {"format": 1, "ruleset_hash": digest, "positions": []})
            return {"proved": 0, "offered": 0, "seconds": {}}

        monkeypatch.setattr(puzzles, "make", old_format)
        outcome = refresher.prove_game(digest, tmp_path / "runs", tmp_path)
        assert outcome.refusal == refresher.ERROR and "still owed re-prove" in outcome.reason
        assert outcome.stored == puzzles.stored_of(tmp_path / "runs", digest)


# ------------------------------------------------------------------ the loop


class TestTheLoop:
    def test_proofs_come_after_scores_and_ladders_one_game_a_heavy_lease(
        self, tmp_path: Path
    ) -> None:
        machine = FakeMachine(owed={"s": ("rescore", "s0")},
                              ladders_owed={"l": ("ladder", "ladder:absent")},
                              proofs_owed={"p": ("prove", "proofs:absent"),
                                           "q": ("re-prove", "proofs:abc")})
        looping = refresher.Refresher(machine, refresher.Ledger(tmp_path), passes=10)
        assert looping.run() == 0
        work = [c[:2] for c in machine.calls
                if c[0] in ("acquire", "repair", "release", "deliver", "ladders", "proofs")]
        assert work == [
            ("acquire", "quiet"), ("repair", "s"), ("release", "quiet"), ("deliver", "s"),
            ("ladders", "c1"),
            ("acquire", "heavy"), ("repair", "l"), ("release", "heavy"),
            ("ladders", "c1"), ("proofs", "c1"),
            ("acquire", "heavy"), ("repair", "p"), ("release", "heavy"),
            ("ladders", "c1"), ("proofs", "c1"),
            ("acquire", "heavy"), ("repair", "q"), ("release", "heavy"),
            ("ladders", "c1"), ("proofs", "c1"),
        ]
        log = refresher.read_log(tmp_path)
        assert [(line["hash"], line["needs"]) for line in log if line["event"] == "repaired"] == [
            ("s", "rescore"), ("l", "ladder"), ("p", "prove"), ("q", "re-prove")]
        status = refresher.read_status(tmp_path)
        assert status is not None and status["proof_counts"] == {"prove": 0, "re-prove": 0}

    def test_the_proofs_are_not_read_while_a_score_or_ladder_is_owed(
        self, tmp_path: Path
    ) -> None:
        machine = FakeMachine(owed={"s": ("rescore", "s0")},
                              proofs_owed={"p": ("prove", "proofs:absent")})
        looping = refresher.Refresher(machine, refresher.Ledger(tmp_path), passes=1)
        assert looping.step() == "repaired"
        assert machine.did("proofs") == []
        assert looping.ledger.status["proof_counts"] is None
        machine.ladders_owed = {"l": ("ladder", "ladder:absent")}
        assert looping.step() == "repaired"
        assert machine.did("proofs") == [] and machine.did("repair")[-1][1] == "l"
        assert looping.step() == "repaired"
        assert machine.did("repair")[-1][1] == "p"

    def test_a_refused_proof_is_not_retried_and_three_refusals_stay_apart(
        self, tmp_path: Path
    ) -> None:
        machine = FakeMachine(owed={"a": ("rescore", "s")},
                              ladders_owed={"a": ("ladder", "ladder:absent")},
                              proofs_owed={"a": ("prove", "proofs:absent")},
                              refuse={"a": refresher.ERROR})
        looping = refresher.Refresher(machine, refresher.Ledger(tmp_path), passes=8)
        assert looping.run() == 0
        assert [c[1] for c in machine.did("repair")] == ["a", "a", "a"]
        refusals = looping.ledger.refusals
        assert sorted(r["stored"] for r in refusals) == [
            "ladder:absent", "proofs:absent", "s"]
        assert looping.ledger.status["proofs_refused"] == 1
        # Its record changed (somebody proved it): the refusal lapses, the others stand.
        machine.proofs_owed = {}
        looping.step()
        assert sorted(r["stored"] for r in looping.ledger.refusals) == ["ladder:absent", "s"]

    def test_only_reaches_the_proofs_too(self, tmp_path: Path) -> None:
        machine = FakeMachine(proofs_owed={"a": ("prove", "x"), "b": ("prove", "y")})
        refresher.Refresher(machine, refresher.Ledger(tmp_path), passes=5,
                            only=frozenset({"b"})).run()
        assert [c[1] for c in machine.did("repair")] == ["b"]

    def test_a_failed_proofs_survey_halts_and_is_logged(self, tmp_path: Path) -> None:
        machine = FakeMachine()

        def broken() -> list[Item]:
            raise RuntimeError("no library")

        machine.proofs = broken  # type: ignore[method-assign]
        assert refresher.Refresher(machine, refresher.Ledger(tmp_path), passes=3).run() == 1
        status = refresher.read_status(tmp_path)
        assert status is not None and "the proofs survey failed" in status["last_refusal"][
            "reason"]


class FixtureMachine(FakeMachine):
    """A fake machine whose proofs are the fixture library's, surveyed by `proof_items` and
    proved by `prove_game` at a test's budget, as the worktree's children would."""

    root: Path = Path()

    def proofs(self) -> list[Item]:
        self.calls.append(("proofs", self.commit))
        return refresher.proof_items(self.root / "runs", self.root)

    def repair(self, it: Item) -> Outcome:
        if not it.proofs:
            return super().repair(it)
        assert self.holding == it.lease == "heavy"
        self.calls.append(("repair", it.hash, self.commit))
        return replace(refresher.prove_game(it.hash, self.root / "runs", self.root, workers=4),
                       commit=self.commit)


def test_the_loop_proves_a_new_game_end_to_end(tmp_path: Path, played, fast) -> None:
    """A game joins the library with its four runs: nobody triggers anything, and the next
    idle pass proves it, once, and the site then offers its positions."""
    digest = lay(tmp_path, SPEC, played, "probe")
    crowd = lay(tmp_path, CROWDED, None, "crowd")
    crowd_runs(tmp_path, crowd)
    machine = FixtureMachine()
    machine.root = tmp_path
    looping = refresher.Refresher(machine, refresher.Ledger(tmp_path / "ledger"), passes=6)
    assert looping.run() == 0
    assert sorted(c[1] for c in machine.did("repair")) == sorted([digest, crowd])
    assert machine.did("deliver") == []
    assert refresher.proof_items(tmp_path / "runs", tmp_path) == []
    record = proven.load(tmp_path / "runs", digest, SPEC)
    assert record is not None and record.basis["near_min"] == puzzles.NEAR_MIN
    assert proven.load(tmp_path / "runs", crowd, CROWDED) is None  # the empty record
    # The log keeps what each proof found and what its stages took.
    said = [line["said"] for line in refresher.read_log(tmp_path / "ledger")
            if line["event"] == "repaired"]
    assert len(said) == 2 and all(" offered; count " in line for line in said)


# ------------------------------------------------------------------ what it says


class TestWhatItSays:
    def test_the_status_says_which_proofs_are_owed_and_next(self, tmp_path: Path) -> None:
        machine = FakeMachine(ladders_owed={"l": ("ladder", "x")},
                              proofs_owed={"p": ("prove", "y"), "r": ("re-prove", "z")})
        looping = refresher.Refresher(machine, refresher.Ledger(tmp_path), passes=1)
        assert looping.step() == "repaired"
        lines = refresher.plain_status(refresher.read_status(tmp_path))
        assert "proofs: read once no score or ladder is owed (scores, then ladders)" in lines
        assert looping.step() == "repaired"
        lines = refresher.plain_status(refresher.read_status(tmp_path))
        assert "proofs owed: 1 prove, 1 re-prove" in lines
        assert any(line.startswith("next: p [p] prove") for line in lines)
        assert "last: p [p] prove" in lines
        machine.refuse = {"r": refresher.ERROR}
        looping.step()
        looping.step()
        lines = refresher.plain_status(refresher.read_status(tmp_path))
        assert "proofs owed: nothing (1 more owed and refused)" in lines

    def test_the_plan_lists_the_proofs_after_the_ladders(
        self, tmp_path: Path, played, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        from chess_variant_sandbox.cli import main

        digest = lay(tmp_path, SPEC, played, "probe")
        monkeypatch.chdir(tmp_path)
        assert main(["refresh", "--plan", "--part", "proofs", "--json", "--runs", "runs"]) == 0
        [row] = json.loads(capsys.readouterr().out)
        assert (row["hash"], row["needs"], row["lease"]) == (digest, "prove", "heavy")
        monkeypatch.setattr(refresher, "queue", lambda runs, weights=None: [])
        monkeypatch.setattr(refresher, "ladder_items", lambda runs, root=None: [
            Item(hash="l", name="ladder", needs="ladder", reason="r", tier=3, measured="",
                 stored="ladder:absent")])
        assert main(["refresh", "--plan", "--runs", "runs"]) == 0
        out = capsys.readouterr().out
        assert out.index(" ladder ") < out.index(" prove ")
        assert ("2 owed: 0 scores, then 1 ladders, then 1 proofs (the proofs until no "
                "ladder is)") in out

    def test_proofs_check_says_what_is_owed_and_proofs_names_a_game(
        self, tmp_path: Path, played, fast, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        from chess_variant_sandbox.cli import main

        digest = lay(tmp_path, SPEC, played, "probe")
        monkeypatch.chdir(tmp_path)
        assert main(["proofs", "--check", "--runs", "runs"]) == 1
        out = capsys.readouterr().out
        assert "prove     puzzle probe" in out and "no record" in out
        assert "1 of 1 owed: 1 prove, 0 re-prove" in out
        assert main(["proofs", "archive/probe.json", "--runs", "runs", "--workers", "4"]) == 0
        capsys.readouterr()
        assert main(["proofs", "--check", "--runs", "runs", "--json"]) == 0
        [found] = json.loads(capsys.readouterr().out)
        assert found["hash"] == digest and not found["owed"]
        assert main(["proofs", "nothing-here", "--check", "--runs", "runs"]) == 2

    def test_refresh_proofs_names_a_game(self, capsys) -> None:
        from chess_variant_sandbox.cli import main

        assert main(["refresh", "--proofs"]) == 2
        assert "--proofs names a game" in capsys.readouterr().err
        assert main(["refresh", "abc", "--proofs", "--ladder"]) == 2


class TestTheRealMachine:
    def test_a_proof_runs_in_the_worktree_into_main_runs(self, repo) -> None:  # noqa: F811
        seen: list[list[str]] = []

        def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            seen.append(list(argv))
            if argv[:2] == ["git", "rev-parse"]:
                return subprocess.CompletedProcess(argv, 0, "c1\n", "")
            if "--plan" in argv:
                row = Item(hash="h", name="h", needs="prove", reason="no record", tier=3,
                           measured="", stored="proofs:absent").as_dict()
                return subprocess.CompletedProcess(argv, 0, json.dumps([row]), "")
            return subprocess.CompletedProcess(argv, 0, json.dumps(
                Outcome(hash="h", needs="prove").as_dict()), "")

        machine = replace(repo, run=run, workers=6)
        [found] = machine.proofs()
        assert found.proofs and found.lease == "heavy"
        outcome = machine.repair(found)
        assert not outcome.refused and outcome.commit == "c1"
        survey, prove = [argv for argv in seen if argv[0] != "git"]
        assert Path(survey[0]) == refresher._python(repo.worktree)
        assert survey[3:7] == ["refresh", "--plan", "--part", "proofs"]
        assert prove[3:6] == ["refresh", "h", "--proofs"]
        assert prove[prove.index("--runs") + 1] == str(repo.main_runs)
        assert prove[prove.index("--workers") + 1] == "6"
        # Nothing staged: a record is written in place in the main checkout's runs/.
        assert not (repo.worktree / "runs").exists()


def _main_runs() -> Path:
    """The main checkout's `runs/`, where the delivered records are, from any worktree."""
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            capture_output=True, text=True, cwd=REPOSITORY, check=False)
    if common.returncode == 0:
        return Path(common.stdout.strip()).parent / "runs"
    return REPOSITORY / "runs"


MAIN_RUNS = _main_runs()


LIBRARY = {digest: (name, spec) for digest, name, spec in puzzles.library(REPOSITORY)}


@pytest.mark.parametrize("digest", sorted(LIBRARY))
def test_every_delivered_record_owes_nothing(digest: str) -> None:
    """`proofs --check` clean for every record the main checkout holds (§6's acceptance,
    then the upkeep's backfill): each replays whole and is current. One case a game of the
    library, which the tree fixes, so every xdist worker collects the same cases while the
    refresher writes records (Q993i: one test over the growing set ran past the duration
    guard). Skips where the game has no record on disk."""
    if not puzzles.record_path(MAIN_RUNS, digest).is_file():
        pytest.skip("no runs/puzzles/ record of this game here")
    name, spec = LIBRARY[digest]
    found = puzzles.standing(digest, name, spec, MAIN_RUNS)
    assert not found.owed, (name, found.reason)
