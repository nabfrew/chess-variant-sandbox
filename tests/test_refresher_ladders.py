"""The refresher keeps ladders current (L7, `docs/plan-leaderboard.md` §2.2).

`refresh --library --watch` rates a missing picker ladder, then tops up a short one, then
rates again one with old players, one ladder a lease, and only once no score is owed. What
these hold down, on fixture libraries and ladders built here (nothing plays: `ladders.rate`
is the seam, as in `test_deploy.py`, and writes the ladder a rating would):

- **The order**: `ladder_items` reads each game through L3's `ladders_check.ladder`, and
  orders missing, then a top-up, then a re-rating (old players, or short with no kept
  games); within each, the launch set, then the oldest ladder, then the hash.
- **Scores keep priority**: the ladders are not even read while a score is owed.
- **One ladder a lease**, under `heavy`, and nothing delivered: a ladder is written in
  place into the main checkout's `runs/opponents/`, the old one moved aside by `rerate`.
- **L3's code is called, not rewritten**: `rate_ladder` calls `ladders.rate` (with
  `top_up` for a short ladder) and `ladders.rerate`; a rating that fails or leaves the
  ladder owing is refused, keyed on its file, and never retried; a ladder's refusal and
  its game's score refusal are two.
- **`--plan` and `--status` say which ladder work is owed and next**, and the real
  machine runs it as a child of the worktree.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from test_refresher import FakeMachine, repo  # noqa: F401  (`repo` is a fixture)

from chess_variant_sandbox import measurement, refresher
from chess_variant_sandbox.refresher import Item, Outcome
from chess_variant_sandbox.spec.canonical import load_spec, spec_hash

REPOSITORY = Path(__file__).resolve().parent.parent

# The picker every game here offers: no rule of thumb, no Fairy-Stockfish to ask.
OFFERED = ("random", "greedy", "minimax:nodes=200000")
SPINE = ["random", "greedy"]
# After every change `opponents.PLAYER_CHANGES` marks (Q502i, 2026-09-29).
RECENT = "2026-10-01"


@pytest.fixture
def checks(monkeypatch: pytest.MonkeyPatch) -> Any:
    """`ladders_check` as the refresher imports it, its picker stood in for."""
    from chess_variant_sandbox.web.opponents import Opponent

    module = refresher._deploy("ladders_check")
    players = [Opponent(agent=agent, label=agent, group="") for agent in OFFERED]
    monkeypatch.setattr(module, "roster", lambda game, root: players)
    monkeypatch.setattr(measurement, "stale_build", lambda: None)
    return module


def tree(root: Path, **games: str) -> dict[str, str]:
    """A library under ``root``: each named ruleset copied into `rulesets/`; stem to hash."""
    (root / "rulesets").mkdir(parents=True, exist_ok=True)
    (root / "archive").mkdir(exist_ok=True)
    (root / "runs" / "opponents").mkdir(parents=True, exist_ok=True)
    digests = {}
    for stem, source in games.items():
        target = root / "rulesets" / f"{stem}.json"
        shutil.copy(REPOSITORY / "rulesets" / f"{source}.json", target)
        digests[stem] = spec_hash(load_spec(target))
    return digests


def stamped(runs: Path, digest: str, players: tuple[str, ...], rated: str) -> None:
    """A ladder written since L3: its stamp, and its games kept beside it."""
    path = runs / "opponents" / f"{digest}.json"
    games = 2
    path.write_text(json.dumps({
        "format": 2, "rated": rated, "games": games, "roster": list(players),
        "spine": SPINE, "standings": [{"name": p, "rating": 0.0} for p in players],
    }), encoding="utf-8")
    line = json.dumps({"seed": 0, "agents": SPINE, "winner": 1, "reason": "", "plies": 3})
    path.with_name(f"{digest}.games.jsonl").write_text((line + "\n") * games,
                                                       encoding="utf-8")


def unstamped(runs: Path, digest: str, players: tuple[str, ...], dated: str) -> None:
    """A ladder from before L3: standings only, its date its file's."""
    path = runs / "opponents" / f"{digest}.json"
    path.write_text(json.dumps({"standings": [{"name": p, "rating": 0.0} for p in players]}),
                    encoding="utf-8")
    then = datetime.fromisoformat(dated).replace(hour=12, tzinfo=timezone.utc).timestamp()
    os.utime(path, (then, then))


@pytest.fixture
def owing(tmp_path: Path, checks: Any) -> tuple[Path, dict[str, str]]:
    """Five games: two missing (one in the launch set), one short with kept games, one
    with old players, one owing nothing."""
    digests = tree(tmp_path, dull="dull", dobutsu="dobutsu", gardner="gardner",
                   courier="courier", makruk="makruk")
    runs = tmp_path / "runs"
    stamped(runs, digests["gardner"], ("random", "greedy"), RECENT)
    unstamped(runs, digests["courier"], OFFERED, "2026-09-09")
    stamped(runs, digests["makruk"], OFFERED, RECENT)
    return tmp_path, digests


# ------------------------------------------------------------------ the order


class TestTheOrder:
    def test_missing_then_a_top_up_then_old_players(self, owing) -> None:
        root, digests = owing
        items = refresher.ladder_items(root / "runs", root)
        by_hash = {digest: stem for stem, digest in digests.items()}
        assert [(by_hash[i.hash], i.needs) for i in items] == [
            ("dobutsu", "ladder"),  # the launch set first within a kind
            ("dull", "ladder"),
            ("gardner", "top-up"),
            ("courier", "re-rate"),
        ]
        assert all(i.lease == "heavy" and i.ladder for i in items)
        assert items[2].reason == "short: minimax:nodes=200000"
        assert items[3].reason.startswith("old players: rated 2026-09-09, before 2026-09-29")
        assert items[0].stored == refresher.LADDER_STORED + "absent"
        assert items[2].stored.startswith(refresher.LADDER_STORED) and items[2].stored != (
            refresher.LADDER_STORED + "absent")

    def test_a_short_ladder_with_no_kept_games_is_rated_again_whole(
        self, tmp_path: Path, checks: Any
    ) -> None:
        digests = tree(tmp_path, dull="dull")
        unstamped(tmp_path / "runs", digests["dull"], ("random", "greedy"), RECENT)
        [found] = refresher.ladder_items(tmp_path / "runs", tmp_path)
        assert found.needs == "re-rate"
        assert found.reason == "short: minimax:nodes=200000 (no kept games)"

    def test_within_a_kind_the_oldest_ladder_first(self, tmp_path: Path, checks: Any) -> None:
        digests = tree(tmp_path, dull="dull", gardner="gardner")
        unstamped(tmp_path / "runs", digests["dull"], OFFERED, "2026-09-20")
        unstamped(tmp_path / "runs", digests["gardner"], OFFERED, "2026-09-10")
        assert [i.hash for i in refresher.ladder_items(tmp_path / "runs", tmp_path)] == [
            digests["gardner"], digests["dull"]]

    def test_a_preview_owes_no_ladder_and_a_library_game_without_one_still_does(
        self, tmp_path: Path, checks: Any
    ) -> None:
        """Q934d: previews get no rated play, so the plan never lists one as owed."""
        digests = tree(tmp_path, dull="dull")
        spec = json.loads((REPOSITORY / "rulesets" / "gardner.json").read_text("utf-8"))
        spec["provenance"] = {**(spec.get("provenance") or {}), "preview": "A preview."}
        (tmp_path / "archive" / "gardner-preview.json").write_text(
            json.dumps(spec), encoding="utf-8")
        preview = spec_hash(load_spec(tmp_path / "archive" / "gardner-preview.json"))
        assert preview != digests["dull"]

        owed = [i.hash for i in refresher.ladder_items(tmp_path / "runs", tmp_path)]
        assert owed == [digests["dull"]]
        assert preview not in [g.digest for g in checks.library(tmp_path)]

        # The same file committed to rulesets/ is a library game, as the loader reads it.
        (tmp_path / "rulesets" / "gardner-preview.json").write_text(
            json.dumps(spec), encoding="utf-8")
        owed = [i.hash for i in refresher.ladder_items(tmp_path / "runs", tmp_path)]
        assert sorted(owed) == sorted([digests["dull"], preview])

    def test_every_score_repair_comes_before_any_ladder(self) -> None:
        def it(needs: str) -> Item:
            return Item(hash="h", name="h", needs=needs, reason="", tier=0, measured="",
                        stored="s")

        everything = [it(needs) for needs in reversed(refresher.QUEUE_ORDER)]
        assert [i.needs for i in sorted(everything, key=lambda i: i.key)] == [
            "refit", "rescore", "reanalyse", "replay", "ladder", "top-up", "re-rate"]


# ------------------------------------------------------------------ one ladder, here


@pytest.fixture
def rating(monkeypatch: pytest.MonkeyPatch, checks: Any) -> list[tuple[str, bool]]:
    """`ladders.rate` stood in for: it writes the ladder a full rating writes, today."""
    ladders = refresher._deploy("ladders")
    asked: list[tuple[str, bool]] = []

    def rate(game: Any, root: Path, runs: Path, say: Any = print, top_up: bool = False) -> bool:
        asked.append((game.digest, top_up))
        say(f"  {game.name} [{game.digest}]: rated")
        stamped(runs, game.digest, OFFERED, RECENT)
        return True

    monkeypatch.setattr(ladders, "rate", rate)
    return asked


class TestOneLadder:
    def test_each_kind_is_paid_by_l3s_own_call(self, owing, rating) -> None:
        root, digests = owing
        runs = root / "runs"
        for stem in ("dull", "gardner", "courier"):
            outcome = refresher.rate_ladder(digests[stem], runs, root)
            assert not outcome.refused, outcome
        assert rating == [(digests["dull"], False), (digests["gardner"], True),
                          (digests["courier"], False)]
        # `rerate` moved the old ladder aside before rating; nothing was written over.
        [aside] = (runs / "opponents").glob("before-*")
        assert (aside / f"{digests['courier']}.json").exists()
        [left] = refresher.ladder_items(runs, root)
        assert left.hash == digests["dobutsu"] and left.needs == "ladder"

    def test_a_ladder_owing_nothing_is_no_refusal_and_plays_nothing(self, owing, rating) -> None:
        root, digests = owing
        outcome = refresher.rate_ladder(digests["makruk"], root / "runs", root)
        assert not outcome.refused and outcome.needs == "" and rating == []

    def test_a_failed_rating_is_refused_and_the_old_ladder_kept(
        self, owing, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root, digests = owing
        before = (root / "runs" / "opponents" / f"{digests['courier']}.json").read_bytes()
        monkeypatch.setattr(refresher._deploy("ladders"), "rate", lambda *a, **k: False)
        outcome = refresher.rate_ladder(digests["courier"], root / "runs", root)
        assert outcome.refusal == refresher.ERROR and outcome.needs == "re-rate"
        assert (root / "runs" / "opponents" / f"{digests['courier']}.json").read_bytes() == before

    def test_a_rating_that_leaves_it_owing_is_refused_as_the_file_it_left(
        self, owing, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root, digests = owing
        runs = root / "runs"

        def short(game: Any, root: Path, runs: Path, **_: Any) -> bool:
            stamped(runs, game.digest, ("random",), RECENT)
            return True

        monkeypatch.setattr(refresher._deploy("ladders"), "rate", short)
        outcome = refresher.rate_ladder(digests["dull"], runs, root)
        assert outcome.refusal == refresher.ERROR and "still owes" in outcome.reason
        assert outcome.stored == refresher.ladder_stored(digests["dull"], runs)

    def test_a_stale_build_is_refused_before_anything_runs(
        self, owing, rating, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root, digests = owing
        monkeypatch.setattr(measurement, "stale_build", lambda: "the core is behind")
        outcome = refresher.rate_ladder(digests["dull"], root / "runs", root)
        assert outcome.refusal == refresher.STALE_BUILD and rating == []

    def test_an_unknown_game_is_a_refusal_not_a_crash(self, owing, rating) -> None:
        root, _digests = owing
        assert refresher.rate_ladder("0" * 12, root / "runs", root).refusal == refresher.ERROR


# ------------------------------------------------------------------ the loop


class FixtureMachine(FakeMachine):
    """A fake machine whose ladder work is the fixture's: surveyed by `ladder_items` and
    paid by `rate_ladder`, as the worktree's children would."""

    root: Path = Path()

    def ladders(self) -> list[Item]:
        self.calls.append(("ladders", self.commit))
        return refresher.ladder_items(self.root / "runs", self.root)

    def repair(self, it: Item) -> Outcome:
        if not it.ladder:
            return super().repair(it)
        assert self.holding == it.lease == "heavy"
        self.calls.append(("repair", it.hash, self.commit))
        return replace(refresher.rate_ladder(it.hash, self.root / "runs", self.root),
                       commit=self.commit)


def machine_on(root: Path, **fields: Any) -> FixtureMachine:
    machine = FixtureMachine(**fields)
    machine.root = root
    return machine


class TestTheLoop:
    def test_scores_first_then_one_ladder_a_lease_in_order(self, owing, rating) -> None:
        root, digests = owing
        machine = machine_on(root, owed={digests["dull"]: ("rescore", "s")})
        looping = refresher.Refresher(machine, refresher.Ledger(root / "ledger"), passes=20)
        assert looping.run() == 0
        work = [c[:2] for c in machine.calls
                if c[0] in ("acquire", "repair", "release", "deliver", "ladders")]
        assert work == [
            # The score: the ladders are not read while it is owed.
            ("acquire", "quiet"), ("repair", digests["dull"]), ("release", "quiet"),
            ("deliver", digests["dull"]),
            ("ladders", "c1"),
            ("acquire", "heavy"), ("repair", digests["dobutsu"]), ("release", "heavy"),
            ("ladders", "c1"),
            ("acquire", "heavy"), ("repair", digests["dull"]), ("release", "heavy"),
            ("ladders", "c1"),
            ("acquire", "heavy"), ("repair", digests["gardner"]), ("release", "heavy"),
            ("ladders", "c1"),
            ("acquire", "heavy"), ("repair", digests["courier"]), ("release", "heavy"),
            ("ladders", "c1"),
        ]
        assert rating == [(digests["dobutsu"], False), (digests["dull"], False),
                          (digests["gardner"], True), (digests["courier"], False)]
        log = refresher.read_log(root / "ledger")
        assert [(line["hash"], line["needs"]) for line in log if line["event"] == "repaired"] == [
            (digests["dull"], "rescore"), (digests["dobutsu"], "ladder"),
            (digests["dull"], "ladder"), (digests["gardner"], "top-up"),
            (digests["courier"], "re-rate")]
        status = refresher.read_status(root / "ledger")
        assert status is not None and status["state"] == "stopped"
        assert status["ladder_counts"] == {"ladder": 0, "top-up": 0, "re-rate": 0}

    def test_a_score_owed_mid_way_comes_before_the_next_ladder(self, owing, rating) -> None:
        root, digests = owing
        machine = machine_on(root)
        looping = refresher.Refresher(machine, refresher.Ledger(root / "ledger"), passes=1)
        assert looping.step() == "repaired"
        machine.owed = {digests["makruk"]: ("replay", "s")}
        assert looping.step() == "repaired"
        assert machine.did("deliver") == [("deliver", digests["makruk"])]
        assert len(machine.did("ladders")) == 1
        assert looping.step() == "repaired"
        assert len(machine.did("ladders")) == 2
        assert rating == [(digests["dobutsu"], False), (digests["dull"], False)]

    def test_a_refused_ladder_is_not_retried_and_leaves_the_score_refusal(
        self, tmp_path: Path, rating
    ) -> None:
        machine = FakeMachine(owed={"a": ("rescore", "s")},
                              ladders_owed={"a": ("ladder", "ladder:absent")},
                              refuse={"a": refresher.ERROR})
        looping = refresher.Refresher(machine, refresher.Ledger(tmp_path), passes=6)
        assert looping.run() == 0
        assert machine.did("repair") == [("repair", "a", "c1"), ("repair", "a", "c1")]
        refusals = looping.ledger.refusals
        assert sorted(r["stored"] for r in refusals) == ["ladder:absent", "s"]
        # The ladder survey lapses only ladder refusals; the score survey only scores'.
        machine.ladders_owed = {}
        looping.step()
        assert [r["stored"] for r in looping.ledger.refusals] == ["s"]
        assert looping.ledger.status["ladders_refused"] == 0

    def test_only_reaches_the_ladders_too(self, tmp_path: Path) -> None:
        machine = FakeMachine(ladders_owed={"a": ("ladder", "x"), "b": ("top-up", "y")})
        refresher.Refresher(machine, refresher.Ledger(tmp_path), passes=5,
                            only=frozenset({"b"})).run()
        assert [c[1] for c in machine.did("repair")] == ["b"]

    def test_a_failed_ladder_survey_halts_and_is_logged(self, tmp_path: Path) -> None:
        machine = FakeMachine()

        def broken() -> list[Item]:
            raise RuntimeError("no deploy")

        machine.ladders = broken  # type: ignore[method-assign]
        assert refresher.Refresher(machine, refresher.Ledger(tmp_path), passes=3).run() == 1
        assert "the ladder survey failed" in refresher.read_status(tmp_path)["last_refusal"][  # type: ignore[index]
            "reason"]


# ------------------------------------------------------------------ what it says


class TestWhatItSays:
    def test_the_status_says_which_ladder_work_is_owed_and_next(self, tmp_path: Path) -> None:
        machine = FakeMachine(owed={"s1": ("rescore", "s")},
                              ladders_owed={"m": ("ladder", "x"), "t": ("top-up", "y"),
                                            "r": ("re-rate", "z")})
        looping = refresher.Refresher(machine, refresher.Ledger(tmp_path), passes=1)
        # A score owed: the ladders are not read, and the status says why.
        assert looping.step() == "repaired"
        lines = refresher.plain_status(refresher.read_status(tmp_path))
        assert "ladders: read once no score is owed (scores first)" in lines
        assert any(line.startswith("next: s1 [s1] rescore") for line in lines)
        assert machine.did("ladders") == []

        # None owed: the ladder work, by kind, and the one taken next.
        assert looping.step() == "repaired"
        lines = refresher.plain_status(refresher.read_status(tmp_path))
        assert "ladders owed: 1 ladder, 1 top-up, 1 re-rate" in lines
        assert "owed: nothing" in lines
        assert any(line.startswith("next: m [m] ladder") for line in lines)
        assert "last: m [m] ladder" in lines

        machine.refuse = {"t": refresher.ERROR}
        looping.step()
        looping.step()
        lines = refresher.plain_status(refresher.read_status(tmp_path))
        assert "ladders owed: 1 re-rate (1 more owed and refused)" in lines

    def test_the_plan_lists_the_ladder_work_after_the_scores(
        self, owing, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        from chess_variant_sandbox.cli import main

        root, digests = owing
        monkeypatch.chdir(root)
        assert main(["refresh", "--plan", "--part", "ladders", "--json",
                     "--runs", "runs"]) == 0
        rows = json.loads(capsys.readouterr().out)
        assert [(r["hash"], r["needs"], r["lease"]) for r in rows] == [
            (digests["dobutsu"], "ladder", "heavy"), (digests["dull"], "ladder", "heavy"),
            (digests["gardner"], "top-up", "heavy"), (digests["courier"], "re-rate", "heavy")]
        monkeypatch.setattr(refresher, "queue", lambda runs, weights=None: [
            Item(hash="s", name="score", needs="replay", reason="r", tier=3, measured="",
                 stored="s")])
        assert main(["refresh", "--plan", "--runs", "runs"]) == 0
        out = capsys.readouterr().out
        assert out.index("replay") < out.index(" ladder ") < out.index("top-up") < out.index(
            "re-rate")
        assert "5 owed: 1 scores, then 4 ladders (the ladders wait until no score is owed)" in out

    def test_a_ladder_is_paid_by_a_named_game_and_never_without_one(
        self, owing, rating, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        from chess_variant_sandbox.cli import main

        root, digests = owing
        monkeypatch.chdir(root)
        assert main(["refresh", "--ladder"]) == 2
        assert main(["refresh", digests["dull"], "--ladder", "--json", "--runs", "runs"]) == 0
        [line] = capsys.readouterr().out.strip().splitlines()
        assert Outcome.from_dict(json.loads(line)).needs == "ladder"
        assert rating == [(digests["dull"], False)]


# ------------------------------------------------------------------ the real machine


class TestTheRealMachine:
    def test_a_ladder_runs_in_the_worktree_into_main_runs(self, repo) -> None:  # noqa: F811
        seen: list[list[str]] = []

        def run(argv, **kwargs):
            seen.append(list(argv))
            if argv[:2] == ["git", "rev-parse"]:
                return subprocess.CompletedProcess(argv, 0, "c1\n", "")
            if "--plan" in argv:
                row = Item(hash="h", name="h", needs="top-up", reason="", tier=3,
                           measured="", stored="ladder:x").as_dict()
                return subprocess.CompletedProcess(argv, 0, json.dumps([row]), "")
            return subprocess.CompletedProcess(argv, 0, json.dumps(
                Outcome(hash="h", needs="top-up").as_dict()), "")

        machine = replace(repo, run=run)
        [found] = machine.ladders()
        assert found.ladder and found.lease == "heavy"
        outcome = machine.repair(found)
        assert not outcome.refused and outcome.commit == "c1"
        survey, rate = [argv for argv in seen if argv[0] != "git"]
        assert Path(survey[0]) == refresher._python(repo.worktree)
        assert survey[3:7] == ["refresh", "--plan", "--part", "ladders"]
        assert rate[3:6] == ["refresh", "h", "--ladder"]
        # Nothing staged: a ladder is rated in place in the main checkout's runs/.
        assert rate[rate.index("--runs") + 1] == str(repo.main_runs)
        assert not (repo.worktree / "runs").exists()

    def test_the_score_survey_reads_the_scores_alone(self, repo) -> None:  # noqa: F811
        seen: list[list[str]] = []

        def run(argv, **kwargs):
            seen.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, "[]", "")

        assert replace(repo, run=run).survey() == []
        assert seen[0][3:7] == ["refresh", "--plan", "--part", "scores"]


def test_ladders_check_does_not_list_a_preview_as_owing(
    tmp_path: Path, checks: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q934d: `ladders_check` reports a library game without a ladder, never a preview."""
    digests = tree(tmp_path, dull="dull")
    spec = json.loads((REPOSITORY / "rulesets" / "gardner.json").read_text("utf-8"))
    spec["provenance"] = {**(spec.get("provenance") or {}), "preview": "A preview."}
    (tmp_path / "archive" / "gardner-preview.json").write_text(
        json.dumps(spec), encoding="utf-8")

    assert checks.main(["--root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "dull" in out and "missing" in out
    assert "gardner" not in out.lower()
    assert "0 of 1 library games have a picker ladder; 1 needs one" in out

    stamped(tmp_path / "runs", digests["dull"], OFFERED, RECENT)
    assert checks.main(["--root", str(tmp_path)]) == 0


def test_ladders_check_is_the_one_ladders_py_reads() -> None:
    """`ladders.py` imports `ladders_check` by its plain name; the refresher must read the
    same module, or a stand-in for its roster reaches one and not the other."""
    refresher._deploy("ladders")
    assert refresher._deploy("ladders_check") is sys.modules["ladders_check"]
    assert sys.modules["ladders"].ladder is sys.modules["ladders_check"].ladder
