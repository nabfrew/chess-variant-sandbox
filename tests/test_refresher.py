"""The refresher (EV5, `docs/archive/plan-evolving-score.md` §6; `chess_variant_sandbox/refresher.py`).

What these hold down, with fakes and fixture runs (nothing plays: `measure`'s expensive
calls are `test_history`'s stubs, on `test_corpus`'s five-square game):

- **The order** (§6.2): cheapest repair first, then the launch set, the top quarter, a game
  on a gate's edge, the oldest estimate and the hash; on a fixture library through the real
  `corpus.survey`.
- **One game**: `refresh <hash>` repairs a stale game as `measure` would, and refuses a
  stale build, a fit that does not reproduce, a gate the repair newly fails and a repair
  that raised.
- **The loop**, over a fake `Machine`: one game per lease, `deliver` after each, no game
  spanning two commits, a refusal skipping one game and never retried, a stale build
  halting until main moves, waiting while another heavy job holds the machine, a rebuild
  when `rust/` moved, the status file.
- **The order never changes a number**: two games repaired in either order give the same
  files.
- **The real machine's parts that need no core**: fast-forward and merge in a real git
  repository, the porcelain reading, staging, delivery through `history.deliver`, the
  lease status reading, a worktree that moved under a repair.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_corpus import SPEC, world  # noqa: F401  (`world` is a fixture)
from test_history import _game_on_disk, stubbed  # noqa: F401  (`stubbed` is a fixture)

from chess_variant_sandbox import corpus, history, measurement, refresher
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS
from chess_variant_sandbox.refresher import Item, Outcome, Stored
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.tournament import fitting

# ------------------------------------------------------------------ fixtures


def status(digest: str, needs: str, name: str = "", current: bool = False) -> Any:
    return SimpleNamespace(hash=digest, name=name or digest, needs="" if current else needs,
                           reason=f"stale ({needs})", current=current)


def item(digest: str, needs: str = "rescore", stored: str = "s0", name: str = "") -> Item:
    return Item(hash=digest, name=name or digest, needs=needs, reason="", tier=3,
                measured="", stored=stored)


def library_game(root: Path, spec: dict[str, Any], index: float = 0.8,
                 viable: bool = True, **moved: Any) -> str:
    """An archived game under ``root``'s `archive/` with a four-run measurement in
    ``root``'s `runs/`, its runs on disk, stamped with today's basis and ``moved``."""
    from chess_variant_sandbox.tournament.analysis import CorpusAnalysis

    archive = root / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    digest = spec_hash(spec)
    (archive / f"{spec['name'].replace(' ', '-')}.json").write_text(json.dumps(spec),
                                                                    encoding="utf-8")
    runs = root / "runs"
    seeds = corpus.LIBRARY_SEEDS
    basis = {**corpus.current_basis(digest, spec=spec, seeds=seeds).as_dict(), **moved}
    out = corpus.measurement_dir(digest, runs)
    out.mkdir(parents=True, exist_ok=True)
    (out / "canonical.json").write_text(json.dumps({
        "hash": digest, "name": spec["name"], "agents": list(corpus.CANONICAL_AGENTS),
        "games_per_pairing": corpus.CANONICAL_GAMES, "seeds": list(seeds), "games": 1024,
        "index": index, "totals_used": 4, "sd": 0.02, "ci_low": index - 0.04,
        "ci_high": index + 0.04, "viable": viable, "violations": [],
        "verdict": {"rule": "x", "accept": 4 if viable else 0, "refuse": 0 if viable else 4},
        "weights_version": basis["weights_version"], "families": {}, "standings": [],
        "runs": [{"seed": seed, "index": index, "sd": 0.02, "viable": viable,
                  "violations": [], "unreadable": [], "origin": "played"} for seed in seeds],
        "basis": basis,
    }), encoding="utf-8")
    (out / f"index-{DEFAULT_WEIGHTS}.json").write_text("{}", encoding="utf-8")
    for seed in seeds:
        run = corpus.run_dir(digest, seed, runs)
        run.mkdir(parents=True, exist_ok=True)
        _game_on_disk(run, seed, "old")
        (run / "tournament.json").write_text(json.dumps({
            "config": {"agents": list(corpus.CANONICAL_AGENTS),
                       "games_per_pairing": corpus.CANONICAL_GAMES, "seed": seed},
            "games": 4, "ratings": {"standings": []},
        }), encoding="utf-8")
        CorpusAnalysis().save(run / "analysis.json")
    return digest


SECOND = {**SPEC, "name": "lancers two",
          "rules": {**SPEC["rules"], "max_plies": 100}}


@pytest.fixture
def library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubbed) -> Path:  # noqa: F811
    """A tree whose `rulesets/` is empty and whose `archive/` and `runs/` are the
    fixture's, as the refresher's worktree is; `corpus` reads the relative defaults."""
    (tmp_path / "rulesets").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(measurement, "stale_build", lambda: None)
    return tmp_path


# ------------------------------------------------------------------ the order


class TestTheOrder:
    def test_the_cheapest_repair_comes_first_whatever_the_value(self) -> None:
        stored = {"a": Stored(index=0.9), "b": Stored(), "c": Stored(), "d": Stored()}
        queue = refresher.order(
            [status("a", "replay", name="dobutsu"), status("b", "reanalyse"),
             status("c", "rescore"), status("d", "refit")], stored)
        assert [i.needs for i in queue] == ["refit", "rescore", "reanalyse", "replay"]

    def test_within_a_repair_value_then_age_then_hash(self) -> None:
        stored = {
            "e1": Stored(index=0.1, measured="2026-09-01"),
            "e2": Stored(index=0.2, measured="2026-08-01"),
            "e3": Stored(index=0.2, measured="2026-08-01"),
            "edge": Stored(index=0.3, split=True, measured="2026-10-01"),
            "top": Stored(index=0.95, measured="2026-10-01"),
            "launch": Stored(index=0.0),
            "gone": Stored(index=0.99, viable=False, measured="2026-10-01"),
            "current": Stored(index=0.97),
        }
        queue = refresher.order(
            [status("e1", "rescore"), status("e3", "rescore"), status("e2", "rescore"),
             status("edge", "rescore"), status("top", "rescore"),
             status("launch", "rescore", name="Three-Check"),
             status("gone", "rescore"), status("current", "", current=True)], stored)
        # Eight games hold a number and seven were not gated out: the top quarter is two,
        # `current` (not owed) and `top`.
        assert [i.hash for i in queue] == ["launch", "top", "edge", "e2", "e3", "e1", "gone"]
        assert [i.tier for i in queue] == [0, 1, 2, 3, 3, 3, 3]

    def test_a_game_never_measured_sorts_as_the_oldest(self) -> None:
        queue = refresher.order([status("b", "replay"), status("a", "replay")],
                                {"b": Stored(measured="2026-01-01"), "a": Stored()})
        assert [i.hash for i in queue] == ["a", "b"]
        assert queue[0].stored == "absent"

    def test_each_repair_takes_its_lease(self) -> None:
        assert {needs: item("x", needs).lease for needs in refresher.REPAIR_ORDER} == {
            "refit": "heavy", "rescore": "quiet", "reanalyse": "heavy", "replay": "heavy"}

    def test_the_launch_set_names_games_the_library_has(self) -> None:
        names = {str(json.loads(path.read_text(encoding="utf-8")).get("name"))
                 for directory in (Path("rulesets"), Path("archive"))
                 for path in (Path(__file__).resolve().parents[1] / directory).glob("*.json")}
        assert set(refresher.LAUNCH_SET) <= names

    def test_on_a_fixture_library_through_the_survey(self, library: Path) -> None:
        stale = library_game(library, SPEC, metrics="an earlier metrics source")
        current = library_game(library, SECOND)
        queue = refresher.queue(Path("runs"))
        assert [(i.hash, i.needs) for i in queue] == [(stale, "rescore")]
        assert queue[0].stored == refresher.stored_of(stale, Path("runs")).digest
        assert current not in {i.hash for i in queue}


# ------------------------------------------------------------------ one game


class TestOneGame:
    def test_a_stale_game_is_repaired_as_measure_would(self, library: Path) -> None:
        digest = library_game(library, SPEC, metrics="an earlier metrics source")
        outcome = refresher.repair(digest, Path("runs"))
        assert not outcome.refused, outcome
        assert outcome.needs == "rescore" and outcome.index == pytest.approx(0.6)
        assert refresher.queue(Path("runs")) == []
        # `corpus` kept what it replaced, in the tree it measured in.
        assert [line.needs for line in history.read(digest, Path("runs"))] == ["rescore"]

    def test_a_stale_build_is_refused_before_anything_runs(
        self, library: Path, monkeypatch: pytest.MonkeyPatch, stubbed  # noqa: F811
    ) -> None:
        digest = library_game(library, SPEC, metrics="an earlier metrics source")
        monkeypatch.setattr(measurement, "stale_build", lambda: "the core is behind")
        outcome = refresher.repair(digest, Path("runs"))
        assert outcome.refusal == refresher.STALE_BUILD
        assert stubbed["score"] == []

    @pytest.mark.parametrize("found", [fitting.REFIT, fitting.FITTED])
    def test_a_fit_that_does_not_reproduce_is_refused_before_a_game_is_played(
        self, library: Path, monkeypatch: pytest.MonkeyPatch, found: str
    ) -> None:
        digest = library_game(library, SPEC)
        owed = SimpleNamespace(current=False, stored={},
                               staleness=SimpleNamespace(refit=True, needs="refit"))
        monkeypatch.setattr(corpus, "read_measurement", lambda *a, **k: owed)
        monkeypatch.setattr(corpus, "check_fit", lambda spec, workers=0, **_ignored: SimpleNamespace(
            outcome=found, failures=("weights moved by 0.3",)))
        monkeypatch.setattr(corpus, "refresh", lambda *a, **k: pytest.fail("it ran"))
        outcome = refresher.repair(digest, Path("runs"))
        assert outcome.refusal == refresher.FIT_NOT_REPRODUCED and outcome.needs == "refit"

    def test_a_gate_the_repair_newly_fails_is_refused(
        self, library: Path, monkeypatch: pytest.MonkeyPatch, stubbed  # noqa: F811
    ) -> None:
        from test_corpus import _Scored

        digest = library_game(library, SPEC, metrics="an earlier metrics source")
        monkeypatch.setattr(corpus, "score_ruleset", lambda *a, analysis=None, **k: _Scored(
            analysis=analysis, viable=False, violations=("decisive_rate=0.1 below 0.2",)))
        outcome = refresher.repair(digest, Path("runs"))
        assert outcome.refusal == refresher.GATE_NEWLY_FAILED
        assert "decisive_rate" in outcome.reason

    def test_a_game_already_failing_its_gate_is_no_refusal(
        self, library: Path, monkeypatch: pytest.MonkeyPatch, stubbed  # noqa: F811
    ) -> None:
        from test_corpus import _Scored

        digest = library_game(library, SPEC, viable=False, metrics="earlier")
        monkeypatch.setattr(corpus, "score_ruleset", lambda *a, analysis=None, **k: _Scored(
            analysis=analysis, viable=False, violations=("decisive_rate=0.1 below 0.2",)))
        assert not refresher.repair(digest, Path("runs")).refused

    def test_a_repair_that_raises_is_a_refusal_not_a_crash(
        self, library: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        digest = library_game(library, SPEC, metrics="earlier")

        def broken(*_a: Any, **_k: Any) -> Any:
            raise OSError("disk full")

        monkeypatch.setattr(corpus, "refresh", broken)
        outcome = refresher.repair(digest, Path("runs"))
        assert outcome.refusal == refresher.ERROR and "disk full" in outcome.reason
        assert outcome.needs == "rescore"

    def test_the_outcome_survives_the_process_boundary(self) -> None:
        outcome = Outcome(hash="h", needs="replay", refusal="", index=0.5, viable=True,
                          commit="c", seconds=1.5, lines=("a", "b"))
        assert Outcome.from_dict(json.loads(json.dumps(outcome.as_dict()))) == outcome
        it = item("h", "replay")
        assert Item.from_dict(json.loads(json.dumps(it.as_dict()))) == it


# ------------------------------------------------------------------ the loop


@dataclass
class FakeMachine:
    """A `Machine` whose queue is a dict of game to (needs, stored digest), whose main
    can be moved by a test, and which writes down everything the loop asks of it."""

    owed: dict[str, tuple[str, str]] = field(default_factory=dict)
    refuse: dict[str, str] = field(default_factory=dict)
    commit: str = "c1"
    main_commit: str = "c1"
    moved_paths: list[str] = field(default_factory=lambda: ["chess_variant_sandbox/x.py"])
    busy_with: str | None = None
    grant: bool = True
    rebuild_says: str | None = None
    changes: list[str] = field(default_factory=list)
    during_repair: Any = None
    calls: list[tuple[str, ...]] = field(default_factory=list)
    holding: str | None = None
    # Commits main holds, for `unmerged`; and a Ctrl-C at the first sleep.
    merged: set[str] = field(default_factory=set)
    interrupt_on_sleep: bool = False
    # What `slow_tier` returns (a green run by default), or raises; and the files it leaves
    # changed (`test_refresher_slow_tier.py`).
    tier: Any = None
    tier_changes: list[str] = field(default_factory=list)
    # The ladder work (L7, `test_refresher_ladders.py`): game to (needs, stored digest).
    ladders_owed: dict[str, tuple[str, str]] = field(default_factory=dict)

    def head(self) -> str:
        return self.commit

    def behind(self) -> bool:
        return self.main_commit != self.commit

    def advance(self) -> list[str]:
        assert self.holding is None, "advanced while a game held its lease"
        self.calls.append(("advance", self.main_commit))
        self.commit = self.main_commit
        return list(self.moved_paths)

    def rebuild(self) -> str | None:
        assert self.holding == "build"
        self.calls.append(("rebuild",))
        return self.rebuild_says

    def busy(self) -> str | None:
        return self.busy_with

    def acquire(self, profile: str, tag: str) -> bool:
        assert self.holding is None, "two leases at once"
        if not self.grant:
            return False
        self.holding = profile
        self.calls.append(("acquire", profile))
        return True

    def release(self) -> None:
        self.calls.append(("release", self.holding or ""))
        self.holding = None

    def survey(self) -> list[Item]:
        self.calls.append(("survey", self.commit))
        rows = [status(h, needs) for h, (needs, _digest) in self.owed.items()]
        stored = {h: Stored(digest=digest) for h, (_needs, digest) in self.owed.items()}
        return refresher.order(rows, stored)

    def ladders(self) -> list[Item]:
        self.calls.append(("ladders", self.commit))
        return sorted((Item(hash=h, name=h, needs=needs, reason="", tier=3, measured="",
                            stored=digest)
                       for h, (needs, digest) in self.ladders_owed.items()),
                      key=lambda it: it.key)

    def repair(self, it: Item) -> Outcome:
        assert self.holding == it.lease
        start = self.commit
        self.calls.append(("repair", it.hash, start))
        if self.during_repair is not None:
            self.during_repair(self)
        assert self.commit == start, "the worktree moved under a repair"
        if it.hash in self.refuse:
            return Outcome(hash=it.hash, needs=it.needs, refusal=self.refuse[it.hash],
                           reason="because", commit=start)
        if it.ladder:
            # Rated in place: nothing is delivered.
            self.ladders_owed.pop(it.hash, None)
            return Outcome(hash=it.hash, needs=it.needs, commit=start)
        return Outcome(hash=it.hash, needs=it.needs, index=0.5, viable=True, commit=start)

    def deliver(self, it: Item) -> None:
        assert self.holding is None
        self.calls.append(("deliver", it.hash))
        self.owed.pop(it.hash, None)

    def changed(self) -> list[str]:
        return list(self.changes)

    def keep(self, it: Item, paths: list[str]) -> str:
        self.calls.append(("keep", it.hash, *paths))
        self.changes = []
        return "k1"

    def discard(self, paths: list[str]) -> None:
        self.calls.append(("discard", *paths))
        self.changes = []

    def unmerged(self, commits: list[str]) -> list[str]:
        return [commit for commit in commits if commit not in self.merged]

    def slow_tier(self) -> refresher.SlowTier:
        assert self.holding == refresher.SLOW_TIER_LEASE
        self.calls.append(("slow tier", self.commit))
        self.changes = [*self.changes, *self.tier_changes]
        if isinstance(self.tier, BaseException):
            raise self.tier
        return self.tier or refresher.SlowTier(commit=self.commit, passed=3)

    def sleep(self, seconds: float) -> None:
        self.calls.append(("sleep",))
        if self.interrupt_on_sleep:
            raise KeyboardInterrupt

    def did(self, kind: str) -> list[tuple[str, ...]]:
        return [call for call in self.calls if call[0] == kind]


def loop(machine: FakeMachine, runs: Path, watch: bool = False,
         passes: int | None = 20) -> refresher.Refresher:
    return refresher.Refresher(machine, refresher.Ledger(runs), watch=watch, passes=passes)


class TestTheLoop:
    def test_one_game_per_lease_and_a_delivery_after_each(self, tmp_path: Path) -> None:
        machine = FakeMachine(owed={"a": ("replay", "s"), "b": ("rescore", "s")})
        assert loop(machine, tmp_path).run() == 0
        work = [c for c in machine.calls if c[0] in ("acquire", "repair", "release", "deliver")]
        assert work == [
            ("acquire", "quiet"), ("repair", "b", "c1"), ("release", "quiet"), ("deliver", "b"),
            ("acquire", "heavy"), ("repair", "a", "c1"), ("release", "heavy"), ("deliver", "a"),
        ]

    def test_no_game_spans_two_commits(self, tmp_path: Path) -> None:
        def move_main(machine: FakeMachine) -> None:
            machine.main_commit = "c2"

        machine = FakeMachine(owed={"a": ("rescore", "s"), "b": ("rescore", "s")},
                              during_repair=move_main)
        loop(machine, tmp_path).run()
        order = [c for c in machine.calls if c[0] in ("repair", "deliver", "advance")]
        # `a` is repaired and delivered at c1; only then does the worktree move.
        assert order == [("repair", "a", "c1"), ("deliver", "a"), ("advance", "c2"),
                         ("repair", "b", "c2"), ("deliver", "b")]

    def test_a_refusal_skips_the_game_and_is_never_retried(self, tmp_path: Path) -> None:
        machine = FakeMachine(owed={"a": ("rescore", "s"), "b": ("replay", "s")},
                              refuse={"a": refresher.GATE_NEWLY_FAILED})
        assert loop(machine, tmp_path).run() == 0
        assert machine.did("repair") == [("repair", "a", "c1"), ("repair", "b", "c1")]
        assert machine.did("deliver") == [("deliver", "b")]
        status = refresher.read_status(tmp_path)
        assert status is not None and status["state"] == "stopped"
        assert status["last_refusal"]["hash"] == "a"
        assert [r["hash"] for r in status["refusals"]] == ["a"]
        assert status["refused_owed"] == 1
        events = [line["event"] for line in refresher.read_log(tmp_path)]
        assert events == ["refused", "repaired"]
        # `refresh --status` says the refused game is still owed, as `measure --check`
        # does, rather than "owed: nothing" (Q693r's first real run).
        assert any("owed: nothing (1 more owed and refused)" in line
                   for line in refresher.plain_status(status))

        # A second run finds `a` still owed with the same stored file and does not touch it.
        again = loop(machine, tmp_path)
        assert again.run() == 0
        assert len(machine.did("repair")) == 2

        # Somebody else repairs it (its stored file changes): it is a new question, and
        # the old refusal lapses (Q693r's first real run); this fake refuses it again, of
        # its new file.
        machine.owed["a"] = ("rescore", "s-new")
        loop(machine, tmp_path).run()
        assert machine.did("repair")[-1] == ("repair", "a", "c1")
        status = refresher.read_status(tmp_path)
        assert status is not None
        assert [(r["hash"], r["stored"]) for r in status["refusals"]] == [("a", "s-new")]
        assert any(line["event"] == "lapsed" and line["hash"] == "a"
                   for line in refresher.read_log(tmp_path))

    def test_a_refusal_of_a_game_owed_nothing_lapses(self, tmp_path: Path) -> None:
        machine = FakeMachine(owed={"a": ("rescore", "s")},
                              refuse={"a": refresher.GATE_NEWLY_FAILED})
        loop(machine, tmp_path).run()
        assert len(refresher.read_status(tmp_path)["refusals"]) == 1  # type: ignore[index]
        machine.owed = {}  # somebody repaired it by hand: it is current
        loop(machine, tmp_path).run()
        assert refresher.read_status(tmp_path)["refusals"] == []  # type: ignore[index]

    def test_only_never_lapses_a_refusal_it_did_not_look_at(self, tmp_path: Path) -> None:
        machine = FakeMachine(owed={"a": ("rescore", "s"), "b": ("rescore", "s")},
                              refuse={"a": refresher.GATE_NEWLY_FAILED})
        loop(machine, tmp_path).run()
        refresher.Refresher(machine, refresher.Ledger(tmp_path), passes=3,
                            only=frozenset({"b"})).run()
        assert [r["hash"] for r in refresher.read_status(tmp_path)["refusals"]] == ["a"]  # type: ignore[index]

    def test_ctrl_c_stops_cleanly_and_is_logged(self, tmp_path: Path) -> None:
        machine = FakeMachine(interrupt_on_sleep=True)
        assert loop(machine, tmp_path, watch=True).run() == 0
        assert refresher.read_status(tmp_path)["state"] == "stopped"  # type: ignore[index]
        assert refresher.read_log(tmp_path)[-1]["event"] == "interrupted"
        assert machine.holding is None

    def test_forgetting_a_refusal_lets_it_be_taken_again(self, tmp_path: Path) -> None:
        machine = FakeMachine(owed={"a": ("rescore", "s")}, refuse={"a": refresher.ERROR})
        loop(machine, tmp_path).run()
        assert refresher.Ledger(tmp_path).forget("a")
        machine.refuse = {}
        loop(machine, tmp_path).run()
        assert machine.did("deliver") == [("deliver", "a")]

    def test_a_stale_build_halts_the_loop_until_main_moves(self, tmp_path: Path) -> None:
        machine = FakeMachine(owed={"a": ("rescore", "s"), "b": ("rescore", "s")},
                              refuse={"a": refresher.STALE_BUILD})
        assert loop(machine, tmp_path).run() == 1
        assert machine.did("repair") == [("repair", "a", "c1")]
        status = refresher.read_status(tmp_path)
        assert status is not None and status["last_refusal"]["kind"] == refresher.STALE_BUILD
        # Not a refusal of the game: it is the build's.
        assert status["refusals"] == []

        watching = loop(machine, tmp_path, watch=True, passes=3)
        machine.refuse = {}
        watching.ledger.status["state"] = "halted"
        watching.step()
        assert len(machine.did("repair")) == 1, "halted: nothing runs while main stands"
        machine.main_commit = "c2"
        watching.step()
        assert machine.did("rebuild") == [("rebuild",)]
        assert machine.did("repair")[-1] == ("repair", "a", "c2")

    def test_it_waits_while_another_job_holds_the_machine(self, tmp_path: Path) -> None:
        machine = FakeMachine(owed={"a": ("rescore", "s")},
                              busy_with="a heavy lease is held by T5-g2")
        loop(machine, tmp_path, watch=True, passes=3).run()
        assert machine.did("acquire") == [] and machine.did("repair") == []
        assert len(machine.did("sleep")) == 3
        machine.busy_with = None
        loop(machine, tmp_path, passes=5).run()
        assert machine.did("deliver") == [("deliver", "a")]

    def test_a_lease_not_granted_is_waited_for_not_taken(self, tmp_path: Path) -> None:
        machine = FakeMachine(owed={"a": ("replay", "s")}, grant=False)
        refresher.Refresher(machine, refresher.Ledger(tmp_path), passes=2).run()
        assert machine.did("repair") == [] and len(machine.did("sleep")) == 2
        assert refresher.read_status(tmp_path)["waiting_for"] == "a heavy lease"  # type: ignore[index]

    def test_a_move_of_rust_rebuilds_under_a_build_lease(self, tmp_path: Path) -> None:
        machine = FakeMachine(owed={"a": ("replay", "s")}, main_commit="c2",
                              moved_paths=["rust/src/lib.rs"])
        loop(machine, tmp_path).run()
        assert machine.calls[:5] == [("advance", "c2"), ("acquire", "build"), ("rebuild",),
                                     ("release", "build"), ("survey", "c2")]
        # The log says the core was rebuilt: nothing else does (Q693r's first real run).
        events = [line["event"] for line in refresher.read_log(tmp_path)]
        assert events[:2] == ["advanced", "rebuilt"]

    def test_a_move_without_rust_rebuilds_nothing(self, tmp_path: Path) -> None:
        machine = FakeMachine(owed={"a": ("replay", "s")}, main_commit="c2")
        loop(machine, tmp_path).run()
        assert machine.did("rebuild") == []

    def test_a_failed_rebuild_halts(self, tmp_path: Path) -> None:
        machine = FakeMachine(owed={"a": ("replay", "s")}, main_commit="c2",
                              moved_paths=["rust/src/lib.rs"], rebuild_says="cargo failed")
        assert loop(machine, tmp_path).run() == 1
        assert machine.did("repair") == []

    def test_fit_evidence_is_kept_and_anything_else_refused(self, tmp_path: Path) -> None:
        machine = FakeMachine(owed={"a": ("refit", "s")},
                              changes=["fitted/reproduced/x.json"])
        loop(machine, tmp_path).run()
        assert machine.did("keep") == [("keep", "a", "fitted/reproduced/x.json")]
        assert machine.did("deliver") == [("deliver", "a")]
        assert refresher.read_status(tmp_path)["owed_merge"] == ["k1"]  # type: ignore[index]

        # Once main holds the fit commit it is owed no merge, and the log says so
        # (Q693r's first real run: the list only ever grew).
        machine.merged = {"k1"}
        loop(machine, tmp_path).run()
        assert refresher.read_status(tmp_path)["owed_merge"] == []  # type: ignore[index]
        assert any(line["event"] == "merged" and line["commits"] == ["k1"]
                   for line in refresher.read_log(tmp_path))

        other = FakeMachine(owed={"b": ("rescore", "s")}, changes=["rulesets/chess.json"])
        loop(other, tmp_path / "other").run()
        assert other.did("deliver") == [] and other.did("discard") == [
            ("discard", "rulesets/chess.json")]
        refusal = refresher.read_status(tmp_path / "other")["last_refusal"]  # type: ignore[index]
        assert refusal["kind"] == refresher.CHANGED_COMMITTED_FILES

    def test_a_refused_repair_puts_back_what_it_changed(self, tmp_path: Path) -> None:
        machine = FakeMachine(owed={"a": ("refit", "s")}, changes=["fitted/a.json"],
                              refuse={"a": refresher.FIT_NOT_REPRODUCED})
        loop(machine, tmp_path).run()
        assert machine.did("discard") == [("discard", "fitted/a.json")]
        assert machine.did("keep") == [] and machine.did("deliver") == []

    def test_the_status_file_says_what_is_owed_and_what_was_done(self, tmp_path: Path) -> None:
        machine = FakeMachine(owed={"a": ("rescore", "s"), "b": ("replay", "s")})
        looping = loop(machine, tmp_path)
        looping.step()
        status = refresher.read_status(tmp_path)
        assert status is not None
        assert status["owed_counts"] == {"refit": 0, "rescore": 1, "reanalyse": 0, "replay": 1}
        assert [row["hash"] for row in status["owed"]] == ["a", "b"]
        assert status["last_done"]["hash"] == "a" and status["commit"] == "c1"
        assert isinstance(status["age_s"], int) and status["age_s"] >= 0
        assert any("owed: 1 rescore, 1 replay" in line
                   for line in refresher.plain_status(status))

    def test_a_failed_survey_or_delivery_halts_and_is_logged(self, tmp_path: Path) -> None:
        machine = FakeMachine(owed={"a": ("rescore", "s")})

        def broken(*_a: Any) -> Any:
            raise RuntimeError("no interpreter")

        machine.survey = broken  # type: ignore[method-assign]
        assert loop(machine, tmp_path).run() == 1
        assert refresher.read_status(tmp_path)["last_refusal"]["kind"] == refresher.ERROR  # type: ignore[index]

        other = FakeMachine(owed={"a": ("rescore", "s"), "b": ("rescore", "s")})
        other.deliver = broken  # type: ignore[method-assign]
        assert loop(other, tmp_path / "other").run() == 1
        assert other.did("repair") == [("repair", "a", "c1")]
        assert [line["event"] for line in refresher.read_log(tmp_path / "other")] == ["halted"]

    def test_only_takes_the_games_it_names(self, tmp_path: Path) -> None:
        machine = FakeMachine(owed={"a": ("rescore", "s"), "b": ("rescore", "s")})
        refresher.Refresher(machine, refresher.Ledger(tmp_path), passes=5,
                            only=frozenset({"b"})).run()
        assert machine.did("deliver") == [("deliver", "b")]

    def test_nothing_owed_is_idle_and_a_watch_sleeps(self, tmp_path: Path) -> None:
        machine = FakeMachine()
        assert loop(machine, tmp_path).run() == 0
        assert machine.did("sleep") == []
        loop(machine, tmp_path, watch=True, passes=2).run()
        assert len(machine.did("sleep")) == 2


# ------------------------------------------------------------------ the order moves no number


class TestTheOrderNeverChangesANumber:
    def test_two_orders_write_the_same_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubbed  # noqa: F811
    ) -> None:
        from test_corpus import _Scored

        def score(spec, records, ratings, weights, analysis=None, **_ignored):
            # A number that depends on the game and its records, never on what ran before.
            seeds = sorted({record.seed for record in records})
            headline = (int(spec_hash(spec)[:4], 16) % 97) / 100 + sum(seeds) / 1e6
            return _Scored(analysis=analysis, weights_version=weights.version,
                           headline=headline)

        monkeypatch.setattr(corpus, "score_ruleset", score)
        monkeypatch.setattr(measurement, "stale_build", lambda: None)
        files: list[dict[str, bytes]] = []
        for name, order in (("one", (SPEC, SECOND)), ("two", (SECOND, SPEC))):
            root = tmp_path / name
            (root / "rulesets").mkdir(parents=True)
            digests = [library_game(root, spec, metrics="earlier") for spec in (SPEC, SECOND)]
            monkeypatch.chdir(root)
            for spec in order:
                assert not refresher.repair(spec_hash(spec), Path("runs")).refused
            files.append({
                f"{digest}/{path.name}": path.read_bytes()
                for digest in digests
                for path in sorted((Path("runs") / digest).iterdir())
                if path.is_file() and path.name != history.HISTORY_FILE
            })
        assert files[0] == files[1]
        assert len(files[0]) == 4


# ------------------------------------------------------------------ the real machine


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> refresher.GitMachine:
    """A main checkout on `main` with a refresher worktree beside it."""
    config = tmp_path / "gitconfig"
    config.write_text("[user]\n\tname = t\n\temail = t@t\n[commit]\n\tgpgsign = false\n",
                      encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    main = tmp_path / "main"
    main.mkdir()
    git(main, "init", "-q", "-b", "main")
    (main / ".gitignore").write_text("runs/\n.claude/worktrees/\n", encoding="utf-8")
    (main / "fitted").mkdir()
    (main / "fitted" / "a.json").write_text("{}", encoding="utf-8")
    git(main, "add", "-A")
    git(main, "commit", "-qm", "one")
    worktree = main / ".claude" / "worktrees" / "refresher"
    git(main, "worktree", "add", "-q", "-b", refresher.BRANCH, str(worktree), "main")
    return refresher.GitMachine(main_checkout=main, worktree=worktree)


class TestTheRealMachine:
    def test_it_fast_forwards_to_main_and_says_what_moved(self, repo) -> None:
        main = repo.main_checkout
        (main / "rust").mkdir()
        (main / "rust" / "lib.rs").write_text("//", encoding="utf-8")
        git(main, "add", "-A")
        git(main, "commit", "-qm", "two")
        assert repo.behind()
        assert repo.advance() == ["rust/lib.rs"]
        assert not repo.behind()
        assert refresher.rebuild_needed(["rust/lib.rs"])
        assert not refresher.rebuild_needed(["chess_variant_sandbox/cli.py"])

    def test_its_own_fit_commits_are_merged_with_main_never_rewritten(self, repo) -> None:
        (repo.worktree / "fitted" / "b.json").write_text("{}", encoding="utf-8")
        assert repo.changed() == ["fitted/b.json"]
        kept = repo.keep(item("h", "refit", name="lancers"), ["fitted/b.json"])
        assert kept == repo.head() and repo.changed() == []
        (repo.main_checkout / "x.txt").write_text("x", encoding="utf-8")
        git(repo.main_checkout, "add", "-A")
        git(repo.main_checkout, "commit", "-qm", "three")
        assert repo.behind()
        assert repo.advance() == ["x.txt"]
        # Ahead of main by its own commit and the merge, and not behind it.
        assert not repo.behind()
        log = git(repo.worktree, "rev-list", "HEAD")
        assert kept in log and git(repo.main_checkout, "rev-parse", "main").strip() in log

        # Owed a merge until main holds it; a commit the branch does not hold (unknown to
        # git, or main's own) is owed nothing.
        main_head = git(repo.main_checkout, "rev-parse", "main").strip()
        assert repo.unmerged([kept, "0" * 40, main_head]) == [kept]
        git(repo.main_checkout, "merge", "-q", "--no-edit", refresher.BRANCH)
        assert repo.unmerged([kept]) == []

    def test_the_porcelain_is_read_whole_and_a_discard_puts_it_back(self, repo) -> None:
        (repo.worktree / "fitted" / "a.json").write_text('{"x": 1}', encoding="utf-8")
        (repo.worktree / "fitted" / "new.json").write_text("{}", encoding="utf-8")
        (repo.worktree / "runs").mkdir()
        (repo.worktree / "runs" / "ignored.json").write_text("{}", encoding="utf-8")
        # The first line starts with a space (" M"); a stripped read would cut its name.
        assert repo.changed() == ["fitted/a.json", "fitted/new.json"]
        repo.discard(repo.changed())
        assert repo.changed() == []
        assert (repo.worktree / "fitted" / "a.json").read_text(encoding="utf-8") == "{}"

    def test_a_worktree_that_moves_under_a_repair_is_refused(self, repo) -> None:
        heads = iter(["c1", "c2"])

        def run(argv, **kwargs):
            if argv[:2] == ["git", "rev-parse"]:
                return subprocess.CompletedProcess(argv, 0, next(heads), "")
            outcome = Outcome(hash="h", needs="rescore", index=0.5)
            return subprocess.CompletedProcess(argv, 0, json.dumps(outcome.as_dict()), "")

        machine = replace(repo, run=run)
        outcome = machine.repair(item("h"))
        assert outcome.refusal == refresher.COMMIT_MOVED

    def test_a_repair_runs_the_worktrees_own_python_on_its_staging_runs(self, repo) -> None:
        seen: list[list[str]] = []

        def run(argv, **kwargs):
            seen.append(list(argv))
            if argv[:2] == ["git", "rev-parse"]:
                return subprocess.CompletedProcess(argv, 0, "c1\n", "")
            return subprocess.CompletedProcess(argv, 0, "noise\n" + json.dumps(
                Outcome(hash="h", needs="rescore", index=0.5).as_dict()), "")

        machine = replace(repo, run=run)
        outcome = machine.repair(item("h"))
        assert not outcome.refused and outcome.commit == "c1"
        [child] = [argv for argv in seen if argv[0] != "git"]
        assert Path(child[0]) == refresher._python(repo.worktree)
        assert child[1:5] == ["-m", "chess_variant_sandbox.cli", "refresh", "h"]
        assert child[child.index("--runs") + 1] == str(repo.worktree / "runs")

    def test_an_unreadable_repair_is_a_refusal(self, repo) -> None:
        def run(argv, **kwargs):
            out = "c1" if argv[0] == "git" else "Traceback ..."
            return subprocess.CompletedProcess(argv, 1, out, "boom")

        assert replace(repo, run=run).repair(item("h")).refusal == refresher.ERROR

    def test_it_yields_to_another_worktrees_heavy_or_quiet_lease(self, repo) -> None:
        def leases(*held: tuple[str, str]):
            payload = {"leases": [{"profile": p, "worktree": w, "tag": "t"} for p, w in held]}

            def run(argv, **kwargs):
                return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")

            return replace(repo, run=run)

        own = str(repo.worktree)
        assert leases(("heavy", "/x/T5")).busy() == "a heavy lease is held by T5 (t)"
        assert leases(("quiet", "/x/bench")).busy() is not None
        assert leases(("suite", "/x/a"), ("build", "/x/b")).busy() is None
        assert leases(("heavy", own)).busy() is None

    def test_the_lease_line_names_the_profile_and_waits(self, repo) -> None:
        seen: list[list[str]] = []

        def run(argv, **kwargs):
            seen.append(list(argv))
            assert kwargs["cwd"] == repo.worktree  # the lease is the worktree's
            return subprocess.CompletedProcess(argv, 0, "", "")

        machine = replace(repo, run=run, wait_s=60)
        assert machine.acquire("quiet", "refresh h")
        assert seen[0][0] == sys.executable
        assert seen[0][1].endswith("compute_lease.py")
        assert seen[0][2:] == ["acquire", "--profile", "quiet", "--tag", "refresh h",
                               "--wait", "60"]


def test_the_runs_named_are_the_runs_surveyed_and_delivered_into(repo, tmp_path) -> None:
    seen: list[list[str]] = []

    def run(argv, **kwargs):
        seen.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "[]", "")

    machine = replace(repo, run=run, runs_dir=tmp_path / "elsewhere")
    machine.survey()
    machine.deliver(item("h"))
    assert seen[0][seen[0].index("--runs") + 1] == str(tmp_path / "elsewhere")
    assert seen[1][-2:] == [str(tmp_path / "elsewhere"), "h"]


class TestStagingAndDelivery:
    def test_staging_copies_what_a_repair_reads_and_nothing_of_the_history(
        self, tmp_path: Path
    ) -> None:
        main = tmp_path / "main-runs"
        digest = library_game(tmp_path, SPEC)
        (tmp_path / "runs").rename(main)
        (main / digest / history.HISTORY_FILE).write_text("{}\n", encoding="utf-8")
        (main / digest / "previous" / "1").mkdir(parents=True)
        staging = tmp_path / "staging"
        (staging / f"{digest}-9").mkdir(parents=True)  # a leftover of an earlier repair
        copied = refresher.stage_game(digest, main, staging)
        assert f"{digest}/canonical.json" in copied
        assert sorted(p.name for p in staging.iterdir()) == sorted(
            [digest] + [f"{digest}-{seed}" for seed in corpus.LIBRARY_SEEDS])
        assert not (staging / digest / history.HISTORY_FILE).exists()
        assert not (staging / digest / "previous").exists()
        assert (staging / f"{digest}-0" / "games.jsonl").exists()

    def test_staged_repaired_and_delivered_the_main_runs_keep_what_was_replaced(
        self, library: Path
    ) -> None:
        digest = library_game(library, SPEC, metrics="an earlier metrics source")
        main = library / "runs"
        before = (main / digest / "canonical.json").read_bytes()
        staging = library / "staging"
        refresher.stage_game(digest, main, staging)
        assert not refresher.repair(digest, staging).refused
        assert history.deliver(staging, main, [digest]) == [digest]
        assert (history.previous_root(digest, main) / "1" / "canonical.json").read_bytes() \
            == before
        [line] = history.read(digest, main)
        assert line.needs == "rescore"
        assert refresher.queue(main) == []

    def test_delivery_is_history_deliver_in_the_worktree(self, repo) -> None:
        seen: list[list[str]] = []

        def run(argv, **kwargs):
            seen.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, "", "")

        replace(repo, run=run).deliver(item("h"))
        assert seen == [[str(refresher._python(repo.worktree)), "-m",
                         "chess_variant_sandbox.history", "deliver",
                         str(repo.worktree / "runs"), str(repo.main_checkout / "runs"), "h"]]


# ------------------------------------------------------------------ it moves nothing


class TestItMovesNothing:
    def test_it_is_in_no_fingerprint_set(self) -> None:
        ours = (measurement.ROOT / "chess_variant_sandbox/refresher.py").resolve()
        for kind in ("engine", "metrics", "analysis", "agents"):
            assert ours not in {p.resolve() for p in measurement.fingerprinted(kind)}, kind

    def test_no_metric_imports_it(self) -> None:
        for path in (measurement.ROOT / "chess_variant_sandbox" / "metrics").rglob("*.py"):
            assert "refresher" not in path.read_text(encoding="utf-8"), path


class TestTheCommand:
    def test_the_plan_is_the_queue_as_json(self, library: Path, capsys) -> None:
        from chess_variant_sandbox.cli import main

        digest = library_game(library, SPEC, metrics="earlier")
        assert main(["refresh", "--plan", "--part", "scores", "--json", "--runs", "runs"]) == 0
        [row] = json.loads(capsys.readouterr().out)
        assert row["hash"] == digest and row["needs"] == "rescore" and row["lease"] == "quiet"
        # The whole plan: the score first, then the game's missing ladder (L7).
        assert main(["refresh", "--plan", "--json", "--runs", "runs"]) == 0
        rows = json.loads(capsys.readouterr().out)
        assert [(r["hash"], r["needs"], r["lease"]) for r in rows] == [
            (digest, "rescore", "quiet"), (digest, "ladder", "heavy")]

    def test_one_game_as_json_is_one_line(self, library: Path, capsys) -> None:
        from chess_variant_sandbox.cli import main

        digest = library_game(library, SPEC, metrics="earlier")
        assert main(["refresh", digest, "--json", "--runs", "runs"]) == 0
        [line] = capsys.readouterr().out.strip().splitlines()
        assert Outcome.from_dict(json.loads(line)).needs == "rescore"

    def test_with_nothing_named_it_says_what_to_name(self, capsys) -> None:
        from chess_variant_sandbox.cli import main

        assert main(["refresh"]) == 2
        assert "--library" in capsys.readouterr().err

    def test_a_named_game_never_shares_a_line_with_a_supervisor_flag(self, capsys) -> None:
        # The lease hook reads `--plan`, `--forget`, `--status` and `--library` as needing
        # no lease, so a repair must never ride on one of them (Q693r).
        from chess_variant_sandbox.cli import main

        for flag in (["--plan"], ["--status"], ["--library"], ["--forget", "abc"]):
            assert main(["refresh", "abc123def456", *flag]) == 2
            assert "name no game" in capsys.readouterr().err
