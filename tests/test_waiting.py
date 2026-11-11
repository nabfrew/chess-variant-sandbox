"""The waiting schedule and `forced_repetition` (Q1184, `docs/plan-player-weaknesses.md` T3b).

The measurement reads records only, so most of this builds records by hand: what the
pre-registration says each shape of duel and control must read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import waiting
from chess_variant_sandbox.measurement import Basis
from chess_variant_sandbox.metrics.components import Components
from chess_variant_sandbox.metrics.index import gate_violations, readable_gate_violations
from chess_variant_sandbox.metrics.scoring import load_weights
from chess_variant_sandbox.metrics.waiting import (
    FORCED_REPETITION,
    WAITING_MAXIMUM,
    forced_repetition,
    waiting_violations,
)
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.record import GameRecord

RUNG = "minimax:nodes=25600"
CONTEMPT = f"{RUNG},contempt=1"


def _game(seed: int, agents: tuple[str, str], winner: int | None, reason: str) -> GameRecord:
    return GameRecord(
        ruleset_hash="h", ruleset_name="g", seed=seed, agents=agents, moves=[],
        winner=winner, reason=reason, plies=20,
    )


def _schedule(duel_winner_contempt: bool, control_repeats: int, control: int = 16) -> list[GameRecord]:
    """Eight seed-pairs (the contempt player at each seat) and a control of ``control`` games."""
    games = []
    for seed in range(500, 508):
        for agents in ((CONTEMPT, RUNG), (RUNG, CONTEMPT)):
            if duel_winner_contempt:
                games.append(_game(seed, agents, agents.index(CONTEMPT), "checkmate"))
            else:
                games.append(_game(seed, agents, None, "repetition"))
    for index in range(control):
        repeated = index < control_repeats
        games.append(_game(500 + index, (RUNG, RUNG), None if repeated else 0,
                           "repetition" if repeated else "checkmate"))
    return games


class TestForcedRepetition:
    def test_a_side_that_can_always_wait_reads_the_control_share(self) -> None:
        value, detail = forced_repetition(_schedule(False, 16))
        assert value == 1.0
        assert detail["score"] == 0.5 and detail["error"] == 0.0
        assert detail["contempt_wins"] is False and detail["r_duel"] == 1.0
        assert detail["rung"] == RUNG and detail["pairs"] == 8

    def test_drift_reads_zero_whatever_the_share(self) -> None:
        value, detail = forced_repetition(_schedule(True, 12))
        assert value == 0.0
        assert detail["contempt_wins"] is True and detail["r_plain"] == 0.75

    def test_no_schedule_is_no_measurement(self) -> None:
        assert forced_repetition([]) == (None, {})
        control_only = [r for r in _schedule(False, 4) if waiting.CONTEMPT not in ",".join(r.agents)]
        assert forced_repetition(control_only) == (None, {})


class TestTheGate:
    def test_it_refuses_above_the_decisive_gates_line(self) -> None:
        assert pytest.approx(1 - 0.2) == WAITING_MAXIMUM
        assert waiting_violations({FORCED_REPETITION: 0.8}) == []
        refused = waiting_violations({FORCED_REPETITION: 0.85})
        assert len(refused) == 1 and refused[0].startswith("forced_repetition=0.850 above maximum 0.8")

    def test_absence_is_not_a_verdict(self) -> None:
        assert waiting_violations({}) == []

    def test_it_stands_beside_the_weights_gates_and_a_blind_run_reads_it(self) -> None:
        weights = load_weights("v13")
        components = Components(raw={FORCED_REPETITION: 1.0})
        assert any(v.startswith("forced_repetition=") for v in gate_violations(components, weights))
        blind = readable_gate_violations(components, weights, ("skill_gradient",))
        assert any(v.startswith("forced_repetition=") for v in blind)


class TestTheRung:
    def test_the_top_searching_rung(self) -> None:
        canonical = ("random", "greedy", "minimax:nodes=400", RUNG)
        assert waiting.top_rung(canonical) == RUNG
        net = ("random", "greedy", "minimax:net=abc,nodes=400", "minimax:net=abc,nodes=25600")
        assert waiting.top_rung(net) == "minimax:net=abc,nodes=25600"
        assert waiting.top_rung(("random", "greedy", "matrix:nodes=1600")) is None


class TestTheStamp:
    def _basis(self, seeds: tuple[int, ...], schedule: str) -> Basis:
        return Basis(
            ruleset_hash="h", weights_version="v13", agents=("random",), games_per_pairing=4,
            seeds=seeds, engine="e", metrics="m", tables="t", analysis="a", players="p",
            waiting=schedule,
        )

    def test_a_library_stamp_without_the_schedule_is_a_rescore(self) -> None:
        library = (0, 1000, 2000, 3000)
        staleness = self._basis(library, "").stale_against(self._basis(library, "s"))
        assert staleness.needs == "rescore"
        assert any("waiting schedule" in reason for reason in staleness.reasons)

    def test_a_one_run_stamp_never_reads_it(self) -> None:
        assert self._basis((0,), "").stale_against(self._basis((0,), "")).needs == ""
        library_reader = self._basis((0, 1000, 2000, 3000), "s")
        staleness = self._basis((0,), "").stale_against(library_reader)
        assert staleness.stored_needs == "" and staleness.missing_seeds == (1000, 2000, 3000)

    def test_it_round_trips_and_keeps_old_shapes(self) -> None:
        stamped = self._basis((0, 1000, 2000, 3000), "s")
        assert Basis.from_dict(stamped.as_dict()) == stamped
        assert "waiting" not in self._basis((0,), "").as_dict()

    def test_the_schedule_names_its_code(self) -> None:
        assert waiting.schedule_id().startswith(waiting.SCHEDULE + "@")


def test_the_schedule_plays_and_is_read(tmp_path: Path) -> None:
    """The real thing at a toy budget: 16 duel games and 16 control games, read back."""
    spec = load_spec(Path("rulesets/dull.json"))
    games = waiting.play(spec, ("random", "minimax:nodes=40"), seed=0, workers=1)
    assert len(games) == 2 * waiting.PAIRS + waiting.CONTROL
    assert sorted({g.seed for g in games}) == list(range(500, 500 + waiting.CONTROL))
    waiting.save(games, tmp_path)
    assert waiting.played(tmp_path)
    loaded = waiting.load(tmp_path)
    assert loaded is not None
    value, detail = forced_repetition(loaded)
    assert value is not None and detail["pairs"] == waiting.PAIRS
    assert waiting.load(tmp_path / "absent") is None


def test_one_schedule_call_plays_the_duels_and_the_controls_games() -> None:
    """`play` through `run_schedules` is `duel_records` and a self-play tournament, game for game."""
    from chess_variant_sandbox.tournament.checkpoint import run_tournament
    from chess_variant_sandbox.tournament.duel import duel_records
    from chess_variant_sandbox.tournament.runner import TournamentConfig

    spec = load_spec(Path("rulesets/dull.json"))
    rung = "minimax:nodes=40"
    played = waiting.play(spec, ("random", rung), seed=0, workers=1)
    duel = duel_records(spec, f"{rung},{waiting.CONTEMPT}", rung, waiting.PAIRS, seed=500, workers=1)
    control = run_tournament(
        spec, TournamentConfig(agents=(rung,), games_per_pairing=waiting.CONTROL, seed=500,
                               workers=1, collect_series=False),
    ).records
    key = [(g.seed, g.agents, g.moves) for g in played]
    assert sorted(key) == sorted((g.seed, g.agents, g.moves) for g in [*duel, *control])


def test_a_library_run_without_the_schedule_plays_it_and_is_scored_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`corpus._run_seed`: a library run whose games stand gets the schedule, then a re-score."""
    from types import SimpleNamespace

    from chess_variant_sandbox import corpus

    monkeypatch.setattr(corpus, "PLAY_WAITING", True)
    monkeypatch.setattr(corpus, "stale_build", lambda: None)
    played: list[int] = []
    monkeypatch.setattr(
        corpus.waiting, "play", lambda spec, agents, seed, workers: played.append(seed) or []
    )
    rescored: list[int] = []
    monkeypatch.setattr(
        corpus, "_rescore_seed", lambda candidate, seed, *a, **k: rescored.append(seed) or "r"
    )
    spec = load_spec(Path("rulesets/dull.json"))
    out = corpus.run_dir("h", 1000, tmp_path)
    out.mkdir(parents=True)
    (out / "games.jsonl").write_text("", encoding="utf-8")
    library: Any = SimpleNamespace(stage="committed", spec=spec, hash="h")
    assert corpus._run_seed(library, 1000, "", tmp_path, None, 1, lambda _m: None) == "r"
    assert played == [1000] and rescored == [1000] and waiting.played(out)
    visitor: Any = SimpleNamespace(stage="candidate", spec=spec, hash="v")
    assert not corpus._plays_waiting(visitor)
