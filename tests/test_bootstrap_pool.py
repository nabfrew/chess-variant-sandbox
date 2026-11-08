"""The bootstrap's draws come first and are scored in a pool, and nothing moves (Q1119).

`bootstrap_index` used to draw a resample, score it and draw the next. It now draws all
of them from the one `random.Random(seed)` first (`bootstrap.draw_resamples`), scores
them here or in a process pool, and collects them in draw order
(`docs/plan-performance.md` C11, survey R13). This is not the refused item, a stream per
worker: the draws are the loop's own, as positions rather than objects.

The oracle is the loop as it stood, kept below. Equality is `==` on floats and on whole
results, never approx, and the per-draw lists are compared in order, since the result's
own sums (`fmean`, `pstdev`) are exact and would not see an order change.
"""

from __future__ import annotations

import math
import pickle
import random
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.metrics import bootstrap
from chess_variant_sandbox.metrics.components import GameReadings, compute_components
from chess_variant_sandbox.metrics.scoring import load_weights
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.analysis import BlunderTrial, CorpusAnalysis, GameAnalysis
from chess_variant_sandbox.tournament.ratings import rate
from chess_variant_sandbox.tournament.record import GameRecord

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"
LADDER = ("random", "greedy", "minimax:nodes=100", "minimax:nodes=400", "minimax:nodes=1600")
WANTED = ("skill_gradient", "lead_changes", "material_volatility", "decisive_rate_worst_half")


def _corpus(seed: int, uneven: bool = False) -> list[GameRecord]:
    rng = random.Random(seed)
    records = []
    game = 0
    for first in LADDER:
        for second in LADDER:
            for _ in range(rng.randrange(1, 6) if uneven else 4):
                plies = rng.randrange(8, 60)
                winner = rng.choice([0, 1, None])
                records.append(
                    GameRecord(
                        ruleset_hash="fixture", ruleset_name="fixture", seed=game,
                        agents=(first, second), moves=[rng.randrange(40) for _ in range(plies)],
                        winner=winner, reason="checkmate" if winner is not None else "ply_limit",
                        plies=plies, branching=[rng.randrange(1, 40) for _ in range(plies)],
                        material=[rng.uniform(-9, 9) for _ in range(plies)],
                        captures=[rng.randrange(2) for _ in range(plies)],
                    )
                )
                game += 1
    return records


def _analysis(records: list[GameRecord], seed: int) -> CorpusAnalysis:
    """Analyses for most games (not all: an unmatched game is a case) and some trials."""
    rng = random.Random(seed)
    games = [
        GameAnalysis(seed=r.seed, agents=r.agents) for r in records if rng.random() < 0.9
    ]
    trials = [
        BlunderTrial(
            seed=r.seed, agents=r.agents, ply=rng.randrange(1, 8),
            original_winner=r.winner, blundering_side=rng.randrange(2),
            new_winner=rng.choice([0, 1, None]),
        )
        for r in rng.sample(records, 7)
    ]
    return CorpusAnalysis(games=games, blunders=trials)


def _old_loop(
    spec: dict[str, Any],
    records: list[GameRecord],
    resamples: int,
    seed: int,
    analysis: CorpusAnalysis | None,
) -> list[tuple[float, dict[str, float]] | None]:
    """The loop as it stood before Q1119: draw, score, draw the next. One entry a draw."""
    weights = load_weights("v2")
    strata: dict[tuple[str, str], list[GameRecord]] = defaultdict(list)
    for record in records:
        strata[record.agents].append(record)
    by_game = {}
    if analysis is not None:
        by_game = {(g.seed, tuple(g.agents)): g for g in analysis.games}
    readings = GameReadings()
    rng = random.Random(seed)
    out: list[tuple[float, dict[str, float]] | None] = []
    for _ in range(resamples):
        resampled: list[GameRecord] = []
        for games in strata.values():
            resampled.extend(rng.choices(games, k=len(games)))
        drawn = None
        if analysis is not None:
            matched = [
                by_game[(r.seed, r.agents)] for r in resampled if (r.seed, r.agents) in by_game
            ]
            trials = (
                rng.choices(analysis.blunders, k=len(analysis.blunders))
                if analysis.blunders
                else []
            )
            drawn = CorpusAnalysis(games=matched, blunders=trials)
        try:
            measured = compute_components(
                spec, resampled, rate(resampled), drawn, readings=readings
            )
            total, _ = weights.combine(weights.score_all(measured.raw))
        except Exception:
            out.append(None)
            continue
        out.append((total, {n: measured.raw[n] for n in WANTED if n in measured.raw}))
    return out


def _scorer(
    spec: dict[str, Any], records: list[GameRecord], analysis: CorpusAnalysis | None
) -> tuple[bootstrap._Scorer, list[bootstrap.Draw]]:
    strata: dict[tuple[str, str], list[GameRecord]] = defaultdict(list)
    for record in records:
        strata[record.agents].append(record)
    ordered = [record for games in strata.values() for record in games]
    by_game = (
        {(g.seed, tuple(g.agents)): g for g in analysis.games} if analysis is not None else {}
    )
    scorer = bootstrap._Scorer(spec, ordered, load_weights("v2"), analysis, by_game, WANTED)
    draws = bootstrap.draw_resamples(
        [len(games) for games in strata.values()],
        len(analysis.blunders) if analysis is not None else None,
        30,
        11,
    )
    return scorer, draws


@pytest.mark.parametrize("uneven", [False, True])
@pytest.mark.parametrize("with_analysis", [False, True])
def test_the_draws_are_the_loops_draws(uneven: bool, with_analysis: bool) -> None:
    """Draw by draw, the same games and trials as the loop drew, in the same order."""
    records = _corpus(4, uneven)
    analysis = _analysis(records, 4) if with_analysis else None
    strata: dict[tuple[str, str], list[GameRecord]] = defaultdict(list)
    for record in records:
        strata[record.agents].append(record)
    ordered = [record for games in strata.values() for record in games]
    draws = bootstrap.draw_resamples(
        [len(games) for games in strata.values()],
        len(analysis.blunders) if analysis is not None else None,
        25,
        7,
    )
    rng = random.Random(7)
    for chosen, trials in draws:
        old: list[GameRecord] = []
        for games in strata.values():
            old.extend(rng.choices(games, k=len(games)))
        assert [ordered[i] for i in chosen] == old
        assert all(ordered[i] is game for i, game in zip(chosen, old, strict=True))
        if analysis is None:
            assert trials is None
        else:
            assert trials is not None
            assert [analysis.blunders[i] for i in trials] == rng.choices(
                analysis.blunders, k=len(analysis.blunders)
            )
    # The stream is spent exactly as the loop spent it.
    follow = random.Random(7)
    for _ in draws:
        for games in strata.values():
            follow.choices(games, k=len(games))
        if analysis is not None:
            follow.choices(analysis.blunders, k=len(analysis.blunders))
    assert follow.getstate() == rng.getstate()


def test_no_trials_draw_nothing() -> None:
    """An analysis with no blunder trials draws nothing for them, as the loop did."""
    draws = bootstrap.draw_resamples([3, 2], 0, 4, 5)
    rng = random.Random(5)
    for chosen, trials in draws:
        assert trials == []
        assert chosen == rng.choices(range(3), k=3) + rng.choices(range(3, 5), k=2)


@pytest.mark.parametrize("with_analysis", [False, True])
def test_each_draw_scores_as_the_loop_scored_it(with_analysis: bool) -> None:
    """In order, entry by entry, a dropped resample in the same place."""
    spec = load_spec(RULESETS / "dull.json")
    records = _corpus(2, uneven=True)
    analysis = _analysis(records, 2) if with_analysis else None
    scorer, draws = _scorer(spec, records, analysis)
    new = bootstrap._score_draws(scorer, draws, workers=1)
    assert sum(entry is not None for entry in new) >= 20
    assert new == _old_loop(spec, records, 30, 11, analysis)


def test_bootstrap_index_is_the_loops() -> None:
    spec = load_spec(RULESETS / "dull.json")
    records = _corpus(3)
    result = bootstrap.bootstrap_index(
        spec, records, load_weights("v2"), resamples=30, seed=11, components_of_interest=WANTED
    )
    assert result is not None
    entries = [e for e in _old_loop(spec, records, 30, 11, None) if e is not None]
    assert result.resamples == len(entries)
    assert result.mean == round(statistics.fmean(total for total, _ in entries), 4)
    assert result.component_mean["skill_gradient"] == round(
        statistics.fmean(e["skill_gradient"] for _, e in entries), 4
    )


def test_the_pool_collects_in_draw_order(tmp_path: Path) -> None:
    """Two workers, two calls: each list is the serial list, entry by entry.

    The tasks `_score_draws` sends, through a pool of its own and without its serial
    fallback, so a pool that could not start fails here rather than passing on the
    serial answer. The second call's records differ, so a worker that kept the first
    call's scorer would answer wrongly.
    """
    spec = load_spec(RULESETS / "dull.json")
    calls = []
    for number, seed in enumerate((6, 8)):
        scorer, draws = _scorer(spec, _corpus(seed, uneven=bool(number)), None)
        path = tmp_path / f"scorer-{number}.pickle"
        path.write_bytes(pickle.dumps(scorer))
        tasks = [(f"test-{number}", str(path), draws[i:i + 7]) for i in range(0, len(draws), 7)]
        calls.append((bootstrap._score_draws(scorer, draws, workers=1), tasks))
    assert all(sum(e is not None for e in serial) >= 20 for serial, _ in calls)
    assert calls[0][0] != calls[1][0]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for serial, tasks in calls:
            pooled = [entry for part in pool.map(bootstrap._score_chunk, tasks) for entry in part]
            assert pooled == serial
        # Inside a pool's worker (`measure`'s re-score pool, Q1118) the bootstrap never
        # starts a pool of its own, however large the run.
        assert pool.submit(bootstrap.pool_workers, 0, 10**9).result() == 1


def test_the_bootstraps_pool_is_the_serial_answer() -> None:
    """Through `_score_draws` itself, forced to two workers, twice in one process."""
    spec = load_spec(RULESETS / "dull.json")
    try:
        for seed in (6, 9):
            scorer, draws = _scorer(spec, _corpus(seed), None)
            assert bootstrap._score_draws(scorer, draws, workers=2) == bootstrap._score_draws(
                scorer, draws, workers=1
            )
    finally:
        bootstrap._shutdown_pool()


def test_who_pools() -> None:
    assert bootstrap.pool_workers(1, 10**9) == 1
    assert bootstrap.pool_workers(-1, 10**9) == 1
    assert bootstrap.pool_workers(3, 1) == 3
    assert bootstrap.pool_workers(0, bootstrap.POOL_MIN_DRAWN - 1) == 1
    assert 1 <= bootstrap.pool_workers(0, bootstrap.POOL_MIN_DRAWN) <= bootstrap.POOL_MAX_WORKERS


# ---- Q1145: the pool's sizing, its lifetime, the chunking, and the helpers ----


def _no_parent(monkeypatch: pytest.MonkeyPatch, cpus: int | None) -> None:
    """A top-level process on a platform with processes and ``cpus`` CPUs."""
    import multiprocessing
    import os
    import sys

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: None)
    monkeypatch.setattr(os, "cpu_count", lambda: cpus)


def test_pool_workers_sizes_by_the_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """0 pools from POOL_MIN_DRAWN on, one CPU short of the machine, at most four."""
    floor = bootstrap.POOL_MIN_DRAWN
    _no_parent(monkeypatch, 3)
    assert bootstrap.pool_workers(0, floor) == 2
    assert bootstrap.pool_workers(0, floor - 1) == 1
    assert bootstrap.pool_workers(2, 1) == 2
    assert bootstrap.pool_workers(1, 10**9) == 1
    _no_parent(monkeypatch, 4)
    assert bootstrap.pool_workers(0, floor) == 3
    _no_parent(monkeypatch, 9)
    assert bootstrap.pool_workers(0, floor) == 4
    _no_parent(monkeypatch, 1)
    assert bootstrap.pool_workers(0, floor) == 1
    _no_parent(monkeypatch, None)  # unknown: as if two, so serial
    assert bootstrap.pool_workers(0, floor) == 1


def test_pool_workers_never_pools_in_a_child_or_a_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import multiprocessing
    import sys

    _no_parent(monkeypatch, 8)
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: object())
    assert bootstrap.pool_workers(0, 10**9) == 1
    _no_parent(monkeypatch, 8)
    monkeypatch.setattr(sys, "platform", "emscripten")
    assert bootstrap.pool_workers(0, 10**9) == 1


def test_the_pool_is_kept_at_one_size_and_replaced_at_another() -> None:
    """No worker starts: an executor spawns its processes on its first task."""
    try:
        first = bootstrap._pool(2)
        assert isinstance(first, ProcessPoolExecutor)
        assert bootstrap._pool(2) is first
        second = bootstrap._pool(3)
        assert second is not first
        assert second._max_workers == 3
        with pytest.raises(RuntimeError, match="after shutdown"):
            first.submit(int)
    finally:
        bootstrap._shutdown_pool()
    assert bootstrap._POOL is None


def test_shutdown_waits_and_cancels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Waiting leaves no worker behind; cancelling drops what a broken call queued."""
    calls: list[dict[str, bool]] = []

    class Held:
        def shutdown(self, **kwargs: bool) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(bootstrap, "_POOL", Held())
    monkeypatch.setattr(bootstrap, "_POOL_SIZE", 2)
    bootstrap._shutdown_pool()
    assert calls == [{"wait": True, "cancel_futures": True}]
    assert (bootstrap._POOL, bootstrap._POOL_SIZE) == (None, 0)


class _InlinePool:
    """A pool that runs its tasks here, so the tasks `_score_draws` cuts can be read."""

    def __init__(self) -> None:
        self.sizes: list[int] = []
        self.tasks: list[Any] = []

    def __call__(self, size: int) -> _InlinePool:
        self.sizes.append(size)
        return self

    def map(self, fn: Any, tasks: Any) -> list[Any]:
        self.tasks = list(tasks)
        return [fn(task) for task in self.tasks]


@pytest.mark.parametrize(
    ("drawn", "workers", "lengths"),
    [(30, 2, [4] * 7 + [2]), (30, 3, [3] * 10), (5, 2, [1] * 5)],
)
def test_the_draws_are_cut_four_runs_a_worker(
    monkeypatch: pytest.MonkeyPatch, drawn: int, workers: int, lengths: list[int]
) -> None:
    """ceil(draws / (workers * 4)) a run, at least one; the joined runs are the serial list."""
    spec = load_spec(RULESETS / "dull.json")
    scorer, draws = _scorer(spec, _corpus(5), None)
    draws = draws[:drawn]
    serial = bootstrap._score_draws(scorer, draws, workers=1)
    pool = _InlinePool()
    monkeypatch.setattr(bootstrap, "_pool", pool)
    monkeypatch.setattr(bootstrap, "_WORKER_SCORER", None)
    assert bootstrap._score_draws(scorer, draws, workers=workers) == serial
    assert pool.sizes == [workers]
    assert [len(chunk) for _, _, chunk in pool.tasks] == lengths
    assert [d for _, _, chunk in pool.tasks for d in chunk] == draws


def test_a_pool_that_breaks_gives_the_serial_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pool that cannot start is shut and the draws are scored here, entry for entry."""
    spec = load_spec(RULESETS / "dull.json")
    scorer, draws = _scorer(spec, _corpus(5), None)
    draws = draws[:6]
    serial = bootstrap._score_draws(scorer, draws, workers=1)
    shut: list[bool] = []

    def broken(size: int) -> Any:
        raise OSError("no processes here")

    monkeypatch.setattr(bootstrap, "_pool", broken)
    monkeypatch.setattr(bootstrap, "_shutdown_pool", lambda: shut.append(True))
    assert bootstrap._score_draws(scorer, draws, workers=2) == serial
    assert shut == [True]


def test_a_draw_without_trials_scores_without_the_analysis() -> None:
    """Trials ``None`` reads no analysis, even where the scorer holds one."""
    spec = load_spec(RULESETS / "dull.json")
    records = _corpus(3)
    analysis = _analysis(records, 3)
    with_analysis, draws = _scorer(spec, records, analysis)
    without, _ = _scorer(spec, records, None)
    chosen, trials = draws[0]
    assert trials is not None
    assert with_analysis.score((chosen, None)) == without.score((chosen, None))
    assert without.score((chosen, None)) is not None


def test_percentile_interpolates_between_neighbours() -> None:
    p = bootstrap._percentile
    assert math.isnan(p([], 0.5))
    assert p([7.0], 0.3) == 7.0
    assert p([1.0, 3.0], 0.5) == 2.0
    assert p([1.0, 3.0], 1.0) == 3.0
    assert p([1.0, 3.0, 5.0], 0.25) == 2.0
    assert p([1.0, 3.0, 5.0], 0.75) == 4.0
    assert p([1.0, 3.0, 5.0], 1.0) == 5.0
    assert p([1.0, 3.0, 5.0], 0.0) == 1.0


def test_component_noise_skips_an_entry_with_no_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An entry whose bootstrap is None adds no noise, and the entries after it still do;
    a component too thin to summarise is skipped, and the components after it are not."""
    raw = {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.00005, "e": 4.0}
    sd = {"a": 0.5, "b": None, "c": 1.5, "d": 2.5, "e": 3.5}

    class Measured:
        def __init__(self, value: float) -> None:
            self.raw = {"x": value}

    class Drawn:
        def __init__(self, value: float) -> None:
            self.component_sd = {"x": value}

    monkeypatch.setattr(bootstrap, "rate", lambda records: None)
    monkeypatch.setattr(
        bootstrap,
        "compute_components",
        lambda spec, records, ratings, analysis: Measured(raw[spec]),
    )
    monkeypatch.setattr(
        bootstrap,
        "bootstrap_index",
        lambda spec, *a, **k: None if (value := sd[spec]) is None else Drawn(value),
    )
    corpus: list[Any] = [(name, [], None) for name in raw]
    # "w" is measured by no entry: it is left out, and "x" after it is still summarised.
    result = bootstrap.component_noise(corpus, load_weights("v2"), ("w", "x"), resamples=3)
    # pstdev of 0, 1, 2, 3.00005, 4 is 1.41425...; the mean of 0.5, 1.5, 2.5, 3.5 is 2.0.
    assert result == {"x": {"corpus_sd": 1.4142, "noise_sd": 2.0, "ratio": 0.71}}
