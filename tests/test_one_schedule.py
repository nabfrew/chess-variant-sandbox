"""Q1117 (Q-C9): one schedule per game, longest first, one analysis pool.

A measurement replays its seeds as one core call, the games started in descending combined
budget, and analyses them in one process pool sized to the physical cores. None of that may
change a record: each game's seed and players are fixed by its own seed's schedule before
anything is ordered, and every result goes back to its place. These hold the order and
the identity.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import pytest

from chess_variant_sandbox import corpus
from chess_variant_sandbox.tournament import TournamentConfig, run_tournament, runner, schedule
from chess_variant_sandbox.tournament.analysis import analyse_corpus


@pytest.fixture(scope="module")
def small_spec() -> dict:
    """A quick game: 5x5, two piece types, royal capture."""
    return {
        "schema_version": 1,
        "name": "skirmish",
        "board": {"width": 5, "height": 5},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
            ]},
            {"id": "R", "value": 5, "moves": [
                {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
            ]},
        ],
        "setup": {"mirror": True, "placements": [
            {"piece": "K", "squares": ["c1"]},
            {"piece": "R", "squares": ["a1", "e1"]},
        ]},
        "rules": {"win": "royal_capture", "max_plies": 120, "no_progress_draw": 40},
    }


ROSTER = ("random", "greedy", "minimax:nodes=200", "minimax:nodes=50")


def _as_played(records: list[Any]) -> list[dict]:
    """Every field of each record, in order, but the one a clock writes."""
    return [
        {key: value for key, value in record.as_dict().items() if key != "duration_ms"}
        for record in records
    ]


class TestTheOrder:
    def test_a_seat_costs_its_nodes_greedy_one_and_random_nothing(self) -> None:
        assert runner.seat_budget("random") == 0
        assert runner.seat_budget("greedy") == 1
        assert runner.seat_budget("minimax:nodes=6400") == 6400
        assert runner.combined_budget("minimax:nodes=400", "greedy") == 401

    def test_the_longest_games_go_first_and_ties_keep_the_schedule_order(self) -> None:
        tasks = schedule(TournamentConfig(agents=ROSTER, games_per_pairing=2, seed=7))
        order = runner.dispatch_order(tasks)

        assert sorted(order) == list(range(len(tasks)))
        budgets = [runner.combined_budget(tasks[i][1], tasks[i][2]) for i in order]
        assert budgets == sorted(budgets, reverse=True)
        for k in range(len(order) - 1):
            if budgets[k] == budgets[k + 1]:
                assert order[k] < order[k + 1], "a tie is broken by the schedule"
        assert tasks[order[0]][1:] == ("minimax:nodes=200", "minimax:nodes=200")

    def test_the_core_is_handed_the_longest_first_and_the_records_come_back_in_schedule_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handed: list[list[int]] = []

        class _Core:
            def run_tournament(self, specs, values, tasks, collect_series, threads):
                handed.append([seed for seed, _, _ in tasks])
                return [seed for seed, _, _ in tasks]

        class _Engine:
            core = _Core()
            spec = None

        monkeypatch.setattr(runner, "eval_values", lambda spec: [])
        monkeypatch.setattr(runner, "build_record", lambda engine, pair, seed, result: (seed, result))
        config = TournamentConfig(agents=ROSTER, games_per_pairing=1, seed=100, workers=3)
        tasks = schedule(config)

        records = runner._run_native(_Engine(), config, tasks, [object()] * len(ROSTER), False)

        order = runner.dispatch_order(tasks)
        assert handed == [[tasks[i][0] for i in order]]
        assert records == [(seed, seed) for seed, _, _ in tasks]


class TestOneScheduleIsTheSchedulesOneByOne:
    @pytest.mark.parametrize("workers", [1, 3])
    def test_each_seed_gets_the_games_its_own_tournament_plays(
        self, small_spec: dict, workers: int
    ) -> None:
        configs = [
            TournamentConfig(agents=ROSTER, games_per_pairing=1, seed=seed, workers=workers)
            for seed in (0, 1000)
        ]
        pooled = runner.run_schedules(small_spec, configs)
        assert [result.config for result in pooled] == configs
        for config, result in zip(configs, pooled, strict=True):
            alone = run_tournament(small_spec, replace(config, workers=1))
            assert _as_played(result.records) == _as_played(alone.records)
            assert [(r.seed, *r.agents) for r in result.records] == schedule(config)
            assert result.ratings.as_dict() == alone.ratings.as_dict()

    def test_in_a_campaign_it_is_the_same_games(self, small_spec: dict, tmp_path) -> None:
        from chess_variant_sandbox.tournament import checkpoint

        configs = [
            TournamentConfig(agents=ROSTER, games_per_pairing=1, seed=seed, workers=2)
            for seed in (5, 9)
        ]
        plain = runner.run_schedules(small_spec, configs)
        with checkpoint.active(tmp_path / "campaign", build="test"):
            resumed = checkpoint.run_schedules(small_spec, configs)
        for first, second in zip(plain, resumed, strict=True):
            assert _as_played(first.records) == _as_played(second.records)

    def test_schedules_that_disagree_on_series_are_refused(self, small_spec: dict) -> None:
        configs = [
            TournamentConfig(agents=ROSTER, seed=0),
            TournamentConfig(agents=ROSTER, seed=1, collect_series=False),
        ]
        with pytest.raises(ValueError, match="one schedule"):
            runner.run_schedules(small_spec, configs)


class _InProcessPool:
    """A stand-in `ProcessPoolExecutor`: the worker initialiser and every task run here,
    so the pooled path is checked without spawning, and the width it asked for is kept."""

    widths: list[int] = []

    def __init__(self, max_workers: int, initializer: Any, initargs: tuple) -> None:
        type(self).widths.append(max_workers)
        initializer(*initargs)

    def __enter__(self) -> _InProcessPool:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def map(self, function: Any, items: Any, chunksize: int = 1) -> Any:
        return [function(item) for item in items]


@pytest.fixture(scope="module")
def corpora(small_spec: dict) -> list[list[Any]]:
    """Two seeds' games, each with enough graded games (eight or more) to be pooled."""
    configs = [
        TournamentConfig(
            agents=("greedy", "minimax:nodes=60"), games_per_pairing=3, seed=seed, workers=2
        )
        for seed in (0, 1000)
    ]
    return [result.records for result in runner.run_schedules(small_spec, configs)]


class TestOneAnalysisPool:
    def test_the_pool_analyses_each_seed_as_its_own_pass_would(
        self, small_spec: dict, corpora: list[list[Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(corpus, "ProcessPoolExecutor", _InProcessPool)
        monkeypatch.setattr(corpus, "physical_cores", lambda: 4)
        _InProcessPool.widths = []

        pooled = corpus.analyse_seeds(small_spec, corpora, workers=7)

        assert _InProcessPool.widths == [4], "one pool, as wide as the cores"
        for records, analysis in zip(corpora, pooled, strict=True):
            alone = analyse_corpus(small_spec, records, seed=corpus.CANONICAL_SEED, workers=1)
            assert analysis.as_dict() == alone.as_dict()

    def test_the_pool_is_never_wider_than_asked(
        self, small_spec: dict, corpora: list[list[Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(corpus, "ProcessPoolExecutor", _InProcessPool)
        monkeypatch.setattr(corpus, "physical_cores", lambda: 8)
        _InProcessPool.widths = []
        corpus.analyse_seeds(small_spec, corpora, workers=2)
        assert _InProcessPool.widths == [2]

    def test_one_worker_starts_no_pool(
        self, small_spec: dict, corpora: list[list[Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(corpus, "ProcessPoolExecutor", _InProcessPool)
        _InProcessPool.widths = []
        serial = corpus.analyse_seeds(small_spec, corpora, workers=1)
        assert _InProcessPool.widths == []
        assert [a.as_dict() for a in serial] == [
            analyse_corpus(small_spec, records, seed=corpus.CANONICAL_SEED, workers=1).as_dict()
            for records in corpora
        ]

    @pytest.mark.slow
    def test_a_real_pool_gives_the_serial_answer(
        self, small_spec: dict, corpora: list[list[Any]]
    ) -> None:
        pooled = corpus.analyse_seeds(small_spec, corpora, workers=2)
        for records, analysis in zip(corpora, pooled, strict=True):
            alone = analyse_corpus(small_spec, records, seed=corpus.CANONICAL_SEED, workers=1)
            assert analysis.as_dict() == alone.as_dict()


def test_a_measurement_plays_its_seeds_as_one_schedule_and_analyses_them_in_one_pool(
    small_spec: dict, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, list[Any]] = {"play": [], "analyse": [], "score": []}

    class _Played:
        def __init__(self, seed: int) -> None:
            self.records = [seed]
            self.ratings = None

        def save(self, out) -> None:
            calls["play"].append(("saved", out.name))

    def play(spec, configs, **_):
        calls["play"].append([config.seed for config in configs])
        return [_Played(config.seed) for config in configs]

    def analyse(spec, corpora, workers=0):
        calls["analyse"].append(corpora)
        return [_Analysis() for _ in corpora]

    class _Analysis:
        def save(self, path) -> None:
            pass

    def score(candidate, seed, out, records, ratings, weighting, analysis, say, origin):
        calls["score"].append((seed, records, origin))
        return seed

    monkeypatch.setattr(corpus, "run_schedules", play)
    monkeypatch.setattr(corpus, "analyse_seeds", analyse)
    monkeypatch.setattr(corpus, "_score_run", score)
    candidate = corpus.Candidate(
        hash="0123456789ab", name="skirmish", spec=small_spec, stage="committed",
        path=tmp_path / "skirmish.json",
    )

    readings = corpus._play_seeds(
        candidate, [0, 1000, 3000], tmp_path, None, 0, lambda _: None, {1000: "adopted"}
    )

    assert calls["play"][0] == [0, 1000, 3000], "one schedule, the seeds in order"
    assert calls["analyse"] == [[[0], [1000], [3000]]], "one pool for every seed"
    assert calls["score"] == [(0, [0], "played"), (1000, [1000], "adopted"), (3000, [3000], "played")]
    assert readings == {0: 0, 1000: 1000, 3000: 3000}


def test_the_physical_cores_are_at_least_one_and_at_most_the_logical() -> None:
    cores = corpus.physical_cores()
    assert 1 <= cores <= (os.cpu_count() or cores)


# Q1144 (C9's follow-up): the re-analyse path queues a game's seeds into that one pool.


@pytest.fixture
def stored_runs(small_spec: dict, tmp_path) -> tuple[corpus.Candidate, list[int]]:
    """Two seeds' games on disk, as a measurement left them, under a scratch runs dir."""
    candidate = corpus.Candidate(
        hash="0123456789ab", name="skirmish", spec=small_spec, stage="committed",
        path=tmp_path / "skirmish.json",
    )
    seeds = [0, 1000]
    configs = [
        TournamentConfig(
            agents=("greedy", "minimax:nodes=60"), games_per_pairing=3, seed=seed, workers=2
        )
        for seed in seeds
    ]
    for seed, result in zip(seeds, runner.run_schedules(small_spec, configs), strict=True):
        result.save(corpus.run_dir(candidate.hash, seed, tmp_path))
    return candidate, seeds


class TestTheReanalysePathSharesThePool:
    def test_a_measurement_re_analyses_its_seeds_in_one_pool(
        self, stored_runs, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        candidate, seeds = stored_runs
        calls: dict[str, list[Any]] = {"analyse": [], "alone": [], "score": []}
        real = corpus.analyse_seeds

        def analyse(spec, corpora, workers=0):
            calls["analyse"].append([len(records) for records in corpora])
            return real(spec, corpora, workers=1)

        def score(candidate, seed, out, records, ratings, weighting, analysis, say, origin):
            calls["score"].append((seed, len(records), analysis is not None))
            return seed

        monkeypatch.setattr(corpus, "analyse_seeds", analyse)
        monkeypatch.setattr(corpus, "_score_run", score)
        monkeypatch.setattr(corpus, "stale_build", lambda: None)
        monkeypatch.setattr(corpus, "read_measurement", lambda *a, **k: None)
        monkeypatch.setattr(corpus, "_store", lambda c, r, readings, *a, **k: readings)
        monkeypatch.setattr(corpus, "run_schedules", lambda *a, **k: pytest.fail("played"))
        real_alone = corpus.analyse_corpus

        def alone(*args, **kwargs):
            calls["alone"].append(kwargs.get("workers"))
            return real_alone(*args, **kwargs)

        monkeypatch.setattr(corpus, "analyse_corpus", alone)

        readings = corpus._measure_seeds(
            candidate, tmp_path, corpus.DEFAULT_WEIGHTS, 0, None,
            {seed: "reanalyse" for seed in seeds},
        )

        assert len(calls["analyse"]) == 1 and len(calls["analyse"][0]) == 2, (
            "one pool for both seeds"
        )
        # Inside the stand-in (workers=1) each corpus is analysed serially, once each.
        assert calls["alone"] == [1, 1]
        assert [seed for seed, _, analysed in calls["score"] if analysed] == seeds
        assert readings == seeds
        for seed in seeds:
            assert (corpus.run_dir(candidate.hash, seed, tmp_path) / "analysis.json").exists()

    def test_the_pooled_files_are_the_ones_each_seed_alone_writes(
        self, stored_runs, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        candidate, seeds = stored_runs
        monkeypatch.setattr(corpus, "stale_build", lambda: None)
        outs = [corpus.run_dir(candidate.hash, seed, tmp_path) for seed in seeds]
        weighting = corpus.load_weights(corpus.DEFAULT_WEIGHTS)

        for out, seed in zip(outs, seeds, strict=True):
            corpus._rescore_seed(
                candidate, seed, out, weighting, 1, lambda _: None, reanalyse=True
            )
        alone = {
            path: path.read_bytes() for out in outs
            for path in sorted(out.glob("*.json")) if path.name != "tournament.json"
        }
        assert any(path.name == "analysis.json" for path in alone)
        for path in alone:
            path.unlink()

        monkeypatch.setattr(corpus, "ProcessPoolExecutor", _InProcessPool)
        monkeypatch.setattr(corpus, "physical_cores", lambda: 4)
        _InProcessPool.widths = []
        done = corpus._reanalyse_seeds(candidate, seeds, tmp_path, 7, lambda _: None)
        for out, seed in zip(outs, seeds, strict=True):
            corpus._rescore_seed(
                candidate, seed, out, weighting, 7, lambda _: None,
                reanalyse=True, analysed=done[seed],
            )

        assert _InProcessPool.widths == [4], "one pool, as wide as the cores"
        assert {path: path.read_bytes() for path in alone} == alone

    def test_a_convergence_re_score_re_analyses_the_kept_seeds_in_one_pool(
        self, stored_runs, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        candidate, seeds = stored_runs
        calls: list[int] = []
        real = corpus.analyse_seeds

        def analyse(spec, corpora, workers=0):
            calls.append(len(corpora))
            return real(spec, corpora, workers=1)

        monkeypatch.setattr(corpus, "analyse_seeds", analyse)
        monkeypatch.setattr(corpus, "stale_build", lambda: None)
        before = sorted(path.name for path in tmp_path.rglob("*"))

        reanalysed = corpus.rescore_kept(candidate, tmp_path, seeds, reanalyse=True)
        assert calls == [2], "one pool for every kept seed"
        assert sorted(path.name for path in tmp_path.rglob("*")) == before, "writes nothing"

        # With each seed's analysis on disk, nothing is analysed and the totals agree.
        weighting = corpus.load_weights(corpus.DEFAULT_WEIGHTS)
        for seed in seeds:
            out = corpus.run_dir(candidate.hash, seed, tmp_path)
            corpus._rescore_seed(
                candidate, seed, out, weighting, 1, lambda _: None, reanalyse=True
            )
        calls.clear()
        assert corpus.rescore_kept(candidate, tmp_path, seeds) == reanalysed
        assert calls == []
