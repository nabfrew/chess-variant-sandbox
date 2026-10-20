"""Reproducibility, scheduling, and rating.

The reproducibility tests are the load-bearing ones. Everything downstream assumes a
game is a pure function of (ruleset, seed, agents); if that breaks, a surprising
tournament result stops being investigable and the whole harness becomes anecdote.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_variant_sandbox.agents import parse_agent
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.tournament import (
    GameRecord,
    TournamentConfig,
    play_game,
    rate,
    read_records,
    replay,
    run_tournament,
    schedule,
    write_records,
)
from chess_variant_sandbox.tournament.play import seat_rng


@pytest.fixture(scope="module")
def small_spec() -> dict:
    """A quick game to run tournaments over: 5x5, two piece types, royal capture."""
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


class TestReproducibility:
    def test_the_same_seed_replays_the_same_game(self, small_spec: dict) -> None:
        engine = Engine(small_spec)
        pair = ("greedy", "minimax:nodes=200")
        first = play_game(engine, pair, seed=7)
        second = play_game(engine, pair, seed=7)
        assert first.moves == second.moves
        assert (first.winner, first.reason) == (second.winner, second.reason)

    def test_different_seeds_diverge(self, small_spec: dict) -> None:
        engine = Engine(small_spec)
        games = {tuple(play_game(engine, ("random", "random"), seed=s).moves) for s in range(8)}
        assert len(games) > 1

    def test_a_fresh_engine_reproduces_a_recorded_game(self, small_spec: dict) -> None:
        """A record has to mean the same thing in the process that reads it back."""
        record = play_game(Engine(small_spec), ("greedy", "greedy"), seed=3)
        states = replay(Engine(small_spec), record)
        assert len(states) == len(record.moves) + 1

    def test_seat_streams_are_independent(self) -> None:
        """One agent's appetite for randomness must not shift the other's draws.

        Otherwise swapping in a differently-implemented agent silently changes the
        opponent's choices too, and a strength comparison measures nothing.
        """
        first = seat_rng(11, 0)
        second = seat_rng(11, 1)
        assert [first.random() for _ in range(5)] != [second.random() for _ in range(5)]
        assert [seat_rng(11, 0).random() for _ in range(5)] == [
            v for v in [seat_rng(11, 0).random() for _ in range(5)]
        ]


class TestSchedule:
    def test_every_pairing_is_played_in_both_seats(self) -> None:
        config = TournamentConfig(agents=("random", "greedy"), games_per_pairing=2)
        tasks = schedule(config)
        pairs = [(first, second) for _, first, second in tasks]
        assert pairs.count(("random", "greedy")) == pairs.count(("greedy", "random")) == 2

    def test_self_play_is_included_by_default(self) -> None:
        """Self-play is the only matchup where strength cancels, so balance needs it."""
        tasks = schedule(TournamentConfig(agents=("random", "greedy"), games_per_pairing=1))
        assert ("random", "random") in [(first, second) for _, first, second in tasks]

    def test_self_play_can_be_switched_off(self) -> None:
        tasks = schedule(
            TournamentConfig(agents=("random", "greedy"), games_per_pairing=1, include_self_play=False)
        )
        assert ("random", "random") not in [(first, second) for _, first, second in tasks]

    def test_seeds_are_unique_so_games_are_not_secretly_repeats(self) -> None:
        tasks = schedule(TournamentConfig(agents=("random", "greedy", "mcts:nodes=50"), games_per_pairing=3))
        seeds = [seed for seed, _, _ in tasks]
        assert len(set(seeds)) == len(seeds)

    def test_a_bad_agent_name_fails_before_any_game_runs(self) -> None:
        with pytest.raises(ValueError, match="unknown agent"):
            schedule(TournamentConfig(agents=("random", "telepath")))


class TestRatings:
    def _record(self, first: str, second: str, winner: int | None) -> GameRecord:
        return GameRecord(
            ruleset_hash="x",
            ruleset_name="x",
            seed=0,
            agents=(first, second),
            moves=[1, 2],
            winner=winner,
            reason="test",
            plies=2,
        )

    def test_a_consistent_winner_rates_higher(self) -> None:
        records = [self._record("strong", "weak", 0) for _ in range(10)]
        records += [self._record("weak", "strong", 1) for _ in range(10)]
        table = rate(records)
        assert table.rating_of("strong") > table.rating_of("weak")

    def test_ratings_do_not_depend_on_the_order_games_were_played(self) -> None:
        """The reason for fitting rather than updating sequentially.

        An index that moves when the games are shuffled would be reporting noise as
        design signal.
        """
        records = (
            [self._record("a", "b", 0) for _ in range(6)]
            + [self._record("b", "a", 0) for _ in range(3)]
            + [self._record("a", "c", None) for _ in range(4)]
            + [self._record("c", "b", 1) for _ in range(5)]
        )
        forward = rate(records)
        backward = rate(list(reversed(records)))
        for standing in forward.standings:
            assert standing.rating == pytest.approx(backward.rating_of(standing.name), abs=0.2)

    def test_an_unbeaten_player_still_gets_a_finite_rating(self) -> None:
        """Without the prior this is where the fit fails to converge, and it is common."""
        records = [self._record("perfect", "hopeless", 0) for _ in range(20)]
        table = rate(records)
        assert table.rating_of("perfect") < float("inf")
        assert table.spread() > 0

    def test_first_player_score_is_tracked_per_seat(self) -> None:
        records = [self._record("a", "a", 0) for _ in range(4)]
        table = rate(records)
        assert table.standings[0].first_player_score == 1.0


class TestRecords:
    def test_records_round_trip_through_disk(self, tmp_path, small_spec: dict) -> None:
        engine = Engine(small_spec)
        original = [play_game(engine, ("random", "greedy"), seed=s) for s in range(3)]
        path = write_records(original, tmp_path / "games.jsonl")
        restored = list(read_records(path))
        assert [r.moves for r in restored] == [r.moves for r in original]
        assert [r.winner for r in restored] == [r.winner for r in original]

    def test_a_truncated_final_line_is_skipped_not_fatal(self, tmp_path) -> None:
        """A killed sweep should still leave every completed game readable."""
        path = tmp_path / "games.jsonl"
        record = GameRecord("h", "n", 0, ("a", "b"), [1], 0, "test", 1)
        write_records([record], path)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write('{"ruleset_hash": "h", "trunc')
        assert len(list(read_records(path))) == 1

    def test_a_corpus_recorded_before_depth_was_carried_still_reads(self) -> None:
        """``from_dict`` does ``cls(**payload)``, so a new field has to default.

        Every run under ``runs/`` predates this pair and is still a valid corpus: the
        index reads persisted records, so a field that made an old file unreadable
        would retire the corpus rather than extend it.
        """
        payload = GameRecord("h", "n", 0, ("a", "b"), [1], 0, "test", 1).as_dict()
        del payload["depth_sum"], payload["searches"]
        restored = GameRecord.from_dict(payload)
        assert (restored.depth_sum, restored.searches) == ((0, 0), (0, 0))
        assert restored.mean_depth(0) is None

    def test_how_deep_a_search_got_survives_the_round_trip(
        self, tmp_path, small_spec: dict
    ) -> None:
        engine = Engine(small_spec)
        original = play_game(engine, ("minimax:nodes=400", "random"), seed=1)
        restored = next(iter(read_records(write_records([original], tmp_path / "g.jsonl"))))

        assert restored.depth_sum == original.depth_sum
        assert restored.searches == original.searches
        # The searching seat searched; the random one did not, and the difference is
        # the point -- an agent with no depth is absent from the read-out rather than
        # sitting at zero beside a search that completed no ply.
        assert original.searches[0] > 0 and original.depth_sum[0] > 0
        assert original.searches[1] == 0
        assert original.mean_depth(0) is not None and original.mean_depth(1) is None


@pytest.mark.slow
def test_a_full_tournament_produces_a_rateable_corpus(small_spec: dict) -> None:
    config = TournamentConfig(
        agents=("random", "greedy", "minimax:nodes=300"), games_per_pairing=2, workers=1
    )
    result = run_tournament(small_spec, config)
    assert len(result.records) == 9 * 2
    assert len(result.ratings.standings) == 3
    assert result.ratings.rating_of("minimax:nodes=300") > result.ratings.rating_of("random")


def test_agent_config_round_trips_through_its_text_form() -> None:
    """Agent identity ends up in game records, so two spellings must not diverge."""
    config = parse_agent("minimax:nodes=5000,mobility=0.1")
    assert str(parse_agent(str(config))) == str(config)


@pytest.mark.parametrize(
    "agent",
    [
        # Q558 retired the fitted objective; stored rosters and records still spell it.
        "minimax:fit=98f172aeac49,nodes=100",
        "greedy:fit=98f172aeac49",
        "matrix:fit=98f172aeac49,nodes=100",
        "minimax:nodes=100,depth=3",
    ],
)
def test_an_option_no_player_reads_is_refused(small_spec: dict, agent: str) -> None:
    """A player sat down without an option it was named with would play something else
    under that name, so an unread option is refused, natively and in the oracle alike."""
    from chess_variant_sandbox.agents import build_agent, native_spec

    engine = Engine(small_spec)
    with pytest.raises(ValueError, match="reads no option"):
        native_spec(agent, engine.spec.hash)
    with pytest.raises(ValueError, match="reads no option"):
        build_agent(agent, engine)


class TestWhereTheCommandWrites:
    """Q335: `chess-variant-sandbox tournament` with no `--out` cannot land on a library run.

    A library game's runs are `runs/<hash>-<seed>` and its measurement is `runs/<hash>`.
    The command's default used to be the first of those, so a one-off at seed 0 on any
    roster replaced a library run's games while the measurement beside them went on
    reading current. `corpus._foreign_run` guards the other direction -- `measure` will
    not replay over somebody's one-off -- and this holds the one-off out of the library.
    """

    def test_the_default_is_no_slot_the_library_uses(self) -> None:
        from chess_variant_sandbox import cli, corpus

        digest = "0123456789ab"
        runs = corpus.RUNS_DIR
        library = {corpus.measurement_dir(digest, runs)} | {
            corpus.run_dir(digest, seed, runs) for seed in corpus.LIBRARY_SEEDS
        }
        assert runs == cli.RUNS_DIR  # the root both modules mean by `runs/`
        for seed in corpus.LIBRARY_SEEDS:
            out = cli._tournament_dir(digest, seed)
            assert out not in library
            # Structurally, not only for these seeds: every library slot sits directly
            # under the runs root, and the default sits one directory further down.
            assert out.parent != runs and out.is_relative_to(runs)

    def test_a_one_off_leaves_the_library_run_it_used_to_overwrite_alone(
        self, small_spec: dict, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        from chess_variant_sandbox import cli, corpus

        spec_path = tmp_path / "skirmish.json"
        spec_path.write_text(json.dumps(small_spec), encoding="utf-8")
        digest = spec_hash(load_spec(spec_path))
        monkeypatch.chdir(tmp_path)  # `runs/` is relative, so this is the real default
        runs = Path("runs")
        library = {
            corpus.run_dir(digest, 0, runs) / "games.jsonl": "the library's games\n",
            corpus.run_dir(digest, 0, runs) / "tournament.json": '{"config": {}}',
            corpus.measurement_dir(digest, runs) / "canonical.json": '{"index": 0.5}',
        }
        for path, text in library.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

        argv = ["tournament", str(spec_path), "--agents", "random", "greedy",
                "--games", "1", "--workers", "1"]
        assert cli.main(argv) == 0

        for path, text in library.items():
            assert path.read_text(encoding="utf-8") == text, f"{path} was written over"
        out = cli._tournament_dir(digest, 0)
        assert out == runs / "tournament" / f"{digest}-0"
        assert len(list(read_records(out / "games.jsonl"))) == 4  # self-play included
        assert json.loads((out / "tournament.json").read_text(encoding="utf-8"))[
            "config"
        ]["agents"] == ["random", "greedy"]
        assert str(out) in capsys.readouterr().out
        assert {p.name for p in runs.iterdir()} == {digest, f"{digest}-0", "tournament"}

    def test_the_help_says_where_output_goes(self, capsys) -> None:
        from chess_variant_sandbox import cli

        with pytest.raises(SystemExit):
            cli.main(["tournament", "--help"])
        said = capsys.readouterr().out
        assert "runs/tournament/<hash>-<seed>" in said
        assert "never a library run" in " ".join(said.split())


# ------------------------------------------------ a roster the core cannot all run ----


class _Outsider:
    """A player the core has never heard of, as Fairy-Stockfish is: its moves come from
    Python, and from its seat's own stream, so a seed still decides its game."""

    def __init__(self) -> None:
        from chess_variant_sandbox.agents import SearchStats

        self.name = "outsider"
        self.stats = SearchStats()

    def select_move(self, state: object, moves: list[int], rng) -> int:
        self.stats.moves += 1
        return moves[rng.randrange(len(moves))]


@pytest.fixture
def outsider(monkeypatch: pytest.MonkeyPatch) -> str:
    """Registered for one test only, where `register` would leave it behind for all."""
    from chess_variant_sandbox.agents import base

    monkeypatch.setitem(base._REGISTRY, "outsider", lambda engine, options: _Outsider())
    return "outsider"


def _as_played(records: list[GameRecord]) -> list[dict]:
    """Every field of each record, in order, but the one a clock writes."""
    return [
        {key: value for key, value in record.as_dict().items() if key != "duration_ms"}
        for record in records
    ]


def _all_in_the_loop(spec: dict, config: TournamentConfig) -> list[GameRecord]:
    """What `run_tournament` played for any roster with an outside player before Q439:
    the whole schedule through the Python loop, one game after another."""
    from chess_variant_sandbox.tournament import runner

    engine = Engine(spec)
    return list(runner._run_python(engine, schedule(config), config.collect_series, False))


class TestAMixedRosterIsSplitByNativeForm:
    """Q439: one outside player no longer sends every game through the Python loop.

    The pairings where both seats are native go to the core as one schedule, and only
    the ones with an outside seat stay in the loop. The loop already handed each native
    pairing to the core, a game per call, so what the split can change is speed and
    nothing else -- which is what these hold it to: the same games, field for field
    and in schedule order, as the whole schedule played through the loop.
    """

    def test_it_plays_the_games_the_loop_played_in_the_order_it_played_them(
        self, small_spec: dict, outsider: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = TournamentConfig(
            agents=("random", outsider, "greedy", "minimax:nodes=200"),
            games_per_pairing=2, seed=11, workers=2,
        )
        split = run_tournament(small_spec, config, progress=True).records

        assert _as_played(split) == _as_played(_all_in_the_loop(small_spec, config))
        assert [(r.seed, *r.agents) for r in split] == schedule(config)
        said = capsys.readouterr().out.splitlines()
        assert "  18 games in the core; 14 with outsider one at a time" in said
        # The core's pieces are two games, one per thread, and never one game on a pool.
        assert [line for line in said if line.endswith("/32 games")] == [
            *(f"  {done}/32 games" for done in range(2, 19, 2)), "  32/32 games"
        ]

    def test_only_the_pairings_with_an_outside_seat_stay_in_the_loop(
        self, small_spec: dict, outsider: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chess_variant_sandbox.tournament import runner

        handed: dict[str, list] = {"core": [], "loop": []}
        native, loop = runner._run_native, runner._run_python

        def to_core(engine, config, tasks, natives, progress, **kwargs):
            handed["core"].append(list(tasks))
            return native(engine, config, tasks, natives, progress, **kwargs)

        def to_loop(engine, tasks, collect_series, progress, **kwargs):
            handed["loop"].append(list(tasks))
            return loop(engine, tasks, collect_series, progress, **kwargs)

        monkeypatch.setattr(runner, "_run_native", to_core)
        monkeypatch.setattr(runner, "_run_python", to_loop)

        config = TournamentConfig(
            agents=("random", outsider, "greedy"), games_per_pairing=1, seed=4
        )
        run_tournament(small_spec, config)
        tasks = schedule(config)
        assert handed["core"] == [[t for t in tasks if outsider not in t]]  # one schedule
        assert handed["loop"] == [[t for t in tasks if outsider in t]]

        # And a roster the core can run whole never reaches the loop at all.
        handed.update(core=[], loop=[])
        native_only = TournamentConfig(agents=("random", "greedy"), games_per_pairing=1)
        run_tournament(small_spec, native_only)
        assert handed == {"core": [schedule(native_only)], "loop": []}


class TestTheRunnerTellsItsCount:
    """Q446: `on_progress` hears how many games are back, as they come back.

    The picker's rating run is one `run_tournament` call now, and its page counts the
    outside engine's games down while they play. The core's games come back in one
    piece unless `progress` asks for twenty, so a quiet run is told once for them and
    then once per game the Python loop plays.
    """

    def test_once_for_the_core_then_once_per_outside_game(
        self, small_spec: dict, outsider: str
    ) -> None:
        config = TournamentConfig(
            agents=("random", outsider, "greedy"), games_per_pairing=1, seed=4
        )
        told: list[int] = []
        records = run_tournament(small_spec, config, on_progress=told.append).records
        tasks = schedule(config)
        core = sum(outsider not in (first, second) for _, first, second in tasks)
        assert 0 < core < len(tasks)
        assert told == list(range(core, len(tasks) + 1))
        # Told, never asked: the games are the ones a run with nobody listening plays.
        assert _as_played(records) == _as_played(run_tournament(small_spec, config).records)

    def test_each_piece_the_core_returns_is_told_when_progress_asks_for_pieces(
        self, small_spec: dict, outsider: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = TournamentConfig(
            agents=("random", outsider, "greedy", "minimax:nodes=200"),
            games_per_pairing=2, seed=11, workers=2,
        )
        told: list[int] = []
        run_tournament(small_spec, config, progress=True, on_progress=told.append)
        capsys.readouterr()
        # Nine pieces of two in the core, then the loop's fourteen one at a time.
        assert told == [*range(2, 19, 2), *range(19, 33)]

    def test_a_roster_the_core_runs_whole_is_told_once(self, small_spec: dict) -> None:
        config = TournamentConfig(agents=("random", "greedy"), games_per_pairing=1)
        told: list[int] = []
        run_tournament(small_spec, config, on_progress=told.append)
        assert told == [len(schedule(config))]

    @pytest.mark.slow
    def test_with_fairy_stockfish_it_plays_the_games_the_loop_played(self) -> None:
        """The case the split is for, against the real binary where there is one.

        Its search is its own, node-limited on one thread, so a game against it is
        reproducible on one build -- which is all this needs: the same build, the same
        seeds, the same games, whichever way the schedule was divided.

        Slow for the engine starts rather than the games: each of its games starts a
        process of its own, about 0.9 s on the 2021 binary against 10 ms of play, and the
        comparison plays each of its four games twice.
        """
        from chess_variant_sandbox import fairy

        spec = load_spec(Path(__file__).resolve().parents[1] / "rulesets" / "gardner.json")
        if not fairy.available():
            pytest.skip("no Fairy-Stockfish binary is installed")
        assert fairy.supports(spec)

        config = TournamentConfig(
            agents=("random", "fairystockfish:nodes=100", "greedy"),
            games_per_pairing=1, seed=5, workers=2, include_self_play=False,
        )
        split = run_tournament(spec, config).records

        assert _as_played(split) == _as_played(_all_in_the_loop(spec, config))
        assert [(r.seed, *r.agents) for r in split] == schedule(config)
        assert {r.agents for r in split if "fairystockfish:nodes=100" in r.agents} == {
            ("random", "fairystockfish:nodes=100"), ("fairystockfish:nodes=100", "random"),
            ("fairystockfish:nodes=100", "greedy"), ("greedy", "fairystockfish:nodes=100"),
        }
