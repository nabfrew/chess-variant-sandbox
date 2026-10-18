"""Small helpers that nothing in the suite called by name (archive/plan-tests F9, Q751b).

Each is a few lines the core or the loop leans on, so a change to one would move a number
or a file somewhere else with no test of its own turning red: the pooled search counters,
the evaluator's live-piece count, the engine's replay of a move list, the loop's recorded
configuration, the ladder allocation's game counts, the duel's lead unit, and the analysis
pool's per-worker state.
"""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chess_variant_sandbox.agents.base import SearchStats
from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.agents.features import _live_of, feature_context
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.loop.orchestrator import LoopConfig
from chess_variant_sandbox.tournament import analysis
from chess_variant_sandbox.tournament.duel import _army_scale
from chess_variant_sandbox.tournament.play import play_game
from chess_variant_sandbox.tournament.runner import (
    _LADDER_WEIGHTS,
    TournamentConfig,
    _pairing_kind,
)

RULESETS = Path(__file__).resolve().parent.parent / "rulesets"


def load(name: str) -> dict[str, Any]:
    return json.loads((RULESETS / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dull() -> Engine:
    return Engine(load("dull"))


class TestSearchStatsMerge:
    def test_every_counter_is_added_and_the_other_is_left_alone(self) -> None:
        mine = SearchStats(nodes=10, moves=3, depth_sum=7, searches=2)
        theirs = SearchStats(nodes=1000, moves=30, depth_sum=70, searches=20)
        mine.merge(theirs)
        assert dataclasses.astuple(mine) == (1010, 33, 77, 22)
        assert dataclasses.astuple(theirs) == (1000, 30, 70, 20)

    def test_pooling_games_gives_the_mean_depth_of_the_pool_not_of_the_means(self) -> None:
        """The two-number design exists so a corpus can be pooled: 2 deep + 4 deep = 3."""
        pool = SearchStats()
        pool.merge(SearchStats(depth_sum=2, searches=1))
        pool.merge(SearchStats(depth_sum=12, searches=3))
        assert pool.depth_sum / pool.searches == 3.5


class TestLiveOf:
    """How many of a type one side still has, whichever way each is facing."""

    def test_counts_every_orientation_of_the_type_for_that_side_only(self) -> None:
        # Two types, two facings: one army owns 4 board values. Side 0 holds 1..4
        # (facing 0: types 0, 1; facing 1: types 0, 1), side 1 holds 5..8, and a wild
        # piece sits in the next stride.
        ctx = SimpleNamespace(side_stride=4, n_types=2, n_squares=10)
        board = bytes([1, 3, 2, 5, 7, 7, 8, 9, 0, 3])
        assert _live_of(board, ctx, 0, 0) == 3  # 1, 3, 3: both facings of type 0
        assert _live_of(board, ctx, 1, 0) == 1  # 2
        assert _live_of(board, ctx, 0, 1) == 3  # 5, 7, 7
        assert _live_of(board, ctx, 1, 1) == 1  # 8

    def test_only_the_squares_of_the_board_are_read(self) -> None:
        """The board bytes may carry hands after the squares; they are not on the board."""
        ctx = SimpleNamespace(side_stride=2, n_types=2, n_squares=3)
        board = bytes([1, 0, 1, 1, 1, 1])
        assert _live_of(board, ctx, 0, 0) == 2

    def test_agrees_with_the_opening_position_of_a_real_spec(self, dull: Engine) -> None:
        ctx = feature_context(dull.spec)
        state = dull.initial_state()
        for side in (0, 1):
            for piece_type in range(dull.spec.n_types):
                expected = sum(
                    1
                    for square in range(dull.spec.n_squares)
                    if state.board[square]
                    and (state.board[square] - 1) // dull.spec.side_stride == side
                    and (state.board[square] - 1) % dull.spec.n_types == piece_type
                )
                assert _live_of(state.board, ctx, piece_type, side) == expected
        total = sum(
            _live_of(state.board, ctx, t, s)
            for s in (0, 1)
            for t in range(dull.spec.n_types)
        )
        assert total == sum(1 for v in state.board[: dull.spec.n_squares] if v)


class TestEnginePlay:
    def test_playing_a_list_is_applying_each_move_in_order(self, dull: Engine) -> None:
        rng = random.Random(3)
        state = dull.initial_state()
        moves: list[int] = []
        by_hand = state
        for _ in range(6):
            move = rng.choice(dull.legal_moves(by_hand))
            moves.append(move)
            by_hand = dull.apply(by_hand, move)
        played = dull.play(state, moves)
        assert played.board == by_hand.board
        assert dull.hash(played) == dull.hash(by_hand)
        assert played.side_to_move == by_hand.side_to_move
        assert played.side_to_move == len(moves) % 2

    def test_no_moves_is_the_same_position_and_the_argument_is_not_changed(
        self, dull: Engine
    ) -> None:
        state = dull.initial_state()
        before = (state.board, dull.hash(state))
        assert dull.hash(dull.play(state, [])) == before[1]
        dull.play(state, dull.legal_moves(state)[:1])
        assert (state.board, dull.hash(state)) == before


class TestLoopConfigAsDict:
    def _config(self, tmp_path: Path, **over: Any) -> LoopConfig:
        return LoopConfig(
            generations=2,
            directory=tmp_path / "design",
            seeds=(Path("rulesets") / "dull.json", Path("rulesets") / "chess.json"),
            **over,
        )

    def test_is_plain_json_with_paths_and_the_roster_as_strings_and_lists(
        self, tmp_path: Path
    ) -> None:
        config = self._config(tmp_path, theme="a game with no draws")
        data = config.as_dict()
        assert json.loads(json.dumps(data)) == data
        assert data["directory"] == str(tmp_path / "design")
        assert data["seeds"] == [str(Path("rulesets/dull.json")), str(Path("rulesets/chess.json"))]
        assert data["agents"] == list(config.agents)
        assert isinstance(data["agents"], list)
        assert data["theme"] == "a game with no draws"

    def test_carries_every_field_of_the_config_and_nothing_else(self, tmp_path: Path) -> None:
        config = self._config(tmp_path, batch=4, promote=2)
        data = config.as_dict()
        assert set(data) == {f.name for f in dataclasses.fields(LoopConfig)}
        assert data["batch"] == 4 and data["promote"] == 2
        assert data["weights"] == config.weights

    def test_is_a_copy_and_leaves_the_config_unconverted(self, tmp_path: Path) -> None:
        config = self._config(tmp_path)
        config.as_dict()["agents"].append("sentinel")
        assert isinstance(config.agents, tuple) and "sentinel" not in config.agents
        assert isinstance(config.directory, Path)
        assert all(isinstance(p, Path) for p in config.seeds)


class TestPairingKind:
    @pytest.mark.parametrize(
        ("first", "second", "kind"),
        [
            ("random", "greedy", "weak"),
            ("minimax:nodes=400", "random", "weak"),
            ("random", "random", "weak"),
            ("minimax:nodes=400", "minimax:nodes=1600", "ladder"),
            ("mcts:nodes=400", "mcts:nodes=1600", "ladder"),
            ("greedy", "greedy", "ladder"),
            ("minimax:nodes=400", "greedy", "mixed"),
            ("mcts:nodes=400", "minimax:nodes=400", "mixed"),
        ],
    )
    def test_names_the_kind_from_the_agent_kinds_and_not_their_budgets(
        self, first: str, second: str, kind: str
    ) -> None:
        assert _pairing_kind(first, second) == kind
        assert _pairing_kind(second, first) == kind

    def test_a_ladder_schedule_gives_each_kind_its_weight_of_games(self) -> None:
        config = TournamentConfig(
            agents=("random", "greedy", "minimax:nodes=400"),
            games_per_pairing=2,
            allocation="ladder",
        )
        assert config.games_for("random", "minimax:nodes=400") == 2 * _LADDER_WEIGHTS["weak"]
        assert config.games_for("greedy", "minimax:nodes=400") == 2 * _LADDER_WEIGHTS["mixed"]
        assert config.games_for("minimax:nodes=400", "minimax:nodes=400") == (
            2 * _LADDER_WEIGHTS["ladder"]
        )
        uniform = dataclasses.replace(config, allocation="uniform")
        assert uniform.games_for("minimax:nodes=400", "minimax:nodes=400") == 2


class TestArmyScale:
    def test_is_one_armys_worth_of_the_derived_table(self, dull: Engine) -> None:
        spec = dull.spec
        values = Evaluator.for_spec(spec).values
        # Independent of the function's own loop: count each type on the board, then
        # price it. Both armies are set up alike in `dull`, so one army is half of all.
        per_type = [0] * spec.n_types
        for square in range(spec.n_squares):
            value = dull.initial_state().board[square]
            if value:
                per_type[(value - 1) % spec.n_types] += 1
        everything = sum(abs(values[t]) * n for t, n in enumerate(per_type))
        assert everything > 0
        assert _army_scale(spec) == pytest.approx(everything / 2)

    def test_a_bigger_army_is_a_bigger_unit(self) -> None:
        small = Engine(load("losalamos")).spec
        large = Engine(load("capablanca")).spec
        assert _army_scale(large) > _army_scale(small) > 0


class TestAnalysisWorkerState:
    @pytest.fixture
    def worker_globals(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Restore the module's per-process state: in a pool each child owns its own."""
        for name in ("_WORKER_ENGINE", "_WORKER_EVALUATOR", "_WORKER_EPSILON"):
            monkeypatch.setattr(analysis, name, getattr(analysis, name))

    def test_init_worker_builds_the_engine_the_evaluator_and_the_epsilon(
        self, worker_globals: None
    ) -> None:
        spec = load("xiangqi")  # its cheapest piece is worth 1.25
        analysis._init_worker(spec)
        engine = analysis._WORKER_ENGINE
        assert isinstance(engine, Engine)
        assert engine.spec.n_squares == Engine(spec).spec.n_squares
        assert isinstance(analysis._WORKER_EVALUATOR, Evaluator)
        # the default of 1.0 is what an uninitialised worker would carry
        assert analysis._epsilon_for(engine) != 1.0
        assert analysis._epsilon_for(engine) == analysis._WORKER_EPSILON

    def test_a_blunder_task_is_the_serial_trial_on_the_same_stream(
        self, worker_globals: None
    ) -> None:
        spec = load("dull")
        engine = Engine(spec)
        record = next(
            r
            for r in (play_game(engine, ("random", "random"), seed) for seed in range(40))
            if r.plies >= 12
        )
        analysis._init_worker(spec)
        trial = analysis._blunder_one((record, 5))
        expected = analysis._one_blunder(engine, record, analysis.blunder_rng(5, record))
        assert trial is not None
        assert trial == expected
        assert trial.seed == record.seed and trial.agents == record.agents
        assert trial.ply == int(record.plies * analysis.BLUNDER_FRACTION)
        assert trial.original_winner == record.winner

    def test_a_blunder_task_draws_from_the_stream_keyed_by_its_run_seed_and_record(
        self, worker_globals: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = load("dull")
        engine = Engine(spec)
        record = play_game(engine, ("random", "random"), 1)
        analysis._init_worker(spec)
        seen: list[float] = []

        def spy(_engine: Engine, _record: Any, rng: random.Random) -> None:
            seen.append(rng.random())

        monkeypatch.setattr(analysis, "_one_blunder", spy)
        for seed in (5, 6):
            analysis._blunder_one((record, seed))
        assert seen == [
            analysis.blunder_rng(5, record).random(),
            analysis.blunder_rng(6, record).random(),
        ]
        assert seen[0] != seen[1]

    def test_a_task_before_the_worker_is_initialised_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(analysis, "_WORKER_ENGINE", None)
        with pytest.raises(AssertionError):
            analysis._blunder_one((None, 0))  # type: ignore[arg-type]
