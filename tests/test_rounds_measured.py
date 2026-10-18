"""Schema 26 (Q918d): a game played in rounds, recorded, analysed and scored end to end.

`docs/archive/plan-simultaneous.md` §6.3 and §7's Q918d box. The record keeps both moves of a
round in seat order and every per-ply series twice; the analysis pass replays it a round at
a time, reads each player's moves against the opponent's equilibrium mix, forces blunders
by that reading, and counts what the resolutions did; every scored component says how it
reads rounds; the measuring roster is the round players' ladder. Asserted against specs:
the mirrored Gardner-sized test spec and a matching-pennies position whose equilibrium is
known.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
from test_matrix_players import PENNIES

from chess_variant_sandbox import corpus
from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS, load_weights, score_ruleset
from chess_variant_sandbox.metrics.components import ROUND_READINGS, compute_components
from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.tournament import TournamentConfig, run_tournament
from chess_variant_sandbox.tournament.analysis import (
    DUST,
    ROUND_COUNTS,
    CorpusAnalysis,
    GameAnalysis,
    _analyse_game,
    _epsilon_for,
    _is_swap,
    _one_blunder,
    _purified,
    analyse_corpus,
    blunder_rng,
    read_matrix,
    read_round,
    replay_rounds,
    round_blunder,
)
from chess_variant_sandbox.tournament.play import eval_values
from chess_variant_sandbox.tournament.record import GameRecord, read_records, write_records

HERE = Path(__file__).parent
ROOT = HERE.parent
GARDNER = HERE / "_simultaneous_gardner.json"
# A round game the ladder does not hold (Gardner's is archived since Q918e).
OFF_LADDER = HERE / "_simultaneous_shogi.json"


def gardner(max_plies: int = 40) -> dict[str, Any]:
    spec = json.loads(GARDNER.read_text(encoding="utf-8"))
    spec["rules"]["max_plies"] = max_plies
    return spec


@pytest.fixture(scope="module")
def played() -> tuple[dict[str, Any], Any]:
    """A small round tournament: every pairing of three round players, two games each."""
    spec = gardner()
    result = run_tournament(
        spec,
        TournamentConfig(
            agents=("random", "greedy", "matrix:nodes=1600"), games_per_pairing=2, seed=7,
            workers=1,
        ),
    )
    return spec, result


@pytest.fixture(scope="module")
def analysed(played) -> CorpusAnalysis:
    spec, result = played
    return analyse_corpus(spec, result.records, blunder_sample=8, seed=0, workers=1)


# --------------------------------------------------------------------- records


class TestTheRecord:
    def test_a_round_is_two_plies_and_every_series_is_written_twice(self, played) -> None:
        _, result = played
        for record in result.records:
            assert record.plies == len(record.moves) and record.plies % 2 == 0
            assert len(record.branching) == record.plies
            for series in (record.material, record.captures):
                assert len(series) == record.plies
                assert series[0::2] == series[1::2]

    def test_branching_is_each_players_own_count(self, played) -> None:
        """`mean_branching` per player (§6.3): a round writes player 0's legal moves on its
        first ply and player 1's on its second, never the matrix's size."""
        spec, result = played
        engine = Engine(spec)
        record = next(r for r in result.records if r.plies >= 6)
        states = replay_rounds(engine, record)
        for index, state in enumerate(states[:-1]):
            assert record.branching[2 * index] == len(engine.legal_moves_for(state, 0))
            assert record.branching[2 * index + 1] == len(engine.legal_moves_for(state, 1))

    def test_a_record_replays_to_its_own_ending_and_series(self, played, tmp_path) -> None:
        """Both moves and the resolution, faithfully: written and read back, the record
        replays round by round to the verdict and the material it recorded."""
        spec, result = played
        engine = Engine(spec)
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        path = write_records(result.records, tmp_path / "games.jsonl")
        for original, read in zip(result.records, read_records(path), strict=True):
            assert read == original
            states = replay_rounds(engine, read)
            outcome = engine.terminal(states[-1])
            assert outcome is not None
            assert (outcome.winner, outcome.reason) == (read.winner, read.reason)
            for index, state in enumerate(states[:-1]):
                assert read.material[2 * index] == round(evaluator.material(state, 0), 4)

    def test_the_alternating_replay_refuses_a_round_game_by_name(self, played) -> None:
        """`play.replay` walks one move at a time; a round game's record is walked by
        `replay_rounds`, and the core says so at the first move rather than mis-reading it."""
        from chess_variant_sandbox.tournament.play import replay

        spec, result = played
        with pytest.raises(ValueError, match="played in rounds"):
            replay(Engine(spec), result.records[0])

    def test_a_record_that_drifted_from_its_spec_is_refused(self, played) -> None:
        spec, result = played
        record = result.records[0]
        swapped = GameRecord(**{**record.__dict__, "moves": record.moves[1:2] + record.moves[0:1]
                                + record.moves[2:]})
        with pytest.raises(ValueError, match="not legal"):
            replay_rounds(Engine(spec), swapped)


# ------------------------------------------------------------------- the probe


class TestTheProbeOnAKnownEquilibrium:
    """Matching pennies (`test_matrix_players.PENNIES`): player 0's A guesses where player
    1's royal K steps, a1 or c1, and meeting it wins; stepping onto b2 instead meets
    nothing. The equilibrium is half and half on the two guesses for both players, so
    each guess is worth half a win to player 0 and the step to b2 nothing."""

    @pytest.fixture
    def reading(self):
        engine = Engine(PENNIES)
        state = engine.initial_state()
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        return engine, state, read_round(engine, evaluator, state)

    def test_the_mixes_are_half_and_half(self, reading) -> None:
        engine, state, found = reading
        named = {engine.describe_move(state, m): w
                 for m, w in zip(found.moves[0], found.strategies[0], strict=True)}
        assert named["Ab1-a1"] == pytest.approx(0.5) and named["Ab1-c1"] == pytest.approx(0.5)
        assert named["Ab1xb2"] == 0.0
        assert found.strategies[1] == pytest.approx([0.5, 0.5])

    def test_the_step_aside_is_the_one_blunder(self, reading) -> None:
        engine, state, found = reading
        epsilon = _epsilon_for(engine)
        assert found.near_best(0, epsilon) == 2
        assert [engine.describe_move(state, m) for m in found.blunders(0, epsilon)] == ["Ab1xb2"]
        # The royal's two steps are worth the same against A's mix: no blunder to force.
        assert found.near_best(1, epsilon) == 2 and found.blunders(1, epsilon) == []

    def test_a_trial_forces_the_blunder_and_only_it(self, reading) -> None:
        engine, state, found = reading
        guess = found.moves[0][[engine.describe_move(state, m) for m in found.moves[0]]
                               .index("Ab1-a1")]
        forced = round_blunder(engine, state, 0, guess, random.Random(1))
        assert engine.describe_move(state, forced) == "Ab1xb2"
        assert round_blunder(engine, state, 1, found.moves[1][0], random.Random(1)) is None

    def test_the_solvers_residue_is_cleared(self) -> None:
        assert _purified([0.5, DUST / 2, 0.5 - DUST / 2]) == pytest.approx([0.5 / (1 - DUST / 2), 0.0,
                                                                          (0.5 - DUST / 2) / (1 - DUST / 2)])
        assert _purified([1.0, 0.0]) == [1.0, 0.0]

    def test_a_dominated_row_is_read_from_the_matrix_alone(self) -> None:
        """`read_matrix` is a pure function of the moves and the cells."""
        found = read_matrix(([10, 11, 12], [20, 21]), [1.0, -1.0, -1.0, 1.0, -5.0, -5.0])
        assert found.values[0] == pytest.approx([0.0, 0.0, -5.0])
        assert found.blunders(0, 1.0) == [12] and found.blunders(1, 1.0) == []


# ------------------------------------------------------------- the pass itself


class TestThePass:
    def test_every_round_is_walked_and_counted(self, played, analysed) -> None:
        _, result = played
        graded = {(r.seed, r.agents): r for r in result.records if "random" not in r.agents}
        assert len(analysed.games) == len(graded)
        for game in analysed.games:
            record = graded[(game.seed, game.agents)]
            assert game.rounds is not None and set(game.rounds) == set(ROUND_COUNTS)
            assert game.rounds["rounds"] * 2 == record.plies == game.walked_plies
            assert len(game.position_keys) == record.plies
            assert game.position_keys[0::2] == game.position_keys[1::2]
            assert sum(game.moves_by_type) == record.plies
            assert len(game.near_best) == len(game.legal_counts) == len(game.capture_counts)
            assert all(0 < n <= legal for n, legal in zip(game.near_best, game.legal_counts,
                                                           strict=True))

    def test_a_round_games_file_round_trips(self, analysed, tmp_path) -> None:
        path = analysed.save(tmp_path / "analysis.json")
        assert CorpusAnalysis.load(path).as_dict() == analysed.as_dict()

    def test_an_alternating_games_file_holds_no_round_field(self) -> None:
        assert "rounds" not in GameAnalysis(seed=0, agents=("a", "b")).as_dict()

    def test_a_collision_both_lose_is_counted(self) -> None:
        engine = Engine(PENNIES)
        state = engine.initial_state()
        pick = {engine.describe_move(state, m): m for p in (0, 1)
                for m in engine.legal_moves_for(state, p)}
        record = GameRecord(
            ruleset_hash=engine.spec.hash, ruleset_name="pennies", seed=0,
            agents=("greedy", "greedy"), moves=[pick["Ab1-a1"], pick["Kb2-a1"]],
            winner=0, reason="royal_capture", plies=2,
        )
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        game = _analyse_game(engine, evaluator, record, _epsilon_for(engine))
        assert game.rounds is not None
        assert game.rounds["collisions"] == 1 and game.rounds["collisions_both_lost"] == 1
        assert game.rounds["void_moves"] == 0 and game.rounds["swaps"] == 0

    def test_a_swap_is_read_off_the_words(self) -> None:
        from chess_variant_sandbox.engine.moves import MOVE_DROP, encode_move

        assert _is_swap(encode_move(3, 8), encode_move(8, 3))
        assert not _is_swap(encode_move(3, 8), encode_move(8, 4))
        assert not _is_swap(encode_move(3, 8, MOVE_DROP), encode_move(8, 3))

    def test_the_blunder_trials_name_a_seat_and_a_round(self, played, analysed) -> None:
        _, result = played
        assert analysed.blunders, "no trial was drawn, so nothing was tested"
        by_game = {(r.seed, r.agents): r for r in result.records}
        for trial in analysed.blunders:
            record = by_game[(trial.seed, trial.agents)]
            assert trial.ply == int(record.plies * 0.34)
            assert trial.blundering_side == trial.ply % 2

    def test_a_trial_is_fixed_by_its_seed_and_record(self, played) -> None:
        spec, result = played
        engine = Engine(spec)
        record = next(r for r in result.records if "random" not in r.agents and r.plies >= 12)
        first = _one_blunder(engine, record, blunder_rng(0, record))
        assert first == _one_blunder(engine, record, blunder_rng(0, record))


# ------------------------------------------------------------------ the score


class TestTheScore:
    def test_every_scored_component_says_how_it_reads_rounds(self) -> None:
        weights = load_weights(DEFAULT_WEIGHTS)
        assert set(weights.weights) <= set(ROUND_READINGS)

    def test_a_round_corpus_scores_with_no_component_absent(self, played, analysed) -> None:
        spec, result = played
        components = compute_components(spec, result.records, result.ratings, analysed)
        weights = load_weights(DEFAULT_WEIGHTS)
        absent = [name for name in weights.weights
                  if name not in components.raw and name not in weights.optional]
        assert absent == []
        assert components.diagnostics["round_readings"] == ROUND_READINGS
        tallies = components.diagnostics["rounds"]
        assert tallies["rounds"] == sum(g.rounds["rounds"] for g in analysed.games)
        assert 0.0 <= tallies["collisions_per_round"] <= 1.0

    def test_the_score_carries_the_readings(self, played, analysed) -> None:
        spec, result = played
        scored = score_ruleset(spec, result.records, result.ratings, analysis=analysed,
                               interval=False)
        assert scored.diagnostics["round_readings"] == ROUND_READINGS

    def test_an_alternating_score_carries_no_readings(self) -> None:
        spec = load_spec(ROOT / "rulesets" / "gardner.json")
        result = run_tournament(spec, TournamentConfig(agents=("random", "greedy"),
                                                       games_per_pairing=1, seed=3, workers=1))
        components = compute_components(spec, result.records, result.ratings)
        assert "round_readings" not in components.diagnostics


# ----------------------------------------------------------------- the roster


class TestTheRoundRoster:
    def test_a_round_game_is_measured_on_the_round_players(self) -> None:
        spec = gardner()
        digest = spec_hash(spec)
        assert corpus.canonical_agents(digest, spec=spec) == corpus.ROUND_AGENTS
        assert corpus.current_basis(digest, spec=spec, seeds=corpus.ONE_SEED).agents == (
            corpus.ROUND_AGENTS
        )

    def test_the_round_ladder_has_the_alternating_ladders_shape(self) -> None:
        """Eight places, the alternating ladder's budgets at sixteen times (Q944a,
        `round-roster-budgets.md` §6), the same factor the design loop's roster maps by."""
        from chess_variant_sandbox.loop.screen import ROUND_BUDGET_FACTOR, round_roster

        budgets = [int(a.split("nodes=")[1]) for a in corpus.CANONICAL_AGENTS if "nodes=" in a]
        assert ("random", "greedy") + tuple(
            f"matrix:nodes={budget * ROUND_BUDGET_FACTOR}" for budget in budgets
        ) == corpus.ROUND_AGENTS
        assert corpus.ROUND_AGENTS[2:] == tuple(
            f"matrix:nodes={n}" for n in (1600, 3200, 6400, 12800, 25600, 102400)
        )
        assert round_roster(
            ("random", "greedy") + tuple(f"minimax:nodes={b}" for b in budgets)
        ) == corpus.ROUND_AGENTS
        assert not corpus.fits_its_roster(corpus.ROUND_AGENTS)

    def test_an_alternating_game_keeps_its_roster(self) -> None:
        spec = load_spec(ROOT / "rulesets" / "gardner.json")
        digest = spec_hash(spec)
        assert corpus.canonical_agents(digest, spec=spec) == corpus.canonical_agents(digest)

    def test_the_round_players_hand_the_core_a_players_digest(self) -> None:
        """The stamp's `players` reads what the runner hands the core for every seat."""
        from chess_variant_sandbox.measurement import players_fingerprint

        spec = gardner()
        assert eval_values(Engine(spec).spec)
        assert players_fingerprint(spec, corpus.ROUND_AGENTS) != ""

    def test_measure_check_reads_a_file_off_the_ladder(self, tmp_path, capsys) -> None:
        """`measure --check <file>` names a round game the ladder does not hold, and says
        it was never measured under the runs directory given."""
        from chess_variant_sandbox import cli

        assert cli.main(["measure", "--check", str(OFF_LADDER), "--runs", str(tmp_path)]) == 1
        said = capsys.readouterr().out
        assert "simultaneous dobutsu" in said and "never measured" in said

    def test_a_file_off_the_ladder_is_a_candidate_where_it_stands(self) -> None:
        candidate = corpus.from_file(OFF_LADDER)
        assert candidate.stage == "candidate" and candidate.path == OFF_LADDER
        assert corpus.expected_seeds(candidate.stage) == corpus.ONE_SEED
        assert corpus.from_file(ROOT / "rulesets" / "gardner.json").stage == "committed"
        # From Q918e the Gardner test spec's mechanics were on the ladder, archived as
        # `archive/gardner-in-rounds`; Q962b retired that preview, so the file is a candidate
        # where it stands again.
        assert corpus.from_file(GARDNER).stage == "candidate"
