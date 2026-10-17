"""Schema 27 (Q935d): a spy game recorded, analysed, scored and measured end to end.

`docs/plan-spy.md` §3.8 and the Q935d row of §4. A spy game's record is the flat move list,
one word a turn and two a round, with every per-ply series once a turn and twice a round;
the analysis pass walks it a step at a time by the phase the engine reports, probes a turn
as an alternating ply (stopping at the seam into rounds), an announced round by what each
side knew and a blind round against the mix, forces blunders by the same readings, and
counts what the phases did (`SPY_COUNTS`); every scored component says how it reads a spy
game; the measuring roster is `corpus.SPY_AGENTS`. Asserted against the spy test specs
(`tests/_spy_*.json`) and hand-built positions on them.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import corpus
from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_PROMOTION, decode_move, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS, load_weights, score_ruleset
from chess_variant_sandbox.metrics.components import (
    SPY_READINGS,
    compute_components,
    plays_with_spies,
    spy_summary,
)
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.tournament import TournamentConfig, run_tournament
from chess_variant_sandbox.tournament.analysis import (
    PLY_STRIDE,
    ROUND_COUNTS,
    SPY_COUNTS,
    SPY_LOSSES,
    CorpusAnalysis,
    GameAnalysis,
    SpyStep,
    _analyse_game,
    _epsilon_for,
    _NodeCounter,
    _one_blunder,
    _probe,
    _round_leaf,
    _score_roots,
    _spy_round_blunder,
    analyse_corpus,
    announced_matrix,
    blunder_rng,
    read_announced,
    replay_spy,
    spy_holders,
    spy_loss,
    spy_steps,
)
from chess_variant_sandbox.tournament.record import GameRecord, read_records, write_records

HERE = Path(__file__).parent
ROOT = HERE.parent
SMALL = HERE / "_spy_small.json"


def _load(name: str) -> dict[str, Any]:
    return json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def played() -> tuple[dict[str, Any], Any]:
    """A small spy tournament: every pairing of the three spy players, two games each,
    forty plies at most."""
    spec = _load("_spy_small")
    spec["rules"]["max_plies"] = 40
    result = run_tournament(
        spec,
        TournamentConfig(
            agents=("random", "greedy", "minimax:nodes=400"), games_per_pairing=2, seed=7,
            workers=1,
        ),
    )
    return spec, result


@pytest.fixture(scope="module")
def analysed(played) -> CorpusAnalysis:
    spec, result = played
    return analyse_corpus(spec, result.records, blunder_sample=8, seed=0, workers=1)


def _square(engine: Engine, name: str) -> int:
    return engine._square_index(name)


def _position(
    engine: Engine, white: dict[str, str], black: dict[str, str], side: int = 0
) -> GameState:
    placement = {square: (piece, 0) for square, piece in white.items()}
    placement.update({square: (piece, 1) for square, piece in black.items()})
    return engine.make_state(placement, side_to_move=side)


def _move(engine: Engine, moves: list[int], origin: str, target: str) -> int:
    a, b = _square(engine, origin), _square(engine, target)
    found = [m for m in moves if decode_move(m)[:2] == (a, b)]
    assert len(found) == 1, (origin, target)
    return found[0]


def _step(engine: Engine, state: GameState, moves: tuple[int, ...]) -> SpyStep:
    """One step from a hand-built position, as `spy_steps` would yield it."""
    phase = engine.phase(state)
    if phase.is_round:
        after, _, events = engine.apply_round_with_phase(state, moves[0], moves[1])
        lists: tuple[list[int], ...] = (
            engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1)
        )
        return SpyStep(0, state, phase, moves, lists, after, events)
    return SpyStep(0, state, phase, moves, (engine.legal_moves(state),),
                   engine.apply(state, moves[0]), ())


def _lost(engine: Engine, step: SpyStep) -> frozenset[int]:
    return spy_holders(step.phase, step.state.side_to_move) - spy_holders(
        engine.phase(step.after), step.after.side_to_move
    )


# --------------------------------------------------------------------- records


class TestTheRecord:
    def test_every_phase_is_persisted_a_ply_a_word(self, played) -> None:
        """A turn writes one word and one entry of every series, a round two in seat order
        and every series twice; the branching is each mover's own count."""
        spec, result = played
        engine = Engine(spec)
        reached_rounds = 0
        for record in result.records:
            assert record.plies == len(record.moves) == len(record.branching)
            assert len(record.material) == len(record.captures) == record.plies
            for step in spy_steps(engine, record, strict=True):
                at = step.ply
                if len(step.moves) == 2:
                    reached_rounds += 1
                    assert record.material[at] == record.material[at + 1]
                    assert record.captures[at] == record.captures[at + 1]
                    assert record.branching[at:at + 2] == [len(step.lists[0]),
                                                           len(step.lists[1])]
                else:
                    assert record.branching[at] == len(step.lists[0])
        assert reached_rounds, "no game reached rounds, so the round arm was not tested"

    def test_a_record_replays_to_its_own_ending_and_series(self, played, tmp_path) -> None:
        spec, result = played
        engine = Engine(spec)
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        path = write_records(result.records, tmp_path / "games.jsonl")
        for original, read in zip(result.records, read_records(path), strict=True):
            assert read == original
            states = replay_spy(engine, read)
            outcome = engine.terminal(states[-1])
            assert outcome is not None
            assert (outcome.winner, outcome.reason) == (read.winner, read.reason)
            for step in spy_steps(engine, read):
                assert read.material[step.ply] == round(evaluator.material(step.state, 0), 4)

    def test_a_record_that_drifted_from_its_spec_is_refused(self, played) -> None:
        spec, result = played
        engine = Engine(spec)
        record = next(r for r in result.records if r.plies >= 4)
        swapped = GameRecord(**{**record.__dict__, "moves": record.moves[1:2] + record.moves[0:1]
                                + record.moves[2:]})
        with pytest.raises(ValueError, match="not legal"):
            replay_spy(engine, swapped)
        # Not strict, the walk stops where the record left the spec, as the other walks do.
        assert list(spy_steps(engine, swapped)) == []

    def test_an_analysis_written_before_the_spy_still_reads(self) -> None:
        """A file the pass wrote before schema 27 (no ``spy``, and no ``rounds`` for an
        alternating game) loads and writes back the same."""
        old = {
            "seed": 3, "agents": ["greedy", "greedy"], "near_best": [2], "legal_counts": [5],
            "moves_by_type": [1, 0], "captures_by_type": [0, 0], "capture_counts": [1],
            "revisited_plies": 0, "walked_plies": 1, "starting_instances": 4,
            "moved_instances": 1, "position_keys": [12],
        }
        game = GameAnalysis.from_dict(old)
        assert game.spy is None and game.rounds is None
        assert game.as_dict() == old


# ------------------------------------------------------------------- the probe


class TestTheProbe:
    def test_a_round_inside_a_turns_probe_is_a_leaf(self) -> None:
        """The seam: a turn's tree that reaches a round values it there and searches no
        further, so the probe never plays a round as a turn."""
        engine = Engine(_load("_spy_small"))
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        state = _position(engine, {"e1": "K", "c2": "R", "a1": "S"}, {"a5": "K", "b4": "N"})
        assert engine.phase(state).name == "announced"
        counter = _NodeCounter(1000)
        value = _probe(engine, evaluator, state, 4, counter)
        assert counter.used == 1
        assert value == _round_leaf(engine, evaluator, state)
        assert value == evaluator.evaluate_round(engine, state, state.side_to_move, (0, 0))

    def test_a_spy_games_turn_is_probed_in_python(self) -> None:
        """`_score_roots` hands a spy game's turn to `_root_scores`, not the core's probe,
        whose tree would cross the seam as turns; every other game keeps the core's."""
        from chess_variant_sandbox.tournament.analysis import _root_scores

        engine = Engine(_load("_spy_small"))
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        state = _position(engine, {"a1": "K", "a4": "R", "c2": "S"}, {"d5": "K", "c4": "S"})
        legal = engine.legal_moves(state)
        assert _score_roots(engine, evaluator, state, legal) == _root_scores(
            engine, evaluator, state, legal
        )

    @pytest.mark.parametrize("announcer", (0, 1))
    def test_an_announced_round_is_read_by_what_each_side_knew(self, announcer: int) -> None:
        """The announcer's moves by their worst answer, the answerer's against the move
        announced, each from its own side; the pure choices those values make."""
        moves = ([10, 11, 12], [20, 21])
        # Player 0's side, rows its moves: row 11 is safe (0 whatever the answer), rows 10
        # and 12 win against one answer and lose to the other.
        cells = [3.0, -2.0, 0.0, 0.0, -5.0, 4.0]
        if announcer == 0:
            found = announced_matrix(moves, cells, 0, 10)
            assert found.values[0] == [-2.0, 0.0, -5.0]
            assert found.values[1] == [-3.0, 2.0]
            assert found.blunders(0, 1.0) == [10, 12] and found.near_best(0, 1.0) == 1
            assert found.strategies == ([1.0, 0.0, 0.0], [0.0, 1.0])
        else:
            found = announced_matrix(moves, cells, 1, 21)
            # Player 1's side is the negation; its worst over player 0's three answers.
            assert found.values[1] == [-3.0, -4.0]
            assert found.values[0] == [-2.0, 0.0, 4.0]
            assert found.blunders(1, 0.5) == [21]
            assert found.strategies == ([0.0, 0.0, 1.0], [0.0, 1.0])

    def test_the_answerer_sees_the_announced_capture(self) -> None:
        """On a board: the answerer's spy, announced at, is worth more dodging than standing,
        and the reading says so for the move that dodges."""
        engine = Engine(_load("_spy_small"))
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        # White holds the spy and answers; Black announces Nb3xc1, at the spy (E5's shape).
        state = _position(engine, {"e1": "K", "c1": "S", "a4": "R"}, {"e5": "K", "b3": "N"})
        assert engine.phase(state).name == "announced" and engine.phase(state).announcer == 1
        lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
        take = _move(engine, lists[1], "b3", "c1")
        found = read_announced(engine, evaluator, state, 1, take, lists)
        assert found is not None
        dodge = _move(engine, lists[0], "c1", "d1")
        stand = _move(engine, lists[0], "e1", "d1")
        values = dict(zip(found.moves[0], found.values[0], strict=True))
        assert values[dodge] > values[stand]


# ------------------------------------------------------------- the pass itself


class TestThePass:
    def test_every_step_is_walked_and_counted(self, played, analysed) -> None:
        _, result = played
        graded = {(r.seed, r.agents): r for r in result.records if "random" not in r.agents}
        assert len(analysed.games) == len(graded)
        for game in analysed.games:
            record = graded[(game.seed, game.agents)]
            spy = game.spy
            assert spy is not None and tuple(spy) == SPY_COUNTS
            assert game.rounds is not None and set(game.rounds) == set(ROUND_COUNTS)
            steps = spy["alternating_plies"] + spy["reply_turns"]
            rounds = spy["announced_rounds"] + spy["blind_rounds"]
            assert steps + 2 * rounds == record.plies == game.walked_plies
            assert game.rounds["rounds"] == rounds
            assert spy["reply_turns"] in (0, 1)
            assert len(game.position_keys) == record.plies
            assert sum(game.moves_by_type) == record.plies
            assert len(game.near_best) == len(game.legal_counts) == len(game.capture_counts)
            losses = [spy[f"lost_ply_{p}"] for p in (0, 1) if spy[f"lost_ply_{p}"] is not None]
            for player in (0, 1):
                assert (spy[f"lost_ply_{player}"] is None) == (spy[f"lost_how_{player}"] is None)
                assert spy[f"lost_how_{player}"] in (None, *SPY_LOSSES)
            # The alternating phase is exactly the plies before the first loss.
            if losses:
                assert min(losses) == spy["alternating_plies"]
            else:
                assert spy["alternating_plies"] == record.plies and rounds == 0

    def test_a_round_starting_on_an_odd_ply_is_still_sampled(self, played, analysed) -> None:
        """A step is probed when one of its plies falls on the stride, so a game whose
        rounds start on an odd ply (E2) still has its rounds read."""
        spec, result = played
        engine = Engine(spec)
        for game in analysed.games:
            record = next(r for r in result.records
                          if (r.seed, r.agents) == (game.seed, game.agents))
            expected = 0
            for step in spy_steps(engine, record):
                if not any((step.ply + k) % PLY_STRIDE == 0 for k in range(len(step.moves))):
                    continue
                if len(step.moves) == 1:
                    expected += int(len(step.lists[0]) > 1)
                elif step.lists[0] and step.lists[1]:
                    expected += sum(1 for moves in step.lists if len(moves) > 1)
            assert len(game.legal_counts) == expected

    def test_the_file_round_trips_and_holds_the_counts(self, analysed, tmp_path) -> None:
        path = analysed.save(tmp_path / "analysis.json")
        assert CorpusAnalysis.load(path).as_dict() == analysed.as_dict()
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert all(set(game["spy"]) == set(SPY_COUNTS) for game in on_disk["games"])

    def test_no_other_games_file_holds_the_spy_counts(self) -> None:
        assert "spy" not in GameAnalysis(seed=0, agents=("a", "b")).as_dict()
        assert "spy" not in GameAnalysis(seed=0, agents=("a", "b"), rounds={}).as_dict()

    def test_a_round_games_analysis_holds_no_spy_counts(self) -> None:
        spec = json.loads((HERE / "_simultaneous_gardner.json").read_text(encoding="utf-8"))
        spec["rules"]["max_plies"] = 20
        result = run_tournament(spec, TournamentConfig(agents=("greedy",), games_per_pairing=1,
                                                       seed=3, workers=1))
        engine = Engine(spec)
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        game = _analyse_game(engine, evaluator, result.records[0], _epsilon_for(engine))
        assert game.spy is None and "spy" not in game.as_dict() and game.rounds is not None

    def test_the_blunder_trials_name_a_step_and_a_seat(self, played, analysed) -> None:
        _, result = played
        assert analysed.blunders, "no trial was drawn, so nothing was tested"
        by_game = {(r.seed, r.agents): r for r in result.records}
        for trial in analysed.blunders:
            record = by_game[(trial.seed, trial.agents)]
            assert trial.ply == int(record.plies * 0.34)
            assert trial.blundering_side in (0, 1)

    def test_a_trial_is_fixed_by_its_seed_and_record(self, played) -> None:
        spec, result = played
        engine = Engine(spec)
        for record in (r for r in result.records if "random" not in r.agents and r.plies >= 12):
            first = _one_blunder(engine, record, blunder_rng(0, record))
            assert first == _one_blunder(engine, record, blunder_rng(0, record))

    def test_a_forced_announcement_is_answered_by_the_records_player(self, played) -> None:
        """An announcer's blunder is short of its best by the maximin; the answerer, who
        sees it, answers as its player would, asked of the core."""
        spec, result = played
        engine = Engine(spec)
        epsilon = _epsilon_for(engine)
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        tried = 0
        for record in (r for r in result.records if "random" not in r.agents):
            for step in spy_steps(engine, record):
                if step.phase.name != "announced":
                    continue
                announcer = step.phase.announcer
                pair = _spy_round_blunder(engine, record, step, announcer, random.Random(3))
                if pair is None:
                    continue
                reading = read_announced(engine, evaluator, step.state, announcer,
                                         step.moves[announcer])
                assert reading is not None
                assert pair[announcer] in reading.blunders(announcer, epsilon)
                assert pair[1 - announcer] in step.lists[1 - announcer]
                tried += 1
                break
        assert tried, "no announced round offered the announcer a blunder"


class TestHowASpyIsLost:
    def test_by_capture_on_a_turn(self) -> None:
        """E1: the rook takes the spy on an alternating turn."""
        engine = Engine(_load("_spy_small"))
        state = _position(engine, {"a1": "K", "a4": "R", "c2": "S"}, {"d5": "K", "c4": "S"})
        step = _step(engine, state, (_move(engine, engine.legal_moves(state), "a4", "c4"),))
        assert _lost(engine, step) == {1}
        assert spy_loss(engine, step, 1) == "capture"

    def test_by_a_blast_on_its_owners_move(self) -> None:
        """E3: the knight's capture explodes beside its own spy."""
        engine = Engine(_load("_spy_blast"))
        state = _position(engine, {"e1": "K", "b2": "N", "c3": "S"},
                          {"d5": "K", "d3": "R", "a5": "S"})
        step = _step(engine, state, (_move(engine, engine.legal_moves(state), "b2", "d3"),))
        assert _lost(engine, step) == {0}
        assert spy_loss(engine, step, 0) == "blast"

    def test_by_a_collision_in_an_announced_round(self) -> None:
        """The answerer's spy steps where the announcer's rook arrives, and loses the
        meeting (a spy that captures nothing beats nothing)."""
        engine = Engine(_load("_spy_small"))
        state = _position(engine, {"e1": "K", "c1": "S"}, {"a5": "K", "a2": "R"})
        assert engine.phase(state).name == "announced"
        rook = _move(engine, engine.legal_moves_for(state, 1), "a2", "c2")
        spy = _move(engine, engine.legal_moves_for(state, 0), "c1", "c2")
        step = _step(engine, state, (spy, rook))
        assert _lost(engine, step) == {0}
        assert spy_loss(engine, step, 0) == "collision"

    def test_by_its_own_promotion(self) -> None:
        engine = Engine(_load("_spy_stock"))
        state = _position(engine, {"e1": "K", "c4": "S"}, {"a5": "K", "e5": "S"})
        promote = [m for m in engine.legal_moves(state)
                   if decode_move(m)[:2] == (_square(engine, "c4"), _square(engine, "c5"))
                   and move_type(m) == MOVE_PROMOTION]
        assert len(promote) == 1
        step = _step(engine, state, (promote[0],))
        assert _lost(engine, step) == {0}
        assert spy_loss(engine, step, 0) == "promotion"


# ------------------------------------------------------------------ the score


class TestTheScore:
    def test_every_scored_component_says_how_it_reads_a_spy_game(self) -> None:
        assert set(load_weights(DEFAULT_WEIGHTS).weights) <= set(SPY_READINGS)

    def test_a_spy_corpus_scores_with_no_component_absent(self, played, analysed) -> None:
        spec, result = played
        components = compute_components(spec, result.records, result.ratings, analysed)
        weights = load_weights(DEFAULT_WEIGHTS)
        absent = [name for name in weights.weights
                  if name not in components.raw and name not in weights.optional]
        assert absent == []
        assert components.diagnostics["spy_readings"] == SPY_READINGS
        assert "round_readings" not in components.diagnostics
        summary = components.diagnostics["spy"]
        assert summary["games"] == len(analysed.games)
        assert 0.0 <= summary["spy_lost_share"] <= 1.0

    def test_the_score_carries_the_readings(self, played, analysed) -> None:
        spec, result = played
        scored = score_ruleset(spec, result.records, result.ratings, analysis=analysed,
                               interval=False)
        assert scored.diagnostics["spy_readings"] == SPY_READINGS

    def test_the_summary_reads_the_counts(self) -> None:
        def game(**lost: Any) -> dict[str, Any]:
            counts: dict[str, Any] = {key: (None if key.startswith("lost_") else 0)
                                      for key in SPY_COUNTS}
            counts.update(alternating_plies=10)
            counts.update(lost)
            return counts

        summary = spy_summary([
            game(),
            game(lost_ply_0=10, lost_how_0="capture", reply_turns=1),
            game(lost_ply_1=14, lost_how_1="capture", lost_ply_0=14, lost_how_0="blast"),
            game(lost_ply_1=20, lost_how_1="collision", lost_ply_0=31, lost_how_0="capture"),
        ])
        assert summary["games"] == 4
        assert summary["spy_lost_share"] == 0.75 and summary["both_lost_share"] == 0.5
        assert summary["both_at_once"] == 1
        assert summary["first_loss_median_ply"] == 14
        assert summary["reply_turn_share"] == 0.25
        assert summary["lost_how"] == {"blast": 1, "capture": 3, "collision": 1}


# ----------------------------------------------------------------- the roster


class TestTheSpyRoster:
    def test_a_spy_game_is_measured_on_the_spy_players(self) -> None:
        spec = _load("_spy_small")
        digest = spec_hash(spec)
        assert plays_with_spies(spec)
        assert corpus.canonical_agents(digest, spec=spec) == corpus.SPY_AGENTS
        assert corpus.current_basis(digest, spec=spec, seeds=corpus.ONE_SEED).agents == (
            corpus.SPY_AGENTS
        )

    def test_the_spy_ladder_is_the_alternating_ladder_on_the_built_in_objective(self) -> None:
        from chess_variant_sandbox.loop.screen import spy_roster

        budgets = [int(a.split("nodes=")[1]) for a in corpus.CANONICAL_AGENTS if "nodes=" in a]
        assert ("random", "greedy") + tuple(
            f"minimax:nodes={budget}" for budget in budgets
        ) == corpus.SPY_AGENTS
        assert spy_roster(corpus.SPY_AGENTS) == corpus.SPY_AGENTS
        assert not corpus.fits_its_roster(corpus.SPY_AGENTS)

    def test_the_spy_players_hand_the_core_a_players_digest(self) -> None:
        from chess_variant_sandbox.measurement import players_fingerprint

        assert players_fingerprint(_load("_spy_small"), corpus.SPY_AGENTS) != ""

    def test_a_round_game_keeps_its_roster(self) -> None:
        spec = json.loads((HERE / "_simultaneous_gardner.json").read_text(encoding="utf-8"))
        assert not plays_with_spies(spec)
        assert corpus.canonical_agents(spec_hash(spec), spec=spec) == corpus.ROUND_AGENTS

    def test_measure_check_reads_a_spy_game(self, tmp_path, capsys) -> None:
        from chess_variant_sandbox import cli

        assert cli.main(["measure", "--check", str(SMALL), "--runs", str(tmp_path)]) == 1
        said = capsys.readouterr().out
        assert "spy small" in said and "never measured" in said


@pytest.mark.slow
def test_measure_measures_a_spy_game_and_then_reads_it_current(tmp_path, capsys) -> None:
    """`measure` plays the spy ladder, analyses and scores, and `measure --check` then
    reads the file current."""
    from chess_variant_sandbox import cli

    assert cli.main(["measure", str(SMALL), "--runs", str(tmp_path), "--workers", "4"]) in (0, 1)
    digest = spec_hash(_load("_spy_small"))
    stored = json.loads((tmp_path / digest / "canonical.json").read_text(encoding="utf-8"))
    assert tuple(stored["basis"]["agents"]) == corpus.SPY_AGENTS
    analysis = json.loads((tmp_path / f"{digest}-0" / "analysis.json").read_text(encoding="utf-8"))
    assert all("spy" in game for game in analysis["games"])
    capsys.readouterr()
    assert cli.main(["measure", "--check", str(SMALL), "--runs", str(tmp_path)]) == 0
    assert "current" in capsys.readouterr().out


@pytest.mark.slow
def test_the_pass_is_the_same_at_one_worker_and_four(played) -> None:
    spec, result = played
    one = analyse_corpus(spec, result.records, blunder_sample=8, seed=0, workers=1)
    four = analyse_corpus(spec, result.records, blunder_sample=8, seed=0, workers=4)
    assert one.as_dict() == four.as_dict()
